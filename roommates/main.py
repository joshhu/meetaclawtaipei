"""舞台三巨頭 v3：Trump / Jensen / Obama + 即時 web search (tool use)。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ddgs import DDGS
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("roommates")

# ---- web search 工具 -----------------------------------------------------

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web with DuckDuckGo for current information, news, or facts. "
            "Use this when you need up-to-date data that may have changed after your training cutoff."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, English usually gives better results than Chinese.",
                },
            },
            "required": ["query"],
        },
    },
}


async def web_search(query: str, max_results: int = 5) -> str:
    """同步 ddgs 包進 async。回傳格式化字串給 LLM 讀。"""
    def _do():
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    try:
        results = await asyncio.wait_for(asyncio.to_thread(_do), timeout=20)
    except Exception as e:
        log.warning("web_search failed: %s", e)
        return f"(搜尋失敗：{e})"

    if not results:
        return "(無搜尋結果)"
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")[:120]
        body = r.get("body", "")[:280]
        href = r.get("href", "")
        lines.append(f"[{i}] {title}\n{body}\n來源: {href}")
    return "\n\n".join(lines)


# ---- agent 設定 ----------------------------------------------------------

STAGE_INTRO = (
    "現在是 2026 年「AI Summit 台北」的開幕論壇，三位重量級講者同台對談。"
    "請完全以你的人物口吻發言，模仿那個人公開的講話風格。"
)

COMMON_RULES = (
    "重要規則：\n"
    "1. 每次只說 1-2 句、最多 60 個中文字（短而有力的金句最讚）\n"
    "2. 用繁體中文（如果是 Trump 可以混一兩個簡單英文字）\n"
    "3. 不要列點、不要 markdown、不要表情符號、不要旁白動作\n"
    "4. 要針對上一個講者剛說的話接話（呼應、補充、或反駁）\n"
    "5. 不要每次自我介紹\n"
    "6. 內容輕鬆幽默，不要碰敏感政治攻擊\n"
)

SEARCH_GUIDE = (
    "你有 web_search 工具可以即時上網。\n"
    "**重要：當話題涉及任何具體事實、產品、人物、新聞、最新進展、規格、數字時，"
    "你應該優先呼叫 web_search 拿到第一手資料再發言**，不要靠記憶猜測。\n"
    "搜尋 query 用英文（結果較豐富），第一次發言時幾乎都該搜尋。"
)


@dataclass
class Agent:
    name: str
    emoji: str
    role: str
    base_url: str
    model: str
    avatar: str = ""
    can_search: bool = False        # 是否帶 tools 參數
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.client = AsyncOpenAI(base_url=self.base_url, api_key="not-needed", timeout=120.0)

    def system_prompt(self) -> str:
        others = [a.name for a in AGENTS if a.name != self.name]
        parts = [
            f"你是 {self.name}。{self.role}",
            STAGE_INTRO,
            f"同台的另外兩位是 {' 和 '.join(others)}。",
            COMMON_RULES,
        ]
        if self.can_search:
            parts.append(SEARCH_GUIDE)
        return "\n\n".join(parts)

    async def speak(self, history: list[dict], topic: str, on_event) -> str:
        """tool loop。on_event(dict) 是 callback 用來推 WS 訊息。"""
        msgs: list[dict] = [{"role": "system", "content": self.system_prompt()}]
        if topic:
            msgs.append({"role": "user", "content": f"今天的論壇主題是：{topic}"})
        for h in history:
            msgs.append({"role": "user", "content": f"[{h['speaker']}]：{h['text']}"})
        msgs.append({"role": "user", "content": f"輪到你（{self.name}）發言。"})

        kwargs = dict(
            model=self.model,
            max_tokens=400,
            temperature=0.85,
            top_p=0.9,
        )
        if self.can_search:
            kwargs["tools"] = [SEARCH_TOOL]
            kwargs["tool_choice"] = "auto"
        kwargs.update(self.extra)

        # tool loop，最多 3 輪以防無限
        for _ in range(3):
            resp = await self.client.chat.completions.create(messages=msgs, **kwargs)
            msg = resp.choices[0].message
            tool_calls = msg.tool_calls or []

            # Fallback: 有些 model（如 nemotron 12B-VL）會把 tool call 寫成 <TOOLCALL>[...]</TOOLCALL>
            # 字串而沒進 tool_calls 欄位，這裡手動解析
            if not tool_calls and msg.content:
                m = re.search(r"<TOOLCALL>\s*(\[.+?\])\s*</TOOLCALL>", msg.content, re.S)
                if m:
                    try:
                        parsed = json.loads(m.group(1))
                        synthetic = []
                        for i, call in enumerate(parsed):
                            synthetic.append(type("TC", (), {
                                "id": f"manual-{i}",
                                "function": type("F", (), {
                                    "name": call.get("name", ""),
                                    "arguments": json.dumps(call.get("arguments", {})),
                                })()
                            })())
                        tool_calls = synthetic
                        log.info("Parsed manual TOOLCALL from %s", self.name)
                    except (json.JSONDecodeError, Exception) as e:
                        log.warning("Manual TOOLCALL parse failed: %s", e)

            if not tool_calls:
                # 結束：取 content
                text = (msg.content or "").strip()
                if not text and hasattr(msg, "reasoning") and msg.reasoning:
                    text = msg.reasoning.strip().split("\n")[-1][:200]
                return self._clean(text)

            # 有 tool calls — 把 assistant 的 tool_call 加進 messages，再執行每個 tool
            msgs.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                if tc.function.name != "web_search":
                    result = f"(unknown tool: {tc.function.name})"
                else:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                        query = args.get("query", "")
                    except json.JSONDecodeError:
                        query = tc.function.arguments
                    await on_event({"type": "searching", "speaker": self.name, "query": query})
                    result = await web_search(query)
                    await on_event({"type": "search_done", "speaker": self.name})
                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result[:3000],  # 限制長度避免 KV cache 爆
                })
        # 超過 3 輪還在 tool — 強制 final
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)
        resp = await self.client.chat.completions.create(messages=msgs, **kwargs)
        return self._clean(resp.choices[0].message.content or "（沒話說）")

    def _clean(self, text: str) -> str:
        text = (text or "").replace("\n", " ").strip()
        if not text:
            return "（沒話說）"
        for prefix in (f"[{self.name}]：", f"[{self.name}]:", f"{self.name}：", f"{self.name}:"):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        return text


AGENTS = [
    Agent(
        name="Obama",
        emoji="🇺🇸",
        avatar="obama.jpg",
        role=(
            "你是前美國總統 Barack Obama。講話風格沉穩有節奏、會用「我們」拉近距離，"
            "喜歡引用希望與團結的概念，常用「Let me be clear」或「讓我這樣說」開頭。"
            "句子要有抑揚頓挫，偶爾用一個短停頓（...）。中文以正式而溫暖的口吻。"
            "你**不**自己上網搜尋，而是根據其他兩位剛才提到的資訊做平衡評論。"
        ),
        base_url="http://localhost:8001/v1",
        model="gemma4",
        can_search=False,
    ),
    Agent(
        name="Jensen",
        emoji="🧥",
        avatar="jensen.jpg",
        role=(
            "你是 NVIDIA CEO 黃仁勳。台裔美國人、招牌黑皮衣。"
            "講話會大量使用 AI、GPU、CUDA、Blackwell、加速運算、AI 工廠 等技術詞，"
            "語氣興奮、對未來樂觀，偶爾混一兩個英文。"
            "**必須用繁體中文，絕對不要簡體字（不能寫「当然/单/广/会」要寫「當然/單/廣/會」）。**"
            "如果話題與技術/產業有關，請積極使用 web_search 抓最新資料佐證。"
        ),
        base_url="http://localhost:8003/v1",
        model="nemotron",
        can_search=True,
    ),
    Agent(
        name="Trump",
        emoji="🇺🇸",
        avatar="trump.jpg",
        role=(
            "你是前美國總統 Donald Trump。講話風格直接、用詞簡單、愛用最高級「最棒的」「前所未有」「相信我」。"
            "短句、自信、有點誇張，常說「many people are saying」「nobody knows X better than me」。"
            "中文盡量口語、有時穿插一兩個英文字（tremendous / huge / believe me）。"
            "輕鬆幽默為主，不要做政治攻擊。"
            "如果話題涉及新聞、數字、最新事件，請用 web_search 工具查一下再發言。"
        ),
        base_url="http://localhost:8002/v1",
        model="qwen36",
        can_search=True,
        extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    ),
]

AGENT_BY_NAME = {a.name: a for a in AGENTS}

# ---- FastAPI app ---------------------------------------------------------

app = FastAPI(title="Roommates v3")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {
        "ok": True,
        "agents": [
            {
                "name": a.name,
                "emoji": a.emoji,
                "avatar": a.avatar,
                "endpoint": a.base_url,
                "can_search": a.can_search,
            }
            for a in AGENTS
        ],
    }


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    log.info("client connected")

    history: list[dict] = []
    topic: str = ""
    rr_idx: int = 0  # round-robin index
    paused: bool = True  # 等用戶設 topic 才開始

    async def send(payload: dict):
        await ws.send_text(json.dumps(payload, ensure_ascii=False))

    await send({
        "type": "hello",
        "agents": [
            {"name": a.name, "emoji": a.emoji, "avatar": a.avatar, "can_search": a.can_search}
            for a in AGENTS
        ],
    })

    # 接收用戶 set_topic 的背景 task
    async def listen_client():
        nonlocal topic, history, rr_idx, paused
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    m = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if m.get("type") == "set_topic":
                    topic = (m.get("topic") or "").strip()
                    history.clear()
                    rr_idx = 0
                    paused = False
                    log.info("topic set: %r", topic)
                    await send({"type": "topic_set", "topic": topic})
                elif m.get("type") == "stop":
                    paused = True
                    log.info("stopped by user")
                    await send({"type": "stopped"})
        except WebSocketDisconnect:
            paused = True

    listener = asyncio.create_task(listen_client())

    try:
        while True:
            if paused or not topic:
                await asyncio.sleep(0.5)
                continue

            agent = AGENTS[rr_idx % len(AGENTS)]
            rr_idx += 1

            await send({"type": "thinking", "candidates": [agent.name]})

            start = time.monotonic()
            try:
                text = await agent.speak(history, topic, send)
            except Exception as e:
                log.exception("speak error: %s", e)
                await send({"type": "error", "message": f"{agent.name}: {str(e)[:200]}"})
                await asyncio.sleep(3)
                continue
            elapsed = time.monotonic() - start

            await send({
                "type": "say",
                "speaker": agent.name,
                "text": text,
                "elapsed_ms": int(elapsed * 1000),
            })

            history.append({"speaker": agent.name, "text": text})
            history = history[-12:]  # 留最近 12 句

            # 等前端打字機跑完 + 1.2s 緩衝給用戶讀完
            # 前端打字速度 40ms/char
            typing_seconds = len(text) * 0.040
            await asyncio.sleep(max(2.5, typing_seconds + 1.2))
    except WebSocketDisconnect:
        log.info("client disconnected")
    except Exception as e:
        log.exception("ws loop error: %s", e)
    finally:
        listener.cancel()
        try:
            await ws.close()
        except Exception:
            pass
