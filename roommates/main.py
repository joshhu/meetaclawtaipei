"""三個 LLM 室友的電子魚缸：FastAPI + WebSocket + race orchestrator。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("roommates")

# ---- agent 設定 ----------------------------------------------------------

ROOM_INTRO = (
    "你和兩個室友在台北合租的公寓客廳。現在是週五晚上，剛吃完飯，"
    "三個人坐在沙發上隨意聊天。請依角色自然發言，回應上一個人說的話。"
)

COMMON_RULES = (
    "重要規則：\n"
    "1. 每次只說 1-2 句、最多 50 個中文字\n"
    "2. 用繁體中文（台灣口語）\n"
    "3. 不要列點、不要 markdown、不要表情符號\n"
    "4. 要回應上一個人說的內容（針對他的話接話）\n"
    "5. 不要每次都自我介紹，把它當朋友聊天\n"
)


@dataclass
class Agent:
    name: str
    emoji: str
    role: str  # 用在 system prompt
    base_url: str
    model: str
    extra: dict = field(default_factory=dict)  # extra body params for OpenAI

    def __post_init__(self):
        self.client = AsyncOpenAI(base_url=self.base_url, api_key="not-needed", timeout=90.0)

    def system_prompt(self) -> str:
        others = [a.name for a in AGENTS if a.name != self.name]
        return (
            f"你是 {self.name}。{self.role}\n\n"
            f"{ROOM_INTRO}\n\n"
            f"你的室友是 {' 和 '.join(others)}。\n\n"
            f"{COMMON_RULES}"
        )

    async def speak(self, history: list[dict]) -> str:
        msgs = [{"role": "system", "content": self.system_prompt()}]
        for h in history:
            # 把其他人說的注入成 user role 的對話
            msgs.append({"role": "user", "content": f"[{h['speaker']}]：{h['text']}"})
        msgs.append({"role": "user", "content": f"輪到你（{self.name}）發言。"})

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            max_tokens=180,
            temperature=0.85,
            top_p=0.9,
            **self.extra,
        )
        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        # Qwen3.6 thinking 模式可能把答案放 reasoning
        if not text and hasattr(msg, "reasoning") and msg.reasoning:
            # 取 reasoning 最後一段當答案（截斷）
            text = msg.reasoning.strip().split("\n")[-1][:120]
        if not text:
            text = "（沒話說）"
        # 清理常見的格式干擾
        text = text.replace("\n", " ").strip()
        # 去掉開頭可能的 [name]: 標記（model 偶爾會這樣輸出）
        for prefix in (f"[{self.name}]：", f"[{self.name}]:", f"{self.name}：", f"{self.name}:"):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        return text


AGENTS = [
    Agent(
        name="Alex",
        emoji="🤔",
        role=(
            "哲學系研究生，個性偏內向但愛分析。講話有層次、會引用一些概念，"
            "但要克制不要長篇大論。常常會質疑前面的人或補充另一個角度。"
        ),
        base_url="http://localhost:8002/v1",
        model="qwen36",
        # 試圖讓 Qwen3.6 不要進 thinking 模式（不一定有效但加上）
        extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    ),
    Agent(
        name="Bella",
        emoji="😊",
        role=(
            "外向社交咖，剛從廣告公司下班。個性熱情、主動帶話題、語氣輕快，"
            "喜歡分享今天遇到的趣事，也愛幫朋友打氣。"
        ),
        base_url="http://localhost:8001/v1",
        model="gemma4",
    ),
    Agent(
        name="Carl",
        emoji="🍳",
        role=(
            "理工男，軟體工程師，個性實事求是，講話簡短直接。"
            "對生活瑣事務實，常常給簡單的解決方案而不是討論。"
        ),
        base_url="http://localhost:8003/v1",
        model="nemotron",
    ),
]

AGENT_BY_NAME = {a.name: a for a in AGENTS}

# ---- race orchestrator ---------------------------------------------------


async def race_round(history: list[dict], last_winner: str | None) -> tuple[str, str, float]:
    """三人同時 generate，最快完成的勝出；上輪 winner 不參與本輪。

    Returns (speaker_name, text, elapsed_seconds)。
    """
    candidates = [a for a in AGENTS if a.name != last_winner]
    if not candidates:
        candidates = list(AGENTS)

    start = time.monotonic()
    tasks: dict[asyncio.Task, Agent] = {}
    for agent in candidates:
        coro = agent.speak(history)
        task = asyncio.create_task(coro, name=f"speak-{agent.name}")
        tasks[task] = agent

    try:
        done, pending = await asyncio.wait(
            tasks.keys(), return_when=asyncio.FIRST_COMPLETED, timeout=90.0
        )
    except Exception as e:
        log.exception("race_round wait failed: %s", e)
        for t in tasks:
            t.cancel()
        raise

    # 取消其他還在跑的
    for t in pending:
        t.cancel()
    # 把取消的等一下、吞掉例外
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    # 取第一個完成且成功的
    winner_task = None
    for t in done:
        if t.exception() is None:
            winner_task = t
            break
    if winner_task is None:
        # 全部失敗，回 fallback
        errs = [str(t.exception()) for t in done]
        log.warning("All candidates failed: %s", errs)
        return ("system", f"(本輪所有人都卡住了：{errs[0][:80]})", time.monotonic() - start)

    speaker = tasks[winner_task].name
    text = winner_task.result()
    elapsed = time.monotonic() - start
    return speaker, text, elapsed


# ---- FastAPI app ---------------------------------------------------------

app = FastAPI(title="Roommates")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {
        "ok": True,
        "agents": [{"name": a.name, "emoji": a.emoji, "endpoint": a.base_url} for a in AGENTS],
    }


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    log.info("client connected")

    history: list[dict] = []
    last_winner: str | None = None

    async def send(payload: dict):
        await ws.send_text(json.dumps(payload, ensure_ascii=False))

    # 初始 hello
    await send({
        "type": "hello",
        "agents": [{"name": a.name, "emoji": a.emoji} for a in AGENTS],
    })

    try:
        while True:
            # 通知前端：全員「思考中」
            await send({
                "type": "thinking",
                "candidates": [a.name for a in AGENTS if a.name != last_winner],
            })

            try:
                speaker, text, elapsed = await race_round(history, last_winner)
            except Exception as e:
                log.exception("round error: %s", e)
                await send({"type": "error", "message": str(e)[:200]})
                await asyncio.sleep(5)
                continue

            await send({
                "type": "say",
                "speaker": speaker,
                "text": text,
                "elapsed_ms": int(elapsed * 1000),
            })

            if speaker != "system":
                history.append({"speaker": speaker, "text": text})
                last_winner = speaker
                # 只留最近 10 輪
                history = history[-10:]

            # cooldown 讓用戶看清楚
            await asyncio.sleep(3.5)

    except WebSocketDisconnect:
        log.info("client disconnected")
    except Exception as e:
        log.exception("ws loop error: %s", e)
        try:
            await ws.close()
        except Exception:
            pass
