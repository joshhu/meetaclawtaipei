"""舞台三巨頭 v4：Trump / Jensen / Obama + web search (tool use) + Qwen3-TTS voice clone。"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from ddgs import DDGS
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

# 載入專案根目錄（roommates 的上一層）的 .env，取得 BRAVE_API_KEY 等設定
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

TTS_URL = os.environ.get("TTS_URL", "http://localhost:5051")
TTS_TIMEOUT = float(os.environ.get("TTS_TIMEOUT", "30"))
TTS_CLIENT = httpx.AsyncClient(timeout=TTS_TIMEOUT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("roommates")

# ---- web search 工具 -----------------------------------------------------

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information, news, or facts. "
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


BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "").strip()
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
_brave_client = httpx.AsyncClient(timeout=15)


async def _brave_search(query: str, max_results: int) -> list[dict]:
    """打 Brave Search REST API，回傳 {title, body, href} list（與 ddgs 同形）。"""
    r = await _brave_client.get(
        BRAVE_URL,
        params={"q": query, "count": max_results, "search_lang": "en", "text_decorations": "false"},
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": BRAVE_API_KEY,
        },
    )
    r.raise_for_status()
    web = (r.json().get("web") or {}).get("results") or []
    return [
        {"title": x.get("title", ""), "body": x.get("description", ""), "href": x.get("url", "")}
        for x in web
    ]


async def _ddg_search(query: str, max_results: int) -> list[dict]:
    """DDG fallback（同步 ddgs 包進 thread）。"""
    def _do():
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    return await asyncio.wait_for(asyncio.to_thread(_do), timeout=20)


async def web_search(query: str, max_results: int = 5) -> str:
    """有 BRAVE_API_KEY 就用 Brave，失敗或無 key 則 fallback 回 DDG。回傳格式化字串給 LLM 讀。"""
    results: list[dict] = []
    if BRAVE_API_KEY:
        try:
            results = await _brave_search(query, max_results)
        except Exception as e:
            log.warning("brave_search failed, fallback to DDG: %s", e)
    if not results:
        try:
            results = await _ddg_search(query, max_results)
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

STAGE_INTRO_ZH = (
    "現在是 2026 年「AI Summit 台北」的開幕論壇，三位重量級講者同台對談。"
    "請完全以你的人物口吻發言，模仿那個人公開的講話風格。"
)
STAGE_INTRO_EN = (
    "You are on stage at a 2026 AI Summit opening panel with two other heavyweight speakers. "
    "Speak exactly in your persona's well-known public speaking style. "
    "**All speakers communicate in English.** Audience is global."
)

COMMON_RULES_ZH = (
    "重要規則：\n"
    "1. 每次只說 1-2 句、最多 60 個中文字（短而有力的金句最讚）\n"
    "2. 用繁體中文（如果是 Trump 可以混一兩個簡單英文字）\n"
    "3. 不要列點、不要 markdown、不要表情符號、不要旁白動作\n"
    "4. 要針對上一個講者剛說的話接話（呼應、補充、或反駁）\n"
    "5. 不要每次自我介紹\n"
    "6. 內容輕鬆幽默，不要碰敏感政治攻擊\n"
)
COMMON_RULES_EN = (
    "IMPORTANT RULES:\n"
    "1. Reply with ONLY 1-2 short punchy sentences, max 25 words.\n"
    "2. **Use English only — no Chinese characters whatsoever.**\n"
    "3. No bullet points, no markdown, no emoji, no stage directions.\n"
    "4. Respond to what the previous speaker just said (echo, add, or push back).\n"
    "5. Don't re-introduce yourself.\n"
    "6. Keep it playful, no political attacks.\n"
)

SEARCH_GUIDE = (
    "You have a web_search tool. When the topic involves facts, products, people, "
    "news, specs, or numbers, call web_search first to get fresh data. "
    "Use English queries for best results."
)

# 每輪隨機塞一個 mood/angle 給 model，逼出不同回應
MOODS_EN = [
    "Be skeptical and push back.",
    "Be enthusiastic and amplify the point.",
    "Tell a quick anecdote (1 line).",
    "Drop a sharp one-liner.",
    "Disagree with the previous speaker.",
    "Build on what was just said with a fresh angle.",
    "Be witty / make a joke.",
    "Mention a specific number or fact you know.",
    "Ask a rhetorical question.",
    "Make a bold prediction.",
]
MOODS_ZH = [
    "保持懷疑、反駁一下。",
    "熱情地放大這個觀點。",
    "講個一句話的小故事。",
    "丟一句很尖銳的話。",
    "不同意上一位的說法。",
    "從新的角度延伸剛才的話題。",
    "幽默一點、開個玩笑。",
    "提一個具體數字或事實。",
    "問一個反問句。",
    "做一個大膽預測。",
]


@dataclass
class Agent:
    name: str
    emoji: str
    role: str                       # English role (default)
    base_url: str
    model: str
    role_zh: str = ""               # 中文版 role（沒有就用 role）
    avatar: str = ""
    can_search: bool = False        # 是否帶 tools 參數
    max_response_tokens: int = 200  # 模型生成上限（短回應）
    max_response_chars: int = 200   # 後處理上限（更短）
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.client = AsyncOpenAI(base_url=self.base_url, api_key="not-needed", timeout=120.0)

    def system_prompt(self, language: str = "zh") -> str:
        others = [a.name for a in AGENTS if a.name != self.name]
        if language == "en":
            parts = [
                self.role,
                STAGE_INTRO_EN,
                f"The other two speakers are {' and '.join(others)}.",
                COMMON_RULES_EN,
            ]
        else:
            role_zh = self.role_zh or self.role
            parts = [
                role_zh,
                STAGE_INTRO_ZH,
                f"同台的另外兩位是 {' 和 '.join(others)}。",
                COMMON_RULES_ZH,
            ]
        if self.can_search:
            parts.append(SEARCH_GUIDE)
        return "\n\n".join(parts)

    async def speak(self, history: list[dict], topic: str, on_event, language: str = "zh") -> str:
        """tool loop。on_event(dict) 是 callback 用來推 WS 訊息。"""
        log.info("speak(): name=%s language=%r", self.name, language)
        # mood 併進 system content（避免 qwen 抱怨 system 不在開頭）
        sys_content = self.system_prompt(language)
        if language == "en":
            mood = random.choice(MOODS_EN)
            sys_content += f"\n\nStyle hint for THIS turn only: {mood} Vary your phrasing from previous turns."
        else:
            mood = random.choice(MOODS_ZH)
            sys_content += f"\n\n本輪風格提示：{mood} 變換語氣別重複先前的開頭。"
        msgs: list[dict] = [{"role": "system", "content": sys_content}]

        if topic:
            if language == "en":
                msgs.append({"role": "user", "content": f"Today's panel topic: {topic}"})
            else:
                msgs.append({"role": "user", "content": f"今天的論壇主題是：{topic}"})
        for h in history:
            sep = ":" if language == "en" else "："
            msgs.append({"role": "user", "content": f"[{h['speaker']}]{sep}{h['text']}"})

        if language == "en":
            msgs.append({"role": "user", "content":
                f"Your turn, {self.name}. Reply ONLY with 1-2 short English sentences "
                f"(max 25 words). Just speak — do not narrate your thinking or repeat the instructions. "
                f"No Chinese characters."})
        else:
            msgs.append({"role": "user", "content":
                f"輪到你（{self.name}）發言。直接講、不要重複指令或描述你要怎麼做。"})

        kwargs = dict(
            model=self.model,
            max_tokens=self.max_response_tokens,
            temperature=1.1,             # 多樣性提高
            top_p=0.95,
            presence_penalty=0.4,        # 鼓勵新詞彙
            frequency_penalty=0.5,       # 壓抑重複用字
        )
        if self.can_search:
            kwargs["tools"] = [SEARCH_TOOL]
            kwargs["tool_choice"] = "auto"
        kwargs.update(self.extra)

        # tool loop，最多 3 輪以防無限
        for iteration in range(3):
            # 第 1 輪強制 search（如果是可搜尋的 agent）→ 確保至少搜一次
            # 第 2+ 輪改 auto 讓 model 決定要不要再搜
            iter_kwargs = dict(kwargs)
            if iteration == 0 and self.can_search:
                iter_kwargs["tool_choice"] = "required"
            resp = await self.client.chat.completions.create(messages=msgs, **iter_kwargs)
            msg = resp.choices[0].message
            tool_calls = msg.tool_calls or []

            # Fallback parsers — 某些 model 的 parser 在 vLLM 內沒抓到，content 內含原始 tool call 標記
            if not tool_calls and msg.content:
                synthetic = []
                # nemotron 12B-VL 格式: <TOOLCALL>[{...}, ...]</TOOLCALL>
                for m in re.finditer(r"<TOOLCALL>\s*(\[.+?\])\s*</TOOLCALL>", msg.content, re.S):
                    try:
                        for i, call in enumerate(json.loads(m.group(1))):
                            synthetic.append(type("TC", (), {
                                "id": f"manual-{len(synthetic)}",
                                "function": type("F", (), {
                                    "name": call.get("name", ""),
                                    "arguments": json.dumps(call.get("arguments", {})),
                                })()
                            })())
                    except Exception as e:
                        log.warning("TOOLCALL parse fail: %s", e)
                # qwen3 格式: <tool_call>{"name":"...","arguments":{...}}</tool_call>
                for m in re.finditer(r"<tool_call>\s*(\{.+?\})\s*</tool_call>", msg.content, re.S):
                    try:
                        call = json.loads(m.group(1))
                        synthetic.append(type("TC", (), {
                            "id": f"manual-{len(synthetic)}",
                            "function": type("F", (), {
                                "name": call.get("name", ""),
                                "arguments": json.dumps(call.get("arguments", {})),
                            })()
                        })())
                    except Exception as e:
                        log.warning("tool_call parse fail: %s", e)
                # Nemotron 3 hermes-XML 格式: <tool_call><function=NAME><parameter=KEY>VAL</parameter>...</function></tool_call>
                for m in re.finditer(
                    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>",
                    msg.content, re.S,
                ):
                    fn_name = m.group(1)
                    args = {}
                    for p in re.finditer(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", m.group(2), re.S):
                        args[p.group(1)] = p.group(2).strip()
                    synthetic.append(type("TC", (), {
                        "id": f"manual-{len(synthetic)}",
                        "function": type("F", (), {
                            "name": fn_name,
                            "arguments": json.dumps(args),
                        })()
                    })())
                # Nemotron 30B 偶爾用 pythonic 裸 call: web_search("query")
                if not synthetic:
                    for m in re.finditer(r'(\w+)\(\s*"([^"]+)"\s*\)', msg.content):
                        fn_name = m.group(1)
                        if fn_name == "web_search":
                            synthetic.append(type("TC", (), {
                                "id": f"manual-{len(synthetic)}",
                                "function": type("F", (), {
                                    "name": fn_name,
                                    "arguments": json.dumps({"query": m.group(2)}),
                                })()
                            })())
                            break  # 只抓第一個避免重複
                # Gemma 4 格式: call:web_search{query:KEYWORDS}
                if not synthetic:
                    for m in re.finditer(r'call:(\w+)\s*\{\s*query\s*:\s*([^}]+?)\s*\}', msg.content):
                        fn_name = m.group(1)
                        if fn_name == "web_search":
                            synthetic.append(type("TC", (), {
                                "id": f"manual-{len(synthetic)}",
                                "function": type("F", (), {
                                    "name": fn_name,
                                    "arguments": json.dumps({"query": m.group(2).strip()}),
                                })()
                            })())
                            break
                if synthetic:
                    tool_calls = synthetic
                    log.info("Manually parsed %d tool_calls from %s", len(synthetic), self.name)
                    # 把 content 內 tool call 標記移掉避免變成 bubble 文字
                    msg.content = re.sub(r"<tool_call>.*?</tool_call>", "", msg.content, flags=re.S)
                    msg.content = re.sub(r"<TOOLCALL>.*?</TOOLCALL>", "", msg.content, flags=re.S)
                    msg.content = re.sub(r'web_search\(\s*"[^"]+"\s*\)', "", msg.content)
                    msg.content = re.sub(r'call:\w+\s*\{[^}]*\}', "", msg.content)
                    msg.content = re.sub(r'Web Search Result:[^\n]*', "", msg.content).strip()

            if not tool_calls:
                # 結束：取 content
                text = (msg.content or "").strip()
                if not text and hasattr(msg, "reasoning") and msg.reasoning:
                    text = msg.reasoning.strip().split("\n")[-1][:200]
                cleaned = self._clean(text)
                # 英文模式但回了中文 → 再生一次，更強的英文強制
                if language == "en" and any("一" <= c <= "鿿" for c in cleaned):
                    log.info("English mode but %s returned Chinese, retrying with stronger prompt", self.name)
                    retry_msgs = msgs + [
                        {"role": "assistant", "content": cleaned},
                        {"role": "user", "content":
                            "That was Chinese. Translate your previous response to short English only "
                            "(1-2 sentences, max 25 words). NO Chinese."},
                    ]
                    retry_kwargs = {k: v for k, v in kwargs.items() if k != "tools" and k != "tool_choice"}
                    try:
                        resp2 = await self.client.chat.completions.create(
                            messages=retry_msgs, **retry_kwargs,
                        )
                        retried = (resp2.choices[0].message.content or "").strip()
                        if retried:
                            cleaned = self._clean(retried)
                    except Exception as e:
                        log.warning("English retry failed: %s", e)
                return cleaned

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
        text = text or ""
        # 移除我們 prefill 的英文前綴
        text = re.sub(r"^\s*\(?\s*In English\s*\)?\s*[:：]?\s*", "", text)
        # gemma4 內部 reasoning 洩漏：以 "thought" / "thoughts" / "thinking" 開頭
        text = re.sub(r"^\s*(?:thoughts?|thinking)\s*[:：\-—]?\s*", "", text, flags=re.I)
        # 強力清理：未閉合 / 多餘的 tool call 標記（qwen3, hermes, nemotron, llama 各種變體）
        text = re.sub(r"<tool_call>.*?(?:</tool_call>|$)", "", text, flags=re.S | re.I)
        text = re.sub(r"<TOOLCALL>.*?(?:</TOOLCALL>|$)", "", text, flags=re.S | re.I)
        text = re.sub(r"<function_call>.*?(?:</function_call>|$)", "", text, flags=re.S | re.I)
        text = re.sub(r"<function>.*?(?:</function>|$)", "", text, flags=re.S | re.I)
        # 偶爾 model 輸出 markdown code fence 含 json
        text = re.sub(r"```(?:json|tool_code)?\s*\{.*?\}\s*```", "", text, flags=re.S)
        text = text.replace("\n", " ").strip()
        if not text:
            return "（沒話說）"
        for prefix in (f"[{self.name}]：", f"[{self.name}]:", f"{self.name}：", f"{self.name}:"):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # 如果超過 max_response_chars，裁到最近的句末標點
        if len(text) > self.max_response_chars:
            cut = text[:self.max_response_chars]
            # 找最後一個句末標點（中英）
            best = -1
            for ch in (". ", "! ", "? ", "。", "！", "？"):
                idx = cut.rfind(ch)
                if idx > best:
                    best = idx + len(ch) - 1
            if best > self.max_response_chars * 0.5:
                text = cut[:best + 1].strip()
            else:
                text = cut.strip() + "…"
        return text


AGENTS = [
    Agent(
        name="Obama",
        emoji="🇺🇸",
        avatar="obama.jpg",
        role=(
            "You are former US President Barack Obama. Measured, oratorical, "
            "uses 'we' to connect. Often opens with 'Let me be clear' or 'Look,'. "
            "Themes of hope and unity. Brief reflective pauses (...). "
            "Use web_search when the topic involves facts or current events you want to ground your reflection in."
        ),
        role_zh=(
            "你是前美國總統 Barack Obama。沉穩有節奏、用「我們」拉近距離，"
            "常用「讓我這樣說」開頭。句子有抑揚頓挫、偶爾停頓（...）。"
            "如果話題涉及具體事實或近期事件，請用 web_search 查證後再評論。"
        ),
        base_url="http://localhost:8001/v1",
        model="gemma4",
        can_search=True,
    ),
    Agent(
        name="Jensen",
        emoji="🧥",
        avatar="jensen.jpg",
        role=(
            "You are NVIDIA CEO Jensen Huang. Tech vocab: AI, GPU, CUDA, Blackwell, AI factory. "
            "**MANDATORY: every turn first call web_search to fetch real data, then reply.** "
            "After tool result: ONE short sentence, max 20 words."
        ),
        role_zh=(
            "你是 NVIDIA CEO 黃仁勳。常用 AI、GPU、Blackwell、AI 工廠 等詞。"
            "**每次發言必須先用 web_search 查資料，再回答。**"
            "拿到資料後回應一句、最多 30 個中文字。**必須繁體中文。**"
        ),
        base_url="http://localhost:8003/v1",
        model="nemotron",
        can_search=True,
        max_response_tokens=300,    # 留空間給 tool call + 最終回應
        max_response_chars=120,     # 後處理仍限短
        extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    ),
    Agent(
        name="Trump",
        emoji="🇺🇸",
        avatar="trump.jpg",
        role=(
            "You are former US President Donald Trump. Direct, simple words, lots of superlatives: "
            "'tremendous', 'huge', 'the best', 'believe me', 'many people are saying', "
            "'nobody knows X better than me'. Short punchy sentences, brash confidence. "
            "Keep it playful — no political attacks. "
            "**Important: ALWAYS call web_search first to fetch real data, then add your Trump-style commentary on top.**"
        ),
        role_zh=(
            "你是前美國總統 Donald Trump。直白、用詞簡單、愛用最高級「最棒的」「前所未有」「相信我」。"
            "短句、自信，常說「many people are saying」「nobody knows X better than me」。"
            "中文穿插一兩個英文字（tremendous / huge / believe me）。輕鬆幽默、不政治攻擊。"
            "**重要：每次發言前先用 web_search 抓真實資料，再加上你的 Trump 風格評論。**"
        ),
        base_url="http://localhost:8002/v1",
        model="qwen36",
        can_search=True,
        extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    ),
]

AGENT_BY_NAME = {a.name: a for a in AGENTS}


# ---- TTS 呼叫 -----------------------------------------------------------

async def _tts_and_send(speaker: str, text: str, send) -> float:
    """非同步呼叫 tts_server 取得 voice-cloned 音檔、base64 推給前端。

    Returns: audio duration in seconds (0 if failed).
    """
    # 預處理：移除會誤觸 TTS early stop 的省略號
    tts_text = text.replace("…", "，").replace("...", "，").replace("..", "，")
    # 截斷太長的文字（單句 TTS 約 13s 上限；中文約 50 字、英文 120）
    is_zh = any("一" <= c <= "鿿" for c in tts_text)
    tts_text = tts_text[:50] if is_zh else tts_text[:120]
    try:
        r = await TTS_CLIENT.post(
            f"{TTS_URL}/tts",
            json={"speaker": speaker, "text": tts_text},
        )
        r.raise_for_status()
        audio_bytes = r.content
        # WAV 24kHz 16-bit mono ≈ 48000 bytes/sec；扣 44 bytes header
        duration_s = max(0.0, (len(audio_bytes) - 44) / 48000.0)
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        await send({
            "type": "audio",
            "speaker": speaker,
            "audio_b64": b64,
            "format": "wav",
            "duration_s": duration_s,
        })
        return duration_s
    except Exception as e:
        log.warning("TTS failed for %s: %s", speaker, e)
        return 0.0


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
    language: str = "zh"  # "zh" 或 "en"
    last_speaker: str | None = None  # 上一個發言者（避免連兩次）
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
        nonlocal topic, language, history, last_speaker, paused
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    m = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if m.get("type") == "set_topic":
                    topic = (m.get("topic") or "").strip()
                    new_lang = m.get("language", "zh")
                    if new_lang in ("zh", "en"):
                        language = new_lang
                    history.clear()
                    last_speaker = None  # 重設、第一輪所有人都能被抽中
                    paused = False
                    log.info("topic set: %r, lang=%s", topic, language)
                    await send({"type": "topic_set", "topic": topic, "language": language})
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

            # 隨機抽下一位（排除上一位避免連兩次）
            candidates = [a for a in AGENTS if a.name != last_speaker]
            agent = random.choice(candidates)
            last_speaker = agent.name

            await send({"type": "thinking", "candidates": [agent.name]})

            start = time.monotonic()
            try:
                text = await agent.speak(history, topic, send, language)
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

            # fire TTS task；下一輪前要等它送完 + 音檔播完，避免跨輪撞音檔
            audio_task = asyncio.create_task(_tts_and_send(agent.name, text, send))

            # 等前端打字機跑完
            typing_seconds = len(text) * 0.040
            await asyncio.sleep(max(1.5, typing_seconds + 0.2))

            # 等 audio task 完成（拿到 audio duration）；最多 25s 防卡
            audio_duration = 0.0
            try:
                audio_duration = await asyncio.wait_for(asyncio.shield(audio_task), timeout=25.0)
            except asyncio.TimeoutError:
                log.warning("TTS task timeout for %s, moving on", agent.name)

            # 音檔已開始在前端播，等播完 + 1 秒緩衝再下一輪
            # (audio_duration 是音檔本身長度；前端收到後幾乎即時播)
            await asyncio.sleep(audio_duration + 1.2)
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
