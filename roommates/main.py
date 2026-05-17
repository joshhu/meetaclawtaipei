"""舞台三巨頭：Trump / Jensen / Obama 演講辯論 — FastAPI + WebSocket race orchestrator。"""

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

STAGE_INTRO = (
    "現在是 2026 年「AI Summit 台北」的開幕論壇，三位重量級講者同台對談："
    "主題是「AI 與人類的下一個十年」。台下坐滿觀眾，你們輪流發言，"
    "互相補充、偶爾交鋒。請完全以你的人物口吻發言，模仿那個人公開的講話風格。"
)

COMMON_RULES = (
    "重要規則：\n"
    "1. 每次只說 1-2 句、最多 50 個中文字（短而有力的金句最讚）\n"
    "2. 用繁體中文（如果是 Trump 可以混一兩個簡單英文字）\n"
    "3. 不要列點、不要 markdown、不要表情符號、不要旁白動作\n"
    "4. 要針對上一個講者剛說的話接話（呼應、補充、或反駁）\n"
    "5. 不要每次自我介紹，假設大家都認識你\n"
    "6. 內容輕鬆幽默為主，不要碰敏感政治攻擊\n"
)


@dataclass
class Agent:
    name: str
    emoji: str
    role: str  # 用在 system prompt
    base_url: str
    model: str
    avatar: str = ""  # 前端用的圖檔名（在 static/）
    extra: dict = field(default_factory=dict)  # extra body params for OpenAI

    def __post_init__(self):
        self.client = AsyncOpenAI(base_url=self.base_url, api_key="not-needed", timeout=90.0)

    def system_prompt(self) -> str:
        others = [a.name for a in AGENTS if a.name != self.name]
        return (
            f"你是 {self.name}。{self.role}\n\n"
            f"{STAGE_INTRO}\n\n"
            f"同台的另外兩位是 {' 和 '.join(others)}。\n\n"
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
        name="Obama",
        emoji="🇺🇸",
        avatar="obama.jpg",
        role=(
            "你是前美國總統 Barack Obama。講話風格沉穩有節奏、會用「我們」拉近距離，"
            "喜歡引用希望與團結的概念，常常用「Let me be clear」或「讓我這樣說」開頭。"
            "句子要有抑揚頓挫，偶爾用一個短停頓（...）。中文以正式而溫暖的口吻。"
        ),
        base_url="http://localhost:8001/v1",
        model="gemma4",
    ),
    Agent(
        name="Jensen",
        emoji="🧥",
        avatar="jensen.jpg",
        role=(
            "你是 NVIDIA CEO 黃仁勳（Jensen Huang）。台裔美國人，喜歡穿黑色皮夾克。"
            "講話會大量使用 AI、GPU、CUDA、Blackwell、加速運算、AI 工廠 這類技術詞，"
            "語氣興奮、對未來樂觀，偶爾混一兩個英文單字。"
            "常說「我們正在進入一個全新的運算時代」、「the more you buy, the more you save」。"
            "**必須用繁體中文，絕對不要用簡體字（不能用「当然/单/广/会」要寫「當然/單/廣/會」）。**"
        ),
        base_url="http://localhost:8003/v1",
        model="nemotron",
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
        ),
        base_url="http://localhost:8002/v1",
        model="qwen36",
        extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    ),
]

AGENT_BY_NAME = {a.name: a for a in AGENTS}

# ---- race orchestrator ---------------------------------------------------


async def race_round(
    history: list[dict],
    last_winner: str | None,
    speak_count: dict[str, int],
) -> tuple[str, str, float]:
    """三人同時 generate，最快完成的勝出；上輪 winner 不參與本輪。

    公平規則：如果任何人 speak_count 比最多者少 3 以上，下一輪只有他能講（solo）。

    Returns (speaker_name, text, elapsed_seconds)。
    """
    max_c = max(speak_count.values()) if speak_count else 0
    starving = [name for name, c in speak_count.items() if max_c - c >= 2]
    if starving:
        # 強制給最久沒講的人 solo
        candidates = [a for a in AGENTS if a.name in starving]
        log.info("Fairness kick-in: %s solo (counts=%s)", starving, speak_count)
    else:
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
        "agents": [
            {"name": a.name, "emoji": a.emoji, "avatar": a.avatar, "endpoint": a.base_url}
            for a in AGENTS
        ],
    }


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    log.info("client connected")

    history: list[dict] = []
    last_winner: str | None = None
    speak_count: dict[str, int] = {a.name: 0 for a in AGENTS}

    async def send(payload: dict):
        await ws.send_text(json.dumps(payload, ensure_ascii=False))

    # 初始 hello
    await send({
        "type": "hello",
        "agents": [{"name": a.name, "emoji": a.emoji, "avatar": a.avatar} for a in AGENTS],
    })

    try:
        while True:
            # 通知前端：全員「思考中」
            await send({
                "type": "thinking",
                "candidates": [a.name for a in AGENTS if a.name != last_winner],
            })

            try:
                speaker, text, elapsed = await race_round(history, last_winner, speak_count)
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
                speak_count[speaker] = speak_count.get(speaker, 0) + 1
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
