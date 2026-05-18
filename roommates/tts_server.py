"""Qwen3-TTS 1.7B voice clone server。

啟動時 load 模型 + 預先生成 3 位的 voice clone prompts，
之後每次請求 ~5-15 秒生成 voice-cloned 音檔。

POST /tts {speaker, text, language} → audio/wav
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tts_server")

VOICES_DIR = Path(__file__).parent / "voices"

# speaker → (wav 檔, txt 檔)
SPEAKERS = {
    "Trump":  ("trump.wav",  "trump.txt"),
    "Obama":  ("obama.wav",  "obama.txt"),
    "Jensen": ("huang.wav",  "huang.txt"),
}

# Module 級 state（fastapi lifespan 初始化）
STATE: dict = {"model": None, "prompts": {}}


def detect_language(text: str) -> str:
    """簡易語言判斷：含 CJK 字 → Chinese，否則 English。"""
    for ch in text:
        if "一" <= ch <= "鿿":
            return "Chinese"
    return "English"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading Qwen3-TTS-12Hz-1.7B-Base ...")
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    STATE["model"] = model
    log.info("Model loaded in %.1fs", time.time() - t0)

    # 預先生成 3 位 voice clone prompts（一次性開銷）
    for speaker, (wav, txt) in SPEAKERS.items():
        wav_path = VOICES_DIR / wav
        txt_path = VOICES_DIR / txt
        if not wav_path.exists() or not txt_path.exists():
            log.warning("Missing voice files for %s (%s, %s) — skipping", speaker, wav_path, txt_path)
            continue
        ref_text = txt_path.read_text(encoding="utf-8").strip()[:300]
        log.info("Building voice clone prompt for %s ...", speaker)
        t0 = time.time()
        try:
            prompt = model.create_voice_clone_prompt(
                ref_audio=str(wav_path),
                ref_text=ref_text,
            )
            STATE["prompts"][speaker] = prompt
            log.info("  %s prompt built in %.1fs", speaker, time.time() - t0)
        except Exception as e:
            log.exception("create_voice_clone_prompt failed for %s: %s", speaker, e)
            # fallback：runtime 才用 ref_audio + ref_text
            STATE["prompts"][speaker] = None

    log.info("Ready. Speakers: %s", list(STATE["prompts"].keys()))
    yield
    log.info("Shutting down")


app = FastAPI(title="Qwen3-TTS server", lifespan=lifespan)


class TTSRequest(BaseModel):
    speaker: str
    text: str
    language: Optional[str] = None  # 不傳就自動偵測
    max_new_tokens: Optional[int] = None  # 不傳就依字長動態估


def estimate_tokens(text: str, language: str) -> int:
    """估算音檔上限。中文每字 ~0.3 秒、英文每字 ~0.4 秒；12Hz tokenizer。"""
    if language == "Chinese":
        secs = max(2.0, len(text) * 0.35 + 1.0)
    else:
        words = max(1, len(text.split()))
        secs = max(2.0, words * 0.45 + 1.0)
    return min(int(secs * 13), 200)  # 13 tokens/sec (留 buffer)，上限 200


@app.get("/health")
async def health():
    return {
        "ok": STATE["model"] is not None,
        "speakers": list(STATE["prompts"].keys()),
    }


@app.post("/tts")
async def tts(req: TTSRequest):
    if STATE["model"] is None:
        raise HTTPException(503, "Model not loaded yet")
    speaker = req.speaker
    if speaker not in SPEAKERS:
        raise HTTPException(400, f"Unknown speaker {speaker}; available={list(SPEAKERS)}")
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(400, "Empty text")
    if len(text) > 300:
        text = text[:300]

    language = req.language or detect_language(text)
    model = STATE["model"]
    prompt = STATE["prompts"].get(speaker)
    max_new_tokens = req.max_new_tokens or estimate_tokens(text, language)

    def _gen():
        kw = dict(text=text, language=language, max_new_tokens=max_new_tokens, do_sample=False)
        if prompt is not None:
            kw["voice_clone_prompt"] = prompt
        else:
            # fallback：直接帶 ref_audio + ref_text
            wav, txt = SPEAKERS[speaker]
            kw["ref_audio"] = str(VOICES_DIR / wav)
            kw["ref_text"] = (VOICES_DIR / txt).read_text(encoding="utf-8").strip()[:300]
        return model.generate_voice_clone(**kw)

    t0 = time.time()
    try:
        wavs, sr = await asyncio.to_thread(_gen)
    except Exception as e:
        log.exception("generate failed: %s", e)
        raise HTTPException(500, f"TTS failed: {e}")
    elapsed = time.time() - t0
    dur = wavs[0].shape[0] / sr
    log.info("TTS [%s] (%s) %d chars → %.2fs audio in %.1fs", speaker, language, len(text), dur, elapsed)

    # 寫成 WAV bytes
    buf = io.BytesIO()
    sf.write(buf, wavs[0], sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.post("/tts_base64")
async def tts_base64(req: TTSRequest):
    """同 /tts 但回傳 JSON {audio_b64, sample_rate, duration_s}，方便 WebSocket 傳輸。"""
    resp = await tts(req)
    audio_bytes = resp.body
    return {
        "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
        "format": "wav",
    }
