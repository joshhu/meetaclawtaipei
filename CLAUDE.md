# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概要

在 NVIDIA DGX Spark（GB10 Blackwell, 128GB UMA, ARM64, CUDA 13）上同時跑 **3 個 vLLM 服務**，並在其上搭一個 demo：三個 LLM 分別扮演 Trump / Obama / Jensen Huang，於虛擬「AI Summit 台北」舞台輪流對談、可 web search、並用 Qwen3-TTS voice clone 本人聲音同步發聲。

兩層結構：
1. **後端推理層** — `docker-compose.yml` 定義的 3 個 vLLM container（OpenAI 相容 API）。
2. **demo 應用層** — `roommates/`：FastAPI + WebSocket 主 server（5050）+ Qwen3-TTS server（5051）。

## 常用指令

```bash
# 一鍵冷啟動全部（3 vLLM → TTS → web server，依序、約 15-20 分鐘）
./start_all.sh

# 停止全部（kill TTS/web，stop 3 個 container，不刪 image）
./stop_all.sh

# 煙霧測試：驗證 3 個 vLLM 健康 + chat 完成 + 記憶體狀態
./test_all.sh

# 單獨啟動 demo（前提：3 個 vLLM 已在跑）
cd roommates && uv sync && ./run_all.sh

# 看狀態 / 日誌 / 監控
docker compose ps
docker compose logs -f gemma4
watch -n 2 'free -h && echo && docker stats --no-stream'
```

vLLM container 必須**一個一個依序啟動**，每個之間 `drop_caches`，等 `/v1/models` 就緒再啟下一個（`start_all.sh` 已封裝此邏輯）。同時拉起會 OOM。

Python 一律用 `uv`（`uv sync` / `uv run`），**不要用 pip**。記憶體一律看 `free -h`（UMA，不是 `nvidia-smi`）。

## 後端架構（docker-compose.yml）

| Port | model name | 模型 | parser 設定 |
|------|-----------|------|-------------|
| 8001 | `gemma4` | nvidia/Gemma-4-31B-IT-NVFP4 | tool: pythonic |
| 8002 | `qwen36` | mmangkad/Qwen3.6-27B-NVFP4 | reasoning: qwen3, tool: hermes |
| 8003 | `nemotron` | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | tool: pythonic, reasoning plugin |
| 8004 | `deepseek-r1` | DeepSeek-R1-Distill-32B-NVFP4 | `optional` profile，預設不啟動 |

**關鍵限制：128GB UMA 只能穩定跑 3 個 28-31B 級模型同時。** 跑第 4 個（deepseek-r1）前必須先停掉一個。`gpu-memory-utilization` 三者合計約 0.90-0.95，餘裕極薄。

- `nano_v3_reasoning_parser.py` 透過 volume 掛進 nemotron container（`/opt/parsers/`）作 reasoning parser plugin。
- README.md 部分表格仍寫舊配置（Nemotron 12B-VL、util 0.30 等）— **以 docker-compose.yml 為準**，它是最新的真實設定。

## demo 應用架構（roommates/）

- **`main.py`** — 主 FastAPI server。`AGENTS` list 定義三個角色（name、persona prompt 中英雙版、對應 vLLM endpoint、能否 search）。`/ws` WebSocket 跑主迴圈：隨機抽一位（排除上一位）→ `Agent.speak()` 產生文字 → 推 `say` event → fire TTS task → **await TTS + 音檔播完才進下一輪**（避免跨輪音檔重疊）。
- **`tts_server.py`** — 獨立 FastAPI server。lifespan 載入 `Qwen/Qwen3-TTS-12Hz-1.7B-Base`（~35s）並預讀 `voices/*.wav` reference audio。`POST /tts` 用 `x_vector_only_mode` voice clone 回傳 WAV。
- **`static/index.html`** — 單檔前端 UI（WebSocket client + 打字機 bubble + Audio 播放）。

### Agent.speak() 的 tool-use loop（main.py 核心複雜點）

各模型的 vLLM tool parser 行為不一致，content 常洩漏未解析的 tool call 標記。`speak()` 因此有大量 **fallback 正則 parser**，手動從 `msg.content` 抽 tool call（涵蓋 `<TOOLCALL>`、`<tool_call>` hermes-XML、qwen3 JSON、pythonic 裸 call `web_search("q")`、gemma4 `call:web_search{query:...}` 等格式）。新增/換模型時，多半要在這裡補對應格式的 parser，並在 `_clean()` 補清理規則。

- 第一輪對可 search 的 agent 強制 `tool_choice="required"`，之後 `auto`，最多 3 輪。
- 英文模式下若模型回了中文，會自動 retry 要求翻成英文。
- `web_search()`：有 `BRAVE_API_KEY`（從專案根 `.env` 載入）走 Brave Search API，失敗或無 key fallback 回 DuckDuckGo（`ddgs`）。

### 角色 → 模型 → 聲音映射

| 講者 | model | 聲音 ref |
|------|-------|----------|
| Trump | qwen36 | voices/trump.wav |
| Obama | gemma4 | voices/obama.wav |
| Jensen | nemotron | voices/huang.wav |

新增角色：改 `main.py` 的 `AGENTS` + `tts_server.py` 的 `SPEAKERS` dict，並放對應 `voices/<name>.wav`（24kHz mono）+ `.txt` 逐字稿，重啟 TTS server。

## 冷啟動 OOM 注意

`start_all.sh` 在 nemotron 剛就緒、記憶體仍處 warmup 尖峰時立刻拉 TTS+web，可能撞尖峰被 OOM killer 殺掉（3 個 LLM 回 200 但 5050/5051 沒監聽）。徵兆：`free -h` avail < 9Gi、swap 接近滿。處理：等記憶體回落後再手動起 TTS+web，或調降 nemotron `gpu-memory-utilization` 騰餘裕。這是記憶體時序問題，非設定壞掉。

## 環境慣例

- 回覆一律用台灣繁體中文。
- secrets 從 `.env` / 環境變數讀，勿寫進程式碼或 commit。
- 服務用 docker/podman，避免 native 安裝。
