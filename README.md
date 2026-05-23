# meetaclawtaipei — DGX Spark 多 LLM 舞台

在 NVIDIA DGX Spark（GB10 Blackwell, 128GB UMA, ARM64, CUDA 13）上同時跑 **3 個 vLLM 服務**，並在其上搭一個 demo：三個 LLM 分別扮演 **Trump / Obama / Jensen Huang**，於虛擬「AI Summit 台北」舞台輪流對談、可即時 web search，並用 **Qwen3-TTS voice clone 本人聲音**同步發聲。

> **實測結論**：128GB UMA 在 28-31B 級模型上能穩定跑 **3 個同時**。原本規劃的第 4 個（DeepSeek-R1-Distill-32B）會 OOM——每個 vLLM process 除了權重還有 KV cache + CUDA graph + encoder cache 共 15-20GB 固定開銷，3 個就吃滿。

## 系統架構

兩層：

```
┌──────────────────────── demo 應用層 (roommates/) ────────────────────────┐
│  主 web server (FastAPI + WebSocket, :5050)   TTS server (Qwen3-TTS, :5051) │
│        三角色輪流發言 + web_search             voice clone 本人聲音           │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 │ OpenAI 相容 API
┌────────────────────────── 後端推理層 (docker-compose) ─────────────────────┐
│  :8001 gemma4      :8002 qwen36      :8003 nemotron      （共用 128GB UMA） │
└────────────────────────────────────────────────────────────────────────────┘
```

## 系統需求

- NVIDIA DGX Spark（GB10 Blackwell, 128GB UMA, Ubuntu 24.04 ARM64, CUDA 13）
- Docker 28+ with NVIDIA runtime
- 至少 80GB 可用磁碟（模型快取）
- 已預先 pull `vllm-node:latest`（NVIDIA 官方 sbsa vLLM container）
- Python 套件用 `uv`（demo 層），不用 pip

## 模型配置（docker-compose.yml）

| Port | model name | 模型 | 量化 | gpu-mem-util | parser |
|------|-----------|------|------|--------------|--------|
| 8001 | `gemma4` | nvidia/Gemma-4-31B-IT-NVFP4 | NVFP4 | 0.40 | tool: pythonic |
| 8002 | `qwen36` | mmangkad/Qwen3.6-27B-NVFP4 | NVFP4 | 0.30 | reasoning: qwen3 / tool: hermes |
| 8003 | `nemotron` | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | NVFP4 | 0.25 | tool: pythonic + reasoning plugin |
| ~~8004~~ | `deepseek-r1` | DeepSeek-R1-Distill-Qwen-32B-NVFP4 | NVFP4 | — | `optional` profile（預設不啟動）|

3 個合計 GPU util ≈ 0.95，幾乎吃滿 128GB UMA，餘裕極薄。

## 快速開始

### 一鍵冷啟動（推薦）

```bash
./start_all.sh      # 依序起 3 個 vLLM → TTS server → 主 web server，約 15-20 分鐘
```

完成後開瀏覽器 `http://<dgx-spark-ip>:5050`，輸入主題、按開始討論，三人輪流發言並出聲。

> ⚠️ **冷啟動 OOM 注意**：3 個 vLLM 剛就緒時記憶體處 warmup 尖峰，若立刻拉 TTS+web 可能被 OOM killer 殺掉（3 個 LLM 回 200 但 5050/5051 沒監聽，`free -h` avail < 9Gi）。徵兆出現時等記憶體回落後再手動起 TTS+web，或調降 nemotron `gpu-memory-utilization` 騰餘裕。這是記憶體時序問題，非設定壞掉。

### 手動分步（除錯用）

```bash
# 1. 確認 image 與模型
docker images vllm-node:latest
ls ~/.cache/huggingface/hub/ | grep -E "Gemma-4-31B|Qwen3.6-27B|Nemotron-3-Nano-30B"

# 2. 必須一個一個來：每個之間 drop_caches、等就緒再啟下一個（同時拉起會 OOM）
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
docker compose up -d gemma4
until curl -s --max-time 2 http://localhost:8001/v1/models | jq -e '.data[0].id' >/dev/null 2>&1; do sleep 10; done

sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
docker compose up -d qwen36
until curl -s --max-time 2 http://localhost:8002/v1/models | jq -e '.data[0].id' >/dev/null 2>&1; do sleep 10; done

sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
docker compose up -d nemotron
until curl -s --max-time 2 http://localhost:8003/v1/models | jq -e '.data[0].id' >/dev/null 2>&1; do sleep 10; done

# 3. 煙霧測試
./test_all.sh

# 4. 只啟動 demo（3 個 vLLM 已在跑時）
cd roommates && uv sync && ./run_all.sh
```

> 首次啟動每個模型載入 5-7 分鐘（讀權重 + flashinfer fp4_gemm autotuning + CUDA graph 編譯）；TTS 模型載入 ~35s。有快取後第二次快很多。

## demo 應用（roommates/）

三個 LLM 扮演三巨頭，主迴圈隨機抽一位（排除上一位避免連兩次）發言，可呼叫 `web_search`，並用 Qwen3-TTS voice clone 同步發聲。主迴圈會 await TTS 與音檔播完才進下一輪，避免跨輪音檔重疊。

| 講者 | model | 能 search? | 聲音 ref |
|------|-------|-----------|----------|
| 🇺🇸 Trump | qwen36 | ✅ | voices/trump.wav |
| 🇺🇸 Obama | gemma4 | ✅ | voices/obama.wav |
| 🧥 Jensen | nemotron | ✅ | voices/huang.wav |

- **web search**：設了 `BRAVE_API_KEY`（從專案根 `.env` 載入）走 Brave Search API，失敗或無 key 則 fallback 回 DuckDuckGo。
- **角色設定**：在 `roommates/main.py` 的 `AGENTS` list（persona prompt 中英雙版 + endpoint + 是否帶 tools）。
- **新增角色**：改 `main.py` 的 `AGENTS` + `tts_server.py` 的 `SPEAKERS`，放對應 `voices/<name>.wav`（24kHz mono）+ `.txt` 逐字稿，重啟 TTS server。

詳見 [`roommates/README.md`](roommates/README.md)。

## 直接打後端 API

OpenAI 相容。`model` 可填 `gemma4`、`qwen36`、`nemotron`：

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"你好"}],"max_tokens":200}'
```

```python
from openai import OpenAI
agents = {
    "general":   OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed"),  # gemma4
    "reasoning": OpenAI(base_url="http://localhost:8002/v1", api_key="not-needed"),  # qwen36
    "nemotron":  OpenAI(base_url="http://localhost:8003/v1", api_key="not-needed"),
}
r = agents["reasoning"].chat.completions.create(
    model="qwen36",
    messages=[{"role": "user", "content": "解開這個邏輯題：..."}],
    max_tokens=1000,
)
```

**注意**：qwen36 / nemotron 預設可走 thinking 模式，`max_tokens` 太小時最終回應會在 `.choices[0].message.reasoning` 而非 `.content`（demo 已對這兩者關 thinking）。

## 啟動第 4 個模型（DeepSeek，可選）

預設不啟動。要跑必須先停掉其中 1 個：

```bash
docker compose stop nemotron
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
docker compose --profile optional up -d deepseek-r1
```

## 管理指令

```bash
./stop_all.sh                                               # 停全部（TTS/web + 3 container，不刪 image）
docker compose ps                                           # 看狀態
docker compose logs -f gemma4                               # 看單一日誌
docker compose restart gemma4                               # 重啟單一
watch -n 2 'free -h && echo && docker stats --no-stream'    # 監控（UMA 用 free -h，非 nvidia-smi）
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null  # 釋放 page cache
```

## 疑難排解

| 症狀 | 解法 |
|------|------|
| `No available memory for the cache blocks` | 提高 `--gpu-memory-utilization`（gemma4 建議 0.40 起跳）|
| 啟動到一半 container exit | KV cache 不夠；提高 utilization 或縮 `--max-model-len` |
| 啟動慢（>10 min）| flashinfer 在做 fp4_gemm autotuning，第一次正常，之後快取 |
| 3 個 LLM 都回 200 但 5050/5051 沒監聽 | 冷啟動 OOM 尖峰殺掉 TTS/web；等記憶體回落後再手動起，或降 nemotron util |
| qwen36/nemotron 回應只有 reasoning | thinking 沒結束；`max_tokens` 加到 800+，或關 thinking |
| demo 沒聲音 | 瀏覽器首次需點頁面解鎖 autoplay；F12 Console 看有無音檔 base64 |
| 系統卡頓 / 嚴重 swap | `drop_caches` 釋放 page cache，或停一個服務 |
| `nvidia` runtime not found | `sudo apt install nvidia-container-toolkit && sudo systemctl restart docker` |

## 檔案

- `docker-compose.yml` — 3 服務定義（+ deepseek-r1 為 optional profile）— **後端真實設定以此為準**
- `start_all.sh` / `stop_all.sh` — 一鍵啟停全 stack
- `test_all.sh` — 健康檢查 + chat 完成測試 + 記憶體狀態
- `nano_v3_reasoning_parser.py` — 掛進 nemotron container 的 reasoning parser plugin
- `roommates/` — demo 應用（FastAPI WebSocket 主 server + Qwen3-TTS server + 前端）
- `.env.example` — `HF_HOME`、`BRAVE_API_KEY` 等環境變數範本
