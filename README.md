# meetaclawtaipei

在 NVIDIA DGX Spark（GB10 Blackwell, 128GB UMA, ARM64, CUDA 13）上同時跑 3 個 vLLM 模型，並用它們打造一個舞台 demo：**Trump、Obama、黃仁勳**三人在「AI Summit 台北」輪流對談，會即時上網查資料，還會用 voice clone 模仿本人聲音講話。

---

## 它由兩部分組成

**1. 推理後端** — 3 個 vLLM 服務，OpenAI 相容 API：

| Port | 名稱 | 模型 | 量化 |
|---|---|---|---|
| 8001 | `gemma4` | Gemma-4-31B-IT | NVFP4 |
| 8002 | `qwen36` | Qwen3.6-27B | NVFP4 |
| 8003 | `nemotron` | Nemotron-3-Nano-30B-A3B | NVFP4 |

> 128GB UMA 只夠穩定跑這 3 個。`docker-compose.yml` 另有第 4 個 `deepseek-r1`（`optional` profile，預設關閉），要跑得先停一個。

**2. 舞台 demo**（`roommates/`）— FastAPI 網頁，三個模型各扮演一位講者：

| 講者 | 用哪個模型 | 聲音 |
|---|---|---|
| Trump | qwen36 | `voices/trump.wav` |
| Obama | gemma4 | `voices/obama.wav` |
| Jensen | nemotron | `voices/huang.wav` |

三人都能呼叫 `web_search`（優先用 Brave，沒設 key 則退回 DuckDuckGo），發言時由獨立的 Qwen3-TTS server voice clone 出聲。

---

## 快速開始

```bash
./start_all.sh
```

這一支腳本會依序帶起整個 stack（3 個 vLLM → TTS server → 網頁 server），首次約 15-20 分鐘。完成後開瀏覽器：

```
http://<dgx-spark-ip>:5050
```

輸入主題、選語言、按開始，三人就會輪流發言並出聲。

停止：

```bash
./stop_all.sh
```

---

## 環境需求

- DGX Spark（GB10 Blackwell, 128GB UMA, Ubuntu 24.04 ARM64, CUDA 13）
- Docker 28+ 含 NVIDIA runtime
- 已 pull `vllm-node:latest`（NVIDIA 官方 sbsa vLLM image）
- 約 80GB 磁碟放模型快取
- demo 的 Python 依賴用 `uv`（非 pip）

選用設定寫在 `.env`（範本見 `.env.example`）：

```
HF_HOME=/home/joshhu/.cache/huggingface
BRAVE_API_KEY=...   # 設了才走 Brave search，否則自動用 DuckDuckGo
```

---

## 手動操作

啟動腳本之外，也可以分開來跑。

**只啟動 demo**（3 個 vLLM 已在跑時）：

```bash
cd roommates
uv sync
./run_all.sh
```

**逐一啟動 vLLM**（除錯用）。每個之間要 `drop_caches`、等就緒再起下一個，同時拉起會 OOM：

```bash
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
docker compose up -d gemma4
until curl -sf http://localhost:8001/v1/models >/dev/null; do sleep 10; done
# qwen36(8002)、nemotron(8003) 同理
```

**直接打 API**：

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"你好"}],"max_tokens":200}'
```

**常用管理指令**：

```bash
docker compose ps                   # 看狀態
docker compose logs -f gemma4       # 看日誌
./test_all.sh                       # 煙霧測試
free -h                             # 看記憶體（UMA 看這個，不是 nvidia-smi）
```

---

## 疑難排解

| 症狀 | 解法 |
|---|---|
| 啟動失敗、container 直接 exit | KV cache 不夠，提高 `--gpu-memory-utilization` 或縮 `--max-model-len` |
| 第一次啟動很慢（>10 分鐘）| flashinfer 在做 fp4 autotuning，正常，之後會快取 |
| 3 個 LLM 都通但網頁打不開 | 冷啟動記憶體尖峰把 TTS/web OOM 掉了，等記憶體回落再手動起，或調降 nemotron 的 util |
| demo 沒聲音 | 瀏覽器首次需點一下頁面解鎖 autoplay |
| 模型只回 reasoning 沒有內容 | thinking 沒結束，把 `max_tokens` 加大 |
| 系統卡頓、狂 swap | `drop_caches` 釋放 page cache，或停掉一個服務 |

---

## 專案結構

```
docker-compose.yml          3 個 vLLM 服務定義（+ optional deepseek-r1）
start_all.sh / stop_all.sh  一鍵啟停整個 stack
test_all.sh                 後端健康檢查
nano_v3_reasoning_parser.py 掛進 nemotron 的 reasoning parser
roommates/                  舞台 demo（網頁 + TTS server）→ 見 roommates/README.md
.env.example                環境變數範本
```
