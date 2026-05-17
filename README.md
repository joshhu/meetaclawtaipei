# meetaclawtaipei — DGX Spark 多 LLM Agent 平台

在 NVIDIA DGX Spark（GB10 Blackwell, 128GB UMA, ARM64, CUDA 13.2）上同時跑 **3 個 vLLM 服務**，作為多 agent 合作系統的後端。Blackwell 原生 NVFP4 / FP8 量化。

> **實測結論**：128GB UMA 在 28B 級模型上能穩定跑 **3 個同時**。原本規劃的 4 個（含 DeepSeek-R1-Distill-32B）會 OOM（即使全部 NVFP4），因為每個 vLLM process 除了權重還有 KV cache + CUDA graph + multimodal encoder cache 共 15-20GB 固定開銷。

## 系統需求

- NVIDIA DGX Spark（GB10 Blackwell, 128GB UMA, Ubuntu 24.04 ARM64, CUDA 13）
- Docker 28+ with NVIDIA runtime
- 至少 80GB 可用磁碟（模型快取）
- 已預先 pull `vllm-node:latest`（NVIDIA 官方 sbsa vLLM container, CUDA 13.2, vLLM 0.21.1rc1）

## 模型配置

| Port | 角色 | 模型 | 量化 | 實際記憶體 |
|------|------|------|------|------------|
| 8001 | 通用 dense + 多模態 | `nvidia/Gemma-4-31B-IT-NVFP4` | NVFP4 | ~50GB |
| 8002 | 長 context + reasoning | `mmangkad/Qwen3.6-27B-NVFP4` | NVFP4 | ~30GB |
| 8003 | 視覺 (VL) | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8` | FP8 | ~20GB |
| ~~8004~~ | ~~深度推理~~ | ~~DeepSeek-R1-Distill-32B-NVFP4~~ | NVFP4 | optional（須先停一個）|

**3 個合計約 106GB（121GB UMA 中），swap 用 ~9GB 緩衝**。

## 快速開始

```bash
# 1. 確認 image 在
docker images vllm-node:latest

# 2. 確認模型已下載
ls /home/joshhu/.cache/huggingface/hub/ | grep -E "Gemma-4-31B|Qwen3.6-27B|Nemotron-Nano-12B"

# 3. 啟動服務（必須一個一個來，每個等就緒再啟下一個）
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

docker compose up -d gemma4
# 等就緒（約 5-6 分鐘，看到 endpoint 回應）
until curl -s --max-time 2 http://localhost:8001/v1/models | jq -e '.data[0].id' >/dev/null 2>&1; do sleep 10; done

docker compose up -d qwen36
until curl -s --max-time 2 http://localhost:8002/v1/models | jq -e '.data[0].id' >/dev/null 2>&1; do sleep 10; done

docker compose up -d nemotron
until curl -s --max-time 2 http://localhost:8003/v1/models | jq -e '.data[0].id' >/dev/null 2>&1; do sleep 10; done

# 4. 煙霧測試
./test_all.sh
```

> 首次啟動每個模型載入 5-7 分鐘（讀權重 + flashinfer fp4_gemm autotuning + CUDA graph 編譯）。第二次有快取會快很多。

## API 用法

OpenAI 相容 API。範例：

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4",
    "messages": [{"role":"user","content":"你好"}],
    "max_tokens": 200
  }'
```

可用 `model` 名稱：`gemma4`、`qwen36`、`nemotron`。

**注意**：Qwen3.6 預設 thinking 模式，會先輸出 reasoning 再給最終答案。`max_tokens` 至少要 500+ 才看得到最終回應，否則回應會在 `.choices[0].message.reasoning` 而非 `.content`。

## 多 agent 整合範例

```python
from openai import OpenAI

agents = {
    "general":   OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed"),
    "reasoning": OpenAI(base_url="http://localhost:8002/v1", api_key="not-needed"),
    "vision":    OpenAI(base_url="http://localhost:8003/v1", api_key="not-needed"),
}

result = agents["reasoning"].chat.completions.create(
    model="qwen36",
    messages=[{"role": "user", "content": "解開這個邏輯題：..."}],
    max_tokens=1000,
)
```

可搭配 LangGraph、AutoGen、CrewAI，或自寫 orchestrator。

## 啟動 4 個模型（DeepSeek 可選）

預設 deepseek-r1 不會啟動。如果要跑它，必須先停掉其中 1 個：

```bash
docker compose stop nemotron
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
docker compose --profile optional up -d deepseek-r1
```

## 管理指令

```bash
docker compose ps                                            # 看狀態
docker compose logs -f gemma4                                # 看單一日誌
docker compose restart gemma4                                # 重啟單一
docker compose down                                          # 停全部
watch -n 2 'free -h && echo && docker stats --no-stream'     # 監控
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null  # 釋放 page cache
```

## 記憶體預算（實測）

| Port | gpu-memory-utilization | 實測佔用 |
|------|------------------------|----------|
| 8001 Gemma 4 31B | 0.40 | ~50GB（含 mmap'd checkpoint）|
| 8002 Qwen3.6 27B | 0.30 | ~30GB |
| 8003 Nemotron 12B-VL | 0.15 | ~20GB |
| **合計** | **0.85** | **~106GB** |
| 系統 + swap 緩衝 | | ~15GB + 9GB swap |

**重要觀察**：UMA 下 vLLM 看到全部 128GB 為 "GPU memory"。每個 process 設 `gpu-memory-utilization=X` 是 X × 128GB 的池，但 OS 也會 mmap 整個 checkpoint 進 buff/cache（30GB+ for 大模型），需在啟動每個服務前 `drop_caches`。

## 疑難排解

| 症狀 | 解法 |
|------|------|
| `ValueError: No available memory for the cache blocks` | 提高 `--gpu-memory-utilization`（gemma4 建議 0.40 起跳）|
| 啟動到一半 container exit | 通常是 KV cache 不夠；提高 utilization 或縮 `--max-model-len` |
| 啟動慢（>10 min） | flashinfer 在做 fp4_gemm autotuning，第一次正常；之後會快取 |
| Qwen3.6 回應只有 reasoning 沒 content | thinking 模式還沒結束；`max_tokens` 加到 800+ |
| 系統卡頓 / 嚴重 swap | 用 `drop_caches` 釋放 page cache，或停一個服務 |
| `nvidia` runtime not found | `sudo apt install nvidia-container-toolkit && sudo systemctl restart docker` |

## 架構圖

```
                ┌─── localhost:8001 (Gemma 4 31B NVFP4) ──┐
                │                                          │
   Multi-Agent ─┼─── localhost:8002 (Qwen3.6 27B NVFP4) ──┤── 共用 128GB UMA
   Orchestrator │                                          │   (GB10 Blackwell)
                └─── localhost:8003 (Nemotron 12B-VL FP8) ┘
```

## 檔案

- `docker-compose.yml` — 3 服務定義（+ deepseek-r1 為 optional profile）
- `test_all.sh` — 健康檢查 + chat 完成測試
- `nano_v3_reasoning_parser.py` — Nemotron 30B-A3B 自訂 parser（目前未用，留作備援）
- `.env.example` — `HF_HOME` 等環境變數範本
