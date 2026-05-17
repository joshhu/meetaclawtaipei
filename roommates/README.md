# 室友三人組 (Roommates)

三個 LLM 在虛擬客廳裡自動聊天的 web demo，跑在 DGX Spark 上。

## 角色

| 角色 | LLM (port) | 個性 |
|------|-----------|------|
| 🤔 **Alex** | qwen36 (8002) | 哲學研究生、愛分析、講話有深度 |
| 😊 **Bella** | gemma4 (8001) | 外向社交咖、熱情主動 |
| 🍳 **Carl** | nemotron (8003) | 理工男、講話簡短直接 |

三人「同時思考」，**最快出口的先說**。前一輪贏家本輪不能再說（避免霸場）。

## 啟動

```bash
# 1. 確認 3 個 vLLM 服務都活著
cd /home/joshhu/workspace/meetaclawtaipei
docker compose ps   # 應該看到 gemma4 / qwen36 / nemotron 都 Up

# 2. 跑這個 demo
cd roommates
uv sync        # 首次：建 venv 裝套件
./run.sh       # 預設 port 5050
```

打開 `http://<dgx-spark-ip>:5050` 看三人自動聊天。

## 觀察重點

- **Carl 應該常贏**：nemotron 模型最小（12B），race 中通常最快
- **Alex 講話最有深度**：qwen 是 thinking 模型
- **Bella 帶話題**：gemma 通用對話強
- **chat log 在右側**：累積完整對話，可看出脈絡

## 設定

修改 `main.py` 中的 `AGENTS` list 可調整角色設定、模型 endpoint。

修改 `COMMON_RULES` 可調整對話風格（句長、語言、是否列點等）。

## API 端點

- `/` — 前端 (index.html)
- `/health` — JSON 健康檢查
- `/ws` — WebSocket：訂閱即收到 race 結果（type=hello/thinking/say/error）

## 疑難排解

| 症狀 | 解法 |
|------|------|
| `curl localhost:8001/v1/models` 沒回應 | `cd .. && docker compose up -d` 先啟動 vLLM |
| Alex 的話都是 "（沒話說）" | Qwen3.6 thinking 把 token 用光，調 `max_tokens` 加大 |
| 對話卡住不動 | F12 看 console / WebSocket 是否斷線 |
| 3 個模型回應變慢 | vLLM 同時 inference 競爭 KV cache，是預期 |
| Port 5050 被佔 | `PORT=5051 ./run.sh` |
