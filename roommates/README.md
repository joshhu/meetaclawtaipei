# AI Summit 三巨頭舞台 (v3 — with web search)

三個 LLM 分別扮演 Trump / Obama / Jensen Huang，在虛擬演講舞台上對談任意主題。**Trump 與 Jensen 能即時上網搜尋（DuckDuckGo）**取得最新資訊；Obama 不搜，根據兩人查到的資料做平衡評論。

跑在 DGX Spark 上。

## 角色

| 講者 | LLM (port) | 個性 → 強項 | 能 search？ |
|---|---|---|---|
| 🇺🇸 **Trump** | qwen36 (8002) | 簡短自信、超 catchphrases | ✅ |
| 🇺🇸 **Obama** | gemma4 (8001) | 沉穩平衡、用「我們」拉近 | ❌（只評論） |
| 🧥 **Jensen Huang** | nemotron (8003) | NVIDIA 自家模型扮自家 CEO、技術詞滿載 | ✅ |

發言順序為 **round-robin**（Obama → Jensen → Trump 輪替），每人輪到時可以呼叫 `web_search` 工具拿即時資料。

## 啟動

```bash
# 1. 確認 3 個 vLLM 服務在跑（要含 --enable-auto-tool-choice 標）
cd /home/joshhu/workspace/meetaclawtaipei
docker compose ps

# 2. 跑 demo
cd roommates
uv sync          # 首次：建 venv 裝套件 (fastapi、uvicorn、openai、ddgs)
./run.sh         # 預設 port 5050
```

打開 `http://<dgx-spark-ip>:5050`：

1. 上方「主題列」輸入今日論壇主題（例：`NVIDIA Rubin 規格`、`DeepSeek V4 速度`、`Llama 5 何時出？`）
2. 按「開始討論」
3. 三人開始輪流發言，能搜尋的（Trump + Jensen）會顯示 🔍 indicator

## UI 介面

- **主題輸入列**：金色按鈕 + 提示文字
- **舞台**：紅幕、金色 backdrop banner、燈光
- **3 個圓形大頭像**：講話時放大發光
- **Speech bubble**：頭頂浮現、打字機效果
- **🔍 搜尋 indicator**：搜尋中的講者頭頂出現綠色脈動標
- **右側對話 log**：累積完整紀錄，含搜尋查詢

## vLLM tool calling 設定

`docker-compose.yml` 中的 tool 參數：
- **qwen36**：`--enable-auto-tool-choice --tool-call-parser=hermes`
- **nemotron**：`--enable-auto-tool-choice --tool-call-parser=pythonic`
  - 加 fallback：手動 parse `<TOOLCALL>[...]</TOOLCALL>` 格式
- **gemma4**：無 tool calling（vLLM 在 4.x 對 Gemma 4 的 parser 支援待加）

## API 端點

- `/` — 前端
- `/static/*` — 圖檔
- `/health` — JSON 健康檢查（含 can_search）
- `/ws` — WebSocket
  - Server → Client: `hello / topic_set / thinking / searching / search_done / say / error`
  - Client → Server: `{type:"set_topic", topic:"..."}`

## 疑難排解

| 症狀 | 解法 |
|---|---|
| 三人不講話 | 確認你有輸入主題並按「開始討論」（v3 不再自動跑）|
| 🔍 沒出現 | LLM 判斷不需要搜；可改主題加「最新」「規格」「2026」等關鍵字鼓勵搜尋 |
| Jensen bubble 顯示 `<TOOLCALL>` | 表示 fallback parser 失敗，看 server log 有無 `Manual TOOLCALL parse failed` |
| 搜尋慢 | DDG 第一次連線較慢；每次 search ~3-8s，總一輪 ~10-20s |
| 對話沒對到主題 | system prompt 強化、或刪除舊 history 重設 |
| Port 5050 被佔 | `PORT=5051 ./run.sh` |

## 圖片來源 (CC / Public Domain)

- `trump.jpg` — 白宮 2017 官方總統肖像 (US Gov, Public Domain)
- `obama.jpg` — 白宮 2012 官方總統肖像 (US Gov, Public Domain)
- `jensen.jpg` — 2023 黃仁勳與 Modi 會晤 (Government of India CC)

## v1 → v2 → v3 演進

| 項目 | v1 (室友) | v2 (舞台) | v3 (search) |
|---|---|---|---|
| 主題 | 客廳閒聊 | AI Summit 對談 | **用戶自訂主題** |
| 角色 | Alex/Bella/Carl | Trump/Obama/Jensen | 同 v2 |
| 頭像 | emoji | 真人照片 | 同 v2 |
| 發言順序 | Race | Race + fairness | **Round-robin** |
| 知識範圍 | 訓練 cutoff | 同 | **即時 web search** |
| 新功能 | — | 場景 + 真人 | tool use, 主題輸入框, 🔍 indicator |
