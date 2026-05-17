# AI Summit 三巨頭舞台 (v2)

三個 LLM 分別扮演 Trump / Obama / Jensen Huang，在虛擬演講舞台上對談「AI 與人類的下一個十年」。

跑在 DGX Spark 上，全自動、不需用戶輸入。

## 角色

| 講者 | LLM (port) | 模型強項 → 對應人物 |
|------|-----------|-----|
| 🇺🇸 **Trump** | qwen36 (8002) | thinking 收斂後的短句金句、自信誇張 |
| 🇺🇸 **Obama** | gemma4 (8001) | 通用對話、沉穩平衡、用「我們」拉近 |
| 🧥 **Jensen Huang** | nemotron (8003) | NVIDIA 自家模型扮自家 CEO，AI/Blackwell/GPU 詞庫超對 |

三人「同時思考、最快出口的先說」(race 模式)，加上**公平規則**：
- 上一輪贏家本輪不參賽
- 任何人比最多者少講 2 輪以上 → 強制 solo

## 啟動

```bash
# 1. 確認 3 個 vLLM 服務在跑
cd /home/joshhu/workspace/meetaclawtaipei
docker compose ps   # 看到 gemma4 / qwen36 / nemotron 都 Up

# 2. 跑 demo
cd roommates
uv sync            # 首次：建 venv 裝套件
./run.sh           # 預設 port 5050
```

打開 `http://<dgx-spark-ip>:5050` 看三人自動對談。

## UI 介面

- 紅色舞台帷幕 + 金色 backdrop banner「AI & THE NEXT DECADE」
- 三人圓形大頭像（金色外框，講話時變紅光放大）
- 頭頂浮現 speech bubble，打字機效果
- 右側完整對談紀錄 (chat log)

## 觀察重點

- **Trump** 講「最棒的」「相信我」「tremendous」「huge」「many people are saying」
- **Obama** 開頭常用「讓我這樣說」、講「我們」與希望
- **Jensen** 大量提 Blackwell、AI 工廠、加速運算、CUDA

## 換角色 / 模型

修改 `main.py` 中的 `AGENTS` list：
- `name`：講者顯示名
- `avatar`：static/ 下的圖檔名
- `role`：人物個性 system prompt
- `base_url` + `model`：對應的 vLLM endpoint

## 圖片來源 (CC / Public Domain)

- `trump.jpg` — 白宮 2017 官方總統肖像，Public Domain (US Gov work)
- `obama.jpg` — 白宮 2012 官方總統肖像，Public Domain (US Gov work)
- `jensen.jpg` — 2023 黃仁勳會晤印度總理 Modi，CC (Government of India Open Data License)

## API 端點

- `/` — 前端 (index.html)
- `/static/*` — 圖檔 + JS
- `/health` — JSON 健康檢查
- `/ws` — WebSocket：訂閱即收到 race 結果 (`type=hello/thinking/say/error`)

## 疑難排解

| 症狀 | 解法 |
|------|------|
| `curl localhost:8001/v1/models` 沒回應 | `cd .. && docker compose up -d` 先啟動 vLLM |
| 某人從不發言 | nemotron 不愛扮政治人物 → 已映射為 Jensen；公平規則會強制輪到 |
| Trump 講話只剩 reasoning 沒 content | Qwen3.6 thinking 模式問題，已用 `enable_thinking: False` 關掉 |
| 對話卡住不動 | F12 看 console / WebSocket 是否斷線 |
| Port 5050 被佔 | `PORT=5051 ./run.sh` |

## 從 v1 升級點

| 項目 | v1 (室友) | v2 (舞台) |
|------|----------|----------|
| 主題 | 公寓客廳室友 | AI Summit 演講舞台 |
| 角色 | Alex/Bella/Carl (虛構) | Trump/Obama/Jensen (真實名人) |
| 頭像 | emoji 🤔😊🍳 | 真人照片（圓形外框）|
| 場景 CSS | 米色客廳 + emoji 家具 | 紅色舞台 + 金色 backdrop + 燈光 |
| 公平規則 | 只有「上輪贏家不參賽」 | 加「最少講者強制 solo」|
