# 舞台 demo — AI Summit 三巨頭

三個 LLM 各扮演 **Trump、Obama、黃仁勳**，在「AI Summit 台北」舞台輪流對談任意主題。發言時會即時上網查資料，並用 voice clone 模仿本人聲音講話。

這是 [meetaclawtaipei](../README.md) 的網頁前端，跑在 3 個 vLLM 後端之上。

---

## 三個角色

| 講者 | 用哪個模型 | endpoint | 聲音 |
|---|---|---|---|
| Trump | qwen36 | :8002 | `voices/trump.wav` |
| Obama | gemma4 | :8001 | `voices/obama.wav` |
| Jensen | nemotron | :8003 | `voices/huang.wav` |

每個角色的人設（中英雙語的 persona prompt、對應的模型、發言長度上限）都定義在 `main.py` 的 `AGENTS`。三人都能呼叫 `web_search`。

---

## 三個程式

| 程式 | 做什麼 | port |
|---|---|---|
| `main.py` | 主網頁，WebSocket 跑對談主迴圈 | 5050 |
| `tts_server.py` | Qwen3-TTS voice clone，把文字變成本人聲音 | 5051 |
| `static/index.html` | 前端 UI（打字機泡泡 + 播音） | — |

---

## 啟動

前提是 3 個 vLLM 後端已經在跑（`docker compose ps` 確認）。

```bash
uv sync          # 首次：建 venv、裝依賴
./run_all.sh     # 自動起 TTS server + 主網頁
```

打開 `http://<dgx-spark-ip>:5050`，輸入主題、選語言（中／英）、按開始。

> 也可以從專案根用 `./start_all.sh` 一鍵把整個 stack（含後端）帶起來。
> 首次啟動 TTS 模型載入約 35 秒，之後每句話生成約 3-10 秒。

---

## 運作方式

主迴圈每一輪：

1. 隨機抽一位講者（不會連續抽同一個）
2. 呼叫該模型發言，過程中它可能先 `web_search` 查資料
3. 文字送到前端跑打字機泡泡
4. 同一段文字送去 TTS server，回傳的音檔在前端播放
5. 等打字機跑完、音檔也播完，才換下一位（避免聲音重疊）

**web search**：設了 `BRAVE_API_KEY`（讀專案根 `.env`）就用 Brave Search，否則退回 DuckDuckGo。每位講者第一次發言會強制查一次，之後由模型自己決定要不要查。

---

## 換成自己的聲音

```bash
# 1. 放一段 10-60 秒、單人清楚的語音（英文較準）
cp my_voice.mp3 voices/myname.mp3

# 2. 轉成 24kHz 單聲道 WAV
ffmpeg -i voices/myname.mp3 -ar 24000 -ac 1 voices/myname.wav

# 3. 在 tts_server.py 的 SPEAKERS 加一行
#    "Myname": ("myname.wav", "myname.txt")

# 4. 在 main.py 的 AGENTS 加一個新角色

# 5. 重啟 TTS server（要重新載入模型）
```

voice clone 只需要 WAV，不靠逐字稿。

---

## 直接打 TTS API

```bash
# 回傳 WAV
curl -X POST http://localhost:5051/tts \
  -H "Content-Type: application/json" \
  -d '{"speaker":"Trump","text":"This is tremendous, believe me."}' --output out.wav

# 健康檢查
curl http://localhost:5051/health
```

語言不指定會自動偵測（有中文字就當中文，否則英文）。

---

## 疑難排解

| 症狀 | 解法 |
|---|---|
| 沒聲音 | 瀏覽器首次需點一下頁面解鎖 autoplay；F12 看 Console 有沒有收到音檔 |
| 聲音太短（< 1 秒）| 文字裡的省略號會讓模型提早停（程式已自動把 `…` 換成逗號）|
| 聲音重疊 | 主迴圈已等音檔播完才換人；若仍發生看 `/tmp/tts.log` |
| 某人都不查資料 | 該模型較保守，強化它 persona 裡的查詢指示，或讓它排在別人後面接話 |
| 記憶體吃緊 | TTS 約 5GB + 3 個 vLLM 約 100GB，停一個 vLLM 或縮 `--max-model-len` |

---

## 一個維護重點

不同模型回報 tool call 的格式不一致，有些 vLLM 沒幫忙解析、直接把標記混在文字裡。`main.py` 的 `Agent.speak()` 裡有一串 fallback 正則專門把這些 tool call 撈出來（涵蓋 `<tool_call>`、hermes XML、`web_search("...")` 等寫法）。**之後換或加模型時，多半要在這裡補上對應格式的解析，並在 `_clean()` 補清理規則。**

---

## 素材來源

- **照片**（`static/*.jpg`）：白宮官方肖像（Trump 2017、Obama 2012，公有領域）、黃仁勳 2023 與 Modi 會晤（Government of India, CC）。
- **聲音**（`voices/`）：公開影像中的單人語音節錄，僅供 demo；逐字稿用 whisper 自動轉錄。
