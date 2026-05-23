# AI Summit 三巨頭舞台

三個 LLM 分別扮演 **Trump / Obama / Jensen Huang**，在虛擬「AI Summit 台北」舞台上對談任意主題。三人會輪流發言、即時 web search 查資料，並用 **Qwen3-TTS voice clone 本人聲音**同步發聲。

這是 [meetaclawtaipei](../README.md) 的 demo 應用層，跑在 3 個 vLLM 後端服務之上。

## 元件

| 元件 | 角色 | port | 啟動方式 |
|------|------|------|----------|
| vLLM × 3 | gemma4 / qwen36 / nemotron LLM 後端 | 8001 / 8002 / 8003 | 上層 `docker compose`（見專案根 README）|
| **TTS server** | Qwen3-TTS-12Hz-1.7B-Base voice clone | 5051 | `tts_server.py` |
| **主 web server** | FastAPI + WebSocket UI | 5050 | `main.py` |

## 角色映射

| 講者 | LLM | endpoint | 能 search? | 聲音來源（voice clone）|
|------|-----|----------|-----------|------------------------|
| 🇺🇸 **Trump** | qwen36 | :8002 | ✅ | `voices/trump.wav` |
| 🇺🇸 **Obama** | gemma4 | :8001 | ✅ | `voices/obama.wav` |
| 🧥 **Jensen** | nemotron | :8003 | ✅ | `voices/huang.wav` |

角色設定都在 `main.py` 的 `AGENTS` list：persona prompt 有中英雙版、各自對應的 vLLM endpoint、是否帶 `web_search` tool、回應長度上限等。

## 啟動

```bash
# 1. 確認 3 個 vLLM 後端在跑
cd /home/joshhu/workspace/meetaclawtaipei
docker compose ps

# 2. 一鍵跑 demo（自動起 TTS server + 主 server）
cd roommates
uv sync          # 首次：建 venv 裝套件（fastapi、openai、ddgs、qwen-tts、torch…）
./run_all.sh     # 主 server :5050、TTS :5051
```

> 也可從專案根用 `./start_all.sh` 一鍵冷啟動整個 stack（3 vLLM → TTS → web）。

首次啟動：TTS 模型載入 ~35s、3 位 voice prompt 預讀 ~2s，之後每次 TTS 生成 3-10s。

打開 `http://<dgx-spark-ip>:5050`，輸入主題、選語言（中／英）、按開始討論。

## 運作流程

```
主迴圈（main.py /ws）
  └─ 隨機抽一位（排除上一位，避免連兩次）
       └─ Agent.speak()  ── tool loop ──┐
            可呼叫 web_search 查資料      │  (5-15s)
            └─→ "say" event → 前端 bubble 打字機
            └─→ POST tts_server /tts {speaker, text}
                   └─→ "audio" event (base64 WAV) → 前端 Audio.play()
       主迴圈 await 打字機 + TTS + 音檔播完，才進下一輪（避免跨輪音檔重疊）
```

WebSocket 事件型別：`hello` / `thinking` / `searching` / `search_done` / `say` / `audio` / `topic_set` / `stopped` / `error`。前端送 `set_topic`（含 topic 與 language）與 `stop`。

### web search（Brave + DDG fallback）

`web_search()`：若專案根 `.env` 設了 `BRAVE_API_KEY` 就走 Brave Search API，失敗或無 key 則 fallback 回 DuckDuckGo（`ddgs`）。第一輪對可 search 的 agent 強制查一次（`tool_choice="required"`），之後交給模型決定，最多 3 輪。

### tool call fallback parser（main.py 核心複雜點）

各模型的 vLLM tool parser 行為不一致，content 常洩漏未解析的 tool call 標記。`Agent.speak()` 因此有一連串 **fallback 正則**，手動從 `msg.content` 抽 tool call，涵蓋：

- `<TOOLCALL>[{...}]</TOOLCALL>`（nemotron 舊格式）
- `<tool_call>{json}</tool_call>`（qwen3）
- `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`（hermes-XML）
- `web_search("query")`（pythonic 裸 call）
- `call:web_search{query:...}`（gemma4）

**換或加模型時，多半要在這裡補對應格式的 parser，並在 `_clean()` 補清理規則**（清掉洩漏標記、reasoning 前綴、超長截斷到句末標點）。英文模式下若模型回了中文，會自動 retry 要求翻成英文。

## 自訂自己的聲音

```bash
# 1. 放音檔（10-60 秒清楚單人語音，英文較準）
cp my_voice.mp3 voices/myname.mp3

# 2. 轉 24kHz mono WAV
ffmpeg -i voices/myname.mp3 -ar 24000 -ac 1 voices/myname.wav

# 3. 用 whisper 自動轉錄逐字稿（或手寫）
uvx --from openai-whisper whisper voices/myname.wav \
  --model base --language en --device cpu \
  --output_format txt --output_dir voices/

# 4. 改 tts_server.py 的 SPEAKERS：加 "Myname": ("myname.wav", "myname.txt")
# 5. 改 main.py 的 AGENTS：加新角色（指定 model endpoint / persona / 聲音）
# 6. 重啟 TTS server（要重 load 模型 ~35s）
```

TTS 用 `x_vector_only_mode`，只需 reference WAV（不靠逐字稿）；`.txt` 留作紀錄。

## 疑難排解

| 症狀 | 解法 |
|------|------|
| 沒聲音 | 瀏覽器首次需點頁面解鎖 autoplay；F12 Console 看有無音檔 base64 |
| 音檔超短（< 1 秒）| 模型 early stop；通常是文字含省略號（已自動把 `…`/`...` 換成「，」）|
| 音檔超長（>15 秒）| `max_new_tokens` 估計過高；調 `tts_server.py` 的 `estimate_tokens` |
| TTS 慢 | 首次推理較慢、之後 3-10s；無 flash-attn 速度有限 |
| 跨輪音檔重疊 | 主迴圈已 await TTS + 音檔播完才下一輪；若仍發生看 `/tmp/tts.log` 與 main server 日誌 |
| 某人從不搜尋 | 該模型較克制；強化 persona prompt 的 search 指示，或它在順序中靠後就會接別人查到的結果 |
| 記憶體吃緊 | TTS ~5GB + 3 個 vLLM ~100GB 緊張；停一個 vLLM 或縮 `--max-model-len` |
| 英文模式回中文 | `speak()` 會自動 retry 翻譯；若仍漏字看主 server 日誌 |

## API（tts_server.py）

```bash
# 回傳 audio/wav
curl -X POST http://localhost:5051/tts \
  -H "Content-Type: application/json" \
  -d '{"speaker":"Trump","text":"This is tremendous, believe me."}' --output out.wav

# 回傳 JSON {audio_b64, format}
curl -X POST http://localhost:5051/tts_base64 -H "Content-Type: application/json" \
  -d '{"speaker":"Jensen","text":"The more you buy, the more you save."}'

curl http://localhost:5051/health   # {"ok":true, "speakers":[...]}
```

`language` 不傳會自動偵測（含 CJK → Chinese，否則 English）。`max_new_tokens` 不傳則依字長動態估。

## 素材來源

- **圖片**（`static/*.jpg`，CC / Public Domain）：`trump.jpg` 白宮 2017 官方肖像、`obama.jpg` 白宮 2012 官方肖像、`jensen.jpg` 2023 黃仁勳與 Modi 會晤（Government of India CC）。
- **音檔**（`voices/`）：真人公開影像中的單人語音節錄（個人 demo 用途），逐字稿由 whisper-base 自動轉錄存於 `voices/*.txt`。
