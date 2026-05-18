# AI Summit 三巨頭舞台 (v4 — TTS voice clone)

三個 LLM 分別扮演 Trump / Obama / Jensen Huang，在虛擬演講舞台上對談任意主題。

**v4 新增：講話時用 Qwen3-TTS 1.7B 模型 voice clone 本人聲音同步發聲**。

## 功能總覽

| 元件 | 角色 | port |
|---|---|---|
| **vLLM × 3** | gemma4/qwen36/nemotron LLM | 8001/8002/8003 |
| **TTS server** | Qwen3-TTS-12Hz-1.7B-Base voice clone | 5051 |
| **主 web** | FastAPI + WebSocket UI | 5050 |

## 角色映射

| 講者 | LLM | 能 search? | 聲音來源（voice clone）|
|---|---|---|---|
| 🇺🇸 **Trump** | qwen36 | ✅ | `voices/trump.mp3`（與習近平國宴致詞）|
| 🇺🇸 **Obama** | gemma4 | ❌（評論） | `voices/obama.mp3`（黑人四分衛訪談）|
| 🧥 **Jensen** | nemotron | ✅ | `voices/huang.mp3`（「電子化為 token」演講）|

## 啟動

```bash
# 1. 確認 3 個 vLLM 服務在跑
cd /home/joshhu/workspace/meetaclawtaipei
docker compose ps

# 2. 一鍵跑 demo（同時啟動 TTS server + 主 server）
cd roommates
uv sync                # 首次：建 venv 裝套件 (fastapi、openai、ddgs、qwen-tts、torch...)
./run_all.sh           # 預設主 server port 5050、TTS 5051
```

首次啟動：
- TTS 模型載入 ~35s
- 3 位 voice prompt 預先 build ~2s
- 之後每次 TTS 生成 3-10s

打開 `http://<dgx-spark-ip>:5050`，輸入主題、按開始討論，三人會輪流發言並出聲。

## v4 架構

```
LLM 產生文字 (5-15s)
    │
    ├─→ "say" event → 前端 bubble 打字機
    │
    └─→ TTS server /tts {speaker, text}
            │
            └─→ "audio" event (base64 WAV) → 前端 Audio.play()

主迴圈 await TTS task 完成才下一輪，避免跨輪音檔衝突
```

## 自訂自己的聲音

放 mp3 進 `voices/` 並提供文字稿：

```bash
# 1. 把音檔放進去（10-60 秒清楚單人語音、英文較準）
cp my_voice.mp3 voices/myname.mp3

# 2. 轉成 24kHz mono WAV
ffmpeg -i voices/myname.mp3 -ar 24000 -ac 1 voices/myname.wav

# 3. 用 whisper 自動轉錄（或手動寫）
uvx --from openai-whisper whisper voices/myname.wav \
  --model base --language en --device cpu \
  --output_format txt --output_dir voices/

# 4. 改 tts_server.py 的 SPEAKERS dict 加入新人
# 5. 改 main.py 的 AGENTS 加入新角色
# 6. 重啟 TTS server (要重 load model ~35s)
```

## 疑難排解

| 症狀 | 解法 |
|---|---|
| 沒聲音 | 瀏覽器第一次需點頁面解鎖 autoplay；F12 看 Console 有無音檔 base64 |
| 音檔超短（< 1 秒）| 模型 early stop；通常是文字含「...」省略號（v4 已自動替換為「，」）|
| 音檔超長（>15 秒）| max_new_tokens 估計過高；可在 tts_server.py 改 estimate_tokens |
| TTS 慢 | 首次推理較慢、之後 3-10s；無 flash-attn 速度有限 |
| 跨輪音檔重疊 | v4 已 await TTS task 完成才下一輪；如果還發生看 main_v4.log |
| Trump 從不搜尋 | qwen 較克制；改 prompt 強化、或他在順序中靠後就會看 Jensen 結果接話 |
| 記憶體吃緊 | TTS 5GB + vLLM 100GB 緊張；停一個 vLLM 或縮 max-model-len |

## 圖片來源 (CC / PD)

- `trump.jpg` — 白宮 2017 官方總統肖像 (Public Domain US Gov)
- `obama.jpg` — 白宮 2012 官方總統肖像 (Public Domain US Gov)
- `jensen.jpg` — 2023 黃仁勳與印度總理 Modi 會晤 (Government of India CC)

## 音檔來源

- 真人公開影像中的單人語音節錄（個人 demo 用途）
- 透過 `voices/*.txt` 提供逐字稿（whisper-base 自動轉錄）

## v1 → v2 → v3 → v4 演進

| 項目 | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| 主題 | 客廳閒聊 | AI Summit | 用戶輸入 | 用戶輸入 |
| 角色 | Alex/Bella/Carl | Trump/Obama/Jensen | 同 v2 | 同 v2 |
| 頭像 | emoji | 真人照片 | 同 v2 | 同 v2 |
| 知識範圍 | 訓練 cutoff | 同 | **+ web search** | 同 v3 |
| 發言節奏 | Race | Race+fairness | Round-robin | RR + 隨機起始 |
| **聲音** | 無 | 無 | 無 | **✨ TTS voice clone** |
