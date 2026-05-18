#!/usr/bin/env bash
# 啟動 v4 全套：TTS server + 主 web server
set -euo pipefail
cd "$(dirname "$0")"

# 1. 確認 3 個 vLLM 服務都在
for p in 8001 8002 8003; do
  if ! curl -s --max-time 2 "http://localhost:$p/v1/models" >/dev/null; then
    echo "❌ vLLM port $p 沒回應，先 docker compose up -d"
    exit 1
  fi
done
echo "✅ 3 個 vLLM 都在"

# 2. 啟動 TTS server (port 5051)
if ! curl -s --max-time 2 http://localhost:5051/health 2>/dev/null | grep -q '"ok":true'; then
  echo "🔊 啟動 TTS server (背景)..."
  nohup uv run uvicorn tts_server:app --host 0.0.0.0 --port 5051 --log-level info > /tmp/tts.log 2>&1 &
  disown
  echo "等 TTS 模型載入..."
  START=$(date +%s)
  while ! curl -s --max-time 2 http://localhost:5051/health 2>/dev/null | grep -q '"ok":true'; do
    if [ $(($(date +%s)-START)) -gt 120 ]; then
      echo "❌ TTS server 啟動 timeout"
      tail /tmp/tts.log
      exit 1
    fi
    sleep 5
  done
  echo "✅ TTS server ready after $(($(date +%s)-START))s"
else
  echo "✅ TTS server 已在跑"
fi

# 3. 啟動主 web server (port 5050)
PORT="${PORT:-5050}"
echo "🚀 主 server http://0.0.0.0:$PORT"
exec uv run uvicorn main:app --host 0.0.0.0 --port "$PORT" --log-level info
