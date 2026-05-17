#!/usr/bin/env bash
# 啟動 roommates web demo
set -euo pipefail
cd "$(dirname "$0")"

# 確認 3 個 vLLM 服務都活著
for p in 8001 8002 8003; do
  if ! curl -s --max-time 2 "http://localhost:$p/v1/models" > /dev/null; then
    echo "❌ vLLM 服務 port $p 沒回應，請先 docker compose up -d 啟動"
    exit 1
  fi
done
echo "✅ 3 個 vLLM 服務都在"

# 啟動 web server
PORT="${PORT:-5050}"
echo "🚀 http://0.0.0.0:$PORT"
exec uv run uvicorn main:app --host 0.0.0.0 --port "$PORT" --log-level info
