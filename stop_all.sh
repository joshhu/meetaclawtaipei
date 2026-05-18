#!/usr/bin/env bash
# 停止全部服務（vLLM container + TTS + 主 server）
set -uo pipefail
cd "$(dirname "$0")"

log()  { echo -e "\033[1;36m[$(date +%H:%M:%S)]\033[0m $*"; }
ok()   { echo -e "\033[1;32m✅\033[0m $*"; }

log "停主 server (port 5050)"
PIDS=$(ps -ef | grep "uvicorn main:app" | grep -v grep | awk '{print $2}')
if [ -n "$PIDS" ]; then
  echo $PIDS | xargs sudo -n kill -9 2>/dev/null || echo $PIDS | xargs kill -9 2>/dev/null
  ok "main server stopped"
fi

log "停 TTS server (port 5051)"
PIDS=$(ps -ef | grep "uvicorn tts_server" | grep -v grep | awk '{print $2}')
if [ -n "$PIDS" ]; then
  echo $PIDS | xargs sudo -n kill -9 2>/dev/null || echo $PIDS | xargs kill -9 2>/dev/null
  ok "TTS server stopped"
fi

log "停 3 個 vLLM container（不刪 image）"
docker compose stop gemma4 qwen36 nemotron 2>&1 | grep -v "^$"
ok "vLLM containers stopped"

log "最終狀態"
docker compose ps
ss -tlnp 2>/dev/null | grep -E ":(5050|5051)" || ok "port 5050/5051 已釋放"
