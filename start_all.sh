#!/usr/bin/env bash
# 完整啟動腳本：從關機後一鍵把整個 stack 帶起來
# - 3 個 vLLM container（gemma4 / qwen36 / nemotron）
# - TTS server (port 5051)
# - 主 web server (port 5050)
#
# 用法：./start_all.sh
# 預估時間：~15-20 分鐘（首次依序載入 3 個大模型）
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(pwd)"
ROOMS="$ROOT/roommates"

log()  { echo -e "\033[1;36m[$(date +%H:%M:%S)]\033[0m $*"; }
ok()   { echo -e "\033[1;32m✅\033[0m $*"; }
warn() { echo -e "\033[1;33m⚠️\033[0m  $*"; }
err()  { echo -e "\033[1;31m❌\033[0m $*"; }

drop_caches() {
  sync
  echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 \
    && log "Dropped page caches" \
    || warn "Couldn't drop caches (sudo password needed?)"
}

wait_for_vllm() {
  local port=$1 name=$2 timeout=${3:-720}
  log "Waiting for $name @ $port (max ${timeout}s)..."
  local start=$(date +%s)
  while true; do
    local elapsed=$(($(date +%s) - start))
    if curl -s --max-time 2 "http://localhost:$port/v1/models" 2>/dev/null \
       | grep -q '"id"'; then
      ok "$name ready (${elapsed}s)"
      return 0
    fi
    if [ $elapsed -gt $timeout ]; then
      err "$name timeout after ${timeout}s"
      docker compose logs --tail 20 "$name"
      return 1
    fi
    sleep 10
  done
}

wait_for_tts() {
  log "Waiting for TTS server @ 5051..."
  local start=$(date +%s)
  while true; do
    local elapsed=$(($(date +%s) - start))
    if curl -s --max-time 2 http://localhost:5051/health 2>/dev/null \
       | grep -q '"ok":true'; then
      ok "TTS ready (${elapsed}s)"
      return 0
    fi
    if [ $elapsed -gt 120 ]; then
      err "TTS timeout"
      tail /tmp/tts.log 2>/dev/null
      return 1
    fi
    sleep 5
  done
}

# ============ Step 1: Docker / 環境檢查 ============
log "Step 1/4: 環境檢查"
if ! systemctl is-active docker >/dev/null 2>&1; then
  err "docker daemon 沒在跑，請執行: sudo systemctl start docker"
  exit 1
fi
ok "docker daemon running"

# ============ Step 2: 啟動 3 個 vLLM container ============
log "Step 2/4: 啟動 3 個 vLLM container（依序、每個間隔 drop_caches）"

# 先檢查既有狀態
EXISTING=$(docker compose ps --services --filter "status=running" 2>/dev/null | grep -E "^(gemma4|qwen36|nemotron)$" || true)
if [ "$(echo "$EXISTING" | wc -l)" = "3" ] && [ -n "$EXISTING" ]; then
  ok "3 個 vLLM 都已在跑，略過"
else
  drop_caches
  log "→ gemma4 (Gemma 4 31B, port 8001)"
  docker compose up -d gemma4
  wait_for_vllm 8001 gemma4

  drop_caches
  log "→ qwen36 (Qwen3.6 27B, port 8002)"
  docker compose up -d qwen36
  wait_for_vllm 8002 qwen36

  drop_caches
  log "→ nemotron (Nemotron 12B-VL, port 8003)"
  docker compose up -d nemotron
  wait_for_vllm 8003 nemotron
fi

# ============ Step 3: TTS server ============
log "Step 3/4: TTS server (Qwen3-TTS-1.7B)"
cd "$ROOMS"
if curl -s --max-time 2 http://localhost:5051/health 2>/dev/null | grep -q '"ok":true'; then
  ok "TTS 已在跑"
else
  log "啟動 TTS server (背景)"
  nohup uv run uvicorn tts_server:app --host 0.0.0.0 --port 5051 --log-level info \
    > /tmp/tts.log 2>&1 &
  disown
  wait_for_tts
fi

# ============ Step 4: 主 web server ============
log "Step 4/4: 主 web server"
if ss -tlnp 2>/dev/null | grep -q ":5050"; then
  ok "主 server 已在跑"
else
  log "啟動主 server (前景，Ctrl+C 結束)"
  exec uv run uvicorn main:app --host 0.0.0.0 --port 5050 --log-level info
fi

echo ""
ok "全部就緒"
echo ""
HOST_IP=$(hostname -I | awk '{print $1}')
echo "🌐 開瀏覽器: http://${HOST_IP}:5050"
echo "🛑 停止: ./stop_all.sh"
