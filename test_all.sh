#!/usr/bin/env bash
# 煙霧測試：驗證 3 個 vLLM 服務都活著並能回應
set -u

echo "=== 1) /v1/models 健康檢查 ==="
for entry in "gemma4:8001" "qwen36:8002" "nemotron:8003"; do
  name="${entry%:*}"; port="${entry#*:}"
  printf "[%-9s @ %s] " "$name" "$port"
  resp=$(curl -s --max-time 5 "http://localhost:$port/v1/models" 2>&1)
  echo "$resp" | jq -e '.data[0].id' >/dev/null 2>&1 \
    && echo "OK -> $(echo "$resp" | jq -r '.data[0].id')" \
    || echo "FAIL"
done

echo ""
echo "=== 2) 簡單 chat 完成測試 ==="
for entry in "gemma4:8001" "qwen36:8002" "nemotron:8003"; do
  name="${entry%:*}"; port="${entry#*:}"
  echo "--- $name @ $port ---"
  # qwen36 是 reasoning 模型，需要更多 token 才能看到最終答案
  if [ "$name" = "qwen36" ]; then
    max_tok=800
  else
    max_tok=120
  fi
  resp=$(curl -s --max-time 90 "http://localhost:$port/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$name\",
      \"messages\": [{\"role\":\"user\",\"content\":\"用一句中文介紹你自己\"}],
      \"max_tokens\": $max_tok,
      \"temperature\": 0.7
    }")
  content=$(echo "$resp" | jq -r '.choices[0].message.content // empty')
  if [ -z "$content" ] || [ "$content" = "null" ]; then
    # 可能在 reasoning 欄位
    echo "[reasoning] $(echo "$resp" | jq -r '.choices[0].message.reasoning // .error // "no response"' | head -c 200)..."
  else
    echo "$content"
  fi
  echo ""
done

echo "=== 3) 記憶體狀態 ==="
free -h
echo ""
echo "=== 4) Container 狀態 ==="
docker compose -f /home/joshhu/workspace/meetaclawtaipei/docker-compose.yml ps
