#!/bin/bash
# Test the Copilot proxy end-to-end
set -e

PORT=${PROXY_PORT:-4000}
BASE="http://localhost:$PORT"
PASS=0
FAIL=0

red()   { echo -e "\033[31m$1\033[0m"; }
green() { echo -e "\033[32m$1\033[0m"; }

# Check proxy is running
echo "=== Health check ==="
if curl -sf "$BASE/health" > /dev/null 2>&1; then
    green "PASS: Proxy is running on port $PORT"
    PASS=$((PASS+1))
else
    red "FAIL: Proxy not running on port $PORT. Start it with: ./start.sh"
    exit 1
fi

# Test non-streaming request (Anthropic Messages format)
echo ""
echo "=== Non-streaming request (Anthropic Messages API) ==="
RESP=$(curl -sf -H "Content-Type: application/json" \
    -H "x-api-key: dummy" \
    -H "anthropic-version: 2023-06-01" \
    -d '{"model":"claude-opus-4-6","messages":[{"role":"user","content":"Reply with exactly: PROXY_OK"}],"max_tokens":20}' \
    "$BASE/v1/messages" 2>&1)

if echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['type']=='message'; assert d['role']=='assistant'; assert len(d['content'])>0" 2>/dev/null; then
    TEXT=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['content'][0]['text'])")
    green "PASS: Got response: $TEXT"
    PASS=$((PASS+1))
else
    red "FAIL: Unexpected response: $RESP"
    FAIL=$((FAIL+1))
fi

# Test streaming request
echo ""
echo "=== Streaming request ==="
STREAM=$(curl -sf -N -H "Content-Type: application/json" \
    -H "x-api-key: dummy" \
    -H "anthropic-version: 2023-06-01" \
    -d '{"model":"claude-opus-4-6","messages":[{"role":"user","content":"Say hi"}],"max_tokens":10,"stream":true}' \
    "$BASE/v1/messages" 2>&1)

HAS_START=$(echo "$STREAM" | grep -c "message_start" || true)
HAS_STOP=$(echo "$STREAM" | grep -c "message_stop" || true)
HAS_DELTA=$(echo "$STREAM" | grep -c "content_block_delta" || true)

if [ "$HAS_START" -gt 0 ] && [ "$HAS_STOP" -gt 0 ] && [ "$HAS_DELTA" -gt 0 ]; then
    green "PASS: Streaming works (start/delta/stop events present)"
    PASS=$((PASS+1))
else
    red "FAIL: Streaming incomplete (start=$HAS_START, delta=$HAS_DELTA, stop=$HAS_STOP)"
    FAIL=$((FAIL+1))
fi

# Test model name mapping
echo ""
echo "=== Model name mapping (claude-sonnet-4-6 → claude-sonnet-4.6) ==="
RESP2=$(curl -sf -H "Content-Type: application/json" \
    -H "x-api-key: dummy" \
    -H "anthropic-version: 2023-06-01" \
    -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"Say ok"}],"max_tokens":10}' \
    "$BASE/v1/messages" 2>&1)

if echo "$RESP2" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['type']=='message'" 2>/dev/null; then
    green "PASS: claude-sonnet-4-6 model works"
    PASS=$((PASS+1))
else
    red "FAIL: claude-sonnet-4-6 failed: $RESP2"
    FAIL=$((FAIL+1))
fi

echo ""
echo "=============================="
if [ "$FAIL" -eq 0 ]; then
    green "All $PASS tests passed!"
else
    red "$FAIL tests failed, $PASS passed"
    exit 1
fi
