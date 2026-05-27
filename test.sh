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
    red "FAIL: Proxy not running on port $PORT. Start it with: ./cc-gh-proxy.py"
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

# Unit test: model-name mapping (no network, runs against the source file directly)
echo ""
echo "=== Model name mapping (unit test, --no-opus on/off) ==="
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
if python3 - "$SCRIPT_DIR/cc-gh-proxy.py" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("proxy", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

cases_default = [
    ("claude-opus-4-6",          "claude-opus-4.6"),
    ("claude-opus-4-7",          "claude-opus-4.7"),
    ("claude-opus-4-6[1m]",      "claude-opus-4.6"),
    ("claude-opus-4-6-20260312", "claude-opus-4.6"),
    ("claude-opus-4",            "claude-opus-4.6"),
    ("claude-sonnet-4-6",        "claude-sonnet-4.6"),
    ("claude-haiku-4-5",         "claude-haiku-4.5"),
]
mod._no_opus = False
for src, want in cases_default:
    got = mod.map_model_name(src)
    assert got == want, f"default: {src} -> {got}, expected {want}"

mod._no_opus = True
mod._no_opus_target = "claude-sonnet-4.6"
opus_inputs = [
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-6[1m]",
    "claude-opus-4-7-20260512",
    "claude-opus-4",
]
for src in opus_inputs:
    got = mod.map_model_name(src)
    assert got == "claude-sonnet-4.6", f"--no-opus: {src} -> {got}, expected claude-sonnet-4.6"
# Sonnet/Haiku must not be rewritten
assert mod.map_model_name("claude-sonnet-4-6") == "claude-sonnet-4.6"
assert mod.map_model_name("claude-haiku-4-5") == "claude-haiku-4.5"

# Custom target honored
mod._no_opus_target = "claude-sonnet-4.5"
assert mod.map_model_name("claude-opus-4-7") == "claude-sonnet-4.5"
print("ok")
PY
then
    green "PASS: map_model_name handles all canonicalization and --no-opus cases"
    PASS=$((PASS+1))
else
    red "FAIL: model mapping unit test failed"
    FAIL=$((FAIL+1))
fi

# Integration test: local OpenAI-compatible upstream via a tiny mock server.
# Spins up a proxy on a free port pointed at a mock /chat/completions endpoint,
# then sends an Anthropic-format request and verifies the translated round-trip.
echo ""
echo "=== Local upstream (--upstream-base-url) round-trip ==="
PROXY_PY="$SCRIPT_DIR/cc-gh-proxy.py"
MOCK_PORT=14834
PROXY_PORT_LOCAL=14835
TMPDIR_LOCAL=$(mktemp -d)
trap 'kill $MOCK_PID $PROXY_PID 2>/dev/null || true; wait $MOCK_PID $PROXY_PID 2>/dev/null || true; rm -rf "$TMPDIR_LOCAL"' EXIT

# Mock OpenAI server: returns a fixed chat-completion response
python3 - "$MOCK_PORT" >"$TMPDIR_LOCAL/mock.log" 2>&1 <<'PY' &
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        body = json.dumps({
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "MOCK_OK"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a, **k): pass
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY
MOCK_PID=$!

# Start proxy pointed at the mock. --upstream-base-url skips gh auth.
PROXY_LOG_DIR="$TMPDIR_LOCAL/logs" python3 "$PROXY_PY" \
    --port "$PROXY_PORT_LOCAL" \
    --upstream-base-url "http://127.0.0.1:$MOCK_PORT/v1" \
    --upstream-model mock-model \
    --log-level WARNING >"$TMPDIR_LOCAL/proxy.log" 2>&1 &
PROXY_PID=$!

# Wait for the proxy to come up (max ~5s)
for i in 1 2 3 4 5 6 7 8 9 10; do
    sleep 0.5
    curl -sf "http://localhost:$PROXY_PORT_LOCAL/health" >/dev/null 2>&1 && break
done

RESP_LOCAL=$(curl -sf -H "Content-Type: application/json" \
    -H "anthropic-version: 2023-06-01" \
    -d '{"model":"claude-haiku-4-5","messages":[{"role":"user","content":"ping"}],"max_tokens":10}' \
    "http://localhost:$PROXY_PORT_LOCAL/v1/messages" 2>&1)

if echo "$RESP_LOCAL" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['type']=='message'; assert d['content'][0]['text']=='MOCK_OK'; assert d['stop_reason']=='end_turn'" 2>/dev/null; then
    green "PASS: local-upstream round-trip translated and forwarded correctly"
    PASS=$((PASS+1))
else
    red "FAIL: local-upstream round-trip: $RESP_LOCAL"
    cat "$TMPDIR_LOCAL/proxy.log" | tail -10
    FAIL=$((FAIL+1))
fi

kill $MOCK_PID $PROXY_PID 2>/dev/null || true
wait $MOCK_PID $PROXY_PID 2>/dev/null || true
trap - EXIT
rm -rf "$TMPDIR_LOCAL"

# Routing test: /chat/completions and /v1/models should be registered routes.
# Without --copilot-auth they return 503 (not 404), proving the dispatcher
# matched them instead of falling through to the 404 branch.
echo ""
echo "=== /chat/completions + /v1/models routing ==="
ROUTE_PORT=14836
TMPDIR_ROUTE=$(mktemp -d)
trap 'kill $ROUTE_PID 2>/dev/null || true; wait $ROUTE_PID 2>/dev/null || true; rm -rf "$TMPDIR_ROUTE"' EXIT

PROXY_LOG_DIR="$TMPDIR_ROUTE/logs" python3 "$PROXY_PY" \
    --port "$ROUTE_PORT" \
    --upstream-base-url "http://127.0.0.1:1/v1" \
    --upstream-model mock-model \
    --log-level WARNING >"$TMPDIR_ROUTE/proxy.log" 2>&1 &
ROUTE_PID=$!

for i in 1 2 3 4 5 6 7 8 9 10; do
    sleep 0.5
    curl -sf "http://localhost:$ROUTE_PORT/health" >/dev/null 2>&1 && break
done

CODE_POST=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}' \
    "http://localhost:$ROUTE_PORT/chat/completions")
CODE_GET=$(curl -s -o /dev/null -w "%{http_code}" \
    "http://localhost:$ROUTE_PORT/v1/models")

if [ "$CODE_POST" = "503" ] && [ "$CODE_GET" = "503" ]; then
    green "PASS: /chat/completions and /v1/models routed (503 without --copilot-auth)"
    PASS=$((PASS+1))
else
    red "FAIL: routing — POST /chat/completions=$CODE_POST GET /v1/models=$CODE_GET (expected 503/503)"
    FAIL=$((FAIL+1))
fi

kill $ROUTE_PID 2>/dev/null || true
wait $ROUTE_PID 2>/dev/null || true
trap - EXIT
rm -rf "$TMPDIR_ROUTE"

echo ""
echo "=============================="
if [ "$FAIL" -eq 0 ]; then
    green "All $PASS tests passed!"
else
    red "$FAIL tests failed, $PASS passed"
    exit 1
fi
