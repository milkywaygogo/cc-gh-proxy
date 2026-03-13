#!/usr/bin/env python3
"""
cc-gh-proxy: Claude Code -> GitHub Copilot pass-through proxy.

GitHub Copilot natively supports the Anthropic Messages API at /v1/messages,
so this proxy only needs to:
  1. Swap the auth header (gh CLI OAuth token)
  2. Map model names (dashes -> dots)
  3. Strip unsupported cache_control fields
  4. Forward requests and responses as-is
"""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import os
import re
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any

COPILOT_HOST: str = "api.githubcopilot.com"
JsonDict = dict[str, Any]

logger: logging.Logger = logging.getLogger("cc-gh-proxy")

# Set in main() before server starts
_log_dir: Path = Path()
_api_key: str | None = None

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="cc-gh-proxy",
        description="Pass-through proxy: Claude Code -> GitHub Copilot (native Anthropic API)",
    )
    p.add_argument(
        "-p", "--port", type=int,
        default=int(os.environ.get("PROXY_PORT", "4000")),
        help="port to listen on (env: PROXY_PORT, default: 4000)",
    )
    p.add_argument(
        "--host",
        default=os.environ.get("PROXY_HOST", "127.0.0.1"),
        help="address to bind to (env: PROXY_HOST, default: 127.0.0.1)",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("PROXY_API_KEY"),
        help="require this key via x-api-key header (env: PROXY_API_KEY)",
    )
    p.add_argument(
        "--log-dir",
        default=os.environ.get("PROXY_LOG_DIR", str(Path(__file__).resolve().parent / "logs")),
        help="log directory (env: PROXY_LOG_DIR)",
    )
    p.add_argument(
        "--log-level",
        default=os.environ.get("PROXY_LOG_LEVEL", "INFO").upper(),
        help="log level (env: PROXY_LOG_LEVEL, default: INFO)",
    )
    return p.parse_args()


def setup_logging(log_dir: Path, level: str) -> None:
    """Configure console and file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(getattr(logging, level, logging.INFO))

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("[proxy] %(message)s"))
    logger.addHandler(console)

    fh = logging.FileHandler(log_dir / "proxy.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_jsonl(entry: JsonDict) -> None:
    """Append a JSON line to the requests log."""
    with open(_log_dir / "requests.jsonl", "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def summarize_request(body: JsonDict) -> str:
    """One-line summary of a request for the console log."""
    model: str = body.get("model", "?")
    stream: bool = body.get("stream", False)
    msgs: list[JsonDict] = body.get("messages", [])
    n_msgs: int = len(msgs)

    # Last user message preview
    last_user: str = ""
    for msg in reversed(msgs):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                last_user = content
            elif isinstance(content, list):
                texts = [b.get("text", "") for b in content if b.get("type") == "text"]
                last_user = " ".join(texts)
            break
    preview: str = last_user[:80].replace("\n", " ")
    if len(last_user) > 80:
        preview += "..."

    flag: str = " [stream]" if stream else ""
    return f"{model} ({n_msgs} msgs{flag}) \"{preview}\""


def summarize_response(
    status: int, body: JsonDict | None, stream_text: str | None
) -> str:
    """One-line summary of a response for the console log."""
    if status != 200:
        error_msg: str = ""
        if body:
            error_msg = body.get("error", {}).get("message", "")[:100]
        return f"HTTP {status}: {error_msg}"

    if body:
        # Non-streaming response
        usage: JsonDict = body.get("usage", {})
        inp: int = usage.get("input_tokens", 0)
        out: int = usage.get("output_tokens", 0)
        cached: int = usage.get("cache_read_input_tokens", 0)
        stop: str = body.get("stop_reason", "?")
        text: str = ""
        for block in body.get("content", []):
            if block.get("type") == "text":
                text = block.get("text", "")[:80].replace("\n", " ")
                break
        cache_info: str = f", cached={cached}" if cached else ""
        return f"OK in={inp} out={out}{cache_info} stop={stop} \"{text}...\""

    if stream_text is not None:
        preview: str = stream_text[:80].replace("\n", " ")
        return f"OK [streamed] \"{preview}...\""

    return f"HTTP {status}"


# ---------------------------------------------------------------------------
# Token & model helpers
# ---------------------------------------------------------------------------

def get_gh_token() -> str:
    result = subprocess.run(
        ["gh", "auth", "token"], capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.error("Failed to get gh token. Run: gh auth refresh -s copilot")
        sys.exit(1)
    return result.stdout.strip()


SSL_CTX: ssl.SSLContext = ssl.create_default_context()


class TokenManager:
    """Thread-safe gh OAuth token with auto-refresh."""

    REFRESH_INTERVAL: float = 3600  # Re-fetch every hour
    RETRY_INTERVAL: float = 30     # Retry on failure after 30s

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._token: str = get_gh_token()
        self._fetched_at: float = time.monotonic()
        logger.info("Token acquired: %s...%s", self._token[:8], self._token[-4:])

    def get_token(self) -> str:
        """Return a valid token, refreshing if stale."""
        age: float = time.monotonic() - self._fetched_at
        if age < self.REFRESH_INTERVAL:
            return self._token

        with self._lock:
            # Double-check after acquiring lock
            if time.monotonic() - self._fetched_at < self.REFRESH_INTERVAL:
                return self._token
            return self._refresh()

    def invalidate(self) -> str:
        """Force a refresh (e.g. after a 401). Returns new token."""
        with self._lock:
            return self._refresh()

    def _refresh(self) -> str:
        try:
            new_token: str = get_gh_token()
            self._token = new_token
            self._fetched_at = time.monotonic()
            logger.info("Token refreshed: %s...%s", new_token[:8], new_token[-4:])
        except SystemExit:
            logger.error("Token refresh failed, keeping old token")
        return self._token


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def map_model_name(model: str) -> str:
    """Map Anthropic model IDs to Copilot model names.

    Claude Code may send:
      claude-opus-4-6, claude-opus-4-6-20260312, claude-sonnet-4-6, etc.
    Copilot expects:
      claude-opus-4.6, claude-sonnet-4.6, claude-haiku-4.5
    """
    static_map: dict[str, str] = {
        "claude-opus-4-6": "claude-opus-4.6",
        "claude-sonnet-4-6": "claude-sonnet-4.6",
        "claude-haiku-4-5": "claude-haiku-4.5",
    }
    if model in static_map:
        return static_map[model]

    # Strip date suffixes: claude-opus-4-6-20260312 -> claude-opus-4-6
    stripped: str = re.sub(r"-\d{8}$", "", model)
    if stripped in static_map:
        return static_map[stripped]

    # Pattern: claude-{tier}-{major}-{minor} -> claude-{tier}-{major}.{minor}
    m = re.match(r"^(claude-(?:opus|sonnet|haiku)-\d+)-(\d+)$", stripped)
    if m:
        return f"{m.group(1)}.{m.group(2)}"

    # Base family: claude-opus-4 -> claude-opus-4.6 (latest known)
    family_map: dict[str, str] = {
        "claude-opus-4": "claude-opus-4.6",
        "claude-sonnet-4": "claude-sonnet-4.6",
        "claude-haiku-4": "claude-haiku-4.5",
    }
    if stripped in family_map:
        return family_map[stripped]

    logger.warning("Unknown model '%s', passing through as-is", model)
    return model


# ---------------------------------------------------------------------------
# Request rewriting
# ---------------------------------------------------------------------------

def strip_cache_control_extras(obj: Any) -> Any:
    """Remove unsupported fields from cache_control objects.

    Claude Code sends cache_control like {"type": "ephemeral", "scope": "..."}
    but Copilot only accepts {"type": "ephemeral"}.
    """
    if isinstance(obj, dict):
        result: JsonDict = {}
        for key, value in obj.items():
            if key == "cache_control" and isinstance(value, dict):
                result[key] = {"type": value["type"]} if "type" in value else value
            else:
                result[key] = strip_cache_control_extras(value)
        return result
    if isinstance(obj, list):
        return [strip_cache_control_extras(item) for item in obj]
    return obj


def rewrite_body(raw_body: bytes) -> tuple[bytes, JsonDict]:
    """Rewrite model names and strip unsupported fields.

    Returns (rewritten_body_bytes, parsed_body_dict).
    """
    body: JsonDict = json.loads(raw_body)
    modified: bool = False

    # Map model name
    original: str = body.get("model", "")
    mapped: str = map_model_name(original)
    if mapped != original:
        body["model"] = mapped
        modified = True

    # Strip unsupported cache_control fields
    cleaned: JsonDict = strip_cache_control_extras(body)
    if cleaned != body:
        body = cleaned
        modified = True

    if modified:
        return json.dumps(body).encode(), body
    return raw_body, body


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        try:
            self._handle_post()
        except (ConnectionResetError, BrokenPipeError):
            logger.debug("Client disconnected")

    def _handle_post(self) -> None:
        # Check API key if configured
        if _api_key:
            client_key: str = self.headers.get("x-api-key", "")
            if client_key != _api_key:
                self.send_error(401, "Invalid or missing API key")
                logger.warning("Rejected request: bad x-api-key")
                return

        t0: float = time.monotonic()
        content_length: int = int(self.headers.get("Content-Length", "0"))
        raw_body: bytes = self.rfile.read(content_length)

        # Rewrite model name and strip unsupported fields
        body_to_send: bytes
        parsed_body: JsonDict
        body_to_send, parsed_body = rewrite_body(raw_body)

        logger.info(">>> %s %s", self.path, summarize_request(parsed_body))

        # Build headers for upstream
        current_token: str = token_manager.get_token()
        upstream_headers: dict[str, str] = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": self.headers.get(
                "Content-Type", "application/json"
            ),
            "Content-Length": str(len(body_to_send)),
        }
        version: str | None = self.headers.get("anthropic-version")
        if version:
            upstream_headers["anthropic-version"] = version

        # Forward anthropic-beta but strip features Copilot doesn't support
        raw_beta: str | None = self.headers.get("anthropic-beta")
        if raw_beta:
            supported = [
                b.strip() for b in raw_beta.split(",")
                if not b.strip().startswith("context-")
            ]
            if supported:
                upstream_headers["anthropic-beta"] = ", ".join(supported)

        # Use http.client for proper streaming support
        conn = http.client.HTTPSConnection(COPILOT_HOST, context=SSL_CTX)
        try:
            conn.request(
                "POST", self.path, body=body_to_send, headers=upstream_headers
            )
            resp: http.client.HTTPResponse = conn.getresponse()

            # Retry once on 401 with a refreshed token
            if resp.status == 401:
                logger.warning("Got 401, refreshing token and retrying")
                resp.read()  # drain response before reusing connection
                conn.close()
                new_token: str = token_manager.invalidate()
                upstream_headers["Authorization"] = f"Bearer {new_token}"
                conn = http.client.HTTPSConnection(COPILOT_HOST, context=SSL_CTX)
                conn.request(
                    "POST", self.path, body=body_to_send, headers=upstream_headers
                )
                resp = conn.getresponse()

            # Forward status
            self.send_response(resp.status)

            # Forward relevant headers
            is_stream: bool = False
            content_type: str = resp.getheader("Content-Type", "")
            if content_type:
                self.send_header("Content-Type", content_type)
                if "event-stream" in content_type:
                    is_stream = True

            cache_control: str | None = resp.getheader("Cache-Control")
            if cache_control:
                self.send_header("Cache-Control", cache_control)

            if not is_stream:
                resp_length: str | None = resp.getheader("Content-Length")
                if resp_length:
                    self.send_header("Content-Length", resp_length)

            self.end_headers()

            # Forward body and collect for logging
            elapsed_ms: float
            if is_stream:
                collected: bytearray = bytearray()
                while True:
                    chunk: bytes = resp.read(1)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    collected.extend(chunk)
                    if chunk == b"\n":
                        self.wfile.flush()

                elapsed_ms = (time.monotonic() - t0) * 1000
                stream_text: str = self._extract_stream_text(
                    collected.decode(errors="replace")
                )
                logger.info(
                    "<<< %dms %s",
                    elapsed_ms,
                    summarize_response(resp.status, None, stream_text),
                )
                log_jsonl({
                    "ts": time.time(),
                    "path": self.path,
                    "request": self._request_log_entry(parsed_body),
                    "response": {
                        "status": resp.status,
                        "stream": True,
                        "text_preview": stream_text[:500],
                        "usage": self._extract_stream_usage(
                            collected.decode(errors="replace")
                        ),
                    },
                    "elapsed_ms": round(elapsed_ms),
                })
            else:
                resp_data: bytes = resp.read()
                self.wfile.write(resp_data)

                elapsed_ms = (time.monotonic() - t0) * 1000
                resp_body: JsonDict | None = None
                try:
                    resp_body = json.loads(resp_data)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

                logger.info(
                    "<<< %dms %s",
                    elapsed_ms,
                    summarize_response(resp.status, resp_body, None),
                )
                log_jsonl({
                    "ts": time.time(),
                    "path": self.path,
                    "request": self._request_log_entry(parsed_body),
                    "response": {
                        "status": resp.status,
                        "stream": False,
                        "body": resp_body,
                    },
                    "elapsed_ms": round(elapsed_ms),
                })

        finally:
            conn.close()

    @staticmethod
    def _request_log_entry(body: JsonDict) -> JsonDict:
        """Create a log-safe version of the request (truncate large fields)."""
        entry: JsonDict = {
            "model": body.get("model", ""),
            "stream": body.get("stream", False),
            "max_tokens": body.get("max_tokens"),
            "n_messages": len(body.get("messages", [])),
        }
        # Include tool names if any
        tools: list[JsonDict] | None = body.get("tools")
        if tools:
            entry["tools"] = [t.get("name", "") for t in tools]
        # Last user message
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    entry["last_user_message"] = content[:500]
                elif isinstance(content, list):
                    texts = [
                        b.get("text", "")
                        for b in content
                        if b.get("type") == "text"
                    ]
                    entry["last_user_message"] = " ".join(texts)[:500]
                break
        return entry

    @staticmethod
    def _extract_stream_text(raw: str) -> str:
        """Extract concatenated text from an Anthropic SSE stream."""
        parts: list[str] = []
        for line in raw.split("\n"):
            if not line.startswith("data: "):
                continue
            try:
                event: JsonDict = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if event.get("type") == "content_block_delta":
                delta: JsonDict = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    parts.append(delta.get("text", ""))
        return "".join(parts)

    @staticmethod
    def _extract_stream_usage(raw: str) -> JsonDict:
        """Extract usage info from an Anthropic SSE stream."""
        usage: JsonDict = {}
        for line in raw.split("\n"):
            if not line.startswith("data: "):
                continue
            try:
                event: JsonDict = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if event.get("type") == "message_start":
                msg_usage: JsonDict = event.get("message", {}).get("usage", {})
                if msg_usage:
                    usage.update(msg_usage)
            elif event.get("type") == "message_delta":
                delta_usage: JsonDict = event.get("usage", {})
                if delta_usage:
                    usage.update(delta_usage)
        return usage

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        # Suppress default BaseHTTPRequestHandler logging (we do our own)
        pass


if __name__ == "__main__":
    args = parse_args()
    _log_dir = Path(args.log_dir)
    _api_key = args.api_key

    setup_logging(_log_dir, args.log_level)
    token_manager = TokenManager()

    logger.info("cc-gh-proxy starting on http://%s:%d", args.host, args.port)
    logger.info("  Upstream: %s", COPILOT_HOST)
    if _api_key:
        logger.info("  API key: required (x-api-key)")
    else:
        logger.info("  API key: not configured (open access)")
    logger.info("  Token auto-refresh: every %ds", TokenManager.REFRESH_INTERVAL)
    logger.info("  Logs: %s", _log_dir)

    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping proxy.")
        server.server_close()
