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
import hmac
import http.client
import ipaddress
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
COPILOT_OAUTH_CLIENT_ID: str = "Iv1.b507a08c87ecfe98"
COPILOT_TOKEN_URL: str = "https://api.github.com/copilot_internal/v2/token"
MAX_BODY_SIZE: int = 10 * 1024 * 1024  # 10 MB
JsonDict = dict[str, Any]


class TokenError(Exception):
    """Raised when the GitHub OAuth token cannot be obtained or refreshed."""

logger: logging.Logger = logging.getLogger("cc-gh-proxy")

# Set in main() before server starts
_log_dir: Path = Path()
_api_key: str | None = None
_log_requests: bool = False  # Log request/response content (opt-in)
_upstream_model: str | None = None  # Override model for all requests

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
    p.add_argument(
        "--log-requests",
        action="store_true",
        default=os.environ.get("PROXY_LOG_REQUESTS", "").lower() in ("1", "true", "yes"),
        help="log request/response content including message text (env: PROXY_LOG_REQUESTS, default: off)",
    )
    p.add_argument(
        "--upstream-model",
        default=os.environ.get("PROXY_UPSTREAM_MODEL"),
        help="force all requests to use this model (env: PROXY_UPSTREAM_MODEL). "
             "Claude models (claude-opus-4.6, claude-sonnet-4.6, claude-haiku-4.5) use "
             "native pass-through. Non-Claude models require --copilot-auth and OpenAI "
             "translation (EXPERIMENTAL). Available: gpt-5-mini (0x), gpt-4.1 (0x), "
             "gpt-5.1, gpt-5.2, gpt-4o, gemini-2.5-pro, grok-code-fast-1. "
             "Codex models (gpt-5.x-codex) use /responses only and are NOT supported",
    )
    p.add_argument(
        "--copilot-auth",
        action="store_true",
        default=os.environ.get("PROXY_COPILOT_AUTH", "").lower() in ("1", "true", "yes"),
        help="use Copilot OAuth app for auth (required for non-Claude models). "
             "Performs a one-time device flow on first run (env: PROXY_COPILOT_AUTH)",
    )
    return p.parse_args()


def setup_logging(log_dir: Path, level: str) -> None:
    """Configure console and file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_dir.chmod(0o700)
    logger.setLevel(getattr(logging, level, logging.INFO))

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("[proxy] %(message)s"))
    logger.addHandler(console)

    # Pre-create with restricted permissions before FileHandler opens it
    log_file = log_dir / "proxy.log"
    fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.close(fd)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _fmt_size(n: int) -> str:
    """Format byte count as human-readable size."""
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def log_jsonl(entry: JsonDict) -> None:
    """Append a JSON line to the requests log."""
    path = _log_dir / "requests.jsonl"
    with open(path, "a", opener=lambda p, f: os.open(p, f, 0o600)) as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _content_size(content: Any) -> int:
    """Estimate the size of a message content field in bytes."""
    if isinstance(content, str):
        return len(content.encode())
    return len(json.dumps(content).encode())


def summarize_messages(body: JsonDict) -> list[str]:
    """Return detail lines describing the messages in the request."""
    messages: list[JsonDict] = body.get("messages", [])
    lines: list[str] = []

    # System prompt size
    system = body.get("system")
    if system:
        lines.append(f"system: {_fmt_size(_content_size(system))}")

    # Count tools defined
    tools: list[JsonDict] | None = body.get("tools")
    if tools:
        lines.append(f"tools: {len(tools)} defined")

    # Role counts
    role_counts: dict[str, int] = {}
    for msg in messages:
        role: str = msg.get("role", "?")
        role_counts[role] = role_counts.get(role, 0) + 1
    if role_counts:
        parts = [f"{r}={c}" for r, c in role_counts.items()]
        lines.append(f"messages: {' '.join(parts)}")

    # Tool use summary: count by tool name, and collect Read paths
    tool_counts: dict[str, int] = {}
    read_paths: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_use":
                name: str = block.get("name", "?")
                tool_counts[name] = tool_counts.get(name, 0) + 1
                if name == "Read" and isinstance(block.get("input"), dict):
                    fpath: str = block["input"].get("file_path", "")
                    if fpath:
                        read_paths.append(fpath)

    if tool_counts:
        parts = [f"{name}({count})" for name, count in sorted(tool_counts.items())]
        lines.append(f"tool_uses: {' '.join(parts)}")

    if read_paths:
        for p in read_paths:
            lines.append(f"  read: {p}")

    return lines


def summarize_request(body: JsonDict) -> str:
    """One-line summary of a request for the console log."""
    model: str = body.get("model", "?")
    stream: bool = body.get("stream", False)
    n_msgs: int = len(body.get("messages", []))
    flag: str = " [stream]" if stream else ""
    summary = f"{model} ({n_msgs} msgs{flag})"

    if _log_requests:
        # Include last user message preview only when content logging is enabled
        last_user: str = ""
        for msg in reversed(body.get("messages", [])):
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
        summary += f' "{preview}"'

    return summary


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
        cache_info: str = f", cached={cached}" if cached else ""
        summary = f"OK in={inp} out={out}{cache_info} stop={stop}"
        if _log_requests:
            text: str = ""
            for block in body.get("content", []):
                if block.get("type") == "text":
                    text = block.get("text", "")[:80].replace("\n", " ")
                    break
            summary += f' "{text}..."'
        return summary

    if stream_text is not None:
        summary = "OK [streamed]"
        if _log_requests:
            preview: str = stream_text[:80].replace("\n", " ")
            summary += f' "{preview}..."'
        return summary

    return f"HTTP {status}"


# ---------------------------------------------------------------------------
# Token & model helpers
# ---------------------------------------------------------------------------

def get_gh_token() -> str:
    result = subprocess.run(
        ["gh", "auth", "token"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise TokenError("Failed to get gh token. Run: gh auth refresh -s copilot")
    return result.stdout.strip()


SSL_CTX: ssl.SSLContext = ssl.create_default_context()


class TokenManager:
    """Thread-safe gh OAuth token with auto-refresh."""

    REFRESH_INTERVAL: float = 3600  # Re-fetch every hour
    RETRY_INTERVAL: float = 30     # Retry on failure after 30s

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        try:
            self._token: str = get_gh_token()
        except TokenError as e:
            logger.error("%s", e)
            sys.exit(1)
        self._fetched_at: float = time.monotonic()
        logger.info("Token acquired successfully")

    def get_token(self) -> str:
        """Return a valid token, refreshing if stale."""
        # Fast lockless pre-check: float read is atomic in CPython (GIL).
        if time.monotonic() - self._fetched_at < self.REFRESH_INTERVAL:
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
            logger.info("Token refreshed successfully")
        except TokenError:
            logger.error("Token refresh failed, keeping old token")
        return self._token


class CopilotTokenManager:
    """Manages a Copilot API token obtained via the Copilot OAuth app.

    Required for non-Claude models on /chat/completions.
    Uses a two-step flow:
      1. GitHub OAuth device flow -> access token (one-time, cached to disk)
      2. Exchange access token -> short-lived Copilot API token (~30 min)
    """

    TOKEN_CACHE_DIR: Path = Path.home() / ".config" / "cc-gh-proxy"
    ACCESS_TOKEN_FILE: str = "copilot-access-token"
    COPILOT_TOKEN_FILE: str = "copilot-api-token.json"

    _HEADERS: dict[str, str] = {
        "Accept": "application/json",
        "Editor-Version": "vscode/1.96.0",
        "Editor-Plugin-Version": "copilot/1.200.0",
        "User-Agent": "GithubCopilot/1.200.0",
    }

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self.TOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.TOKEN_CACHE_DIR.chmod(0o700)
        self._access_token: str = self._load_or_create_access_token()
        self._copilot_token: str = ""
        self._copilot_expires_at: float = 0
        self._refresh_copilot_token()

    def get_token(self) -> str:
        """Return a valid Copilot API token, refreshing if near expiry."""
        if time.time() < self._copilot_expires_at - 300:  # 5 min buffer
            return self._copilot_token
        with self._lock:
            if time.time() < self._copilot_expires_at - 300:
                return self._copilot_token
            self._refresh_copilot_token()
            return self._copilot_token

    def invalidate(self) -> str:
        """Force a refresh."""
        with self._lock:
            self._refresh_copilot_token()
            return self._copilot_token

    def _load_or_create_access_token(self) -> str:
        """Load cached access token or run device flow."""
        token_file: Path = self.TOKEN_CACHE_DIR / self.ACCESS_TOKEN_FILE
        if token_file.exists():
            token: str = token_file.read_text().strip()
            if token:
                logger.info("Copilot access token loaded from cache")
                return token

        # Device code flow
        logger.info("Starting Copilot OAuth device flow...")
        import urllib.request
        import urllib.parse

        # Step 1: Get device code
        data: bytes = urllib.parse.urlencode({
            "client_id": COPILOT_OAUTH_CLIENT_ID,
            "scope": "read:user",
        }).encode()
        req = urllib.request.Request(
            "https://github.com/login/device/code",
            data=data,
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            device_info: JsonDict = json.loads(resp.read())

        device_code: str = device_info["device_code"]
        user_code: str = device_info["user_code"]
        verification_uri: str = device_info["verification_uri"]
        interval: int = device_info.get("interval", 5)

        print(f"\n  Copilot OAuth: Open {verification_uri}")
        print(f"  Enter code: {user_code}\n")
        print("  Waiting for authorization...", flush=True)

        # Step 2: Poll for access token
        poll_data: bytes = urllib.parse.urlencode({
            "client_id": COPILOT_OAUTH_CLIENT_ID,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }).encode()

        while True:
            time.sleep(interval)
            poll_req = urllib.request.Request(
                "https://github.com/login/oauth/access_token",
                data=poll_data,
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(poll_req) as resp:
                poll_resp: JsonDict = json.loads(resp.read())

            if "access_token" in poll_resp:
                token = poll_resp["access_token"]
                # Cache it
                fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                os.write(fd, token.encode())
                os.close(fd)
                logger.info("Copilot access token acquired and cached")
                return token

            error: str = poll_resp.get("error", "")
            if error == "authorization_pending":
                continue
            elif error == "slow_down":
                interval += 5
                continue
            else:
                raise TokenError(f"Device flow failed: {poll_resp}")

    def _refresh_copilot_token(self) -> None:
        """Exchange access token for a short-lived Copilot API token."""
        import urllib.request

        headers: dict[str, str] = {
            **self._HEADERS,
            "Authorization": f"token {self._access_token}",
        }
        req = urllib.request.Request(COPILOT_TOKEN_URL, headers=headers)
        try:
            with urllib.request.urlopen(req) as resp:
                data: JsonDict = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 401 or e.code == 404:
                # Access token may be revoked — clear cache and re-auth
                cache_file: Path = self.TOKEN_CACHE_DIR / self.ACCESS_TOKEN_FILE
                if cache_file.exists():
                    cache_file.unlink()
                raise TokenError(
                    f"Copilot token exchange failed (HTTP {e.code}). "
                    "Access token may be invalid. Restart the proxy to re-authenticate."
                )
            raise

        self._copilot_token = data["token"]
        expires_at: int | str = data.get("expires_at", 0)
        if isinstance(expires_at, str):
            expires_at = int(expires_at)
        # GitHub returns unix timestamp in seconds
        self._copilot_expires_at = float(expires_at) if expires_at > 10_000_000_000 else float(expires_at)
        if self._copilot_expires_at < 10_000_000_000:
            # already in seconds, keep as-is
            pass

        # Derive API base URL from token if present
        import re as _re
        match = _re.search(r"proxy-ep=([^;\s]+)", self._copilot_token)
        if match:
            host: str = match.group(1).replace("proxy.", "api.")
            if not host.startswith("http"):
                host = f"https://{host}"
            logger.info("Copilot API token acquired (expires in %dm, endpoint: %s)",
                        (self._copilot_expires_at - time.time()) / 60, host)
        else:
            logger.info("Copilot API token acquired (expires in %dm)",
                        (self._copilot_expires_at - time.time()) / 60)

        # Cache the token
        cache_file = self.TOKEN_CACHE_DIR / self.COPILOT_TOKEN_FILE
        fd = os.open(cache_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.write(fd, json.dumps(data).encode())
        os.close(fd)


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


# Module-level constants for model name mapping (avoid rebuilding per call)
_MODEL_STATIC_MAP: dict[str, str] = {
    "claude-opus-4-6": "claude-opus-4.6",
    "claude-sonnet-4-6": "claude-sonnet-4.6",
    "claude-haiku-4-5": "claude-haiku-4.5",
}
_MODEL_FAMILY_MAP: dict[str, str] = {
    "claude-opus-4": "claude-opus-4.6",
    "claude-sonnet-4": "claude-sonnet-4.6",
    "claude-haiku-4": "claude-haiku-4.5",
}


def map_model_name(model: str) -> str:
    """Map Anthropic model IDs to Copilot model names.

    Claude Code may send:
      claude-opus-4-6, claude-opus-4-6[1m], claude-opus-4-6-20260312, etc.
    Copilot expects:
      claude-opus-4.6, claude-sonnet-4.6, claude-haiku-4.5
    """
    # Strip bracket suffixes: claude-opus-4-6[1m] -> claude-opus-4-6
    model = re.sub(r"\[[^\]]*\]$", "", model)

    if model in _MODEL_STATIC_MAP:
        return _MODEL_STATIC_MAP[model]

    # Strip date suffixes: claude-opus-4-6-20260312 -> claude-opus-4-6
    stripped: str = re.sub(r"-\d{8}$", "", model)
    if stripped in _MODEL_STATIC_MAP:
        return _MODEL_STATIC_MAP[stripped]

    # Pattern: claude-{tier}-{major}-{minor} -> claude-{tier}-{major}.{minor}
    m = re.match(r"^(claude-(?:opus|sonnet|haiku)-\d+)-(\d+)$", stripped)
    if m:
        return f"{m.group(1)}.{m.group(2)}"

    # Base family: claude-opus-4 -> claude-opus-4.6 (latest known)
    if stripped in _MODEL_FAMILY_MAP:
        return _MODEL_FAMILY_MAP[stripped]

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


# Anthropic Messages API top-level fields (allowlist)
ALLOWED_BODY_FIELDS: set[str] = {
    "model", "messages", "max_tokens",
    "temperature", "top_p", "top_k", "stop_sequences",
    "system",
    "tools", "tool_choice",
    "stream",
    "thinking",
    "metadata",
    "service_tier",
}


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

    # Drop any fields not in the Anthropic Messages API spec
    unknown = [k for k in body if k not in ALLOWED_BODY_FIELDS]
    if unknown:
        for key in unknown:
            logger.debug("Stripping unsupported field: %s", key)
        body = {k: v for k, v in body.items() if k in ALLOWED_BODY_FIELDS}
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
# OpenAI format translation (for non-Claude models)
# ---------------------------------------------------------------------------

def _is_claude_model(model: str) -> bool:
    """Return True if the model name is a Claude/Anthropic model."""
    return model.startswith("claude-")


def anthropic_to_openai(body: JsonDict, model: str) -> JsonDict:
    """Convert Anthropic Messages API request to OpenAI Chat Completions format."""
    messages: list[JsonDict] = []

    # Handle system prompt
    system: Any = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text: str = "\n".join(
                block["text"] for block in system if block.get("type") == "text"
            )
            if text:
                messages.append({"role": "system", "content": text})

    # Convert messages
    for msg in body.get("messages", []):
        role: str = msg["role"]
        content: Any = msg.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            parts: list[str] = []
            tool_calls: list[JsonDict] = []
            tool_results: list[JsonDict] = []

            for block in content:
                btype: str | None = block.get("type")
                if btype == "text":
                    parts.append(block["text"])
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"]),
                        },
                    })
                elif btype == "tool_result":
                    result_content: Any = block.get("content", "")
                    if isinstance(result_content, list):
                        result_content = "\n".join(
                            b.get("text", "") for b in result_content
                            if b.get("type") == "text"
                        )
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": str(result_content),
                    })

            if role == "assistant":
                m: JsonDict = {"role": "assistant"}
                if parts:
                    m["content"] = "\n".join(parts)
                if tool_calls:
                    m["tool_calls"] = tool_calls
                messages.append(m)
            elif role == "user":
                if tool_results:
                    messages.extend(tool_results)
                if parts:
                    messages.append({"role": "user", "content": "\n".join(parts)})
            else:
                if parts:
                    messages.append({"role": role, "content": "\n".join(parts)})

    oai: JsonDict = {
        "model": model,
        "messages": messages,
        "max_tokens": body.get("max_tokens", 4096),
    }

    if body.get("temperature") is not None:
        oai["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        oai["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        oai["stop"] = body["stop_sequences"]
    if body.get("stream"):
        oai["stream"] = True
        oai["stream_options"] = {"include_usage": True}

    # Convert tools
    tools: list[JsonDict] | None = body.get("tools")
    if tools:
        oai_tools: list[JsonDict] = []
        for tool in tools:
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        oai["tools"] = oai_tools

    return oai


def openai_to_anthropic(oai_resp: JsonDict, model: str) -> JsonDict:
    """Convert OpenAI Chat Completions response to Anthropic Messages format."""
    choice: JsonDict = oai_resp["choices"][0]
    msg: JsonDict = choice["message"]

    content: list[JsonDict] = []
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})

    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            content.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": json.loads(tc["function"]["arguments"]),
            })

    stop_map: dict[str, str] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }

    usage: JsonDict = oai_resp.get("usage", {})

    return {
        "id": oai_resp.get("id", "msg_proxy"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_map.get(choice.get("finish_reason", ""), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _format_sse(event: str, data: JsonDict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def openai_stream_to_anthropic_events(raw: str, model: str) -> str:
    """Convert collected OpenAI SSE stream to Anthropic SSE stream."""
    chunks: list[JsonDict] = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        data: str = line[6:]
        if data == "[DONE]":
            break
        try:
            chunks.append(json.loads(data))
        except json.JSONDecodeError:
            continue

    if not chunks:
        return ""

    # Extract usage from final chunk
    input_tokens: int = 0
    output_tokens: int = 0
    for chunk in chunks:
        usage: JsonDict | None = chunk.get("usage")
        if usage:
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

    parts: list[str] = []

    # message_start
    parts.append(_format_sse("message_start", {
        "type": "message_start",
        "message": {
            "id": chunks[0].get("id", "msg_proxy"),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    }))

    content_started: bool = False
    tool_index: int = -1

    for chunk in chunks:
        choices: list[JsonDict] = chunk.get("choices", [])
        if not choices:
            continue
        choice: JsonDict = choices[0]
        delta: JsonDict = choice.get("delta", {})

        # Text content
        if delta.get("content"):
            if not content_started:
                parts.append(_format_sse("content_block_start", {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }))
                content_started = True
            parts.append(_format_sse("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": delta["content"]},
            }))

        # Tool calls
        if delta.get("tool_calls"):
            for tc in delta["tool_calls"]:
                if tc.get("id"):
                    tool_index += 1
                    block_index: int = (1 if content_started else 0) + tool_index
                    parts.append(_format_sse("content_block_start", {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": {},
                        },
                    }))
                if tc.get("function", {}).get("arguments"):
                    block_index = (1 if content_started else 0) + tool_index
                    parts.append(_format_sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": tc["function"]["arguments"],
                        },
                    }))

        # Finish
        if choice.get("finish_reason"):
            stop_map: dict[str, str] = {
                "stop": "end_turn",
                "length": "max_tokens",
                "tool_calls": "tool_use",
            }
            stop: str = stop_map.get(choice["finish_reason"], "end_turn")

            total_blocks: int = (1 if content_started else 0) + max(0, tool_index + 1)
            for i in range(total_blocks):
                parts.append(_format_sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": i,
                }))

            parts.append(_format_sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            }))

    parts.append(_format_sse("message_stop", {"type": "message_stop"}))
    return "".join(parts)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        try:
            self._handle_post()
        except (ConnectionResetError, BrokenPipeError):
            logger.debug("Client disconnected")
        except Exception:
            logger.exception("Unhandled error handling request")
            try:
                self.send_error(500, "Internal server error")
            except Exception:
                pass

    def _handle_post(self) -> None:
        # Only allow the Anthropic messages endpoint
        if not self.path.startswith("/v1/messages"):
            self.send_error(404, "Not found")
            return

        # Check API key if configured
        if _api_key:
            client_key: str = self.headers.get("x-api-key", "")
            if not hmac.compare_digest(client_key, _api_key):
                self.send_error(401, "Invalid or missing API key")
                logger.warning("Rejected request: bad x-api-key")
                return

        t0: float = time.monotonic()
        try:
            content_length: int = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Invalid Content-Length")
            return
        if content_length > MAX_BODY_SIZE:
            self.send_error(413, "Request body too large")
            return
        raw_body: bytes = self.rfile.read(content_length)

        # Rewrite model name and strip unsupported fields
        body_to_send: bytes
        parsed_body: JsonDict
        body_to_send, parsed_body = rewrite_body(raw_body)

        # Determine the effective upstream model
        effective_model: str = _upstream_model or parsed_body.get("model", "")
        use_openai: bool = _upstream_model is not None and not _is_claude_model(_upstream_model)

        if use_openai:
            # Override model in body for logging
            parsed_body["model"] = _upstream_model
        elif _upstream_model:
            # Force a specific Claude model
            parsed_body["model"] = _upstream_model
            body_to_send = json.dumps(parsed_body).encode()

        req_size: int = len(body_to_send)
        route: str = "openai" if use_openai else "native"
        logger.info(">>> %s %s (%s) [%s]", self.path, summarize_request(parsed_body), _fmt_size(req_size), route)
        for detail in summarize_messages(parsed_body):
            logger.info("    %s", detail)

        if use_openai:
            self._handle_openai_path(t0, parsed_body, effective_model)
        else:
            self._handle_native_path(t0, body_to_send, parsed_body)

    def _handle_native_path(
        self, t0: float, body_to_send: bytes, parsed_body: JsonDict,
    ) -> None:
        """Forward request to Copilot's native Anthropic /v1/messages endpoint."""
        req_size: int = len(body_to_send)

        # Build headers for upstream
        current_token: str = token_manager.get_token()
        upstream_headers: dict[str, str] = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": "application/json",
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

        conn = http.client.HTTPSConnection(COPILOT_HOST, context=SSL_CTX)
        try:
            conn.request(
                "POST", "/v1/messages", body=body_to_send, headers=upstream_headers
            )
            resp: http.client.HTTPResponse = conn.getresponse()

            # Retry once on 401 with a refreshed token
            if resp.status == 401:
                logger.warning("Got 401, refreshing token and retrying")
                resp.read()
                conn.close()
                new_token: str = token_manager.invalidate()
                upstream_headers["Authorization"] = f"Bearer {new_token}"
                conn = http.client.HTTPSConnection(COPILOT_HOST, context=SSL_CTX)
                conn.request(
                    "POST", "/v1/messages", body=body_to_send, headers=upstream_headers
                )
                resp = conn.getresponse()

            # Forward status and headers
            self.send_response(resp.status)
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

            self._forward_and_log(resp, is_stream, t0, req_size, parsed_body)
        finally:
            conn.close()

    def _handle_openai_path(
        self, t0: float, parsed_body: JsonDict, model: str,
    ) -> None:
        """Translate to OpenAI format, send to /chat/completions, translate back."""
        if copilot_token_manager is None:
            self.send_error(503, "Non-Claude models require --copilot-auth")
            logger.error("OpenAI path requested but copilot_token_manager not initialized")
            return

        is_stream: bool = parsed_body.get("stream", False)
        oai_body: JsonDict = anthropic_to_openai(parsed_body, model)
        oai_bytes: bytes = json.dumps(oai_body).encode()
        req_size: int = len(oai_bytes)

        current_token: str = copilot_token_manager.get_token()
        upstream_headers: dict[str, str] = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": "application/json",
            "Content-Length": str(len(oai_bytes)),
            "Editor-Version": "vscode/1.96.0",
            "Editor-Plugin-Version": "copilot/1.200.0",
            "User-Agent": "GithubCopilot/1.200.0",
            "Copilot-Integration-Id": "vscode-chat",
        }

        conn = http.client.HTTPSConnection(COPILOT_HOST, context=SSL_CTX)
        try:
            conn.request(
                "POST", "/chat/completions", body=oai_bytes, headers=upstream_headers
            )
            resp: http.client.HTTPResponse = conn.getresponse()

            # Retry once on 401/403
            if resp.status in (401, 403):
                logger.warning("Got %d, refreshing Copilot token and retrying", resp.status)
                resp.read()
                conn.close()
                new_token: str = copilot_token_manager.invalidate()
                upstream_headers["Authorization"] = f"Bearer {new_token}"
                conn = http.client.HTTPSConnection(COPILOT_HOST, context=SSL_CTX)
                conn.request(
                    "POST", "/chat/completions", body=oai_bytes, headers=upstream_headers
                )
                resp = conn.getresponse()

            if resp.status != 200:
                # Forward error as-is
                resp_data: bytes = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_data)))
                self.end_headers()
                self.wfile.write(resp_data)
                elapsed_ms: float = (time.monotonic() - t0) * 1000
                logger.info("<<< %dms HTTP %d (%s -> %s)",
                            elapsed_ms, resp.status,
                            _fmt_size(req_size), _fmt_size(len(resp_data)))
                return

            if is_stream:
                # Collect full OpenAI stream, then translate and send as Anthropic SSE
                collected: bytearray = bytearray()
                while True:
                    chunk: bytes = resp.read(4096)
                    if not chunk:
                        break
                    collected.extend(chunk)

                oai_raw: str = collected.decode(errors="replace")
                anthropic_stream: str = openai_stream_to_anthropic_events(oai_raw, model)
                resp_bytes: bytes = anthropic_stream.encode()

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(resp_bytes)
                self.wfile.flush()

                resp_size: int = len(resp_bytes)
                elapsed_ms = (time.monotonic() - t0) * 1000
                stream_text: str = self._extract_stream_text(anthropic_stream)
                logger.info(
                    "<<< %dms %s (%s -> %s)",
                    elapsed_ms,
                    summarize_response(200, None, stream_text),
                    _fmt_size(req_size), _fmt_size(resp_size),
                )
                stream_resp_log: JsonDict = {
                    "status": 200,
                    "stream": True,
                    "translated": True,
                    "usage": self._extract_stream_usage(anthropic_stream),
                    "req_bytes": req_size,
                    "resp_bytes": resp_size,
                }
                if _log_requests:
                    stream_resp_log["text_preview"] = stream_text[:500]
                log_jsonl({
                    "ts": time.time(),
                    "path": self.path,
                    "request": self._request_log_entry(parsed_body),
                    "response": stream_resp_log,
                    "elapsed_ms": round(elapsed_ms),
                })
            else:
                resp_data = resp.read()
                oai_resp: JsonDict = json.loads(resp_data)
                anthropic_resp: JsonDict = openai_to_anthropic(oai_resp, model)
                resp_bytes = json.dumps(anthropic_resp).encode()

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)

                resp_size = len(resp_bytes)
                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.info(
                    "<<< %dms %s (%s -> %s)",
                    elapsed_ms,
                    summarize_response(200, anthropic_resp, None),
                    _fmt_size(req_size), _fmt_size(resp_size),
                )
                nonstream_resp_log: JsonDict = {
                    "status": 200,
                    "stream": False,
                    "translated": True,
                    "req_bytes": req_size,
                    "resp_bytes": resp_size,
                }
                if anthropic_resp:
                    nonstream_resp_log["usage"] = anthropic_resp.get("usage", {})
                    nonstream_resp_log["stop_reason"] = anthropic_resp.get("stop_reason")
                if _log_requests:
                    nonstream_resp_log["body"] = anthropic_resp
                log_jsonl({
                    "ts": time.time(),
                    "path": self.path,
                    "request": self._request_log_entry(parsed_body),
                    "response": nonstream_resp_log,
                    "elapsed_ms": round(elapsed_ms),
                })
        finally:
            conn.close()

    def _forward_and_log(
        self, resp: http.client.HTTPResponse, is_stream: bool,
        t0: float, req_size: int, parsed_body: JsonDict,
    ) -> None:
        """Forward upstream response to client and log it."""
        elapsed_ms: float
        if is_stream:
            collected: bytearray = bytearray()
            while True:
                chunk: bytes = resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                collected.extend(chunk)
                if b"\n" in chunk:
                    self.wfile.flush()

            resp_size: int = len(collected)
            elapsed_ms = (time.monotonic() - t0) * 1000
            decoded_stream: str = collected.decode(errors="replace")
            stream_text: str = self._extract_stream_text(decoded_stream)
            logger.info(
                "<<< %dms %s (%s -> %s)",
                elapsed_ms,
                summarize_response(resp.status, None, stream_text),
                _fmt_size(req_size), _fmt_size(resp_size),
            )
            stream_resp_log: JsonDict = {
                "status": resp.status,
                "stream": True,
                "usage": self._extract_stream_usage(decoded_stream),
                "req_bytes": req_size,
                "resp_bytes": resp_size,
            }
            if _log_requests:
                stream_resp_log["text_preview"] = stream_text[:500]
            log_jsonl({
                "ts": time.time(),
                "path": self.path,
                "request": self._request_log_entry(parsed_body),
                "response": stream_resp_log,
                "elapsed_ms": round(elapsed_ms),
            })
        else:
            resp_data: bytes = resp.read()
            resp_size = len(resp_data)
            self.wfile.write(resp_data)

            elapsed_ms = (time.monotonic() - t0) * 1000
            resp_body: JsonDict | None = None
            try:
                resp_body = json.loads(resp_data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

            logger.info(
                "<<< %dms %s (%s -> %s)",
                elapsed_ms,
                summarize_response(resp.status, resp_body, None),
                _fmt_size(req_size), _fmt_size(resp_size),
            )
            nonstream_resp_log: JsonDict = {
                "status": resp.status,
                "stream": False,
                "req_bytes": req_size,
                "resp_bytes": resp_size,
            }
            if resp_body:
                nonstream_resp_log["usage"] = resp_body.get("usage", {})
                nonstream_resp_log["stop_reason"] = resp_body.get("stop_reason")
            if _log_requests and resp_body:
                nonstream_resp_log["body"] = resp_body
            log_jsonl({
                "ts": time.time(),
                "path": self.path,
                "request": self._request_log_entry(parsed_body),
                "response": nonstream_resp_log,
                "elapsed_ms": round(elapsed_ms),
            })

    @staticmethod
    def _request_log_entry(body: JsonDict) -> JsonDict:
        """Create a log-safe version of the request (truncate large fields)."""
        entry: JsonDict = {
            "model": body.get("model", ""),
            "stream": body.get("stream", False),
            "max_tokens": body.get("max_tokens"),
            "n_messages": len(body.get("messages", [])),
        }
        # Include metadata.user_id if present (Claude Code session identifier)
        metadata: JsonDict | None = body.get("metadata")
        if isinstance(metadata, dict) and metadata.get("user_id"):
            entry["user_id"] = metadata["user_id"]
        # Include tool names if any
        tools: list[JsonDict] | None = body.get("tools")
        if tools:
            entry["tools"] = [t.get("name", "") for t in tools]
        # Last user message — only when content logging is enabled
        if _log_requests:
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
        # Route through our logger instead of BaseHTTPRequestHandler's default
        logger.debug(format, *args)


def _is_loopback(host: str) -> bool:
    """Return True if host resolves to a loopback address."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


if __name__ == "__main__":
    args = parse_args()
    _log_dir = Path(args.log_dir)
    _api_key = args.api_key
    _log_requests = args.log_requests
    _upstream_model = args.upstream_model

    setup_logging(_log_dir, args.log_level)
    token_manager = TokenManager()

    # Initialize Copilot token manager for non-Claude models
    copilot_token_manager: CopilotTokenManager | None = None
    need_copilot_auth: bool = args.copilot_auth or (
        _upstream_model is not None and not _is_claude_model(_upstream_model)
    )
    if need_copilot_auth:
        try:
            copilot_token_manager = CopilotTokenManager()
        except TokenError as e:
            logger.error("Copilot auth failed: %s", e)
            sys.exit(1)

    logger.info("cc-gh-proxy starting on http://%s:%d", args.host, args.port)
    logger.info("  Upstream: %s", COPILOT_HOST)
    if _upstream_model:
        if _is_claude_model(_upstream_model):
            logger.info("  Upstream model: %s (native Anthropic pass-through)", _upstream_model)
        else:
            logger.info("  Upstream model: %s (EXPERIMENTAL: OpenAI translation)", _upstream_model)
    if _api_key:
        logger.info("  API key: required (x-api-key)")
    else:
        logger.info("  API key: not configured (open access)")
    logger.info("  Token auto-refresh: every %ds", TokenManager.REFRESH_INTERVAL)
    logger.info("  Logs: %s", _log_dir)
    if _log_requests:
        logger.info("  Request logging: ENABLED (message content will be persisted)")

    if not _is_loopback(args.host):
        logger.warning(
            "Proxy is binding to %s — NOT a loopback address. "
            "Requests and API keys are transmitted in cleartext over the network.",
            args.host,
        )

    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping proxy.")
        server.server_close()
