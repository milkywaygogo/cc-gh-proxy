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
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any

COPILOT_HOST: str = "api.githubcopilot.com"
TAVILY_HOST: str = "api.tavily.com"
TAVILY_PRICING: dict[str, float] = {"basic": 0.005, "advanced": 0.008}
COPILOT_OAUTH_CLIENT_ID: str = "Iv1.b507a08c87ecfe98"
COPILOT_TOKEN_URL: str = "https://api.github.com/copilot_internal/v2/token"
MAX_BODY_SIZE: int = 10 * 1024 * 1024  # 10 MB
# Beta features Copilot doesn't support — strip these from anthropic-beta header.
# Add new prefixes here as Claude releases features Copilot doesn't understand.
_STRIP_BETA_PREFIXES: tuple[str, ...] = (
    "context-",          # e.g. context-1m-2025-08-07
    "advisor-tool-",     # e.g. advisor-tool-2026-03-01
)
JsonDict = dict[str, Any]


class TokenError(Exception):
    """Raised when the GitHub OAuth token cannot be obtained or refreshed."""

logger: logging.Logger = logging.getLogger("cc-gh-proxy")

# Set in main() before server starts
_log_dir: Path = Path()
_api_key: str | None = None
_log_requests: bool = False  # Log request/response content (opt-in)
_upstream_model: str | None = None  # Override model for all requests
_upstream_base_url: str | None = None  # OpenAI-compatible upstream URL (bypasses Copilot)
_upstream_api_key: str | None = None  # Bearer token for --upstream-base-url
_no_opus: bool = False  # Map any claude-opus-* request to a sonnet model
_no_opus_target: str = "claude-sonnet-4.6"  # Target sonnet model when --no-opus is set

# Tavily configuration. When set, Claude Code's "WebSearch executor" requests
# (the small follow-up call CC fires with tools=[web_search_*]) are served by
# Tavily instead of Copilot — Tavily handles the actual web search and returns
# extracted page content inline so the model rarely needs follow-up WebFetch.
_tavily_api_key: str | None = None
_tavily_search_depth: str = "advanced"  # "basic" or "advanced"
_tavily_max_results: int = 5
_tavily_spend_lock: threading.Lock = threading.Lock()
_tavily_spend: dict[str, float] = {}

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
    p.add_argument(
        "--upstream-base-url",
        default=os.environ.get("PROXY_UPSTREAM_BASE_URL"),
        help="OpenAI-compatible base URL to route requests to instead of Copilot "
             "(e.g. http://localhost:11434/v1 for Ollama). Bypasses both gh and "
             "Copilot OAuth. Combine with --upstream-model to set the model name "
             "(env: PROXY_UPSTREAM_BASE_URL)",
    )
    p.add_argument(
        "--upstream-api-key",
        default=os.environ.get("PROXY_UPSTREAM_API_KEY"),
        help="Bearer token for --upstream-base-url (env: PROXY_UPSTREAM_API_KEY)",
    )
    p.add_argument(
        "--tavily-api-key",
        default=os.environ.get("PROXY_TAVILY_API_KEY"),
        help="Tavily API key. When set, requests whose `tools` contains ONLY "
             "`web_search_*` / `web_fetch_*` server tools (Claude Code's "
             "WebSearch executor) are served by Tavily. Tavily returns "
             "extracted page content in search results, so the model rarely "
             "needs follow-up WebFetch (env: PROXY_TAVILY_API_KEY)",
    )
    p.add_argument(
        "--tavily-search-depth",
        choices=("basic", "advanced"),
        default=os.environ.get("PROXY_TAVILY_SEARCH_DEPTH", "advanced"),
        help="Tavily search depth. 'advanced' ($0.008/search) returns "
             "extracted page content; 'basic' ($0.005/search) returns snippets "
             "only (env: PROXY_TAVILY_SEARCH_DEPTH, default: advanced)",
    )
    p.add_argument(
        "--tavily-max-results",
        type=int,
        default=int(os.environ.get("PROXY_TAVILY_MAX_RESULTS", "5")),
        help="Maximum results per Tavily search "
             "(env: PROXY_TAVILY_MAX_RESULTS, default: 5)",
    )
    p.add_argument(
        "--no-opus",
        action="store_true",
        default=os.environ.get("PROXY_NO_OPUS", "").lower() in ("1", "true", "yes"),
        help="rewrite any claude-opus-* model to --no-opus-target before "
             "forwarding. Useful for avoiding the high premium-request cost "
             "of Opus on GitHub Copilot (env: PROXY_NO_OPUS)",
    )
    p.add_argument(
        "--no-opus-target",
        default=os.environ.get("PROXY_NO_OPUS_TARGET", "claude-sonnet-4.6"),
        help="target Copilot model when --no-opus rewrites an Opus request "
             "(env: PROXY_NO_OPUS_TARGET, default: claude-sonnet-4.6). "
             "Copilot currently only ships Sonnet 4.6 and 4.5, not 4.7",
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


def _opus_to_sonnet(name: str) -> str:
    """If --no-opus is enabled, rewrite any claude-opus-* model to the configured
    sonnet target. Version is NOT preserved because Copilot does not ship a sonnet
    for every opus version (e.g. opus 4.7 has no matching sonnet 4.7)."""
    if _no_opus and name.startswith("claude-opus-"):
        logger.info("Downgrading %s -> %s (--no-opus)", name, _no_opus_target)
        return _no_opus_target
    return name


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
        return _opus_to_sonnet(_MODEL_STATIC_MAP[model])

    # Strip date suffixes: claude-opus-4-6-20260312 -> claude-opus-4-6
    stripped: str = re.sub(r"-\d{8}$", "", model)
    if stripped in _MODEL_STATIC_MAP:
        return _opus_to_sonnet(_MODEL_STATIC_MAP[stripped])

    # Pattern: claude-{tier}-{major}-{minor} -> claude-{tier}-{major}.{minor}
    m = re.match(r"^(claude-(?:opus|sonnet|haiku)-\d+)-(\d+)$", stripped)
    if m:
        return _opus_to_sonnet(f"{m.group(1)}.{m.group(2)}")

    # Base family: claude-opus-4 -> claude-opus-4.6 (latest known)
    if stripped in _MODEL_FAMILY_MAP:
        return _opus_to_sonnet(_MODEL_FAMILY_MAP[stripped])

    logger.warning("Unknown model '%s', passing through as-is", model)
    return _opus_to_sonnet(model)


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


def _today_str() -> str:
    return time.strftime("%Y-%m-%d")


def _is_pure_websearch_request(body: JsonDict) -> bool:
    """True if `tools` is non-empty and every entry is a web_search_* or
    web_fetch_* server tool. This is the Tavily-eligible subset of the
    "pure server-tool" pattern."""
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            return False
        ttype = tool.get("type")
        if not isinstance(ttype, str):
            return False
        if not (ttype.startswith("web_search_") or ttype.startswith("web_fetch_")):
            return False
    return True


def _extract_search_query(body: JsonDict) -> str:
    """Pull the user's search query out of an executor-pattern request body.

    Claude Code puts the query verbatim in the last user message's content,
    either as a plain string or as a list of `{type:"text", text:...}` blocks.
    """
    messages = body.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return ""
    last = messages[-1]
    if not isinstance(last, dict):
        return ""
    content = last.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "")
                if isinstance(t, str):
                    texts.append(t)
        return " ".join(texts).strip()
    return ""


def _tavily_search(query: str) -> JsonDict:
    """Synchronous Tavily search call. Raises on transport / non-200."""
    import urllib.request
    import urllib.error

    payload: bytes = json.dumps({
        "api_key": _tavily_api_key,
        "query": query,
        "search_depth": _tavily_search_depth,
        "max_results": _tavily_max_results,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }).encode()
    req = urllib.request.Request(
        f"https://{TAVILY_HOST}/search",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
        return json.loads(resp.read())


def _tavily_to_search_results(data: JsonDict) -> list[JsonDict]:
    """Map Tavily `results` into Anthropic `web_search_result` blocks.

    Anthropic's native shape uses an opaque `encrypted_content` blob — we
    stuff the extracted page content there so the model has the same field
    layout it expects. `page_age` is left null since Tavily doesn't expose it.
    """
    out: list[JsonDict] = []
    raw = data.get("results") or []
    if not isinstance(raw, list):
        return out
    for r in raw:
        if not isinstance(r, dict):
            continue
        out.append({
            "type": "web_search_result",
            "url": r.get("url") or "",
            "title": r.get("title") or "",
            "encrypted_content": (r.get("content") or "").strip(),
            "page_age": None,
        })
    return out


def _format_tavily_results(query: str, data: JsonDict) -> str:
    """Format a Tavily response as Markdown for injection back into CC."""
    parts: list[str] = [f"# Search results for: {query}\n"]
    answer = data.get("answer")
    if isinstance(answer, str) and answer.strip():
        parts.append(f"**Answer:** {answer.strip()}\n")
    results = data.get("results") or []
    if not isinstance(results, list) or not results:
        parts.append("_No results._")
        return "\n".join(parts)
    for i, r in enumerate(results, 1):
        if not isinstance(r, dict):
            continue
        title = r.get("title") or "(no title)"
        url = r.get("url") or ""
        content = (r.get("content") or "").strip()
        parts.append(f"\n## {i}. [{title}]({url})\n")
        if content:
            parts.append(content)
    return "\n".join(parts)


def _tavily_spend_today() -> float:
    with _tavily_spend_lock:
        return _tavily_spend.get(_today_str(), 0.0)


def _record_tavily_spend(usd: float) -> float:
    with _tavily_spend_lock:
        today = _today_str()
        _tavily_spend[today] = _tavily_spend.get(today, 0.0) + usd
        return _tavily_spend[today]


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
    elif msg.get("reasoning"):
        # Reasoning models (e.g. Gemma 4 via Ollama) put thinking text in `reasoning`
        # and may produce no `content` if max_tokens is too small. Surface it as text
        # so the response is not silently empty.
        content.append({"type": "text", "text": msg["reasoning"]})

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


def _stream_responses_to_chat_completions(
    resp: http.client.HTTPResponse, wfile: Any, model: str,
) -> int:
    """Translate an OpenAI Responses-API SSE stream to a Chat-Completions SSE
    stream, writing chunks to `wfile` as they're produced. Returns total bytes
    written.

    Used when a client (e.g. Cursor) sends a Responses-API body to
    /v1/chat/completions but expects to read back a chat-completion SSE stream.
    Translation is incremental — each upstream event is converted and flushed
    immediately, so token-level latency is preserved.
    """
    chat_id: str = f"chatcmpl-{int(time.time() * 1000)}"
    created: int = int(time.time())
    bytes_written: int = 0

    # State shared across events
    state: dict[str, Any] = {
        "tool_indices": {},     # output_index/item_id -> tool_call index
        "next_tool_index": 0,
        "sent_role": False,
        "saw_tool_call": False,
        "finish_reason": "stop",
    }

    def emit(
        delta: JsonDict,
        finish_reason: str | None = None,
        usage: JsonDict | None = None,
    ) -> None:
        nonlocal bytes_written
        chunk: JsonDict = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }
        if usage is not None:
            chunk["usage"] = usage
        line: bytes = f"data: {json.dumps(chunk)}\n\n".encode()
        try:
            wfile.write(line)
            wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            raise
        bytes_written += len(line)

    def handle_event(event_type: str, data: JsonDict) -> None:
        if event_type == "response.created" or event_type == "response.in_progress":
            if not state["sent_role"]:
                emit({"role": "assistant", "content": ""})
                state["sent_role"] = True
            return

        if event_type == "response.output_text.delta":
            text: str = data.get("delta", "")
            if text:
                emit({"content": text})
            return

        if event_type == "response.output_item.added":
            item: JsonDict = data.get("item", {}) or {}
            if item.get("type") == "function_call":
                # Key by output_index when present (multiple parallel tool calls)
                key: str = (
                    str(data.get("output_index"))
                    if data.get("output_index") is not None
                    else (item.get("id") or item.get("call_id") or "")
                )
                if key not in state["tool_indices"]:
                    idx: int = state["next_tool_index"]
                    state["tool_indices"][key] = idx
                    state["next_tool_index"] += 1
                    state["saw_tool_call"] = True
                    emit({
                        "tool_calls": [{
                            "index": idx,
                            "id": item.get("call_id") or item.get("id") or "",
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": "",
                            },
                        }],
                    })
            return

        if event_type == "response.function_call_arguments.delta":
            key = (
                str(data.get("output_index"))
                if data.get("output_index") is not None
                else (data.get("item_id") or "")
            )
            idx_opt: int | None = state["tool_indices"].get(key)
            if idx_opt is None:
                # Some streams emit args without a prior output_item.added —
                # allocate an index on the fly.
                idx_opt = state["next_tool_index"]
                state["tool_indices"][key] = idx_opt
                state["next_tool_index"] += 1
                state["saw_tool_call"] = True
            delta_str: str = data.get("delta", "")
            if delta_str:
                emit({
                    "tool_calls": [{
                        "index": idx_opt,
                        "function": {"arguments": delta_str},
                    }],
                })
            return

        if event_type == "response.completed":
            resp_obj: JsonDict = data.get("response", {}) or {}
            u: JsonDict = resp_obj.get("usage", {}) or {}
            usage: JsonDict | None = None
            if u:
                usage = {
                    "prompt_tokens": u.get("input_tokens", 0),
                    "completion_tokens": u.get("output_tokens", 0),
                    "total_tokens": u.get(
                        "total_tokens",
                        u.get("input_tokens", 0) + u.get("output_tokens", 0),
                    ),
                }
            finish: str = "tool_calls" if state["saw_tool_call"] else "stop"
            emit({}, finish_reason=finish, usage=usage)
            return

        if event_type == "response.incomplete":
            # Hit max_output_tokens, content_filter, etc.
            reason_obj: JsonDict = (data.get("response", {}) or {}).get(
                "incomplete_details", {}
            ) or {}
            reason: str = reason_obj.get("reason", "")
            finish_map: dict[str, str] = {
                "max_output_tokens": "length",
                "content_filter": "content_filter",
            }
            emit({}, finish_reason=finish_map.get(reason, "stop"))
            return

        if event_type in ("response.failed", "error"):
            emit({}, finish_reason="stop")
            return

    # Incremental SSE parser. Read raw bytes, split on \n\n event boundaries.
    buf: bytes = b""
    try:
        while True:
            chunk_bytes: bytes = resp.read(4096)
            if not chunk_bytes:
                break
            buf += chunk_bytes
            while b"\n\n" in buf:
                raw_event, buf = buf.split(b"\n\n", 1)
                event_type: str = ""
                data_lines: list[str] = []
                for line in raw_event.split(b"\n"):
                    if line.startswith(b"event:"):
                        event_type = line[6:].strip().decode("utf-8", "replace")
                    elif line.startswith(b"data:"):
                        data_lines.append(
                            line[5:].lstrip().decode("utf-8", "replace")
                        )
                if not data_lines:
                    continue
                data_str: str = "\n".join(data_lines)
                if data_str == "[DONE]":
                    continue
                try:
                    event_data: JsonDict = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if not event_type:
                    event_type = event_data.get("type", "")
                handle_event(event_type, event_data)
    except (BrokenPipeError, ConnectionResetError):
        return bytes_written

    # Closing [DONE] sentinel
    try:
        done_line: bytes = b"data: [DONE]\n\n"
        wfile.write(done_line)
        wfile.flush()
        bytes_written += len(done_line)
    except (BrokenPipeError, ConnectionResetError):
        pass
    return bytes_written


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
    reasoning_buffer: list[str] = []

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
        elif delta.get("reasoning"):
            # Reasoning models stream thinking text in delta.reasoning. Buffer it
            # so we can surface it as text if no real content arrives.
            reasoning_buffer.append(delta["reasoning"])

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

            # Reasoning fallback: if we buffered reasoning but never started a real
            # text block (and no tool calls), emit reasoning as text now.
            if reasoning_buffer and not content_started and tool_index < 0:
                parts.append(_format_sse("content_block_start", {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }))
                parts.append(_format_sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "".join(reasoning_buffer)},
                }))
                content_started = True

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

    def _check_api_key(self) -> bool:
        """Validate proxy --api-key from x-api-key or Authorization: Bearer.

        Cursor and other OpenAI-compatible clients send Authorization: Bearer,
        Claude Code sends x-api-key. Accept either.
        """
        if not _api_key:
            return True
        client_key: str = self.headers.get("x-api-key", "")
        if not client_key:
            auth: str = self.headers.get("Authorization", "")
            if auth.lower().startswith("bearer "):
                client_key = auth[7:].strip()
        if not hmac.compare_digest(client_key, _api_key):
            self.send_error(401, "Invalid or missing API key")
            logger.warning("Rejected request: bad api key")
            return False
        return True

    def _handle_post(self) -> None:
        # Dispatch by path. Anthropic Messages on /v1/messages; OpenAI Chat
        # Completions passthrough on /chat/completions and /v1/chat/completions.
        if self.path.startswith("/chat/completions") or self.path.startswith("/v1/chat/completions"):
            if not self._check_api_key():
                return
            self._handle_chat_passthrough()
            return
        if self.path.startswith("/responses") or self.path.startswith("/v1/responses"):
            if not self._check_api_key():
                return
            self._handle_responses_passthrough()
            return
        if not self.path.startswith("/v1/messages"):
            self.send_error(404, "Not found")
            return

        if not self._check_api_key():
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

        # Tavily routing: when configured, intercept Claude Code's "WebSearch
        # executor" pattern (a request whose `tools` contains ONLY
        # web_search_*/web_fetch_* server tools) and serve it from Tavily.
        # Mixed requests (server tool alongside Read/Bash/...) stay on Copilot.
        pre_parsed: JsonDict | None = None
        if _tavily_api_key is not None:
            try:
                pre_parsed = json.loads(raw_body)
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
                return

        if (
            _tavily_api_key is not None
            and pre_parsed is not None
            and _is_pure_websearch_request(pre_parsed)
        ):
            req_size: int = len(raw_body)
            logger.info(
                ">>> %s %s (%s) [tavily]",
                self.path, summarize_request(pre_parsed), _fmt_size(req_size),
            )
            for detail in summarize_messages(pre_parsed):
                logger.info("    %s", detail)
            self._handle_tavily_path(t0, raw_body, pre_parsed)
            return

        # Rewrite model name and strip unsupported fields
        body_to_send: bytes
        parsed_body: JsonDict
        body_to_send, parsed_body = rewrite_body(raw_body)

        # Determine the effective upstream model and routing
        effective_model: str = _upstream_model or parsed_body.get("model", "")
        use_openai: bool = bool(_upstream_base_url) or (
            _upstream_model is not None and not _is_claude_model(_upstream_model)
        )

        if use_openai:
            # Override model in body for logging only when explicitly set
            if _upstream_model:
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

    def _handle_chat_passthrough(self) -> None:
        """Forward OpenAI /chat/completions requests to Copilot unchanged.

        Used by Cursor and other OpenAI-compatible clients. Auth header is
        swapped for the Copilot OAuth token; body and response are forwarded
        as-is (no Anthropic/OpenAI translation).
        """
        if copilot_token_manager is None:
            self.send_error(503, "/chat/completions passthrough requires --copilot-auth")
            logger.error("/chat/completions hit but copilot_token_manager not initialized")
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
        req_size: int = len(raw_body)

        # Best-effort parse for logging only; bytes are forwarded as-is.
        model: str = ""
        is_stream: bool = False
        n_messages: int = 0
        peek: JsonDict | None = None
        try:
            peek = json.loads(raw_body) if raw_body else {}
            if isinstance(peek, dict):
                model = str(peek.get("model", ""))
                is_stream = bool(peek.get("stream", False))
                msgs = peek.get("messages")
                if isinstance(msgs, list):
                    n_messages = len(msgs)
        except json.JSONDecodeError:
            pass

        logger.info(
            ">>> %s model=%s stream=%s msgs=%d (%s) [chat-passthrough]",
            self.path, model or "?", is_stream, n_messages, _fmt_size(req_size),
        )

        # Cursor (and some other clients) hard-route GPT-5.x / reasoning
        # models to /chat/completions but send a Responses-API-style body
        # (`input` instead of `messages`, plus `reasoning`, `text`, `store`,
        # etc.). Copilot's /chat/completions then rejects it with the
        # cryptic "messages must be non-empty". Detect this shape and
        # internally re-route to Copilot's /responses endpoint instead.
        if (
            n_messages == 0
            and isinstance(peek, dict)
            and "input" in peek
        ):
            logger.info(
                "chat-passthrough body looks like Responses API "
                "(keys=%s) — rerouting to /responses (translate=%s)",
                sorted(peek.keys()), is_stream,
            )
            self._forward_responses_passthrough(
                t0, raw_body, model, is_stream,
                translate_to_chat=True,
            )
            return

        # Plain empty-messages body that isn't Responses-API-shaped:
        # surface a useful warning so the user can debug their client.
        if n_messages == 0 and isinstance(peek, dict):
            logger.warning(
                "chat-passthrough request has empty/missing `messages` "
                "(keys=%s). This will likely fail with "
                "\"messages must be non-empty\".",
                sorted(peek.keys()),
            )

        def _build_headers(token: str) -> dict[str, str]:
            return {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Content-Length": str(req_size),
                "Editor-Version": "vscode/1.96.0",
                "Editor-Plugin-Version": "copilot/1.200.0",
                "User-Agent": "GithubCopilot/1.200.0",
                "Copilot-Integration-Id": "vscode-chat",
                "Accept": "text/event-stream" if is_stream else "application/json",
            }

        current_token: str = copilot_token_manager.get_token()
        conn = http.client.HTTPSConnection(COPILOT_HOST, context=SSL_CTX)
        try:
            conn.request(
                "POST", "/chat/completions",
                body=raw_body, headers=_build_headers(current_token),
            )
            resp: http.client.HTTPResponse = conn.getresponse()

            if resp.status in (401, 403):
                logger.warning("Got %d on /chat/completions, refreshing token", resp.status)
                resp.read()
                conn.close()
                new_token: str = copilot_token_manager.invalidate()
                conn = http.client.HTTPSConnection(COPILOT_HOST, context=SSL_CTX)
                conn.request(
                    "POST", "/chat/completions",
                    body=raw_body, headers=_build_headers(new_token),
                )
                resp = conn.getresponse()

            # Forward status and content-type. Stream bytes through.
            content_type: str = resp.getheader("Content-Type") or "application/json"
            self.send_response(resp.status)
            self.send_header("Content-Type", content_type)
            if "event-stream" in content_type:
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                resp_size: int = 0
                while True:
                    chunk: bytes = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    resp_size += len(chunk)
                    if b"\n" in chunk:
                        self.wfile.flush()
            else:
                data: bytes = resp.read()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                resp_size = len(data)

            elapsed_ms: float = (time.monotonic() - t0) * 1000
            logger.info(
                "<<< %dms HTTP %d [chat-passthrough] (%s -> %s)",
                elapsed_ms, resp.status,
                _fmt_size(req_size), _fmt_size(resp_size),
            )
            log_jsonl({
                "ts": time.time(),
                "path": self.path,
                "request": {
                    "model": model,
                    "stream": is_stream,
                    "n_messages": n_messages,
                    "passthrough": "chat",
                },
                "response": {
                    "status": resp.status,
                    "stream": is_stream,
                    "req_bytes": req_size,
                    "resp_bytes": resp_size,
                },
                "elapsed_ms": round(elapsed_ms),
            })
        finally:
            conn.close()

    def _handle_responses_passthrough(self) -> None:
        """Forward OpenAI-style /v1/responses requests to Copilot unchanged.

        Cursor and other OpenAI-compatible clients use the Responses API
        (`input` field instead of `messages`) for GPT-5.x and reasoning
        models. Auth header is swapped for the Copilot OAuth token; body
        and response are forwarded as-is.
        """
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
        req_size: int = len(raw_body)

        model: str = ""
        is_stream: bool = False
        n_input: int = 0
        try:
            peek: JsonDict = json.loads(raw_body) if raw_body else {}
            if isinstance(peek, dict):
                model = str(peek.get("model", ""))
                is_stream = bool(peek.get("stream", False))
                inp = peek.get("input")
                if isinstance(inp, list):
                    n_input = len(inp)
                elif isinstance(inp, str):
                    n_input = 1
        except json.JSONDecodeError:
            pass

        logger.info(
            ">>> %s model=%s stream=%s input=%d (%s) [responses-passthrough]",
            self.path, model or "?", is_stream, n_input, _fmt_size(req_size),
        )
        self._forward_responses_passthrough(t0, raw_body, model, is_stream)

    def _forward_responses_passthrough(
        self, t0: float, raw_body: bytes, model: str, is_stream: bool,
        translate_to_chat: bool = False,
    ) -> None:
        """Forward a Responses-API request body to Copilot's /responses endpoint.

        When `translate_to_chat` is True, the upstream SSE stream is translated
        into Chat-Completions SSE chunks on the fly (used when Cursor sends a
        Responses-API body to /v1/chat/completions and expects a chat-completion
        response in return).
        """
        if copilot_token_manager is None:
            self.send_error(503, "/responses passthrough requires --copilot-auth")
            logger.error("/responses hit but copilot_token_manager not initialized")
            return
        req_size: int = len(raw_body)

        def _build_headers(token: str) -> dict[str, str]:
            return {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Content-Length": str(req_size),
                "Editor-Version": "vscode/1.96.0",
                "Editor-Plugin-Version": "copilot/1.200.0",
                "User-Agent": "GithubCopilot/1.200.0",
                "Copilot-Integration-Id": "vscode-chat",
                "Accept": "text/event-stream" if is_stream else "application/json",
            }

        current_token: str = copilot_token_manager.get_token()
        conn = http.client.HTTPSConnection(COPILOT_HOST, context=SSL_CTX)
        try:
            conn.request(
                "POST", "/responses",
                body=raw_body, headers=_build_headers(current_token),
            )
            resp: http.client.HTTPResponse = conn.getresponse()

            if resp.status in (401, 403):
                logger.warning("Got %d on /responses, refreshing token", resp.status)
                resp.read()
                conn.close()
                new_token: str = copilot_token_manager.invalidate()
                conn = http.client.HTTPSConnection(COPILOT_HOST, context=SSL_CTX)
                conn.request(
                    "POST", "/responses",
                    body=raw_body, headers=_build_headers(new_token),
                )
                resp = conn.getresponse()

            content_type: str = resp.getheader("Content-Type") or "application/json"
            is_event_stream: bool = "event-stream" in content_type
            do_translate: bool = (
                translate_to_chat and is_event_stream and resp.status == 200
            )

            self.send_response(resp.status)
            if do_translate:
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                resp_size: int = _stream_responses_to_chat_completions(
                    resp, self.wfile, model,
                )
            elif is_event_stream:
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                resp_size = 0
                while True:
                    chunk: bytes = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    resp_size += len(chunk)
                    if b"\n" in chunk:
                        self.wfile.flush()
            else:
                self.send_header("Content-Type", content_type)
                data: bytes = resp.read()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                resp_size = len(data)

            elapsed_ms: float = (time.monotonic() - t0) * 1000
            mode: str = "responses->chat" if do_translate else "responses-passthrough"
            logger.info(
                "<<< %dms HTTP %d [%s] (%s -> %s)",
                elapsed_ms, resp.status, mode,
                _fmt_size(req_size), _fmt_size(resp_size),
            )
            log_jsonl({
                "ts": time.time(),
                "path": self.path,
                "request": {
                    "model": model,
                    "stream": is_stream,
                    "passthrough": "responses",
                    "translated": do_translate,
                },
                "response": {
                    "status": resp.status,
                    "stream": is_stream,
                    "req_bytes": req_size,
                    "resp_bytes": resp_size,
                },
                "elapsed_ms": round(elapsed_ms),
            })
        finally:
            conn.close()

    def _handle_tavily_path(
        self, t0: float, raw_body: bytes, parsed_body: JsonDict,
    ) -> None:
        """Serve a CC WebSearch executor request from Tavily.

        Replaces the upstream call entirely: we extract the search query,
        call Tavily, and synthesize a streaming Anthropic-format response
        containing one `text` block of Markdown-formatted search results
        with extracted page content. CC folds this back into the main
        Copilot turn as a `tool_result`, and the model can answer without
        any follow-up WebFetch.
        """
        assert _tavily_api_key is not None
        req_size: int = len(raw_body)

        query: str = _extract_search_query(parsed_body)
        if not query:
            self.send_error(400, "Could not extract search query from request")
            logger.warning("Refused [tavily]: empty search query")
            return

        try:
            tavily_resp: JsonDict = _tavily_search(query)
        except Exception as e:
            logger.exception("Tavily call failed")
            err_msg = f"Tavily error: {e}"
            self.send_error(502, err_msg)
            return

        cost: float = TAVILY_PRICING.get(_tavily_search_depth, 0.01)
        today_total: float = _record_tavily_spend(cost)

        is_stream: bool = bool(parsed_body.get("stream", False))
        model: str = parsed_body.get("model") or "claude-tavily"
        n_results: int = len(tavily_resp.get("results") or [])

        msg_id: str = f"msg_tavily_{int(time.time() * 1000)}"
        tool_use_id: str = f"srvtoolu_{int(time.time() * 1000)}"
        search_results: list[JsonDict] = _tavily_to_search_results(tavily_resp)
        summary_text: str = _format_tavily_results(query, tavily_resp)
        usage: JsonDict = {
            "input_tokens": 0,
            "output_tokens": max(1, len(summary_text) // 4),
            "server_tool_use": {"web_search_requests": 1 if n_results else 0},
        }

        if is_stream:
            sse_body: bytes = self._tavily_to_anthropic_sse(
                msg_id, tool_use_id, query, search_results, summary_text,
                model, usage,
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(sse_body)
            self.wfile.flush()
            resp_size: int = len(sse_body)
        else:
            non_stream_body: JsonDict = {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": tool_use_id,
                        "name": "web_search",
                        "input": {"query": query},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": tool_use_id,
                        "content": search_results,
                    },
                    {"type": "text", "text": summary_text},
                ],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": usage,
            }
            ns_bytes: bytes = json.dumps(non_stream_body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(ns_bytes)))
            self.end_headers()
            self.wfile.write(ns_bytes)
            resp_size = len(ns_bytes)

        elapsed_ms: float = (time.monotonic() - t0) * 1000
        logger.info(
            "<<< %dms OK [tavily] (%s -> %s) results=%d",
            elapsed_ms, _fmt_size(req_size), _fmt_size(resp_size), n_results,
        )
        logger.info(
            "    cost: $%.4f  today: $%.2f", cost, today_total,
        )

        log_jsonl({
            "ts": time.time(),
            "path": self.path,
            "request": self._request_log_entry(parsed_body),
            "response": {
                "status": 200,
                "stream": is_stream,
                "tavily": True,
                "search_depth": _tavily_search_depth,
                "results": n_results,
                "query": query[:200],
                "req_bytes": req_size,
                "resp_bytes": resp_size,
                "cost_usd": round(cost, 4),
                "spend_today_usd": round(today_total, 4),
            },
            "elapsed_ms": round(elapsed_ms),
        })

    @staticmethod
    def _tavily_to_anthropic_sse(
        msg_id: str,
        tool_use_id: str,
        query: str,
        search_results: list[JsonDict],
        summary_text: str,
        model: str,
        usage: JsonDict,
    ) -> str:
        """Build a synthetic Anthropic-format SSE stream that mimics a real
        `web_search` server-tool turn. Emits three content blocks:

          0. `server_tool_use`           — the tool invocation
          1. `web_search_tool_result`    — Anthropic-shaped result list
          2. `text`                      — Markdown summary (fallback for
                                            consumers that ignore the result
                                            block)

        This shape lets Claude Code's executor count the search and feed the
        structured results back to the main agent.
        """
        parts: list[str] = []
        parts.append(_format_sse("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }))

        parts.append(_format_sse("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": tool_use_id,
                "name": "web_search",
                "input": {},
            },
        }))
        parts.append(_format_sse("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps({"query": query}),
            },
        }))
        parts.append(_format_sse("content_block_stop", {
            "type": "content_block_stop", "index": 0,
        }))

        parts.append(_format_sse("content_block_start", {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": tool_use_id,
                "content": search_results,
            },
        }))
        parts.append(_format_sse("content_block_stop", {
            "type": "content_block_stop", "index": 1,
        }))

        text: str = summary_text or "_(no results)_"
        parts.append(_format_sse("content_block_start", {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "text", "text": ""},
        }))
        chunk_size: int = 4096
        for i in range(0, len(text), chunk_size):
            parts.append(_format_sse("content_block_delta", {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "text_delta", "text": text[i:i + chunk_size]},
            }))
        parts.append(_format_sse("content_block_stop", {
            "type": "content_block_stop", "index": 2,
        }))

        parts.append(_format_sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": usage,
        }))
        parts.append(_format_sse("message_stop", {"type": "message_stop"}))
        return "".join(parts)

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
                if not b.strip().startswith(_STRIP_BETA_PREFIXES)
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
        if _upstream_base_url:
            self._handle_local_openai_path(t0, parsed_body, model)
            return
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

    def _handle_local_openai_path(
        self, t0: float, parsed_body: JsonDict, model: str,
    ) -> None:
        """Translate Anthropic -> OpenAI and forward to a local OpenAI-compatible endpoint."""
        assert _upstream_base_url is not None
        url = urllib.parse.urlparse(_upstream_base_url)
        if not url.hostname:
            self.send_error(500, "Invalid --upstream-base-url")
            return
        is_https: bool = url.scheme == "https"
        port: int = url.port or (443 if is_https else 80)
        base_path: str = url.path.rstrip("/")
        full_path: str = f"{base_path}/chat/completions"

        is_stream: bool = parsed_body.get("stream", False)
        oai_body: JsonDict = anthropic_to_openai(parsed_body, model)
        oai_bytes: bytes = json.dumps(oai_body).encode()
        req_size: int = len(oai_bytes)

        upstream_headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Content-Length": str(len(oai_bytes)),
            "Accept": "text/event-stream" if is_stream else "application/json",
        }
        if _upstream_api_key:
            upstream_headers["Authorization"] = f"Bearer {_upstream_api_key}"

        if is_https:
            conn = http.client.HTTPSConnection(url.hostname, port=port, context=SSL_CTX)
        else:
            conn = http.client.HTTPConnection(url.hostname, port=port)

        try:
            conn.request("POST", full_path, body=oai_bytes, headers=upstream_headers)
            resp: http.client.HTTPResponse = conn.getresponse()

            if resp.status != 200:
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
                    "upstream": _upstream_base_url,
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
                    "upstream": _upstream_base_url,
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
            stream_usage: JsonDict = self._extract_stream_usage(decoded_stream)
            stream_resp_log: JsonDict = {
                "status": resp.status,
                "stream": True,
                "usage": stream_usage,
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
            return
        if self.path in ("/models", "/v1/models"):
            self._handle_models_passthrough()
            return
        self.send_error(404)

    def _handle_models_passthrough(self) -> None:
        """Forward GET /models (and /v1/models) to Copilot for Cursor probes."""
        if not self._check_api_key():
            return
        if copilot_token_manager is None:
            self.send_error(503, "/models passthrough requires --copilot-auth")
            return
        token: str = copilot_token_manager.get_token()
        headers: dict[str, str] = {
            "Authorization": f"Bearer {token}",
            "Editor-Version": "vscode/1.96.0",
            "Editor-Plugin-Version": "copilot/1.200.0",
            "User-Agent": "GithubCopilot/1.200.0",
            "Copilot-Integration-Id": "vscode-chat",
        }
        conn = http.client.HTTPSConnection(COPILOT_HOST, context=SSL_CTX)
        try:
            conn.request("GET", "/models", headers=headers)
            resp: http.client.HTTPResponse = conn.getresponse()
            if resp.status in (401, 403):
                resp.read()
                conn.close()
                new_token: str = copilot_token_manager.invalidate()
                headers["Authorization"] = f"Bearer {new_token}"
                conn = http.client.HTTPSConnection(COPILOT_HOST, context=SSL_CTX)
                conn.request("GET", "/models", headers=headers)
                resp = conn.getresponse()
            data: bytes = resp.read()
            self.send_response(resp.status)
            self.send_header(
                "Content-Type", resp.getheader("Content-Type") or "application/json",
            )
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            logger.info("<<< GET /models HTTP %d (%s)", resp.status, _fmt_size(len(data)))
        finally:
            conn.close()

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
    _upstream_base_url = args.upstream_base_url
    _upstream_api_key = args.upstream_api_key
    _no_opus = args.no_opus
    _no_opus_target = args.no_opus_target
    _tavily_api_key = args.tavily_api_key
    _tavily_search_depth = args.tavily_search_depth
    _tavily_max_results = args.tavily_max_results

    setup_logging(_log_dir, args.log_level)

    # In local-upstream mode we bypass both Copilot paths entirely.
    token_manager: TokenManager | None = None
    copilot_token_manager: CopilotTokenManager | None = None
    if _upstream_base_url:
        if not _upstream_model:
            logger.warning(
                "--upstream-base-url set without --upstream-model; "
                "request model names will be passed through as-is"
            )
    else:
        token_manager = TokenManager()
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
    if _upstream_base_url:
        logger.info("  Upstream: %s (local OpenAI-compatible)", _upstream_base_url)
        if _upstream_api_key:
            logger.info("  Upstream API key: configured")
    else:
        logger.info("  Upstream: %s", COPILOT_HOST)
    if _upstream_model:
        if _upstream_base_url:
            logger.info("  Upstream model: %s (local)", _upstream_model)
        elif _is_claude_model(_upstream_model):
            logger.info("  Upstream model: %s (native Anthropic pass-through)", _upstream_model)
        else:
            logger.info("  Upstream model: %s (EXPERIMENTAL: OpenAI translation)", _upstream_model)
    if _api_key:
        logger.info("  API key: required (x-api-key or Authorization: Bearer)")
    else:
        logger.info("  API key: not configured (open access)")
    if copilot_token_manager is not None:
        logger.info("  OpenAI /chat/completions passthrough: ENABLED (Copilot upstream)")
    if _no_opus:
        logger.info("  Opus downgrade: ENABLED (claude-opus-* -> %s)", _no_opus_target)
    if _tavily_api_key:
        logger.info(
            "  Tavily search: ENABLED -> %s (depth=%s, max_results=%d)",
            TAVILY_HOST, _tavily_search_depth, _tavily_max_results,
        )
    if token_manager is not None:
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
