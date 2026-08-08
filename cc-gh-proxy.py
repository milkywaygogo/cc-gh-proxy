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
import queue
import re
import ssl
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any

try:
    import duckdb  # optional; telemetry disables itself if unavailable
except ImportError:  # pragma: no cover - exercised only when dep missing
    duckdb = None  # type: ignore[assignment]

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
_opus_model: str = "claude-opus-5"  # Every claude-opus-* request resolves here
_no_opus: bool = False  # Map any claude-opus-* request to a sonnet model
_no_opus_target: str = "claude-sonnet-5"  # Target sonnet model when --no-opus is set

# Telemetry (DuckDB) — set in main(). When _telemetry is None, telemetry is off.
_telemetry: TelemetryWriter | None = None
# Optional override for the Claude Code projects dir (used to resolve session
# names). None -> auto-detect (CLAUDE_CONFIG_DIR or ~/.claude). Set in main().
_claude_projects_dir: Path | None = None

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
             "Claude models (claude-opus-5, claude-sonnet-4.6, claude-haiku-4.5) use "
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
        "--opus-model",
        default=os.environ.get("PROXY_OPUS_MODEL"),
        help="Copilot model that every claude-opus-* request resolves to, "
             "regardless of the version Claude Code asks for "
             "(env: PROXY_OPUS_MODEL). When unset, auto-discovered at startup "
             "from GET /models (fallback: claude-opus-5). Ignored when "
             "--no-opus is set",
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
        default=os.environ.get("PROXY_NO_OPUS_TARGET"),
        help="target Copilot model when --no-opus rewrites an Opus request "
             "(env: PROXY_NO_OPUS_TARGET). When unset, auto-discovered at "
             "startup as the newest Sonnet from GET /models "
             "(fallback: claude-sonnet-5)",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=os.environ.get("PROXY_VERBOSE", "").lower() in ("1", "true", "yes"),
        help="mirror the full request-by-request log to the console. By "
             "default only the startup banner and warnings/errors are printed "
             "(env: PROXY_VERBOSE, default: off)",
    )
    p.add_argument(
        "--duckdb-path",
        default=os.environ.get("PROXY_DUCKDB_PATH"),
        help="path to a DuckDB file for per-request telemetry. Defaults to "
             "<log-dir>/usage.duckdb. Requires the `duckdb` pip package; if it "
             "is not installed, telemetry is disabled (env: PROXY_DUCKDB_PATH)",
    )
    p.add_argument(
        "--no-duckdb",
        action="store_true",
        default=os.environ.get("PROXY_NO_DUCKDB", "").lower() in ("1", "true", "yes"),
        help="disable DuckDB telemetry entirely (env: PROXY_NO_DUCKDB)",
    )
    p.add_argument(
        "--duckdb-flush-interval",
        type=float,
        default=float(os.environ.get("PROXY_DUCKDB_FLUSH_INTERVAL", "2.0")),
        help="seconds between DuckDB flushes. The writer opens/appends/closes "
             "the file each flush so duckdb.exe can query it between flushes "
             "(env: PROXY_DUCKDB_FLUSH_INTERVAL, default: 2.0)",
    )
    p.add_argument(
        "--claude-projects-dir",
        default=os.environ.get("PROXY_CLAUDE_PROJECTS_DIR"),
        help="path to Claude Code's projects directory, used to resolve the "
             "session_name telemetry column. Defaults to "
             "CLAUDE_CONFIG_DIR/projects or ~/.claude/projects "
             "(env: PROXY_CLAUDE_PROJECTS_DIR)",
    )
    return p.parse_args()


class _ConsoleFilter(logging.Filter):
    """Decide which records reach the console.

    Default: only warnings/errors and explicitly tagged banner records
    (the startup summary). The per-request INFO stream stays out of the shell
    (it is still written to proxy.log). With --verbose, everything passes.
    """

    def __init__(self, verbose: bool) -> None:
        super().__init__()
        self._verbose = verbose

    def filter(self, record: logging.LogRecord) -> bool:
        if self._verbose:
            return True
        return record.levelno >= logging.WARNING or getattr(record, "banner", False)


class _LowerLevelFormatter(logging.Formatter):
    """Render the level name lowercased, e.g. [info] / [warning] / [error]."""

    def format(self, record: logging.LogRecord) -> str:
        original = record.levelname
        record.levelname = original.lower()
        try:
            return super().format(record)
        finally:
            record.levelname = original


def banner(msg: str, *args: object) -> None:
    """Log an INFO line that is always shown on the console (startup summary)."""
    logger.info(msg, *args, extra={"banner": True})


def setup_logging(log_dir: Path, level: str, verbose: bool = False) -> None:
    """Configure console and file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_dir.chmod(0o700)
    logger.setLevel(getattr(logging, level, logging.INFO))

    # Console shows the startup banner + warnings/errors so the shell stays
    # clean; the full request-by-request log (INFO) goes to proxy.log only.
    # --verbose mirrors the entire INFO stream to the console too.
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(_LowerLevelFormatter("[%(levelname)s] %(message)s"))
    console.addFilter(_ConsoleFilter(verbose))
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
        logger.info("Token acquired successfully", extra={"banner": True})

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
# Opus is absent here on purpose: every Opus request is pinned by _resolve_opus.
_MODEL_STATIC_MAP: dict[str, str] = {
    "claude-sonnet-4-6": "claude-sonnet-4.6",
    "claude-haiku-4-5": "claude-haiku-4.5",
}
_FALLBACK_OPUS_MODEL: str = "claude-opus-5"
_FALLBACK_SONNET_MODEL: str = "claude-sonnet-5"
_FALLBACK_HAIKU_MODEL: str = "claude-haiku-4.5"
# Fallback defaults used before / until startup discovery refreshes them.
# Bare ``claude-sonnet`` / ``claude-haiku`` always resolve to the series latest.
_MODEL_FAMILY_MAP: dict[str, str] = {
    "claude-sonnet": _FALLBACK_SONNET_MODEL,
    "claude-sonnet-4": "claude-sonnet-4.6",  # latest within major 4, not absolute newest
    "claude-haiku": _FALLBACK_HAIKU_MODEL,
    "claude-haiku-4": "claude-haiku-4.5",
}
# Populated by startup discovery (may be empty if fetch failed / skipped).
_latest_claude_models: dict[str, str] = {}

# Shared editor headers for Copilot /models and chat/completions auth.
_COPILOT_EDITOR_HEADERS: dict[str, str] = {
    "Editor-Version": "vscode/1.96.0",
    "Editor-Plugin-Version": "copilot/1.200.0",
    "User-Agent": "GithubCopilot/1.200.0",
    "Copilot-Integration-Id": "vscode-chat",
}

# claude-{tier}-{major}[.{minor}][-suffix]  e.g. claude-opus-5, claude-sonnet-4.6,
# claude-opus-4.8-fast
_CLAUDE_MODEL_ID_RE: re.Pattern[str] = re.compile(
    r"^claude-(?P<tier>opus|sonnet|haiku)-"
    r"(?P<major>\d+)(?:\.(?P<minor>\d+))?"
    r"(?:-(?P<suffix>.+))?$"
)


def _claude_model_sort_key(model_id: str) -> tuple[int, int, int] | None:
    """Sort key for Copilot Claude IDs: higher major/minor wins; bare > -fast."""
    m = _CLAUDE_MODEL_ID_RE.match(model_id)
    if not m:
        return None
    major: int = int(m.group("major"))
    minor: int = int(m.group("minor") or 0)
    bare: int = 1 if m.group("suffix") is None else 0
    return (major, minor, bare)


def discover_latest_claude_models(model_ids: list[str]) -> dict[str, str]:
    """Return newest model id per tier (opus/sonnet/haiku) from a catalog."""
    best: dict[str, tuple[tuple[int, int, int], str]] = {}
    for mid in model_ids:
        m = _CLAUDE_MODEL_ID_RE.match(mid)
        if not m:
            continue
        key = _claude_model_sort_key(mid)
        if key is None:
            continue
        tier: str = m.group("tier")
        prev = best.get(tier)
        if prev is None or key > prev[0]:
            best[tier] = (key, mid)
    return {tier: mid for tier, (_, mid) in best.items()}


def build_family_map(model_ids: list[str]) -> dict[str, str]:
    """Map ``claude-{sonnet|haiku}-{major}`` → newest model of that major."""
    best_major: dict[tuple[str, int], tuple[tuple[int, int, int], str]] = {}
    for mid in model_ids:
        m = _CLAUDE_MODEL_ID_RE.match(mid)
        if not m:
            continue
        key = _claude_model_sort_key(mid)
        if key is None:
            continue
        tier: str = m.group("tier")
        if tier == "opus":
            continue  # opus always goes through _resolve_opus
        major: int = int(m.group("major"))
        slot: tuple[str, int] = (tier, major)
        prev = best_major.get(slot)
        if prev is None or key > prev[0]:
            best_major[slot] = (key, mid)
    return {
        f"claude-{tier}-{major}": mid
        for (tier, major), (_, mid) in best_major.items()
    }


def fetch_copilot_model_ids(token: str, *, timeout: float = 15.0) -> list[str]:
    """GET api.githubcopilot.com/models and return model id strings."""
    headers: dict[str, str] = {
        **_COPILOT_EDITOR_HEADERS,
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    conn = http.client.HTTPSConnection(COPILOT_HOST, context=SSL_CTX, timeout=timeout)
    try:
        conn.request("GET", "/models", headers=headers)
        resp: http.client.HTTPResponse = conn.getresponse()
        raw: bytes = resp.read()
        if resp.status != 200:
            raise RuntimeError(
                f"GET /models HTTP {resp.status}: {raw[:200]!r}"
            )
        payload: Any = json.loads(raw)
    finally:
        conn.close()

    items: Any = payload.get("data", []) if isinstance(payload, dict) else payload
    ids: list[str] = []
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict) and it.get("id"):
                ids.append(str(it["id"]))
            elif isinstance(it, str):
                ids.append(it)
    return ids


def apply_discovered_claude_defaults(
    model_ids: list[str],
    *,
    update_opus: bool,
    update_no_opus_target: bool,
) -> dict[str, str]:
    """Apply catalog discovery to opus pin, no-opus target, and family map."""
    global _opus_model, _no_opus_target, _latest_claude_models

    latest: dict[str, str] = discover_latest_claude_models(model_ids)
    _latest_claude_models = dict(latest)
    family: dict[str, str] = build_family_map(model_ids)
    # Bare family aliases → absolute newest in that series.
    if "sonnet" in latest:
        family["claude-sonnet"] = latest["sonnet"]
    if "haiku" in latest:
        family["claude-haiku"] = latest["haiku"]
    if family:
        _MODEL_FAMILY_MAP.clear()
        _MODEL_FAMILY_MAP.update(family)

    if update_opus and "opus" in latest:
        _opus_model = latest["opus"]
    if update_no_opus_target and "sonnet" in latest:
        _no_opus_target = latest["sonnet"]
    return latest


def _resolve_opus(name: str) -> str:
    """Resolve a claude-opus-* request to the Opus model actually sent upstream.

    --no-opus wins when set. Version is NOT preserved in either direction:
    Copilot does not ship a sonnet for every opus version (e.g. opus 4.7 has no
    matching sonnet 4.7), and pinning Opus keeps Claude Code's dated model IDs
    from silently selecting an older Opus than the one configured.
    """
    if _no_opus:
        logger.info("Downgrading %s -> %s (--no-opus)", name, _no_opus_target)
        return _no_opus_target
    if name != _opus_model:
        logger.info("Pinning %s -> %s (--opus-model)", name, _opus_model)
    return _opus_model


def map_model_name(model: str) -> str:
    """Map Anthropic model IDs to Copilot model names.

    Claude Code may send:
      claude-opus-4-6, claude-opus-4-6[1m], claude-opus-4-6-20260312, etc.
    Copilot expects:
      claude-opus-5, claude-sonnet-4.6, claude-haiku-4.5
    """
    # Strip bracket suffixes: claude-opus-4-6[1m] -> claude-opus-4-6
    model = re.sub(r"\[[^\]]*\]$", "", model)
    # Strip date suffixes: claude-opus-4-6-20260312 -> claude-opus-4-6
    stripped: str = re.sub(r"-\d{8}$", "", model)

    if stripped == "claude-opus" or stripped.startswith("claude-opus-"):
        return _resolve_opus(stripped)

    if stripped in _MODEL_STATIC_MAP:
        return _MODEL_STATIC_MAP[stripped]

    # Pattern: claude-{tier}-{major}-{minor} -> claude-{tier}-{major}.{minor}
    m = re.match(r"^(claude-(?:sonnet|haiku)-\d+)-(\d+)$", stripped)
    if m:
        return f"{m.group(1)}.{m.group(2)}"

    # Base family: claude-sonnet / claude-sonnet-4 / claude-haiku → discovered
    # newest (bare tier → absolute latest; major-only → newest of that major).
    if stripped in _MODEL_FAMILY_MAP:
        mapped = _MODEL_FAMILY_MAP[stripped]
        if mapped != stripped:
            logger.info("Pinning %s -> %s (latest family)", stripped, mapped)
        return mapped

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


def rewrite_body(raw_body: bytes) -> tuple[bytes, JsonDict, str]:
    """Rewrite model names and strip unsupported fields.

    Returns (rewritten_body_bytes, parsed_body_dict, original_model_name).
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
        return json.dumps(body).encode(), body, original
    return raw_body, body, original


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
# Telemetry (DuckDB)
# ---------------------------------------------------------------------------

# Editable pricing table for the *counterfactual* cost columns. These are NOT
# what Copilot bills — they are Anthropic public list prices (USD per 1M
# tokens) applied locally so you can compare models. `premium` is the GitHub
# Copilot premium-request multiplier (closer to your real cost driver).
# Keyed by the upstream (dotted) model name. Add rows as new models ship.
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4.6":   {"in": 15.0, "out": 75.0, "cache_read": 1.50, "cache_write": 18.75, "premium": 10.0},
    "claude-opus-4.7":   {"in": 15.0, "out": 75.0, "cache_read": 1.50, "cache_write": 18.75, "premium": 10.0},
    "claude-opus-4.8":   {"in": 15.0, "out": 75.0, "cache_read": 1.50, "cache_write": 18.75, "premium": 10.0},
    "claude-opus-5":     {"in": 15.0, "out": 75.0, "cache_read": 1.50, "cache_write": 18.75, "premium": 10.0},
    "claude-sonnet-4.6": {"in": 3.0,  "out": 15.0, "cache_read": 0.30, "cache_write": 3.75,  "premium": 1.0},
    "claude-sonnet-4.5": {"in": 3.0,  "out": 15.0, "cache_read": 0.30, "cache_write": 3.75,  "premium": 1.0},
    "claude-sonnet-5":   {"in": 3.0,  "out": 15.0, "cache_read": 0.30, "cache_write": 3.75,  "premium": 1.0},
    "claude-haiku-4.5":  {"in": 1.0,  "out": 5.0,  "cache_read": 0.10, "cache_write": 1.25,  "premium": 0.33},
}

# Ordered column list — the single source of truth for both the DDL and the
# parameterized INSERT (keeps them from drifting apart).
TELEMETRY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("request_id", "TEXT PRIMARY KEY"),
    ("ts", "TIMESTAMP"),
    ("session_id", "TEXT"),
    ("session_name", "TEXT"),
    ("account_id", "TEXT"),
    ("response_message_id", "TEXT"),
    ("requested_model", "TEXT"),
    ("upstream_model", "TEXT"),
    ("downgraded", "BOOLEAN"),
    ("downgrade_to", "TEXT"),
    ("route", "TEXT"),
    ("upstream_host", "TEXT"),
    ("retry_count", "INTEGER"),
    ("beta_requested", "TEXT"),
    ("beta_stripped", "TEXT"),
    ("n_messages", "INTEGER"),
    ("system_bytes", "INTEGER"),
    ("tool_count", "INTEGER"),
    ("tool_names", "TEXT[]"),
    ("max_tokens", "INTEGER"),
    ("temperature", "DOUBLE"),
    ("top_p", "DOUBLE"),
    ("top_k", "INTEGER"),
    ("thinking_enabled", "BOOLEAN"),
    ("thinking_budget", "INTEGER"),
    ("stream", "BOOLEAN"),
    ("req_bytes", "INTEGER"),
    ("input_tokens", "INTEGER"),
    ("output_tokens", "INTEGER"),
    ("cache_read_tokens", "INTEGER"),
    ("cache_creation_tokens", "INTEGER"),
    ("cache_creation_5m", "INTEGER"),
    ("cache_creation_1h", "INTEGER"),
    ("total_tokens", "INTEGER"),
    ("cache_hit_ratio", "DOUBLE"),
    ("web_search_requests", "INTEGER"),
    ("status", "INTEGER"),
    ("stop_reason", "TEXT"),
    ("completed", "BOOLEAN"),
    ("elapsed_ms", "INTEGER"),
    ("ttft_ms", "INTEGER"),
    ("output_tokens_per_sec", "DOUBLE"),
    ("resp_bytes", "INTEGER"),
    ("response_tool_names", "TEXT[]"),
    ("error_type", "TEXT"),
    ("error_message", "TEXT"),
    ("est_anthropic_cost_usd", "DOUBLE"),
    ("copilot_premium_multiplier", "DOUBLE"),
    ("tavily_cost_usd", "DOUBLE"),
)

TELEMETRY_DDL: str = (
    "CREATE TABLE IF NOT EXISTS requests (\n  "
    + ",\n  ".join(f"{name} {decl}" for name, decl in TELEMETRY_COLUMNS)
    + "\n)"
)

_USER_ID_RE: re.Pattern[str] = re.compile(
    r"_account_(?P<account>.*?)_session_(?P<session>.+)$"
)


def parse_user_id(user_id: str | None) -> tuple[str | None, str | None]:
    """Split Claude Code's metadata.user_id into (session_id, account_id).

    Two formats are seen in the wild:
      1. A JSON object string (current Claude Code), e.g.
         {"device_id": "...", "account_uuid": "", "session_id": "<uuid>"}
      2. The legacy underscore string:
         user_<hash>_account_<account-uuid>_session_<session-uuid>

    For the JSON form, account_id prefers account_uuid but falls back to
    device_id (account_uuid is often empty under Copilot auth) so there is a
    stable per-install identifier to group by. Returns (None, None) when the
    field is missing or unparseable.
    """
    if not user_id or not isinstance(user_id, str):
        return None, None
    s = user_id.strip()
    if s.startswith("{"):
        try:
            obj = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            obj = None
        if isinstance(obj, dict):
            session = obj.get("session_id") or None
            account = obj.get("account_uuid") or obj.get("device_id") or None
            return session, account
    m = _USER_ID_RE.search(s)
    if not m:
        return None, None
    return m.group("session") or None, m.group("account") or None


# Session-name resolution -----------------------------------------------------
# Claude Code stores each conversation transcript at
#   <claude-config-dir>/projects/<encoded-cwd>/<session-id>.jsonl
# and *appends* a human-readable title line as the conversation grows. Two
# shapes exist: the auto-generated "ai-title"/"aiTitle" (the common case) and
# the explicit "custom-title"/"customTitle" set via /rename. We resolve
# session_id -> name by globbing for that file and reading the title. Results
# are cached with a TTL so renames are eventually picked up without re-scanning
# on every request.
_SESSION_NAME_TTL: float = 300.0  # seconds
_session_name_cache: dict[str, tuple[float, str | None]] = {}
_session_name_lock: threading.Lock = threading.Lock()


def claude_projects_dir() -> Path:
    """Return the Claude Code projects directory (cross-platform).

    Resolution order:
      1. --claude-projects-dir / PROXY_CLAUDE_PROJECTS_DIR (explicit override)
      2. CLAUDE_CONFIG_DIR/projects (Claude Code's own override env var)
      3. ~/.claude/projects (default; Path.home() handles Windows + POSIX)
    """
    if _claude_projects_dir is not None:
        return _claude_projects_dir
    cfg = os.environ.get("CLAUDE_CONFIG_DIR")
    if cfg:
        return Path(cfg) / "projects"
    return Path.home() / ".claude" / "projects"


def _read_session_title(path: Path) -> str | None:
    """Read the current human-readable title from a transcript, if present.

    Claude Code records the title as a standalone line it *appends* whenever
    the title is (re)generated — typically after several messages, so it is
    NOT in the header. Two line shapes exist:
      - {"type": "ai-title", "aiTitle": ...}         — auto-generated; the
        newer format and the common case for almost every session.
      - {"type": "custom-title", "customTitle": ...} — set explicitly via
        /rename; the legacy format and an explicit user override.

    We scan the whole file and keep the last line of each kind (titles are
    re-emitted as the conversation grows, so last-one-wins). An explicit
    custom-title takes precedence over the auto-generated ai-title; otherwise
    the ai-title is used. Returns None when neither is present (e.g. sessions
    too short to have been titled). (An earlier version read only
    `custom-title`, so every session that had only an auto ai-title — i.e. most
    of them — resolved to NULL.)
    """
    custom: str | None = None
    ai: str | None = None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "-title" not in line:  # cheap pre-filter before JSON parse
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                ttype = obj.get("type")
                if ttype == "custom-title":
                    val = obj.get("customTitle")
                    if isinstance(val, str) and val.strip():
                        custom = val.strip()  # keep scanning; last one wins
                elif ttype == "ai-title":
                    val = obj.get("aiTitle")
                    if isinstance(val, str) and val.strip():
                        ai = val.strip()
    except OSError:
        return None
    # Explicit user rename wins over the auto-generated title.
    return custom or ai


def lookup_session_name(session_id: str | None) -> str | None:
    """Resolve a Claude Code session UUID to its human-readable name.

    Returns None if no transcript or no title (ai-title/custom-title) is found.
    Cached with a TTL (negative results too, so unnamed sessions don't re-scan
    constantly). Safe to call from the writer thread; never raises.
    """
    if not session_id:
        return None
    now = time.monotonic()
    with _session_name_lock:
        cached = _session_name_cache.get(session_id)
        if cached is not None and now - cached[0] < _SESSION_NAME_TTL:
            return cached[1]

    name: str | None = None
    try:
        projects = claude_projects_dir()
        if projects.is_dir():
            # session_id is a UUID, so the filename is unique across projects.
            for p in projects.glob(f"**/{session_id}.jsonl"):
                name = _read_session_title(p)
                if name:
                    break
    except OSError:
        name = None

    with _session_name_lock:
        _session_name_cache[session_id] = (now, name)
    return name


def build_request_tel(
    parsed_body: JsonDict,
    *,
    request_id: str,
    route: str,
    requested_model: str,
    upstream_model: str,
    beta_requested: str | None,
) -> dict[str, Any]:
    """Build the request-time portion of a telemetry row.

    Response-time fields (tokens, status, timing, ...) are filled in later by
    the handler before the row is enqueued.
    """
    system = parsed_body.get("system")
    tools: list[JsonDict] = parsed_body.get("tools") or []
    tool_names = [t.get("name", "") for t in tools if isinstance(t, dict)]

    thinking = parsed_body.get("thinking")
    thinking_enabled = (
        isinstance(thinking, dict) and thinking.get("type") == "enabled"
    )
    thinking_budget = (
        thinking.get("budget_tokens") if isinstance(thinking, dict) else None
    )

    metadata = parsed_body.get("metadata")
    user_id = metadata.get("user_id") if isinstance(metadata, dict) else None
    session_id, account_id = parse_user_id(user_id)

    downgraded = bool(_no_opus and requested_model.startswith("claude-opus-"))

    return {
        "request_id": request_id,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "session_id": session_id,
        "session_name": None,  # resolved in the writer thread (off hot path)
        "account_id": account_id,
        "response_message_id": None,
        "requested_model": requested_model,
        "upstream_model": upstream_model,
        "downgraded": downgraded,
        "downgrade_to": _no_opus_target if downgraded else None,
        "route": route,
        "upstream_host": None,
        "retry_count": 0,
        "beta_requested": beta_requested,
        "beta_stripped": None,
        "n_messages": len(parsed_body.get("messages", [])),
        "system_bytes": _content_size(system) if system else 0,
        "tool_count": len(tools),
        "tool_names": tool_names,
        "max_tokens": parsed_body.get("max_tokens"),
        "temperature": parsed_body.get("temperature"),
        "top_p": parsed_body.get("top_p"),
        "top_k": parsed_body.get("top_k"),
        "thinking_enabled": thinking_enabled,
        "thinking_budget": thinking_budget,
        "stream": bool(parsed_body.get("stream", False)),
        "req_bytes": None,
        # response-time fields default to None/0 until filled in
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_creation_5m": 0,
        "cache_creation_1h": 0,
        "total_tokens": 0,
        "cache_hit_ratio": None,
        "web_search_requests": 0,
        "status": None,
        "stop_reason": None,
        "completed": False,
        "elapsed_ms": None,
        "ttft_ms": None,
        "output_tokens_per_sec": None,
        "resp_bytes": None,
        "response_tool_names": [],
        "error_type": None,
        "error_message": None,
        "est_anthropic_cost_usd": 0.0,
        "copilot_premium_multiplier": 0.0,
        "tavily_cost_usd": 0.0,
    }


def finalize_usage_tel(
    tel: dict[str, Any], usage: JsonDict, model: str,
) -> None:
    """Fold an Anthropic `usage` dict into a telemetry row (in place)."""
    inp = usage.get("input_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or 0
    cr = usage.get("cache_read_input_tokens", 0) or 0
    cw = usage.get("cache_creation_input_tokens", 0) or 0
    tel["input_tokens"] = inp
    tel["output_tokens"] = out
    tel["cache_read_tokens"] = cr
    tel["cache_creation_tokens"] = cw

    cache_creation = usage.get("cache_creation")
    if isinstance(cache_creation, dict):
        tel["cache_creation_5m"] = cache_creation.get("ephemeral_5m_input_tokens", 0) or 0
        tel["cache_creation_1h"] = cache_creation.get("ephemeral_1h_input_tokens", 0) or 0

    server_tool_use = usage.get("server_tool_use")
    if isinstance(server_tool_use, dict):
        tel["web_search_requests"] = server_tool_use.get("web_search_requests", 0) or 0

    tel["total_tokens"] = inp + out + cr + cw
    denom = inp + cr
    tel["cache_hit_ratio"] = round(cr / denom, 4) if denom else None

    cost, premium = _est_cost(model, usage)
    tel["est_anthropic_cost_usd"] = cost
    tel["copilot_premium_multiplier"] = premium


def _est_cost(model: str, usage: dict[str, Any]) -> tuple[float, float]:
    """Return (est_anthropic_cost_usd, copilot_premium_multiplier) for a usage dict.

    Cost is a counterfactual list-price estimate, not real Copilot billing.
    """
    p = PRICING.get(model)
    if not p:
        return 0.0, 0.0
    inp = usage.get("input_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or 0
    cr = usage.get("cache_read_input_tokens", 0) or 0
    cw = usage.get("cache_creation_input_tokens", 0) or 0
    cost = (
        inp * p["in"]
        + out * p["out"]
        + cr * p["cache_read"]
        + cw * p["cache_write"]
    ) / 1_000_000.0
    return round(cost, 6), p.get("premium", 0.0)


class TelemetryWriter:
    """Background DuckDB writer.

    A single thread owns all DB access via a queue. Request threads only
    enqueue dict rows (never blocks the response). The writer batches rows and
    flushes them by opening the DB, appending, and closing — so the file lock
    is released between flushes and `duckdb.exe` can query the DB live.
    """

    # Cap on rows retained across failed flushes (e.g. while another process
    # holds the DB lock). Beyond this we drop the oldest — they remain in
    # requests.jsonl, the source of truth, so nothing is truly lost.
    MAX_PENDING: int = 10_000

    def __init__(self, db_path: Path, flush_interval: float = 2.0,
                 max_batch: int = 100) -> None:
        self._db_path = db_path
        self._flush_interval = max(0.25, flush_interval)
        self._max_batch = max_batch
        self._q: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._col_names: list[str] = [name for name, _ in TELEMETRY_COLUMNS]
        self._lock_warned: bool = False
        self._thread = threading.Thread(
            target=self._run, name="telemetry-writer", daemon=True
        )

    def start(self) -> None:
        # Validate we can open the DB and create/migrate the table up-front; if
        # this fails, the caller disables telemetry rather than starting the thread.
        con = duckdb.connect(str(self._db_path))
        try:
            self._ensure_schema(con)
        finally:
            con.close()
        self._thread.start()

    def _ensure_schema(self, con: Any) -> None:
        """Create the table if missing, and add any columns absent from an
        older DB (so the schema can evolve without dropping existing data)."""
        con.execute(TELEMETRY_DDL)
        existing = {
            row[0] for row in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'requests'"
            ).fetchall()
        }
        for name, decl in TELEMETRY_COLUMNS:
            if name not in existing:
                col_type = decl.replace("PRIMARY KEY", "").strip()
                con.execute(f"ALTER TABLE requests ADD COLUMN {name} {col_type}")
                logger.info("Telemetry: added new column '%s' to requests table", name)

    def enqueue(self, row: dict[str, Any]) -> None:
        self._q.put(row)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal shutdown and wait for the final flush."""
        self._q.put(None)
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        # `pending` carries rows that could not be flushed yet (e.g. the DB is
        # locked by another process); they are retried on the next flush.
        pending: list[dict[str, Any]] = []
        stopping = False
        while not stopping:
            try:
                item = self._q.get(timeout=self._flush_interval)
                if item is None:
                    stopping = True
                else:
                    pending.append(item)
                    while len(pending) < self._max_batch:
                        try:
                            nxt = self._q.get_nowait()
                        except queue.Empty:
                            break
                        if nxt is None:
                            stopping = True
                            break
                        pending.append(nxt)
            except queue.Empty:
                pass  # interval elapsed — flush whatever is pending

            if pending and self._flush(pending):
                pending = []
            elif len(pending) > self.MAX_PENDING:
                drop = len(pending) - self.MAX_PENDING
                logger.warning(
                    "Telemetry: %d rows still unflushed (DB locked?); dropping "
                    "%d oldest. They remain in requests.jsonl.",
                    len(pending), drop,
                )
                pending = pending[drop:]

        # Final drain on shutdown — one best-effort attempt.
        if pending:
            if self._flush(pending):
                logger.info("Telemetry: flushed %d pending rows on shutdown", len(pending))
            else:
                logger.warning(
                    "Telemetry: %d rows could not be flushed on shutdown "
                    "(DB locked?); they remain in requests.jsonl", len(pending),
                )

    def _flush(self, batch: list[dict[str, Any]]) -> bool:
        """Append a batch to DuckDB. Returns True on success.

        On failure (e.g. the DB file is locked by another process) the rows are
        NOT consumed — the caller retains them and retries on the next flush.
        """
        try:
            con = duckdb.connect(str(self._db_path))
        except Exception as e:
            if not self._lock_warned:
                logger.warning(
                    "Telemetry: cannot open %s (%s). Rows are buffered and will "
                    "be written once the DB is free (e.g. close other clients "
                    "like DataGrip).", self._db_path, e,
                )
                self._lock_warned = True
            return False
        self._lock_warned = False
        try:
            self._ensure_schema(con)
            # Resolve session names off the request hot path, here in the
            # writer thread (cached, so this is cheap for repeated sessions).
            for r in batch:
                if r.get("session_id") and not r.get("session_name"):
                    r["session_name"] = lookup_session_name(r["session_id"])
            placeholders = ", ".join("?" for _ in self._col_names)
            sql = (
                f"INSERT OR IGNORE INTO requests ({', '.join(self._col_names)}) "
                f"VALUES ({placeholders})"
            )
            rows = [[r.get(name) for name in self._col_names] for r in batch]
            con.executemany(sql, rows)
            return True
        except Exception:
            # A genuine insert error (not a lock) — drop to avoid a poison batch
            # looping forever. Data is still in requests.jsonl.
            logger.exception("Telemetry: insert failed; dropping %d rows", len(batch))
            return True
        finally:
            con.close()


def record_telemetry(row: dict[str, Any]) -> None:
    """Enqueue a telemetry row if telemetry is enabled. Never raises."""
    if _telemetry is None:
        return
    try:
        _telemetry.enqueue(row)
    except Exception:
        logger.debug("Telemetry enqueue failed", exc_info=True)


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
        request_id: str = uuid.uuid4().hex
        beta_requested: str | None = self.headers.get("anthropic-beta")
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
            tavily_model: str = pre_parsed.get("model", "")
            tel = build_request_tel(
                pre_parsed,
                request_id=request_id,
                route="tavily",
                requested_model=tavily_model,
                upstream_model=tavily_model,
                beta_requested=beta_requested,
            )
            self._handle_tavily_path(t0, raw_body, pre_parsed, tel)
            return

        # Rewrite model name and strip unsupported fields
        body_to_send: bytes
        parsed_body: JsonDict
        original_model: str
        body_to_send, parsed_body, original_model = rewrite_body(raw_body)

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

        tel = build_request_tel(
            parsed_body,
            request_id=request_id,
            route=route,
            requested_model=original_model,
            upstream_model=effective_model if use_openai else parsed_body.get("model", ""),
            beta_requested=beta_requested,
        )
        tel["req_bytes"] = req_size
        tel["upstream_host"] = _upstream_base_url or COPILOT_HOST

        if use_openai:
            self._handle_openai_path(t0, parsed_body, effective_model, tel)
        else:
            self._handle_native_path(t0, body_to_send, parsed_body, tel)

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
        tel: dict[str, Any],
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
        tel["req_bytes"] = req_size
        tel["upstream_host"] = TAVILY_HOST

        query: str = _extract_search_query(parsed_body)
        if not query:
            self.send_error(400, "Could not extract search query from request")
            logger.warning("Refused [tavily]: empty search query")
            tel["status"] = 400
            tel["error_type"] = "empty_search_query"
            tel["elapsed_ms"] = round((time.monotonic() - t0) * 1000)
            record_telemetry(tel)
            return

        try:
            tavily_resp: JsonDict = _tavily_search(query)
        except Exception as e:
            logger.exception("Tavily call failed")
            err_msg = f"Tavily error: {e}"
            self.send_error(502, err_msg)
            tel["status"] = 502
            tel["error_type"] = "tavily_error"
            tel["error_message"] = str(e)[:500]
            tel["elapsed_ms"] = round((time.monotonic() - t0) * 1000)
            record_telemetry(tel)
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

        finalize_usage_tel(tel, usage, model)
        tel["response_message_id"] = msg_id
        tel["status"] = 200
        tel["stop_reason"] = "end_turn"
        tel["completed"] = True
        tel["resp_bytes"] = resp_size
        tel["elapsed_ms"] = round(elapsed_ms)
        tel["tavily_cost_usd"] = round(cost, 4)
        record_telemetry(tel)

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
        tel: dict[str, Any],
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
            betas = [b.strip() for b in raw_beta.split(",")]
            supported = [
                b for b in betas if not b.startswith(_STRIP_BETA_PREFIXES)
            ]
            stripped = [b for b in betas if b.startswith(_STRIP_BETA_PREFIXES)]
            if stripped:
                tel["beta_stripped"] = ", ".join(stripped)
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
                tel["retry_count"] = tel.get("retry_count", 0) + 1
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

            self._forward_and_log(resp, is_stream, t0, req_size, parsed_body, tel)
        finally:
            conn.close()

    def _handle_openai_path(
        self, t0: float, parsed_body: JsonDict, model: str,
        tel: dict[str, Any],
    ) -> None:
        """Translate to OpenAI format, send to /chat/completions, translate back."""
        if _upstream_base_url:
            self._handle_local_openai_path(t0, parsed_body, model, tel)
            return
        if copilot_token_manager is None:
            self.send_error(503, "Non-Claude models require --copilot-auth")
            logger.error("OpenAI path requested but copilot_token_manager not initialized")
            tel["status"] = 503
            tel["error_type"] = "no_copilot_auth"
            tel["elapsed_ms"] = round((time.monotonic() - t0) * 1000)
            record_telemetry(tel)
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
                tel["retry_count"] = tel.get("retry_count", 0) + 1
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
                tel["status"] = resp.status
                tel["error_type"] = f"http_{resp.status}"
                tel["error_message"] = resp_data.decode(errors="replace")[:500]
                tel["req_bytes"] = req_size
                tel["resp_bytes"] = len(resp_data)
                tel["elapsed_ms"] = round(elapsed_ms)
                record_telemetry(tel)
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
                self._record_openai_stream_tel(
                    tel, anthropic_stream, model, 200, req_size, resp_size, elapsed_ms,
                )
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
                self._record_openai_nonstream_tel(
                    tel, anthropic_resp, model, 200, req_size, resp_size, elapsed_ms,
                )
        finally:
            conn.close()

    def _record_openai_stream_tel(
        self, tel: dict[str, Any], anthropic_stream: str, model: str,
        status: int, req_size: int, resp_size: int, elapsed_ms: float,
    ) -> None:
        """Capture telemetry for a translated OpenAI streaming response.

        Note: the OpenAI usage shape has no cache fields, so cache_* stay 0.
        """
        finalize_usage_tel(tel, self._extract_stream_usage(anthropic_stream), model)
        tel["status"] = status
        tel["stop_reason"] = self._extract_stream_stop_reason(anthropic_stream)
        tel["completed"] = '"type": "message_stop"' in anthropic_stream
        tel["response_message_id"] = self._extract_stream_message_id(anthropic_stream)
        tel["response_tool_names"] = self._extract_stream_tool_names(anthropic_stream)
        tel["resp_bytes"] = resp_size
        tel["elapsed_ms"] = round(elapsed_ms)
        record_telemetry(tel)

    def _record_openai_nonstream_tel(
        self, tel: dict[str, Any], anthropic_resp: JsonDict, model: str,
        status: int, req_size: int, resp_size: int, elapsed_ms: float,
    ) -> None:
        """Capture telemetry for a translated OpenAI non-streaming response."""
        finalize_usage_tel(tel, anthropic_resp.get("usage", {}), model)
        tel["status"] = status
        tel["stop_reason"] = anthropic_resp.get("stop_reason")
        tel["completed"] = True
        tel["response_message_id"] = anthropic_resp.get("id")
        tel["response_tool_names"] = [
            b.get("name", "")
            for b in anthropic_resp.get("content", [])
            if isinstance(b, dict) and b.get("type") == "tool_use"
        ]
        tel["resp_bytes"] = resp_size
        tel["elapsed_ms"] = round(elapsed_ms)
        record_telemetry(tel)

    def _handle_local_openai_path(
        self, t0: float, parsed_body: JsonDict, model: str,
        tel: dict[str, Any],
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
                tel["status"] = resp.status
                tel["error_type"] = f"http_{resp.status}"
                tel["error_message"] = resp_data.decode(errors="replace")[:500]
                tel["req_bytes"] = req_size
                tel["resp_bytes"] = len(resp_data)
                tel["elapsed_ms"] = round(elapsed_ms)
                record_telemetry(tel)
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
                self._record_openai_stream_tel(
                    tel, anthropic_stream, model, 200, req_size, resp_size, elapsed_ms,
                )
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
                self._record_openai_nonstream_tel(
                    tel, anthropic_resp, model, 200, req_size, resp_size, elapsed_ms,
                )
        finally:
            conn.close()
    def _forward_and_log(
        self, resp: http.client.HTTPResponse, is_stream: bool,
        t0: float, req_size: int, parsed_body: JsonDict,
        tel: dict[str, Any],
    ) -> None:
        """Forward upstream response to client and log it."""
        elapsed_ms: float
        model: str = tel.get("upstream_model") or parsed_body.get("model", "")
        if is_stream:
            collected: bytearray = bytearray()
            ttft_ms: int | None = None
            while True:
                chunk: bytes = resp.read(4096)
                if not chunk:
                    break
                if ttft_ms is None and b"text_delta" in chunk:
                    ttft_ms = round((time.monotonic() - t0) * 1000)
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

            finalize_usage_tel(tel, stream_usage, model)
            tel["status"] = resp.status
            tel["stop_reason"] = self._extract_stream_stop_reason(decoded_stream)
            tel["completed"] = "event: message_stop" in decoded_stream or (
                '"type": "message_stop"' in decoded_stream
            )
            tel["resp_bytes"] = resp_size
            tel["elapsed_ms"] = round(elapsed_ms)
            tel["ttft_ms"] = ttft_ms
            tel["response_message_id"] = self._extract_stream_message_id(decoded_stream)
            tel["response_tool_names"] = self._extract_stream_tool_names(decoded_stream)
            gen_s = (elapsed_ms - (ttft_ms or 0)) / 1000.0
            if gen_s > 0 and tel["output_tokens"]:
                tel["output_tokens_per_sec"] = round(tel["output_tokens"] / gen_s, 2)
            if resp.status != 200:
                tel["error_type"] = f"http_{resp.status}"
            record_telemetry(tel)
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

            tel["status"] = resp.status
            tel["resp_bytes"] = resp_size
            tel["elapsed_ms"] = round(elapsed_ms)
            if resp_body and resp.status == 200:
                finalize_usage_tel(tel, resp_body.get("usage", {}), model)
                tel["stop_reason"] = resp_body.get("stop_reason")
                tel["completed"] = True
                tel["response_message_id"] = resp_body.get("id")
                tel["response_tool_names"] = [
                    b.get("name", "")
                    for b in resp_body.get("content", [])
                    if isinstance(b, dict) and b.get("type") == "tool_use"
                ]
            else:
                tel["error_type"] = f"http_{resp.status}"
                if resp_body:
                    tel["error_message"] = json.dumps(
                        resp_body.get("error", {}))[:500]
            record_telemetry(tel)

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

    @staticmethod
    def _iter_stream_events(raw: str) -> "list[JsonDict]":
        """Parse all JSON events out of an Anthropic SSE stream."""
        events: list[JsonDict] = []
        for line in raw.split("\n"):
            if not line.startswith("data: "):
                continue
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                continue
        return events

    @classmethod
    def _extract_stream_stop_reason(cls, raw: str) -> str | None:
        for event in cls._iter_stream_events(raw):
            if event.get("type") == "message_delta":
                stop = event.get("delta", {}).get("stop_reason")
                if stop:
                    return stop
        return None

    @classmethod
    def _extract_stream_message_id(cls, raw: str) -> str | None:
        for event in cls._iter_stream_events(raw):
            if event.get("type") == "message_start":
                return event.get("message", {}).get("id")
        return None

    @classmethod
    def _extract_stream_tool_names(cls, raw: str) -> list[str]:
        names: list[str] = []
        for event in cls._iter_stream_events(raw):
            if event.get("type") == "content_block_start":
                block = event.get("content_block", {})
                if block.get("type") == "tool_use":
                    names.append(block.get("name", ""))
        return names

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
            **_COPILOT_EDITOR_HEADERS,
            "Authorization": f"Bearer {token}",
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
    # None means "auto-discover from Copilot /models at startup".
    _opus_model_explicit: bool = args.opus_model is not None
    _no_opus_target_explicit: bool = args.no_opus_target is not None
    _opus_model = args.opus_model or _FALLBACK_OPUS_MODEL
    _no_opus = args.no_opus
    _no_opus_target = args.no_opus_target or _FALLBACK_SONNET_MODEL
    _tavily_api_key = args.tavily_api_key
    _tavily_search_depth = args.tavily_search_depth
    _tavily_max_results = args.tavily_max_results
    _claude_projects_dir = Path(args.claude_projects_dir) if args.claude_projects_dir else None

    setup_logging(_log_dir, args.log_level, verbose=args.verbose)

    # DuckDB telemetry — optional, disables itself gracefully if the duckdb
    # package is missing or the DB can't be opened.
    if not args.no_duckdb:
        if duckdb is None:
            logger.warning(
                "Telemetry: --duckdb requested but the `duckdb` package is not "
                "installed. Run `pip install duckdb` to enable. Disabling telemetry."
            )
        else:
            db_path = Path(args.duckdb_path) if args.duckdb_path else _log_dir / "usage.duckdb"
            try:
                writer = TelemetryWriter(db_path, flush_interval=args.duckdb_flush_interval)
                writer.start()
                _telemetry = writer
            except Exception:
                logger.exception("Telemetry: failed to initialize DuckDB at %s; disabling", db_path)
                _telemetry = None

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

        # Discover newest opus/sonnet/haiku from Copilot's catalog and use them
        # as defaults (unless the user pinned --opus-model / --no-opus-target).
        discover_token: str | None = None
        if copilot_token_manager is not None:
            try:
                discover_token = copilot_token_manager.get_token()
            except TokenError as e:
                logger.warning("Model discovery: Copilot token unavailable: %s", e)
        if discover_token is None and token_manager is not None:
            try:
                discover_token = token_manager.get_token()
            except TokenError as e:
                logger.warning("Model discovery: gh token unavailable: %s", e)
        if discover_token is not None:
            try:
                catalog: list[str] = fetch_copilot_model_ids(discover_token)
                latest: dict[str, str] = apply_discovered_claude_defaults(
                    catalog,
                    update_opus=not _opus_model_explicit,
                    update_no_opus_target=not _no_opus_target_explicit,
                )
                if latest:
                    logger.info(
                        "Model discovery: opus=%s sonnet=%s haiku=%s",
                        latest.get("opus", "?"),
                        latest.get("sonnet", "?"),
                        latest.get("haiku", "?"),
                        extra={"banner": True},
                    )
                else:
                    logger.warning(
                        "Model discovery: no claude opus/sonnet/haiku ids in catalog"
                    )
            except Exception as e:
                logger.warning(
                    "Model discovery failed (%s); using fallbacks "
                    "opus=%s sonnet=%s haiku=%s",
                    e, _opus_model, _no_opus_target, _FALLBACK_HAIKU_MODEL,
                )
        else:
            logger.warning(
                "Model discovery skipped (no auth token); using fallbacks "
                "opus=%s sonnet=%s",
                _opus_model, _no_opus_target,
            )

    banner("cc-gh-proxy starting on http://%s:%d", args.host, args.port)
    if _upstream_base_url:
        banner("  Upstream: %s (local OpenAI-compatible)", _upstream_base_url)
        if _upstream_api_key:
            banner("  Upstream API key: configured")
    else:
        banner("  Upstream: %s", COPILOT_HOST)
    if _upstream_model:
        if _upstream_base_url:
            banner("  Upstream model: %s (local)", _upstream_model)
        elif _is_claude_model(_upstream_model):
            banner("  Upstream model: %s (native Anthropic pass-through)", _upstream_model)
        else:
            banner("  Upstream model: %s (EXPERIMENTAL: OpenAI translation)", _upstream_model)
    if _api_key:
        banner("  API key: required (x-api-key or Authorization: Bearer)")
    else:
        banner("  API key: not configured (open access)")
    if copilot_token_manager is not None:
        banner("  OpenAI /chat/completions passthrough: ENABLED (Copilot upstream)")
    if _no_opus:
        banner("  Opus downgrade: ENABLED (claude-opus-* -> %s)", _no_opus_target)
    else:
        banner("  Opus model: claude-opus-* -> %s", _opus_model)
    banner(
        "  Sonnet default: %s",
        _latest_claude_models.get("sonnet", _FALLBACK_SONNET_MODEL),
    )
    banner(
        "  Haiku default: %s",
        _latest_claude_models.get("haiku", _FALLBACK_HAIKU_MODEL),
    )
    if _tavily_api_key:
        banner(
            "  Tavily search: ENABLED -> %s (depth=%s, max_results=%d)",
            TAVILY_HOST, _tavily_search_depth, _tavily_max_results,
        )
    if token_manager is not None:
        banner("  Token auto-refresh: every %ds", TokenManager.REFRESH_INTERVAL)
    if _telemetry is not None:
        banner(
            "  Telemetry: ENABLED -> %s (flush every %.1fs)",
            _telemetry._db_path, args.duckdb_flush_interval,
        )
        banner("  Session names: resolved from %s", claude_projects_dir())
    elif not args.no_duckdb and duckdb is not None:
        banner("  Telemetry: disabled (init failed)")
    else:
        banner("  Telemetry: disabled")
    banner("  Logs: %s", _log_dir)
    if _log_requests:
        banner("  Request logging: ENABLED (message content will be persisted)")

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
    finally:
        if _telemetry is not None:
            logger.info("Flushing telemetry...")
            _telemetry.stop()
