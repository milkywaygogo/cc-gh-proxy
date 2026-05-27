# cc-gh-proxy

Routes Claude Code requests through GitHub Copilot instead of Anthropic's API directly.

> **Status: Experimental** — works for the author, may eat your tokens.

```
Claude Code ──► localhost:4000 ──► api.githubcopilot.com
              (Anthropic Messages API, pass-through)
```

GitHub Copilot natively supports the Anthropic Messages API, so the proxy is a
thin pass-through that only:
1. Swaps the auth header (gh CLI OAuth token)
2. Maps model names (`claude-opus-4-6` -> `claude-opus-4.6`)
3. Strips unsupported `cache_control` fields

No format conversion. Real token counts. Native streaming.

## Prerequisites

- **GitHub CLI** (`gh`)
- **Python 3.10+** (no pip dependencies)
- **GitHub Copilot license** on your GitHub account

## One-time setup

### 1. Install GitHub CLI

```bash
# Debian/Ubuntu
sudo apt install gh

# macOS
brew install gh

# Or see https://cli.github.com/
```

### 2. Authenticate and add the `copilot` scope

```bash
# Log in (if not already authenticated)
gh auth login

# Add the copilot scope to your token
gh auth refresh --hostname github.com -s copilot

# Verify
gh auth status  # should show 'copilot' in Token scopes
gh auth token   # should print a token
```

### 3. Verify Copilot access

Not necessary, but you can test that your token works with the Copilot API:

```bash
GH_TOKEN=$(gh auth token)
curl -s -H "Authorization: Bearer $GH_TOKEN" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-opus-4.6","messages":[{"role":"user","content":"Say hello"}],"max_tokens":5}' \
  https://api.githubcopilot.com/v1/messages
```

You should get a JSON response with `"type": "message"`.

### 4. Configure your project

Create `.claude/settings.json` in the project you want to route through Copilot:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:4000"
  }
}
```

If you start the proxy with `--api-key`, also set `ANTHROPIC_API_KEY`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:4000",
    "ANTHROPIC_API_KEY": "my-secret-key"
  }
}
```

Alternatively, you can set these as regular environment variables instead of using
`settings.json`:

```bash
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=my-secret-key  # only if proxy uses --api-key
```

Only projects/sessions with these settings use the proxy. All others use Anthropic directly.

## Usage

### Start the proxy

```bash
./cc-gh-proxy.py
```

Or with options:

```bash
./cc-gh-proxy.py --port 4000 --host 127.0.0.1 --api-key my-secret-key
```

Or in the background:

```bash
./cc-gh-proxy.py &
```

### Start Claude Code

```bash
cd your-project  # must have .claude/settings.json with ANTHROPIC_BASE_URL
claude
```

### Verify it works

With the proxy running:

```bash
./test.sh
```

### Stop the proxy

```bash
# If running in foreground: Ctrl+C
# If running in background:
kill $(lsof -ti:4000)
```

## Supported models

| Claude Code sends     | Copilot receives      |
|-----------------------|-----------------------|
| `claude-opus-4-6`     | `claude-opus-4.6`     |
| `claude-sonnet-4-6`   | `claude-sonnet-4.6`   |
| `claude-haiku-4-5`    | `claude-haiku-4.5`    |

Date-stamped variants (e.g. `claude-opus-4-6-20260312`) and base family names
(e.g. `claude-opus-4`) are also mapped automatically. Newer minor versions
(e.g. `claude-opus-4-7` -> `claude-opus-4.7`) are forwarded through the same
pattern, but Copilot only accepts versions it actually ships - check with the
Copilot status page if you get HTTP 400 `model_not_supported`.

## Routing modes

The proxy supports three upstream modes, picked at startup:

1. **Native Anthropic pass-through (default)** - requests go to
   `api.githubcopilot.com/v1/messages` unchanged. Works for Claude Opus,
   Sonnet, and Haiku. Auth: `gh auth token`.
2. **Copilot OpenAI-translated** - `--upstream-model gpt-5-mini` (or
   `gpt-4.1`, `gemini-2.5-pro`, etc.) routes to
   `api.githubcopilot.com/chat/completions` with Anthropic <-> OpenAI
   translation. Requires `--copilot-auth` (one-time device flow).
   Useful when your Claude premium-request quota is exhausted - `gpt-5-mini`
   and `gpt-4.1` cost 0x premium requests on Pro plans.
3. **Local / third-party OpenAI-compatible** - `--upstream-base-url`
   bypasses Copilot entirely and routes to any OpenAI-compatible endpoint
   (Ollama, vLLM, etc.) with the same translation. No GitHub auth needed.

### Avoiding Opus quota burn (`--no-opus`)

Opus consumes the highest premium-request multiplier. With `--no-opus`,
any incoming `claude-opus-*` request is rewritten to `--no-opus-target`
(default `claude-sonnet-4.6`) before forwarding. Useful when you want to
keep using Claude Code with its default model selection without burning
through your monthly Opus quota.

```bash
./cc-gh-proxy.py --no-opus
# or pin the target:
./cc-gh-proxy.py --no-opus --no-opus-target claude-sonnet-4.5
```

The substitution happens after canonicalization, so date and bracket
suffixes are handled. Sonnet and Haiku requests pass through unchanged.

### Using a local model (Ollama)

If your Copilot quota is exhausted (and you don't want `gpt-5-mini`), you
can route through a local Ollama install. Pull a tool-capable model and
point the proxy at Ollama's OpenAI-compatible endpoint:

```bash
ollama pull gemma4:26b   # or any tool-capable model
./cc-gh-proxy.py \
  --upstream-base-url http://localhost:11434/v1 \
  --upstream-model gemma4:26b
```

Caveats:
- Local models are noticeably slower and weaker at structured tool calls
  than Claude. Expect a typical Claude Code agent loop to take 5-10x longer.
- Reasoning models that stream `delta.reasoning` (Gemma 4, DeepSeek-R1)
  are handled: if `max_tokens` is too small to produce real `content`, the
  buffered reasoning text is surfaced instead so responses are not silently
  empty. Bump `max_tokens` to get proper output.

### Using a non-Claude Copilot model

```bash
./cc-gh-proxy.py --upstream-model gpt-5-mini --copilot-auth
```

The first run prints a device-flow URL and code; visit GitHub to authorize
the Copilot OAuth app. The token is cached under `~/.config/cc-gh-proxy/`.

### Using Cursor (or any OpenAI-compatible client) with Copilot

The proxy also exposes a raw OpenAI `/chat/completions` passthrough so tools
like Cursor that only speak the OpenAI API can use your Copilot subscription.
No translation — the body is forwarded unchanged to Copilot's OpenAI endpoint.

```bash
./cc-gh-proxy.py --copilot-auth
```

In Cursor: Settings → Models → "Override OpenAI Base URL" =
`http://localhost:4000/v1`. Use any Copilot-supported model name (`gpt-4o`,
`gpt-5-mini`, etc.). If you set `--api-key`, paste it as the OpenAI API key —
the proxy accepts it via either `x-api-key` or `Authorization: Bearer`.

Only `/chat/completions` and `/models` are exposed. OpenAI's `/responses`
endpoint (used by Codex) is not — Copilot doesn't implement it.

### WebSearch via Tavily

GitHub Copilot **does not execute Anthropic's server-side tools** like
`web_search_20250305` / `web_fetch_20250305`. As a result, Claude Code's
`WebSearch` returns "Did 0 searches" when routed through Copilot.

Pass `--tavily-api-key` and the proxy will intercept Claude Code's
WebSearch sub-call and serve it from [Tavily](https://tavily.com) instead:

```bash
./cc-gh-proxy.py --tavily-api-key tvly-...
```

How it works:
1. Claude Code fires a small executor request whose `tools` list contains
   only `web_search_*` / `web_fetch_*` (no `Read`, `Bash`, etc.). The proxy
   recognizes this exact shape and intercepts it before reaching Copilot.
2. The user's query is extracted from the last message and sent to Tavily.
3. The Tavily response is repackaged as a streaming Anthropic SSE turn with
   three content blocks: `server_tool_use` (so Claude Code's UI reports
   `Did N searches`), `web_search_tool_result` (the structured results, with
   Tavily's extracted page content in `encrypted_content`), and a text
   block with a Markdown summary as fallback.
4. Mixed requests (server tool injected alongside client tools) stay on
   Copilot — those are normal conversation turns and don't need Tavily.

Pricing: `$0.008` per `advanced` search, `$0.005` per `basic`. Each Tavily
call is logged to `requests.jsonl` as `"tavily": true` with `cost_usd` and
`spend_today_usd` running totals.

Why Tavily and not direct Anthropic web_search? Anthropic charges `$10 / 1k`
searches (~30x more expensive) and Claude Code's `WebFetch` follow-ups
often fail on bot-protected hosts (Microsoft Learn, Azure docs, etc.). The
`advanced` Tavily mode returns extracted page content inline, so the model
typically doesn't need to issue a follow-up `WebFetch` at all.

Note: `WebFetch` is a client-side Claude Code tool — it runs from your
local machine and never traverses the proxy. Socket / TLS errors on
`WebFetch` are network-side (often bot mitigation) and cannot be fixed by
the proxy.

## Configuration

All options can be set via CLI arguments or environment variables. CLI takes precedence.

| CLI flag              | Environment variable      | Default               | Description                                                |
|-----------------------|---------------------------|-----------------------|------------------------------------------------------------|
| `-p, --port`          | `PROXY_PORT`              | `4000`                | Port the proxy listens on                                  |
| `--host`              | `PROXY_HOST`              | `127.0.0.1`           | Address to bind to                                         |
| `--api-key`           | `PROXY_API_KEY`           | *(none)*              | Require this key via `x-api-key` header                    |
| `--log-dir`           | `PROXY_LOG_DIR`           | `./logs`              | Directory for log files                                    |
| `--log-level`         | `PROXY_LOG_LEVEL`         | `INFO`                | Log level (`DEBUG` for more)                               |
| `--log-requests`      | `PROXY_LOG_REQUESTS`      | off                   | Log message text and response bodies (privacy-sensitive)   |
| `--upstream-model`    | `PROXY_UPSTREAM_MODEL`    | *(none)*              | Force all requests to use this model                       |
| `--copilot-auth`      | `PROXY_COPILOT_AUTH`      | off                   | Use Copilot OAuth app (required for non-Claude models)     |
| `--upstream-base-url` | `PROXY_UPSTREAM_BASE_URL` | *(none)*              | OpenAI-compatible base URL (bypasses Copilot)              |
| `--upstream-api-key`  | `PROXY_UPSTREAM_API_KEY`  | *(none)*              | Bearer token for `--upstream-base-url`                     |
| `--no-opus`           | `PROXY_NO_OPUS`           | off                   | Rewrite `claude-opus-*` to `--no-opus-target`              |
| `--no-opus-target`    | `PROXY_NO_OPUS_TARGET`    | `claude-sonnet-4.6`   | Target model for `--no-opus` rewrites                      |
| `--tavily-api-key`    | `PROXY_TAVILY_API_KEY`    | *(none)*              | Enable Tavily WebSearch interception (see below)           |
| `--tavily-search-depth` | `PROXY_TAVILY_SEARCH_DEPTH` | `advanced`        | Tavily depth: `basic` ($0.005) or `advanced` ($0.008)      |
| `--tavily-max-results` | `PROXY_TAVILY_MAX_RESULTS` | `5`                  | Maximum results per Tavily search                          |

### API key authentication

When `--api-key` is set, the proxy validates the `x-api-key` header on every request.
Claude Code sends this header automatically when `ANTHROPIC_API_KEY` is set (either in
`.claude/settings.json` or as an environment variable). Requests with a missing or wrong
key get a 401.

Note: setting `ANTHROPIC_API_KEY` makes Claude Code show "API Usage Billing" in the
status bar, but no Anthropic charges occur since requests go to Copilot, not Anthropic.

## Logs

The proxy writes three levels of logging:

- **Console** (stderr) — compact one-liners for each request/response
- **`logs/proxy.log`** — same as console but timestamped, for reviewing later
- **`logs/requests.jsonl`** — machine-readable, one JSON object per request with
  full details (model, messages, usage, timing, response text)

## Troubleshooting

### "Failed to get gh token"

```bash
gh auth refresh --hostname github.com -s copilot
```

### "Access to this endpoint is forbidden"

Your GitHub account doesn't have a Copilot subscription, or the `copilot` scope
is missing:

```bash
gh auth status  # should show 'copilot' in Token scopes
```

### Proxy is running but Claude Code hangs

The `gh` token may have expired. Restart the proxy to refresh it.

## Files

```
cc-gh-proxy/
├── cc-gh-proxy.py   # The proxy server
├── test.sh          # End-to-end test suite
├── CLAUDE.md        # Project context for Claude Code
├── LICENSE          # MIT
└── logs/            # Created at runtime (gitignored)
    ├── proxy.log
    └── requests.jsonl
```

## License

MIT
