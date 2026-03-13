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

Only projects with this setting use the proxy. All other projects use Anthropic directly.

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
(e.g. `claude-opus-4`) are also mapped automatically.

## Configuration

All options can be set via CLI arguments or environment variables. CLI takes precedence.

| CLI flag       | Environment variable | Default      | Description                              |
|----------------|----------------------|--------------|------------------------------------------|
| `-p, --port`   | `PROXY_PORT`         | `4000`       | Port the proxy listens on                |
| `--host`       | `PROXY_HOST`         | `127.0.0.1`  | Address to bind to                       |
| `--api-key`    | `PROXY_API_KEY`      | *(none)*     | Require this key via `x-api-key` header  |
| `--log-dir`    | `PROXY_LOG_DIR`      | `./logs`     | Directory for log files                  |
| `--log-level`  | `PROXY_LOG_LEVEL`    | `INFO`       | Log level (`DEBUG` for more)             |

### API key authentication

When `--api-key` is set, the proxy validates the `x-api-key` header on every request.
Claude Code sends this header automatically when `ANTHROPIC_API_KEY` is set in the
project's `.claude/settings.json`. Requests with a missing or wrong key get a 401.

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
