# cc-gh-proxy

Pass-through proxy: Claude Code -> GitHub Copilot using Copilot's **native Anthropic Messages API**.

Unlike every other copilot proxy out there, this does NOT translate between OpenAI and Anthropic formats. GitHub Copilot natively supports the Anthropic Messages API at `https://api.githubcopilot.com/v1/messages` — we just forward requests as-is with auth swapped and minor field cleanup.

## Architecture

```
Claude Code --> localhost:4000 --> api.githubcopilot.com/v1/messages
             (swap auth token,    (native Anthropic Messages API,
              map model names,     no format translation)
              strip cache_control)
```

## Key decisions

- **No dependencies**: stdlib-only Python (http.client, http.server, json, ssl, threading, argparse). No pip install needed.
- **Native Anthropic pass-through**: Copilot's `/v1/messages` endpoint accepts Anthropic format directly. All other community tools (copilot-api, claude-code-copilot, copilot-proxy) use the OpenAI Chat Completions endpoint and translate formats, losing token counts and adding complexity.
- **OAuth tokens only**: GitHub PATs (classic and fine-grained) don't work with the Copilot API. Must use `gh auth token` with copilot scope.
- **Per-project activation**: Projects opt in via `.claude/settings.json` with `ANTHROPIC_BASE_URL=http://localhost:4000`. Other projects use Anthropic directly.
- **API key gating**: Optional `--api-key` validates the `x-api-key` header Claude Code sends (set via `ANTHROPIC_API_KEY` in the project). Prevents unauthorized use of the proxy. Claude Code shows "API Usage Billing" but no Anthropic charges occur since requests go to Copilot, not Anthropic.
- **Model name mapping**: Claude Code sends dashes (`claude-opus-4-6`), Copilot expects dots (`claude-opus-4.6`). Bracket suffixes (`[1m]`), date suffixes, and base family names also handled.
- **cache_control stripping**: Claude Code sends `{"type": "ephemeral", "scope": "turn"}` but Copilot only accepts `{"type": "ephemeral"}`.
- **Beta header filtering**: Strips `context-*` beta features from `anthropic-beta` header (Copilot rejects unknown betas like `context-1m-2025-08-07`).
- **Token auto-refresh**: `TokenManager` re-fetches `gh auth token` hourly and retries on 401.
- **Threading**: `ThreadingHTTPServer` handles concurrent requests from Claude Code.
- **CLI + env config**: All options (port, host, api-key, log-dir, log-level) configurable via CLI args or env vars.

## Files

- `cc-gh-proxy.py` — the proxy server (~500 lines)
- `test.sh` — end-to-end test suite (health, non-streaming, streaming, model mapping)
- `README.md` — user-facing docs with setup, usage, troubleshooting

## Known issues / history

- `urllib.request` buffers SSE — that's why we use `http.client.HTTPSConnection` with chunked reads
- The `get_gh_token()` function raises `TokenError` on failure, which `TokenManager._refresh()` catches gracefully
- Setting `ANTHROPIC_API_KEY` for proxy auth makes Claude Code show "API Usage Billing" — harmless since no requests reach Anthropic
