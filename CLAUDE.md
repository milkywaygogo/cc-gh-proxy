# cc-gh-proxy

Pass-through proxy: Claude Code -> GitHub Copilot using Copilot's **native Anthropic Messages API**.

For Claude models: native pass-through to `/v1/messages` with no format translation.
For non-Claude models (GPT-5 mini, GPT-5.1, etc.): translates between Anthropic and OpenAI formats via `/chat/completions` (experimental).

## Architecture

```
Claude models:
  Claude Code --> localhost:4000 --> api.githubcopilot.com/v1/messages
               (swap auth token,    (native Anthropic Messages API)
                map model names,
                strip cache_control)

Non-Claude models (--upstream-model gpt-5-mini):
  Claude Code --> localhost:4000 --> api.githubcopilot.com/chat/completions
               (Anthropic→OpenAI      (OpenAI Chat Completions API,
                translation,           requires Copilot OAuth token)
                OpenAI→Anthropic
                response translation)

WebSearch (when --tavily-api-key is set):
  Claude Code --> localhost:4000 --> api.tavily.com/search
               (intercept executor   (extracted page content,
                requests with        $0.005-0.008 per call)
                tools=[web_search_*],
                synthesize Anthropic
                server_tool_use +
                web_search_tool_result
                SSE response)
```

## Key decisions

- **No dependencies**: stdlib-only Python (http.client, http.server, json, ssl, threading, argparse). No pip install needed.
- **Native Anthropic pass-through**: Copilot's `/v1/messages` endpoint accepts Anthropic format directly. All other community tools (copilot-api, claude-code-copilot, copilot-proxy) use the OpenAI Chat Completions endpoint and translate formats, losing token counts and adding complexity.
- **OAuth tokens only**: GitHub PATs (classic and fine-grained) don't work with the Copilot API. Must use `gh auth token` with copilot scope.
- **Per-project activation**: Projects opt in via `.claude/settings.json` with `ANTHROPIC_BASE_URL=http://localhost:4000`, or by setting the env var directly. Other projects use Anthropic directly.
- **API key gating**: Optional `--api-key` validates the `x-api-key` header Claude Code sends (set via `ANTHROPIC_API_KEY` in settings.json or env var). Prevents unauthorized use of the proxy. Claude Code shows "API Usage Billing" but no Anthropic charges occur since requests go to Copilot, not Anthropic.
- **Model name mapping**: Claude Code sends dashes (`claude-opus-4-6`), Copilot expects dots (`claude-opus-4.6`). Bracket suffixes (`[1m]`), date suffixes, and base family names also handled.
- **cache_control stripping**: Claude Code sends `{"type": "ephemeral", "scope": "turn"}` but Copilot only accepts `{"type": "ephemeral"}`.
- **Beta header filtering**: Strips `context-*` beta features from `anthropic-beta` header (Copilot rejects unknown betas like `context-1m-2025-08-07`).
- **Token auto-refresh**: `TokenManager` re-fetches `gh auth token` hourly and retries on 401.
- **Threading**: `ThreadingHTTPServer` handles concurrent requests from Claude Code.
- **CLI + env config**: All options (port, host, api-key, log-dir, log-level) configurable via CLI args or env vars.
- **Non-Claude model support** (EXPERIMENTAL): `--upstream-model gpt-5-mini` translates Anthropic↔OpenAI formats. Requires `--copilot-auth` which does a one-time OAuth device flow to get a Copilot API token. Available: gpt-5-mini (0x), gpt-4.1 (0x), gpt-5.1, gpt-5.2, gpt-4o, gemini-2.5-pro. Codex models (/responses only) not supported.
- **Two auth modes**: `gh auth token` for Claude models on `/v1/messages`; Copilot OAuth app (`Iv1.b507a08c87ecfe98`) device flow for non-Claude models on `/chat/completions`.
- **Tavily WebSearch routing**: Copilot does not execute Anthropic's server-side `web_search_*`/`web_fetch_*` tools, so Claude Code's WebSearch silently returns 0 results when proxied. Workaround: intercept Claude Code's "WebSearch executor" pattern — a sub-request whose `tools` list contains *only* server tools, with no client tools (Read/Bash/Edit/...) — and serve it from Tavily. The proxy synthesizes a streaming Anthropic-format response containing three content blocks:
  1. `server_tool_use` (so Claude Code's UI counts the search and reports `Did N searches`)
  2. `web_search_tool_result` with one `web_search_result` per Tavily hit; Tavily's extracted page text goes in `encrypted_content`
  3. `text` block with a Markdown summary as a fallback for consumers that ignore the structured block
  Mixed requests (server tool definition alongside client tools) stay on Copilot — those are the main conversation turns where Copilot just ignores the unsupported tool definition. `WebFetch` is a client-side CC tool, never proxied; failures (e.g. on `azure.microsoft.com`) are TLS/bot-mitigation issues on the target host, not a proxy problem.

## Files

- `cc-gh-proxy.py` — the proxy server (~500 lines)
- `test.sh` — end-to-end test suite (health, non-streaming, streaming, model mapping)
- `README.md` — user-facing docs with setup, usage, troubleshooting

## Known issues / history

- `urllib.request` buffers SSE — that's why we use `http.client.HTTPSConnection` with chunked reads
- The `get_gh_token()` function raises `TokenError` on failure, which `TokenManager._refresh()` catches gracefully
- Setting `ANTHROPIC_API_KEY` for proxy auth makes Claude Code show "API Usage Billing" — harmless since no requests reach Anthropic
