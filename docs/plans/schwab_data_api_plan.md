# Schwab Market-Data API — Plan

**Working document — not a skill doc yet.** Becomes `docs/skills/47_schwab_data_api.md` once Phase 0 lands.

**One-line intent.** Expose the existing `SchwabMarketDataProvider` methods (chain fetches, price, snapshots) as a small HTTP JSON API running on the same machine as the trading agent, so a remote agent — Claude session on a laptop, a Cowork process on another Mac, a scheduled task on any tailnet node — can pull live Schwab data over Tailscale without having its own Schwab OAuth flow.

**Why this shape.** You already have working Schwab OAuth on the agent machine. Rebuilding auth in every consuming process duplicates state (three copies of tokens, three refresh clocks, three chances to lock the account). One process owns the tokens; everything else calls its HTTP surface. Tailscale takes care of transport — no public internet exposure, no shared TLS certs, ACLs decide which of your nodes can reach the endpoint.

---

## Endpoint surface

Read-only. Every route below is a thin passthrough to the `SchwabMarketDataProvider` you already have, using the same normalized dict shapes. No new schemas, no new logic in the server layer beyond auth + rate limiting.

| Method | Route | Wraps | Query / Body |
|---|---|---|---|
| GET | `/health` | — | Liveness. Returns `{"status": "ok"}`. |
| GET | `/ready` | `SchwabOAuth.get_access_token()` | Readiness. Returns `{"status": "ready"}` when token refresh works, `503` otherwise. |
| GET | `/price/{ticker}` | `get_current_price` | Just the ticker. Returns `{"ticker": "SPY", "price": 542.30, "as_of": "..."}`. |
| GET | `/chain/{underlying}` | `fetch_option_chain` | Query: `expiration=YYYY-MM-DD`, `option_type=call\|put`. Returns the same list-of-dicts shape the chain scanner already consumes. |
| POST | `/quotes` | `fetch_option_quotes` | Body: `{"symbols": ["SPY260724C00762000", ...]}`. Returns per-symbol quote dicts. |
| POST | `/snapshots` | `fetch_batch_snapshots` | Body: `{"tickers": ["SPY", "QQQ", ...]}`. Returns per-ticker snapshot dicts with mark, day change, etc. |
| GET | `/market-status` | `market_hours.is_within_market_hours` | Returns `{"open": true, "next_open": "...", "next_close": "..."}` — pure calendar math, no Schwab call. |

Explicitly NOT exposed — and the CI invariant scanner will fail the PR if any of these show up:

- Order placement (any side, any strategy)
- Account state (positions, balances, orders)
- Journal reads / writes
- Preset config mutations
- Anything that talks to Alpaca (the executor stays out of this server entirely)

The read-only contract lives in code as a `SERVER_READ_ONLY = True` module-level constant plus a conformance test that greps the server module for any import of `executor`, `journal_kb`, or `submit_order`.

---

## Auth model

Two layers, both simple.

**Layer 1 — Tailscale network gate.** The server binds to your tailnet address (`100.x.y.z`) or to `localhost` behind `tailscale serve`. Only devices on your tailnet can reach the socket. Tailscale ACLs let you restrict which of your nodes can hit it (`tag:trading-data` on the server, `tag:trading-clients` on the laptop / cowork machine, ACL rule allowing `tag:trading-clients → tag:trading-data:8765`). This is enough for a solo tailnet with no shared devices.

**Layer 2 — API key in `Authorization` header.** Defense-in-depth for two cases: a device on the tailnet that gets compromised, or a friend / family Tailscale share. One shared secret in an env var (`SCHWAB_API_SERVER_KEY`), sent as `Authorization: Bearer <key>` on every request. Requests without the header get a `401` with zero information leaked. This layer is optional — if the env var is unset the server logs a WARNING at startup and runs on Tailscale-only gating.

Not doing OAuth on the endpoint, not doing per-user tokens. If you get to the point of "multiple users with different permissions" — you're past what this system should be doing anyway.

---

## Response shapes + error handling

- **Happy path**: 200 with the same dict shape the internal Python API returns. No wrapping envelope, no meta block, no cursor — this is a passthrough.
- **Upstream Schwab failure**: 502 Bad Gateway with `{"error": "schwab_upstream", "detail": "<sanitized message>"}`. Token-refresh failures get a specific `{"error": "schwab_auth"}` so the client can tell you to re-login.
- **Client error** (missing param, bad ticker, unknown expiration): 400 with the specific parsing error.
- **Missing / bad API key**: 401 with `{"error": "unauthorized"}`. Same body for missing header and wrong key — no oracle.
- **Rate limit tripped**: 429 with `Retry-After` header + `{"error": "rate_limited"}`.
- **Genuine bug**: 500 with `{"error": "internal"}`. The traceback goes to the server's log file, never in the response body.

Every request writes one access-log line: `ts | remote_addr | method | path | status | latency_ms | ticker_or_symbol`. Rotates daily via `logging.handlers.TimedRotatingFileHandler`. The log becomes the audit trail — you can see who hit what, when, and whether Schwab was cranky.

---

## Rate limiting + caching

**Cache-first.** The chain fetches already cache 3 minutes in `SchwabMarketDataProvider._option_cache`. The server layer just calls the same provider, so cache hits are free — Schwab isn't hit twice for the same chain across two clients. Add a similar cache for `get_current_price` (60 seconds TTL — prices move but not that fast) and `fetch_batch_snapshots` (90 seconds).

**Rate limit per client key.** Schwab's Trader API is 120 requests/minute per account. If one client hammers `/chain` on 20 tickers every 5 seconds we hit the ceiling fast. Add a token-bucket rate limit per API key: 60 req/min sustained, 20 burst. Configurable via env var (`SCHWAB_API_RATE_LIMIT_RPM=60`). Return `429 Retry-After: <seconds>` on trip.

**Per-endpoint circuit breaker.** If Schwab returns 5xx three times in 30 seconds, the endpoint enters "open circuit" mode for 60 seconds — every request gets an immediate 503 without even trying upstream, so we don't compound the outage. Reset on the first successful call after the timeout.

---

## Deployment

**Separate process.** The API server runs as its own Python process, distinct from the trading agent. A crash in one doesn't take out the other, they can be upgraded independently, and the API server can restart without disrupting the agent's cycle. Both processes import the same `SchwabMarketDataProvider` class — they just construct their own instance and don't share memory.

**macOS launchd** (your Mac): a plist under `~/Library/LaunchAgents/com.balakrishna.schwab-api.plist` that auto-starts on login, restarts on crash, redirects stdout / stderr to `~/Library/Logs/schwab-api.{out,err}.log`. `launchctl load` once, then it's persistent.

**Linux systemd** (if you ever move to the pi): a unit file at `/etc/systemd/system/schwab-api.service` with `Restart=on-failure`, `RestartSec=5`, and the working directory pointing at the trading-agent checkout.

**Ports**: default `8765` on the tailnet address. Configurable via `SCHWAB_API_PORT`. The trading agent, Streamlit dashboard, and Long-Term Evaluator all stay on their existing ports — no collision.

**Startup checklist logged at INFO on process start**: bound address, Tailscale hostname if detectable, API key presence (never the key itself), Schwab OAuth token expiry timestamp, cache TTLs, rate-limit config. That's the "am I actually running the right thing?" one-liner an operator scans on restart.

---

## Framework choice

**FastAPI + Uvicorn.** Reasons:

- Native async, so a `/snapshots` call for 15 tickers can dispatch concurrently to the underlying Schwab batch endpoint instead of blocking N times.
- Pydantic models give you request/response schema validation for free — a bad `option_type=option_type=call` typo returns 400 before it ever hits the Schwab layer.
- Auto-generated OpenAPI docs at `/docs` — the future remote agent can introspect the API surface with one HTTP call and generate tool bindings from it.
- Ubiquitous, well-documented, no surprises. Bare `http.server` is smaller but rate-limiting + auth middleware in Flask/http.server is more code than it saves.

Adds two dependencies to `pyproject.toml`: `fastapi>=0.110`, `uvicorn[standard]>=0.27`. Both pure Python, both trusted, both already in most macOS Python setups.

---

## Phased build

Each phase is ~1 session unless noted. Every phase includes skill doc updates + conformance tests + SDD housekeeping.

### Phase 0 — Skill 47 design doc (½ session)

Draft `docs/skills/47_schwab_data_api.md` covering: theory (why a single-owner data server), endpoint surface, auth model, response shapes, rate limiting + caching, deployment. Cross-references to skill 16 (provider routing), skill 34 (ExceptionMonitor for Schwab auth failures), skill 42 (portfolio-review scheduler as an existing local-only consumer that could optionally switch to the API).

### Phase 1 — Core server (1 session)

- `trading_agent/data_server/app.py` — FastAPI app + route handlers. Every route delegates to a shared `SchwabMarketDataProvider` instance.
- `trading_agent/data_server/auth.py` — API key middleware, `Authorization: Bearer <key>` check.
- `trading_agent/data_server/cache.py` — in-process TTL caches for price and batch snapshots (chain cache already lives on the provider).
- `trading_agent/data_server/__main__.py` — CLI entry (`python -m trading_agent.data_server --port 8765 --bind 100.x.y.z`).
- 15-20 conformance tests: happy-path per endpoint, auth rejection, upstream error mapping, cache hit/miss, invariant test that verifies no `executor` / `journal_kb` / `submit_order` imports.

Deliverables at end of Phase 1: `python -m trading_agent.data_server` runs on your Mac, `curl -H "Authorization: Bearer $KEY" http://100.x.y.z:8765/price/SPY` returns live SPY.

### Phase 2 — Rate limit + circuit breaker + launchd unit (½ session)

- Token-bucket rate limiter (`slowapi` or hand-rolled — probably hand-rolled, one file, easier to conformance-test).
- Per-endpoint circuit breaker (3 upstream 5xx in 30s → open 60s).
- `deploy/launchd/com.balakrishna.schwab-api.plist` template + a `README.md` for installation.
- Access-log rotation config.

### Phase 3 — MCP wrapper (optional, ½ session)

For Claude Code / Cowork consumers: a thin MCP server (`trading_agent/data_server/mcp_bridge.py`) that exposes the same routes as MCP tools (`get_current_price`, `fetch_option_chain`, `fetch_option_quotes`, `fetch_batch_snapshots`, `is_market_open`). Reuses every underlying handler — same cache, same auth, same rate limit. Just a second front-end.

Only ship if you actually want to plug the data server into a Claude session as first-class tools. If your consumer is a Python script or shell, the HTTP layer is enough.

### Phase 4 — Consumer refactor (½ session, optional)

Convert the Long-Term Evaluator's chain fetcher (skill 40 §3.4) and the portfolio-review scheduler (skill 42) to optionally use the API instead of instantiating their own `SchwabMarketDataProvider`. Gated behind an env var (`USE_SCHWAB_API=1`). Off by default; flip on if you want to prove the API works end-to-end against real internal consumers.

---

## Decisions to make before Phase 0

Three, each with a sensible default. Reply with anything you'd override.

**1. FastAPI vs Flask vs bare stdlib.** Default: FastAPI. Overrides only if you've hit specific problems with FastAPI or want minimum dependencies.

**2. API key auth in Phase 1 or Phase 2.** Default: Phase 1 — Tailscale + API key together from day one, since your Tailscale might have shared devices and defense-in-depth is cheap. Override to Phase 2 only if you want the fastest possible ship in Phase 1.

**3. MCP wrapper — ship or defer.** Default: defer to Phase 3 as optional. If the consuming agent is Claude and you'd rather have native MCP tools than HTTP-via-curl, promote it into Phase 1 and I'll build the FastAPI + MCP fronts together.

---

## What NOT to build here

Repeating for emphasis — these are out of scope for this server:

- **Order placement.** If the future includes a Schwab executor, it lives in `trading_agent/executor_schwab.py`, gets its own OAuth scope, its own Preview → Confirm → Place UI, and does NOT ride this data server's auth. The whole point of "data server" is that a compromised API key can only leak read-only market data — nothing can transfer money.
- **Personal data.** No account balances, no positions, no P&L, no journal contents. If the remote agent wants to know "am I bullish or bearish?" it computes that itself from the price data — not by asking this server.
- **Alpaca fallback.** If Schwab is down, the server returns 502. Do not silently substitute Alpaca — the caller might make different decisions based on which feed answered.

---

## Next step

Say the word and I'll draft `skill_47_schwab_data_api.md` + Phase 1's `trading_agent/data_server/app.py` in one commit. The skill goes first per the SDD process; the app plus conformance tests come with it. If you'd rather pin down one of the three decisions above first, tell me and I'll re-shape.
