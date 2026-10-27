# Schwab Data API — Local HTTP Server

> **One-line summary:** FastAPI server that exposes the existing `SchwabMarketDataProvider` methods as read-only JSON endpoints over the operator's tailnet. One process owns Schwab OAuth tokens; every other consumer (Cowork sessions, remote Claude, ad-hoc curl from any tailnet node) reaches Schwab via HTTP with a shared bearer token. Bearer-token + Tailscale-network gating for auth. Read-only invariant enforced in code and by conformance test — never any route that could place an order or mutate account state.
> **Source of truth:** [`trading_agent/data_server/app.py`](../../trading_agent/data_server/app.py), [`trading_agent/data_server/auth.py`](../../trading_agent/data_server/auth.py), [`trading_agent/data_server/cache.py`](../../trading_agent/data_server/cache.py), [`trading_agent/data_server/config.py`](../../trading_agent/data_server/config.py).
> **Phase:** 2  •  **Group:** ops
> **Depends on:** `16_market_data_provider_routing.md` (the `SchwabMarketDataProvider` this server wraps), `34_exception_monitor.md` (Schwab auth failures still page the operator through the same channel), `18_order_submission_idempotency.md` (this server is the FIRST thing to explicitly refuse to expose the executor).
> **Consumed by:** any remote agent / process on the operator's tailnet.

---

## 1. Theory & Objective

The trading agent already carries a working Schwab OAuth flow — refresh tokens rotate every ~30 minutes, hard-expire on a 7-day absolute clock, and the adapter caches option chains for 3 minutes. Rebuilding that in every consuming process is expensive: two copies of the tokens race on refresh, two OAuth code-paste dances at bootstrap, two rate-limit buckets that can starve each other. One machine already owns the state — this skill lets other machines reach through it.

Concrete triggers for building this over just handing every consumer its own OAuth:

- Multiple remote consumers. Two agents on two Macs each holding tokens doubles the failure surface.
- Ephemeral consumers. A Cowork session that spins up for 90 seconds and dies shouldn't OAuth-dance.
- Untrusted consumers. A Claude session with a revocable API key is safer than a Claude session with your Schwab refresh token.
- Central rate limiting. If a consumer misbehaves it burns THIS server's quota, not your trading account's — the middle layer throttles before Schwab sees anything.
- Audit trail. One access log = one grep to answer "who fetched what when."

The server is **read-only by design and by construction**. Order placement lives in a separate module (`executor.py` today, `executor_schwab.py` when Trader-API integration lands). This server cannot import either. A compromised bearer token here can only leak market-data — never move money. The invariant is enforced by a conformance test that greps for `executor`, `submit_order`, `journal_kb`, and `place_order` imports.

## 2. Endpoint surface

Sixteen routes. The first eight are thin passthroughs to `SchwabMarketDataProvider` — no new schemas, no new logic beyond auth + caching + rate limiting. The three research routes (§3.5, 2026-10-26) and five market-context routes (§3.6, 2026-10-27) put news, analyst research, sentiment and the skill-68 market inputs behind the same server and key, so every data point comes from one place.

```text
GET  /health                                     — liveness, no auth
GET  /ready                                      — readiness: Schwab returns data (auth_probe, A8)
GET  /price/{ticker}                             — current price
GET  /chain/{underlying}?expiration=&option_type=
                                                 — option chain (call|put)
POST /quotes           body {"symbols": [...]}   — batch option/stock quote lookup
POST /snapshots        body {"tickers": [...]}   — batch stock snapshots with indicators
GET  /market-status                              — is_within_market_hours() + next open/close
GET  /fundamentals/{ticker}                      — P/E, EPS, market cap, dividend, beta, 52w range, ROE/ROA, revenue growth, MRQ margins, debt ratios
GET  /news/{ticker}?days=30&limit=60             — Alpaca news + Yahoo RSS, de-duplicated, spread across days (§3.5)
GET  /research/{ticker}                          — analyst estimates/revisions/targets (Yahoo) + latest quality-at-lows row
GET  /sentiment/{ticker}                         — Claude drop-cause label (stored) + analyst stance + news flow
GET  /macro-events?days=45                       — FOMC / CPI / NFP releases ahead + calendar coverage (§3.6)
GET  /credit-stress                              — HYG/IEF % under its 50-day, CAUTION threshold, last 20 readings
GET  /vol/{ticker}                               — 30d ATM IV, 20d realized, IV/RV, short-premium verdict
GET  /gamma/{ticker}                             — latest recorded dealer gamma (SPY/QQQ/IWM, 16:05 ET)
GET  /market-context                             — market state snapshot + the four above in one call
```

**Explicitly NOT exposed** — CI-verified via `test_readonly_no_forbidden_imports`:

- Order placement (any side, any strategy)
- Account state (positions, balances, orders, buying power)
- Journal writes, and journal reads other than the quality-at-lows shadow rows that `/research` and `/sentiment` serve (no trades, positions or P&L)
- Preset config mutations
- Anything that touches `trading_agent.executor` or Alpaca's trading APIs

## 3. Reference Python Implementation

### 3.1 App entrypoint

```python
# trading_agent/data_server/app.py
def build_app(
    *,
    provider: Any = None,
    api_key: Optional[str] = None,
    cache_ttls: Optional[CacheTTLs] = None,
) -> "FastAPI":
    """Construct the FastAPI app.

    ``provider`` is a MarketDataPort — production wires the real
    ``SchwabMarketDataProvider``; tests pass a stub. ``api_key`` is
    the shared bearer token; when unset the server logs a WARNING
    at startup and runs Tailscale-only.
    """
```

Every route dispatches through the provider. The provider is dependency-injected at app construction so tests can hand in a stub with deterministic outputs.

### 3.2 Auth middleware

```python
# trading_agent/data_server/auth.py
def check_bearer(
    authorization_header: Optional[str],
    *,
    expected_key: Optional[str],
) -> None:
    """Enforce Authorization: Bearer <expected_key> when set.

    * expected_key is None → auth disabled; no-op.
    * missing / malformed / wrong header → AuthError.

    Uses hmac.compare_digest for constant-time equality so an attacker
    can't time-side-channel the correct key.
    """
```

A thin `build_fastapi_dependency(expected_key)` factory wraps `check_bearer` for FastAPI's `Depends()` machinery; the pure logic sits in `check_bearer` and is directly unit-testable without FastAPI.

### 3.3 Cache

```python
# trading_agent/data_server/cache.py
class TTLCache:
    """Thread-safe TTL cache. Separate instances per endpoint so a
    slow /snapshots doesn't invalidate /price entries.
    """
    def get_or_set(self, key: str, fetch: Callable[[], Any]) -> Any:
        ...
```

Chain fetches already cache 3 min inside the provider — the server layer only adds caches for `/price` (60s) and `/snapshots` (90s), since those don't have provider-side caching today. All TTLs are configurable via env vars (`SCHWAB_API_PRICE_TTL_SEC`, `SCHWAB_API_SNAPSHOT_TTL_SEC`).

### 3.4 CLI entry

```python
# trading_agent/data_server/__main__.py
def main(argv: Optional[List[str]] = None) -> int:
```

Invocation from a shell (not Python — just documented here for operators):

```
python -m trading_agent.data_server --port 8765 --bind 100.115.216.79
```

Environment variables the CLI reads:

- `SCHWAB_API_SERVER_KEY` — shared bearer token, **required** and at least 24 characters (2026-10-08): the server exits with code 4 at startup when it is missing or short.
- `SCHWAB_API_ALLOW_NO_KEY` — `true` opts back in to keyless, tailnet-only mode (network gating only). Never combine with Tailscale Funnel. It does not waive the length check for a key that is set.
- `SCHWAB_API_PORT` — port to bind (default 8765).
- `SCHWAB_API_BIND` — address to bind (default 127.0.0.1 — you must override for tailnet exposure).
- `SCHWAB_API_CACHE_ENABLED` — **master cache switch** (default `false`). When false, every price/snapshot/chain request goes live to Schwab; the per-endpoint TTL vars below are ignored. Flip to `true`/`1`/`yes`/`on` when rate limits become a concern.
- `SCHWAB_API_PRICE_TTL_SEC` — price cache TTL when master switch is on (default 60). Ignored when `SCHWAB_API_CACHE_ENABLED=false`.
- `SCHWAB_API_SNAPSHOT_TTL_SEC` — snapshot cache TTL when master switch is on (default 90). Ignored when disabled.
- `SCHWAB_API_CHAIN_TTL_SEC` — option-chain provider cache TTL when master switch is on (default 180). Ignored when disabled — chain cache is force-set to 0 regardless of this value.
- `SCHWAB_API_LOG_FILE` — access-log destination (default stdout).

Startup log surfaces the resolved mode so the operator can verify at
a glance. When caching is off:

```
Schwab Data API starting — bind=100.115.216.79 port=8765 auth=bearer-token
    cache=DISABLED (every request live — set SCHWAB_API_CACHE_ENABLED=true to enable)
```

When on:

```
cache=ENABLED (price_ttl=60s snapshot_ttl=90s)
```

### 3.5 Research routes (2026-10-26)

```python
# trading_agent/data_server/research.py — pure helpers + injectable sources
@dataclass
class ResearchSources:
    fetch_news: Callable[[str, int, int], List[NewsItem]]      # news_aggregator.ticker_news
    fetch_estimates: Callable[[str], Optional[Dict]]           # analyst_estimates.get_estimates
    quality_row: Callable[[str], Optional[Dict]]               # latest quality_lows.jsonl row
    now: Callable[[], datetime]

def research(ticker, src) -> Dict          # {"analyst_estimates", "quality_lows", "notes", "as_of"}
def sentiment(ticker, news_rows, src, notes) -> Dict
    # {"drop_cause_llm": {label, confidence, summary, model, news_from, scan_date} | None,
    #  "drop_cause_rule", "quality_lows_stage",
    #  "analyst": {revision_trend, revisions_up/down_30d, rec_score_now/3m_ago,
    #              recommendation_drift: upgrading|downgrading|steady, target_upside_pct},
    #  "news_flow": {items_last_7d, items_prior_23d, rate_ratio, capped},
    #  "latest_headlines": [...5], "notes": [...], "as_of"}

# app.py — build_app(provider=..., config=..., research_sources=None)
news_cache = TTLCache(NEWS_TTL_SEC)        # 600 s, on regardless of SCHWAB_API_CACHE_ENABLED

@app.get("/news/{ticker}")                 # plain def → FastAPI thread pool
def news(ticker, days=30, limit=60): ...   # 400 outside days 1-90 / limit 1-200
```

The logic stays in `news_aggregator.py`, `analyst_estimates.py` and `quality_lows.py` (skill 67); the server only exposes it. MCP tools `get_news`, `get_research`, `get_sentiment` (skill 48) read these routes with a 45 s timeout.

### 3.6 Market-context routes (2026-10-27, skill 68)

```python
# trading_agent/data_server/market_context.py
@dataclass
class ContextSources:
    history: Callable[[str], Any]                 # provider.fetch_historical_prices (yfinance)
    atm_iv: Callable[[str, date], Optional[float]]   # macro_vol_gates.latest_atm_iv (vol_surface rows)
    gamma_rows: Callable[[], Dict]                # gamma_exposure.latest_rows
    market_state: Callable[[], Optional[Dict]]    # market_state.read_state (the agent's snapshot)
    events: Callable[[], List[MacroEvent]]        # macro_calendar.all_events
    now_et: Callable[[], datetime]
    adjusted: Optional[Callable] = None           # market_state.yfinance_adjusted_history (credit pair)

# app.py — build_app(..., context_sources=None)
context_cache = TTLCache(900)        # /credit-stress and /vol: daily-history inputs
```

Same thresholds and functions as the agent (`credit_stress_pct`, `iv_rv_ratio`, `pending_events`, `MarketStateConfig`, the active preset's `vrp_min_ratio`), so the server and the agent cannot disagree. MCP: `get_macro_events`, `get_credit_stress`, `get_vol`, `get_gamma`, `get_market_context`. `/chain` contracts gained `open_interest` (None when Schwab omits it) for the gamma recorder.

## 4. Edge Cases / Guardrails

- **Read-only invariant enforced by conformance test.** `test_readonly_no_forbidden_imports` greps `data_server/*.py` for imports of `executor`, `journal_kb`, `submit_order`, `place_order`, `TradingClient`. Any hit fails CI. Belt for the "one bad refactor and we're now placing orders" scenario; suspenders for the "compromised bearer token" scenario.
- **Auth key comparison is constant-time.** `hmac.compare_digest` is used instead of `==` so an attacker can't time-side-channel the correct key from the failure response.
- **401 body is identical for "missing header" and "wrong key".** No information leak: `{"error": "unauthorized"}` verbatim in both cases.
- **Provider is injected, not imported at module load.** `build_app(provider=...)` accepts any `MarketDataPort` — tests hand in a `FakeProvider` with pre-baked returns. No env-dependent global state at module load.
- **Cache is per-instance, not module-global.** `TTLCache` instances live on the app object so two apps in the same process don't share state. Matters for parallel pytest runs.
- **Schwab upstream failures map to 502, NOT 500.** `SchwabAuthError` → 502 with `{"error": "schwab_auth"}`; any other requests-side exception → 502 with `{"error": "schwab_upstream"}`. 500 is reserved for genuine bugs in the server layer itself.
- **Empty batch body → 400, not 200 with empty response.** POST /quotes with `{"symbols": []}` is a client bug (no work asked for). Reject at request-validation time with 400. Same for /snapshots.
- **Ticker parameter validation.** The ticker is stripped and **uppercased first** (so `spy` is served as `SPY`), then must match `^[A-Z][A-Z0-9.-]{0,9}$` — alphanumeric plus `.` (BRK.B) and `-` (rare). Reject anything else at 400. Prevents `../` path traversal and shell-injection footguns even though we never shell out.
- **Access log never contains the API key.** The `Authorization` header is redacted before logging. Failed auth attempts log `<AUTH_FAILED>` in the auth field, not the attempted value.
- **/ready probes real data (A8, 2026-10-26).** With a provider that has `auth_probe()` (Schwab), `/ready` is 200 only when a token can be obtained *and* `/markets` returns a body; otherwise 503 with the probe's detail ("Schwab token unavailable: …" / "Schwab /markets returned no data"). Before, it called `is_market_open()`, which swallows auth failures and falls back to Alpaca's clock — `/ready` said "ready" through the 2026-10-20 10-hour token outage. The liveness watchdog (skill 64) pages on the 503 in session. Providers without `auth_probe` keep the old `is_market_open()` probe.
- **Option quotes carry `delta` (2026-10-26).** `fetch_option_quotes` (and so `POST /quotes`) returns `{"bid", "ask", "mid", "delta"}`; `delta` is Schwab's `QuoteOption.delta`, None when absent — never 0.0. The agent's book net-delta reading (skill 37, E4) uses it from the monitor's re-mark quotes.
- **/health has no auth.** Liveness probes from launchd or an external uptime service must work without a key. `/ready` also has no auth — it reveals only Schwab-token validity (a boolean), not any market data.
- **CORS is off by default.** No `Access-Control-Allow-Origin` header. If a browser-side consumer ever appears, add it explicitly then. Curl and Python `requests` don't care about CORS.
- **The trading agent process does NOT depend on this server.** If the server is down, the trading agent's cycle proceeds normally with its own in-process `SchwabMarketDataProvider`. Vice versa: if the trading agent is down, the API server keeps serving. Full crash isolation.

- **Key source + fingerprint at startup (2026-09-29).** `main()` loads `.env` *before* `ServerConfig.from_env()` (previously the key came only from the launching shell) and logs `API key source=<shell env | shell env (overrides .env) | .env | unset> fingerprint=<sha256[:8]>`. The key itself is never logged. If a shell export shadows a *different* `.env` key it logs a WARNING with both fingerprints and the `env -u SCHWAB_API_SERVER_KEY …` restart command — the stale-export case that made every MCP call 401. Operators compare with `printf %s "$SCHWAB_API_SERVER_KEY" | shasum -a 256 | cut -c1-8` on the client side; `e3b0c442` = empty.

- **Refuses to start without a strong key (2026-10-08).** The server can be published to the internet with Tailscale Funnel (runbook 08 §8b), where the bearer key is the only protection. `key_policy_error()` in `__main__.py` stops startup (exit 4, before the Schwab provider is built) when `SCHWAB_API_SERVER_KEY` is missing, blank or shorter than 24 characters, unless `SCHWAB_API_ALLOW_NO_KEY=true` is set for a tailnet-only server. The error message never echoes the key. Under systemd (`Restart=always`) a refused start retries every 30 s and logs the reason to `/tmp/trading-agent.data-server.err.log` until the key is fixed.

- **Growth fields in `/fundamentals` (2026-10-26, skill 67).** `rev_change_ttm_pct`, `rev_change_year_pct`, `gross_margin_mrq`, `net_profit_margin_mrq`, `return_on_investment`, `current_ratio`, `interest_coverage`, `lt_debt_to_equity`, `total_debt_to_capital`, `short_int_days_to_cover` — mapped from the raw Schwab block checked with `SCHWAB_API_FUNDAMENTALS_INCLUDE_RAW=true` on 2026-10-25. Revenue growth and days-to-cover read exact 0.0 → None (Schwab's "not provided"; `revChangeIn` and `epsChangeYear` were 0.0 on every ticker and are not mapped). New fields reach clients only after the data server restarts. No analyst forecasts here — those come from `analyst_estimates.py` (Yahoo).

- **Research routes are not Schwab data (2026-10-26).** `/news`, `/research` and `/sentiment` call Alpaca's news API (the paper keys in `.env`), Yahoo's RSS and Yahoo's analyst tables — never Schwab, so they work when the token is dead and never count toward Schwab's rate limit. Each source is optional: a failed analyst lookup or quality-lows read becomes `null` plus a `notes` entry, not an error for the whole response; only `/news` itself returns 502 (`news_upstream`) when the fetch raises. Note that `ticker_news` swallows per-source failures and returns `[]`, which is cached for 10 minutes like any result.
- **No LLM per request.** `/sentiment` serves the Claude label the 15:40 ET quality-at-lows scan stored (`llm_cause`, names at `watch` or later); other names get `drop_cause_llm: null` with a note. A live Claude call takes about a minute and spends usage, so it stays in the scan.
- **News flow is approximate when capped.** Past `limit` items, `ticker_news` takes items round-robin across days, so `items_last_7d` vs `items_prior_23d` undercounts busy days; `news_flow.capped` is true in that case.
- **Analyst cache file.** `/research` and `/sentiment` go through `get_estimates`, which refetches from Yahoo after 20 h and rewrites `trade_journal/analyst_estimates.json` (temp + rename) — the only file the server writes. Relative to the service's working directory (the repo).

- **Market-context routes read local snapshots (2026-10-27).** `/market-context` returns the agent's last `market_state.json` (with `age_seconds`), `/vol` the newest vol-surface row ≤ 4 days old, `/gamma` the newest gamma row — none of them call Schwab. A failed piece of `/market-context` becomes `null` plus a `notes` entry; `/credit-stress` alone returns 502 `context_upstream` when the history fetch raises. `/credit-stress` and `/vol` are cached 15 minutes (their inputs are daily).
- **Gamma rows need the restarted server.** `open_interest` appears on `/chain` only after the data server restarts on the 2026-10-27 code; the recorder refuses to write rows without it.

## 5. Cross-References

- `16_market_data_provider_routing.md` — the underlying provider this server wraps. When the surface routing picks Schwab, we're calling the same code path this server exposes.
- `18_order_submission_idempotency.md` — the executor whose contract we deliberately don't re-implement here. The `SERVER_READ_ONLY = True` invariant is symmetric to "the executor is the ONLY place that submits orders."
- `34_exception_monitor.md` — Schwab auth failures inside the server still bubble up to `ExceptionMonitor.record(...)` so the operator's Telegram error channel pages once per failure mode per day.
- `68_macro_events_vrp_credit_gamma.md` — the market inputs behind `/macro-events`, `/credit-stress`, `/vol`, `/gamma`, `/market-context`.
- `67_quality_at_lows_sleeve.md` — owner of the news, analyst-estimate and drop-cause logic the research routes expose.
- `48_claude_code_mcp_surface.md` — `get_news` / `get_research` / `get_sentiment` passthroughs.
- `42_portfolio_alert_scheduler.md` — a current in-process consumer of `SchwabMarketDataProvider` that could optionally refactor to use this API (Phase 4 in `docs/plans/schwab_data_api_plan.md`).

---

*Last verified against repo HEAD on 2026-10-27.*
