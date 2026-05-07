# Market-Data Provider Routing

> **One-line summary:** Per-surface dispatch between Alpaca / Schwab / Yahoo market-data adapters via env-var routing, so the live agent, watchlist, and backtester can each read from a different feed independently.
> **Source of truth:** [`trading_agent/market_data_factory.py`](../../trading_agent/market_data_factory.py), [`trading_agent/market_data_schwab.py`](../../trading_agent/market_data_schwab.py), [`trading_agent/market_data_yahoo.py`](../../trading_agent/market_data_yahoo.py), [`trading_agent/schwab_oauth.py`](../../trading_agent/schwab_oauth.py)
> **Phase:** 2  •  **Group:** architecture
> **Depends on:** `00_sdlc_and_conventions.md` (hexagonal port pattern)
> **Consumed by:** `agent.py:156` (live cycle), `streamlit/watchlist_ui.py:_get_data_provider`, `streamlit/live_monitor.py:_is_market_open_cached`

---

## 1. Theory & Objective

The agent's cycle, the watchlist tab, and the backtester all read market data, but their fidelity needs differ.  The live cycle is the only surface that must size live option spreads — it benefits most from real-time Schwab quotes (free for Schwab brokerage holders) versus Alpaca's 15-min-delayed `indicative` feed.  The watchlist only draws charts and computes regime indicators, so it can run on Alpaca, Schwab, or yfinance with no impact on trading.  The backtester is offline historical replay; yfinance is fine.

Three concrete adapters all satisfy the same `MarketDataPort` Protocol (`ports.py:60-117`):
- `MarketDataProvider` (Alpaca) — full-fidelity options + snapshots, 15-min indicative on free tier.
- `SchwabMarketDataProvider` — real-time quotes + Greeks via Schwab Trader API (OAuth 2.0).
- `YahooMarketDataProvider` — yfinance for historical / intraday / spot; **no** options, **no** live bid/ask.

The factory `build_market_data_provider(...)` walks an env-var priority chain and returns the right concrete class.  Adding a fourth provider (IBKR, Polygon, Tradier, etc.) is one branch in the factory plus one new adapter file.

## 2. Routing logic

```text
build_market_data_provider(*, alpaca_*, surface=None) -> MarketDataProvider

Priority order:
  1. MARKET_DATA_PROVIDER_<SURFACE>      (e.g. _LIVE, _WATCHLIST, _BACKTEST)
  2. MARKET_DATA_PROVIDER                (global default)
  3. "alpaca"                            (hardcoded fallback)

Provider names are case-insensitive: "Schwab", "SCHWAB", "schwab" all valid.
Empty per-surface var falls through to global. Unknown names → fall through to next level.
```

Recognised provider strings: `alpaca` (default), `schwab`, `yahoo`.

## 3. Reference Python Implementation

### `trading_agent/market_data_factory.py:53-91`
```python
def _resolve_provider_name(surface: Optional[str]) -> str:
    surface_norm = (surface or "").strip().upper()
    if surface_norm:
        per_surface_var = f"MARKET_DATA_PROVIDER_{surface_norm}"
        per_surface = os.environ.get(per_surface_var, "").strip().lower()
        if per_surface:
            if per_surface not in _KNOWN_PROVIDERS:
                logger.warning(...)
            else:
                return per_surface
    global_var = os.environ.get("MARKET_DATA_PROVIDER", "").strip().lower()
    if global_var in _KNOWN_PROVIDERS:
        return global_var
    return "alpaca"


def build_market_data_provider(*, alpaca_api_key, alpaca_secret_key,
                                alpaca_data_url=..., alpaca_base_url=...,
                                surface=None) -> MarketDataProvider:
    provider_name = _resolve_provider_name(surface)
    if provider_name == "schwab":
        from trading_agent.market_data_schwab import SchwabMarketDataProvider
        return SchwabMarketDataProvider(...)
    if provider_name == "yahoo":
        from trading_agent.market_data_yahoo import YahooMarketDataProvider
        return YahooMarketDataProvider(...)
    return MarketDataProvider(...)
```

### Callsites — three surfaces wired through the factory

| File:line | Surface arg |
|---|---|
| `agent.py:156` | `surface="live"` |
| `streamlit/watchlist_ui.py:_get_data_provider` | `surface="watchlist"` |
| `streamlit/live_monitor.py:_is_market_open_cached` | `surface="live"` (so dashboard market-open badge matches the agent) |

The `_fetch_account_cached` callsite (`live_monitor.py:1128`) **intentionally stays Alpaca-direct** — `get_account_info()` is Alpaca-specific account state with no Schwab/Yahoo equivalent shape, and the executor talks to Alpaca regardless of which market-data provider is selected.

### Schwab OAuth — the only adapter that needs an auth dance

Schwab uses OAuth 2.0 authorization-code flow.  Access tokens last **30 min** (refreshed silently); refresh tokens last **7 days, absolute** (re-auth weekly).

`SchwabOAuth.get_access_token()` is called on every API request.  It:
1. Loads tokens from `~/.schwab_tokens.json` (override via `SCHWAB_TOKEN_PATH`).
2. If access token has < `REFRESH_LEEWAY_SEC` (120s) remaining → POST to `https://api.schwabapi.com/v1/oauth/token` with `grant_type=refresh_token`, persists the rotated refresh token atomically (temp+rename per CLAUDE.md soft rule).
3. If the refresh token has expired → raises `RuntimeError` with the exact CLI command to re-auth.

The one-time login is invoked manually:
```bash
python -m trading_agent.schwab_oauth login
```

### Refresh retry policy (added 2026-05-06)

Pre-2026-05-06 `_refresh` was one-shot: it raised `RuntimeError` on the first non-200 response. A single transient 5xx or network blip during the 30-min refresh window halted the agent until manual intervention. For live trading that's an unacceptable single point of failure — a 30-second retry budget is cheap relative to watching the operator's screen forever.

```text
REFRESH_MAX_ATTEMPTS    = 3
REFRESH_BACKOFF_BASE    = 2.0      # 2s after attempt 1, 4s after attempt 2
REFRESH_BACKOFF_CAP_S   = 8.0      # cap so total never exceeds cycle runway
```

Decision tree per attempt:
- **200** → success, ingest token, persist atomically.
- **429 OR 5xx** → transient, sleep and retry.
- **400 / 401 / other 4xx** → PERMANENT (refresh-token revoked, client credentials wrong, scope issue) — short-circuit immediately and raise `RuntimeError("...permanently failed...refresh-token likely revoked; re-auth required.")`. No retries wasted on permanent failures.
- **Network exception** (RequestException) → transient, sleep and retry.
- **All attempts exhausted** → raise `RuntimeError("...failed after N attempts...")`. Surfaced to the operator via the Schwab adapter's WARNING log path; in practice the agent's downstream "no data" symptoms (price=0, regime=unknown) cause cycle skips that the dashboard's staleness beacon then surfaces.

```python
# trading_agent/schwab_oauth.py:334-462 — see source for the full
# retry loop. Worst-case latency at all 3 attempts: 15s × 3 + (2s + 4s)
# backoff = ~51s. Inside the cycle's 270s hard guard.
```

### Snapshot + position-fetch retry (added 2026-05-06)

`MarketDataProvider.fetch_batch_snapshots` and `PositionMonitor.fetch_open_positions` both got 2-attempt retry with brief backoff on the same date. Rationale:

- `fetch_batch_snapshots` — 09:30 ET market-open surge regularly produced 10s read-timeouts on the snapshot endpoint, blanking regime classification for the entire 5-min cycle. `SNAPSHOT_RETRY_ATTEMPTS=2`, `SNAPSHOT_RETRY_BACKOFF_S=0.5`.
- `fetch_open_positions` — preserves the fail-closed `None` return on retry exhaustion (so the dedup gate still fails closed) but recovers from a single 100ms TCP reset instead of skipping the cycle. `POSITION_FETCH_RETRY_ATTEMPTS=2`, `POSITION_FETCH_RETRY_BACKOFF_S=0.5`.

Both also emit `journal_kb.log_warning` rows (`action="warning"`) on retry exhaustion so the dashboard surfaces the connectivity issue without requiring a log scrape. See skill 19 for the warning-row schema.

### Symbol-format quirk — Schwab vs everyone else

Schwab option symbols are **space-padded to 21 chars**: `"AMZN  220617C03170000"` (root padded to 6).  Standard OCC compact: `"AMZN220617C03170000"`.  The adapter exposes `to_schwab_symbol()` / `from_schwab_symbol()` translators so the agent always speaks compact OCC; conversion happens at the adapter boundary in `fetch_option_chain` and `fetch_option_quotes`.

```python
# trading_agent/market_data_schwab.py:75-100
def to_schwab_symbol(occ: str) -> str:
    """Compact OCC → Schwab-padded.  Pass-through for non-options."""
    m = _OCC_COMPACT_RE.match(occ or "")
    if not m:
        return occ
    root, ymd, cp, strike = m.groups()
    return f"{root:<6}{ymd}{cp}{strike}"


def from_schwab_symbol(schwab: str) -> str:
    """Schwab-padded → compact.  Inverse."""
    m = _OCC_SCHWAB_RE.match(schwab or "")
    if not m:
        return (schwab or "").strip()
    root, ymd, cp, strike = m.groups()
    return f"{root}{ymd}{cp}{strike}"
```

## 4. Edge Cases / Guardrails

- **No tokens on disk** — `SchwabOAuth.get_access_token()` raises `RuntimeError("No Schwab tokens on disk yet…")`.  `SchwabMarketDataProvider._get` catches this, logs a one-time WARNING with the exact CLI fix (`python -m trading_agent.schwab_oauth login`), then drops to DEBUG so the log isn't spammed every cycle.  All Schwab-routed surfaces will see no data until login completes.
- **Refresh token expired (>7 days)** — Same `RuntimeError` path.  Operator must re-run the login flow.
- **Refresh-time transient failure (5xx, 429, network blip)** — `_refresh` retries up to 3 times with exponential backoff (2s/4s, capped at 8s).  Pre-2026-05-06 a single 5xx halted the agent until manual restart; the retry budget is tuned to recover from typical gateway hiccups without racing the cycle's 270s hard guard.  See §3 "Refresh retry policy" above.
- **Refresh-time permanent failure (4xx ≠ 429)** — Refresh-token revoked, client credentials wrong, scope misconfigured.  `_refresh` short-circuits on the first 4xx (no retries) and raises `RuntimeError("...permanently failed...refresh-token likely revoked; re-auth required.")`.  Operator runs the login CLI and the next cycle resumes.
- **401 mid-flight** — `_get` clears the cached token (`oauth._tokens = None`), refreshes once, retries the request.  If the retry also fails, returns `None` and the caller decides how to degrade.
- **Snapshot batch timeout (Alpaca)** — `fetch_batch_snapshots` retries once with 0.5s backoff (typical 09:30 ET market-open queue surge).  After exhaustion returns `{}` and the agent's per-cycle skip behaviour kicks in.  Same retry shape on `fetch_open_positions` (`PositionMonitor`), preserving the fail-closed `None` semantics so the dedup gate stays safe.
- **Schwab IV is in percent** — `volatility=22.31` means 22.31%, not 0.2231.  The adapter divides by 100 in `_normalize_contract` so the rest of the agent sees decimals (matches Alpaca/yfinance convention).
- **Schwab option symbols have spaces** — Always translate at the adapter boundary; never let a padded symbol leak into the executor or journal.
- **Yahoo provider on a live surface** — `fetch_option_chain` returns `None` and `fetch_option_quotes` returns `{}`.  The agent's cycle will skip every ticker (no chains → no plans).  Yahoo is intentionally unsupported for the live surface; stick to Alpaca/Schwab there.
- **Yahoo `get_underlying_bid_ask` returns `None`** — Yahoo's free feed has no real-time NBBO.  The risk manager treats `None` as "no quote" (the liquidity guardrail soft-passes).  Use Yahoo only on surfaces that don't need this gate.
- **Boolean params** — Schwab's URL parser accepts only lowercase `"true"`/`"false"` in query strings, not Python's `True`/`False` (which `requests` serializes as `"True"`/`"False"`).  The adapter passes booleans as lowercase strings explicitly (`market_data_schwab.py:fetch_intraday_bars`).
- **Empty `candles` from `/pricehistory`** — Adapter logs a WARNING with the exact request params so operators can see whether the empty response is a bad period/frequency combo, an unavailable symbol, or extended-hours filtering.
- **Account-info path stays Alpaca-direct** — `live_monitor.py:_fetch_account_cached` never goes through the factory.  Schwab has no equivalent endpoint shape and the executor still talks to Alpaca; routing through the factory would just break the broker-state cache.

## 5. Cross-References

- `00_sdlc_and_conventions.md` — explains the hexagonal port pattern and why the agent core depends on `MarketDataPort` instead of any concrete provider class.
- `15_backtest_live_parity.md` — the backtester wires through `decide()` independently of market-data provider routing; the `MARKET_DATA_PROVIDER_BACKTEST` env var is reserved for a future enhancement (the backtester currently uses its own historical port).
- `18_order_submission_idempotency.md` — companion safety story on the EXECUTION side; together skills 16/18 cover all retry paths against external vendors.
- `19_journal_schema.md` — `action="warning"` schema used to surface retry-exhausted vendor failures to the dashboard.
- Setup steps: see [`README.md`](../../README.md) §"Multi-provider market data" for env-var examples and the Schwab OAuth setup walkthrough.

---

*Last verified against repo HEAD on 2026-05-06.*
