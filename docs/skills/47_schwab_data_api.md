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

Seven routes. Every route is a thin passthrough to `SchwabMarketDataProvider` — no new schemas, no new logic beyond auth + caching + rate limiting.

```text
GET  /health                                     — liveness, no auth
GET  /ready                                      — readiness (Schwab token refresh works)
GET  /price/{ticker}                             — current price
GET  /chain/{underlying}?expiration=&option_type=
                                                 — option chain (call|put)
POST /quotes           body {"symbols": [...]}   — batch option/stock quote lookup
POST /snapshots        body {"tickers": [...]}   — batch stock snapshots with indicators
GET  /market-status                              — is_within_market_hours() + next open/close
```

**Explicitly NOT exposed** — CI-verified via `test_readonly_no_forbidden_imports`:

- Order placement (any side, any strategy)
- Account state (positions, balances, orders, buying power)
- Journal reads / writes
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

- `SCHWAB_API_SERVER_KEY` — shared bearer token. Missing → Tailscale-only auth.
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

## 4. Edge Cases / Guardrails

- **Read-only invariant enforced by conformance test.** `test_readonly_no_forbidden_imports` greps `data_server/*.py` for imports of `executor`, `journal_kb`, `submit_order`, `place_order`, `TradingClient`. Any hit fails CI. Belt for the "one bad refactor and we're now placing orders" scenario; suspenders for the "compromised bearer token" scenario.
- **Auth key comparison is constant-time.** `hmac.compare_digest` is used instead of `==` so an attacker can't time-side-channel the correct key from the failure response.
- **401 body is identical for "missing header" and "wrong key".** No information leak: `{"error": "unauthorized"}` verbatim in both cases.
- **Provider is injected, not imported at module load.** `build_app(provider=...)` accepts any `MarketDataPort` — tests hand in a `FakeProvider` with pre-baked returns. No env-dependent global state at module load.
- **Cache is per-instance, not module-global.** `TTLCache` instances live on the app object so two apps in the same process don't share state. Matters for parallel pytest runs.
- **Schwab upstream failures map to 502, NOT 500.** `SchwabAuthError` → 502 with `{"error": "schwab_auth"}`; any other requests-side exception → 502 with `{"error": "schwab_upstream"}`. 500 is reserved for genuine bugs in the server layer itself.
- **Empty batch body → 400, not 200 with empty response.** POST /quotes with `{"symbols": []}` is a client bug (no work asked for). Reject at request-validation time with 400. Same for /snapshots.
- **Ticker parameter validation.** Accept `^[A-Z][A-Z0-9.-]{0,9}$` — uppercase alphanumeric plus `.` (BRK.B) and `-` (rare). Reject anything else at 400. Prevents `../` path traversal and shell-injection footguns even though we never shell out.
- **Access log never contains the API key.** The `Authorization` header is redacted before logging. Failed auth attempts log `<AUTH_FAILED>` in the auth field, not the attempted value.
- **/health has no auth.** Liveness probes from launchd or an external uptime service must work without a key. `/ready` also has no auth — it reveals only Schwab-token validity (a boolean), not any market data.
- **CORS is off by default.** No `Access-Control-Allow-Origin` header. If a browser-side consumer ever appears, add it explicitly then. Curl and Python `requests` don't care about CORS.
- **The trading agent process does NOT depend on this server.** If the server is down, the trading agent's cycle proceeds normally with its own in-process `SchwabMarketDataProvider`. Vice versa: if the trading agent is down, the API server keeps serving. Full crash isolation.

## 5. Cross-References

- `16_market_data_provider_routing.md` — the underlying provider this server wraps. When the surface routing picks Schwab, we're calling the same code path this server exposes.
- `18_order_submission_idempotency.md` — the executor whose contract we deliberately don't re-implement here. The `SERVER_READ_ONLY = True` invariant is symmetric to "the executor is the ONLY place that submits orders."
- `34_exception_monitor.md` — Schwab auth failures inside the server still bubble up to `ExceptionMonitor.record(...)` so the operator's Telegram error channel pages once per failure mode per day.
- `42_portfolio_alert_scheduler.md` — a current in-process consumer of `SchwabMarketDataProvider` that could optionally refactor to use this API (Phase 4 in `docs/plans/schwab_data_api_plan.md`).

---

*Last verified against repo HEAD on 2026-09-27.*
