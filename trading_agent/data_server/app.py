"""FastAPI app exposing Schwab market-data endpoints. Skill 47.

Every route is a thin passthrough to a ``MarketDataPort`` provider —
production wires ``SchwabMarketDataProvider``, tests hand in a stub.
The provider is dependency-injected at ``build_app()`` time so no
env-dependent global state materializes at import.

Route summary (skill 47 §2):
    GET  /health
    GET  /ready
    GET  /price/{ticker}
    GET  /chain/{underlying}?expiration=&option_type=
    POST /quotes                  {"symbols": [...]}
    POST /snapshots               {"tickers": [...]}
    GET  /fundamentals/{ticker}
    GET  /market-status
    GET  /news/{ticker}?days=&limit=   Alpaca + Yahoo RSS (§3.5)
    GET  /research/{ticker}            analyst estimates + quality-at-lows row
    GET  /sentiment/{ticker}           Claude label + analyst stance + news flow

Read-only invariant: this file imports nothing from ``trading_agent.executor``,
``trading_agent.journal_kb``, or anything named ``submit_order`` /
``place_order``. Enforced by conformance test
``test_skill_47_readonly_no_forbidden_imports``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from trading_agent.data_server.auth import build_fastapi_dependency
from trading_agent.data_server.cache import TTLCache
from trading_agent.data_server.config import CacheTTLs, ServerConfig
from trading_agent.data_server import research as _research
from trading_agent.data_server.research import ResearchSources

logger = logging.getLogger(__name__)

# Skill 47 §4 — ticker validation. Uppercase alphanumeric + . (BRK.B) + -.
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


# ── Request models (module-level, skill 47 §3.5) ────────────────────────
# Defining these OUTSIDE build_app() is critical: FastAPI's parameter
# introspection reads the annotation via __annotations__, and when the
# model class lives inside a closure the resolution fails on some
# Pydantic/FastAPI version combos — the parameter falls back to
# query-param treatment (POST returns 422 "field required as query"
# and /openapi.json 500s on schema gen). Module-level definition
# guarantees FastAPI sees the class cleanly.
class QuotesRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=1, max_length=200)


class SnapshotsRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=1, max_length=100)


class MarketDataPort:
    """Protocol the app expects from any injected provider. Documented
    as a duck-typed base so ``SchwabMarketDataProvider`` and test stubs
    both satisfy it structurally without needing to subclass.

    Required methods (all synchronous, return dict-like data):
        get_current_price(ticker: str) -> float
        fetch_option_chain(underlying, expiration_date, option_type) -> list[dict]
        fetch_option_quotes(symbols: list[str]) -> list[dict]
        fetch_batch_snapshots(tickers: list[str]) -> dict[str, dict]

    Optional methods (route returns 501 if absent):
        fetch_fundamentals(ticker: str) -> dict   # skill 47 fundamentals extension
    """


def build_app(
    *,
    provider: Any,
    config: Optional[ServerConfig] = None,
    research_sources: Optional[ResearchSources] = None,
) -> "Any":
    """Construct the FastAPI application.

    ``provider`` must satisfy ``MarketDataPort``. ``config`` defaults
    to ``ServerConfig()`` (auth disabled, default TTLs) for the
    hermetic-test path. ``research_sources`` defaults to the live news /
    analyst / quality-at-lows readers (skill 47 §3.5).
    """
    from fastapi import Body, Depends, FastAPI, HTTPException

    cfg = config or ServerConfig()
    ttls: CacheTTLs = cfg.ttls
    price_cache = TTLCache(ttls.price_sec)
    snapshot_cache = TTLCache(ttls.snapshot_sec)
    # News is cached whatever the Schwab cache switch says: it is not
    # Schwab data, and Alpaca + Yahoo take seconds per ticker.
    news_cache = TTLCache(_research.NEWS_TTL_SEC)
    src = research_sources or _research.default_sources()
    auth_dep = build_fastapi_dependency(cfg.api_key)

    app = FastAPI(
        title="Schwab Data API",
        version="1.0.0",
        description=(
            "Read-only local server exposing SchwabMarketDataProvider "
            "over HTTP. Skill 47."
        ),
    )

    # ── Helper: validate a ticker OR raise 400 ───────────────────────
    def _validate_ticker(ticker: str) -> str:
        t = (ticker or "").strip().upper()
        if not _TICKER_RE.match(t):
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_ticker", "value": ticker},
            )
        return t

    def _upstream_error(exc: Exception, kind: str = "schwab_upstream"):
        """Map a provider-side exception to a 502 with a stable body.

        Distinguishes ``schwab_auth`` (token refresh failed → operator
        needs to re-login) from generic ``schwab_upstream`` (network,
        429, 5xx). The client can decide whether to retry immediately
        or surface a "please re-auth" message.
        """
        msg = str(exc) or exc.__class__.__name__
        low = msg.lower()
        if "auth" in low or "token" in low or "unauthorized" in low:
            return HTTPException(
                status_code=502,
                detail={"error": "schwab_auth", "detail": msg[:200]},
            )
        return HTTPException(
            status_code=502,
            detail={"error": kind, "detail": msg[:200]},
        )

    # ── Routes ────────────────────────────────────────────────────────

    @app.get("/health", tags=["meta"])
    async def health() -> Dict[str, str]:
        """Liveness. No auth. Returns immediately without touching Schwab."""
        return {"status": "ok"}

    @app.get("/ready", tags=["meta"])
    async def ready() -> Dict[str, Any]:
        """Readiness. No auth. Attempts one cheap provider call so the
        caller learns whether Schwab OAuth is currently working. Returns
        503 if the underlying call fails."""
        try:
            # A8 (2026-10-26): providers with ``auth_probe`` (Schwab) must
            # return data — is_market_open alone falls back to Alpaca's
            # clock and hid a dead Schwab token for 10 hours on 10-20.
            auth = getattr(provider, "auth_probe", None)
            if callable(auth):
                ok, detail = auth()
                if not ok:
                    raise RuntimeError(detail)
                return {"status": "ready"}
            # Cheap probe — many providers expose is_market_open().
            probe = getattr(provider, "is_market_open", None)
            if callable(probe):
                probe()
            return {"status": "ready"}
        except Exception as exc:                                # noqa: BLE001
            raise HTTPException(
                status_code=503,
                detail={"error": "not_ready", "detail": str(exc)[:200]},
            )

    @app.get("/price/{ticker}", dependencies=[Depends(auth_dep)])
    async def price(ticker: str) -> Dict[str, Any]:
        t = _validate_ticker(ticker)
        try:
            value = price_cache.get_or_set(
                t, lambda: float(provider.get_current_price(t)),
            )
        except HTTPException:
            raise
        except Exception as exc:                                # noqa: BLE001
            raise _upstream_error(exc)
        return {"ticker": t, "price": value}

    @app.get("/chain/{underlying}", dependencies=[Depends(auth_dep)])
    async def chain(
        underlying: str,
        expiration: str,
        option_type: str = "call",
    ) -> Dict[str, Any]:
        t = _validate_ticker(underlying)
        opt = option_type.lower().strip()
        if opt not in ("call", "put"):
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_option_type", "value": option_type},
            )
        # ISO-date shape check — full parse happens in the provider.
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", expiration or ""):
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_expiration", "value": expiration},
            )
        try:
            contracts = provider.fetch_option_chain(
                underlying=t, expiration_date=expiration, option_type=opt,
            ) or []
        except Exception as exc:                                # noqa: BLE001
            raise _upstream_error(exc)
        return {
            "underlying":  t,
            "expiration":  expiration,
            "option_type": opt,
            "count":       len(contracts),
            "contracts":   contracts,
        }

    @app.post("/quotes", dependencies=[Depends(auth_dep)])
    async def quotes(req: QuotesRequest = Body(...)) -> Dict[str, Any]:
        try:
            data = provider.fetch_option_quotes(req.symbols)
        except Exception as exc:                                # noqa: BLE001
            raise _upstream_error(exc)
        return {"count": len(data) if hasattr(data, "__len__") else 0,
                "quotes": data}

    @app.post("/snapshots", dependencies=[Depends(auth_dep)])
    async def snapshots(req: SnapshotsRequest = Body(...)) -> Dict[str, Any]:
        # Validate each ticker before hitting Schwab.
        tickers = [_validate_ticker(t) for t in req.tickers]
        # Cache key spans the full set — different subsets don't share.
        key = ",".join(sorted(tickers))
        try:
            data = snapshot_cache.get_or_set(
                key, lambda: provider.fetch_batch_snapshots(tickers),
            )
        except HTTPException:
            raise
        except Exception as exc:                                # noqa: BLE001
            raise _upstream_error(exc)
        return {"count": len(data) if hasattr(data, "__len__") else 0,
                "snapshots": data}

    @app.get("/fundamentals/{ticker}", dependencies=[Depends(auth_dep)])
    async def fundamentals(ticker: str) -> Dict[str, Any]:
        t = _validate_ticker(ticker)
        # Fundamentals change at earnings cadence, not by the minute —
        # any caching decision here is decoupled from the price/snapshot
        # TTLs. When the master cache switch is off we return live every
        # time (matches skill 47 §3.4 semantics); when on, we defer to
        # the provider's own instrument-level cache. No new server-side
        # TTLCache instance is created; the LLM watchlist-curation flow
        # calls this at most once per ticker per day.
        try:
            data = provider.fetch_fundamentals(t)
        except AttributeError as exc:
            # Provider doesn't implement fundamentals — surface a 501 so
            # a stub-provider in tests can gracefully answer "not here".
            raise HTTPException(
                status_code=501,
                detail={"error": "not_implemented",
                        "reason": "provider does not implement fetch_fundamentals",
                        "underlying": str(exc)},
            )
        except Exception as exc:                                # noqa: BLE001
            raise _upstream_error(exc)
        return {"ticker": t, "fundamentals": data or {}}

    @app.get("/market-status", dependencies=[Depends(auth_dep)])
    async def market_status() -> Dict[str, Any]:
        # market_hours does not touch Schwab — it's pure calendar math.
        from datetime import datetime, timezone
        from trading_agent.market_hours import is_within_market_hours
        now = datetime.now(timezone.utc)
        try:
            is_open = bool(is_within_market_hours(now))
        except Exception as exc:                                # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail={"error": "internal", "detail": str(exc)[:200]},
            )
        return {"open": is_open, "as_of": now.isoformat()}

    # ── Research (skill 47 §3.5) ─────────────────────────────────────
    # Plain ``def`` routes: the fetchers block for seconds (HTTP to
    # Alpaca / Yahoo), so FastAPI runs them in its thread pool instead
    # of stalling the event loop that serves /price and /quotes.

    def _news_rows(t: str, days: int, limit: int) -> List[Dict[str, Any]]:
        return news_cache.get_or_set(
            f"{t}|{days}|{limit}",
            lambda: _research.news_rows(src.fetch_news(t, days, limit)),
        )

    @app.get("/news/{ticker}", tags=["research"], dependencies=[Depends(auth_dep)])
    def news(ticker: str, days: int = _research.NEWS_DAYS,
             limit: int = _research.NEWS_MAX_ITEMS) -> Dict[str, Any]:
        t = _validate_ticker(ticker)
        if not (1 <= days <= 90) or not (1 <= limit <= 200):
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_range", "days": days, "limit": limit,
                        "allowed": "days 1-90, limit 1-200"},
            )
        try:
            rows = _news_rows(t, days, limit)
        except Exception as exc:                                # noqa: BLE001
            raise _upstream_error(exc, kind="news_upstream")
        return {"ticker": t, "days": days, "count": len(rows), "items": rows,
                "sources": ["alpaca_news", "yahoo_rss"]}

    @app.get("/research/{ticker}", tags=["research"], dependencies=[Depends(auth_dep)])
    def research(ticker: str) -> Dict[str, Any]:
        return _research.research(_validate_ticker(ticker), src)

    @app.get("/sentiment/{ticker}", tags=["research"], dependencies=[Depends(auth_dep)])
    def sentiment(ticker: str) -> Dict[str, Any]:
        t = _validate_ticker(ticker)
        notes: List[str] = []
        try:
            rows = _news_rows(t, _research.NEWS_DAYS, _research.NEWS_MAX_ITEMS)
        except Exception as exc:                                # noqa: BLE001
            logger.warning("[%s] news for sentiment failed: %s", t, exc)
            notes.append(f"news unavailable: {str(exc)[:120]}")
            rows = None
        return _research.sentiment(t, rows, src, notes)

    # Attach caches to app.state so tests can invalidate them.
    app.state.price_cache = price_cache
    app.state.news_cache = news_cache
    app.state.snapshot_cache = snapshot_cache
    app.state.config = cfg
    return app
