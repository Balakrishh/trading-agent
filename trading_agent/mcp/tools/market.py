"""Market-data + alerts read wrappers (skill 48 §2).

Delegates to the skill-47 data server surface when it's reachable
(preferred — one place owns Schwab OAuth), else to the in-process
market data provider as fallback. Both paths are read-only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import os
import re
import urllib.request
import urllib.error
import json


def _data_server_url() -> Optional[str]:
    """Return the data-server base URL if configured, else None."""
    base = os.environ.get("SCHWAB_API_BASE_URL", "").strip()
    return base or None


def _data_server_request(path: str, body: Optional[Dict[str, Any]] = None,
                         timeout: float = 5) -> Optional[Dict[str, Any]]:
    """GET ``path`` (or POST ``body`` as JSON — still a read, e.g.
    ``/quotes``) on the data server. None when unconfigured, unreachable
    or the server rejects the request."""
    base = _data_server_url()
    if not base:
        return None
    key = os.environ.get("SCHWAB_API_SERVER_KEY", "")
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(f"{base.rstrip('/')}/{path.lstrip('/')}",
                                 data=data)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _data_server_get(path: str, timeout: float = 5) -> Optional[Dict[str, Any]]:
    return _data_server_request(path, timeout=timeout)


# Compact OCC option symbol: root, YYMMDD, C/P, strike × 1000 (8 digits).
_OCC_RE = re.compile(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$")


def get_quote(symbol: str) -> Dict[str, Any]:
    """Current mark for an equity or option symbol.

    Equities use the data server's ``/price/{ticker}``; option symbols
    (compact OCC, e.g. ``GLD261106P00379000``) use ``/quotes`` and
    return bid / ask / mid with ``price`` = mid. ``/price`` rejects
    option symbols, which used to surface as "data server unreachable".
    """
    if not symbol:
        raise ValueError("symbol is required")
    sym = symbol.upper().strip()
    if _OCC_RE.match(sym):
        remote = _data_server_request("quotes", {"symbols": [sym]})
        if remote is not None:
            q = (remote.get("quotes") or {}).get(sym)
            if not q:
                return {"symbol": sym, "price": None, "source": "data_server",
                        "note": "no quote for this contract"}
            return {"symbol": sym, "price": q.get("mid"), "bid": q.get("bid"),
                    "ask": q.get("ask"), "mid": q.get("mid"),
                    "source": "data_server"}
    else:
        remote = _data_server_get(f"price/{sym}")
        if remote is not None:
            return remote
    # Fallback: in-process. Best-effort — market_data_factory build
    # requires credentials the MCP process may not have.
    return {"symbol": sym, "price": None,
            "source": "unavailable", "note": "data server unreachable"}


def get_chain(
    underlying: str,
    expiration: Optional[str] = None,
    option_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Option chain — thin passthrough to the data server."""
    if not underlying:
        raise ValueError("underlying is required")
    q = []
    if expiration:
        q.append(f"expiration={expiration}")
    if option_type:
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        q.append(f"option_type={option_type}")
    suffix = ("?" + "&".join(q)) if q else ""
    remote = _data_server_get(f"chain/{underlying.upper()}{suffix}")
    if remote is not None:
        return remote
    return {"underlying": underlying.upper(),
            "source": "unavailable",
            "note": "data server unreachable"}


def get_market_status() -> Dict[str, Any]:
    """Whether the market is currently open (US equity session)."""
    remote = _data_server_get("market-status")
    if remote is not None:
        return remote
    try:
        from trading_agent.market_hours import is_within_market_hours
        return {"open": bool(is_within_market_hours()),
                "source": "in-process"}
    except Exception as exc:  # noqa: BLE001
        return {"open": None, "source": "unavailable", "error": str(exc)}


def get_market_state() -> Dict[str, Any]:
    """The agent's latest whole-market risk state (skill 58).

    Reads ``trade_journal/market_state.json`` (written every cycle):
    state, reasons, inputs (SPY trend, VIX, VIX/VIX3M, breadth), the
    entry gate (size multiplier, allowed strategies, CSP pause) and
    ``age_seconds``. In CAUTION / DEFENSIVE it adds a SPY put-spread
    hedge suggestion sized to half the account — a suggestion to stage
    through /propose, never an order.
    """
    from trading_agent import market_state

    snap = market_state.read_state()
    if snap is None:
        return {"state": None, "source": "unavailable",
                "note": "no snapshot yet — the agent writes one each cycle "
                        "when market_state_enabled is on"}
    out = {**snap, "source": "agent_cycle"}
    spy = (snap.get("inputs") or {}).get("spy_price")
    if snap.get("state") in (market_state.CAUTION, market_state.DEFENSIVE) and spy:
        out["hedge_suggestion"] = market_state.hedge_suggestion(
            float(snap.get("account_balance") or 0.0), float(spy))
    elif snap.get("state") == market_state.CAPITULATION:
        out["hedge_note"] = ("Puts are most expensive at capitulation — reduce "
                             "exposure rather than buying protection now.")
    return out


def get_trading_halt() -> Dict[str, Any]:
    """Whether new entries are paused (kill switch / drawdown governor,
    skill 62): paused, reason, set_by, set_at, and the day / week equity
    baselines. Read-only — pausing and resuming is the operator's CLI:
    ``python -m trading_agent.trading_halt pause --reason … | resume``."""
    from dataclasses import asdict
    from trading_agent import trading_halt as th
    return {**asdict(th.load()), "source": "trading_halt.json"}


def get_fundamentals(ticker: str) -> Dict[str, Any]:
    """Return the fundamentals block for one equity ticker.

    Passthrough to the skill-47 data server's ``/fundamentals/{ticker}``
    route. When the data server is unreachable, returns a structured
    ``{"source": "unavailable"}`` row rather than raising — the LLM
    decides whether to ask the operator to bring the server up.

    Fields (all present as dict keys; individual values may be None):

    - Identity: ``ticker``, ``cusip``, ``description``, ``exchange``,
      ``asset_type``
    - Valuation: ``pe_ratio``, ``peg_ratio``, ``pb_ratio``, ``eps_ttm``,
      ``market_cap``, ``book_value_per_share``
    - Dividend: ``dividend_yield``, ``dividend_amount``,
      ``dividend_date``, ``next_dividend_pay_date``
    - Trading: ``beta``, ``high_52w``, ``low_52w``,
      ``vol_avg_1d``, ``vol_avg_10d``, ``vol_avg_3mo``,
      ``shares_outstanding``, ``float_shares`` (share COUNT — Schwab's
      confusingly-named ``marketCapFloat`` field is actually shares
      floating, not a dollar figure)
    - Profitability: ``roe``, ``roa``, ``gross_margin_ttm``,
      ``net_profit_margin_ttm``, ``operating_margin_ttm``
    - Short interest: ``short_int_to_float``

    Set ``SCHWAB_API_FUNDAMENTALS_INCLUDE_RAW=true`` to have the
    provider include the raw Schwab ``fundamental`` block as
    ``_raw_fundamental`` — useful when a field looks wrong and you
    want to see the exact upstream keys.
    - Provenance: ``as_of`` (ISO-8601 UTC)
    """
    if not ticker:
        raise ValueError("ticker is required")
    remote = _data_server_get(f"fundamentals/{ticker.upper()}")
    if remote is not None:
        return remote
    return {"ticker": ticker.upper(),
            "fundamentals": {},
            "source": "unavailable",
            "note": "data server unreachable"}


# Research routes fetch Alpaca + Yahoo on a cache miss (several seconds).
RESEARCH_TIMEOUT_S = 45


def _research_get(route: str, ticker: str, query: str = "") -> Dict[str, Any]:
    if not ticker:
        raise ValueError("ticker is required")
    t = ticker.upper()
    remote = _data_server_get(f"{route}/{t}{query}", timeout=RESEARCH_TIMEOUT_S)
    if remote is not None:
        return remote
    return {"ticker": t, "source": "unavailable", "note": "data server unreachable"}


def get_news(ticker: str, days: Any = 30, limit: Any = 60) -> Dict[str, Any]:
    """Recent news for a ticker (Alpaca + Yahoo RSS) from the data server.

    Skill-47 ``/news/{ticker}`` passthrough: items newest first with
    ``published_at``, ``source``, ``publisher``, ``title``, ``summary``,
    ``url``; at most ``limit`` items spread across the last ``days`` days
    so one busy week does not hide the event that started a move.
    """
    return _research_get("news", ticker, f"?days={int(days)}&limit={int(limit)}")


def get_research(ticker: str) -> Dict[str, Any]:
    """Analyst estimates + the latest quality-at-lows row for a ticker.

    Skill-47 ``/research/{ticker}`` passthrough. ``analyst_estimates``:
    consensus EPS and 30/90-day revisions, revenue/EPS growth, analyst
    up/down revision counts, mean price target and upside, recommendation
    score now vs 3 months ago (1 strong buy … 5 strong sell),
    ``revision_trend``. ``quality_lows``: the scan's newest row (stage,
    price features, drop cause, Claude ``llm_cause``) or None.
    """
    return _research_get("research", ticker)


def get_sentiment(ticker: str) -> Dict[str, Any]:
    """Sentiment roll-up for a ticker from the data server.

    Skill-47 ``/sentiment/{ticker}`` passthrough: Claude's stored
    drop-cause label (from the 15:40 ET quality-at-lows scan, names at
    watch or later only), the rule-based cause, analyst stance
    (revision trend, recommendation drift, target upside) and news flow
    (items in the last 7 days vs the 23 before). No LLM runs per call.
    """
    return _research_get("sentiment", ticker)


def get_recent_alerts(hours: Any = 24) -> Dict[str, Any]:
    """Read the ExceptionMonitor's recent-events log (skill 34).

    ``hours`` is coerced from string/float since some MCP clients JSON-
    serialize numeric arguments as strings.
    """
    try:
        hours = int(hours)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"hours must be an integer, got {hours!r}") from exc
    if hours <= 0:
        raise ValueError("hours must be positive")
    try:
        from trading_agent.journal_reader import JournalReader
        reader = JournalReader()
        return {
            "window_hours": hours,
            "silenced_today": [
                str(e) for e in reader.silenced_exceptions_today()
            ],
            "error_count_today": reader.error_count_today(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"window_hours": hours, "error": str(exc), "alerts": []}
