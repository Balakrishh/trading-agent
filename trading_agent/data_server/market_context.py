"""Market-context routes for the data server (skill 47 §3.6, skill 68).

Added 2026-10-27 with the macro-event, implied-vs-realized and
credit-stress inputs, so every data point the agent trades on is also
readable from the one local server:

* ``/macro-events``      — FOMC / CPI / NFP releases ahead + calendar coverage;
* ``/credit-stress``     — HYG/IEF % under its 50-day average, the CAUTION
  threshold and recent history;
* ``/vol/{ticker}``      — 30-day ATM IV (latest vol-surface row), 20-day
  realized vol, IV/RV and whether a short-premium trade passes the floor;
* ``/gamma/{ticker}``    — latest recorded dealer-gamma row (16:05 ET timer);
* ``/market-context``    — the agent's last market-state snapshot plus the
  four above for SPY in one call.

Like the research routes, the logic lives in its own modules
(``macro_calendar``, ``market_state``, ``macro_vol_gates``,
``gamma_exposure``); this file only exposes it. Daily history comes
from the provider's ``fetch_historical_prices`` (yfinance, not Schwab).
Read-only: the only files read are the agent's snapshots and recorder
rows; nothing is written.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

HISTORY_DAYS = 200
CREDIT_HISTORY_POINTS = 20


def _now_et() -> datetime:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)


@dataclass
class ContextSources:
    """Injectable readers (tests hand in stubs)."""
    history: Callable[[str], Any]                       # ticker → daily OHLCV frame
    atm_iv: Callable[[str, date], Optional[float]]
    gamma_rows: Callable[[], Dict[str, Dict[str, Any]]]
    market_state: Callable[[], Optional[Dict[str, Any]]]
    events: Callable[[], List[Any]]
    now_et: Callable[[], datetime] = field(default=_now_et)
    adjusted: Optional[Callable[[str], Any]] = None     # dividend-adjusted (credit pair)


def default_sources(provider: Any) -> ContextSources:
    from trading_agent import gamma_exposure, macro_calendar, macro_vol_gates, market_state

    def _history(t: str) -> Any:
        fetch = getattr(provider, "fetch_historical_prices", None)
        if fetch is None:
            raise RuntimeError("provider has no fetch_historical_prices")
        return fetch(t, period_days=HISTORY_DAYS)

    return ContextSources(history=_history, adjusted=market_state.yfinance_adjusted_history,
                          atm_iv=macro_vol_gates.latest_atm_iv,
                          gamma_rows=gamma_exposure.latest_rows,
                          market_state=market_state.read_state,
                          events=macro_calendar.all_events)


def macro_events(src: ContextSources, days: int) -> Dict[str, Any]:
    from trading_agent import macro_calendar as mc
    from trading_agent.macro_vol_gates import pending_events
    now = src.now_et()
    evs = src.events()
    ahead = pending_events(now.date(), now.date() + timedelta(days=days), now, mc.KINDS, evs)
    return {
        "as_of_et": now.isoformat(timespec="minutes"),
        "days": days,
        "events": [{"kind": e.kind, "date": e.day.isoformat(),
                    "release_et": e.release_et.strftime("%H:%M"),
                    "days_away": (e.day - now.date()).days} for e in ahead],
        "next": ahead[0].label() if ahead else None,
        "coverage_warnings": mc.coverage_warnings(now.date(), evs),
    }


def credit_stress(src: ContextSources) -> Dict[str, Any]:
    from trading_agent.market_state import (CREDIT_PAIR, CREDIT_SMA, MarketStateConfig,
                                            _aligned_ratio, credit_stress_pct)
    adj = src.adjusted or src.history
    ratio = _aligned_ratio(adj(CREDIT_PAIR[0]), adj(CREDIT_PAIR[1]))
    cfg = MarketStateConfig()
    hist = []
    for k in range(min(CREDIT_HISTORY_POINTS, max(len(ratio) - CREDIT_SMA + 1, 0)), 0, -1):
        hist.append(credit_stress_pct(ratio[:len(ratio) - k + 1]))
    now = credit_stress_pct(ratio)
    return {
        "pair": "/".join(CREDIT_PAIR), "sma_days": CREDIT_SMA,
        "stress_pct": now,
        "caution_at_pct": cfg.caution_credit_stress,
        "caution_exit_below_pct": cfg.caution_exit_credit_stress,
        "triggers_caution": now is not None and now >= cfg.caution_credit_stress,
        "recent": hist,
        "note": "% the HYG/IEF ratio sits under its 50-day average; negative = above it",
    }


def vol(ticker: str, src: ContextSources, min_ratio: Optional[float] = None) -> Dict[str, Any]:
    from trading_agent.macro_vol_gates import iv_rv_ratio
    from trading_agent.shadow_pop import realized_vol
    now = src.now_et()
    notes: List[str] = []
    rv = None
    try:
        df = src.history(ticker)
        col = "Close" if "Close" in df.columns else "close"
        rv = realized_vol([float(x) for x in df[col].dropna().tolist()])
    except Exception as exc:                            # noqa: BLE001 — missing input → null + note
        notes.append(f"realized vol unavailable: {str(exc)[:120]}")
    iv = src.atm_iv(ticker, now.date())
    if iv is None:
        notes.append("no vol-surface row in the last 4 days (15:50 ET timer)")
    if min_ratio is None:
        from trading_agent.strategy_presets import load_active_preset
        try:
            min_ratio = float(getattr(load_active_preset(), "vrp_min_ratio", 1.0))
        except Exception:                               # noqa: BLE001 — preset unreadable → default
            min_ratio = 1.0
    ratio = iv_rv_ratio(iv, rv)
    return {
        "ticker": ticker, "iv_atm_30d": iv,
        "rv_20d": round(rv, 4) if rv is not None else None,
        "iv_rv_ratio": ratio, "vrp_min_ratio": min_ratio,
        "short_premium_allowed": None if ratio is None or min_ratio <= 0 else ratio >= min_ratio,
        "notes": notes, "as_of_et": now.isoformat(timespec="minutes"),
    }


def gamma(ticker: str, src: ContextSources) -> Dict[str, Any]:
    row = src.gamma_rows().get(ticker)
    return {"ticker": ticker, "gamma": row,
            "note": None if row else "not recorded (16:05 ET timer covers SPY, QQQ, IWM)"}


def market_context(src: ContextSources) -> Dict[str, Any]:
    out: Dict[str, Any] = {"notes": []}
    for key, fn in (("market_state", src.market_state),
                    ("macro_events", lambda: macro_events(src, 30)),
                    ("credit_stress", lambda: credit_stress(src)),
                    ("spy_vol", lambda: vol("SPY", src)),
                    ("gamma", lambda: {t: src.gamma_rows().get(t) for t in ("SPY", "QQQ", "IWM")})):
        try:
            out[key] = fn()
        except Exception as exc:                        # noqa: BLE001 — one source must not sink the rest
            logger.warning("market context %s failed: %s", key, exc)
            out[key] = None
            out["notes"].append(f"{key} unavailable: {str(exc)[:120]}")
    return out
