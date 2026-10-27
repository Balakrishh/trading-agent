"""macro_vol_gates.py — macro-event and implied-vs-realized entry gates (skill 68).

Added 2026-10-27, live on the paper account from the start (the operator
asked for live rules on paper rather than a shadow period). Both gates
apply only to **short-premium** structures — the ones that lose when the
underlying moves more than the options priced:

1. **Macro event guard.** No new short-premium trade on an equity ETF
   whose life (today → expiration) is at most ``macro_event_max_dte``
   days and contains a ``macro_event_kinds`` release that has not
   happened yet. ``scripts/research/macro_vrp_credit_study.py`` (2025 →
   2026-10, April 2025 excluded): jobs-report days moved SPY / QQQ / IWM
   / XLK / SMH 1.4–1.9× a normal day (median |r|) and 1.2–2.0× in
   variance; CPI days only slightly (variance not elevated); FOMC days
   not at all close-to-close; TLT / GLD / XLU barely react to any. One
   1.5–2× day is ~10 % of a 21-session trade's variance but 25–40 % of a
   5-session one — so only short trades are gated, only for NFP by
   default, and bonds / metals / commodities are exempt.
2. **Implied vs realized (variance risk premium).** Short premium pays
   when implied vol exceeds what the underlying actually does. SPY
   2007 → 2026: with VIX / trailing 20-day realized vol under 1.0, SPY
   stayed inside its ±1σ (VIX) 21-session range 77–82 % of the time;
   above 1.5 it did 87–88 %. A short-premium trade needs the ticker's
   30-day ATM IV (yesterday's ``vol_surface`` row, skill 66) ≥
   ``vrp_min_ratio`` × its 20-day realized vol. Missing IV or RV never
   blocks (sentinel: ``vrp_available`` False).

Debit spreads and calendars are long premium and never gated here.
Every journalled plan carries the context (``macro_events_in_trade``,
``macro_next_event``, ``iv_atm_30d``, ``iv_rv_ratio``) so the rules can
be re-tested on outcomes.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from trading_agent import macro_calendar as mc
from trading_agent.market_state import (BEAR_CALL, BOUNCE_BULL_PUT, BULL_PUT, IRON_BUTTERFLY,
                                        IRON_CONDOR, MEAN_REVERSION, NON_EQUITY_TICKERS)

SHORT_PREMIUM_STRATEGIES = frozenset({BULL_PUT, BEAR_CALL, IRON_CONDOR, IRON_BUTTERFLY,
                                      MEAN_REVERSION, BOUNCE_BULL_PUT})
VOL_SURFACE_PATH = Path(os.environ.get("TRADING_AGENT_VOL_SURFACE",
                                       "trade_journal/vol_surface.jsonl"))
IV_MAX_AGE_DAYS = 4              # a Friday row still serves Monday / a holiday Tuesday


def is_short_premium(strategy: str) -> bool:
    return strategy in SHORT_PREMIUM_STRATEGIES


def pending_events(today: date, expiration: date, now_et: datetime,
                   kinds: Iterable[str] = mc.KINDS,
                   events: Optional[List[mc.MacroEvent]] = None) -> List[mc.MacroEvent]:
    """Releases from now to ``expiration`` that have not happened yet (a
    release earlier today is already in the price)."""
    out = []
    for e in mc.events_between(today, expiration, events, kinds):
        if e.day == today and now_et.time() >= e.release_et:
            continue
        out.append(e)
    return out


def macro_event_failure(ticker: str, strategy: str, expiration: date, now_et: datetime,
                        kinds: Iterable[str], max_dte: int,
                        events: Optional[List[mc.MacroEvent]] = None) -> Optional[str]:
    """Reason a short-premium entry is blocked, else None."""
    if not is_short_premium(strategy) or ticker in NON_EQUITY_TICKERS or max_dte <= 0:
        return None
    today = now_et.date()
    dte = (expiration - today).days
    if dte > max_dte:
        return None
    hits = pending_events(today, expiration, now_et, kinds, events)
    if not hits:
        return None
    return (f"macro_event_guard ({', '.join(e.label() for e in hits)} inside a {dte}-day "
            f"{strategy}; short premium ≤ {max_dte} DTE waits until after the release)")


# ── Implied vs realized ───────────────────────────────────────────────

_iv_cache: Dict[Tuple[str, float], Dict[str, Tuple[float, date]]] = {}


def _atm_iv_index(path: Path) -> Dict[str, Tuple[float, date]]:
    """ticker → (30-day ATM IV, row date) from the newest vol_surface row
    per ticker; cached per file mtime (the agent re-reads once a cycle)."""
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return {}
    if key in _iv_cache:
        return _iv_cache[key]
    out: Dict[str, Tuple[float, date]] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    day = date.fromisoformat(row["date"])
                except (ValueError, KeyError, TypeError):
                    continue
                iv = None
                for ex in row.get("expiries") or []:
                    if ex.get("target_dte") == 30 and ex.get("atm_iv"):
                        iv = float(ex["atm_iv"])
                if iv is None and (row.get("term") or {}).get("iv30"):
                    iv = float(row["term"]["iv30"])
                t = row.get("ticker")
                if iv and t and (t not in out or day >= out[t][1]):
                    out[t] = (iv, day)
    except OSError:
        return {}
    _iv_cache.clear()
    _iv_cache[key] = out
    return out


def latest_atm_iv(ticker: str, today: date, path: Optional[Path] = None,
                  max_age_days: int = IV_MAX_AGE_DAYS) -> Optional[float]:
    hit = _atm_iv_index(Path(path or VOL_SURFACE_PATH)).get(ticker)
    if hit is None or (today - hit[1]).days > max_age_days:
        return None
    return hit[0]


def iv_rv_ratio(iv: Optional[float], rv: Optional[float]) -> Optional[float]:
    if not iv or not rv or rv <= 0:
        return None
    return round(iv / rv, 3)


def vrp_failure(strategy: str, iv: Optional[float], rv: Optional[float],
                min_ratio: float) -> Optional[str]:
    if not is_short_premium(strategy) or min_ratio <= 0:
        return None
    ratio = iv_rv_ratio(iv, rv)
    if ratio is None or ratio >= min_ratio:
        return None
    return (f"vrp_below_min (30d ATM IV {iv:.1%} / 20d realized {rv:.1%} = {ratio:.2f} "
            f"< {min_ratio:g}: premium too cheap to sell)")


def context_fields(ticker: str, expiration: Optional[date], now_et: datetime,
                   iv: Optional[float], rv: Optional[float],
                   events: Optional[List[mc.MacroEvent]] = None) -> Dict[str, Any]:
    """Journal fields for a plan (whatever its structure)."""
    evs = events if events is not None else mc.all_events()
    today = now_et.date()
    nxt = mc.next_event(today + timedelta(days=1), evs)
    later_today = pending_events(today, today, now_et, mc.KINDS, evs)
    nxt = later_today[0] if later_today else nxt
    inside = (pending_events(today, expiration, now_et, mc.KINDS, evs)
              if expiration is not None else [])
    ratio = iv_rv_ratio(iv, rv)
    return {
        "macro_next_event": nxt.label() if nxt else None,
        "macro_events_in_trade": [e.label() for e in inside],
        "iv_atm_30d": iv,
        "iv_rv_ratio": ratio,
        "vrp_available": ratio is not None,
    }
