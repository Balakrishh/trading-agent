"""book_delta.py — beta-weighted net delta of the book, shadow budget (E4).

Backlog 11.1 / E4 (2026-10-26). Every entry from 10-14 to 10-22 was a
bearish put debit (IWM, TLT ×3, XLU, GLD ×2), and the account's daily P&L
had ≈ 0 correlation with SPY: the book was one directional bet that no
count-based cap saw. This measures the tilt each cycle:

    share delta (per underlying) = Σ leg delta × signed qty × 100
    SPY-equivalent $             = share delta × spot × beta to SPY
    net beta delta %             = Σ SPY-equivalent $ / equity

and, in ``shadow`` mode, journals (``net_delta_shadow``) every candidate
that a cap of ``max_net_beta_delta_pct`` would block: the book is past the
budget and the candidate leans the same way. Neutral structures (delta
≈ 0 by construction) are never blocked. Nothing is blocked for real —
``live`` is not built; the shadow rows decide whether it should be.

Leg deltas come from the monitor's mid re-mark quotes (Schwab's
QuoteOption ``delta``), betas from the cached daily price history: no
extra API calls. A leg without a delta makes that underlying "unknown";
the book number then covers the rest and lists what it left out.

State (the latest reading + today's shadow keys, so a candidate is
journalled once a day although the process restarts every cycle) lives
in ``trade_journal/book_delta.json``.
"""
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

STATE_PATH = Path(os.environ.get("TRADING_AGENT_BOOK_DELTA", "trade_journal/book_delta.json"))
BETA_SESSIONS = 60
_OCC = re.compile(r"^([A-Z]{1,6})\d{6}[CP]\d{8}$")


def underlying_of(symbol: str) -> Optional[str]:
    m = _OCC.match(symbol or "")
    return m.group(1) if m else None


def share_deltas(legs: Iterable[Any], quotes: Dict[str, Dict]) -> Dict[str, Optional[float]]:
    """underlying → net share delta of its option legs (Σ Δ × qty × 100);
    None when any of its legs has no delta in ``quotes``."""
    out: Dict[str, Optional[float]] = {}
    for leg in legs:
        u = underlying_of(getattr(leg, "symbol", ""))
        if not u:
            continue
        d = (quotes.get(leg.symbol) or {}).get("delta")
        if d is None or out.get(u, 0.0) is None:
            out[u] = None
            continue
        out[u] = out.get(u, 0.0) + float(d) * int(leg.qty) * 100
    return {u: (None if v is None else round(v, 2)) for u, v in out.items()}


def beta_from_closes(closes: List[float], spy: List[float], n: int = BETA_SESSIONS) -> Optional[float]:
    """OLS beta of daily log returns over the last ``n`` sessions (both
    series aligned at the end). None with fewer than n + 1 closes."""
    if len(closes) < n + 1 or len(spy) < n + 1:
        return None
    try:
        a = [math.log(closes[i] / closes[i - 1]) for i in range(len(closes) - n, len(closes))]
        b = [math.log(spy[i] / spy[i - 1]) for i in range(len(spy) - n, len(spy))]
    except (ValueError, ZeroDivisionError):
        return None
    ma, mb = sum(a) / n, sum(b) / n
    var = sum((x - mb) ** 2 for x in b)
    if var <= 0:
        return None
    return round(sum((x - ma) * (y - mb) for x, y in zip(a, b)) / var, 3)


def book_reading(deltas: Dict[str, Optional[float]], spots: Dict[str, Optional[float]],
                 betas: Dict[str, Optional[float]], equity: float) -> Dict[str, Any]:
    """SPY-equivalent dollar delta per underlying and for the book."""
    rows, total, unknown = {}, 0.0, []
    for u, d in sorted(deltas.items()):
        spot, beta = spots.get(u), betas.get(u)
        if d is None or not spot or beta is None:
            unknown.append(u)
            rows[u] = {"share_delta": d, "spot": spot, "beta": beta, "spy_equiv": None}
            continue
        eq = round(d * spot * beta, 2)
        total += eq
        rows[u] = {"share_delta": d, "spot": spot, "beta": beta, "spy_equiv": eq}
    return {"spy_equiv_dollars": round(total, 2),
            "pct_of_equity": round(total / equity, 4) if equity > 0 else None,
            "unknown": unknown, "by_underlying": rows}


def plan_share_delta(plan: Any) -> Optional[float]:
    """Share delta of one contract of a plan (buy +, sell −); None when a
    leg has no delta."""
    total = 0.0
    for leg in getattr(plan, "legs", []) or []:
        d = getattr(leg, "delta", None)
        if d is None:
            return None
        total += float(d) * (1 if leg.action == "buy" else -1) * 100
    return round(total, 2)


def shadow_verdict(book_pct: Optional[float], candidate_delta: Optional[float],
                   budget_pct: float) -> Optional[str]:
    """Reason a live cap would block the candidate, else None: the book is
    past ±budget and the candidate's delta has the same sign."""
    if book_pct is None or candidate_delta is None or budget_pct <= 0:
        return None
    if abs(book_pct) <= budget_pct or abs(candidate_delta) < 1.0:
        return None
    if (book_pct > 0) == (candidate_delta > 0):
        side = "long" if book_pct > 0 else "short"
        return (f"net_delta_budget (book {book_pct:+.0%} of equity {side}, "
                f"> {budget_pct:.0%}; candidate Δ {candidate_delta:+.0f} sh/contract adds to it)")
    return None


def load_state(path: Optional[Path] = None) -> Dict[str, Any]:
    try:
        return json.loads((path or STATE_PATH).read_text())
    except (OSError, ValueError):
        return {}


def save_state(state: Dict[str, Any], path: Optional[Path] = None) -> None:
    path = path or STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(path)


def record_reading(reading: Dict[str, Any], day: str, path: Optional[Path] = None) -> Dict[str, Any]:
    """Store the latest reading; today's shadow keys survive, older ones reset."""
    state = load_state(path)
    keys = state.get("shadow_keys") if state.get("day") == day else []
    state = {"day": day, "as_of_utc": datetime.now(timezone.utc).isoformat(),
             "reading": reading, "shadow_keys": keys or []}
    save_state(state, path)
    return state


def first_shadow_today(key: str, day: str, path: Optional[Path] = None) -> bool:
    """True (and remembered) the first time ``key`` is seen on ``day``."""
    state = load_state(path)
    if state.get("day") != day:
        state = {**state, "day": day, "shadow_keys": []}
    keys = list(state.get("shadow_keys") or [])
    if key in keys:
        return False
    keys.append(key)
    state["shadow_keys"] = keys
    save_state(state, path)
    return True
