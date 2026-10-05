"""shadow_pop.py — realized-volatility POP, logged next to the delta POP.

Backlog §3 / §6.8 (2026-10-05). Shadow mode only: nothing here gates a
trade. Every journaled plan gets

* ``pop_delta``   — the probability of profit the scorers use (|Δ| read as
  P(ITM): credit vertical 1 − |Δshort|, iron condor 1 − |Δp| − |Δc|,
  debit vertical |Δ| interpolated at the breakeven);
* ``pop_rv``      — P(profit at expiry) under a zero-drift lognormal with
  the ticker's 20-day realized volatility, measured at the breakeven;
* ``rv_20d``      — that annualized realized volatility.

After 4+ weeks of paper trades, comparing both against outcomes says
whether implied (delta) or realized volatility predicts better.
Calendars have no single-expiry breakeven and get ``None``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


def realized_vol(closes: Sequence[float], window: int = 20) -> Optional[float]:
    """Annualized close-to-close volatility over the last ``window`` returns."""
    c = [float(x) for x in closes if x and x > 0]
    if len(c) < window + 1:
        return None
    rets = [math.log(b / a) for a, b in zip(c[-window - 1:-1], c[-window:])]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252)


def prob_above(spot: float, level: float, dte: int, sigma: float) -> float:
    """P(S_T > level), lognormal, zero drift."""
    if level <= 0:
        return 1.0
    t = max(dte, 1) / 365.0
    vt = sigma * math.sqrt(t)
    if vt <= 0:
        return 1.0 if spot > level else 0.0
    d = (math.log(spot / level) - 0.5 * vt * vt) / vt
    return 0.5 * (1.0 + math.erf(d / math.sqrt(2.0)))


def _legs(plan: Any, action: str, opt: Optional[str] = None) -> List[Any]:
    return [l for l in plan.legs if l.action == action
            and (opt is None or l.option_type == opt)]


def shadow_pop(plan: Any, spot: float, sigma: Optional[float],
               dte: int) -> Dict[str, Optional[float]]:
    """``{pop_delta, pop_rv, breakeven_low, breakeven_high}`` for a plan
    (values ``None`` where a structure or an input does not allow one)."""
    out: Dict[str, Optional[float]] = {"pop_delta": None, "pop_rv": None,
                                       "breakeven_low": None, "breakeven_high": None}
    if not getattr(plan, "legs", None) or getattr(plan, "far_expiration", ""):
        return out
    net = float(plan.net_credit)
    sp, sc = _legs(plan, "sell", "put"), _legs(plan, "sell", "call")
    lp, lc = _legs(plan, "buy", "put"), _legs(plan, "buy", "call")
    lo = hi = None
    if net >= 0:                                  # credit structures
        if sp:
            lo = sp[0].strike - net
        if sc:
            hi = sc[0].strike + net
        out["pop_delta"] = max(0.0, 1.0 - sum(abs(l.delta) for l in sp + sc))
    else:                                         # debit verticals
        debit, width = -net, float(plan.spread_width or 0.0)
        if lc and sc:
            lo = lc[0].strike + debit
            dl, ds = abs(lc[0].delta), abs(sc[0].delta)
        elif lp and sp:
            hi = lp[0].strike - debit
            dl, ds = abs(lp[0].delta), abs(sp[0].delta)
        else:
            return out
        if width > 0:
            out["pop_delta"] = dl + (ds - dl) * (debit / width)
    out["breakeven_low"], out["breakeven_high"] = lo, hi
    if sigma and spot > 0 and (lo is not None or hi is not None):
        p_above_lo = prob_above(spot, lo, dte, sigma) if lo is not None else 1.0
        p_above_hi = prob_above(spot, hi, dte, sigma) if hi is not None else 0.0
        out["pop_rv"] = max(0.0, p_above_lo - p_above_hi)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
