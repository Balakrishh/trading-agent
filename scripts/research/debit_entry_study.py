"""debit_entry_study.py — cheaper debit structures and trend confirmation.

Research for the 2026-11-01 build (docs/plans/research_debit_entry.md).
Two questions about the debit verticals the agent trades in CAUTION:

1. **Fit to budget.** Under CAUTION the per-trade cap halves to ~1 % and
   1,140 valid plans (Oct 26–30) were rejected for size. Does a cheaper
   structure — the long leg further out of the money, or a narrower
   width — keep the edge of the 0.50-delta spread?
2. **Trend confirmation.** Four of seven debit losers in October were
   bearish TLT / XLU entries while price sat above its 20-day average
   and rising. Does requiring price on the trade's side of the 20-day
   SMA with 10-day momentum agreeing beat the live pullback filter?

Model (deliberately simple; the comparison is between variants, not an
absolute P&L forecast):

* daily closes from yfinance (unadjusted), 2008 → today;
* volatility = max(20-day realized, 8 %) × 1.10 (implied usually sits
  over realized — skill 68 measured 1.1–1.3 on these ETFs);
* entry every 5th session per ticker when the trend regime agrees:
  bear = close < SMA50 and SMA50 lower than 10 sessions ago (bull mirror);
* 30-calendar-day vertical, long strike at the target |Δ|, short strike
  ``width_pct`` × spot further out; entry at Black–Scholes value × 1.03
  (natural-ish fill), skipped when (width − debit) / debit < 1.0 (the
  agent's ``debit_min_reward_risk``);
* marked daily at Black–Scholes with the entry volatility; exits at the
  agent's rules — stop at 50 % of the debit lost, target at 50 % gained,
  else at expiry; exit value × 0.97.

    python scripts/research/debit_entry_study.py [--start 2008-01-01]
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

TICKERS = ("SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "XLU", "XLE", "XLF",
           "XLK", "XLI", "XLV", "SMH", "EEM", "XLY", "GDX")
DTE = 30
STEP = 5
STOP, TARGET = 0.5, 0.5
ENTRY_SLIP, EXIT_SLIP = 1.03, 0.97
MIN_RR = 1.0
VARIANTS = {                      # name → (long |Δ|, width % of spot)
    "0.50Δ w2%": (0.50, 0.02),
    "0.50Δ w1%": (0.50, 0.01),
    "0.40Δ w2%": (0.40, 0.02),
    "0.35Δ w2%": (0.35, 0.02),
    "0.40Δ w1.5%": (0.40, 0.015),
    "0.50Δ w0.75%": (0.50, 0.0075),
    "0.50Δ w0.5%": (0.50, 0.005),
}


def ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(s: float, k: float, t: float, v: float, call: bool) -> float:
    if t <= 0 or v <= 0:
        return max(0.0, s - k) if call else max(0.0, k - s)
    vt = v * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * vt * vt) / vt
    d2 = d1 - vt
    return s * ncdf(d1) - k * ncdf(d2) if call else k * ncdf(-d2) - s * ncdf(-d1)


def strike_for_delta(s: float, t: float, v: float, delta: float, call: bool) -> float:
    """Strike whose BS |Δ| equals ``delta`` (zero rates)."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(delta if call else 1 - delta)
    return s * math.exp(0.5 * v * v * t - z * v * math.sqrt(t))


def sma(xs: List[float], i: int, n: int) -> Optional[float]:
    return sum(xs[i - n + 1:i + 1]) / n if i >= n - 1 else None


def atr(h, l, c, i, n=14) -> Optional[float]:
    if i < n:
        return None
    trs = [max(h[j] - l[j], abs(h[j] - c[j - 1]), abs(l[j] - c[j - 1])) for j in range(i - n + 1, i + 1)]
    return sum(trs) / n


def rv20(c: List[float], i: int) -> Optional[float]:
    if i < 21:
        return None
    r = [math.log(c[j] / c[j - 1]) for j in range(i - 19, i + 1)]
    m = sum(r) / len(r)
    return math.sqrt(sum((x - m) ** 2 for x in r) / (len(r) - 1)) * math.sqrt(252)


def simulate(c: List[float], i: int, call: bool, long_d: float, width_pct: float,
             vol: float) -> Optional[Tuple[float, float]]:
    """(P&L per $1 of debit, debit) for one trade, None when skipped."""
    s0, t0 = c[i], DTE / 365.0
    k_long = strike_for_delta(s0, t0, vol, long_d, call)
    k_short = k_long + (1 if call else -1) * width_pct * s0
    width = abs(k_short - k_long)

    def value(s, t):
        return (bs(s, k_long, t, vol, call) - bs(s, k_short, t, vol, call))
    debit = value(s0, t0) * ENTRY_SLIP
    if debit <= 0 or (width - debit) / debit < MIN_RR:
        return None
    sessions = int(DTE * 252 / 365)
    for d in range(1, sessions + 1):
        if i + d >= len(c):
            return None
        t = max(DTE - d * 365 / 252, 0) / 365.0
        v = value(c[i + d], t) * EXIT_SLIP
        if v <= debit * (1 - STOP) or v >= debit * (1 + TARGET) or d == sessions:
            return (v - debit) / debit, debit
    return None


def filters(direction: str, c, h, l, i) -> Dict[str, Optional[bool]]:
    p, s20, s50, a = c[i], sma(c, i, 20), sma(c, i, 50), atr(h, l, c, i)
    r10 = c[i] / c[i - 10] - 1 if i >= 10 else None
    if None in (s20, s50, a, r10) or not a:
        return {}
    if direction == "bear":
        pullback = p < s50 and (s20 - p) / a <= 1.0
        confirm = p < s20 and r10 < 0
    else:
        pullback = p > s50 and (p - s20) / a <= 1.0
        confirm = p > s20 and r10 > 0
    return {"none": True, "pullback (live)": pullback, "confirm": confirm,
            "pullback+confirm": pullback and confirm}


def run(start: str) -> None:
    import yfinance as yf
    stats: Dict[tuple, List[float]] = defaultdict(list)
    halves: Dict[tuple, List[float]] = defaultdict(list)
    for tk in TICKERS:
        df = yf.Ticker(tk).history(start=start, auto_adjust=False)
        if df.empty:
            continue
        c, h, l = (df[x].astype(float).tolist() for x in ("Close", "High", "Low"))
        dates = [d.date() for d in df.index]
        mid = dates[len(dates) // 2]
        for i in range(210, len(c) - 25, STEP):
            s50, s50_10 = sma(c, i, 50), sma(c, i - 10, 50)
            if s50 is None or s50_10 is None:
                continue
            if c[i] < s50 and s50 < s50_10:
                direction = "bear"
            elif c[i] > s50 and s50 > s50_10:
                direction = "bull"
            else:
                continue
            f = filters(direction, c, h, l, i)
            if not f:
                continue
            r = rv20(c, i)
            if r is None:
                continue
            vol = max(r, 0.08) * 1.10
            for vname, (ld, wp) in VARIANTS.items():
                res = simulate(c, i, direction == "bull", ld, wp, vol)
                if res is None:
                    continue
                pl, debit = res
                for fname, ok in f.items():
                    if ok:
                        stats[(direction, fname, vname)].append(pl)
                        halves[(direction, fname, vname, dates[i] >= mid)].append(pl)
                        stats[(direction, fname, vname, "cost")].append(debit / c[i])
    print(f"{'dir':4} {'filter':18} {'variant':12} {'n':>5} {'win':>5} {'avg/$':>7} "
          f"{'1st':>7} {'2nd':>7} {'cost%S':>6}")
    for (d, fn, vn), xs in sorted((k, v) for k, v in stats.items() if len(k) == 3):
        a = halves[(d, fn, vn, False)]
        b = halves[(d, fn, vn, True)]
        cost = stats[(d, fn, vn, "cost")]
        print(f"{d:4} {fn:18} {vn:12} {len(xs):5d} {sum(x > 0 for x in xs) / len(xs):5.0%} "
              f"{sum(xs) / len(xs):+7.3f} {sum(a) / max(len(a), 1):+7.3f} "
              f"{sum(b) / max(len(b), 1):+7.3f} {sum(cost) / len(cost):6.2%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2008-01-01")
    run(ap.parse_args().start)
