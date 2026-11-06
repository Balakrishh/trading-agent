"""debit_exit_study.py — exit rules and entry choices for debit verticals.

Research for the 2026-11-06 build (docs/plans/research_debit_exits.md).
The paper account's debit spreads were often up first and stopped later
(TLT 10-26 +$100 → −$155, XLU +$100 → −$210). This compares exit
policies by direction, plus a few entry choices, on the same simple
model as ``debit_entry_study.py``:

* daily closes from yfinance (unadjusted), 2008 → today, 16 ETFs;
* volatility = max(20-day realized, 8 %) × 1.10;
* an entry every 5th session per ticker when the trend agrees (bear:
  close < SMA50, SMA50 falling over 10 sessions; bull mirror) and the
  live pullback filter passes (≤ 1 ATR from the SMA20, right side of the
  SMA50);
* 30-day vertical, long leg at the target |Δ|, short leg ``width`` ×
  spot further out; entry at value × 1.03, skipped when reward/risk < 1;
* marked daily at value × 0.97 with the live exit rules: stop at
  ``stop`` × debit lost; the trail arms at ``arm`` × debit gained, closes
  when profit falls to max(floor × debit, peak × (1 − giveback)), at the
  ceiling (0.90 × max profit) or, once armed, at ≤ 7 days to expiry;
  otherwise held to expiry.

P&L is per $1 of debit. Daily marks miss intraday paths (the live agent
checks every ~95 s), and the model has no put skew. Compare variants;
do not read the absolute level as a forecast.

    python scripts/research/debit_exit_study.py [--start 2008-01-01] [--cache DIR]
"""
from __future__ import annotations

import argparse
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from debit_entry_study import (DTE, ENTRY_SLIP, EXIT_SLIP, MIN_RR, STEP, TICKERS,
                               atr, bs, rv20, sma, strike_for_delta)

SESS = int(DTE * 252 / 365)
CEILING_OF_MAX = 0.90          # trail_debit_ceiling_pct
TRAIL_MAX_HOLD_DTE = 7         # trail_max_hold_dte (calendar days)


@dataclass(frozen=True)
class Side:
    stop: Optional[float] = 0.5        # fraction of debit lost; None = no stop
    arm: float = 0.5                   # trail arms at this gain (× debit)
    giveback: float = 0.25
    floor: float = 0.0                 # lock never below floor × debit
    width: float = 0.02                # × spot
    delta: float = 0.50


@dataclass(frozen=True)
class Policy:
    bull: Side = Side()
    bear: Side = Side()
    bear_needs_spy_below_sma50: bool = False
    trend200: bool = False             # bull above its SMA200, bear below
    bull_max_rv_pct: Optional[float] = None   # RV20 percentile vs its last year
    bear_max_rv_pct: Optional[float] = None


LIVE = Policy()                                        # 2026-11-06 live rules


@dataclass(frozen=True)
class Costs:
    """Pricing and fill assumptions. ``markup`` = implied ÷ realized vol
    used to price the spread (skill 68 measured 1.1–1.3 on these ETFs);
    ``entry_slip`` / ``exit_slip`` scale the model value at the fill;
    ``fixed`` is a per-share cost added on entry and taken on exit (the
    bid/ask in cents, which a %-of-value haircut understates on cheap
    spreads)."""
    markup: float = 1.10
    entry_slip: float = ENTRY_SLIP
    exit_slip: float = EXIT_SLIP
    fixed: float = 0.0
    stop_slip: Optional[float] = None    # stops fill here instead (they close at natural)


BASE_COSTS = Costs()


def load(start: str, cache: Optional[Path]) -> Dict[str, Tuple[list, list, list, list]]:
    fp = cache / f"closes_{start}.pkl" if cache else None
    if fp and fp.exists():
        return pickle.loads(fp.read_bytes())
    import yfinance as yf
    out = {}
    for tk in TICKERS:
        df = yf.Ticker(tk).history(start=start, auto_adjust=False)
        if df.empty:
            continue
        out[tk] = ([d.date() for d in df.index],
                   *(df[x].astype(float).tolist() for x in ("Close", "High", "Low")))
    if fp:
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(pickle.dumps(out))
    return out


def trade(c: List[float], i: int, call: bool, side: Side, vol: float,
          costs: Costs = BASE_COSTS) -> Optional[float]:
    """P&L per $1 of debit for one trade; None when skipped / no data."""
    s0, t0 = c[i], DTE / 365.0
    kl = strike_for_delta(s0, t0, vol, side.delta, call)
    ks = kl + (1 if call else -1) * side.width * s0
    width = abs(ks - kl)

    def val(s, t):
        return bs(s, kl, t, vol, call) - bs(s, ks, t, vol, call)
    debit = val(s0, t0) * costs.entry_slip + costs.fixed
    if debit <= 0 or (width - debit) / debit < MIN_RR:
        return None
    ceiling = CEILING_OF_MAX * (width - debit) / debit
    armed, peak = False, 0.0
    for d in range(1, SESS + 1):
        if i + d >= len(c):
            return None
        days_left = DTE - d * 365 / 252
        v = val(c[i + d], max(days_left, 0) / 365.0)
        p = (max(v * costs.exit_slip - costs.fixed, 0.0) - debit) / debit
        if side.stop is not None and p <= -side.stop:
            if costs.stop_slip is not None:
                p = (max(v * costs.stop_slip - costs.fixed, 0.0) - debit) / debit
            return p
        if not armed and p >= side.arm:
            armed, peak = True, p
        if armed:
            peak = max(peak, p)
            if p >= ceiling or p <= max(side.floor, peak * (1 - side.giveback)):
                return p
            if days_left <= TRAIL_MAX_HOLD_DTE:
                return p
        if d == SESS:
            return p
    return None


def entries(data, start_idx: int = 210):
    """(ticker, date, index, direction, vol) for every trend + pullback entry."""
    spy = data.get("SPY")
    spy_by_date = {d: k for k, d in enumerate(spy[0])} if spy else {}
    for tk, (dates, c, h, l) in data.items():
        for i in range(start_idx, len(c) - 25, STEP):
            s20, s50, s50_10 = sma(c, i, 20), sma(c, i, 50), sma(c, i - 10, 50)
            a, r = atr(h, l, c, i), rv20(c, i)
            if None in (s20, s50, s50_10, a, r) or not a:
                continue
            if c[i] < s50 and s50 < s50_10:
                direction, ok = "bear", (s20 - c[i]) / a <= 1.0
            elif c[i] > s50 and s50 > s50_10:
                direction, ok = "bull", (c[i] - s20) / a <= 1.0
            else:
                continue
            if not ok:
                continue
            k = spy_by_date.get(dates[i])
            spy_below = (k is not None and k >= 50 and spy[1][k] < sma(spy[1], k, 50))
            s200 = sma(c, i, 200)
            yield (tk, dates[i], i, direction, max(r, 0.08) * 1.10, spy_below,
                   None if s200 is None else c[i] > s200, rv_percentile(c, i))


def rv_percentile(c: List[float], i: int, lookback: int = 200, every: int = 5) -> Optional[float]:
    """Share of the last ``lookback`` sessions' RV20 readings (every 5th) at
    or below today's — the ticker's own volatility regime. 200 matches the
    live ``RegimeClassifier._compute_iv_rank`` over the ~220 fetched bars."""
    if i < lookback + 21:
        return None
    cur = rv20(c, i)
    hist = [rv20(c, j) for j in range(i - lookback + 1, i + 1, every)]
    hist = [h for h in hist if h is not None]
    return sum(h <= cur for h in hist) / len(hist) if hist and cur is not None else None


def run(data, policy: Policy, ents, costs: Costs = BASE_COSTS) -> List[Tuple[str, object, str, float]]:
    rows = []
    for tk, day, i, direction, vol, spy_below, above200, rvp in ents:
        if direction == "bear" and policy.bear_needs_spy_below_sma50 and not spy_below:
            continue
        if policy.trend200 and above200 is not (direction == "bull"):
            continue
        cap = policy.bull_max_rv_pct if direction == "bull" else policy.bear_max_rv_pct
        if cap is not None and (rvp is None or rvp > cap):
            continue
        side = policy.bull if direction == "bull" else policy.bear
        p = trade(data[tk][1], i, direction == "bull", side, vol * costs.markup / 1.10, costs)
        if p is not None:
            rows.append((tk, day, direction, p))
    return rows


def stats(xs: List[float]) -> str:
    if not xs:
        return "   0     –        –"
    win = sum(x > 0 for x in xs) / len(xs)
    return f"{len(xs):5d} {win:5.0%} {sum(xs) / len(xs):+8.3f}"


def summary(rows, split) -> Dict[str, str]:
    out = {}
    for d in ("bull", "bear", "all"):
        xs = [r for r in rows if d == "all" or r[2] == d]
        a = [r[3] for r in xs if r[1] < split]
        b = [r[3] for r in xs if r[1] >= split]
        allp = [r[3] for r in xs]
        out[d] = (f"{stats(allp)}  {sum(a) / max(len(a), 1):+7.3f} {sum(b) / max(len(b), 1):+7.3f}")
    return out


PLAN_EXITS = Policy(bull=Side(stop=0.8), bear=Side(arm=0.4, giveback=0.25))
FILTERS = dict(trend200=True, bull_max_rv_pct=0.8, bear_max_rv_pct=0.5)
FINAL = replace(PLAN_EXITS, **FILTERS)

VARIANTS: Dict[str, Policy] = {
    "live (stop 50, arm 50/give 25, 2 %)": LIVE,
    "exits: bear arm30/give40/floor10": replace(LIVE, bear=Side(arm=0.3, giveback=0.4, floor=0.1)),
    "exits: bear arm40/give25": replace(LIVE, bear=Side(arm=0.4, giveback=0.25)),
    "exits: bull stop 80": replace(LIVE, bull=Side(stop=0.8)),
    "exits: both": PLAN_EXITS,
    "widths 1 % (both), live exits": replace(LIVE, bull=Side(width=0.01), bear=Side(width=0.01)),
    "filter: trend200, live exits": replace(LIVE, trend200=True),
    "filter: trend200 + RV pct, live exits": replace(LIVE, **FILTERS),
    "FINAL: exits + trend200 + RV pct": FINAL,
    "FINAL with bear arm30/give40": replace(FINAL, bear=Side(arm=0.3, giveback=0.4, floor=0.1)),
    "FINAL with bull stop 70": replace(FINAL, bull=Side(stop=0.7)),
    "FINAL, bear RV pct <= 33 %": replace(FINAL, bear_max_rv_pct=0.33),
    "FINAL, bear RV pct <= 67 %": replace(FINAL, bear_max_rv_pct=0.67),
    "FINAL, bull RV pct <= 67 %": replace(FINAL, bull_max_rv_pct=0.67),
    "FINAL, no bull RV cap": replace(FINAL, bull_max_rv_pct=None),
    "FINAL + bear only when SPY < SMA50": replace(FINAL, bear_needs_spy_below_sma50=True),
}


def sensitivity() -> Dict[str, Policy]:
    out = {}
    for arm in (0.2, 0.3, 0.4, 0.5):
        for give in (0.25, 0.4, 0.5):
            for floor in (0.0, 0.1, 0.2):
                if floor >= arm:
                    continue
                out[f"bear arm{arm:.0%} give{give:.0%} floor{floor:.0%}"] = replace(
                    FINAL, bear=Side(arm=arm, giveback=give, floor=floor))
    for stop in (0.4, 0.5, 0.6):
        out[f"bear stop {stop:.0%}"] = replace(FINAL, bear=replace(FINAL.bear, stop=stop))
    for stop in (0.5, 0.6, 0.7, 0.8, None):
        out[f"bull stop {'none' if stop is None else f'{stop:.0%}'}"] = replace(
            FINAL, bull=Side(stop=stop))
    for arm, give in ((0.3, 0.4), (0.5, 0.25), (0.5, 0.4)):
        out[f"bull arm{arm:.0%} give{give:.0%} (stop 80)"] = replace(
            FINAL, bull=Side(stop=0.8, arm=arm, giveback=give))
    return out


def by_year(rows) -> Dict[int, Tuple[str, str]]:
    out = {}
    years = sorted({r[1].year for r in rows})
    for y in years:
        out[y] = tuple(stats([r[3] for r in rows if r[1].year == y and r[2] == d]) for d in ("bull", "bear"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--part", default="all", choices=("all", "variants", "sensitivity", "years", "tickers"))
    args = ap.parse_args()
    data = load(args.start, Path(args.cache) if args.cache else None)
    ents = list(entries(data))
    dates = sorted({e[1] for e in ents})
    split = dates[len(dates) // 2]
    print(f"entries: {len(ents)}  split at {split}\n")
    head = f"{'variant':42} {'dir':4}     n   win    avg/$     1st     2nd"
    if args.part in ("all", "variants"):
        print(head)
        for name, pol in VARIANTS.items():
            for d, line in summary(run(data, pol, ents), split).items():
                print(f"{name:42} {d:4} {line}")
            print()
    if args.part in ("all", "sensitivity"):
        print(head)
        for name, pol in sensitivity().items():
            s = summary(run(data, pol, ents), split)
            d = "bear" if name.startswith("bear") else "bull"
            print(f"{name:42} {d:4} {s[d]}")
    if args.part in ("all", "years"):
        live = run(data, LIVE, ents)
        plan = run(data, FINAL, ents)
        ly, py = by_year(live), by_year(plan)
        print(f"\n{'year':5} {'live bull':>20} {'plan bull':>20} {'live bear':>20} {'plan bear':>20}")
        for y in ly:
            print(f"{y:5} {ly[y][0]:>20} {py[y][0]:>20} {ly[y][1]:>20} {py[y][1]:>20}")
    if args.part in ("all", "tickers"):
        live = run(data, LIVE, ents)
        plan = run(data, FINAL, ents)
        print(f"\n{'ticker':6} {'live':>20} {'plan':>20}")
        for tk in data:
            print(f"{tk:6} {stats([r[3] for r in live if r[0] == tk]):>20} "
                  f"{stats([r[3] for r in plan if r[0] == tk]):>20}")


if __name__ == "__main__":
    main()
