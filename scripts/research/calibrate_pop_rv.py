"""Calibrate the realized-volatility POP against what actually happened.

Backlog B2 (2026-10-13). The live scorers read win probability from the
option delta — the market's own (implied) odds — so expected value is
≈ 0 before costs by construction. ``shadow_pop`` logs a second estimate,
``pop_rv``: a zero-drift lognormal with the 20-day realized volatility.
Before any trade decision leans on it (backlog B3) this script asks, over
daily history:

1. **Calibration** — when ``pop_rv`` says 80 %, does the short strike
   survive ~80 % of the time? (Buckets of predicted probability vs the
   realized rate, and the Brier score.)
2. **Variance risk premium** — for tickers with a volatility index
   (SPY ^VIX, QQQ ^VXN, IWM ^RVX, GLD ^GVZ, EEM ^VXEEM) strikes are placed
   the way the live agent places them, at a target *implied* delta; then
   the implied POP (≈ 1 − |Δ|, what the scorers use), ``pop_rv`` and the
   realized survival rate are compared. Realized > implied means option
   sellers were paid more than the risk they carried.

Strikes are one-sided (a short put below spot, a short call above) at
z ∈ {0.5, 0.75, 1.0} standard deviations (≈ 31 / 23 / 16 delta), horizons
of 14 / 21 / 30 calendar days. Samples overlap day to day, so counts
overstate independent evidence; read differences of a few points as noise.

Research only: downloads from Yahoo, writes nothing, never imported by the
live agent. Uses ``trading_agent.shadow_pop`` itself (no re-implementation).

    python scripts/research/calibrate_pop_rv.py [--start 2019-01-01] [--tickers SPY,QQQ]
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trading_agent.shadow_pop import prob_above, realized_vol  # noqa: E402

WATCHLIST = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "EEM", "GLD", "TLT", "GDX",
             "META", "AMZN"]
VOL_INDEX = {"SPY": "^VIX", "QQQ": "^VXN", "IWM": "^RVX", "GLD": "^GVZ", "EEM": "^VXEEM"}
DTES = (14, 21, 30)
ZS = (0.5, 0.75, 1.0)
BUCKETS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]


def closes(symbol: str, start: str) -> pd.Series:
    import yfinance as yf  # type: ignore
    df = yf.Ticker(symbol).history(start=start, auto_adjust=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    s = df["Close"].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s


def samples(px: pd.Series, iv: Optional[pd.Series], begin: str
            ) -> List[Tuple[str, int, float, float, Optional[float], bool, str]]:
    """(side, dte, z, pop_rv, pop_iv, survived, trend) for every day × dte ×
    z × side; trend = "up" when the close is above its 50-day average.
    Strike distance uses implied vol when available (as the live agent's
    delta targeting does), else realized vol."""
    out = []
    dates = px.index
    vals = px.values
    sma50 = px.rolling(50).mean().values
    for i in range(21, len(vals)):
        if dates[i] < pd.Timestamp(begin):
            continue
        rv = realized_vol(vals[i - 20: i + 1], 20)
        if not rv:
            continue
        sig_iv = None
        if iv is not None:
            v = iv.get(dates[i])
            sig_iv = float(v) / 100.0 if v is not None and not np.isnan(v) else None
        spot = float(vals[i])
        if np.isnan(sma50[i]):
            continue
        trend = "up" if spot > sma50[i] else "down"
        for dte in DTES:
            j = dates.searchsorted(dates[i] + pd.Timedelta(days=dte))
            if j >= len(vals):
                continue
            end = float(vals[j])
            t = dte / 365.0
            sig_k = sig_iv or rv
            for z in ZS:
                k_put = spot * math.exp(-z * sig_k * math.sqrt(t))
                k_call = spot * math.exp(z * sig_k * math.sqrt(t))
                p_rv_put = prob_above(spot, k_put, dte, rv)
                p_rv_call = 1.0 - prob_above(spot, k_call, dte, rv)
                p_iv_put = prob_above(spot, k_put, dte, sig_iv) if sig_iv else None
                p_iv_call = (1.0 - prob_above(spot, k_call, dte, sig_iv)) if sig_iv else None
                out.append(("put", dte, z, p_rv_put, p_iv_put, end > k_put, trend))
                out.append(("call", dte, z, p_rv_call, p_iv_call, end < k_call, trend))
    return out


def report(name: str, rows) -> Dict[str, float]:
    if not rows:
        print(f"\n{name}: no samples")
        return {}
    p_rv = np.array([r[3] for r in rows])
    won = np.array([r[5] for r in rows], dtype=float)
    brier_rv = float(np.mean((p_rv - won) ** 2))
    print(f"\n{name}: {len(rows):,} samples (overlapping)  realized survival {won.mean():.1%}  "
          f"mean pop_rv {p_rv.mean():.1%}  Brier(rv) {brier_rv:.4f}")
    print("   pop_rv bucket   n        predicted  realized   gap")
    for lo, hi in BUCKETS:
        m = (p_rv >= lo) & (p_rv < hi if hi < 1.0 else p_rv <= hi)
        if m.sum() < 30:
            continue
        print(f"   {lo:.1f}–{hi:.1f}      {int(m.sum()):7,}  {p_rv[m].mean():8.1%}  "
              f"{won[m].mean():8.1%}  {won[m].mean() - p_rv[m].mean():+6.1%}")
    out = {"n": len(rows), "survival": won.mean(), "pop_rv": p_rv.mean(), "brier_rv": brier_rv}
    iv_rows = [r for r in rows if r[4] is not None]
    if iv_rows:
        p_iv = np.array([r[4] for r in iv_rows])
        p_rv2 = np.array([r[3] for r in iv_rows])
        won2 = np.array([r[5] for r in iv_rows], dtype=float)
        out.update(pop_iv=p_iv.mean(), survival_iv_rows=won2.mean(),
                   brier_iv=float(np.mean((p_iv - won2) ** 2)),
                   brier_rv_iv_rows=float(np.mean((p_rv2 - won2) ** 2)))
        print(f"   strikes at implied delta: n {len(iv_rows):,}  implied POP {p_iv.mean():.1%}  "
              f"pop_rv {p_rv2.mean():.1%}  realized {won2.mean():.1%}  "
              f"→ sellers' edge vs implied {won2.mean() - p_iv.mean():+.1%}  "
              f"Brier iv {out['brier_iv']:.4f} vs rv {out['brier_rv_iv_rows']:.4f}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--tickers", default=",".join(WATCHLIST))
    args = ap.parse_args(argv)
    fetch_from = str((pd.Timestamp(args.start) - pd.Timedelta(days=60)).date())
    pooled = []
    by_cell: Dict[Tuple[str, int, float], list] = defaultdict(list)
    summary = {}
    for t in [x.strip().upper() for x in args.tickers.split(",") if x.strip()]:
        px = closes(t, fetch_from)
        if px.empty:
            print(f"{t}: no price data")
            continue
        iv = closes(VOL_INDEX[t], fetch_from) if t in VOL_INDEX else None
        if iv is not None and iv.empty:
            iv = None
        rows = samples(px, iv, args.start)
        summary[t] = report(f"{t} ({'implied: ' + VOL_INDEX[t] if iv is not None else 'no vol index'})", rows)
        pooled += rows
        for r in rows:
            by_cell[(r[0], r[1], r[2])].append(r)
    report("ALL TICKERS POOLED", pooled)
    print("\nBy side × trend (pooled; the agent sells puts in up-trends, calls in"
          " down-trends): n, pop_rv, implied, realized")
    for side in ("put", "call"):
        for trend in ("up", "down"):
            rs = [r for r in pooled if r[0] == side and r[6] == trend]
            if not rs:
                continue
            iv = [r for r in rs if r[4] is not None]
            ivs = (f"  [vol-index tickers: implied {np.mean([r[4] for r in iv]):.1%}, "
                   f"realized {np.mean([r[5] for r in iv]):.1%}, n {len(iv):,}]") if iv else ""
            print(f"   {side:4} {trend:4}  n {len(rs):7,}  pop_rv {np.mean([r[3] for r in rs]):.1%}  "
                  f"realized {np.mean([r[5] for r in rs]):.1%}{ivs}")
    print("\nBy side × DTE × z (pooled): n, pop_rv, realized, gap; implied POP where available")
    for key in sorted(by_cell):
        rs = by_cell[key]
        p = np.mean([r[3] for r in rs])
        w = np.mean([r[5] for r in rs])
        iv = [r[4] for r in rs if r[4] is not None]
        ivs = f"  implied {np.mean(iv):.1%}" if iv else ""
        print(f"   {key[0]:4} {key[1]:2}d z={key[2]:.2f}  n {len(rs):6,}  pop_rv {p:.1%}  "
              f"realized {w:.1%}  gap {w - p:+.1%}{ivs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
