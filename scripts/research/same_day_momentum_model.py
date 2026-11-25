"""same_day_momentum_model.py — two years of the same-day momentum debit, priced by model.

Companion to same_day_replay.py (real 0DTE quotes, only a few weeks). Here:
SPY intraday bars from yfinance (1 h for ~2 years, 5 min for ~60 days) and
a Black–Scholes price for the 0DTE $5 vertical with σ = ``k`` × the prior
VIX close and time in trading minutes (k ≈ 0.6 calibrated on the logged
0DTE straddles, 2026-11-09 … 11-23; range 0.36–0.89, so 0.45 / 0.75 are
shown too). Entry at fixed times: buy the ≈ ATM vertical in the direction
of SPY's move since the open (``fade`` = the opposite side, the control);
+50 % / −50 % exits checked at each bar close, else held to 16:00
(intrinsic). Costs: $0.02 per spread per side.

    python scripts/research/same_day_momentum_model.py
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import time as dtime

import numpy as np
import yfinance as yf

WIDTH, COST = 5.0, 0.02
TARGET, STOP = 0.5, 0.5


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def call(s, k, sig, t):
    if t <= 0 or sig <= 0:
        return max(0.0, s - k)
    d1 = (math.log(s / k) + 0.5 * sig * sig * t) / (sig * math.sqrt(t))
    return s * ncdf(d1) - k * ncdf(d1 - sig * math.sqrt(t))


def put(s, k, sig, t):
    return call(s, k, sig, t) - s + k


def vertical(s, k, side, sig, t):
    if side > 0:
        return call(s, k, sig, t) - call(s, k + WIDTH, sig, t)
    return put(s, k, sig, t) - put(s, k - WIDTH, sig, t)


def t_left(ts):
    mins = 16 * 60 - (ts.hour * 60 + ts.minute)
    return max(0, mins) / (390 * 252)


def sessions(interval, period):
    bars = yf.download("SPY", interval=interval, period=period, progress=False, auto_adjust=False)
    bars = bars.xs("SPY", axis=1, level="Ticker") if bars.columns.nlevels > 1 else bars
    bars.index = bars.index.tz_convert("America/New_York")
    vix = yf.download("^VIX", period="3y", progress=False, auto_adjust=False)["Close"]["^VIX"]
    vix = {k.date(): float(v) for k, v in vix.items()}
    vdays = sorted(vix)
    step = 60 if interval == "1h" else 5
    for day, g in bars.groupby(bars.index.date):
        prior = [d for d in vdays if d < day]
        if not prior or len(g) < 4:
            continue
        opn = float(g["Open"].iloc[0])
        # (bar close time, close price)
        path = []
        for ts, row in g.iterrows():
            end = (ts + np.timedelta64(step, "m")).time()
            path.append((min(end, dtime(16, 0)), float(row["Close"])))
        yield day, opn, path, vix[prior[-1]]


def trade(path, i0, side, sig):
    t0, s0 = path[i0]
    k = round(s0)
    debit = vertical(s0, k, side, sig, t_left(t0)) + COST
    if debit <= 0.05 or debit >= WIDTH:
        return None
    for t, s in path[i0 + 1:]:
        v = vertical(s, k, side, sig, t_left(t)) - COST
        p = (v - debit) / debit
        if p >= TARGET or p <= -STOP or t >= dtime(16, 0):
            return p, 100 * (v - debit)
    return None


def run(interval, period, entries, ks=(0.45, 0.6, 0.75)):
    print(f"\n=== SPY {interval} bars, {period} ===")
    rows = defaultdict(list)            # (k, rule, bucket) → [(p, usd)]
    days = 0
    for day, opn, path, vix in sessions(interval, period):
        days += 1
        bucket = "VIX<18" if vix < 18 else ("VIX 18–25" if vix < 25 else "VIX≥25")
        for et in entries:
            i0 = next((i for i, (t, _) in enumerate(path) if t >= et), None)
            if i0 is None or i0 >= len(path) - 1:
                continue
            move = path[i0][1] - opn
            if move == 0:
                continue
            for kk in ks:
                for rule, side in (("momentum", 1 if move > 0 else -1), ("fade", -1 if move > 0 else 1)):
                    r = trade(path, i0, side, kk * vix / 100)
                    if r:
                        for b in (bucket, "all"):
                            rows[(kk, rule, b)].append(r)
    print(f"sessions: {days}")
    print(f"{'σ=k·VIX':8} {'rule':9} {'bucket':10} {'n':>5} {'win':>4} {'avg/$':>7} {'$/trade':>8}  CI95 avg/$ (bootstrap)")
    rng = np.random.default_rng(0)
    for key in sorted(rows, key=lambda x: (x[0], x[1], x[2] != "all", x[2])):
        xs = np.array([p for p, _ in rows[key]])
        usd = np.array([u for _, u in rows[key]])
        bs = [rng.choice(xs, len(xs)).mean() for _ in range(1000)]
        print(f"{key[0]:<8} {key[1]:9} {key[2]:10} {len(xs):5d} {(xs > 0).mean():4.0%} {xs.mean():+7.3f} {usd.mean():+8.1f}"
              f"  [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")


def main():
    run("1h", "730d", (dtime(10, 30), dtime(11, 30), dtime(12, 30), dtime(13, 30)))
    run("5m", "60d", (dtime(10, 0), dtime(11, 0), dtime(12, 30), dtime(14, 0)))


if __name__ == "__main__":
    main()
