"""hiv_put_debit_study.py — buy put debit spreads when volatility is extreme?

Question (2026-11-19): the HighIV block (20-day RV percentile > 95) and the
bear vol-regime cap (≤ 50) stop every put debit spread in a sell-off. Do
bearish put debits on high-vol days pay on real quotes, and does a quicker
target help?

Real end-of-day SPY / QQQ / IWM chains; entries, selection and exits are
debit_chain_backtest's (bear trend: close < SMA50 with a falling SMA50, the
live pullback filter ≤ 1 ATR from the 20-day SMA; 0.50Δ long, 25–35 DTE,
1–2.5 % wide; entry mid + ½, exits judged at mid with the live trail).
Splits by RV rank and by a DEFENSIVE proxy (SPY below its SMA-200). Every
5th session per ticker, as the earlier studies; month-block bootstrap CI.

    python scripts/research/hiv_put_debit_study.py [--data ~/data/options_chains]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from debit_chain_backtest import NEW, load, select, signals, simulate  # noqa: E402
from debit_entry_study import sma  # noqa: E402

LIVE = replace(NEW, filters=False)
EXITS = {
    "live (arm 40, trail 25)": LIVE,
    "arm 25": replace(LIVE, bear_arm=0.25),
    "arm 25, stop 35": replace(LIVE, bear_arm=0.25, bear_stop=0.35),
}
BUCKETS = {
    "rank ≤ 50 (live-allowed)": lambda r: r["rank"] <= 0.50,
    "rank 50–95": lambda r: 0.50 < r["rank"] <= 0.95,
    "rank > 95 (HighIV)": lambda r: r["rank"] > 0.95,
    "rank > 95, SPY < SMA200": lambda r: r["rank"] > 0.95 and r["spy_below200"],
    "rank > 95, SPY > SMA200": lambda r: r["rank"] > 0.95 and not r["spy_below200"],
    "rank > 80, SPY < SMA200": lambda r: r["rank"] > 0.80 and r["spy_below200"],
}


def spy_below200(data: Path):
    u, _, _ = load(data, "SPY")
    c, d = u["close"].tolist(), u["date"].tolist()
    return {d[i]: c[i] < sma(c, i, 200) for i in range(200, len(c)) if sma(c, i, 200)}


def month_ci(rows, n_boot=2000, seed=0):
    by_m = defaultdict(list)
    for r in rows:
        by_m[(r["day"].year, r["day"].month)].append(r["pnl"])
    blocks = list(by_m.values())
    if len(blocks) < 5:
        return "n/a"
    rng = np.random.default_rng(seed)
    means = [np.mean(np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]))
             for _ in range(n_boot)]
    return f"[{np.percentile(means, 2.5):+.3f}, {np.percentile(means, 97.5):+.3f}]"


def collect(data: Path, tickers):
    below = spy_below200(data)
    rows = []
    for tk in tickers:
        u, by_day, by_leg = load(data, tk)
        for day, direction, _above200, rank in signals(u):
            if direction != "bear" or rank is None or day not in by_day:
                continue
            trade = select(by_day[day], "bear")
            if trade is None:
                continue
            for ename, rules in EXITS.items():
                p = simulate(trade, by_leg, day, rules, "bear")
                if p is not None:
                    rows.append(dict(tk=tk, day=day, exits=ename, pnl=p, rank=rank,
                                     spy_below200=below.get(day, False), debit=trade["natural"]))
        print(f"  {tk}: done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(Path.home() / "data/options_chains"))
    ap.add_argument("--tickers", default="SPY,QQQ,IWM")
    args = ap.parse_args()
    rows = collect(Path(args.data), args.tickers.split(","))
    print(f"\n{'bucket':28} {'exits':24} {'n':>4} {'win':>5} {'avg/$':>7}  {'CI95 (month blocks)':>18}  2008–16  2017–25")
    for bname, keep in BUCKETS.items():
        for ename in EXITS:
            sub = [r for r in rows if r["exits"] == ename and keep(r)]
            if not sub:
                continue
            p = np.array([r["pnl"] for r in sub])
            a = [r["pnl"] for r in sub if r["day"].year < 2017]
            b = [r["pnl"] for r in sub if r["day"].year >= 2017]
            print(f"{bname:28} {ename:24} {len(p):4d} {(p > 0).mean():5.0%} {p.mean():+7.3f}  {month_ci(sub):>18}"
                  f"  {np.mean(a) if a else float('nan'):+7.3f}  {np.mean(b) if b else float('nan'):+7.3f}")
        print()
    print("per ticker, rank > 95, live exits:")
    for tk in args.tickers.split(","):
        p = [r["pnl"] for r in rows if r["tk"] == tk and r["exits"] == next(iter(EXITS)) and r["rank"] > 0.95]
        if p:
            print(f"  {tk}: n {len(p)}  win {np.mean(np.array(p) > 0):.0%}  avg {np.mean(p):+.3f}")


if __name__ == "__main__":
    main()
