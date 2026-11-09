"""bear_call_backtest.py — bear call spreads on real option quotes, gated by trend and IV/RV.

Research for docs/plans/research_credit_vol.md §6 (backlog §15 "re-test
bear calls on index ETFs with the IV/RV floor"). Bear calls have been
off since 2026-10-14 (B6: short calls survived 2–6 points less often
than their delta implied, even in down-trends). Question: does a high
implied-vs-realized ratio rescue them on SPY / QQQ / IWM?

Every session, for each shape (call spread; DTE, short |Δ|, width as %
of spot) and exit set, one trade is selected and simulated with
``credit_chain_backtest`` (real end-of-day bid / ask, fills halfway
mid → natural). Variants then filter those rows by gate and walk them in
date order holding one position per ticker at a time — the agent's
cadence — and report per-trade stats, a month-block bootstrap and a 2 %-
risk account.

Gates: ``bear`` = close below SMA50 and SMA50 falling over 10 sessions
(the agent's bearish regime); ``vrp`` = ATM implied vol of the chosen
expiry ÷ RV20 ≥ the floor (the live ``vrp_min_ratio`` gate uses 30-day
ATM IV ÷ RV20); HighIV (rank > 95) always blocks.

    python scripts/research/bear_call_backtest.py --data DIR [--tickers SPY,QQQ,IWM]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from credit_chain_backtest import AGENT, NO_PROX, labels, load, select, simulate
from debit_entry_study import sma
from put_spread_playbook_backtest import report

SHAPES = {"21d 0.20Δ 1%": ("call", 21, 0.20, 0.01), "21d 0.15Δ 1%": ("call", 21, 0.15, 0.01),
          "14d 0.20Δ 1%": ("call", 14, 0.20, 0.01), "21d 0.20Δ 2%": ("call", 21, 0.20, 0.02)}
EXITS = {"agent": AGENT, "no-prox": NO_PROX}


@dataclass(frozen=True)
class Variant:
    shape: str = "21d 0.20Δ 1%"
    exits: str = "agent"
    bear: bool = True
    vrp_min: float = 0.0
    rank_max: float = 0.95


VARIANTS = {
    "bear, any IV/RV":                    Variant(),
    "bear, IV/RV ≥ 1.0 (live floor)":     Variant(vrp_min=1.0),
    "bear, IV/RV ≥ 1.2":                  Variant(vrp_min=1.2),
    "bear, IV/RV ≥ 1.3":                  Variant(vrp_min=1.3),
    "bear, IV/RV ≥ 1.2, no-prox":         Variant(exits="no-prox", vrp_min=1.2),
    "bear, IV/RV ≥ 1.2, 0.15Δ":           Variant(shape="21d 0.15Δ 1%", vrp_min=1.2),
    "bear, IV/RV ≥ 1.2, 14d":             Variant(shape="14d 0.20Δ 1%", vrp_min=1.2),
    "bear, IV/RV ≥ 1.2, 2% wide":         Variant(shape="21d 0.20Δ 2%", vrp_min=1.2),
    "any trend, IV/RV ≥ 1.2":             Variant(bear=False, vrp_min=1.2),
    "bear, IV/RV ≥ 1.2, rank ≤ 80":       Variant(vrp_min=1.2, rank_max=0.80),
}


def collect(data: Path, tickers: List[str]) -> List[dict]:
    """One simulated trade per (ticker, session, shape, exits)."""
    rows = []
    for tk in tickers:
        u, by_day, by_leg = load(data, tk, max_dte=30)
        c, dates = u["close"].tolist(), u["date"].tolist()
        closes = dict(zip(dates, c))
        for i in range(221, len(c) - 1):
            day = by_day.get(dates[i])
            if day is None:
                continue
            lab = labels(c, i)
            if lab["rank"] is None or not lab["rv"]:
                continue
            bear = c[i] < sma(c, i, 50) < sma(c, i - 10, 50)
            for sname, shape in SHAPES.items():
                tr = select(day, shape, c[i])
                if tr is None:
                    continue
                for ename, ex in EXITS.items():
                    out = simulate(tr, by_leg, closes, dates[i], ex)
                    if out:
                        rows.append(dict(tk=tk, day=dates[i], shape=sname, exits=ename, pnl=out[0],
                                         why=out[1], exit_day=out[2], bear=bear, rank=lab["rank"],
                                         vrp=tr["atm_iv"] / lab["rv"]))
        print(f"  {tk}: done", flush=True)
    return rows


def one_at_a_time(rows: List[dict], v: Variant):
    busy: Dict[str, date] = defaultdict(lambda: date.min)
    out = []
    for r in sorted((r for r in rows if r["shape"] == v.shape and r["exits"] == v.exits),
                    key=lambda r: (r["day"], r["tk"])):
        if r["day"] <= busy[r["tk"]] or r["rank"] > v.rank_max or r["vrp"] < v.vrp_min:
            continue
        if v.bear and not r["bear"]:
            continue
        out.append((r["tk"], r["day"], r["exit_day"], r["pnl"], r["why"]))
        busy[r["tk"]] = r["exit_day"]
    return out


def bucket_table(rows: List[dict]):
    """All entries (overlapping), 21d 0.20Δ 1 %, agent exits: avg per $ by IV/RV and trend."""
    sub = [r for r in rows if r["shape"] == "21d 0.20Δ 1%" and r["exits"] == "agent" and r["rank"] <= 0.95]
    edges = [0, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 99]
    print("\nall entries, 21d 0.20Δ 1%, agent exits — n / win / avg per $ by IV/RV")
    print(f"{'IV/RV':>12} {'bear':>20} {'not bear':>20}")
    for lo, hi in zip(edges, edges[1:]):
        cells = []
        for flag in (True, False):
            x = np.array([r["pnl"] for r in sub if lo <= r["vrp"] < hi and r["bear"] is flag])
            cells.append(f"{len(x):5d} {(x > 0).mean() if len(x) else 0:4.0%} {x.mean() if len(x) else 0:+7.3f}")
        print(f"{f'[{lo}, {hi})':>12} {cells[0]:>20} {cells[1]:>20}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--tickers", default="SPY,QQQ,IWM")
    args = ap.parse_args()
    rows = collect(Path(args.data), args.tickers.split(","))
    bucket_table(rows)
    report({name: one_at_a_time(rows, v) for name, v in VARIANTS.items()})


if __name__ == "__main__":
    main()
