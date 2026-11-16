"""hiv_quick_target_study.py — trade high-vol days with a quicker profit target?

The live HighIV filter (20-day RV percentile > 95) blocks every entry, so the
agent sits flat for weeks after a shock. Question (2026-11-12): sell the
vol-band put spread anyway on those days but take profit sooner (20–40 % of
credit instead of 50 %)?

Real end-of-day SPY / QQQ / IWM quotes, fills halfway mid → natural on entry
and exit, agent stops (50 % of max loss, 3× credit) without the proximity
exit. One position per ticker at a time; 2 %-risk account. Splits: RV rank
> 95 vs the live 50–80 band, and ticker above / below its SMA-200.

    python scripts/research/hiv_quick_target_study.py [--data ~/data/options_chains]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from credit_chain_backtest import Exits, labels, load, select, simulate  # noqa: E402
from put_spread_playbook_backtest import report  # noqa: E402

SHAPES = {"21d": ("put", 21, 0.25, 0.02), "14d": ("put", 14, 0.25, 0.02)}
EXITS = {f"tp{int(t * 100)}": Exits(target=t, proximity=None) for t in (0.2, 0.3, 0.4, 0.5)}
EXITS["tp30 2×stop"] = Exits(target=0.3, hard_mult=2.0, proximity=None)


def collect(data: Path, tickers):
    rows = []
    for tk in tickers:
        u, by_day, by_leg = load(data, tk, max_dte=26)
        c, dates = u["close"].tolist(), u["date"].tolist()
        closes = dict(zip(dates, c))
        pos = {d: j for j, d in enumerate(dates)}
        for i in range(221, len(c) - 1):
            day = by_day.get(dates[i])
            if day is None:
                continue
            lab = labels(c, i)
            if lab["rank"] is None:
                continue
            for sname, shape in SHAPES.items():
                tr = select(day, shape, c[i])
                if tr is None:
                    continue
                for ename, ex in EXITS.items():
                    out = simulate(tr, by_leg, closes, dates[i], ex)
                    if out:
                        rows.append(dict(tk=tk, day=dates[i], shape=sname, exits=ename, pnl=out[0],
                                         why=out[1], exit_day=out[2], hold=pos.get(out[2], i + 99) - i,
                                         rank=lab["rank"], above200=lab["above200"]))
        print(f"  {tk}: done", flush=True)
    return rows


def walk(rows, shape, exits, keep):
    busy, out = defaultdict(lambda: date.min), []
    for r in sorted((r for r in rows if r["shape"] == shape and r["exits"] == exits),
                    key=lambda r: (r["day"], r["tk"])):
        if r["day"] <= busy[r["tk"]] or not keep(r):
            continue
        out.append((r["tk"], r["day"], r["exit_day"], r["pnl"], r["why"]))
        busy[r["tk"]] = r["exit_day"]
    return out


GROUPS = {
    "HiIV": lambda r: r["rank"] > 0.95,
    "HiIV >200": lambda r: r["rank"] > 0.95 and r["above200"],
    "HiIV <200": lambda r: r["rank"] > 0.95 and not r["above200"],
    "band 50-80": lambda r: 0.50 <= r["rank"] <= 0.80,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(Path.home() / "data/options_chains"))
    ap.add_argument("--tickers", default="SPY,QQQ,IWM")
    args = ap.parse_args()
    rows = collect(Path(args.data), args.tickers.split(","))
    for shape in SHAPES:
        print(f"\n=== {shape} 0.25Δ 2% put spread ===")
        report({f"{g} {e}": walk(rows, shape, e, keep) for g, keep in GROUPS.items() for e in EXITS})
        print("median hold (sessions):")
        for g, keep in GROUPS.items():
            cells = []
            for e in EXITS:
                hs = sorted(r["hold"] for r in rows if r["shape"] == shape and r["exits"] == e and keep(r))
                cells.append(f"{e} {hs[len(hs) // 2] if hs else 0}")
            print(f"  {g:12} " + "  ".join(cells))


if __name__ == "__main__":
    main()
