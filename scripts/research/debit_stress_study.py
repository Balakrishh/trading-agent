"""debit_stress_study.py — stress tests for the 2026-11-06 debit-spread rules.

Research for docs/plans/research_debit_stress.md. Takes the live rules
(``debit_exit_study.FINAL``: put debits take profit / arm the trail at
40 %, call debits stop at 80 %, 200-day trend + volatility-regime entry
filters) and the pre-2026-11-06 rules (``LIVE``) and asks how they hold
up when the assumptions are wrong:

* costs — implied vol 1.0–1.4 × realized; fills 3 % → 5 % off model;
  a fixed $0.05 / share bid-ask cost per side;
* market periods — crises and calm bull runs;
* rolling 3-year windows;
* parameter noise — every threshold jittered ±25 % at once, 200 draws;
* the account — 2 trades a week drawn at random from the eligible set,
  risk 1.5 % of equity each, 3,000 bootstrapped 1-year paths: return,
  max drawdown, losing streaks;
* a month-block bootstrap of the mean P&L per $ (trades in the same
  month are correlated).

Same model and caveats as ``debit_exit_study.py`` (daily marks, no put
skew). ``--cache DIR`` reuses the downloaded closes.

    python scripts/research/debit_stress_study.py --cache DIR [--part all|costs|periods|rolling|noise|account|ci]
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import debit_exit_study as dx
from debit_exit_study import FINAL, LIVE, Costs, Side, entries, load, run

PERIODS: Dict[str, Tuple[date, date]] = {
    "2008-09 crisis":     (date(2008, 9, 1), date(2009, 6, 30)),
    "2010 flash / 2011":  (date(2010, 4, 15), date(2011, 12, 31)),
    "2015-16 selloffs":   (date(2015, 7, 1), date(2016, 3, 31)),
    "2018 Q4":            (date(2018, 9, 15), date(2019, 1, 31)),
    "2020 Covid":         (date(2020, 2, 1), date(2020, 6, 30)),
    "2022 bear":          (date(2022, 1, 1), date(2022, 12, 31)),
    "2025-26":            (date(2025, 1, 1), date(2026, 12, 31)),
    "2013 calm bull":     (date(2013, 1, 1), date(2013, 12, 31)),
    "2017 calm bull":     (date(2017, 1, 1), date(2017, 12, 31)),
    "2021 bull":          (date(2021, 1, 1), date(2021, 12, 31)),
    "2023-24 bull":       (date(2023, 1, 1), date(2024, 12, 31)),
}


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def line(rows) -> str:
    xs = [r[3] for r in rows]
    if not xs:
        return "    0     –        –"
    return f"{len(xs):5d} {sum(x > 0 for x in xs) / len(xs):5.0%} {mean(xs):+8.3f}"


def part_costs(data, ents):
    print("\n## Costs — all entries; n / win / avg per $\n")
    print(f"{'costs':44} {'live':>22} {'final':>22}")
    grid = []
    for mk in (1.0, 1.1, 1.2, 1.3, 1.4):
        grid.append((f"IV/RV {mk:.1f}, fills 3 %", Costs(markup=mk)))
    grid += [
        ("IV/RV 1.1, fills 5 %", Costs(entry_slip=1.05, exit_slip=0.95)),
        ("IV/RV 1.1, fills 3 % + $0.05/side", Costs(fixed=0.05)),
        ("IV/RV 1.2, fills 5 % + $0.05/side", Costs(markup=1.2, entry_slip=1.05, exit_slip=0.95, fixed=0.05)),
        ("IV/RV 1.3, fills 5 % + $0.05/side", Costs(markup=1.3, entry_slip=1.05, exit_slip=0.95, fixed=0.05)),
    ]
    for name, c in grid:
        print(f"{name:44} {line(run(data, LIVE, ents, c)):>22} {line(run(data, FINAL, ents, c)):>22}")


def part_periods(data, ents):
    live, final = run(data, LIVE, ents), run(data, FINAL, ents)
    print("\n## Market periods\n")
    print(f"{'period':20} {'live':>22} {'final':>22} {'final bull':>22} {'final bear':>22}")
    for name, (a, b) in PERIODS.items():
        lv = [r for r in live if a <= r[1] <= b]
        fn = [r for r in final if a <= r[1] <= b]
        print(f"{name:20} {line(lv):>22} {line(fn):>22} "
              f"{line([r for r in fn if r[2] == 'bull']):>22} {line([r for r in fn if r[2] == 'bear']):>22}")


def part_rolling(data, ents):
    live, final = run(data, LIVE, ents), run(data, FINAL, ents)
    print("\n## Rolling 3-year windows (start each January)\n")
    print(f"{'window':11} {'live':>22} {'final':>22}")
    neg_l = neg_f = n = 0
    for y in range(2009, 2025):
        a, b = date(y, 1, 1), date(y + 2, 12, 31)
        lv = [r for r in live if a <= r[1] <= b]
        fn = [r for r in final if a <= r[1] <= b]
        n += 1
        neg_l += mean([r[3] for r in lv]) < 0
        neg_f += mean([r[3] for r in fn]) < 0
        print(f"{y}-{y + 2:<6} {line(lv):>22} {line(fn):>22}")
    print(f"\nnegative windows: live {neg_l}/{n}, final {neg_f}/{n}")


def jitter(rng, x, pct=0.25, lo=None, hi=None):
    v = x * (1 + rng.uniform(-pct, pct))
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def part_noise(data, ents, draws=200, seed=7):
    rng = random.Random(seed)
    out = []
    for _ in range(draws):
        pol = replace(
            FINAL,
            bull=Side(stop=jitter(rng, 0.8, hi=0.99), arm=jitter(rng, 0.5), giveback=jitter(rng, 0.25)),
            bear=Side(stop=jitter(rng, 0.5), arm=jitter(rng, 0.4), giveback=jitter(rng, 0.25)),
            bull_max_rv_pct=jitter(rng, 0.8, hi=1.0), bear_max_rv_pct=jitter(rng, 0.5, hi=1.0))
        rows = run(data, pol, ents)
        out.append((mean([r[3] for r in rows]), sum(r[3] > 0 for r in rows) / len(rows), len(rows)))
    avgs = sorted(o[0] for o in out)
    wins = sorted(o[1] for o in out)
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
    print(f"\n## Parameter noise — every threshold ±25 % at once, {draws} draws\n")
    print(f"avg per $: 5th {q(avgs, .05):+.3f}  median {q(avgs, .5):+.3f}  95th {q(avgs, .95):+.3f}  "
          f"positive in {sum(a > 0 for a in avgs)}/{draws}")
    print(f"win rate:  5th {q(wins, .05):.0%}  median {q(wins, .5):.0%}  95th {q(wins, .95):.0%}")
    live_avg = mean([r[3] for r in run(data, LIVE, ents)])
    print(f"draws beating the old live rules ({live_avg:+.3f}): {sum(a > live_avg for a in avgs)}/{draws}")


def account_paths(rows, *, paths=3000, weeks=52, per_week=2, risk=0.015, seed=11, equity=28_700.0):
    """Bootstrapped 1-year paths: each week draw ``per_week`` trades from a
    random historical week that had candidates (weeks keep their mix of
    bull / bear and their correlation)."""
    by_week = defaultdict(list)
    for r in rows:
        y, w, _ = r[1].isocalendar()
        by_week[(y, w)].append(r[3])
    weeks_list = [v for v in by_week.values() if v]
    rng = random.Random(seed)
    finals, dds, streaks = [], [], []
    for _ in range(paths):
        eq, peak, dd, streak, worst = equity, equity, 0.0, 0, 0
        for _ in range(weeks):
            wk = rng.choice(weeks_list)
            for p in rng.sample(wk, min(per_week, len(wk))):
                eq += eq * risk * p
                streak = streak + 1 if p <= 0 else 0
                worst = max(worst, streak)
            peak = max(peak, eq)
            dd = max(dd, (peak - eq) / peak)
        finals.append(eq / equity - 1)
        dds.append(dd)
        streaks.append(worst)
    finals.sort(); dds.sort(); streaks.sort()
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"ret_p5": q(finals, .05), "ret_p50": q(finals, .5), "ret_p95": q(finals, .95),
            "p_loss": sum(f < 0 for f in finals) / paths,
            "dd_p50": q(dds, .5), "dd_p95": q(dds, .95),
            "streak_p50": q(streaks, .5), "streak_p95": q(streaks, .95)}


def part_account(data, ents):
    print("\n## The account — 2 trades / week, 1.5 % of equity at risk each, 3,000 one-year paths\n")
    print(f"{'rules / costs':34} {'ret p5':>7} {'p50':>7} {'p95':>7} {'P(loss)':>8} "
          f"{'maxDD p50':>9} {'p95':>6} {'loss streak p50':>15} {'p95':>4}")
    for name, pol, c in (("live, base costs", LIVE, dx.BASE_COSTS),
                         ("final, base costs", FINAL, dx.BASE_COSTS),
                         ("final, IV/RV 1.2 + 5 % + $0.05", FINAL,
                          Costs(markup=1.2, entry_slip=1.05, exit_slip=0.95, fixed=0.05))):
        s = account_paths(run(data, pol, ents, c))
        print(f"{name:34} {s['ret_p5']:+7.1%} {s['ret_p50']:+7.1%} {s['ret_p95']:+7.1%} {s['p_loss']:8.0%} "
              f"{s['dd_p50']:9.1%} {s['dd_p95']:6.1%} {s['streak_p50']:15d} {s['streak_p95']:4d}")


def block_ci(rows, draws=2000, seed=3):
    by_month = defaultdict(list)
    for r in rows:
        by_month[(r[1].year, r[1].month)].append(r[3])
    months = list(by_month.values())
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        xs = []
        for _ in range(len(months)):
            xs.extend(rng.choice(months))
        means.append(mean(xs))
    means.sort()
    return means[int(.025 * draws)], means[int(.975 * draws)], sum(m > 0 for m in means) / draws


def part_ci(data, ents):
    print("\n## Month-block bootstrap of the mean P&L per $ (95 % interval)\n")
    for name, pol in (("live", LIVE), ("final", FINAL)):
        rows = run(data, pol, ents)
        for d in ("bull", "bear", "all"):
            xs = [r for r in rows if d == "all" or r[2] == d]
            lo, hi, p = block_ci(xs)
            print(f"{name:6} {d:4} mean {mean([r[3] for r in xs]):+.3f}  95 % [{lo:+.3f}, {hi:+.3f}]  "
                  f"share of draws > 0: {p:.0%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--part", default="all",
                    choices=("all", "costs", "periods", "rolling", "noise", "account", "ci"))
    args = ap.parse_args()
    data = load(args.start, Path(args.cache) if args.cache else None)
    ents = list(entries(data))
    parts = {"costs": part_costs, "periods": part_periods, "rolling": part_rolling,
             "noise": part_noise, "account": part_account, "ci": part_ci}
    for k, fn in parts.items():
        if args.part in ("all", k):
            fn(data, ents)


if __name__ == "__main__":
    main()
