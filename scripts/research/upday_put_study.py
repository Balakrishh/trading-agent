"""Sell a 21d 0.25Δ 2%-wide put spread at the close of a big up day — does it pay?

Setups (index ETF, entry at that session's close, no-prox exits, fills halfway mid→natural):
  all days         — every session (baseline)
  up ≥ X%          — the day's close-to-close gain ≥ X
  today-like       — up ≥ 1.5% AND close < SMA20 AND close < SMA50 AND close > SMA200
Reported per setup: n, win %, avg P&L per $ risked, how often the short put finished
in the money / was touched, and the 21-session forward return. One position per
ticker at a time (agent cadence) for the stats; the RV-rank > 95 days are shown
separately because the live HighIV block would have skipped them.
"""
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from credit_chain_backtest import NO_PROX, labels, load, select, simulate  # noqa: E402
from debit_entry_study import sma  # noqa: E402
from upday_put_study_rows import ROWS  # noqa: E402

SHAPE = ("put", 21, 0.25, 0.02)
DATA = Path.home() / "data/options_chains"
rows = []
for tk in ("SPY", "QQQ", "IWM"):
    u, by_day, by_leg = load(DATA, tk, max_dte=26)
    c, dates = u["close"].tolist(), u["date"].tolist()
    closes = dict(zip(dates, c))
    pos = {d: j for j, d in enumerate(dates)}
    for i in range(221, len(c) - 22):
        day = by_day.get(dates[i])
        if day is None:
            continue
        tr = select(day, SHAPE, c[i])
        if tr is None:
            continue
        out = simulate(tr, by_leg, closes, dates[i], NO_PROX)
        if not out:
            continue
        ks = tr["legs"][0]["ks"]
        path = [closes[d] for d in dates[i + 1:] if d <= tr["exp"] and d in closes]
        lab = labels(c, i)
        rows.append(dict(tk=tk, day=dates[i], exit_day=out[2], pnl=out[0], why=out[1],
                         ret1=c[i] / c[i - 1] - 1, fwd21=c[i + 21] / c[i] - 1,
                         below20=c[i] < sma(c, i, 20), below50=c[i] < sma(c, i, 50),
                         above200=c[i] > sma(c, i, 200), rank=lab["rank"] or 0,
                         hold=pos.get(out[2], i + 99) - i,
                         itm=bool(path) and path[-1] < ks, touched=bool(path) and min(path) < ks))
    print(f"{tk}: {len(rows)} rows", flush=True)


def one_at_a_time(rs):
    busy, out = defaultdict(lambda: date.min), []
    for r in sorted(rs, key=lambda r: (r["day"], r["tk"])):
        if r["day"] > busy[r["tk"]]:
            out.append(r)
            busy[r["tk"]] = r["exit_day"]
    return out


def month_ci(rs, n_boot=2000, seed=0):
    by_m = defaultdict(list)
    for r in rs:
        by_m[(r["day"].year, r["day"].month)].append(r["pnl"])
    blocks = list(by_m.values())
    if len(blocks) < 5:
        return "  n/a"
    rng = np.random.default_rng(seed)
    means = [np.mean(np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]))
             for _ in range(n_boot)]
    return f"[{np.percentile(means, 2.5):+.3f}, {np.percentile(means, 97.5):+.3f}]"


SETUPS = {
    "all days": lambda r: True,
    "up ≥ 1.5%": lambda r: r["ret1"] >= 0.015,
    "up ≥ 2.5%": lambda r: r["ret1"] >= 0.025,
    "below 20 & 50, above 200 (any day)": lambda r: r["below20"] and r["below50"] and r["above200"],
    "today-like: up ≥1.5%, <20 & <50, >200": lambda r: r["ret1"] >= 0.015 and r["below20"] and r["below50"] and r["above200"],
    "up ≥ 1.5%, below 200": lambda r: r["ret1"] >= 0.015 and not r["above200"],
}
print(f"\n{'setup':40} {'blk':>4} {'n':>5} {'win':>5} {'avg/$':>7} {'CI95 (month blocks)':>20} "
      f"{'ITM@exp':>8} {'touched':>8} {'fwd21 med':>9} {'fwd21 p10':>9}")
for name, f in SETUPS.items():
    for blk_name, keep in (("ok", lambda r: r["rank"] <= 0.95), ("HiIV", lambda r: r["rank"] > 0.95)):
        sub = one_at_a_time([r for r in rows if f(r) and keep(r)])
        if not sub:
            continue
        p = np.array([r["pnl"] for r in sub])
        fw = np.array([r["fwd21"] for r in sub])
        print(f"{name:40} {blk_name:>4} {len(sub):5d} {(p > 0).mean():5.0%} {p.mean():+7.3f} {month_ci(sub):>20} "
              f"{np.mean([r['itm'] for r in sub]):8.0%} {np.mean([r['touched'] for r in sub]):8.0%} "
              f"{np.median(fw):+9.1%} {np.percentile(fw, 10):+9.1%} {np.median([r['hold'] for r in sub]):8.0f} "
              + " ".join(f"{np.mean([r['why'] == 'target' and r['hold'] <= k for r in sub]):7.0%}" for k in (1, 2, 5))
              + f" {np.mean([r['pnl'] for r in sub if r['why'] == 'target' and r['hold'] <= 2] or [0]):+11.3f}")

import pickle  # noqa: E402
ROWS.parent.mkdir(parents=True, exist_ok=True)
pickle.dump(rows, open(ROWS, "wb"))
print("\ntoday-like entries (any RV rank):")
for r in one_at_a_time([r for r in rows if SETUPS["today-like: up ≥1.5%, <20 & <50, >200"](r)]):
    print(f"  {r['tk']} {r['day']} up {r['ret1']:+.1%} rank {r['rank']:.2f} → {r['why']:6} "
          f"{r['pnl']:+.2f}/$ in {r['hold']}d  fwd21 {r['fwd21']:+.1%}")
