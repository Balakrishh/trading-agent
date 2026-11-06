"""debit_chain_backtest.py — the agent's debit-vertical rules on real option quotes.

Research for docs/plans/research_debit_stress.md. The earlier studies
priced spreads with Black–Scholes and assumed fills; this one trades
real end-of-day chains (bid, ask, delta per contract, 2008-2025) from
the MIT-licensed dataset published with lambdaclass/
options_portfolio_backtester (release ``data-v1``: SPY, QQQ, IWM).

Entries (every 5th session, as the other studies): trend regime (close
vs SMA50 and its 10-session slope), the live pullback filter, and for
the new rules the 200-day trend and volatility-regime filters.
Selection copies ``decision_engine.decide_debit_spread``:

* expiration: the highest DTE in [25, 35] (``dte_debit`` 30 ± 5);
* long leg: |Δ| closest to 0.50; short leg: the listed strike nearest
  long ∓ w × strike for w in 1 / 1.5 / 2 / 2.5 % (``width_grid_pct``);
* each leg must pass the liquidity gate (rejected only when the bid/ask
  is wider than BOTH $0.15 and 5 % of mid); natural ≤ mid × 1.05;
  (width − natural) / natural ≥ 1; the cheapest natural wins.

Marks are the real end-of-day quotes of the same two contracts: mid and
natural (sell the long at its bid, buy the short at its ask). Stops are
judged on mid (as live). The profit target / trail is judged on natural
(old rules) or mid (new rules). Fills: entry at mid + ``entry_frac`` ×
(natural − mid); closes at mid − ``close_frac`` × (mid − natural); a
mid-judged profit close is skipped that day when the fill would keep
less than half the mid profit (the capped close ladder). Positions still
open on the last session before expiry close there (DTE safety).
Daily data only: no intraday path, no 3-cycle debounce.

    python scripts/research/debit_chain_backtest.py --data DIR [--tickers SPY,QQQ,IWM]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq

from debit_entry_study import atr, rv20, sma
from debit_exit_study import rv_percentile

WIDTHS = (0.01, 0.015, 0.02, 0.025)
DTE_LO, DTE_HI = 25, 35
STEP = 5


@dataclass(frozen=True)
class Rules:
    measure: str = "natural"           # profit target / trail judged on "natural" or "mid"
    bull_stop: float = 0.5
    bear_stop: float = 0.5
    bull_arm: float = 0.5
    bear_arm: float = 0.5
    giveback: float = 0.25
    ceiling_of_max: float = 0.90
    filters: bool = False              # trend200 + vol regime
    bull_max_rank: float = 0.80
    bear_max_rank: float = 0.50
    entry_frac: float = 1.0            # 1 = natural, 0 = mid
    close_frac: float = 0.5            # 0.5 = halfway mid → natural, 1 = natural
    mid_floor: float = 0.5             # mid-judged profit close keeps ≥ this share of the mid profit


OLD = Rules()
OLD_LADDER = replace(OLD, entry_frac=0.5)
NEW = Rules(measure="mid", bull_stop=0.8, bear_arm=0.4, filters=True, entry_frac=0.5)


def load(data: Path, tk: str):
    t = pq.read_table(data / f"{tk}_options.parquet",
                      columns=["date", "expiration", "strike", "type", "bid", "ask", "delta"]).to_pandas()
    t["type"] = t["type"].astype(str)
    t["dte"] = (t["expiration"] - t["date"]).dt.days
    t = t[(t["dte"] >= 0) & (t["dte"] <= 50) & (t["ask"] > 0)]
    t["date"] = t["date"].dt.date
    t["expiration"] = t["expiration"].dt.date
    u = pd.read_parquet(data / f"{tk}_underlying.parquet", columns=["date", "close", "high", "low"])
    u["date"] = pd.to_datetime(u["date"]).dt.date
    u = u.sort_values("date").drop_duplicates("date")
    by_day = {d: g for d, g in t.groupby("date", sort=False)}
    by_leg = {k: g.set_index("date")[["bid", "ask"]] for k, g in t.groupby(["expiration", "strike", "type"], sort=False)}
    return u, by_day, by_leg


def gate_ok(bid: float, ask: float) -> bool:
    mid = (bid + ask) / 2
    return not ((ask - bid) > 0.15 and (ask - bid) > 0.05 * mid)


def select(day_df: pd.DataFrame, direction: str) -> Optional[dict]:
    opt = "call" if direction == "bull" else "put"
    d = day_df[(day_df["type"] == opt) & (day_df["dte"] >= DTE_LO) & (day_df["dte"] <= DTE_HI) & (day_df["bid"] > 0)]
    if d.empty:
        return None
    exp = d["expiration"].max()
    d = d[d["expiration"] == exp]
    long_row = d.iloc[(d["delta"].abs() - 0.5).abs().argsort().iloc[0]]
    if not gate_ok(long_row.bid, long_row.ask):
        return None
    kl = float(long_row.strike)
    strikes = d.set_index("strike")
    best = None
    for w in WIDTHS:
        target = kl + (w if opt == "call" else -w) * kl
        ks = float(min(strikes.index, key=lambda k: abs(k - target)))
        if ks == kl or (ks > kl) != (opt == "call"):
            continue
        s = strikes.loc[ks]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[0]
        if s.bid <= 0 or not gate_ok(s.bid, s.ask):
            continue
        natural = long_row.ask - s.bid
        mid = (long_row.bid + long_row.ask) / 2 - (s.bid + s.ask) / 2
        width = abs(ks - kl)
        if natural <= 0 or mid <= 0 or natural > mid * 1.05 or (width - natural) / natural < 1.0:
            continue
        if best is None or natural < best["natural"]:
            best = dict(exp=exp, opt=opt, kl=kl, ks=ks, width=width, natural=natural, mid=mid)
    return best


def simulate(trade: dict, by_leg, start: date, rules: Rules, direction: str) -> Optional[float]:
    lg = by_leg.get((trade["exp"], trade["kl"], trade["opt"]))
    sh = by_leg.get((trade["exp"], trade["ks"], trade["opt"]))
    if lg is None or sh is None:
        return None
    q = lg.join(sh, lsuffix="_l", rsuffix="_s", how="inner").sort_index()
    q = q[q.index > start]
    if q.empty:
        return None
    debit = trade["mid"] + rules.entry_frac * (trade["natural"] - trade["mid"])
    stop = rules.bull_stop if direction == "bull" else rules.bear_stop
    arm = rules.bull_arm if direction == "bull" else rules.bear_arm
    ceiling = rules.ceiling_of_max * (trade["width"] - debit) / debit
    armed, peak = False, 0.0
    rows = list(q.itertuples())
    for k, r in enumerate(rows):
        if min(r.bid_l, r.bid_s) < 0 or r.ask_l <= 0 or r.ask_s <= 0:
            continue
        mid = (r.bid_l + r.ask_l) / 2 - (r.bid_s + r.ask_s) / 2
        nat = r.bid_l - r.ask_s
        fill = mid - rules.close_frac * (mid - nat)
        p_mid, p_fill = (mid - debit) / debit, (fill - debit) / debit
        p_meas = p_mid if rules.measure == "mid" else (nat - debit) / debit
        last = k == len(rows) - 1 or (trade["exp"] - r.Index).days <= 1
        if p_mid <= -stop or last:
            return p_fill
        if not armed and p_meas >= arm:
            armed, peak = True, p_meas
        if armed:
            peak = max(peak, p_meas)
            close = (p_meas >= ceiling or p_meas <= peak * (1 - rules.giveback)
                     or (trade["exp"] - r.Index).days <= 7)
            if close:
                if rules.measure == "mid" and p_fill < rules.mid_floor * max(p_mid, 0):
                    continue                     # capped ladder: no fill today, retry tomorrow
                return p_fill
    return None


def signals(u: pd.DataFrame):
    c, h, l = u["close"].tolist(), u["high"].tolist(), u["low"].tolist()
    dates = u["date"].tolist()
    for i in range(221, len(c) - 25, STEP):
        s20, s50, s50_10, s200 = sma(c, i, 20), sma(c, i, 50), sma(c, i - 10, 50), sma(c, i, 200)
        a = atr(h, l, c, i)
        if None in (s20, s50, s50_10, s200, a) or not a:
            continue
        if c[i] < s50 and s50 < s50_10:
            direction, ok = "bear", (s20 - c[i]) / a <= 1.0
        elif c[i] > s50 and s50 > s50_10:
            direction, ok = "bull", (c[i] - s20) / a <= 1.0
        else:
            continue
        if ok:
            yield dates[i], direction, c[i] > s200, rv_percentile(c, i)


def run(data: Path, tickers: List[str], variants: Dict[str, Rules]):
    results = defaultdict(list)           # variant → [(tk, date, dir, pnl, debit)]
    for tk in tickers:
        u, by_day, by_leg = load(data, tk)
        for day, direction, above200, rank in signals(u):
            df = by_day.get(day)
            if df is None:
                continue
            trade = select(df, direction)
            if trade is None:
                continue
            for name, rules in variants.items():
                if rules.filters:
                    if above200 is not (direction == "bull"):
                        continue
                    cap = rules.bull_max_rank if direction == "bull" else rules.bear_max_rank
                    if rank is None or rank > cap:
                        continue
                p = simulate(trade, by_leg, day, rules, direction)
                if p is not None:
                    results[name].append((tk, day, direction, p, trade["natural"]))
        print(f"  loaded {tk}: {len(by_day)} days", flush=True)
    return results


def fmt(xs: List[float]) -> str:
    if not xs:
        return "    0    –       –"
    return f"{len(xs):5d} {sum(x > 0 for x in xs) / len(xs):4.0%} {sum(xs) / len(xs):+7.3f}"


def report(results, split=date(2017, 1, 1)):
    print(f"\n{'variant':40} {'dir':4} {'n':>5} {'win':>4} {'avg/$':>7}   1st     2nd   med debit")
    for name, rows in results.items():
        for d in ("bull", "bear", "all"):
            xs = [r for r in rows if d == "all" or r[2] == d]
            a = [r[3] for r in xs if r[1] < split]
            b = [r[3] for r in xs if r[1] >= split]
            deb = sorted(r[4] for r in xs)
            print(f"{name:40} {d:4} {fmt([r[3] for r in xs])} {sum(a)/max(1,len(a)):+7.3f} "
                  f"{sum(b)/max(1,len(b)):+7.3f}  {deb[len(deb)//2] if deb else float('nan'):6.2f}")
        print()
    first = next(iter(results))
    tks = sorted({r[0] for rows in results.values() for r in rows})
    print(f"{'ticker':6}" + "".join(f" {n[:28]:>30}" for n in results))
    for tk in tks:
        print(f"{tk:6}" + "".join(f" {fmt([r[3] for r in rows if r[0] == tk]):>30}" for rows in results.values()))


VARIANTS = {
    "old: natural target, entry at natural": OLD,
    "old + entry ladder (mid + ½)": OLD_LADDER,
    "new exits only (no filters)": replace(NEW, filters=False),
    "old exits + filters": replace(OLD_LADDER, filters=True),
    "NEW (live 2026-11-06)": NEW,
    "NEW, closes at natural": replace(NEW, close_frac=1.0),
    "NEW, entry at natural": replace(NEW, entry_frac=1.0),
    "NEW, bear arm 30 / give 40": replace(NEW, bear_arm=0.3, giveback=0.25),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--tickers", default="IWM,QQQ,SPY")
    args = ap.parse_args()
    res = run(Path(args.data), args.tickers.split(","), VARIANTS)
    report(res)


if __name__ == "__main__":
    main()
