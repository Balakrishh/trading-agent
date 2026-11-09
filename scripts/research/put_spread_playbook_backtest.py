"""put_spread_playbook_backtest.py — a candidate bull-put playbook vs the agent's own credit scan.

Research for docs/plans/research_credit_vol.md §5. Follows up §2 of
that report, which found 0.25Δ bull put spreads best at vol rank 50–80
and the 1 % strike-proximity exit costly. Here, on the same real
end-of-day chains (SPY / QQQ / IWM), trades are taken the way the
agent would hold them — one position per ticker at a time, the next
entry no earlier than the session after the close — so the results
read as an account, not overlapping samples.

Selectors:

* ``scan``  — the agent's live credit scan: ``decide()`` with the live
  preset (``dte_grid`` × ``delta_grid`` × ``width_grid_pct``, the C/W
  floor ``|Δ| × (1 + edge_buffer)``, ``min_pop``) on chain slices at the
  expirations ``next_weekly_expiration`` picks; the top candidate.
* ``fixed`` — the expiry nearest 21 DTE, short |Δ| 0.25, long 2 % of
  spot below; no C/W floor.

Gates: ``bull`` = the agent's bullish regime (close above SMA50 and
SMA50 rising over 10 sessions); ``rank`` = vol rank inside a band;
HighIV (rank > 95) always blocks, as live. Exits from
``credit_chain_backtest.Exits``. Fills halfway mid → natural.

Account: 2 % of equity at risk per trade, positions on the three ETFs
may overlap; equity is updated at each exit.

    python scripts/research/put_spread_playbook_backtest.py --data DIR [--preset STRATEGY_PRESET.json]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from credit_chain_backtest import AGENT, HOLD, NO_PROX, Exits, labels, load, select, simulate
from debit_entry_study import sma

from trading_agent.calendar_utils import next_weekly_expiration
from trading_agent.decision_engine import ChainSlice, DecisionInput, decide
from trading_agent.strategy_presets import load_active_preset

RISK_PER_TRADE = 0.02


@dataclass(frozen=True)
class Variant:
    selector: str                          # "scan" | "fixed"
    exits: Exits
    bull: bool = True
    rank_lo: float = 0.0
    rank_hi: float = 0.95


VARIANTS = {
    "agent scan, agent exits, bull":           Variant("scan", AGENT),
    "agent scan, no-prox, bull":               Variant("scan", NO_PROX),
    "fixed, agent exits, bull":                Variant("fixed", AGENT),
    "fixed, no-prox, bull":                    Variant("fixed", NO_PROX),
    "fixed, no-prox, bull, rank 50-80":        Variant("fixed", NO_PROX, rank_lo=0.5, rank_hi=0.8),
    "fixed, no-prox, rank 50-80 (any trend)":  Variant("fixed", NO_PROX, bull=False, rank_lo=0.5, rank_hi=0.8),
    "fixed, tp-only, bull, rank 50-80":        Variant("fixed", HOLD, rank_lo=0.5, rank_hi=0.8),
    "fixed, agent exits, bull, rank 50-80":    Variant("fixed", AGENT, rank_lo=0.5, rank_hi=0.8),
}


def scan_pick(day, today: date, spot: float, preset) -> Optional[dict]:
    """The agent's bull-put pick on one historical session, via decide()."""
    puts = day[(day["type"] == "put") & (day["bid"] > 0)]
    slices, seen = [], set()
    for target in preset.dte_grid:
        lo, hi = max(1, target - preset.dte_window_days), target + preset.dte_window_days
        try:
            exp = next_weekly_expiration(today=today, target_dte=target, dte_min=lo, dte_max=hi)
        except Exception:                  # noqa: BLE001 — no weekly in the window
            continue
        if exp in seen:
            continue
        seen.add(exp)
        x = puts[puts["expiration"] == exp]
        if x.empty:
            continue
        contracts = [dict(strike=float(r.strike), delta=float(r.delta), bid=float(r.bid), ask=float(r.ask),
                          symbol=f"{exp}P{r.strike}", type="put") for r in x.itertuples()]
        slices.append(ChainSlice(expiration=exp.isoformat(), dte=(exp - today).days, contracts=contracts))
    if not slices:
        return None
    out = decide(DecisionInput(side="bull_put", chain_slices=slices, preset=preset, spot=spot), max_candidates=1)
    if not out.candidates:
        return None
    c = out.candidates[0]
    exp = date.fromisoformat(c.expiration)
    x = puts[puts["expiration"] == exp].drop_duplicates("strike").set_index("strike")
    s, l = x.loc[c.short_strike], x.loc[c.long_strike]
    nat, mid = s.bid - l.ask, (s.bid + s.ask) / 2 - (l.bid + l.ask) / 2
    if nat <= 0 or mid <= 0:
        return None
    w = abs(c.short_strike - c.long_strike)
    return dict(exp=exp, legs=[dict(opt="put", ks=float(c.short_strike), kl=float(c.long_strike), w=w,
                                    nat=nat, mid=mid, d=abs(c.short_delta))],
                nat=nat, mid=mid, w=w)


def run(data: Path, tickers: List[str], preset) -> Dict[str, List[Tuple[str, date, date, float, str]]]:
    trades = defaultdict(list)             # variant → [(tk, entry, exit, pnl/$, why)]
    for tk in tickers:
        u, by_day, by_leg = load(data, tk, max_dte=40)
        c, dates = u["close"].tolist(), u["date"].tolist()
        closes = dict(zip(dates, c))
        busy_until = {name: date.min for name in VARIANTS}
        picks: Dict[Tuple[str, date], Optional[dict]] = {}
        for i in range(221, len(c) - 1):
            today, day = dates[i], by_day.get(dates[i])
            if day is None:
                continue
            lab = labels(c, i)
            rank = lab["rank"]
            if rank is None:
                continue
            bull = c[i] > sma(c, i, 50) > sma(c, i - 10, 50)
            for name, v in VARIANTS.items():
                if today <= busy_until[name] or (v.bull and not bull) or not (v.rank_lo <= rank < v.rank_hi):
                    continue
                key = (v.selector, today)
                if key not in picks:
                    picks[key] = (scan_pick(day, today, c[i], preset) if v.selector == "scan"
                                  else select(day, ("put", 21, 0.25, 0.02), c[i]))
                tr = picks[key]
                if tr is None:
                    continue
                out = simulate(tr, by_leg, closes, today, v.exits)
                if out is None:
                    continue
                pnl, why, exit_day = out
                trades[name].append((tk, today, exit_day, pnl, why))
                busy_until[name] = exit_day
        print(f"  {tk}: done", flush=True)
    return trades


def account(rows) -> dict:
    eq, peak, mdd, by_year = 1.0, 1.0, 0.0, defaultdict(lambda: 1.0)
    for tk, entry, exit_day, pnl, why in sorted(rows, key=lambda r: r[2]):
        g = RISK_PER_TRADE * pnl
        eq *= 1 + g
        by_year[exit_day.year] *= 1 + g
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    yrs = sorted(by_year)
    n_years = max(1, yrs[-1] - yrs[0] + 1) if yrs else 1
    return dict(cagr=eq ** (1 / n_years) - 1, mdd=mdd, losing=sum(v < 1 for v in by_year.values()),
                years=len(by_year), by_year=dict(by_year))


def report(trades):
    print(f"\n{'variant':42} {'n':>5} {'win':>4} {'avg/$':>7} {'2008-16':>8} {'2017-25':>8} "
          f"{'CAGR':>6} {'maxDD':>6} {'lose yrs':>8}  CI95 avg/$ (month blocks)")
    rng = np.random.default_rng(0)
    for name, rows in trades.items():
        if not rows:
            print(f"{name:42}     0")
            continue
        x = np.array([r[3] for r in rows])
        early = [r[3] for r in rows if r[1].year < 2017]
        late = [r[3] for r in rows if r[1].year >= 2017]
        months = defaultdict(list)
        for r in rows:
            months[(r[1].year, r[1].month)].append(r[3])
        blocks = list(months.values())
        bs = []
        for _ in range(2000):
            pick = [blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]
            bs.append(np.mean([v for b in pick for v in b]))
        a = account(rows)
        print(f"{name:42} {len(x):5d} {(x > 0).mean():4.0%} {x.mean():+7.3f} {np.mean(early) if early else 0:+8.3f} "
              f"{np.mean(late) if late else 0:+8.3f} {a['cagr']:+6.1%} {a['mdd']:6.1%} {a['losing']:>3}/{a['years']:<4}"
              f"  [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
    print("\nper ticker (n / win / avg):")
    for name, rows in trades.items():
        cells = []
        for tk in ("SPY", "QQQ", "IWM"):
            x = [r[3] for r in rows if r[0] == tk]
            cells.append(f"{tk} {len(x):4d} {np.mean([v > 0 for v in x]) if x else 0:4.0%} {np.mean(x) if x else 0:+.3f}")
        print(f"{name:42} " + "   ".join(cells))
    print("\nexit reasons:")
    for name, rows in trades.items():
        why = defaultdict(int)
        for r in rows:
            why[r[4]] += 1
        print(f"{name:42} " + " ".join(f"{k}={v / len(rows):.0%}" for k, v in sorted(why.items())))
    print("\naccount return by year (2 % risk per trade):")
    names = list(trades)
    years = sorted({y for rows in trades.values() for y in account(rows)["by_year"]}) if trades else []
    print("year " + "".join(f"{n[:14]:>16}" for n in names))
    accts = [account(trades[n])["by_year"] if trades[n] else {} for n in names]
    for y in years:
        print(f"{y} " + "".join(f"{(a.get(y, 1) - 1):+16.1%}" for a in accts))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--tickers", default="SPY,QQQ,IWM")
    ap.add_argument("--preset", default=None, help="preset file for the agent scan (default: the active one)")
    args = ap.parse_args()
    preset = load_active_preset(Path(args.preset)) if args.preset else load_active_preset()
    print("scan preset:", preset.to_summary_line()[:200])
    report(run(Path(args.data), args.tickers.split(","), preset))


if __name__ == "__main__":
    main()
