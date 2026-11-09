"""credit_chain_backtest.py — short-premium spreads on real option quotes, by volatility regime.

Research for docs/plans/research_credit_vol.md. Asks whether selling
defined-risk premium for 1–3 weeks pays after a volatility spike — the
"volatility is when people make money" idea — on the same real
end-of-day chains as ``debit_chain_backtest.py`` (SPY / QQQ / IWM,
release ``data-v1`` of lambdaclass/options_portfolio_backtester).

Entries every session (overlapping; per-trade averages, not an account
curve). Structures: put credit spread (bull put) and iron condor. For
each (target DTE, short |Δ|, width as % of spot):

* expiration: the listed one nearest the target DTE within ±3 days;
* short leg: |Δ| closest to the target; long leg: the listed strike
  nearest short ∓ width × spot;
* each leg passes the live liquidity gate (rejected only when the
  bid/ask is wider than BOTH $0.15 and 5 % of mid); natural credit > 0;
* ``floor``: credit / width ≥ |Δshort| (the live C/W floor with
  edge_buffer 0, judged on mid) — reported both ways.

Entry fill: mid − ``entry_frac`` × (mid − natural). Daily marks on the
same contracts' real quotes; closes at mid + ``close_frac`` × (natural −
mid). Exit rules (``Exits``) mirror ``PositionMonitor`` for credit
spreads: profit target (50 % of credit, on mid), stop at 50 % of max
loss, hard stop at 3× credit, strike proximity (close within 1 % of a
short strike), DTE safety (close on the last session before expiry).
Positions reaching expiry without a close settle at intrinsic on the
expiry close. Daily closes only: intraday breaches are missed, which
flatters the stops and proximity rule.

Volatility labels on the entry day: ``rank`` = the live RV20
percentile (``rv_percentile``); ``vrp`` = ATM implied vol of the chosen
expiry ÷ RV20; ``receding`` = rank hit ≥ 95 in the last 10 sessions and
RV20 is below its 5-session-ago value.

    python scripts/research/credit_chain_backtest.py --data DIR [--tickers SPY,QQQ,IWM]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq

from debit_chain_backtest import gate_ok
from debit_entry_study import rv20, sma
from debit_exit_study import rv_percentile


@dataclass(frozen=True)
class Exits:
    target: Optional[float] = 0.50     # of credit, on mid
    stop_max_loss: Optional[float] = 0.50
    hard_mult: Optional[float] = 3.0   # loss ≥ k × credit
    proximity: Optional[float] = 0.01  # |S − K_short| / K_short
    close_frac: float = 0.5            # closes halfway mid → natural


AGENT = Exits()
NO_PROX = Exits(proximity=None)
HOLD = Exits(stop_max_loss=None, hard_mult=None, proximity=None)
STOP2X = Exits(stop_max_loss=None, hard_mult=2.0, proximity=None)
EXITS = {"agent": AGENT, "no-prox": NO_PROX, "tp-only": HOLD, "2x-stop": STOP2X}

SHAPES = [  # (structure, dte, |Δ|, width % of spot)
    ("put", 7, 0.15, 0.02), ("put", 7, 0.25, 0.02),
    ("put", 14, 0.15, 0.02), ("put", 14, 0.25, 0.02),
    ("put", 21, 0.15, 0.02), ("put", 21, 0.25, 0.02), ("put", 21, 0.25, 0.01),
    ("condor", 14, 0.15, 0.02), ("condor", 21, 0.15, 0.02), ("condor", 21, 0.20, 0.02),
]
ENTRY_FRAC = 0.5


def load(data: Path, tk: str, max_dte: int = 26):
    t = pq.read_table(data / f"{tk}_options.parquet",
                      columns=["date", "expiration", "strike", "type", "bid", "ask",
                               "delta", "implied_volatility"]).to_pandas()
    t["type"] = t["type"].astype(str)
    t["dte"] = (t["expiration"] - t["date"]).dt.days
    t = t[(t["dte"] >= 0) & (t["dte"] <= max_dte) & (t["ask"] > 0)]
    t["date"] = t["date"].dt.date
    t["expiration"] = t["expiration"].dt.date
    u = pd.read_parquet(data / f"{tk}_underlying.parquet", columns=["date", "close"])
    u["date"] = pd.to_datetime(u["date"]).dt.date
    u = u.sort_values("date").drop_duplicates("date")
    by_day = {d: g for d, g in t.groupby("date", sort=False)}
    by_leg = {k: g.set_index("date")[["bid", "ask"]] for k, g in t.groupby(["expiration", "strike", "type"], sort=False)}
    return u, by_day, by_leg


def _row(d: pd.DataFrame, k: float):
    r = d.loc[d["strike"] == k]
    return None if r.empty else r.iloc[0]


def vertical(d: pd.DataFrame, opt: str, delta: float, width: float, spot: float) -> Optional[dict]:
    """One credit vertical on expiry slice ``d`` (put: sell higher, buy lower)."""
    x = d[(d["type"] == opt) & (d["bid"] > 0)]
    if x.empty:
        return None
    s = x.iloc[(x["delta"].abs() - delta).abs().argsort().iloc[0]]
    target = s.strike - width * spot if opt == "put" else s.strike + width * spot
    kl = float(min(d.loc[d["type"] == opt, "strike"], key=lambda k: abs(k - target)))
    if kl == s.strike or (kl < s.strike) != (opt == "put"):
        return None
    l = _row(d[d["type"] == opt], kl)
    if l is None or not gate_ok(s.bid, s.ask) or not gate_ok(l.bid, l.ask):
        return None
    nat = s.bid - l.ask
    mid = (s.bid + s.ask) / 2 - (l.bid + l.ask) / 2
    if nat <= 0 or mid <= 0:
        return None
    return dict(opt=opt, ks=float(s.strike), kl=kl, w=abs(s.strike - kl), nat=nat, mid=mid,
                d=abs(float(s.delta)))


def select(day: pd.DataFrame, shape, spot: float) -> Optional[dict]:
    kind, dte, delta, width = shape
    near = day[(day["dte"] - dte).abs() <= 3]
    if near.empty:
        return None
    exp = near.iloc[(near["dte"] - dte).abs().argsort().iloc[0]]["expiration"]
    d = day[day["expiration"] == exp]
    legs = [vertical(d, "put", delta, width, spot)]
    if kind == "condor":
        legs.append(vertical(d, "call", delta, width, spot))
    if any(v is None for v in legs):
        return None
    atm = d.iloc[(d["strike"] - spot).abs().argsort().iloc[:2]]["implied_volatility"].mean()
    mid = sum(v["mid"] for v in legs)
    w = max(v["w"] for v in legs)
    return dict(exp=exp, legs=legs, nat=sum(v["nat"] for v in legs), mid=mid, w=w,
                floor=all(v["mid"] / v["w"] >= v["d"] for v in legs), atm_iv=float(atm))


def simulate(tr: dict, by_leg, closes: Dict[date, float], start: date, ex: Exits) -> Optional[Tuple[float, str, date]]:
    """Return (P&L per $ risked, exit reason, exit session)."""
    frames = []
    for v in tr["legs"]:
        a, b = by_leg.get((tr["exp"], v["ks"], v["opt"])), by_leg.get((tr["exp"], v["kl"], v["opt"]))
        if a is None or b is None:
            return None
        frames.append(a.join(b, lsuffix="_s", rsuffix="_l", how="inner"))
    q = frames[0] if len(frames) == 1 else frames[0].join(frames[1], rsuffix="_c", how="inner")
    q = q[q.index > start].sort_index()
    credit = tr["mid"] - ENTRY_FRAC * (tr["mid"] - tr["nat"])
    risk = tr["w"] - credit
    if risk <= 0:
        return None
    shorts = [v["ks"] for v in tr["legs"]]
    for r in q.itertuples():
        vals = [(r.bid_s, r.ask_s, r.bid_l, r.ask_l)]
        if len(frames) == 2:
            vals.append((r.bid_s_c, r.ask_s_c, r.bid_l_c, r.ask_l_c))
        if any(x < 0 for v in vals for x in v):
            continue
        cmid = sum((bs + as_) / 2 - (bl + al) / 2 for bs, as_, bl, al in vals)
        cnat = sum(as_ - bl for bs, as_, bl, al in vals)
        fill = cmid + ex.close_frac * (cnat - cmid)
        pl_mid, pl_fill = credit - cmid, credit - fill
        spot = closes.get(r.Index)
        last = (tr["exp"] - r.Index).days <= 1
        reason = None
        if ex.hard_mult and -pl_mid >= ex.hard_mult * credit:
            reason = "hard"
        elif ex.stop_max_loss and -pl_mid >= ex.stop_max_loss * risk:
            reason = "stop"
        elif ex.target and pl_mid >= ex.target * credit:
            reason = "target"
        elif ex.proximity and spot and any(abs(spot - k) / k <= ex.proximity for k in shorts):
            reason = "prox"
        elif last:
            reason = "dte"
        if reason:
            return max(-risk, min(credit, pl_fill)) / risk, reason, r.Index
    spot = closes.get(tr["exp"])
    if spot is None:
        return None
    intr = sum(max(0.0, min(v["w"], (v["ks"] - spot) if v["opt"] == "put" else (spot - v["ks"])))
               for v in tr["legs"])
    return (credit - intr) / risk, "expiry", tr["exp"]


def labels(c: List[float], i: int) -> dict:
    rank = rv_percentile(c, i)
    ranks = [rv_percentile(c, j) for j in range(i - 9, i + 1)]
    r_now, r_5 = rv20(c, i), rv20(c, i - 5)
    return dict(rank=rank, rv=r_now,
                receding=bool(max((x or 0) for x in ranks) >= 0.95 and r_now < r_5),
                above200=c[i] > sma(c, i, 200), above50=c[i] > sma(c, i, 50))


def run(data: Path, tickers: List[str]):
    rows = []   # dict per (trade, exits)
    for tk in tickers:
        u, by_day, by_leg = load(data, tk)
        c, dates = u["close"].tolist(), u["date"].tolist()
        closes = dict(zip(dates, c))
        for i in range(221, len(c) - 1):
            day = by_day.get(dates[i])
            if day is None:
                continue
            lab = labels(c, i)
            if lab["rank"] is None:
                continue
            for shape in SHAPES:
                tr = select(day, shape, c[i])
                if tr is None:
                    continue
                vrp = tr["atm_iv"] / lab["rv"] if lab["rv"] else None
                for name, ex in EXITS.items():
                    out = simulate(tr, by_leg, closes, dates[i], ex)
                    if out:
                        rows.append(dict(tk=tk, day=dates[i], shape=shape, exits=name, pnl=out[0],
                                         why=out[1], floor=tr["floor"], vrp=vrp, credit_w=tr["mid"] / tr["w"],
                                         **lab))
        print(f"  {tk}: {len(by_day)} days", flush=True)
    return pd.DataFrame(rows)


def bucket(rank: float) -> str:
    return "<50" if rank < 0.5 else "50-80" if rank < 0.8 else "80-95" if rank < 0.95 else ">=95"


def _cell(x: pd.Series) -> str:
    if len(x) < 20:
        return f"{'n<20':>17}"
    return f"{len(x):5d} {(x > 0).mean():4.0%} {x.mean():+6.3f}"


def report(df: pd.DataFrame):
    df = df.copy()
    df["bucket"] = df["rank"].map(bucket)
    df["era"] = df["day"].map(lambda d: "2008-16" if d.year < 2017 else "2017-25")
    df["shape_s"] = df["shape"].map(lambda s: f"{s[0]} {s[1]}d {s[2]:.2f}Δ {s[3]:.0%}")
    order = ["<50", "50-80", "80-95", ">=95"]
    pd.set_option("display.width", 220)
    for ex in EXITS:
        sub = df[df["exits"] == ex]
        print(f"\n=== exits: {ex} — n / win / avg per $ risked, by RV rank on entry ===")
        print(f"{'shape':24}" + "".join(f"{b:>18}" for b in order) + f"{'all':>18}{'receding':>18}{'2008-16':>18}{'2017-25':>18}")
        for s, g in sub.groupby("shape_s", sort=False):
            cells = [_cell(g.loc[g["bucket"] == b, "pnl"]) for b in order]
            cells += [_cell(g["pnl"]), _cell(g.loc[g["receding"], "pnl"]),
                      _cell(g.loc[g["era"] == "2008-16", "pnl"]), _cell(g.loc[g["era"] == "2017-25", "pnl"])]
            print(f"{s:24}" + "".join(f"{x:>18}" for x in cells))
    sub = df[df["exits"] == "agent"]
    print("\n=== agent exits: by VRP (ATM IV / RV20) ===")
    sub = sub.assign(vrpb=pd.cut(sub["vrp"], [0, 0.9, 1.1, 1.3, 1.6, 99]))
    for s, g in sub.groupby("shape_s", sort=False):
        print(f"{s:24}" + "".join(f"  {str(k):>11} {_cell(gg['pnl'])}" for k, gg in g.groupby("vrpb", observed=True)))
    print("\n=== agent exits: C/W floor passes vs fails, trend, tickers ===")
    for s, g in sub.groupby("shape_s", sort=False):
        print(f"{s:24} floor ok {_cell(g.loc[g['floor'], 'pnl'])}  floor fail {_cell(g.loc[~g['floor'], 'pnl'])}"
              f"  >SMA200 {_cell(g.loc[g['above200'], 'pnl'])}  <SMA200 {_cell(g.loc[~g['above200'], 'pnl'])}")
    for (tk, s), g in sub.groupby(["tk", "shape_s"], sort=False):
        print(f"{tk:4} {s:24} {_cell(g['pnl'])}  >=80 {_cell(g.loc[g['rank'] >= 0.8, 'pnl'])}")
    print("\n=== agent exits: exit reasons (share) ===")
    print(sub.groupby("shape_s", sort=False)["why"].value_counts(normalize=True).unstack().round(2).to_string())
    print("\n=== agent exits, rank >= 80: by year ===")
    hi = sub[sub["rank"] >= 0.8].assign(y=lambda x: x["day"].map(lambda d: d.year))
    print(hi.pivot_table(index="y", columns="shape_s", values="pnl", aggfunc="mean").round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--tickers", default="SPY,QQQ,IWM")
    ap.add_argument("--save", help="write the per-trade rows to this parquet file")
    args = ap.parse_args()
    df = run(Path(args.data), args.tickers.split(","))
    report(df)
    if args.save:
        df.assign(shape=df["shape"].map(str)).to_pickle(args.save)


if __name__ == "__main__":
    main()
