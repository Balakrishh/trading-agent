"""post_capitulation_study.py — sell bull put spreads on the bounce after a panic?

Question (2026-11-20): after a CAPITULATION day (skill 58: VIX ≥ 35, or
VIX/VIX3M ≥ 1.10, or SPY RSI ≤ 25 with VIX ≥ 28) the agent trades nothing
until SPY reclaims its SMA-20 (RECOVERY). Does an earlier trigger — the
VIX term structure back under 1.0, or VIX well off its peak — find
profitable bull put spreads before that?

Real end-of-day SPY / QQQ / IWM chains (credit_chain_backtest: entry halfway
mid → natural, agent stops 50 % of max loss / 3× credit, no proximity exit).
VIX / VIX3M daily closes from yfinance. A day is "post-capitulation" when the
last CAPITULATION day was within ``WINDOW`` sessions and today is not one.
Entries on every such day the trigger holds; one position per ticker at a
time. Baselines: enter right on the CAPITULATION day, and the live RECOVERY
rule (SPY > SMA-20).

    python scripts/research/post_capitulation_study.py [--data ~/data/options_chains]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from credit_chain_backtest import Exits, load, select, simulate  # noqa: E402
from debit_entry_study import sma  # noqa: E402
from put_spread_playbook_backtest import report  # noqa: E402

WINDOW = 30
SHAPES = {"14d 0.15Δ": ("put", 14, 0.15, 0.02), "21d 0.15Δ": ("put", 21, 0.15, 0.02),
          "21d 0.25Δ": ("put", 21, 0.25, 0.02)}
EXITS = {"tp50": Exits(target=0.5, proximity=None), "tp30": Exits(target=0.3, proximity=None)}


def rsi14(c, i):
    if i < 15:
        return None
    gains = [max(0.0, c[j] - c[j - 1]) for j in range(i - 13, i + 1)]
    losses = [max(0.0, c[j - 1] - c[j]) for j in range(i - 13, i + 1)]
    g, l = sum(gains) / 14, sum(losses) / 14
    return 100.0 if l == 0 else 100 - 100 / (1 + g / l)


def market_days(data: Path):
    """date → dict(cap, since_cap, term, vix_off_peak, spy_above20)."""
    u, _, _ = load(data, "SPY")
    c, d = u["close"].tolist(), u["date"].tolist()
    vx = yf.download("^VIX ^VIX3M", start="2007-01-01", progress=False, auto_adjust=False)["Close"]
    vix = {k.date(): v for k, v in vx["^VIX"].dropna().items()}
    v3 = {k.date(): v for k, v in vx["^VIX3M"].dropna().items()}
    out, last_cap, peak = {}, None, 0.0
    for i in range(200, len(c)):
        day = d[i]
        v, t3, r = vix.get(day), v3.get(day), rsi14(c, i)
        if v is None or t3 is None:
            continue
        term = v / t3
        cap = v >= 35 or term >= 1.10 or (r is not None and r <= 25 and v >= 28)
        if cap:
            peak = v if last_cap is None or i - last_cap > WINDOW else max(peak, v)
            last_cap = i
        since = None if last_cap is None else i - last_cap
        out[day] = dict(cap=cap, since=since, term=term, vix=v,
                        off_peak=(1 - v / peak) if peak else 0.0, above20=c[i] > sma(c, i, 20),
                        above200=c[i] > sma(c, i, 200))
    return out


TRIGGERS = {
    "on the CAPITULATION day": lambda m: m["cap"],
    "post-cap, term < 1.00": lambda m: m["term"] < 1.00,
    "post-cap, term < 0.95": lambda m: m["term"] < 0.95,
    "post-cap, VIX ≥ 25% off peak": lambda m: m["off_peak"] >= 0.25,
    "post-cap, term < 1.00 & VIX ≥ 25% off": lambda m: m["term"] < 1.00 and m["off_peak"] >= 0.25,
    "post-cap, live RECOVERY (SPY > SMA20)": lambda m: m["above20"],
}


def in_window(m, trig):
    if trig == "on the CAPITULATION day":
        return m["cap"]
    return not m["cap"] and m["since"] is not None and 0 < m["since"] <= WINDOW


def collect(data: Path, tickers, mkt):
    rows = []
    for tk in tickers:
        u, by_day, by_leg = load(data, tk, max_dte=26)
        c, dates = u["close"].tolist(), u["date"].tolist()
        closes = dict(zip(dates, c))
        for i in range(200, len(c) - 1):
            m = mkt.get(dates[i])
            if m is None or m["since"] is None or m["since"] > WINDOW or dates[i] not in by_day:
                continue
            for sname, shape in SHAPES.items():
                tr = select(by_day[dates[i]], shape, c[i])
                if tr is None:
                    continue
                for ename, ex in EXITS.items():
                    out = simulate(tr, by_leg, closes, dates[i], ex)
                    if out:
                        rows.append(dict(tk=tk, day=dates[i], shape=sname, exits=ename,
                                         pnl=out[0], why=out[1], exit_day=out[2], m=m))
        print(f"  {tk}: done", flush=True)
    return rows


def walk(rows, shape, exits, trig):
    busy, out = defaultdict(lambda: date.min), []
    keep = TRIGGERS[trig]
    for r in sorted((r for r in rows if r["shape"] == shape and r["exits"] == exits),
                    key=lambda r: (r["day"], r["tk"])):
        if r["day"] <= busy[r["tk"]] or not in_window(r["m"], trig) or not keep(r["m"]):
            continue
        out.append((r["tk"], r["day"], r["exit_day"], r["pnl"], r["why"]))
        busy[r["tk"]] = r["exit_day"]
    return out


def episodes(mkt):
    eps, start, prev = [], None, None
    for day in sorted(mkt):
        m = mkt[day]
        if m["cap"] and (prev is None or (m["since"] == 0 and (start is None or (day - prev).days > 45))):
            start = day
            eps.append(day)
        if m["cap"]:
            prev = day
    return eps


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(Path.home() / "data/options_chains"))
    ap.add_argument("--tickers", default="SPY,QQQ,IWM")
    args = ap.parse_args()
    data = Path(args.data)
    mkt = market_days(data)
    eps = episodes(mkt)
    print(f"capitulation episodes (first day, >45 days apart): {len(eps)}")
    print("  " + " ".join(str(e) for e in eps))
    rows = collect(data, args.tickers.split(","), mkt)
    for shape in SHAPES:
        for ename in EXITS:
            print(f"\n=== {shape} 2% put spread, {ename} ===")
            report({t: walk(rows, shape, ename, t) for t in TRIGGERS})


if __name__ == "__main__":
    main()
