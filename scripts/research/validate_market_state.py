"""Replay the market risk-state classifier over SPY / VIX / VIX3M history.

Skill 58 §4 — threshold validation (backlog §1). For every trading day it
rebuilds the same inputs the live agent computes (SPY 20/50/200-day
averages and RSI-14 from daily closes, VIX and VIX3M closes, breadth over
the equity ETFs of the default watchlist), classifies with the previous
day's state, and reports:

* how many days fell in each state, per year;
* for each named stress episode, the first CAUTION / DEFENSIVE /
  CAPITULATION day and SPY's drawdown from its prior high on that day;
* by state, how often SPY then fell ≥ 5 % below that day's close within
  the next 30 calendar days — a proxy for a 30-DTE bull put going wrong.

Research only: downloads from Yahoo, writes nothing, never imported by the
live agent.

    python scripts/research/validate_market_state.py [--start 2019-01-01]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trading_agent import market_state as ms  # noqa: E402

BREADTH = ["QQQ", "IWM", "DIA", "XLF", "XLE", "EEM"]
EPISODES = {
    "2020 COVID crash": ("2020-02-01", "2020-04-30"),
    "2022 bear market": ("2022-01-01", "2022-10-31"),
    "2023 regional banks": ("2023-02-15", "2023-03-31"),
    "2024 Aug carry unwind": ("2024-07-15", "2024-08-31"),
    "2025 tariff drawdown": ("2025-02-15", "2025-05-15"),
}


def _closes(symbol: str, start: str) -> pd.Series:
    import yfinance as yf
    df = yf.Ticker(symbol).history(start=start, auto_adjust=False)
    s = df["Close"].dropna()
    s.index = s.index.tz_localize(None).normalize()
    return s


def replay(start: str):
    warm = (pd.Timestamp(start) - pd.Timedelta(days=330)).strftime("%Y-%m-%d")
    spy, vix, vix3m = _closes("SPY", warm), _closes("^VIX", warm), _closes("^VIX3M", warm)
    others = {t: _closes(t, warm) for t in BREADTH}
    sma50 = {t: s.rolling(50).mean() for t, s in others.items()}
    days = spy.index[spy.index >= pd.Timestamp(start)]
    rows, prior = [], None
    for d in days:
        hist = spy.loc[:d].tolist()
        above = n = 0
        for t, s in others.items():
            if d in s.index and pd.notna(sma50[t].get(d)):
                n += 1
                above += s[d] > sma50[t][d]
        inp = ms.MarketInputs(
            spy_price=hist[-1], spy_sma20=ms._sma(hist, 20), spy_sma50=ms._sma(hist, 50),
            spy_sma200=ms._sma(hist, 200), spy_rsi=ms._rsi(hist),
            vix=float(vix[d]) if d in vix.index else None,
            vix3m=float(vix3m[d]) if d in vix3m.index else None,
            breadth=(above / n) if n else None, breadth_n=n)
        r = ms.classify_market_state(inp, prior)
        prior = r.state
        rows.append({"date": d, "state": r.state, "spy": hist[-1]})
    out = pd.DataFrame(rows).set_index("date")
    out["peak"] = spy.loc[:days[-1]].cummax().reindex(out.index)
    fwd_min = [spy.loc[d + pd.Timedelta(days=1): d + pd.Timedelta(days=30)].min()
               for d in out.index]
    out["fwd30_min_ret"] = pd.Series(fwd_min, index=out.index) / out["spy"] - 1
    return out


def report(df: pd.DataFrame) -> None:
    print("## Days per state by year\n")
    tab = df.groupby([df.index.year, "state"]).size().unstack(fill_value=0)
    print(tab.reindex(columns=[s for s in ms.STATES if s in tab.columns]).to_string(), "\n")

    print("## Stress episodes — first warning day\n")
    for name, (a, b) in EPISODES.items():
        ep = df.loc[a:b]
        if ep.empty:
            continue
        trough = ep["spy"].idxmin()
        dd = ep["spy"].min() / ep["peak"].loc[trough] - 1
        line = [f"{name}: trough {trough.date()} ({dd:.1%} from high)"]
        for st in (ms.CAUTION, ms.DEFENSIVE, ms.CAPITULATION):
            hit = ep[ep["state"] == st]
            if hit.empty:
                line.append(f"{st}: never")
            else:
                d0 = hit.index[0]
                line.append(f"{st}: {d0.date()} (SPY {ep['spy'][d0] / ep['peak'][d0] - 1:+.1%} from high)")
        print(" • ".join(line))
    print()

    print("## SPY falls ≥ 5 % within 30 days, by state (bull-put danger proxy)\n")
    valid = df.dropna(subset=["fwd30_min_ret"])
    for st in ms.STATES:
        g = valid[valid["state"] == st]
        if len(g):
            hit = (g["fwd30_min_ret"] <= -0.05).mean()
            print(f"{st:13s} days={len(g):5d}  P(−5 % within 30d)={hit:6.1%}  "
                  f"median worst={g['fwd30_min_ret'].median():+.1%}")
    flips = Counter(zip(df["state"].shift(), df["state"]))
    changes = sum(v for (a, b), v in flips.items() if a != b and isinstance(a, str))
    print(f"\nState changes: {changes} over {len(df)} days "
          f"({changes / max(1, len(df)) * 252:.1f} per year)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-06-01")
    report(replay(ap.parse_args().start))


if __name__ == "__main__":
    main()
