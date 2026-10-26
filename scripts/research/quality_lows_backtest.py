"""Backtest the quality-at-lows price signal (skill 67, backlog §12.4).

Replays ``quality_lows.price_features`` + ``quality_lows.stage`` day by
day over daily bars (the live scanner's own functions — no shadow copy)
and buys the stock at the next open the first session a ticker reaches
``turn`` or ``confirmed``. One open trade per ticker at a time.

Exits (the stock version of skill 67 §2): stop on a close 8 % below the
base low; take profit on a close within 3 % of the 52-week high at entry;
after a +10 % gain, exit on a close below SMA-50; time stop after 126
sessions. Each trade is compared with SPY over the same dates, and fixed
63 / 126 / 252-session forward returns are reported too.

Limits, stated in the output:
* price signal only — quality and drop-cause need point-in-time
  fundamentals and estimates, which there is no free history of, so the
  backtest treats every ticker as quality / sentiment;
* today's universe (survivorship: names that collapsed and left the
  watchlist are missing; names added *because* they recovered are in);
* stocks, not options — no historical option prices; skill 67 §12.5
  judges the option structure on live chain prices in shadow.

    python scripts/research/quality_lows_backtest.py [--years 5] [--tickers NFLX,ORCL]
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading_agent import quality_lows as ql  # noqa: E402

WINDOW = 300            # bars handed to price_features each day (≥ 252)
STOP_BELOW_BASE = 0.08
TARGET_NEAR_HIGH = 0.03
TRAIL_AFTER_GAIN = 0.10
MAX_HOLD = 126


def load(ticker: str, years: int):
    import yfinance as yf
    # Adjusted prices: a 10-for-1 split must not look like a 90 % crash.
    df = yf.Ticker(ticker).history(period=f"{years}y", auto_adjust=True)
    if df is None or df.empty:
        return None
    return df


def simulate(df, spy, cfg=ql.DEFAULT_CONFIG) -> List[Dict]:
    h, l, c, o = (list(map(float, df[k])) for k in ("High", "Low", "Close", "Open"))
    dates = list(df.index)
    spy_close = {d.date(): float(v) for d, v in zip(spy.index, spy["Close"])}
    spy_open = {d.date(): float(v) for d, v in zip(spy.index, spy["Open"])}
    trades: List[Dict] = []
    i = WINDOW
    n = len(c)
    while i < n - 1:
        pf = ql.price_features(h[i - WINDOW + 1:i + 1], l[i - WINDOW + 1:i + 1], c[i - WINDOW + 1:i + 1], cfg)
        st = ql.stage(pf, True, "sentiment", cfg) if pf else "excluded"
        if st not in ("turn", "confirmed"):
            i += 1
            continue
        e = i + 1
        entry = o[e]
        stop = pf["base_low"] * (1 - STOP_BELOW_BASE)
        target = pf["high_52w"] * (1 - TARGET_NEAR_HIGH)
        x, reason = None, None
        for j in range(e, min(n, e + MAX_HOLD)):
            sma50 = sum(c[j - 49:j + 1]) / 50
            if c[j] <= stop:
                x, reason = j, "stop"
            elif c[j] >= target:
                x, reason = j, "near_high"
            elif c[j] >= entry * (1 + TRAIL_AFTER_GAIN) and c[j] < sma50:
                x, reason = j, "trail"
            if x is not None:
                break
        if x is None:
            x, reason = min(n - 1, e + MAX_HOLD - 1), "time" if e + MAX_HOLD - 1 < n else "open"
        d0, d1 = dates[e].date(), dates[x].date()
        s0, s1 = spy_open.get(d0), spy_close.get(d1)
        fwd = {}
        for k in (63, 126, 252):
            if e + k < n:
                sk = spy_close.get(dates[e + k].date())
                r = c[e + k] / entry - 1
                fwd[k] = (r, (sk / s0 - 1) if (sk and s0) else None)
        trades.append({"entry_date": str(d0), "exit_date": str(d1), "reason": reason,
                       "ret": c[x] / entry - 1, "spy": (s1 / s0 - 1) if (s0 and s1) else None,
                       "hold": x - e, "stage": st, "fwd": fwd})
        i = x + 1
    return trades


def _fmt(xs: List[float]) -> str:
    if not xs:
        return "n=0"
    return (f"n={len(xs)} mean {statistics.mean(xs) * 100:+.1f}% median "
            f"{statistics.median(xs) * 100:+.1f}% win {sum(x > 0 for x in xs) / len(xs) * 100:.0f}%")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--tickers", default="")
    args = ap.parse_args(argv)
    tickers = ([t.strip().upper() for t in args.tickers.split(",") if t.strip()]
               if args.tickers else ql.universe())
    spy = load("SPY", args.years)
    all_trades = []
    for t in tickers:
        df = load(t, args.years)
        if df is None or len(df) < WINDOW + 60:
            print(f"{t:6} skipped (history {0 if df is None else len(df)} bars)")
            continue
        tr = simulate(df, spy)
        for x in tr:
            x["ticker"] = t
        all_trades.extend(tr)
        ex = [x["ret"] - x["spy"] for x in tr if x["spy"] is not None]
        print(f"{t:6} trades {len(tr):2}  ret {_fmt([x['ret'] for x in tr])}  "
              f"vs SPY {(statistics.mean(ex) * 100 if ex else float('nan')):+.1f}%")
    closed = [x for x in all_trades if x["reason"] != "open"]
    print()
    print(f"Universe {len(tickers)} tickers, {args.years}y adjusted daily bars; "
          f"{len(all_trades)} signals ({len(closed)} closed)")
    print("Trade returns (rule exits):   ", _fmt([x["ret"] for x in closed]))
    ex = [x["ret"] - x["spy"] for x in closed if x["spy"] is not None]
    print("Excess vs SPY, same dates:    ", _fmt(ex))
    print("Exit reasons:", {r: sum(x["reason"] == r for x in all_trades)
                            for r in ("stop", "near_high", "trail", "time", "open")})
    print("Median hold (sessions):", statistics.median([x["hold"] for x in closed]) if closed else "-")
    for k in (63, 126, 252):
        pairs = [x["fwd"][k] for x in all_trades if k in x["fwd"] and x["fwd"][k][1] is not None]
        print(f"Fixed {k:3}-session forward:  stock {_fmt([p[0] for p in pairs])} | "
              f"excess {_fmt([p[0] - p[1] for p in pairs])}")
    print()
    print("Limits: price signal only (no point-in-time quality / estimates), today's universe "
          "(survivorship), stock returns not options.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
