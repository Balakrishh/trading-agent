"""Macro-event, implied-vs-realized and credit-stress study (skill 68).

Three questions, each answered on daily closes (yfinance) before the
agent uses the input live:

1. **Macro event days.** How much larger is each ETF's close-to-close
   move on FOMC / CPI / NFP days (``macro_calendar``) than on other
   days, and what share of a 21-session trade's variance does one event
   day carry?  FOMC 2015 →; CPI / NFP 2025 → (FRED's public calendar).
2. **Implied vs realized (variance risk premium).** For SPY, with VIX as
   the 30-day implied vol: how often is the realized vol of the next 21
   sessions below VIX, and how often does SPY stay inside a ±1σ (VIX)
   21-session range, bucketed by VIX / trailing 20-day realized vol?
   Short premium needs implied > realized, so this sets the
   ``vrp_min_ratio`` floor.
3. **Credit stress.** HYG / IEF (high-yield vs Treasuries, a credit
   spread proxy with history since 2007; FRED's HY OAS has only 3 years)
   — when the ratio sits ≥ k % under its 50-day average, how do SPY's
   next 21 sessions look (return, max drawdown, P(drawdown ≥ 5 %)) —
   overall, and on days SPY is above its 50-day (when the trend rule
   says NORMAL — does credit add anything)?

Limits printed with the output: close-to-close only (FOMC's 14:00 move
and CPI / NFP's 08:30 gap are inside the event day's close-to-close);
VIX is SPY-only implied history — other ETFs use the same rule with
their chain ATM IV live; overlapping 21-session windows (not
independent samples).

    python scripts/research/macro_vrp_credit_study.py [--start 2007-01-01]
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading_agent import macro_calendar as mc  # noqa: E402
from trading_agent.market_state import credit_stress_pct  # noqa: E402
from trading_agent.shadow_pop import realized_vol  # noqa: E402

ETFS = ("SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLU", "XLV", "XLY", "XLI",
        "SMH", "TLT", "GLD", "EEM", "GDX")
HORIZON = 21


def closes(symbol: str, start: str) -> Dict[date, float]:
    import yfinance as yf
    df = yf.Ticker(symbol).history(start=start, interval="1d", auto_adjust=True)
    return {ts.date(): float(c) for ts, c in df["Close"].items() if c == c and c > 0}


def log_returns(px: Dict[date, float]) -> Dict[date, float]:
    days = sorted(px)
    return {d: math.log(px[d] / px[p]) for p, d in zip(days, days[1:])}


# ── 1. Macro event days ───────────────────────────────────────────────

def event_study(rets: Dict[date, float], event_days: Dict[str, set]) -> Dict[str, Tuple[int, float, float]]:
    """kind → (n, mean |r| ratio vs non-event days, mean r² ratio)."""
    all_events = set().union(*event_days.values())
    base = [r for d, r in rets.items() if d not in all_events]
    if not base:
        return {}
    b_abs = statistics.fmean(abs(r) for r in base)
    b_sq = statistics.fmean(r * r for r in base)
    out = {}
    for kind, days in event_days.items():
        ev = [rets[d] for d in days if d in rets]
        if len(ev) >= 8:
            out[kind] = (len(ev), statistics.fmean(abs(r) for r in ev) / b_abs,
                         statistics.fmean(r * r for r in ev) / b_sq)
    return out


# ── 2. Variance risk premium (SPY vs VIX) ─────────────────────────────

def vrp_table(spy: Dict[date, float], vix: Dict[date, float],
              edges: Sequence[float] = (0.0, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 99.0)) -> List[str]:
    days = [d for d in sorted(spy) if d in vix]
    px = [spy[d] for d in days]
    rows = {i: [] for i in range(len(edges) - 1)}
    for i in range(21, len(days) - HORIZON):
        rv_past = realized_vol(px[i - 20:i + 1], 20)
        rv_next = realized_vol(px[i:i + HORIZON + 1], HORIZON)
        iv = vix[days[i]] / 100.0
        if not rv_past or not rv_next or iv <= 0:
            continue
        ratio = iv / rv_past
        move = abs(math.log(px[i + HORIZON] / px[i]))
        one_sigma = iv * math.sqrt(HORIZON / 252)
        for b in rows:
            if edges[b] <= ratio < edges[b + 1]:
                rows[b].append((rv_next < iv, move < one_sigma, iv - rv_next))
                break
    out = ["IV/RV20 bucket     n      P(RV21<IV)  P(|move|<1σ)  mean IV−RV21"]
    for b, xs in rows.items():
        if not xs:
            continue
        n = len(xs)
        out.append(f"[{edges[b]:4.2f}, {edges[b + 1]:5.2f})  {n:6d}   "
                   f"{sum(x[0] for x in xs) / n:9.1%}   {sum(x[1] for x in xs) / n:10.1%}   "
                   f"{statistics.fmean(x[2] for x in xs):+10.3f}")
    return out


# ── 3. Credit stress (HYG / IEF) ──────────────────────────────────────

def forward_stats(px: List[float], i: int) -> Tuple[float, float]:
    """(21-session return, max drawdown from entry close) starting at i."""
    path = px[i:i + HORIZON + 1]
    peak_dd = min(p / path[0] - 1 for p in path)
    return path[-1] / path[0] - 1, peak_dd


def credit_table(spy: Dict[date, float], hyg: Dict[date, float], ief: Dict[date, float],
                 thresholds: Sequence[float] = (1.0, 2.0, 3.0, 5.0)) -> List[str]:
    days = [d for d in sorted(spy) if d in hyg and d in ief]
    sp = [spy[d] for d in days]
    ratio = [hyg[d] / ief[d] for d in days]
    samples = []                         # (stress %, above SMA-50, fwd ret, fwd maxdd)
    for i in range(200, len(days) - HORIZON):
        stress = credit_stress_pct(ratio[:i + 1])
        if stress is None:
            continue
        sma50 = statistics.fmean(sp[i - 49:i + 1])
        r, dd = forward_stats(sp, i)
        samples.append((stress, sp[i] > sma50, r, dd))

    def line(label: str, xs: List[Tuple]) -> str:
        if not xs:
            return f"{label:34s} n=0"
        n = len(xs)
        return (f"{label:34s} n={n:5d}  fwd21 {statistics.fmean(x[2] for x in xs):+6.2%}  "
                f"maxDD {statistics.fmean(x[3] for x in xs):+6.2%}  "
                f"P(DD≤−5%) {sum(x[3] <= -0.05 for x in xs) / n:5.1%}")

    out = [line("all days", samples)]
    for scope, keep in (("", lambda s: True), (" & SPY>SMA50", lambda s: s[1])):
        for k in thresholds:
            out.append(line(f"stress ≥ {k:g}%{scope}", [s for s in samples if s[0] >= k and keep(s)]))
        out.append(line(f"stress < 1%{scope}", [s for s in samples if s[0] < 1 and keep(s)]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default="2007-01-01")
    args = ap.parse_args(argv)

    evs = mc.all_events()
    by_kind = {k: {e.day for e in evs if e.kind == k} for k in mc.KINDS}
    print(f"Events: FOMC {len(by_kind['FOMC'])} (2015→), CPI {len(by_kind['CPI'])}, "
          f"NFP {len(by_kind['NFP'])} (2025→)\n")

    print("1. Event-day move vs other days (ratio of mean |r|, mean r²); "
          f"r² share of a {HORIZON}-session window for one event day = ratio / (ratio + {HORIZON - 1})")
    print(f"{'ticker':7s} " + "  ".join(f"{k:>22s}" for k in mc.KINDS))
    for t in ETFS:
        try:
            rets = {d: r for d, r in log_returns(closes(t, "2015-01-01")).items()}
        except Exception as exc:                       # noqa: BLE001 — one ticker must not stop the study
            print(f"{t:7s} unavailable: {exc}")
            continue
        res = event_study(rets, by_kind)
        cells = []
        for k in mc.KINDS:
            if k in res:
                n, a, s = res[k]
                cells.append(f"n={n:3d} |r|×{a:4.2f} r²×{s:4.2f}")
            else:
                cells.append(f"{'n<8':>22s}")
        print(f"{t:7s} " + "  ".join(f"{c:>22s}" for c in cells))

    print("\n2. SPY: VIX vs the next 21 sessions' realized vol, by VIX / trailing RV20")
    spy = closes("SPY", args.start)
    vix = closes("^VIX", args.start)
    for line in vrp_table(spy, vix):
        print("  " + line)

    print("\n3. Credit stress: HYG/IEF % under its 50-day average → SPY next 21 sessions")
    for line in credit_table(spy, closes("HYG", args.start), closes("IEF", args.start)):
        print("  " + line)
    print("\nLimits: close-to-close; overlapping windows; VIX history is SPY only; "
          "CPI/NFP sample 2025→ only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
