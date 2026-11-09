"""liquidity_check.py — would the agent's leg gate let a ticker trade? (read-only)

For each ticker, through the local data server's ``/chain`` (no Schwab
session of its own), at the expiries the agent would use (~30 DTE for
debit verticals, ~21 DTE for credit verticals) it reads the legs the
agent would pick — the 0.50Δ call / put (debit long leg) and the
0.20–0.25Δ put / call (credit short leg) — and reports their bid / ask
width in cents and % of mid, open interest, and whether each passes the
live leg gate (rejected only when wider than BOTH ``max_leg_spread_cents``
and ``max_leg_spread_pct_mid``). It also prices the cheapest call debit
the agent's width grid allows and checks the fair-price rule (natural ≤
mid × 1.05) and the per-trade budget.

Run in the regular session (after 09:45 ET); pre-market option quotes
are wide and meaningless.

    python scripts/research/liquidity_check.py --tickers GOOGL,MSFT,XLC,XLP,XLB [--budget 570]
"""
from __future__ import annotations

import argparse
import os
from datetime import date
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(os.path.join(os.getcwd(), ".env"))

from trading_agent.calendar_utils import next_weekly_expiration  # noqa: E402
from trading_agent.mcp.tools.market import _data_server_get  # noqa: E402
from trading_agent.strategy_presets import load_active_preset  # noqa: E402


def chain(tk: str, exp: date, opt: str) -> List[Dict]:
    r = _data_server_get(f"chain/{tk}?expiration={exp.isoformat()}&option_type={opt}", timeout=15) or {}
    return [c for c in r.get("contracts") or [] if c.get("bid", 0) > 0 and c.get("ask", 0) > 0
            and c.get("delta") is not None]


def leg(cs: List[Dict], target: float) -> Optional[Dict]:
    return min(cs, key=lambda c: abs(abs(c["delta"]) - target)) if cs else None


def gate(c: Dict, pre) -> bool:
    w, mid = c["ask"] - c["bid"], (c["ask"] + c["bid"]) / 2
    return not (w > pre.max_leg_spread_cents and w > pre.max_leg_spread_pct_mid * mid)


def describe(c: Optional[Dict], pre) -> str:
    if c is None:
        return "no quote"
    w, mid = c["ask"] - c["bid"], (c["ask"] + c["bid"]) / 2
    oi = c.get("open_interest")
    return (f"K{c['strike']:g} Δ{abs(c['delta']):.2f} {c['bid']:.2f}/{c['ask']:.2f} "
            f"{w * 100:.0f}¢ {w / mid:.1%}{'' if oi is None else f' OI {oi:,}'} "
            f"{'PASS' if gate(c, pre) else 'FAIL'}")


def call_debit(calls: List[Dict], pre, budget: float) -> str:
    long_c = leg(calls, 0.50)
    if long_c is None or not gate(long_c, pre):
        return "long leg fails the gate"
    best = None
    for w in pre.width_grid_pct:
        target = long_c["strike"] * (1 + w)
        short_c = min(calls, key=lambda c: abs(c["strike"] - target))
        if short_c["strike"] <= long_c["strike"] or not gate(short_c, pre):
            continue
        nat = long_c["ask"] - short_c["bid"]
        mid = (long_c["bid"] + long_c["ask"]) / 2 - (short_c["bid"] + short_c["ask"]) / 2
        width = short_c["strike"] - long_c["strike"]
        ok = nat <= mid * (1 + pre.debit_max_overpay) and (width - nat) / nat >= pre.debit_min_reward_risk
        row = (nat * 100 <= budget, ok, -width, nat, mid, width)
        if best is None or row > best:
            best = row
    if best is None:
        return "no short leg passes"
    fits, ok, _, nat, mid, width = best
    return (f"${width:g} wide: natural ${nat:.2f} vs mid ${mid:.2f} "
            f"({nat / mid - 1:+.1%}) {'fair' if ok else 'NOT fair'}, "
            f"{'fits' if fits else 'over'} the ${budget:,.0f} budget")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tickers", required=True)
    ap.add_argument("--budget", type=float, default=570.0, help="per-trade max loss, $")
    args = ap.parse_args()
    pre = load_active_preset()
    today = date.today()
    e30 = next_weekly_expiration(today=today, target_dte=30, dte_min=25, dte_max=35)
    e21 = next_weekly_expiration(today=today, target_dte=21, dte_min=16, dte_max=26)
    print(f"gate: reject when wider than ${pre.max_leg_spread_cents:.2f} AND "
          f"{pre.max_leg_spread_pct_mid:.0%} of mid | expiries {e30} (debit) / {e21} (credit)\n")
    for tk in [t.strip().upper() for t in args.tickers.split(",") if t.strip()]:
        spot = (_data_server_get(f"price/{tk}") or {}).get("price")
        c30, p30, p21, c21 = chain(tk, e30, "call"), chain(tk, e30, "put"), chain(tk, e21, "put"), chain(tk, e21, "call")
        print(f"{tk}  spot {spot}")
        print(f"   call 0.50Δ {e30}: {describe(leg(c30, 0.50), pre)}")
        print(f"   put  0.50Δ {e30}: {describe(leg(p30, 0.50), pre)}")
        print(f"   put  0.25Δ {e21}: {describe(leg(p21, 0.25), pre)}")
        print(f"   call 0.20Δ {e21}: {describe(leg(c21, 0.20), pre)}")
        print(f"   call debit: {call_debit(c30, pre, args.budget)}")


if __name__ == "__main__":
    main()
