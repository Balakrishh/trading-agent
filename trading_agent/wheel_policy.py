"""wheel_policy.py — single source of truth for Wheel trade handling (skill 40 §2.9).

The Wheel is two single-leg credit trades:
  * Cash-Secured Put (CSP) — sell a put you are willing to be assigned on.
  * Covered Call (CC)      — once assigned, sell calls against the shares.

Every module that stages, submits, monitors or reconciles those trades
imports the names below instead of re-declaring them, so the evaluator's
exit anchors, the position monitor's exit rules and the promote CLI's
checks cannot drift apart.
"""
from __future__ import annotations

from trading_agent.chain_scanner import _quote_credit_single
from trading_agent.strategy import SpreadLeg, SpreadPlan

CSP_STRATEGY = "Cash-Secured Put"
CC_STRATEGY = "Covered Call"
WHEEL_STRATEGIES = frozenset({CSP_STRATEGY, CC_STRATEGY})

# Exit anchors (skill 40 §2.6).
TAKE_PROFIT_PCT_OF_CREDIT = 0.50     # buy back at 50 % of the credit
CSP_STOP_ABS_DELTA = 0.45            # close a CSP once |Δ| reaches this

# Promote-time guard: one CSP may not tie up more than this share of equity.
MAX_CSP_COLLATERAL_PCT_OF_EQUITY = 0.40

# Journal exit_signal values written when an expired Wheel leg is reconciled
# against broker share holdings (wheel_lifecycle.py).
EXIT_ASSIGNED = "assigned"                  # CSP expired ITM → shares delivered
EXIT_CALLED_AWAY = "called_away"            # CC expired ITM → shares delivered away
EXIT_EXPIRED_WORTHLESS = "expired_worthless"


def build_single_leg_plan(
    *,
    ticker: str,
    strategy_name: str,
    symbol: str,
    strike: float,
    option_type: str,
    delta: float,
    bid: float,
    ask: float,
    expiration: str,
    reasoning: str = "",
) -> SpreadPlan:
    """SpreadPlan for one short option (the shape the executor, trade-plan
    file and position monitor already understand).

    ``spread_width`` carries the per-share collateral (the strike) so the
    existing ``credit_to_width_ratio`` column stays meaningful:
    credit / strike = premium yield on collateral. ``max_loss`` for a CSP
    is (strike − credit) × 100 — the stock going to zero. A covered call's
    loss lives in the shares it covers, so its option-level max_loss is 0.
    """
    if strategy_name not in WHEEL_STRATEGIES:
        raise ValueError(f"not a Wheel strategy: {strategy_name!r}")
    credit = round(_quote_credit_single(bid=float(bid), ask=float(ask)), 2)
    strike = float(strike)
    max_loss = round((strike - credit) * 100, 2) if strategy_name == CSP_STRATEGY else 0.0
    return SpreadPlan(
        ticker=ticker.upper(),
        strategy_name=strategy_name,
        regime="wheel",
        legs=[SpreadLeg(
            symbol=symbol, strike=strike, action="sell",
            option_type=option_type, delta=float(delta), theta=0.0,
            bid=float(bid), ask=float(ask), mid=round((float(bid) + float(ask)) / 2, 4),
        )],
        spread_width=strike,
        net_credit=credit,
        max_loss=max_loss,
        credit_to_width_ratio=round(credit / strike, 4) if strike else 0.0,
        expiration=expiration,
        reasoning=reasoning,
    )
