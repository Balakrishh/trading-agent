"""debit_policy.py — single source of truth for debit structures (skill 59).

Backlog §6.3 (call / put debit spreads for trend + low volatility), §6.4
(bounce bull put) and §6.5 (calendar spreads for sideways + low
volatility). The planner, RiskManager, executor, position monitor and
market-state gates import the names and helpers below instead of
re-declaring them.

Sign convention: a debit plan's ``net_credit`` is **negative** (−debit),
``max_loss`` is the debit × 100 (all you can lose on a long spread), and
``max_debit`` caps what the executor may pay at submission.
"""
from __future__ import annotations

from typing import Any, Optional

from trading_agent.strategy import SpreadLeg, SpreadPlan

CALL_DEBIT_STRATEGY = "Call Debit Spread"
PUT_DEBIT_STRATEGY = "Put Debit Spread"
CALENDAR_STRATEGY = "Calendar Spread"
# A bull put sold after an oversold bounce is still a credit spread — it
# keeps the credit exit rules but gets its own name so the market-state
# gate, the regime-shift exit and the 6.6 scorecard can tell it apart.
BOUNCE_BULL_PUT_STRATEGY = "Bounce Bull Put Spread"

DEBIT_VERTICALS = frozenset({CALL_DEBIT_STRATEGY, PUT_DEBIT_STRATEGY})
DEBIT_STRATEGIES = DEBIT_VERTICALS | {CALENDAR_STRATEGY}

KIND_TO_STRATEGY = {
    "call_debit": CALL_DEBIT_STRATEGY,
    "put_debit": PUT_DEBIT_STRATEGY,
    "calendar": CALENDAR_STRATEGY,
}


def is_debit_plan(plan: Any) -> bool:
    """True for a debit structure (by name, or a negative net credit)."""
    return (getattr(plan, "strategy_name", "") in DEBIT_STRATEGIES
            or float(getattr(plan, "net_credit", 0.0) or 0.0) < 0)


def plan_debit(plan: Any, live_net: Optional[float] = None) -> float:
    """Debit per share (positive) of a debit plan, from a signed net
    credit (``live_net`` when given, else the plan's)."""
    net = plan.net_credit if live_net is None else live_net
    return round(-float(net), 2)


def build_debit_plan(*, ticker: str, regime: str, cand: Any,
                     reasoning: str = "") -> SpreadPlan:
    """SpreadPlan for a scored ``DebitCandidate`` (decision_engine):
    buy the long leg, sell the short leg."""
    strategy = KIND_TO_STRATEGY[cand.kind]
    debit = round(float(cand.debit), 2)

    def leg(symbol, strike, action, delta, bid, ask):
        return SpreadLeg(symbol=symbol, strike=float(strike), action=action,
                         option_type=cand.option_type, delta=float(delta), theta=0.0,
                         bid=float(bid), ask=float(ask), mid=round((bid + ask) / 2, 4))

    width = float(cand.width)
    return SpreadPlan(
        ticker=ticker, strategy_name=strategy, regime=regime,
        legs=[leg(cand.long_symbol, cand.long_strike, "buy", cand.long_delta,
                  cand.long_bid, cand.long_ask),
              leg(cand.short_symbol, cand.short_strike, "sell", cand.short_delta,
                  cand.short_bid, cand.short_ask)],
        spread_width=width,
        net_credit=-debit,
        max_loss=round(debit * 100, 2),
        credit_to_width_ratio=round(-debit / width, 4) if width else 0.0,
        expiration=cand.expiration,
        reasoning=reasoning,
        max_debit=round(float(cand.max_debit), 2),
        far_expiration=getattr(cand, "far_expiration", "") or "",
    )
