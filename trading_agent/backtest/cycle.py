"""
cycle.py — one PERCEIVE → CLASSIFY → PLAN → RISK → EXECUTE step.

This is the parity seam: every primitive named below is imported from
the *live* agent module so live and backtest can never drift apart by
construction.

  PERCEIVE     trading_agent.market_data.MarketDataProvider.compute_sma /
               compute_rsi / compute_bollinger_bands / sma_slope
               (static methods — no broker connection needed)
  CLASSIFY     trading_agent.regime.RegimeClassifier._determine_regime
               (instance method, but only reads self.BOLLINGER_NARROW_THRESHOLD)
  PLAN         trading_agent.decision_engine.decide
  RISK         trading_agent.risk_manager.RiskManager.evaluate
  EXECUTE      trading_agent.executor.calculate_position_qty,
               then SimAccount.apply_open

The runner calls ``run_one_cycle`` once per ``intraday_decision`` event
on the clock. The function returns a ``CycleOutcome`` which the runner
inspects to decide whether to journal an entry or move on.

What this function does NOT do
------------------------------
* Network I/O. All data is consumed from ``HistoricalPort`` (which is
  cursor-bound — see ``historical_port.LookaheadError``).
* Sentiment / earnings gating. The live ``IntelligenceConfig`` Tier-2
  gate is intentionally absent here — the legacy backtester also
  skipped it, and the documented residual drift is acknowledged in
  the live↔backtest parity skill.
* Position management. Closing happens in the runner via
  ``SimPosition.evaluate_exit`` + ``SimPosition.close``; this function
  only handles entries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from trading_agent.backtest.account import SimAccount
from trading_agent.backtest.black_scholes import implied_vol
from trading_agent.backtest.historical_port import HistoricalPort
from trading_agent.backtest.sim_position import SimPosition
from trading_agent.backtest.synthetic_chain import (
    build_chain_config_from_preset,
    build_chain_slice,
)
from trading_agent.calendar_utils import next_weekly_expiration
from trading_agent.decision_engine import DecisionInput, decide
from trading_agent.executor import calculate_position_qty
from trading_agent.market_data import MarketDataProvider
from trading_agent.regime import Regime, RegimeClassifier
from trading_agent.risk_manager import RiskManager
from trading_agent.strategy import SpreadLeg, SpreadPlan

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Cycle output
# --------------------------------------------------------------------------

@dataclass
class CycleOutcome:
    """
    Result of one ``run_one_cycle`` call. The runner uses ``status``
    to decide whether to open a position; the other fields surface in
    the journal regardless.
    """
    ticker:        str
    t:             datetime
    spot:          float
    regime:        Regime
    status:        str             # "opened" | "no_candidate" | "risk_rejected" | "no_data"
    reason:        str
    spread_plan:   Optional[SpreadPlan] = None
    qty:           int = 0
    sigma_used:    float = 0.0
    new_position:  Optional[SimPosition] = None


# --------------------------------------------------------------------------
# PERCEIVE — recover regime + ATM IV from historical bars
# --------------------------------------------------------------------------

def _classify_from_history(ticker: str, df: pd.DataFrame,
                           classifier: RegimeClassifier) -> tuple[Regime, str, float]:
    """
    Run the live ``RegimeClassifier._determine_regime`` against the
    historical OHLCV ending at the cursor. Returns
    ``(regime, reasoning, current_price)``.

    We don't go through ``RegimeClassifier.classify`` because that
    method calls Alpaca's snapshot API for the current price; the
    backtester's "current price" is the close of the last bar in the
    cursor-bounded frame.
    """
    close = df["Close"]
    sma_50 = MarketDataProvider.compute_sma(close, 50)
    sma_200 = MarketDataProvider.compute_sma(close, 200)
    upper, middle, lower = MarketDataProvider.compute_bollinger_bands(close, 20, 2.0)
    upper_3std, _, lower_3std = MarketDataProvider.compute_bollinger_bands(close, 20, 3.0)
    sma_50_slope = MarketDataProvider.sma_slope(sma_50, lookback=5)

    current_price = float(close.iloc[-1])
    sma_50_now = float(sma_50.iloc[-1])
    sma_200_now = float(sma_200.iloc[-1])
    bb_width = float((upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1])

    u3 = float(upper_3std.iloc[-1])
    l3 = float(lower_3std.iloc[-1])
    mr_upper = current_price >= u3
    mr_lower = current_price <= l3
    mr_signal = mr_upper or mr_lower
    mr_dir = "upper" if mr_upper else ("lower" if mr_lower else "")

    regime, reasoning = classifier._determine_regime(
        current_price, sma_50_now, sma_200_now, sma_50_slope,
        bb_width, mr_signal, mr_dir, u3, l3,
    )
    return regime, reasoning, current_price


def _atm_iv_proxy(close: pd.Series, window: int = 20) -> float:
    """
    Recover an annualised IV proxy from realised vol — the same trick
    ``RegimeClassifier._compute_iv_rank`` uses internally. ``window``
    is the rolling lookback in bars (20 trading days = ~1 month, the
    canonical "1-month realized vol" window). Returns 0.0 when there
    isn't enough history.
    """
    if len(close) < window + 1:
        return 0.0
    log_returns = np.log(close / close.shift(1)).dropna()
    if log_returns.empty:
        return 0.0
    rolling_std = log_returns.rolling(window=window).std().dropna()
    if rolling_std.empty:
        return 0.0
    # Annualise — 252 trading days a year.
    return float(rolling_std.iloc[-1] * np.sqrt(252.0))


# --------------------------------------------------------------------------
# PLAN — convert a SpreadCandidate into a SpreadPlan the RiskManager understands
# --------------------------------------------------------------------------

def _candidate_to_spread_plan(*,
                              ticker: str,
                              candidate,  # SpreadCandidate (duck-typed)
                              regime: Regime,
                              reasoning: str) -> SpreadPlan:
    """
    Map a ``decide()`` candidate into a ``SpreadPlan`` exactly the way
    ``StrategyPlanner._plan_bull_put`` / ``_plan_bear_call`` do, but
    pure-functionally so we don't have to construct a full
    ``StrategyPlanner``. The mapping is one-to-one — every leg's
    bid/ask/delta survives, the credit and ratio are computed from the
    candidate's own fields.
    """
    opt_type = "put" if candidate.side == "bull_put" else "call"
    legs = [
        SpreadLeg(
            symbol=candidate.short_symbol,
            strike=float(candidate.short_strike),
            action="sell",
            option_type=opt_type,
            delta=float(candidate.short_delta),
            theta=0.0,  # not surfaced by decide(); fine, RM doesn't read theta
            bid=float(candidate.short_bid),
            ask=float(candidate.short_ask),
            mid=round((candidate.short_bid + candidate.short_ask) / 2.0, 4),
        ),
        SpreadLeg(
            symbol=candidate.long_symbol,
            strike=float(candidate.long_strike),
            action="buy",
            option_type=opt_type,
            # Long leg sign mirrors short side; same opt_type both legs.
            delta=-float(candidate.short_delta),
            theta=0.0,
            bid=float(candidate.long_bid),
            ask=float(candidate.long_ask),
            mid=round((candidate.long_bid + candidate.long_ask) / 2.0, 4),
        ),
    ]
    width = float(candidate.width)
    credit = float(candidate.credit)
    max_loss = round((width - credit) * 100.0, 2)
    strategy_name = ("Bull Put Spread" if candidate.side == "bull_put"
                     else "Bear Call Spread")
    return SpreadPlan(
        ticker=ticker,
        strategy_name=strategy_name,
        regime=regime.value,
        legs=legs,
        spread_width=width,
        net_credit=round(credit, 2),
        max_loss=max_loss,
        credit_to_width_ratio=round(credit / width, 4) if width > 0 else 0.0,
        expiration=candidate.expiration,
        reasoning=reasoning[:240],
    )


# --------------------------------------------------------------------------
# Side selection — same regime → side mapping the live planner uses
# --------------------------------------------------------------------------

def _side_for_regime(regime: Regime) -> Optional[str]:
    """
    Returns the spread side the live ``StrategyPlanner.plan`` would pick
    for this regime, or ``None`` when the regime maps to a strategy the
    backtester doesn't yet replicate (Iron Condor / Mean Reversion).

    Skipping IC / MR is a deliberate scope cut for the rewrite — both
    strategies need additional plumbing (two-side decide() composition
    for IC, separate target-Δ logic for MR) and the user's brief was
    "credit-spread P&L parity", which the two vertical sides cover.
    """
    if regime == Regime.BULLISH:
        return "bull_put"
    if regime == Regime.BEARISH:
        return "bear_call"
    return None


# --------------------------------------------------------------------------
# Top-level cycle
# --------------------------------------------------------------------------

def run_one_cycle(*,
                  ticker: str,
                  t: datetime,
                  port: HistoricalPort,
                  preset,
                  account: SimAccount,
                  classifier: RegimeClassifier,
                  risk_manager: RiskManager,
                  ) -> CycleOutcome:
    """
    Run one PERCEIVE → CLASSIFY → PLAN → RISK → EXECUTE step.

    Returns a ``CycleOutcome``. The runner reads ``status`` to decide
    whether a new ``SimPosition`` was opened (``"opened"``); other
    statuses are journalled but don't change the equity curve.

    Order of operations matches ``agent.py``'s per-ticker block so the
    failure modes line up — e.g. "no_data" mirrors the live agent's
    ``InsufficientDataError`` skip; "risk_rejected" mirrors the live
    RiskManager veto path.
    """
    # PERCEIVE — load history, compute regime + IV proxy
    df = port.fetch_underlying_daily(ticker)
    if df is None or df.empty or len(df) < 200:
        return CycleOutcome(ticker=ticker, t=t, spot=0.0,
                            regime=Regime.SIDEWAYS,
                            status="no_data",
                            reason=f"insufficient_history: {0 if df is None else len(df)} bars")
    regime, reasoning, spot = _classify_from_history(ticker, df, classifier)
    sigma_proxy = _atm_iv_proxy(df["Close"])
    if sigma_proxy <= 0:
        return CycleOutcome(ticker=ticker, t=t, spot=spot, regime=regime,
                            status="no_data",
                            reason="iv_proxy_unavailable")

    # CLASSIFY → side
    side = _side_for_regime(regime)
    if side is None:
        return CycleOutcome(ticker=ticker, t=t, spot=spot, regime=regime,
                            status="no_candidate",
                            reason=f"regime_{regime.value}_unsupported_side")

    # PLAN — synth chain across the preset DTE grid, decide()
    cfg = build_chain_config_from_preset(side, preset)
    chain_slices = []
    for dte_target in preset.dte_grid:
        # Pick the same weekly expiration the live planner would.
        dte_min, dte_max = (max(1, dte_target - preset.dte_window_days),
                            dte_target + preset.dte_window_days)
        try:
            exp = next_weekly_expiration(t.date(), dte_target, dte_min, dte_max)
        except Exception as exc:  # pragma: no cover — calendar should always work
            logger.warning("[%s] expiration pick failed: %s", ticker, exc)
            continue
        if exp <= t.date():
            continue
        chain_slices.append(build_chain_slice(
            ticker=ticker, side=side, spot=spot,
            sigma_annual=sigma_proxy, now=t.date(), expiration=exp,
            cfg=cfg,
        ))
    if not chain_slices:
        return CycleOutcome(ticker=ticker, t=t, spot=spot, regime=regime,
                            status="no_candidate",
                            reason="no_resolvable_expirations")

    output = decide(DecisionInput(side=side, chain_slices=chain_slices,
                                  preset=preset),
                    max_candidates=1)
    if not output.candidates:
        top_reason = max(output.diagnostics.rejects_by_reason.items(),
                         key=lambda kv: kv[1], default=("unknown", 0))[0]
        return CycleOutcome(ticker=ticker, t=t, spot=spot, regime=regime,
                            status="no_candidate",
                            reason=f"decide_no_candidate_{top_reason}")
    candidate = output.candidates[0]
    plan = _candidate_to_spread_plan(
        ticker=ticker, candidate=candidate, regime=regime, reasoning=reasoning,
    )

    # RISK
    verdict = risk_manager.evaluate(
        plan,
        account_balance=account.equity,
        account_type="paper",
        market_open=True,        # backtester runs only inside RTH already
        # Underlying liquidity gate — synthesize a tight bid/ask around spot
        # so the gate passes; the live agent reads NBBO from Alpaca, but
        # historically we assume liquid index ETFs.
        underlying_bid_ask=(round(spot - 0.01, 2), round(spot + 0.01, 2)),
        # No buying-power check in backtest (no margin metadata to read).
    )
    if not verdict.approved:
        return CycleOutcome(ticker=ticker, t=t, spot=spot, regime=regime,
                            status="risk_rejected",
                            reason=verdict.checks_failed[0] if verdict.checks_failed else "unknown",
                            spread_plan=plan)

    # EXECUTE — sizing via the live primitive, then book the open
    qty = calculate_position_qty(plan, account_balance=account.equity,
                                 max_risk_pct=risk_manager.max_risk_pct,
                                 live_credit=plan.net_credit)
    if qty <= 0:
        return CycleOutcome(ticker=ticker, t=t, spot=spot, regime=regime,
                            status="risk_rejected",
                            reason="qty_zero_after_sizing",
                            spread_plan=plan, qty=0)

    # Recover σ_entry from the synthetic credit so re-marks scale from
    # the same number the candidate was priced at. The synthetic chain
    # used ``sigma_proxy`` to build mids; using that value directly as
    # σ_entry is exact (no IV inversion needed).
    sigma_entry = sigma_proxy
    vix_entry = port.vix_at(t) or 0.0

    expiration_dt = date.fromisoformat(plan.expiration)
    position = SimPosition(
        ticker=ticker, side=side,
        short_strike=float(candidate.short_strike),
        long_strike=float(candidate.long_strike),
        spread_width=float(candidate.width),
        expiration=expiration_dt,
        qty=qty,
        credit_open=float(candidate.credit),
        sigma_entry=sigma_entry,
        vix_entry=vix_entry,
        entry_t=t,
        short_delta_entry=float(candidate.short_delta),
        current_mark=float(candidate.credit),  # debit-to-close = credit-collected at t=0
        current_short_delta=float(candidate.short_delta),
        current_t=t,
        sigma_current=sigma_entry,
        vix_current=vix_entry,
    )
    account.apply_open(credit_per_share=float(candidate.credit),
                       qty=qty,
                       spread_width=float(candidate.width))

    return CycleOutcome(
        ticker=ticker, t=t, spot=spot, regime=regime,
        status="opened",
        reason=f"opened {plan.strategy_name} qty={qty} cr=${candidate.credit:.2f}",
        spread_plan=plan, qty=qty,
        sigma_used=sigma_proxy, new_position=position,
    )


__all__ = [
    "CycleOutcome",
    "run_one_cycle",
]
