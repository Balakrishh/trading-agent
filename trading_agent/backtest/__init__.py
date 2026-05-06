"""
trading_agent.backtest — historical replay package.

Goal: replay the *exact* live decision pipeline against historical bars
without duplicating any of its primitives. The package is intentionally
small; its only job is to **simulate the inputs** the live agent
receives (equity, market data, synthetic chains) and to **wire those
inputs through the live primitives** (``decide()``, ``RiskManager``,
``calculate_position_qty``, ``PositionMonitor`` exit logic). Order
acceptance is assumed: if a plan clears risk, the simulator books it at
the haircut credit and re-marks every bar.

Module layout
-------------
``black_scholes``      Pure BS pricing + Greeks (delta, gamma, theta, vega).
                       No project deps. Vectorised when given numpy arrays.
``clock``              Calendar-aware timeline iterator
                       (NYSE trading days × intraday bar times).
``historical_port``    Wraps ``MarketDataProvider`` + yfinance with a
                       hard cursor — accessing bars past ``now_t``
                       raises ``LookaheadError``.
``synthetic_chain``    Builds ``ChainSlice`` (the dict shape
                       ``decide()`` consumes) from a spot price + IV
                       curve + the preset's strike grid.
``account``            ``SimAccount`` — frozen-dataclass-style cash
                       + equity ledger with ``apply_fill`` /
                       ``apply_mark`` mutators.
``sim_position``       ``SimPosition`` — open spread bookkeeping with
                       VIX-proxy IV scaling for daily marks.
``cycle``              ``run_one_cycle`` — single PERCEIVE → CLASSIFY
                       → PLAN → RISK → EXECUTE step. Calls the live
                       ``decide()`` / ``RiskManager`` / executor sizing.
``runner``             ``BacktestRunner`` — drives the clock and emits
                       a ``BacktestResult`` (equity curve + closed
                       trades) for the UI to render.

Live↔backtest parity invariants — see ``docs/skills/15_backtest_live_parity.md``.
"""

from __future__ import annotations

from trading_agent.backtest.black_scholes import (
    bs_price,
    bs_delta,
    bs_theta,
    bs_gamma,
    bs_vega,
    implied_vol,
    norm_cdf,
)
from trading_agent.backtest.account import SimAccount
from trading_agent.backtest.sim_position import SimPosition, ClosedTrade
from trading_agent.backtest.historical_port import HistoricalPort, LookaheadError

__all__ = [
    "bs_price",
    "bs_delta",
    "bs_theta",
    "bs_gamma",
    "bs_vega",
    "implied_vol",
    "norm_cdf",
    "SimAccount",
    "SimPosition",
    "ClosedTrade",
    "HistoricalPort",
    "LookaheadError",
]
