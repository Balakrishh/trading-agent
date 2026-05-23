"""
account.py — SimAccount: equity + cash ledger for the simulator.

Why this exists
---------------
The live agent reads ``account_balance`` from Alpaca to feed the
RiskManager's max-loss guardrail (#4) and the executor's position
sizing (``calculate_position_qty``). The simulator must answer the same
question — "how much equity do we have right now?" — without an
external broker call. ``SimAccount`` is the answer: a dataclass that
tracks cash + open-position mark-to-market, exposes ``equity`` as the
sum, and offers explicit ``apply_*`` mutators so the runner's state
transitions are auditable.

Conventions
-----------
* All amounts in dollars. No share/contract scaling here — that's the
  caller's job (executor sizing returns ``qty`` of contracts, the
  caller multiplies by ``credit × 100`` before booking).
* ``cash`` includes credits received and debits paid (commissions,
  closing debits). It does *not* include the unrealised P&L of open
  spreads — that's tracked separately in ``open_market_value`` so we
  can decompose realised vs. unrealised on the equity curve.
* ``equity`` is the property the RiskManager reads. It's always
  ``cash + open_market_value`` so an unrealised loss on an open spread
  reduces the headroom for the next trade — same as a real margin
  account.

This is *not* a SimPosition; positions live in
``trading_agent.backtest.sim_position`` and the runner is responsible
for keeping ``open_market_value`` in sync via ``apply_mark()`` after
each re-mark pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)

# Per-leg commission in dollars — same value the legacy backtester
# pinned against (4 legs round-trip = $2.60). Sourced from the
# ``COMMISSION_ROUND_TRIP`` constant the old backtest_ui.py declared
# at the top of the file. Keeping it on a 2-leg-per-side basis here
# (open + close) means the runner can charge it incrementally instead
# of ambushing equity at close.
COMMISSION_PER_LEG = 0.65


@dataclass
class EquityPoint:
    """One point on the equity curve, recorded by the runner per mark."""
    t:                datetime
    cash:             float
    open_market_value: float
    equity:           float
    open_spread_count: int
    realised_pnl:     float
    unrealised_pnl:   float


@dataclass
class SimAccount:
    """
    Mutable cash + equity ledger. Frozen-dataclass would be safer but
    the simulator updates this every cycle and ``dataclasses.replace``
    on a hot path is wasteful — instead we keep mutators explicit and
    discourage external attribute assignment.
    """

    starting_equity:    float = 100_000.0
    cash:               float = 100_000.0
    open_market_value:  float = 0.0
    realised_pnl:       float = 0.0
    open_spread_count:  int   = 0
    equity_curve:       List[EquityPoint] = field(default_factory=list)

    @classmethod
    def fresh(cls, starting_equity: float = 100_000.0) -> "SimAccount":
        """Construct a clean ledger seeded with ``starting_equity`` cash."""
        return cls(
            starting_equity=starting_equity,
            cash=starting_equity,
            open_market_value=0.0,
            realised_pnl=0.0,
            open_spread_count=0,
            equity_curve=[],
        )

    @property
    def equity(self) -> float:
        """``cash + open_market_value`` — the number RiskManager reads."""
        return self.cash + self.open_market_value

    @property
    def unrealised_pnl(self) -> float:
        """Headroom for the live ``unrealised_pl`` reading on the dashboard."""
        return self.open_market_value

    # ------------------------------------------------------------------
    # Mutators — the runner is the only legitimate caller
    # ------------------------------------------------------------------

    def apply_open(self, *, credit_per_share: float, qty: int,
                   spread_width: float,
                   slippage_per_share: float = 0.0,
                   commission_per_leg: float = COMMISSION_PER_LEG) -> float:
        """
        Book an opening fill. Credit lands in cash; the position's
        initial mark-to-market is recorded via ``open_market_value``
        as ``-credit × qty × 100`` so equity is *unchanged* at the
        instant of the fill (we just received credit but owe back the
        spread's current value).

        Returns the dollar credit received (for the caller's logs).
        Charges 2 legs of commission per contract on open.

        Slippage (skill 38, backtester improvement #1)
        ----------------------------------------------
        ``slippage_per_share`` is subtracted from the BS-mid credit
        before booking. Models the gap between BS-fair-value mid and
        the actual fill price (typically 1-3 ticks per leg, i.e.
        $0.10-$0.30 per share for a 2-leg spread). Default 0 keeps
        legacy callers unaffected; the BacktestRunner threads its
        configured value through.

        ``commission_per_leg`` overrides the module-level default so
        the operator can sweep commission models (Alpaca paper $0,
        Alpaca live $0.65, Schwab $0.65, IB $0.50, etc.) without
        editing the constant.
        """
        if qty <= 0:
            return 0.0
        # Slippage reduces the credit we actually pocket. Two legs per
        # credit spread → slippage_per_share applies to the net credit
        # (already in per-spread units, not per-leg).
        effective_credit = max(0.0, credit_per_share - slippage_per_share)
        credit_dollars = effective_credit * qty * 100.0
        commission = commission_per_leg * 2 * qty
        self.cash += credit_dollars - commission
        # The spread is *short* premium — its market value to us is
        # negative until we close it for less than we sold it.
        self.open_market_value -= credit_dollars
        self.open_spread_count += 1
        logger.debug(
            "SimAccount.apply_open: qty=%d credit=$%.2f (slip=$%.2f) "
            "commission=$%.2f → cash=$%.2f equity=$%.2f",
            qty, credit_dollars, slippage_per_share * qty * 100.0,
            commission, self.cash, self.equity,
        )
        return credit_dollars

    def apply_mark(self, *, total_open_market_value: float) -> None:
        """
        Replace the aggregate open-spread mark-to-market with the value
        the runner just computed across every open position. The runner
        sums each position's ``current_mark_dollars`` and passes the
        total here, which keeps SimAccount stateless about positions
        themselves (single-responsibility).
        """
        self.open_market_value = total_open_market_value

    def apply_close(self, *, credit_per_share: float, qty: int,
                    closing_debit_per_share: float,
                    slippage_per_share: float = 0.0,
                    commission_per_leg: float = COMMISSION_PER_LEG) -> float:
        """
        Book a close. We pay back ``closing_debit × qty × 100`` and
        retire that leg of ``open_market_value``.

        Returns the realised P&L for the trade (positive = win),
        already net of commission.

        Slippage (skill 38, backtester improvement #1)
        ----------------------------------------------
        ``slippage_per_share`` is ADDED to the BS-mid debit before
        booking — i.e. we pay MORE to close than fair value suggests,
        symmetric to the open-side slippage. Same default and same
        threading from BacktestRunner as apply_open.

        IMPORTANT — symmetric accounting (2026-05-23 fix):
        ``credit_per_share`` here is the SAME value the caller passed
        to ``apply_open`` (typically ``pos.credit_open``, the BS-mid
        credit). But the open booked only the SLIPPED credit, so the
        close must symmetrically subtract slippage from
        ``credit_per_share`` before computing realised P&L and
        reconciling ``open_market_value``. Otherwise the slippage
        drag shows as half the expected value (only the close-side
        slip is counted; the open-side slip leaks into equity).
        """
        if qty <= 0:
            return 0.0
        # Mirror apply_open's slippage handling so realised P&L
        # accounts for slippage on BOTH legs of the round-trip.
        effective_open_credit = max(0.0, credit_per_share - slippage_per_share)
        credit_dollars = effective_open_credit * qty * 100.0
        # Slippage on close: we pay more to exit than BS mid implies.
        effective_debit = closing_debit_per_share + slippage_per_share
        debit_dollars = effective_debit * qty * 100.0
        commission = commission_per_leg * 2 * qty
        self.cash -= debit_dollars + commission
        # When we close, the position no longer contributes to
        # open_market_value — undo the negative mark we recorded on
        # open. The reversal MUST use the same effective_open_credit
        # the open booked, otherwise equity has a phantom drift equal
        # to slippage × qty × 100.
        self.open_market_value += credit_dollars
        self.open_spread_count = max(0, self.open_spread_count - 1)
        realised = (credit_dollars - debit_dollars) - commission
        self.realised_pnl += realised
        logger.debug(
            "SimAccount.apply_close: qty=%d credit=$%.2f debit=$%.2f "
            "(slip=$%.2f) commission=$%.2f → realised=$%.2f "
            "total_realised=$%.2f cash=$%.2f",
            qty, credit_dollars, debit_dollars,
            slippage_per_share * qty * 100.0,
            commission, realised, self.realised_pnl, self.cash,
        )
        return realised

    # ------------------------------------------------------------------
    # Equity-curve recording
    # ------------------------------------------------------------------

    def snapshot(self, t: datetime) -> EquityPoint:
        """
        Append one point to ``equity_curve`` using the current ledger
        state. Returns the appended point so callers can also stash it.
        """
        pt = EquityPoint(
            t=t,
            cash=round(self.cash, 2),
            open_market_value=round(self.open_market_value, 2),
            equity=round(self.equity, 2),
            open_spread_count=self.open_spread_count,
            realised_pnl=round(self.realised_pnl, 2),
            unrealised_pnl=round(self.unrealised_pnl, 2),
        )
        self.equity_curve.append(pt)
        return pt


__all__ = [
    "SimAccount",
    "EquityPoint",
    "COMMISSION_PER_LEG",
]
