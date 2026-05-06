"""
sim_position.py — open spread bookkeeping with VIX-proxy IV scaling.

Why VIX-proxy IV scaling
------------------------
The simulator opens spreads at σ_entry (the IV implied by today's
synthetic chain — see ``synthetic_chain.build_chain_slice``). On every
later re-mark, we *don't* have a fresh option chain to read σ from.
Three options were on the table:

  (a) Hold σ_entry constant → P&L is pure delta-drift, ignores
      vol-regime shifts. Wrong: a spread that *should* explode in price
      during a vol spike never moves on the simulator's books.
  (b) Re-imply σ from the next-day chain → no chain exists, would
      require synthesizing the entire chain again with some other σ
      proxy. Circular.
  (c) Scale σ_entry by the realised VIX move:
        σ_t = σ_entry × (vix_t / vix_entry)
      This is the standard "VIX is roughly proportional to realised
      ATM IV on SPY/QQQ" assumption — accurate enough for short-dated
      index spreads, requires no extra data beyond ^VIX, and is
      directionally correct (IV rises with vol, falls when it cools).

We chose (c). The scaling lives entirely on this class because re-marking
is the *only* place σ matters after entry. The runner asks the
``HistoricalPort`` for today's VIX close, calls ``remark()``, and the
position re-prices via Black-Scholes with the scaled σ.

ClosedTrade
-----------
The runner emits a ``ClosedTrade`` per closed spread to the result's
trade log. The dashboard renders these in a sortable DataFrame and the
journal exporter writes one JSONL row per trade keyed on
``run_mode="backtest"``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from trading_agent.backtest.black_scholes import bs_price, bs_delta
from trading_agent.position_monitor import ExitSignal, STRATEGY_REGIME_MAP
from trading_agent.regime import Regime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClosedTrade:
    """
    Final state of one fully-closed spread. Sortable on every field
    so the UI can offer "show me the worst losers", "best by EV",
    etc. without re-fetching.
    """
    ticker:           str
    side:             str           # "bull_put" | "bear_call"
    entry_t:          datetime
    exit_t:           datetime
    expiration:       date
    short_strike:     float
    long_strike:      float
    spread_width:     float
    credit_open:      float         # per-share credit collected
    debit_close:      float         # per-share debit paid to close
    qty:              int
    realised_pnl:     float         # dollars, net of commission
    exit_signal:      str           # ExitSignal.value (e.g. "profit_target")
    exit_reason:      str
    sigma_entry:      float
    sigma_exit:       float
    days_held:        int


@dataclass
class SimPosition:
    """
    One open vertical credit spread on the simulator's books.

    The position is identified by ticker + side + strikes; we don't
    bother with a separate position-id because two simultaneous opens
    on the same ticker+strikes would collapse to one in the live
    position monitor too (Alpaca aggregates by symbol).
    """
    ticker:        str
    side:          str           # "bull_put" | "bear_call"
    short_strike:  float
    long_strike:   float
    spread_width:  float
    expiration:    date
    qty:           int
    credit_open:   float          # per-share credit collected at fill
    sigma_entry:   float          # IV implied at fill time
    vix_entry:     float          # ^VIX close on entry day (or 0 if unavailable)
    entry_t:       datetime
    short_delta_entry: float = 0.0  # signed Δshort at fill, for diagnostics

    # Mutable state — updated by remark() each cycle:
    current_mark:        float = 0.0   # per-share mid value of the spread (debit)
    current_short_delta: float = 0.0
    current_t:           datetime = field(default=datetime.utcnow())
    sigma_current:       float = 0.0   # last σ used in BS pricing
    vix_current:         float = 0.0   # last VIX read

    # Closing state — populated by close():
    closed:           bool = False
    debit_close:      Optional[float] = None
    realised_pnl:     Optional[float] = None
    exit_signal:      Optional[ExitSignal] = None
    exit_reason:      str = ""
    exit_t:           Optional[datetime] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def opt_type(self) -> str:
        return "put" if self.side == "bull_put" else "call"

    @property
    def short_long_signs(self) -> tuple[int, int]:
        """(+1, -1) — short leg debit positive, long leg debit negative."""
        return (+1, -1)

    @property
    def current_mark_dollars(self) -> float:
        """
        Total dollar value of the open spread to *us* (the seller).

        We sold the spread for credit; the spread's market debit
        (``current_mark × qty × 100``) is what we'd pay to close. The
        unrealised P&L vs. entry is therefore
        ``(credit_open − current_mark) × qty × 100``. We return the
        signed value the SimAccount expects for ``open_market_value``:
        ``-current_mark × qty × 100`` (a debit liability).
        """
        return -self.current_mark * self.qty * 100.0

    @property
    def days_held(self) -> int:
        if self.exit_t is None:
            return max(0, (self.current_t.date() - self.entry_t.date()).days)
        return max(0, (self.exit_t.date() - self.entry_t.date()).days)

    # ------------------------------------------------------------------
    # Re-mark — VIX-proxy IV scaling, then BS price both legs
    # ------------------------------------------------------------------

    def remark(self, *, t: datetime, spot: float,
               vix_t: Optional[float], r: float = 0.0) -> None:
        """
        Re-price the spread at time ``t`` using:

          σ_t = σ_entry × (vix_t / vix_entry)

        falling back to ``σ_entry`` when ``vix_t`` or ``vix_entry`` is
        unavailable / zero. The scaled σ is then fed into BS for both
        legs, the spread mid is the difference, and ``current_mark``
        + ``current_short_delta`` are updated.

        Spot must be > 0; ``t`` must be on-or-after entry. Past
        expiration the BS formula collapses to intrinsic, which is the
        correct settlement value (caller still has to call ``close()``
        to realise it on the ledger).
        """
        if spot <= 0:
            logger.warning("[%s] SimPosition.remark: non-positive spot=%.4f, "
                           "skipping mark", self.ticker, spot)
            return
        # Time to expiration in years for BS (clip to 0 at/after expiry).
        dte_days = max(0, (self.expiration - t.date()).days)
        t_years = dte_days / 365.0

        # σ scaling — only when we have both endpoints. Falls back to
        # σ_entry rather than zero so a missing VIX read can't accidentally
        # wipe the spread to intrinsic mid-life.
        if vix_t is not None and vix_t > 0 and self.vix_entry > 0:
            sigma_t = self.sigma_entry * (vix_t / self.vix_entry)
            # Clip to a sane band — extreme VIX prints during liquidity
            # crises (e.g. Mar 2020 intraday spikes to 80+) shouldn't
            # let σ blow past 5.0 and start over-pricing OTM puts.
            sigma_t = min(5.0, max(0.01, sigma_t))
        else:
            sigma_t = self.sigma_entry

        short_price = bs_price(
            spot, self.short_strike, t_years, sigma_t, r=r,
            option_type=self.opt_type,
        )
        long_price = bs_price(
            spot, self.long_strike, t_years, sigma_t, r=r,
            option_type=self.opt_type,
        )
        # Debit to close = short − long for credit spreads (we buy back
        # the short leg, sell off the long leg).
        debit = max(0.0, short_price - long_price)
        self.current_mark = round(debit, 4)
        self.current_short_delta = round(
            bs_delta(spot, self.short_strike, t_years, sigma_t, r=r,
                     option_type=self.opt_type),
            4,
        )
        self.current_t = t
        self.sigma_current = round(sigma_t, 4)
        self.vix_current = round(vix_t or 0.0, 4)

    # ------------------------------------------------------------------
    # Exit-signal evaluation — mirrors PositionMonitor._check_exit
    # ------------------------------------------------------------------

    def evaluate_exit(self, *, t: datetime, spot: float,
                      current_regime: Optional[Regime],
                      profit_target_pct: float = 0.50,
                      hard_stop_multiplier: float = 3.0,
                      strike_proximity_pct: float = 0.01,
                      ) -> tuple[ExitSignal, str]:
        """
        Replicate the live ``PositionMonitor._check_exit`` rule set
        against the simulator's last re-mark. Returns
        ``(ExitSignal, reason)`` exactly as the live monitor does so
        the runner's bookkeeping is interchangeable.

        Rule order matches PositionMonitor:
          1. Hard stop (loss ≥ 3× credit)
          2. Profit target (gain ≥ 50% of credit)
          3. Strike proximity (underlying within 1% of short strike)
          4. DTE safety (expiration is the next trading day)
          5. Regime shift (current regime ≠ strategy's expected regime)
        """
        credit_value = self.credit_open * 100.0  # per-contract dollar credit
        # Unrealised P&L per contract = (credit - current_mark) * 100
        unrealised_per_contract = (self.credit_open - self.current_mark) * 100.0
        loss_per_contract = -unrealised_per_contract  # positive when losing

        # 1. Hard stop
        hard_stop = credit_value * hard_stop_multiplier
        if hard_stop > 0 and loss_per_contract >= hard_stop:
            return (ExitSignal.HARD_STOP,
                    f"Loss ${loss_per_contract:.2f} ≥ {hard_stop_multiplier:.0f}× "
                    f"credit ${credit_value:.2f} (threshold=${hard_stop:.2f})")

        # 2. Profit target
        profit_thr = credit_value * profit_target_pct
        if profit_thr > 0 and unrealised_per_contract >= profit_thr:
            return (ExitSignal.PROFIT_TARGET,
                    f"Profit ${unrealised_per_contract:.2f} ≥ "
                    f"{profit_target_pct*100:.0f}% of credit ${credit_value:.2f}")

        # 3. Strike proximity
        if spot > 0 and self.short_strike > 0:
            proximity = abs(spot - self.short_strike) / self.short_strike
            if proximity <= strike_proximity_pct:
                return (ExitSignal.STRIKE_PROXIMITY,
                        f"Underlying ${spot:.2f} within {proximity*100:.2f}% of "
                        f"short strike ${self.short_strike:.0f}")

        # 4. DTE safety — last trading day before expiration after 15:30 ET.
        # In sim we don't have a real clock-time gate; use date comparison.
        next_trading_day = (self.expiration - t.date()).days
        if next_trading_day == 1 and t.hour >= 15 and t.minute >= 30:
            return (ExitSignal.DTE_SAFETY,
                    f"Expiration {self.expiration} is the next trading day; "
                    f"closing to avoid last-day gamma risk")

        # 5. Regime shift
        if current_regime is not None:
            # We need the strategy_name to look up STRATEGY_REGIME_MAP,
            # which keys on "Bull Put Spread"/"Bear Call Spread" strings.
            strategy_name = ("Bull Put Spread" if self.side == "bull_put"
                             else "Bear Call Spread")
            expected = STRATEGY_REGIME_MAP.get(strategy_name)
            if expected is not None and current_regime != expected:
                return (ExitSignal.REGIME_SHIFT,
                        f"Regime shifted to {current_regime.value} but holding "
                        f"{strategy_name} (expects {expected.value})")

        return (ExitSignal.HOLD, "")

    # ------------------------------------------------------------------
    # Close — finalize the position
    # ------------------------------------------------------------------

    def close(self, *, t: datetime,
              exit_signal: ExitSignal, reason: str) -> ClosedTrade:
        """
        Mark the position closed and return a ``ClosedTrade``. The
        runner is responsible for telling SimAccount about the cash
        movement (``apply_close``) — this method only stamps the
        position's exit fields and returns the immutable record.
        """
        self.closed = True
        self.exit_signal = exit_signal
        self.exit_reason = reason
        self.exit_t = t
        self.debit_close = self.current_mark
        # Realised P&L per contract (per share × 100 inside SimAccount):
        self.realised_pnl = round(
            (self.credit_open - self.current_mark) * self.qty * 100.0, 2,
        )
        return ClosedTrade(
            ticker=self.ticker,
            side=self.side,
            entry_t=self.entry_t,
            exit_t=t,
            expiration=self.expiration,
            short_strike=self.short_strike,
            long_strike=self.long_strike,
            spread_width=self.spread_width,
            credit_open=self.credit_open,
            debit_close=self.current_mark,
            qty=self.qty,
            realised_pnl=self.realised_pnl,
            exit_signal=exit_signal.value,
            exit_reason=reason,
            sigma_entry=self.sigma_entry,
            sigma_exit=self.sigma_current,
            days_held=self.days_held,
        )


__all__ = [
    "SimPosition",
    "ClosedTrade",
]
