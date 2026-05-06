"""
Tests for ``trading_agent.backtest.sim_position.SimPosition``.

Pin the VIX-proxy IV scaling, the re-mark P&L identity, and the
exit-rule parity with ``PositionMonitor._check_exit``.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from trading_agent.backtest.black_scholes import bs_price
from trading_agent.backtest.sim_position import SimPosition
from trading_agent.position_monitor import ExitSignal
from trading_agent.regime import Regime


def _make_position(*, sigma_entry=0.20, vix_entry=18.0, qty=5,
                   side="bull_put", spot_at_entry=500.0):
    """Helper — open a synthetic SPY 490/485 put spread at t0."""
    entry_t = datetime(2026, 5, 4, 9, 30)
    expiration = date(2026, 5, 18)
    # Recover credit from BS at the synthesised σ_entry so re-mark
    # at t=0 returns the same number.
    t_years = (expiration - entry_t.date()).days / 365.0
    short = bs_price(spot_at_entry, 490, t_years, sigma_entry, option_type="put")
    long_ = bs_price(spot_at_entry, 485, t_years, sigma_entry, option_type="put")
    credit = max(0.05, short - long_)
    return SimPosition(
        ticker="SPY", side=side,
        short_strike=490.0, long_strike=485.0, spread_width=5.0,
        expiration=expiration, qty=qty, credit_open=credit,
        sigma_entry=sigma_entry, vix_entry=vix_entry,
        entry_t=entry_t,
        current_mark=credit, current_t=entry_t,
        sigma_current=sigma_entry, vix_current=vix_entry,
    )


# --------------------------------------------------------------------------
# remark — IV scaling
# --------------------------------------------------------------------------

class TestRemark:
    def test_no_vix_change_keeps_sigma_constant(self):
        pos = _make_position(sigma_entry=0.20, vix_entry=18.0)
        pos.remark(t=datetime(2026, 5, 6, 16, 0), spot=500.0, vix_t=18.0)
        assert pos.sigma_current == pytest.approx(0.20, abs=1e-3)

    def test_vix_doubles_sigma_doubles(self):
        pos = _make_position(sigma_entry=0.20, vix_entry=18.0)
        pos.remark(t=datetime(2026, 5, 6, 16, 0), spot=500.0, vix_t=36.0)
        # σ_t = 0.20 × (36/18) = 0.40
        assert pos.sigma_current == pytest.approx(0.40, abs=1e-3)

    def test_missing_vix_falls_back_to_sigma_entry(self):
        pos = _make_position(sigma_entry=0.20, vix_entry=18.0)
        pos.remark(t=datetime(2026, 5, 6, 16, 0), spot=500.0, vix_t=None)
        assert pos.sigma_current == pytest.approx(0.20)

    def test_zero_vix_entry_disables_scaling(self):
        pos = _make_position(sigma_entry=0.20, vix_entry=0.0)
        pos.remark(t=datetime(2026, 5, 6, 16, 0), spot=500.0, vix_t=30.0)
        assert pos.sigma_current == pytest.approx(0.20)

    def test_remark_collapses_to_intrinsic_at_expiration(self):
        pos = _make_position()
        # Spot 480 — short 490 put is ITM by $10, long 485 ITM by $5.
        # Spread debit-to-close at expiry = 10 - 5 = 5 = max loss.
        pos.remark(t=datetime(2026, 5, 18, 16, 0), spot=480.0, vix_t=18.0)
        assert pos.current_mark == pytest.approx(5.0, abs=0.01)

    def test_remark_skips_when_spot_non_positive(self):
        pos = _make_position()
        before_mark = pos.current_mark
        pos.remark(t=datetime(2026, 5, 6, 16, 0), spot=0.0, vix_t=18.0)
        # Mark unchanged
        assert pos.current_mark == before_mark


# --------------------------------------------------------------------------
# evaluate_exit — parity with PositionMonitor._check_exit
# --------------------------------------------------------------------------

class TestEvaluateExit:
    def test_profit_target_fires_at_50_percent(self):
        pos = _make_position()
        # Push mark down to 50% of credit
        pos.current_mark = pos.credit_open * 0.50
        sig, _ = pos.evaluate_exit(
            t=datetime(2026, 5, 5, 16, 0), spot=500.0, current_regime=None,
        )
        assert sig == ExitSignal.PROFIT_TARGET

    def test_hard_stop_fires_at_3x_credit_loss(self):
        pos = _make_position()
        # Loss per contract = (mark − credit) × 100. To hit hard_stop:
        # loss ≥ credit × 100 × 3 → mark ≥ credit + 3 × credit = 4 × credit.
        pos.current_mark = pos.credit_open * 4.0
        sig, _ = pos.evaluate_exit(
            t=datetime(2026, 5, 5, 16, 0), spot=500.0, current_regime=None,
        )
        assert sig == ExitSignal.HARD_STOP

    def test_strike_proximity_fires_when_near_short_strike(self):
        pos = _make_position()
        # Spot within 1% of short strike 490 → fires
        sig, _ = pos.evaluate_exit(
            t=datetime(2026, 5, 5, 16, 0), spot=490.5, current_regime=None,
        )
        assert sig == ExitSignal.STRIKE_PROXIMITY

    def test_regime_shift_fires_for_bull_put_in_bearish(self):
        pos = _make_position()  # default side = bull_put
        sig, _ = pos.evaluate_exit(
            t=datetime(2026, 5, 5, 16, 0), spot=500.0,
            current_regime=Regime.BEARISH,
        )
        assert sig == ExitSignal.REGIME_SHIFT

    def test_hold_when_nothing_fires(self):
        pos = _make_position()
        sig, _ = pos.evaluate_exit(
            t=datetime(2026, 5, 5, 16, 0), spot=500.0,
            current_regime=Regime.BULLISH,
        )
        assert sig == ExitSignal.HOLD


# --------------------------------------------------------------------------
# close — produces a ClosedTrade with correct realised P&L
# --------------------------------------------------------------------------

def test_close_returns_closedtrade_with_realised_pnl():
    pos = _make_position()
    # Take profit half the credit
    pos.current_mark = pos.credit_open * 0.50
    pos.current_t = datetime(2026, 5, 5, 16, 0)
    closed = pos.close(t=datetime(2026, 5, 5, 16, 0),
                       exit_signal=ExitSignal.PROFIT_TARGET,
                       reason="Took profit at 50%")
    assert closed.exit_signal == "profit_target"
    assert closed.qty == 5
    # close() rounds realised_pnl to 2 decimals (cents); compare with abs=$0.01.
    assert closed.realised_pnl == pytest.approx(
        (pos.credit_open - pos.current_mark) * 5 * 100.0, abs=0.01,
    )
    assert closed.sigma_entry == pytest.approx(pos.sigma_entry)
    assert pos.closed is True
