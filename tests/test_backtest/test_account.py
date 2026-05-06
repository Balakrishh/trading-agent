"""
Tests for ``trading_agent.backtest.account.SimAccount``.

These pin the ledger semantics: open credit lands in cash but is
balanced by an equal-and-opposite open_market_value mark, so equity is
flat at the instant of fill. Closing for less than the credit yields a
positive realised P&L that flows into ``cash`` and ``realised_pnl``.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from trading_agent.backtest.account import COMMISSION_PER_LEG, SimAccount


@pytest.fixture
def account():
    return SimAccount.fresh(starting_equity=100_000.0)


# --------------------------------------------------------------------------
# Initial state
# --------------------------------------------------------------------------

def test_fresh_account_state(account):
    assert account.cash == 100_000.0
    assert account.equity == 100_000.0
    assert account.open_market_value == 0.0
    assert account.realised_pnl == 0.0
    assert account.open_spread_count == 0
    assert account.equity_curve == []


# --------------------------------------------------------------------------
# apply_open
# --------------------------------------------------------------------------

class TestApplyOpen:
    def test_credit_lands_in_cash_minus_commission(self, account):
        # 5 contracts at $0.40 credit = $200 gross, minus $0.65 × 2 × 5 = $6.50
        account.apply_open(credit_per_share=0.40, qty=5, spread_width=5.0)
        expected_cash = 100_000.0 + 200.0 - (COMMISSION_PER_LEG * 2 * 5)
        assert account.cash == pytest.approx(expected_cash)

    def test_open_market_value_is_negative_credit(self, account):
        account.apply_open(credit_per_share=0.40, qty=5, spread_width=5.0)
        # We owe back the credit until we close: omv = -200
        assert account.open_market_value == pytest.approx(-200.0)

    def test_equity_drops_only_by_commission_at_open(self, account):
        # Equity = cash + omv = (100000 + 200 - 6.50) + (-200) = 99993.50
        account.apply_open(credit_per_share=0.40, qty=5, spread_width=5.0)
        assert account.equity == pytest.approx(100_000.0 - (COMMISSION_PER_LEG * 2 * 5))

    def test_open_spread_count_increments(self, account):
        account.apply_open(credit_per_share=0.40, qty=5, spread_width=5.0)
        account.apply_open(credit_per_share=0.30, qty=3, spread_width=5.0)
        assert account.open_spread_count == 2

    def test_zero_qty_is_noop(self, account):
        account.apply_open(credit_per_share=0.40, qty=0, spread_width=5.0)
        assert account.cash == 100_000.0
        assert account.open_market_value == 0.0
        assert account.open_spread_count == 0


# --------------------------------------------------------------------------
# apply_close
# --------------------------------------------------------------------------

class TestApplyClose:
    def test_winning_close_yields_positive_realised_pnl(self, account):
        # Open at 0.40 credit, close at 0.10 debit → win 0.30 × 100 × 5 = $150
        # minus 2 × open commission + 2 × close commission = $13.
        account.apply_open(credit_per_share=0.40, qty=5, spread_width=5.0)
        realised = account.apply_close(credit_per_share=0.40, qty=5,
                                       closing_debit_per_share=0.10)
        # 5 contracts × $0.30 × 100 = $150 gross, minus $6.50 close commission
        expected = 150.0 - (COMMISSION_PER_LEG * 2 * 5)
        assert realised == pytest.approx(expected)

    def test_losing_close_yields_negative_realised_pnl(self, account):
        # Open at 0.40, close at 1.00 → lose 0.60 × 100 × 5 = -$300
        account.apply_open(credit_per_share=0.40, qty=5, spread_width=5.0)
        realised = account.apply_close(credit_per_share=0.40, qty=5,
                                       closing_debit_per_share=1.00)
        expected = -300.0 - (COMMISSION_PER_LEG * 2 * 5)
        assert realised == pytest.approx(expected)

    def test_close_decrements_open_spread_count(self, account):
        account.apply_open(credit_per_share=0.40, qty=5, spread_width=5.0)
        assert account.open_spread_count == 1
        account.apply_close(credit_per_share=0.40, qty=5, closing_debit_per_share=0.10)
        assert account.open_spread_count == 0

    def test_close_retires_open_market_value_slice(self, account):
        # After open, omv = -200 (one position).  After close it should be
        # back to 0 because we've squared the position.
        account.apply_open(credit_per_share=0.40, qty=5, spread_width=5.0)
        account.apply_close(credit_per_share=0.40, qty=5,
                            closing_debit_per_share=0.10)
        assert account.open_market_value == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Equity curve
# --------------------------------------------------------------------------

def test_snapshot_records_equity_point(account):
    pt = account.snapshot(datetime(2026, 5, 3, 16, 0))
    assert len(account.equity_curve) == 1
    assert pt.equity == pytest.approx(100_000.0)
    assert pt.cash == pytest.approx(100_000.0)
    assert pt.open_market_value == pytest.approx(0.0)


def test_snapshot_after_open_records_drawdown(account):
    account.apply_open(credit_per_share=0.40, qty=5, spread_width=5.0)
    pt = account.snapshot(datetime(2026, 5, 3, 16, 0))
    # Equity at instant-of-open ≈ starting minus commission
    assert pt.equity == pytest.approx(100_000.0 - COMMISSION_PER_LEG * 2 * 5)


def test_apply_mark_replaces_open_market_value(account):
    account.apply_open(credit_per_share=0.40, qty=5, spread_width=5.0)
    # Imagine the spread mark dropped 50% mid-life: total omv = -100
    account.apply_mark(total_open_market_value=-100.0)
    assert account.open_market_value == pytest.approx(-100.0)
    # Equity = cash (193.50) + (-100) = 99893.50 ... unrealised gain shows
    # up vs the at-open equity of 99993.50.
    assert account.equity == pytest.approx(
        100_000.0 - COMMISSION_PER_LEG * 2 * 5 + 100.0
    )
