"""Conformance tests for skill 44 — position-monitor contract scaling + post-fill grace.

Pins the two invariants that would have prevented the SPY 2026-07-02 incident:

  1. Hard-stop / stop-loss / profit-target thresholds MUST scale by
     ``spread.contracts_open`` — otherwise a 12-contract position trips
     hard-stop at 1/12 of the intended loss.
  2. Post-fill grace period MUST hold exit signals for the first
     ``post_fill_grace_seconds`` after ``spread.opened_at`` — otherwise
     a stale mark on the immediate post-fill cycle triggers a phantom
     hard-stop before the spread quote has settled.

Reproduces the exact numeric fingerprint of the incident where possible.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from trading_agent.position_monitor import (
    ExitSignal,
    PositionMonitor,
    PositionSnapshot,
    SpreadPosition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _leg(qty: int = 1, unrealized_pl: float = 0.0, symbol: str = "SPY260724C00762000",
         side: str = "short") -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol, qty=qty, side=side,
        avg_entry_price=0.3, current_price=0.3,
        market_value=0.0, cost_basis=0.0,
        unrealized_pl=unrealized_pl, unrealized_plpc=0.0,
        asset_class="us_option",
    )


def _spread(*, contracts: int, unrealized_pl: float,
            original_credit: float = 0.31, max_loss: float = 69.0,
            opened_at: str = "") -> SpreadPosition:
    """Mirror the SPY incident: credit=$0.31/share ($31/contract),
    max_loss=$69/contract, N contracts."""
    return SpreadPosition(
        underlying="SPY",
        strategy_name="Bear Call Spread",
        legs=[
            _leg(qty=contracts, symbol="SPY260724C00762000", side="short"),
            _leg(qty=contracts, symbol="SPY260724C00772000", side="long"),
        ],
        original_credit=original_credit,
        max_loss=max_loss,
        spread_width=1.0,
        net_unrealized_pl=unrealized_pl,
        contracts_open=contracts,
        opened_at=opened_at,
    )


def _monitor(**kwargs) -> PositionMonitor:
    """Build a PositionMonitor with grace=0 by default so tests exercise
    the exit paths directly. Override per-test where the grace path is
    the subject under test."""
    defaults = {
        "api_key": "k", "secret_key": "s",
        "hard_stop_multiplier": 3.0,
        "profit_target_pct": 0.50,
        "post_fill_grace_seconds": 0,
    }
    defaults.update(kwargs)
    return PositionMonitor(**defaults)


# ---------------------------------------------------------------------------
# §1 — Contract-count scaling
# ---------------------------------------------------------------------------

def test_hard_stop_scales_by_contract_count():
    """Skill 44 §1 — a 12-contract position must trip hard-stop at
    3× TOTAL credit ($1116), not at 3× per-contract credit ($93)."""
    monitor = _monitor()
    # Twelve contracts, $80 total unrealized loss. Per-contract that's
    # < 3× $31 = $93, so pre-fix code would trigger hard_stop.
    # Post-fix: threshold is 3× ($31 × 12) = $1116, so still HOLD.
    spread = _spread(contracts=12, unrealized_pl=-80.0)
    signal, reason = monitor._check_exit(spread, current_regimes={})
    assert signal != ExitSignal.HARD_STOP, (
        f"multi-contract hard-stop over-triggered: {signal.value} — {reason}"
    )


def test_hard_stop_still_fires_at_correct_position_threshold():
    """A 12-contract position that has genuinely lost > 3× total credit
    ($1116) must still fire hard_stop."""
    monitor = _monitor()
    spread = _spread(contracts=12, unrealized_pl=-1200.0)
    signal, reason = monitor._check_exit(spread, current_regimes={})
    assert signal == ExitSignal.HARD_STOP
    assert "12×$31.00" in reason or "12×" in reason


def test_hard_stop_pre_fix_reproduces_incident_numbers():
    """The SPY 2026-07-02 incident's numbers: loss=$162 on a 12-contract
    position where per-contract credit=$31. Under the fixed code this
    stays HOLD (threshold is $1116, not $93)."""
    monitor = _monitor()
    spread = _spread(contracts=12, unrealized_pl=-162.0)
    signal, _ = monitor._check_exit(spread, current_regimes={})
    assert signal != ExitSignal.HARD_STOP, (
        "Fixed code must NOT re-trigger the SPY 2026-07-02 phantom hard-stop."
    )


def test_stop_loss_scales_by_contract_count():
    """max_loss threshold at 50% must scale per contract too."""
    monitor = _monitor(stop_loss_pct=0.50)
    # 4 contracts × $69/contract max_loss × 0.5 threshold = $138 position-loss threshold.
    # A $100 unrealized loss is below that → HOLD.
    spread = _spread(contracts=4, unrealized_pl=-100.0)
    signal, _ = monitor._check_exit(spread, current_regimes={})
    assert signal != ExitSignal.STOP_LOSS


def test_profit_target_scales_by_contract_count():
    """profit_target at 50% must fire at 50% of TOTAL credit, not per-contract."""
    monitor = _monitor(profit_target_pct=0.50)
    # 3 contracts × $31 credit × 0.5 = $46.5 profit threshold.
    # $50 profit crosses the position threshold → PROFIT_TARGET fires.
    spread = _spread(contracts=3, unrealized_pl=+50.0)
    signal, reason = monitor._check_exit(spread, current_regimes={})
    assert signal == ExitSignal.PROFIT_TARGET


def test_single_contract_still_behaves_identically():
    """1-contract position: threshold math is unchanged from pre-fix."""
    monitor = _monitor()
    # $100 loss > 3× $31 credit = $93 threshold.
    spread = _spread(contracts=1, unrealized_pl=-100.0)
    signal, _ = monitor._check_exit(spread, current_regimes={})
    assert signal == ExitSignal.HARD_STOP


# ---------------------------------------------------------------------------
# §2 — Post-fill grace period
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seconds_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=n)).isoformat()


def test_grace_period_holds_exit_within_window():
    """Skill 44 §2 — within grace_seconds of opened_at, always HOLD
    regardless of the unrealized-loss reading."""
    monitor = _monitor(post_fill_grace_seconds=60)
    # Just-opened position with impossibly-large phantom loss (10× max).
    spread = _spread(
        contracts=1, unrealized_pl=-1000.0, opened_at=_seconds_ago(5),
    )
    signal, reason = monitor._check_exit(spread, current_regimes={})
    assert signal == ExitSignal.HOLD
    assert "grace" in reason.lower()


def test_grace_period_expires_and_exits_resume():
    """After grace_seconds, normal exit paths engage."""
    monitor = _monitor(post_fill_grace_seconds=60)
    spread = _spread(
        contracts=1, unrealized_pl=-1000.0, opened_at=_seconds_ago(120),
    )
    signal, _ = monitor._check_exit(spread, current_regimes={})
    assert signal == ExitSignal.HARD_STOP


def test_grace_period_skipped_when_opened_at_empty():
    """Inferred / legacy spreads with no opened_at must NOT be blocked
    forever — the gate is skipped when the timestamp is unknown."""
    monitor = _monitor(post_fill_grace_seconds=60)
    spread = _spread(contracts=1, unrealized_pl=-1000.0, opened_at="")
    signal, _ = monitor._check_exit(spread, current_regimes={})
    assert signal == ExitSignal.HARD_STOP


def test_grace_period_tolerates_malformed_timestamp():
    """Malformed opened_at falls through gracefully instead of crashing."""
    monitor = _monitor(post_fill_grace_seconds=60)
    spread = _spread(
        contracts=1, unrealized_pl=-1000.0, opened_at="not-a-timestamp",
    )
    # Should not raise — should evaluate normally.
    signal, _ = monitor._check_exit(spread, current_regimes={})
    assert signal == ExitSignal.HARD_STOP


def test_grace_period_zero_disables_gate():
    """post_fill_grace_seconds=0 means exit paths engage immediately —
    used by tests that want to exercise the threshold logic directly."""
    monitor = _monitor(post_fill_grace_seconds=0)
    spread = _spread(
        contracts=1, unrealized_pl=-1000.0, opened_at=_now_iso(),
    )
    signal, _ = monitor._check_exit(spread, current_regimes={})
    assert signal == ExitSignal.HARD_STOP


# ---------------------------------------------------------------------------
# Sanity: SpreadPosition new fields default safely
# ---------------------------------------------------------------------------

def test_spread_position_defaults_are_backward_compatible():
    """New fields must default to safe values so any existing caller
    that constructs SpreadPosition without them still works."""
    s = SpreadPosition(
        underlying="SPY", strategy_name="Bear Call Spread",
        legs=[], original_credit=0.31, max_loss=69.0,
        spread_width=1.0, net_unrealized_pl=0.0,
    )
    assert s.contracts_open == 1
    assert s.opened_at == ""
