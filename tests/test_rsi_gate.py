"""
test_rsi_gate.py — pin down the RSI strategy gate's decision matrix.

The gate sits between Phase II (CLASSIFY) and Phase III (PLAN) in
``agent.py`` and refines the strategy choice based on RSI. These tests
lock in the band boundaries (inclusive lower, exclusive upper) and the
override semantics (skip / proceed / downgrade-to-vertical) so the
behaviour can't drift silently when someone tunes RSI thresholds in
the future.

The gate is a pure function — no I/O, no state — so the tests run fast
and don't need any agent / streamlit / broker stubs.
"""

from __future__ import annotations

import pytest

from trading_agent.regime import Regime
from trading_agent.rsi_gate import RsiGateDecision, evaluate_rsi_gate


# ── Mean-reversion bypass ───────────────────────────────────────────


@pytest.mark.parametrize("rsi", [10.0, 30.0, 50.0, 70.0, 90.0])
def test_mean_reversion_bypasses_gate(rsi):
    """RSI is irrelevant for mean-reversion — 3-σ touch is the signal."""
    d = evaluate_rsi_gate(Regime.MEAN_REVERSION, rsi)
    assert d.allow is True
    assert d.override_regime is None
    assert "mean-reversion" in d.reason.lower()


# ── Bullish regime ──────────────────────────────────────────────────


@pytest.mark.parametrize("rsi", [50.0, 60.0, 69.9])
def test_bullish_below_overbought_proceeds(rsi):
    """Bull Put OK while RSI hasn't reached overbought (70)."""
    d = evaluate_rsi_gate(Regime.BULLISH, rsi)
    assert d.allow is True
    assert d.override_regime is None
    assert "Bull Put" in d.reason


@pytest.mark.parametrize("rsi", [70.0, 80.0, 95.0])
def test_bullish_at_or_above_70_skips(rsi):
    """Don't sell puts after price has already pushed into overbought."""
    d = evaluate_rsi_gate(Regime.BULLISH, rsi)
    assert d.allow is False
    assert d.override_regime is None
    assert "overbought" in d.reason.lower()


def test_bullish_boundary_inclusive_at_70():
    """Boundary contract: RSI=70.0 exactly → SKIP (≥ 70, not > 70)."""
    d = evaluate_rsi_gate(Regime.BULLISH, 70.0)
    assert d.allow is False, "RSI=70 must trigger overbought skip"


def test_bullish_just_below_70_proceeds():
    """Boundary contract: RSI=69.999 → proceed."""
    d = evaluate_rsi_gate(Regime.BULLISH, 69.999)
    assert d.allow is True


# ── Bearish regime ──────────────────────────────────────────────────


@pytest.mark.parametrize("rsi", [40.0, 35.0, 30.001])
def test_bearish_above_oversold_proceeds(rsi):
    """Bear Call OK while RSI hasn't reached oversold (30)."""
    d = evaluate_rsi_gate(Regime.BEARISH, rsi)
    assert d.allow is True
    assert d.override_regime is None
    assert "Bear Call" in d.reason


@pytest.mark.parametrize("rsi", [30.0, 20.0, 5.0])
def test_bearish_at_or_below_30_skips(rsi):
    """Don't sell calls after price has already pushed into oversold."""
    d = evaluate_rsi_gate(Regime.BEARISH, rsi)
    assert d.allow is False
    assert "oversold" in d.reason.lower()


def test_bearish_boundary_inclusive_at_30():
    """Boundary contract: RSI=30.0 exactly → SKIP (≤ 30, not < 30)."""
    d = evaluate_rsi_gate(Regime.BEARISH, 30.0)
    assert d.allow is False


# ── Sideways regime — the meat of the gate ───────────────────────────


@pytest.mark.parametrize("rsi", [45.0, 50.0, 54.999])
def test_sideways_neutral_band_keeps_iron_condor(rsi):
    """RSI ∈ [45, 55) → true neutral, full Iron Condor."""
    d = evaluate_rsi_gate(Regime.SIDEWAYS, rsi)
    assert d.allow is True
    assert d.override_regime is None  # no override → planner picks IC for SIDEWAYS
    assert "neutral" in d.reason.lower() or "iron condor" in d.reason.lower()


@pytest.mark.parametrize("rsi", [55.0, 60.0, 64.999])
def test_sideways_lean_bullish_downgrades_to_bull_put(rsi):
    """RSI ∈ [55, 65) → downgrade to Bull Put only (skip the call wing)."""
    d = evaluate_rsi_gate(Regime.SIDEWAYS, rsi)
    assert d.allow is True
    assert d.override_regime == Regime.BULLISH
    assert "lean-bullish" in d.reason.lower()
    assert "bull put" in d.reason.lower()


@pytest.mark.parametrize("rsi", [35.0, 40.0, 44.999])
def test_sideways_lean_bearish_downgrades_to_bear_call(rsi):
    """RSI ∈ [35, 45) → downgrade to Bear Call only (skip the put wing)."""
    d = evaluate_rsi_gate(Regime.SIDEWAYS, rsi)
    assert d.allow is True
    assert d.override_regime == Regime.BEARISH
    assert "lean-bearish" in d.reason.lower()
    assert "bear call" in d.reason.lower()


@pytest.mark.parametrize("rsi", [65.0, 70.0, 80.0, 99.0])
def test_sideways_strong_up_momentum_skips(rsi):
    """RSI ≥ 65 in a sideways regime → momentum too active, skip."""
    d = evaluate_rsi_gate(Regime.SIDEWAYS, rsi)
    assert d.allow is False
    assert "momentum too active" in d.reason.lower()


@pytest.mark.parametrize("rsi", [34.999, 30.0, 20.0, 0.0])
def test_sideways_strong_down_momentum_skips(rsi):
    """RSI < 35 in a sideways regime → momentum too active, skip."""
    d = evaluate_rsi_gate(Regime.SIDEWAYS, rsi)
    assert d.allow is False


def test_sideways_band_boundaries_no_overlap():
    """Every real RSI ∈ [0, 100] must produce exactly one verdict.

    This guards against a regression where a future tweak to the
    boundary constants accidentally creates a gap or overlap.
    """
    for rsi_int in range(0, 101):
        d = evaluate_rsi_gate(Regime.SIDEWAYS, float(rsi_int))
        # Exactly one of these must be true:
        is_neutral_ic    = d.allow and d.override_regime is None
        is_lean_bullish  = d.allow and d.override_regime == Regime.BULLISH
        is_lean_bearish  = d.allow and d.override_regime == Regime.BEARISH
        is_skip          = not d.allow
        truthy_count = sum(
            1 for x in (is_neutral_ic, is_lean_bullish, is_lean_bearish, is_skip) if x
        )
        assert truthy_count == 1, (
            f"RSI={rsi_int} produced ambiguous decision: "
            f"neutral_ic={is_neutral_ic} lean_bull={is_lean_bullish} "
            f"lean_bear={is_lean_bearish} skip={is_skip}"
        )


# ── Specific real-world cases (regression tests) ────────────────────


def test_dia_today_would_have_downgraded_to_bull_put():
    """The DIA Iron Condor entered today had RSI=62.77, regime=sideways.

    This is the user-cited motivation for the gate. With the gate
    enabled, the trade should have been downgraded to a Bull Put
    (skip the call wing — momentum still active to the upside).
    """
    d = evaluate_rsi_gate(Regime.SIDEWAYS, 62.77)
    assert d.allow is True
    assert d.override_regime == Regime.BULLISH
    assert "lean-bullish" in d.reason.lower()


def test_truly_neutral_market_keeps_iron_condor():
    """RSI=50, sideways → classic Iron Condor setup, gate does nothing."""
    d = evaluate_rsi_gate(Regime.SIDEWAYS, 50.0)
    assert d.allow is True
    assert d.override_regime is None


# ── Decision shape ──────────────────────────────────────────────────


def test_decision_is_frozen_dataclass():
    """RsiGateDecision is immutable — callers can't accidentally mutate it."""
    d = evaluate_rsi_gate(Regime.SIDEWAYS, 50.0)
    assert isinstance(d, RsiGateDecision)
    with pytest.raises(Exception):
        d.allow = False


def test_decision_always_carries_reason():
    """Every branch must populate ``reason`` for journal audit."""
    cases = [
        (Regime.MEAN_REVERSION, 50.0),
        (Regime.BULLISH, 50.0),
        (Regime.BULLISH, 75.0),
        (Regime.BEARISH, 50.0),
        (Regime.BEARISH, 25.0),
        (Regime.SIDEWAYS, 50.0),
        (Regime.SIDEWAYS, 60.0),
        (Regime.SIDEWAYS, 40.0),
        (Regime.SIDEWAYS, 80.0),
        (Regime.SIDEWAYS, 20.0),
    ]
    for regime, rsi in cases:
        d = evaluate_rsi_gate(regime, rsi)
        assert d.reason and len(d.reason) > 10, (
            f"({regime}, {rsi}) returned empty/short reason: {d.reason!r}"
        )
