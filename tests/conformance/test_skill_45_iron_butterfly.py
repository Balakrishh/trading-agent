"""Conformance tests for skill 45 — Iron Butterfly scoring.

Pinned behaviors:

- POP formula: ``min(1.0, 2·C / W)`` — skill 45 §2.1
- EV formula: ``POP·C − (1 − POP)·(W − C)`` per share — skill 45 §2.2
- ATM-shorts gate: both shorts within IB_ATM_DELTA_TOLERANCE of |Δ|=0.5 — skill 45 §4
- Asymmetric-wing gate — skill 45 §4
- CI invariant 2: scoring lives in chain_scanner.py — pinned by import path
"""

from __future__ import annotations

import math

import pytest

from trading_agent.chain_scanner import (
    IB_ATM_DELTA_TOLERANCE,
    IB_REJECT_CREDIT_GE_WING,
    IB_REJECT_CREDIT_NON_POSITIVE_IB,
    IB_REJECT_DTE_NON_POSITIVE_IB,
    IB_REJECT_EV_NON_POSITIVE_IB,
    IB_REJECT_NOT_ATM_SHORTS,
    IB_REJECT_POP_BELOW_MIN_IB,
    IB_REJECT_WING_TOO_NARROW,
    IB_REJECT_WINGS_ASYMMETRIC,
    _ev_per_dollar_risked_ib,
    _pop_from_ib_structure,
    _score_iron_butterfly,
    _score_iron_butterfly_with_reason,
)


# ---------------------------------------------------------------------------
# §2.1 — POP formula
# ---------------------------------------------------------------------------

def test_pop_from_ib_structure_matches_formula():
    """Skill 45 §2.1 — POP_IB = min(1.0, 2C/W)."""
    # 2C/W = 2·1/5 = 0.4
    assert _pop_from_ib_structure(credit=1.0, wing_width=5.0) == pytest.approx(0.4)
    # 2C/W = 2·2/5 = 0.8
    assert _pop_from_ib_structure(credit=2.0, wing_width=5.0) == pytest.approx(0.8)
    # 2C/W ≥ 1.0 caps at 1.0 (degenerate but well-defined)
    assert _pop_from_ib_structure(credit=0.6, wing_width=1.0) == pytest.approx(1.0)


def test_pop_from_ib_structure_returns_zero_on_degenerate_inputs():
    """Non-positive credit, non-positive wing, or credit≥wing → POP=0."""
    assert _pop_from_ib_structure(credit=0.0, wing_width=5.0) == 0.0
    assert _pop_from_ib_structure(credit=1.0, wing_width=0.0) == 0.0
    assert _pop_from_ib_structure(credit=-1.0, wing_width=5.0) == 0.0
    assert _pop_from_ib_structure(credit=5.0, wing_width=5.0) == 0.0
    assert _pop_from_ib_structure(credit=6.0, wing_width=5.0) == 0.0


# ---------------------------------------------------------------------------
# §2.2 — EV formula
# ---------------------------------------------------------------------------

def test_ev_per_dollar_risked_ib_matches_formula():
    """Skill 45 §2.2 — EV = POP·C − (1−POP)·(W−C), normalized by max_loss."""
    # C=2, W=5 → POP=0.8, max_loss=3
    # EV per share = 0.8·2 − 0.2·3 = 1.6 − 0.6 = 1.0
    # EV/$risked = 1.0 / 3.0 = 0.333
    ev = _ev_per_dollar_risked_ib(credit=2.0, wing_width=5.0)
    assert ev == pytest.approx(1.0 / 3.0, abs=1e-4)


def test_ev_per_dollar_risked_ib_returns_none_on_degenerate():
    assert _ev_per_dollar_risked_ib(credit=0.0, wing_width=5.0) is None
    assert _ev_per_dollar_risked_ib(credit=5.0, wing_width=5.0) is None
    assert _ev_per_dollar_risked_ib(credit=6.0, wing_width=5.0) is None
    assert _ev_per_dollar_risked_ib(credit=1.0, wing_width=0.0) is None


# ---------------------------------------------------------------------------
# §3.1 — Scorer happy path + reject taxonomy
# ---------------------------------------------------------------------------

_ATM_CALL_DELTA = 0.50
_ATM_PUT_DELTA = -0.50


def test_scorer_happy_path_returns_five_field_tuple():
    """Skill 45 §3.1 — accepted candidate returns (pop, cw, ev, annualized)."""
    out = _score_iron_butterfly(
        credit=2.0, wing_width=5.0,
        short_call_delta=_ATM_CALL_DELTA,
        short_put_delta=_ATM_PUT_DELTA,
        dte=30, min_pop=0.40,
    )
    assert out is not None
    pop, cw, ev, ann = out
    assert pop == pytest.approx(0.8)
    assert cw == pytest.approx(0.4)
    assert ev == pytest.approx(1.0 / 3.0, abs=1e-4)
    # annualized = EV/$risked × 365/DTE
    assert ann == pytest.approx((1.0 / 3.0) * (365.0 / 30.0), abs=1e-4)


def test_scorer_rejects_when_shorts_not_atm():
    """§4 — |Δ_short| must be within IB_ATM_DELTA_TOLERANCE of 0.5."""
    out = _score_iron_butterfly(
        credit=2.0, wing_width=5.0,
        short_call_delta=0.30,       # NOT ATM
        short_put_delta=_ATM_PUT_DELTA,
        dte=30, min_pop=0.40,
    )
    assert out is None
    verbose = _score_iron_butterfly_with_reason(
        credit=2.0, wing_width=5.0,
        short_call_delta=0.30,
        short_put_delta=_ATM_PUT_DELTA,
        dte=30, min_pop=0.40,
    )
    assert verbose["reason"] == IB_REJECT_NOT_ATM_SHORTS
    assert verbose["short_call_delta"] == pytest.approx(0.30)


def test_scorer_rejects_asymmetric_wings():
    """§4 — call-side and put-side wing widths must be equal (broken-wing
    variants are skill 46, not IB)."""
    verbose = _score_iron_butterfly_with_reason(
        credit=2.0, wing_width=5.0,
        short_call_delta=_ATM_CALL_DELTA,
        short_put_delta=_ATM_PUT_DELTA,
        dte=30, min_pop=0.40,
        call_wing_width=5.0, put_wing_width=6.0,
    )
    assert verbose["reason"] == IB_REJECT_WINGS_ASYMMETRIC


def test_scorer_rejects_when_pop_below_min():
    """§4 — POP < min_pop → reject with pop_below_min."""
    # 2C/W = 0.2 (below 0.40 min_pop)
    verbose = _score_iron_butterfly_with_reason(
        credit=0.5, wing_width=5.0,
        short_call_delta=_ATM_CALL_DELTA,
        short_put_delta=_ATM_PUT_DELTA,
        dte=30, min_pop=0.40,
    )
    assert verbose["reason"] == IB_REJECT_POP_BELOW_MIN_IB
    assert verbose["pop"] == pytest.approx(0.2)


def test_scorer_rejects_dte_non_positive():
    verbose = _score_iron_butterfly_with_reason(
        credit=2.0, wing_width=5.0,
        short_call_delta=_ATM_CALL_DELTA,
        short_put_delta=_ATM_PUT_DELTA,
        dte=0, min_pop=0.40,
    )
    assert verbose["reason"] == IB_REJECT_DTE_NON_POSITIVE_IB


def test_scorer_rejects_wing_too_narrow():
    verbose = _score_iron_butterfly_with_reason(
        credit=2.0, wing_width=0.0,
        short_call_delta=_ATM_CALL_DELTA,
        short_put_delta=_ATM_PUT_DELTA,
        dte=30, min_pop=0.40,
    )
    assert verbose["reason"] == IB_REJECT_WING_TOO_NARROW


def test_scorer_rejects_credit_non_positive():
    verbose = _score_iron_butterfly_with_reason(
        credit=0.0, wing_width=5.0,
        short_call_delta=_ATM_CALL_DELTA,
        short_put_delta=_ATM_PUT_DELTA,
        dte=30, min_pop=0.40,
    )
    assert verbose["reason"] == IB_REJECT_CREDIT_NON_POSITIVE_IB


def test_scorer_rejects_credit_ge_wing_width():
    """credit ≥ wing_width means the trade is a net debit or degenerate."""
    verbose = _score_iron_butterfly_with_reason(
        credit=5.0, wing_width=5.0,
        short_call_delta=_ATM_CALL_DELTA,
        short_put_delta=_ATM_PUT_DELTA,
        dte=30, min_pop=0.40,
    )
    assert verbose["reason"] == IB_REJECT_CREDIT_GE_WING


# ---------------------------------------------------------------------------
# ATM tolerance edge cases
# ---------------------------------------------------------------------------

def test_atm_tolerance_accepts_at_boundary():
    """|Δ| = 0.5 + IB_ATM_DELTA_TOLERANCE is exactly at boundary → accept."""
    boundary_delta = 0.5 + IB_ATM_DELTA_TOLERANCE - 1e-9   # just inside
    out = _score_iron_butterfly(
        credit=2.0, wing_width=5.0,
        short_call_delta=boundary_delta,
        short_put_delta=-boundary_delta,
        dte=30, min_pop=0.40,
    )
    assert out is not None


def test_atm_tolerance_rejects_just_beyond():
    beyond = 0.5 + IB_ATM_DELTA_TOLERANCE + 0.01
    verbose = _score_iron_butterfly_with_reason(
        credit=2.0, wing_width=5.0,
        short_call_delta=beyond,
        short_put_delta=-beyond,
        dte=30, min_pop=0.40,
    )
    assert verbose["reason"] == IB_REJECT_NOT_ATM_SHORTS


def test_signed_delta_handled_correctly_for_puts():
    """Puts have Δ ∈ [-1, 0]. The ATM check must use |Δ| not raw Δ."""
    out = _score_iron_butterfly(
        credit=2.0, wing_width=5.0,
        short_call_delta=+0.50,
        short_put_delta=-0.50,       # signed negative for put
        dte=30, min_pop=0.40,
    )
    assert out is not None


# ---------------------------------------------------------------------------
# CI invariant 2 — scorer lives in chain_scanner.py
# ---------------------------------------------------------------------------

def test_scorer_lives_in_chain_scanner():
    """CI invariant 2 — _score_* helpers may only be defined in
    chain_scanner.py OR decision_engine.py. Iron Butterfly is a pure
    scoring primitive so it lives with the vertical scorer's siblings."""
    from trading_agent import chain_scanner
    assert _score_iron_butterfly.__module__ == chain_scanner.__name__
    assert _score_iron_butterfly_with_reason.__module__ == chain_scanner.__name__


# ---------------------------------------------------------------------------
# PresetConfig knobs are wired
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# §3 — decide_iron_butterfly orchestrator (Phase 1.5)
# ---------------------------------------------------------------------------

def _make_ib_chain(spot: float = 100.0):
    """Synthetic chain with an ATM strike + one wing on each side.

    Wing prices are cheap enough that net credit > W/3 (the algebraic
    threshold for positive EV under POP ≈ 2C/W). See skill 45 §2.2 for
    the derivation.
    """
    # 100 strike (ATM), 95 put wing, 105 call wing.
    return [
        {"symbol": "TEST100C", "strike": 100.0, "type": "call",
         "delta": 0.50, "bid": 1.20, "ask": 1.30, "dte": 30},
        {"symbol": "TEST100P", "strike": 100.0, "type": "put",
         "delta": -0.50, "bid": 1.20, "ask": 1.30, "dte": 30},
        {"symbol": "TEST105C", "strike": 105.0, "type": "call",
         "delta": 0.30, "bid": 0.15, "ask": 0.25, "dte": 30},
        {"symbol": "TEST95P", "strike": 95.0, "type": "put",
         "delta": -0.30, "bid": 0.15, "ask": 0.25, "dte": 30},
    ]


class _StubPreset:
    iron_butterfly_dte_grid = (30,)
    iron_butterfly_wing_width_pct = (0.05,)   # 5% → wing = $5 on $100 spot
    iron_butterfly_min_pop = 0.20             # loose so synthetic passes


def test_decide_iron_butterfly_produces_candidate():
    """Skill 45 §3 — a valid ATM structure produces an accepted candidate."""
    from trading_agent.chain_scanner import IronButterflyCandidate
    from trading_agent.decision_engine import (
        ChainSlice,
        DecisionInput,
        decide_iron_butterfly,
    )

    slc = ChainSlice(expiration="2026-08-01", dte=30,
                     contracts=_make_ib_chain())
    inp = DecisionInput(side="iron_butterfly", chain_slices=[slc],
                        preset=_StubPreset())
    out = decide_iron_butterfly(inp)
    assert len(out.candidates) == 1
    c = out.candidates[0]
    assert isinstance(c, IronButterflyCandidate)
    assert c.strategy == "iron_butterfly"
    assert c.center_strike == 100.0
    assert c.wing_width == 5.0
    # Credit: mid(100C) + mid(100P) - mid(105C) - mid(95P)
    #       = 1.25 + 1.25 - 0.20 - 0.20 = 2.10
    assert c.credit == pytest.approx(2.10, abs=0.02)
    assert c.max_profit == pytest.approx(c.credit)
    assert c.max_loss == pytest.approx(5.0 - c.credit, abs=0.02)


def test_decide_iron_butterfly_empty_chain_produces_no_candidate():
    from trading_agent.decision_engine import (
        ChainSlice,
        DecisionInput,
        decide_iron_butterfly,
    )
    inp = DecisionInput(
        side="iron_butterfly",
        chain_slices=[ChainSlice(expiration="2026-08-01", dte=30, contracts=[])],
        preset=_StubPreset(),
    )
    out = decide_iron_butterfly(inp)
    assert out.candidates == []


def test_decide_iron_butterfly_sorts_by_annualized_score():
    """Multiple accepted candidates come back highest-annualized first."""
    from trading_agent.decision_engine import (
        ChainSlice,
        DecisionInput,
        decide_iron_butterfly,
    )
    # Two DTE slices: 30d and 60d. The shorter DTE has higher annualized score
    # for the same EV/$risked ratio, so 30d should win.
    slc_30 = ChainSlice(expiration="2026-08-01", dte=30,
                        contracts=_make_ib_chain())
    slc_60 = ChainSlice(expiration="2026-09-01", dte=60,
                        contracts=_make_ib_chain())
    inp = DecisionInput(side="iron_butterfly",
                        chain_slices=[slc_60, slc_30],
                        preset=_StubPreset())
    out = decide_iron_butterfly(inp)
    assert len(out.candidates) >= 2
    assert out.candidates[0].dte == 30    # higher annualized wins


# ---------------------------------------------------------------------------
# §5 — strategy.py dispatch wiring
# ---------------------------------------------------------------------------

def test_strategy_planner_has_plan_iron_butterfly_method():
    """Skill 45 §5 — StrategyPlanner class must expose
    _plan_iron_butterfly for the sideways-regime dispatch path."""
    from trading_agent.strategy import StrategyPlanner
    assert hasattr(StrategyPlanner, "_plan_iron_butterfly")


def test_strategy_dispatch_reads_iron_butterfly_enabled_flag():
    """The sideways-regime branch of Strategy.plan_trade must gate
    the IB attempt on preset.iron_butterfly_enabled. Grep the source
    to confirm the flag consult exists near the IC branch."""
    from trading_agent import strategy
    src = open(strategy.__file__).read()
    assert "iron_butterfly_enabled" in src
    # And the fallback to IC must still be present so a rejected IB
    # doesn't leave the sideways regime unhandled.
    assert "_plan_iron_condor" in src


def test_preset_config_iron_butterfly_defaults():
    """New knobs default safely — enabled=False, sane grids."""
    from trading_agent.strategy_presets import PresetConfig
    # Pick any concrete preset shape (fields required by non-default):
    p = PresetConfig(
        name="test", max_delta=0.25, dte_vertical=21,
        dte_iron_condor=21, dte_mean_reversion=7,
        dte_window_days=5, width_mode="pct_of_spot",
        width_value=0.02, min_credit_ratio=0.25, max_risk_pct=0.02,
    )
    assert p.iron_butterfly_enabled is False
    assert p.iron_butterfly_min_pop == pytest.approx(0.40)
    assert 21 in p.iron_butterfly_dte_grid
    assert all(0 < w < 0.10 for w in p.iron_butterfly_wing_width_pct)
