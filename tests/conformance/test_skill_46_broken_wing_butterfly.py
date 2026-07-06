"""Conformance tests for skill 46 — Broken-Wing Butterfly scoring.

Pinned behaviors:

- POP formula: ``min(1.0, 4·C / (W_put + W_call))`` — skill 46 §2.1
- EV formula: equal-probability side split — skill 46 §2.2
- Direction inference: wider wing determines bullish/bearish — skill 46 §2.3
- Symmetric wings rejected (use IB instead) — skill 46 §4
- Wing ratio band [1.20, 3.00] — skill 46 §4
- CI invariant 2: scorer lives in chain_scanner.py — pinned by import path
"""

from __future__ import annotations

import pytest

from trading_agent.chain_scanner import (
    BWB_REJECT_CREDIT_GE_MIN_WING,
    BWB_REJECT_CREDIT_NON_POSITIVE_BWB,
    BWB_REJECT_DTE_NON_POSITIVE_BWB,
    BWB_REJECT_EV_NON_POSITIVE_BWB,
    BWB_REJECT_NOT_ATM_SHORTS_BWB,
    BWB_REJECT_POP_BELOW_MIN_BWB,
    BWB_REJECT_WING_RATIO_OUT_OF_BAND,
    BWB_REJECT_WING_TOO_NARROW_BWB,
    BWB_REJECT_WINGS_EQUAL,
    BWB_WING_RATIO_MAX,
    BWB_WING_RATIO_MIN,
    IB_ATM_DELTA_TOLERANCE,
    _bwb_direction,
    _ev_per_dollar_risked_bwb,
    _pop_from_bwb_structure,
    _score_broken_wing_butterfly,
    _score_broken_wing_butterfly_with_reason,
)

_ATM_CALL_DELTA = 0.50
_ATM_PUT_DELTA = -0.50


# ---------------------------------------------------------------------------
# §2.1 — POP formula
# ---------------------------------------------------------------------------

def test_pop_matches_average_wing_formula():
    """Skill 46 §2.1 — POP = min(1.0, 4C / (W_put + W_call))."""
    # C=1, put_wing=8, call_wing=4 → 4·1/12 = 0.333
    assert _pop_from_bwb_structure(1.0, 8.0, 4.0) == pytest.approx(1.0/3.0, abs=1e-4)


def test_pop_reduces_to_ib_when_wings_equal():
    """Symmetric wings → BWB POP = IB POP = 2C/W."""
    from trading_agent.chain_scanner import _pop_from_ib_structure
    bwb = _pop_from_bwb_structure(2.0, 5.0, 5.0)
    ib = _pop_from_ib_structure(2.0, 5.0)
    assert bwb == pytest.approx(ib)


def test_pop_returns_zero_on_degenerate_inputs():
    assert _pop_from_bwb_structure(0.0, 5.0, 3.0) == 0.0
    assert _pop_from_bwb_structure(1.0, 0.0, 3.0) == 0.0
    assert _pop_from_bwb_structure(1.0, 5.0, 0.0) == 0.0
    # credit ≥ min wing → 0
    assert _pop_from_bwb_structure(3.0, 5.0, 3.0) == 0.0


# ---------------------------------------------------------------------------
# §2.2 — EV formula
# ---------------------------------------------------------------------------

def test_ev_matches_equal_probability_split():
    """Skill 46 §2.2 — EV = POP·C − (1−POP)/2 · (W_p + W_c − 2C),
    normalized by max(max_loss_put, max_loss_call)."""
    # C=2, W_put=8, W_call=4
    # POP = 4·2/12 = 0.667
    # ev_per_share = 0.667·2 − (1-0.667)/2 · (12 − 4) = 1.333 − 1.333 = 0
    # Edge case: exactly zero EV → returns 0 which is NOT positive → reject via scorer
    ev = _ev_per_dollar_risked_bwb(2.0, 8.0, 4.0)
    assert ev == pytest.approx(0.0, abs=1e-3)


def test_ev_positive_for_favorable_structure():
    """A high-credit BWB has positive EV."""
    # C=2.5, W_put=6, W_call=4
    # POP = 4·2.5/10 = 1.0 (capped)
    # ev_per_share = 1.0·2.5 − 0 = 2.5
    # max_loss = max(6-2.5, 4-2.5) = 3.5
    # EV/$risked = 2.5 / 3.5 ≈ 0.714
    ev = _ev_per_dollar_risked_bwb(2.5, 6.0, 4.0)
    assert ev == pytest.approx(2.5 / 3.5, abs=1e-3)


# ---------------------------------------------------------------------------
# §2.3 — Direction inference
# ---------------------------------------------------------------------------

def test_direction_bullish_when_put_wing_wider():
    assert _bwb_direction(put_wing=8.0, call_wing=4.0) == "bullish"


def test_direction_bearish_when_call_wing_wider():
    assert _bwb_direction(put_wing=4.0, call_wing=8.0) == "bearish"


def test_direction_symmetric_when_equal():
    assert _bwb_direction(put_wing=5.0, call_wing=5.0) == "symmetric"


# ---------------------------------------------------------------------------
# §3.1 — Scorer happy path + reject taxonomy
# ---------------------------------------------------------------------------

def test_scorer_happy_path_bullish():
    """Bullish BWB (put wing wider) accepted; returns direction='bullish'."""
    out = _score_broken_wing_butterfly(
        credit=2.5, put_wing=6.0, call_wing=4.0,
        short_call_delta=_ATM_CALL_DELTA,
        short_put_delta=_ATM_PUT_DELTA,
        dte=30, min_pop=0.30,
    )
    assert out is not None
    direction, pop, cw_ratio, ev, ann = out
    assert direction == "bullish"
    assert pop == pytest.approx(1.0)   # capped
    # cw_ratio = C / avg_wing = 2.5 / 5.0 = 0.5
    assert cw_ratio == pytest.approx(0.5)


def test_scorer_happy_path_bearish():
    """Bearish BWB (call wing wider) accepted; returns direction='bearish'."""
    out = _score_broken_wing_butterfly(
        credit=2.5, put_wing=4.0, call_wing=6.0,
        short_call_delta=_ATM_CALL_DELTA,
        short_put_delta=_ATM_PUT_DELTA,
        dte=30, min_pop=0.30,
    )
    assert out is not None
    direction, *_ = out
    assert direction == "bearish"


def test_scorer_rejects_symmetric_wings():
    """§4 — symmetric wings are Iron Butterfly, not BWB."""
    verbose = _score_broken_wing_butterfly_with_reason(
        credit=2.0, put_wing=5.0, call_wing=5.0,
        short_call_delta=_ATM_CALL_DELTA,
        short_put_delta=_ATM_PUT_DELTA,
        dte=30, min_pop=0.30,
    )
    assert verbose["reason"] == BWB_REJECT_WINGS_EQUAL


def test_scorer_rejects_wing_ratio_below_min():
    """Wing ratio < 1.20 → too close to symmetric IB."""
    # put=4.5, call=4.0 → ratio 1.125, below 1.20
    verbose = _score_broken_wing_butterfly_with_reason(
        credit=2.0, put_wing=4.5, call_wing=4.0,
        short_call_delta=_ATM_CALL_DELTA,
        short_put_delta=_ATM_PUT_DELTA,
        dte=30, min_pop=0.30,
    )
    assert verbose["reason"] == BWB_REJECT_WING_RATIO_OUT_OF_BAND
    assert verbose["wing_ratio"] < BWB_WING_RATIO_MIN


def test_scorer_rejects_wing_ratio_above_max():
    """Wing ratio > 3.00 → one wing degenerate."""
    # put=15, call=4 → ratio 3.75, above 3.00
    verbose = _score_broken_wing_butterfly_with_reason(
        credit=2.0, put_wing=15.0, call_wing=4.0,
        short_call_delta=_ATM_CALL_DELTA,
        short_put_delta=_ATM_PUT_DELTA,
        dte=30, min_pop=0.30,
    )
    assert verbose["reason"] == BWB_REJECT_WING_RATIO_OUT_OF_BAND
    assert verbose["wing_ratio"] > BWB_WING_RATIO_MAX


def test_scorer_rejects_credit_ge_min_wing():
    """credit ≥ min wing → negative max_loss on the narrower side."""
    verbose = _score_broken_wing_butterfly_with_reason(
        credit=4.5, put_wing=6.0, call_wing=4.0,   # narrower wing = 4.0
        short_call_delta=_ATM_CALL_DELTA,
        short_put_delta=_ATM_PUT_DELTA,
        dte=30, min_pop=0.30,
    )
    assert verbose["reason"] == BWB_REJECT_CREDIT_GE_MIN_WING


def test_scorer_rejects_credit_non_positive():
    verbose = _score_broken_wing_butterfly_with_reason(
        credit=0.0, put_wing=6.0, call_wing=4.0,
        short_call_delta=_ATM_CALL_DELTA,
        short_put_delta=_ATM_PUT_DELTA,
        dte=30, min_pop=0.30,
    )
    assert verbose["reason"] == BWB_REJECT_CREDIT_NON_POSITIVE_BWB


def test_scorer_rejects_dte_non_positive():
    verbose = _score_broken_wing_butterfly_with_reason(
        credit=2.5, put_wing=6.0, call_wing=4.0,
        short_call_delta=_ATM_CALL_DELTA,
        short_put_delta=_ATM_PUT_DELTA,
        dte=0, min_pop=0.30,
    )
    assert verbose["reason"] == BWB_REJECT_DTE_NON_POSITIVE_BWB


def test_scorer_rejects_wing_too_narrow():
    verbose = _score_broken_wing_butterfly_with_reason(
        credit=2.5, put_wing=0.0, call_wing=4.0,
        short_call_delta=_ATM_CALL_DELTA,
        short_put_delta=_ATM_PUT_DELTA,
        dte=30, min_pop=0.30,
    )
    assert verbose["reason"] == BWB_REJECT_WING_TOO_NARROW_BWB


def test_scorer_rejects_shorts_not_atm():
    """Both shorts must be within IB_ATM_DELTA_TOLERANCE of |Δ|=0.5."""
    verbose = _score_broken_wing_butterfly_with_reason(
        credit=2.5, put_wing=6.0, call_wing=4.0,
        short_call_delta=0.30,                 # NOT ATM
        short_put_delta=_ATM_PUT_DELTA,
        dte=30, min_pop=0.30,
    )
    assert verbose["reason"] == BWB_REJECT_NOT_ATM_SHORTS_BWB


def test_scorer_rejects_pop_below_min():
    """POP < min_pop → reject."""
    # C=0.5, put=6, call=4 → POP = 4·0.5/10 = 0.2 < min_pop=0.30
    verbose = _score_broken_wing_butterfly_with_reason(
        credit=0.5, put_wing=6.0, call_wing=4.0,
        short_call_delta=_ATM_CALL_DELTA,
        short_put_delta=_ATM_PUT_DELTA,
        dte=30, min_pop=0.30,
    )
    assert verbose["reason"] == BWB_REJECT_POP_BELOW_MIN_BWB


def test_scorer_rejects_ev_non_positive():
    """Structure where math admits candidacy but EV is zero or negative."""
    # C=2, W_put=8, W_call=4 → POP=0.667, EV=0 exactly.
    verbose = _score_broken_wing_butterfly_with_reason(
        credit=2.0, put_wing=8.0, call_wing=4.0,
        short_call_delta=_ATM_CALL_DELTA,
        short_put_delta=_ATM_PUT_DELTA,
        dte=30, min_pop=0.30,
    )
    assert verbose["reason"] == BWB_REJECT_EV_NON_POSITIVE_BWB


# ---------------------------------------------------------------------------
# ATM tolerance shared with IB
# ---------------------------------------------------------------------------

def test_atm_tolerance_shared_with_ib():
    """BWB reuses IB_ATM_DELTA_TOLERANCE — same ATM detection semantics."""
    boundary_delta = 0.5 + IB_ATM_DELTA_TOLERANCE - 1e-9
    out = _score_broken_wing_butterfly(
        credit=2.5, put_wing=6.0, call_wing=4.0,
        short_call_delta=boundary_delta,
        short_put_delta=-boundary_delta,
        dte=30, min_pop=0.30,
    )
    assert out is not None


# ---------------------------------------------------------------------------
# CI invariant 2 — scorer lives in chain_scanner.py
# ---------------------------------------------------------------------------

def test_scorer_lives_in_chain_scanner():
    from trading_agent import chain_scanner
    assert _score_broken_wing_butterfly.__module__ == chain_scanner.__name__
    assert _score_broken_wing_butterfly_with_reason.__module__ == chain_scanner.__name__


# ---------------------------------------------------------------------------
# PresetConfig knobs
# ---------------------------------------------------------------------------

def test_preset_config_bwb_defaults():
    from trading_agent.strategy_presets import PresetConfig
    p = PresetConfig(
        name="test", max_delta=0.25, dte_vertical=21,
        dte_iron_condor=21, dte_mean_reversion=7,
        dte_window_days=5, width_mode="pct_of_spot",
        width_value=0.02, min_credit_ratio=0.25, max_risk_pct=0.02,
    )
    assert p.broken_wing_butterfly_enabled is False
    assert p.broken_wing_butterfly_min_pop == pytest.approx(0.40)
    # Grid entries ascending
    grid = p.broken_wing_butterfly_wing_width_pct
    assert list(grid) == sorted(grid)
