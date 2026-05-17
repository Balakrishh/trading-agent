"""Conformance test: skill 01 — POP from short delta.

Skill 01 §2 documents the formula::

    POP ≈ 1 − |Δshort|

This test asserts the live ``_pop_from_delta`` implementation matches
that identity exactly, plus the §4 edge case that POP is clamped at
zero (a delta magnitude > 1 is structurally impossible but the
implementation defends against it anyway).

Failure modes caught:
- Someone replaces |Δ| with raw Δ (sign-bearing) — breaks call-side POP
- Someone removes the max(0.0, ...) clamp — POP could go negative on
  bad chain data
- Someone "corrects" the formula to use Black-Scholes N(d2) — POP is
  intentionally the linear approximation in this codebase
"""

from __future__ import annotations

import pytest

from trading_agent.chain_scanner import _pop_from_delta


@pytest.mark.parametrize("delta,expected_pop", [
    # Skill 01 §2 worked example: Δ-0.30 → POP 0.70
    (-0.30, 0.70),
    (+0.30, 0.70),
    # Centered at ATM: |Δ| = 0.50 → 50/50 odds
    (-0.50, 0.50),
    (+0.50, 0.50),
    # Far OTM: tiny delta, near-certain win
    (-0.05, 0.95),
    # Deep ITM: low POP (sanity)
    (-0.85, 0.15),
])
def test_skill_01_pop_matches_one_minus_abs_delta(
    delta: float, expected_pop: float,
) -> None:
    actual = _pop_from_delta(delta)
    assert actual == pytest.approx(expected_pop), (
        f"Skill 01 §2: POP({delta}) should be 1−|Δ| = {expected_pop}, "
        f"got {actual}. Check chain_scanner._pop_from_delta."
    )


def test_skill_01_pop_is_sign_agnostic() -> None:
    """The |·| in the formula means put and call shorts of equal
    magnitude must give identical POP."""
    for mag in (0.10, 0.20, 0.35, 0.45):
        assert _pop_from_delta(-mag) == _pop_from_delta(+mag)


def test_skill_01_pop_clamped_at_zero() -> None:
    """Skill 01 §4: a delta-magnitude > 1 is structurally impossible
    on a real options chain (it'd mean cumulative probability > 100%),
    but the implementation defends against bad upstream data by
    clamping the POP floor at zero. Validates the ``max(0.0, ...)``
    guard."""
    assert _pop_from_delta(1.5) == 0.0
    assert _pop_from_delta(-1.5) == 0.0
