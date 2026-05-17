"""Conformance test: skill 05 — EV per $-risked scoring.

Skill 05 §2 documents the EV formula::

    EV_per_$_risked = (POP × CW − (1 − POP) × (1 − CW)) / (1 − CW)

where POP = 1 − |Δshort| and CW = credit / width.

This test asserts the live ``_ev_per_dollar_risked`` implementation
matches that math across a representative grid + verifies the §4
sentinel-return behavior for invalid inputs.

Failure modes caught:
- Formula simplification gone wrong (e.g., dropping the /(1−CW))
- Returns 0 instead of None for invalid input → downstream scoring
  divides by None and crashes
- Sign error on (1 − POP) × (1 − CW) — would invert the loss term
"""

from __future__ import annotations

import pytest

from trading_agent.chain_scanner import _ev_per_dollar_risked


@pytest.mark.parametrize("credit,width,short_delta,expected_ev", [
    # Skill 05 §2 worked example: credit $2 on $5 width at Δ-0.30
    #   POP = 0.70, CW = 0.40
    #   EV_per_W = 0.70×0.40 − 0.30×0.60 = 0.28 − 0.18 = 0.10
    #   EV_per_$ = 0.10 / 0.60 ≈ 0.1667
    (2.0, 5.0, -0.30, 0.10 / 0.60),
    # Higher credit (still under-width) at same delta: more EV
    (3.0, 5.0, -0.30, (0.70 * 0.60 - 0.30 * 0.40) / 0.40),
    # At the C/W breakeven (CW = |Δ|): EV = 0
    (1.5, 5.0, -0.30, 0.0),
    # Higher-delta short: same width + credit → different EV
    (2.0, 5.0, -0.20, (0.80 * 0.40 - 0.20 * 0.60) / 0.60),
])
def test_skill_05_ev_matches_documented_formula(
    credit: float, width: float, short_delta: float, expected_ev: float,
) -> None:
    actual = _ev_per_dollar_risked(credit, width, short_delta)
    assert actual is not None, (
        f"Skill 05: expected a number but got None for "
        f"credit={credit}, width={width}, Δ={short_delta}."
    )
    assert actual == pytest.approx(expected_ev, abs=1e-6), (
        f"Skill 05 §2: EV/$risked({credit},{width},{short_delta}) "
        f"should be {expected_ev}, got {actual}. "
        f"Check chain_scanner._ev_per_dollar_risked."
    )


class TestSkill05SentinelReturns:
    """Skill 05 §4 — explicit ``None`` sentinel for structural failures."""

    def test_zero_credit_returns_none(self) -> None:
        """Credit ≤ 0 means there's no spread to score — return None."""
        assert _ev_per_dollar_risked(0.0, 5.0, -0.30) is None
        assert _ev_per_dollar_risked(-0.5, 5.0, -0.30) is None

    def test_zero_width_returns_none(self) -> None:
        """Width ≤ 0 means the spread is degenerate — return None."""
        assert _ev_per_dollar_risked(2.0, 0.0, -0.30) is None
        assert _ev_per_dollar_risked(2.0, -1.0, -0.30) is None

    def test_credit_ge_width_returns_none(self) -> None:
        """Credit ≥ width means this is a debit spread, not a credit
        spread — the scoring formula doesn't apply. Return None."""
        assert _ev_per_dollar_risked(5.0, 5.0, -0.30) is None
        assert _ev_per_dollar_risked(6.0, 5.0, -0.30) is None
