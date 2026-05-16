"""Conformance test: skill 03 — Credit-to-Width floor.

Skill 03 §2 documents the formula::

    required C/W = |Δ_short| × (1 + edge_buffer)

This test asserts the live ``_cw_floor`` implementation in
``chain_scanner.py`` produces exactly that value across a range of
deltas + edge buffers. If the formula drifts, this test flunks
loud before any trade decision relies on the new (incorrect) floor.

Failure mode caught:
- Someone "simplifies" _cw_floor to drop the edge_buffer multiplier
- Someone renames edge_buffer and breaks the math
- Someone changes from |delta| to delta (signed)
"""

from __future__ import annotations

import pytest

from trading_agent.chain_scanner import _cw_floor


@pytest.mark.parametrize("delta,edge_buffer,expected", [
    # Skill 03 §2 example: delta-0.30, edge 0.10 → floor 0.33
    (-0.30, 0.10, 0.33),
    (0.30, 0.10, 0.33),    # call side same magnitude
    (-0.30, 0.0,  0.30),   # zero edge buffer → bare break-even
    (-0.45, 0.10, 0.495),  # higher delta scales linearly
    (-0.15, 0.20, 0.18),   # lower delta with bigger buffer
])
def test_skill_03_cw_floor_matches_documented_formula(
    delta: float, edge_buffer: float, expected: float,
) -> None:
    actual = _cw_floor(delta, edge_buffer)
    assert abs(actual - expected) < 1e-9, (
        f"Skill 03 §2 says |Δ|×(1+edge) should give {expected} for "
        f"delta={delta}, edge={edge_buffer} — got {actual}. "
        f"Re-verify the formula in chain_scanner.py:_cw_floor."
    )


def test_skill_03_cw_floor_uses_absolute_delta() -> None:
    """The |Δ| in the formula must be absolute value — sign-agnostic."""
    # Same magnitude, opposite signs → same floor.
    assert _cw_floor(-0.30, 0.10) == _cw_floor(0.30, 0.10)
