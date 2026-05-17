"""Conformance test: skill 02 — Strike snapping to grid.

Skill 02 §2 documents the snap formula::

    snapped = grid_step × max(1, ceil(raw_width / grid_step))

i.e. round UP to the nearest multiple of the chain's strike grid,
with a floor of one full step. The "round up" direction is critical
— rounding down would silently narrow the spread and break the
C/W floor downstream.

Skill 02 §4 covers two edge cases:
  * `raw_width < grid_step` snaps up to `grid_step` (one strike)
  * `raw_width` exactly on a grid multiple stays put

Failure modes caught:
- Someone "fixes" the snapper to round to nearest (would round 4.0 to 5.0
  but 4.4 to 5.0 too — fine on $5 grids but wrong on $1 grids)
- Someone removes the max(1, ...) floor — sub-step widths return 0
- Someone replaces `ceil` with `round` — produces snap-down for some inputs
"""

from __future__ import annotations

import pytest

from trading_agent.chain_scanner import ChainScanner


class TestSkill02SnapDirection:
    """Skill 02 §2 — direction-of-rounding contract: always UP."""

    @pytest.mark.parametrize("raw,step,expected", [
        # Sub-step → snap up to one full step (skill 02 §4 floor)
        (0.5, 1.0, 1.0),
        (0.5, 5.0, 5.0),
        # Just-over a step → snap up to next multiple
        (1.1, 1.0, 2.0),
        (5.1, 5.0, 10.0),
        # Exactly on grid → stay
        (5.0, 5.0, 5.0),
        (1.0, 1.0, 1.0),
        # Mid-stride → snap up
        (4.0, 5.0, 5.0),
        (7.0, 5.0, 10.0),
        (12.0, 5.0, 15.0),
    ])
    def test_snap_rounds_up(self, raw: float, step: float, expected: float) -> None:
        actual = ChainScanner._snap_width_to_grid(raw, step)
        assert actual == expected, (
            f"Skill 02 §2: snap({raw}, {step}) should be {expected}, "
            f"got {actual}. Snapping must round UP — rounding down would "
            f"narrow the spread and break the C/W floor."
        )


def test_skill_02_one_step_floor() -> None:
    """Skill 02 §4: a width below the grid step still snaps to one
    full step (never zero). Zero would be a degenerate spread."""
    # Tiny raw → must produce at least one step
    assert ChainScanner._snap_width_to_grid(0.01, 5.0) >= 5.0
    assert ChainScanner._snap_width_to_grid(0.0, 1.0) >= 1.0


def test_skill_02_fractional_grid_step() -> None:
    """Penny-pilot chains have $0.50 grids — the snapper must handle
    fractional steps without rounding to integer multiples."""
    # 0.7 with 0.5 grid → snap to 1.0
    assert ChainScanner._snap_width_to_grid(0.7, 0.5) == 1.0
    # 1.3 with 0.5 grid → snap to 1.5
    assert ChainScanner._snap_width_to_grid(1.3, 0.5) == 1.5
