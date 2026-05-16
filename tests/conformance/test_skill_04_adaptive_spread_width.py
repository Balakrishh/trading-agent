"""Conformance test: skill 04 — Adaptive spread width.

Skill 04 §3 documents the snap-to-grid behavior::

    snap UP to the nearest grid step (never down)

Critical because a snap-down would produce a tighter spread than the
preset's width_grid_pct requested, silently reducing credit and
breaking the C/W floor downstream.

Skill 04 §6 (added 2026-05-15) documents the named-preset grid shapes.
This test pins those grids so a future contributor can't accidentally
drop `0.005` from BALANCED.width_grid_pct without flunking conformance.
"""

from __future__ import annotations

import pytest

from trading_agent.chain_scanner import ChainScanner
from trading_agent.strategy_presets import (
    AGGRESSIVE, BALANCED, CONSERVATIVE,
)


class TestSkill04SnapBehavior:
    """Skill 04 §3 — snap_width_to_grid rounds UP."""

    def test_snap_up_below_step(self) -> None:
        # 4.0 with 5-wide grid → 5.0 (up, not 0)
        assert ChainScanner._snap_width_to_grid(4.0, 5.0) == 5.0

    def test_snap_up_above_step(self) -> None:
        # 7.0 with 5-wide grid → 10.0 (up to next multiple)
        assert ChainScanner._snap_width_to_grid(7.0, 5.0) == 10.0

    def test_snap_on_step_stays_put(self) -> None:
        # Exact multiple should NOT snap further
        assert ChainScanner._snap_width_to_grid(5.0, 5.0) == 5.0

    def test_snap_never_goes_below_one_step(self) -> None:
        # 0.5 with 1-wide grid → 1.0 (minimum one strike step)
        assert ChainScanner._snap_width_to_grid(0.5, 1.0) >= 1.0


class TestSkill04PresetGrids:
    """Skill 04 §6 — 2026-05-15 grid retune.

    All three named presets must contain specific grid entries. The
    most important one is BALANCED's 0.005 entry — without it, SPY/QQQ
    are budget-incompatible at any other width.
    """

    def test_balanced_grid_includes_narrow_widths_for_high_spot_tickers(self) -> None:
        """BALANCED must offer 0.5% width for SPY/QQQ budget-fit."""
        assert 0.005 in BALANCED.width_grid_pct, (
            "Skill 04 §6 requires BALANCED.width_grid_pct to include 0.005. "
            "Without it, high-spot tickers (SPY $733, QQQ $700+) can't "
            "produce a budget-eligible spread at any width — see "
            "the 'include SPY without raising budget' discussion."
        )

    def test_balanced_grid_includes_delta_040(self) -> None:
        """BALANCED extends to delta-0.40 for IV-rich premium capture."""
        assert 0.40 in BALANCED.delta_grid

    def test_balanced_grid_includes_45_dte(self) -> None:
        """BALANCED extends to 45-DTE for more total theta."""
        assert 45 in BALANCED.dte_grid

    def test_conservative_grid_stays_conservative(self) -> None:
        """Conservative must NOT carry the 0.5% width (per skill 04 §6:
        'shouldn't be trading SPY-style spreads in the first place')."""
        assert 0.005 not in CONSERVATIVE.width_grid_pct, (
            "Skill 04 §6 explicitly excludes 0.5% width from CONSERVATIVE — "
            "the preset is meant to keep budget-safe trades only."
        )
        # And the max delta should stay at-or-below 0.30
        assert max(CONSERVATIVE.delta_grid) <= 0.30

    def test_aggressive_grid_extends_higher(self) -> None:
        """Aggressive admits up to delta-0.45 + 3% width."""
        assert 0.45 in AGGRESSIVE.delta_grid
        assert 0.030 in AGGRESSIVE.width_grid_pct


class TestSkill04GoldilocksZone:
    """Skill 04 §7 — the IVRank Goldilocks zone (added 2026-05-15).

    Documents that defense_first blocks new entries when IVRank > 95.
    This test pins the threshold so a refactor can't silently change
    it without updating the skill.
    """

    def test_default_high_iv_threshold_is_95(self) -> None:
        """Skill 04 §7: 'defense_first filter blocks ... IVRank > 95'."""
        from trading_agent.regime import RegimeClassifier
        import inspect
        sig = inspect.signature(RegimeClassifier._compute_iv_rank)
        # Find the high_iv_pct param default
        for name, param in sig.parameters.items():
            if name == "high_iv_pct":
                assert param.default == 95.0, (
                    f"Skill 04 §7 documents IVRank>95 as the defense_first "
                    f"threshold. RegimeClassifier._compute_iv_rank default "
                    f"is {param.default} — re-stamp skill 04 or revert."
                )
                return
        pytest.fail("RegimeClassifier._compute_iv_rank has no high_iv_pct param")
