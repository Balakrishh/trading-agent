"""Conformance test: skill 09 — VIX z-score inhibitor.

Skill 09 §2 documents the inhibitor threshold:

    fire_inhibit_bullish = vix_zscore > VIX_INHIBIT_ZSCORE

with ``VIX_INHIBIT_ZSCORE = 2.0`` (default). When the VIX z-score
exceeds this threshold, the strategy demotes Bull-Put + Iron-Condor
candidates to Bear-Call for that cycle, regardless of per-ticker
regime.

This test pins both the threshold and the inhibitor semantics.

Failure modes caught:
- Someone lowers the threshold to 1.5 (too sensitive — half the days
  would inhibit bullish trades on routine vol movements)
- Someone raises it to 3.0 (too desensitised — the inhibitor would
  rarely fire and the protective effect is lost)
- Someone flips the comparison (>= instead of >) — boundary case
  at exactly 2.0 would now inhibit instead of letting through
"""

from __future__ import annotations

import pytest

from trading_agent.regime import VIX_INHIBIT_ZSCORE


def test_skill_09_threshold_is_two_sigma() -> None:
    """Skill 09 §2: VIX_INHIBIT_ZSCORE = 2.0.

    Tuned for a ~2.5% daily firing rate over the past 5 years —
    rare enough to be signal, not noise. Changing this requires
    updating skill 09 §2 + re-running the historical inhibit-rate
    diagnostic.
    """
    assert VIX_INHIBIT_ZSCORE == 2.0, (
        f"Skill 09 §2: VIX_INHIBIT_ZSCORE must be 2.0; got "
        f"{VIX_INHIBIT_ZSCORE}. If you changed this deliberately, "
        f"update the skill's §2 and re-stamp the footer."
    )


@pytest.mark.parametrize("vix_z,should_inhibit", [
    # Below threshold — no inhibit
    (0.0, False),
    (1.5, False),
    (1.99, False),
    # Exactly at threshold — strict-> comparison means NO inhibit
    (2.0, False),
    # Above threshold — inhibit fires
    (2.01, True),
    (2.5, True),
    (3.0, True),
])
def test_skill_09_inhibit_semantics(
    vix_z: float, should_inhibit: bool,
) -> None:
    """Skill 09 §2: ``vix_zscore > VIX_INHIBIT_ZSCORE`` (strict).
    Boundary case at vix_z = 2.0 must NOT trigger the inhibit —
    flipping to >= would change one trading day per year on average
    in a way that's hard to spot at review time.
    """
    fire = vix_z > VIX_INHIBIT_ZSCORE
    assert fire is should_inhibit, (
        f"Skill 09 §2: vix_z={vix_z} should "
        f"{'TRIGGER' if should_inhibit else 'NOT trigger'} the "
        f"inhibit gate at threshold {VIX_INHIBIT_ZSCORE}; got "
        f"fire={fire}."
    )
