"""Conformance test: skill 29 — Per-leg liquidity gate.

Skill 29 §2 documents the AND-of-thresholds gate::

    REJECT if  spread > max_leg_spread_cents
         AND  pct_of_mid > max_leg_spread_pct_mid

Skill 29 §4 lists the canonical worked examples (GLD wide, XLF penny,
SPY tight). This test asserts each example produces the documented
verdict using the default 0.15 / 0.05 thresholds.

Failure mode caught:
- Someone "fixes" the AND to OR — XLF penny options would suddenly
  fail and the agent stops trading them
- Someone tightens the absolute cap to e.g. 0.05 — SPY options
  start failing the gate they're documented as passing
"""

from __future__ import annotations

import pytest

from trading_agent.chain_scanner import _leg_spread_too_wide


class TestSkill29DocumentedExamples:
    """Skill 29 §4 — worked examples with default thresholds (0.15 / 0.05).

    Each test value pair is quoted verbatim from the skill so the test
    fails if the documented examples ever drift from the actual gate.
    """

    @pytest.mark.parametrize("label,bid,ask,expect", [
        # Skill 29 §4: "GLD 2026-05-15: 35¢ spread, 6.6% of mid → REJECTED"
        ("GLD wide-spread put", 5.15, 5.50, True),
        # Skill 29 §4: "XLF penny: 5¢ spread, 66% of mid → passes via absolute"
        ("XLF penny option", 0.05, 0.10, False),
        # Skill 29 §4: "SPY tight: 10¢ on $4.05 mid (2.5%) → passes"
        ("SPY tight spread", 4.00, 4.10, False),
    ])
    def test_documented_example_matches_gate(
        self, label: str, bid: float, ask: float, expect: bool,
    ) -> None:
        actual = _leg_spread_too_wide(bid, ask, 0.15, 0.05)
        assert actual is expect, (
            f"Skill 29 §4 documents {label} as "
            f"{'REJECTED' if expect else 'ACCEPTED'} at default thresholds; "
            f"actual gate returned {'REJECTED' if actual else 'ACCEPTED'}. "
            f"Re-verify chain_scanner._leg_spread_too_wide."
        )


def test_skill_29_and_not_or_semantics() -> None:
    """Skill 29 §1 explicitly explains the AND-of-thresholds design
    rationale: 'A leg has to fail BOTH to be rejected'. A future
    contributor flipping AND→OR would break XLF tradability.

    Verify: a leg failing ONLY the relative cap (high % of mid but
    low absolute spread) must NOT trigger the gate.
    """
    # 10¢ spread on $0.30 mid = 33% of mid (fails relative cap 5%)
    # But 10¢ absolute is below the 15¢ cap (passes absolute)
    # AND-semantics → not rejected. OR-semantics would reject.
    assert _leg_spread_too_wide(0.25, 0.35, 0.15, 0.05) is False, (
        "Skill 29 §1: a leg failing only the relative cap should NOT "
        "trip the gate (AND-of-thresholds). If this fails, the gate "
        "has been changed to OR-semantics — XLF penny options will "
        "stop trading."
    )
