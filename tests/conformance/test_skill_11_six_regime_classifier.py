"""Conformance test: skill 11 — Six-regime classifier.

Skill 11 §1 lists the regime enum members that drive strategy
selection. Despite the skill's name ("six-regime"), the canonical
enum carries four values — the "six" refers to the six-way priority
order in the strategy-selection table (skill 11 §3) which composes
regime + leadership-z + VIX-z into final strategy choice.

The four enum members are:
  * BULLISH
  * BEARISH
  * SIDEWAYS
  * MEAN_REVERSION

Any change to the enum requires updating skill 11's §2 + every
strategy router that switches on regime.

Failure modes caught:
- Someone renames an enum member (silently breaks every str() comparison)
- Someone adds a new member without updating skill 11
- Someone changes a member's string value (e.g., "bullish" → "bull")
  which breaks every journal-row regime field
"""

from __future__ import annotations

from trading_agent.regime import Regime


def test_skill_11_enum_has_four_canonical_members() -> None:
    """Skill 11 §2: the four canonical regimes."""
    members = {m.name for m in Regime}
    expected = {"BULLISH", "BEARISH", "SIDEWAYS", "MEAN_REVERSION"}
    assert members == expected, (
        f"Skill 11 §2: Regime enum must have exactly "
        f"{expected}; got {members}. If a new regime was added, "
        f"update skill 11 §2 + the strategy-router dispatch table."
    )


def test_skill_11_bullish_str_value_is_lowercase() -> None:
    """Skill 11 §2: enum string values are lowercase (journal rows
    record them verbatim — changing case would break dashboard
    rendering + every regime-match downstream)."""
    assert Regime.BULLISH.value == "bullish"


def test_skill_11_bearish_str_value() -> None:
    assert Regime.BEARISH.value == "bearish"


def test_skill_11_sideways_str_value() -> None:
    assert Regime.SIDEWAYS.value == "sideways"


def test_skill_11_mean_reversion_str_value() -> None:
    """Skill 11 §2: mean_reversion (with underscore) — not "mean-reversion"
    or "MEAN REVERSION". Journal regex matching relies on the snake_case
    form."""
    assert Regime.MEAN_REVERSION.value == "mean_reversion"


def test_skill_11_enum_is_hashable() -> None:
    """Skill 11 §3: the strategy router uses Regime members as dict
    keys for dispatch. Members must be hashable (Enum guarantees this
    but we pin it as a contract)."""
    dispatch = {r: r.name for r in Regime}
    assert len(dispatch) == 4
