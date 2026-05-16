"""Conformance test: skill 17 — Close-failure + position cap.

Skill 17 §3 documents the per-ticker cap constant + the new (2026-05-15)
per-sector cap sibling. This test pins both constants so a refactor
can't silently raise them without updating the skill.

Failure mode caught:
- Someone bumps MAX_POSITIONS_PER_TICKER to 2 thinking "more positions
  = more income" without realising it bypasses the 2026-05-12 GLD
  dedup-gate-bypass guardrail
- Someone removes the per-sector cap thinking it's redundant with
  per-ticker — the 'XLF + KRE + KBE' concentration risk re-opens
"""

from __future__ import annotations

from trading_agent.agent import MAX_POSITIONS_PER_TICKER
from trading_agent.sector_map import MAX_POSITIONS_PER_SECTOR


def test_skill_17_per_ticker_cap_is_one() -> None:
    """Skill 17 §3: MAX_POSITIONS_PER_TICKER = 1 (hard cap)."""
    assert MAX_POSITIONS_PER_TICKER == 1, (
        f"Skill 17 §3 documents MAX_POSITIONS_PER_TICKER = 1. "
        f"Got {MAX_POSITIONS_PER_TICKER}. If you raised this, also "
        f"update skill 17 §3 to reflect the new cap and document the "
        f"intentional stacked-spread policy."
    )


def test_skill_17_per_sector_cap_default_is_two() -> None:
    """Skill 17 'Sibling: per-sector position cap' section:
    MAX_POSITIONS_PER_SECTOR default of 2 prevents over-concentration."""
    assert MAX_POSITIONS_PER_SECTOR == 2, (
        f"Skill 17 sibling section documents MAX_POSITIONS_PER_SECTOR = 2. "
        f"Got {MAX_POSITIONS_PER_SECTOR}. Raising this allows e.g. "
        f"XLF + KRE + KBE all simultaneously — confirm sector "
        f"diversification is still intended."
    )


def test_skill_17_cap_imports_from_sector_map_module() -> None:
    """Skill 17 says the sector cap lives in trading_agent.sector_map.
    Verify the import path remains stable (consumers of the cap rely
    on this exact module location)."""
    import trading_agent.sector_map as sm
    assert hasattr(sm, "MAX_POSITIONS_PER_SECTOR")
    assert hasattr(sm, "sector_for")
    assert hasattr(sm, "TICKER_SECTOR_MAP")
