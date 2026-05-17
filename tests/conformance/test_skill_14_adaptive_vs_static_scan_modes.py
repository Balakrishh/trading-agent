"""Conformance test: skill 14 — Adaptive vs static scan modes.

Skill 14 §2 documents the ``scan_mode`` field on PresetConfig:

  * "static"   → planner uses scalar ``max_delta``, ``dte_*``, ``width_*``
  * "adaptive" → scanner sweeps ``delta_grid × dte_grid × width_grid_pct``

The default is "static" so legacy presets keep working unchanged.
Only "adaptive" engages the chain scanner's grid sweep.

Failure modes caught:
- Someone changes the default to "adaptive" — every legacy preset
  that hasn't explicitly opted in starts sweeping (potentially
  pulling thousands of grid points per cycle)
- The literal type drifts from "static"/"adaptive" to something
  else, silently breaking the dispatch in chain_scanner
"""

from __future__ import annotations

from dataclasses import replace

from trading_agent.strategy_presets import BALANCED, PresetConfig


def test_skill_14_default_scan_mode_is_static() -> None:
    """Skill 14 §2: PresetConfig.scan_mode defaults to 'static' so
    legacy presets without an explicit mode setting keep their
    pre-adaptive behavior."""
    # BALANCED is constructed without explicitly passing scan_mode.
    assert BALANCED.scan_mode == "static", (
        f"Skill 14 §2: BALANCED.scan_mode default must be 'static'; "
        f"got {BALANCED.scan_mode!r}. Changing this is a breaking "
        f"behavior change — every legacy preset would suddenly start "
        f"sweeping the adaptive grid."
    )


def test_skill_14_adaptive_value_is_literal_string() -> None:
    """Skill 14 §3: scan_mode is a ``Literal["static", "adaptive"]``
    type. Construction with the literal string must succeed."""
    adaptive_balanced = replace(BALANCED, scan_mode="adaptive")
    assert adaptive_balanced.scan_mode == "adaptive"


def test_skill_14_adaptive_uses_grid_fields() -> None:
    """Skill 14 §2: in adaptive mode the chain scanner reads
    ``delta_grid``, ``dte_grid``, ``width_grid_pct`` (which are
    tuples). Those fields must exist on PresetConfig."""
    p = BALANCED
    assert hasattr(p, "delta_grid")
    assert hasattr(p, "dte_grid")
    assert hasattr(p, "width_grid_pct")
    # Grids are tuples (frozen=True dataclass requirement).
    assert isinstance(p.delta_grid, tuple)
    assert isinstance(p.dte_grid, tuple)
    assert isinstance(p.width_grid_pct, tuple)


def test_skill_14_grid_fields_are_non_empty() -> None:
    """A preset with empty grids would silently produce zero
    candidates per cycle in adaptive mode — caught here at config
    construction rather than at trade time."""
    for grid in (BALANCED.delta_grid, BALANCED.dte_grid, BALANCED.width_grid_pct):
        assert len(grid) > 0
