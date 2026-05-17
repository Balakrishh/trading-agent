"""Conformance test: skill 13 — Preset system & hot-reload.

Skill 13 §2 documents the contract for ``load_active_preset`` and
``save_active_preset``:

  * The active preset lives in ``STRATEGY_PRESET.json``
  * Missing/malformed/unknown-name → fall back to BALANCED (no crash)
  * Saving a custom preset persists the full overrides dict
  * Round-trip save → load reproduces the same PresetConfig

This conformance test pins the documented fallback + round-trip.
The full hot-reload integration (file watcher → cache invalidation
→ next cycle picks up the new values) is integration-scoped and
covered by ``tests/test_streamlit/test_live_monitor.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from trading_agent.strategy_presets import (
    BALANCED, CONSERVATIVE, AGGRESSIVE, PRESETS,
    load_active_preset, save_active_preset,
)


def test_skill_13_three_named_presets_registered() -> None:
    """Skill 13 §2: three named presets — conservative / balanced /
    aggressive — are the canonical menu the dashboard exposes."""
    assert set(PRESETS.keys()) == {"conservative", "balanced", "aggressive"}
    assert PRESETS["balanced"] is BALANCED
    assert PRESETS["conservative"] is CONSERVATIVE
    assert PRESETS["aggressive"] is AGGRESSIVE


def test_skill_13_missing_file_falls_back_to_balanced(tmp_path: Path) -> None:
    """Skill 13 §4: a missing STRATEGY_PRESET.json must not crash —
    the loader returns BALANCED as a safe default."""
    fake_path = tmp_path / "does_not_exist.json"
    result = load_active_preset(path=fake_path)
    assert result.name == BALANCED.name, (
        f"Skill 13 §4: load_active_preset on missing file should fall "
        f"back to BALANCED; got preset name {result.name!r}."
    )


def test_skill_13_malformed_json_falls_back(tmp_path: Path) -> None:
    """Skill 13 §4: corrupt JSON file shouldn't take down the agent."""
    bad = tmp_path / "STRATEGY_PRESET.json"
    bad.write_text("{ this is not valid JSON ::: ")
    result = load_active_preset(path=bad)
    assert result.name == BALANCED.name


def test_skill_13_unknown_profile_name_falls_back(tmp_path: Path) -> None:
    """Skill 13 §4: a JSON file naming a non-existent profile (e.g.
    'experimental') falls back to BALANCED rather than KeyError."""
    bogus = tmp_path / "STRATEGY_PRESET.json"
    bogus.write_text(json.dumps({"profile": "experimental_unknown"}))
    result = load_active_preset(path=bogus)
    assert result.name == BALANCED.name


def test_skill_13_save_then_load_round_trip(tmp_path: Path) -> None:
    """Skill 13 §3: save_active_preset writes a JSON the loader can
    read back into the same PresetConfig."""
    target = tmp_path / "STRATEGY_PRESET.json"
    save_active_preset("aggressive", "auto", path=target)
    loaded = load_active_preset(path=target)
    assert loaded.name == "aggressive"
