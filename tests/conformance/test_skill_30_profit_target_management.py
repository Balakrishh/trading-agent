"""Conformance test: skill 30 — Profit-target management.

Skill 30 §2 documents that ``profit_target_pct`` is a per-preset tunable
and the three built-in presets carry distinct defaults:

  conservative  → 0.60
  balanced      → 0.50
  aggressive    → 0.40

§3.3 documents that ``agent.py`` threads ``preset.profit_target_pct``
into ``PositionMonitor`` so the live exit rule matches the preset.
§3.4 documents the backtester parity wiring.

Failure modes caught:
- Someone changes a preset default silently (e.g. bumps Balanced
  from 0.50 → 0.70) without updating the skill — the trade-off math
  in §1 no longer matches the code
- Someone removes the field from ``PresetConfig`` — the live agent
  falls back to PositionMonitor's hardcoded 0.50, but Conservative
  and Aggressive would silently regress to Balanced behavior
- Someone wires the live exit but forgets the backtester (or
  vice versa) — skill 15 (backtest↔live parity) is broken silently
"""

from __future__ import annotations

import inspect

from trading_agent.strategy_presets import (
    CONSERVATIVE, BALANCED, AGGRESSIVE, PresetConfig,
)


def test_skill_30_field_exists_on_preset_config() -> None:
    """Skill 30 §3.2: ``profit_target_pct`` must be a PresetConfig field."""
    fields = {f.name for f in PresetConfig.__dataclass_fields__.values()}
    assert "profit_target_pct" in fields, (
        "Skill 30 §3.2: PresetConfig must carry a profit_target_pct field. "
        "Removing it makes Conservative/Aggressive silently regress to "
        "PositionMonitor's hardcoded 0.50 default."
    )


def test_skill_30_default_is_50_percent() -> None:
    """Skill 30 §3.2: the field's default value is 0.50 (industry standard)."""
    field = PresetConfig.__dataclass_fields__["profit_target_pct"]
    assert field.default == 0.50, (
        f"Skill 30 §3.2: profit_target_pct default must be 0.50; got "
        f"{field.default}. The 50% rule is documented as the baseline; "
        f"changing this shifts the meaning of every Custom preset that "
        f"doesn't explicitly override it."
    )


def test_skill_30_preset_specific_defaults() -> None:
    """Skill 30 §2: the three built-in presets carry distinct defaults.

    Conservative rides winners further (0.60), Aggressive recycles capital
    faster (0.40), Balanced is the industry-standard 0.50.
    """
    assert CONSERVATIVE.profit_target_pct == 0.60, (
        f"Skill 30 §2: Conservative must close at 60% of credit; got "
        f"{CONSERVATIVE.profit_target_pct}. Conservative trades fire less "
        f"often and need bigger $/trade to justify the universe."
    )
    assert BALANCED.profit_target_pct == 0.50, (
        f"Skill 30 §2: Balanced must close at 50% of credit (industry "
        f"standard); got {BALANCED.profit_target_pct}."
    )
    assert AGGRESSIVE.profit_target_pct == 0.40, (
        f"Skill 30 §2: Aggressive must close at 40% of credit; got "
        f"{AGGRESSIVE.profit_target_pct}. Aggressive cycles fire often and "
        f"benefit from faster capital recycling."
    )


def test_skill_30_overlay_round_trips_through_persistence() -> None:
    """Skill 30 §4: the overlay must survive a save → load round-trip
    and out-of-range values must fall back to the preset default."""
    import json
    import tempfile
    from pathlib import Path

    from trading_agent.strategy_presets import (
        save_active_preset, load_active_preset,
    )

    with tempfile.TemporaryDirectory() as d:
        fp = Path(d) / "STRATEGY_PRESET.json"
        save_active_preset(
            profile="balanced",
            directional_bias="auto",
            profit_target_pct=0.65,
            path=fp,
        )
        loaded = load_active_preset(fp)
        assert abs(loaded.profit_target_pct - 0.65) < 1e-9, (
            f"Skill 30 §4: overlay round-trip failed; expected 0.65, got "
            f"{loaded.profit_target_pct}."
        )

    # Out-of-range overlay should fall back, not crash.
    with tempfile.TemporaryDirectory() as d:
        fp = Path(d) / "STRATEGY_PRESET.json"
        fp.write_text(json.dumps({
            "profile": "aggressive",
            "directional_bias": "auto",
            "profit_target_pct": 1.5,   # invalid — > 0.95
        }))
        loaded = load_active_preset(fp)
        assert loaded.profit_target_pct == AGGRESSIVE.profit_target_pct, (
            f"Skill 30 §4: out-of-range overlay must fall back to preset "
            f"default {AGGRESSIVE.profit_target_pct}; got "
            f"{loaded.profit_target_pct}."
        )


def test_skill_30_agent_threads_preset_into_position_monitor() -> None:
    """Skill 30 §3.3: agent.py must pass preset.profit_target_pct into
    PositionMonitor's constructor.

    Pinning the wire-up textually catches a silent regression where the
    field stays on PresetConfig but the agent stops reading it — which
    would let presets diverge from the actual exit rule without any test
    failing.
    """
    # Read the file directly (not via import) so this test doesn't depend
    # on the live agent's full transitive deps (pandas_market_calendars, etc.)
    # being installed in the conformance-test environment.
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(encoding="utf-8")
    assert "profit_target_pct=self.preset.profit_target_pct" in src, (
        "Skill 30 §3.3: agent.py must thread self.preset.profit_target_pct "
        "into the PositionMonitor() instantiation. Without this, custom "
        "presets silently fall back to the hardcoded 0.50 default."
    )


def test_skill_30_backtester_threads_preset_into_evaluate_exit() -> None:
    """Skill 30 §3.4: backtest/runner.py must pass preset.profit_target_pct
    into evaluate_exit so backtest results match live behavior.

    Pinning skill 15's parity invariant — a divergence here would silently
    overstate Aggressive-preset performance in backtests because the
    backtester would still use 0.50 while live uses 0.40.
    """
    # Read directly — see rationale in
    # test_skill_30_agent_threads_preset_into_position_monitor.
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "backtest" / "runner.py").read_text(
        encoding="utf-8"
    )
    assert 'self.preset, "profit_target_pct"' in src, (
        "Skill 30 §3.4: backtest/runner.py must thread "
        "self.preset.profit_target_pct into evaluate_exit. Without this, "
        "backtests diverge from live behavior (skill 15 parity broken)."
    )
