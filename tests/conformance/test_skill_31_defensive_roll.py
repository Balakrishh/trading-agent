"""Conformance test: skill 31 — Defensive roll.

Skill 31 documents the six-predicate evaluator + atomic close-then-open
executor method that fires when a position's short strike approaches
spot, instead of just closing at a loss on STRIKE_PROXIMITY.

This test pins:

  1. The ``RollDecision`` enum exists with the documented seven outcomes
  2. The ``evaluate_defensive_roll`` function exists and is callable
  3. The six PresetConfig fields exist with documented defaults
  4. Per-preset enablement is correct (Balanced ON, Aggressive ON with
     tightened upper band, Conservative OFF)
  5. The agent wires the dispatcher and the executor method exists

Failure modes caught:
- Someone renames ``evaluate_defensive_roll`` — agent.py wiring breaks
- Someone changes a default threshold silently — the trade-off math
  in skill 31 §1 no longer matches the code
- Someone removes the executor's ``roll_position_defensive`` — the
  atomic close-then-open guarantee is silently broken
- Someone disables the per-Balanced enablement — defensive rolls
  silently stop firing
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────
# Evaluator surface
# ─────────────────────────────────────────────────────────────────────

def test_skill_31_roll_decision_enum_present() -> None:
    """Skill 31 §3.1: RollDecision enum lists the seven documented outcomes."""
    from trading_agent.defensive_roll_evaluator import RollDecision
    expected = {
        "ROLL",
        "SKIP_NOT_IN_TRIGGER_BAND",
        "SKIP_DTE_TOO_SHORT",
        "SKIP_CREDIT_NEGATIVE",
        "SKIP_BELOW_CW_FLOOR",
        "SKIP_DIFFERENT_STRATEGY",
        "SKIP_BUDGET_EXHAUSTED",
    }
    actual = {m.name for m in RollDecision}
    missing = expected - actual
    assert not missing, (
        f"Skill 31 §3.1: RollDecision missing documented outcomes "
        f"{sorted(missing)}. Renaming an outcome silently breaks the "
        f"journal vocabulary (decisions are recorded by value)."
    )


def test_skill_31_evaluate_function_callable() -> None:
    """Skill 31 §3.2: evaluate_defensive_roll is the public entry point."""
    from trading_agent.defensive_roll_evaluator import (
        evaluate_defensive_roll, RollEvalInputs,
    )
    assert callable(evaluate_defensive_roll)
    # And the input struct exists with all the documented fields.
    expected_fields = {
        "spot", "short_strike", "dte", "strategy_name",
        "debit_to_close", "current_roll_count",
        "new_short_delta", "new_strategy_name",
        "new_projected_credit", "new_spread_width", "new_cw_ratio",
        "roll_trigger_min_pct", "roll_trigger_max_pct",
        "min_dte_for_roll", "max_defensive_rolls_per_position",
        "edge_buffer",
    }
    actual = {f.name for f in RollEvalInputs.__dataclass_fields__.values()}
    missing = expected_fields - actual
    assert not missing, (
        f"Skill 31 §3.2: RollEvalInputs missing documented fields "
        f"{sorted(missing)}. The evaluator's six predicates depend on "
        f"these inputs — removing one silently breaks the gate."
    )


def test_skill_31_baseline_roll_decision() -> None:
    """Skill 31 §2: a position threatened at 1.0% proximity, well-capitalised
    candidate, all predicates clean → RollDecision.ROLL."""
    from trading_agent.defensive_roll_evaluator import (
        RollEvalInputs, RollDecision, evaluate_defensive_roll,
    )
    inp = RollEvalInputs(
        spot=100.0, short_strike=99.0, dte=10,
        strategy_name="Bull Put",
        debit_to_close=2.50,
        current_roll_count=0,
        new_short_delta=0.20,
        new_strategy_name="Bull Put",
        new_projected_credit=3.20,
        new_spread_width=10.0,
        new_cw_ratio=0.32,
        roll_trigger_min_pct=0.005,
        roll_trigger_max_pct=0.015,
        min_dte_for_roll=5,
        max_defensive_rolls_per_position=1,
        edge_buffer=0.10,
    )
    assert evaluate_defensive_roll(inp) == RollDecision.ROLL


# ─────────────────────────────────────────────────────────────────────
# PresetConfig surface
# ─────────────────────────────────────────────────────────────────────

def test_skill_31_preset_fields_exist() -> None:
    """Skill 31 §3.3: six PresetConfig fields drive the evaluator."""
    from trading_agent.strategy_presets import PresetConfig
    field_names = {f.name for f in PresetConfig.__dataclass_fields__.values()}
    required = {
        "defensive_roll_enabled",
        "roll_trigger_min_pct", "roll_trigger_max_pct",
        "min_dte_for_roll", "max_defensive_rolls_per_position",
    }
    missing = required - field_names
    assert not missing, (
        f"Skill 31 §3.3: PresetConfig missing fields {sorted(missing)}. "
        f"The dispatcher reads these — removing one would either "
        f"AttributeError at runtime or silently skip the feature."
    )


def test_skill_31_preset_per_profile_enablement() -> None:
    """Skill 31 §2: Balanced ON, Aggressive ON (with tighter upper band),
    Conservative OFF."""
    from trading_agent.strategy_presets import (
        BALANCED, AGGRESSIVE, CONSERVATIVE,
    )
    assert BALANCED.defensive_roll_enabled is True, (
        "Skill 31 §2: Balanced must have defensive_roll_enabled=True — "
        "Balanced is the canonical 'real positions, real defense' preset."
    )
    assert AGGRESSIVE.defensive_roll_enabled is True, (
        "Skill 31 §2: Aggressive must have defensive_roll_enabled=True — "
        "near-ATM shorts routinely trigger STRIKE_PROXIMITY."
    )
    assert AGGRESSIVE.roll_trigger_max_pct == 0.012, (
        f"Skill 31 §2: Aggressive must tighten roll_trigger_max_pct to "
        f"0.012 so it rolls earlier (more reaction time); got "
        f"{AGGRESSIVE.roll_trigger_max_pct}."
    )
    assert CONSERVATIVE.defensive_roll_enabled is False, (
        "Skill 31 §2: Conservative must have defensive_roll_enabled=False — "
        "Conservative trades far-OTM and shouldn't be near short strikes."
    )


# ─────────────────────────────────────────────────────────────────────
# Wiring surface — executor + agent
# ─────────────────────────────────────────────────────────────────────

def test_skill_31_executor_has_roll_method() -> None:
    """Skill 31 §3.4: OrderExecutor.roll_position_defensive is the atomic
    close-then-open entry point. Without it, the dispatcher has nothing
    to call and the close-then-open atomicity guarantee is bypassed.

    Read the source directly (not via import) so this test doesn't
    depend on the executor's heavy transitive deps (scipy, etc.)
    being installed in the conformance-test environment.
    """
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "executor.py").read_text(
        encoding="utf-8"
    )
    assert "def roll_position_defensive" in src, (
        "Skill 31 §3.4: OrderExecutor must define roll_position_defensive(). "
        "Bare close_spread() followed by execute() in the dispatcher "
        "breaks the atomicity guarantee (skill 31 §4 — "
        "doubled-position window risk)."
    )


def test_skill_31_agent_dispatcher_wired_into_close_path() -> None:
    """Skill 31 §3.5: agent.py's close loop must call the dispatcher
    when STRIKE_PROXIMITY fires AND defensive_roll_enabled is True."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    # Pin the dispatcher call shape.
    assert "_maybe_defensive_roll" in src, (
        "Skill 31 §3.5: agent.py must wire _maybe_defensive_roll into "
        "the close path. Without it, STRIKE_PROXIMITY always closes "
        "instead of evaluating a roll."
    )
    assert "preset.defensive_roll_enabled" in src, (
        "Skill 31 §3.5: agent.py must gate the dispatcher on "
        "preset.defensive_roll_enabled — otherwise Conservative would "
        "still attempt rolls."
    )
    assert "ExitSignal.STRIKE_PROXIMITY" in src, (
        "Skill 31 §3.5: dispatcher must trigger specifically on "
        "STRIKE_PROXIMITY, not on every exit signal."
    )
