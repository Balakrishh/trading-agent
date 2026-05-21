"""Conformance test: skill 33 — PDT-aware DTE cap.

Skill 33 documents the per-cycle DTE cap applied to the strategy
planner when the trading account is sub-$25K. Pins:

  1. PresetConfig.pdt_dte_cap field exists with default 14
  2. StrategyPlanner.apply_pdt_dte_cap method exists
  3. Per-strategy "_orig" attribute names match the documented contract
     (so a refactor can't accidentally drop the immutable shadow values
     that make the cap reactive across cycles)
  4. Agent calls apply_pdt_dte_cap once per cycle after computing
     pdt_restricted, with cap=preset.pdt_dte_cap when restricted and
     cap=None otherwise

Failure modes caught:
- Someone hardcodes a numeric cap inside StrategyPlanner instead of
  reading the preset — operator can't tune per profile
- Someone removes the _orig attributes — cap becomes irreversible
  (when restored, planner uses the last-capped value instead of original)
- Someone changes the agent to call apply_pdt_dte_cap only when
  pdt_restricted is True — when balance crosses back above $25K, the
  cap never gets removed
"""

from __future__ import annotations


def test_skill_33_preset_field_exists_with_default_14() -> None:
    """Skill 33 §3.1: PresetConfig must carry pdt_dte_cap with default 14."""
    from trading_agent.strategy_presets import PresetConfig
    field = PresetConfig.__dataclass_fields__.get("pdt_dte_cap")
    assert field is not None, (
        "Skill 33 §3.1: PresetConfig must define pdt_dte_cap. Removing "
        "this field makes the cap untunable per-preset."
    )
    assert field.default == 14, (
        f"Skill 33 §3.1: pdt_dte_cap default must be 14; got "
        f"{field.default}. The default is documented as the trade-off "
        f"point between theta runway and overnight drift risk on "
        f"sub-$25K accounts."
    )


def test_skill_33_apply_method_exists_with_documented_signature() -> None:
    """Skill 33 §3.2: StrategyPlanner.apply_pdt_dte_cap is the canonical
    per-cycle entry point. Read source directly so the test doesn't need
    scipy/pandas_market_calendars in the conformance environment."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "strategy.py").read_text(
        encoding="utf-8"
    )
    assert "def apply_pdt_dte_cap" in src, (
        "Skill 33 §3.2: StrategyPlanner must define apply_pdt_dte_cap. "
        "Renaming silently breaks the agent's per-cycle wire-up."
    )
    # Pin the signature shape — Optional[int] parameter
    assert "apply_pdt_dte_cap(self, cap: Optional[int])" in src, (
        "Skill 33 §3.2: apply_pdt_dte_cap must accept Optional[int]. "
        "Required so cap=None can restore the original DTE values "
        "when the account crosses back above $25K."
    )


def test_skill_33_original_dte_values_stored_separately() -> None:
    """Skill 33 §4: the planner must keep immutable _orig copies of each
    DTE value so cap=None can restore them. Without _orig storage, the
    restoration would copy already-capped values back into themselves."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "strategy.py").read_text(
        encoding="utf-8"
    )
    for orig in ("_dte_vertical_orig",
                 "_dte_iron_condor_orig",
                 "_dte_mean_reversion_orig"):
        assert orig in src, (
            f"Skill 33 §4: planner must store {orig} as the immutable "
            f"shadow of the preset's DTE value. Without it, "
            f"apply_pdt_dte_cap(None) can't reverse a prior cap."
        )


def test_skill_33_cap_uses_min_not_assign() -> None:
    """Skill 33 §2: cap formula must be min(original, cap) — a plain
    assignment would over-cap a strategy whose preset is already below
    the cap (e.g. Mean Reversion 14d gets clamped to 14d incorrectly
    becomes 14d still — but only if the cap is also 14)."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "strategy.py").read_text(
        encoding="utf-8"
    )
    assert "min(self._dte_vertical_orig, cap)" in src, (
        "Skill 33 §2: cap must use min() against the original, not a "
        "plain assignment. Otherwise a cap higher than the original "
        "would silently raise DTE — which the cap is supposed to prevent."
    )
    assert "min(self._dte_iron_condor_orig, cap)" in src
    assert "min(self._dte_mean_reversion_orig, cap)" in src


def test_skill_33_agent_calls_apply_in_run_cycle() -> None:
    """Skill 33 §3.3: agent must call apply_pdt_dte_cap once per cycle
    after computing pdt_restricted. Per-cycle invocation makes the cap
    reactive to intraday balance transitions."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "self.strategy_planner.apply_pdt_dte_cap(cap)" in src, (
        "Skill 33 §3.3: agent must call self.strategy_planner."
        "apply_pdt_dte_cap(cap) inside run_cycle. Construction-time "
        "application would miss intraday balance transitions."
    )
    # Critical: the cap value must be conditional on pdt_restricted.
    assert "self.preset.pdt_dte_cap if pdt_restricted else None" in src, (
        "Skill 33 §3.3: cap must be None when not PDT-restricted "
        "(i.e., account >= $25K). Otherwise the cap engages even at "
        "high equity, needlessly shortening DTE."
    )


def test_skill_33_cap_is_idempotent() -> None:
    """Skill 33 §4: re-applying the same cap is a no-op. Pin this by
    looking at the source pattern — the function unconditionally takes
    min(original, cap), so calling twice produces the same result."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "strategy.py").read_text(
        encoding="utf-8"
    )
    # The method body must NOT compute min against the CURRENT value
    # (which would compound under repeated calls). It must reference _orig.
    helper_start = src.find("def apply_pdt_dte_cap")
    helper_end = src.find("\n    # ---", helper_start + 1)
    body = src[helper_start:helper_end if helper_end > 0 else None]
    # Check there's no `min(self._dte_vertical, cap)` (without _orig) —
    # that would compound caps under repeated calls.
    forbidden = "min(self._dte_vertical, cap)"
    assert forbidden not in body, (
        f"Skill 33 §4: apply_pdt_dte_cap body must NOT use {forbidden!r}. "
        f"Capping against the current value (instead of _orig) compounds "
        f"under repeated calls — three calls with the same cap end up "
        f"with three different DTE values."
    )
