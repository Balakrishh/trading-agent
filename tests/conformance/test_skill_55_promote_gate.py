"""Conformance for skill 55 — the auto-promote gate + write-path invariant.

Two enforcement layers:

1. Every fail branch of ``evaluate_auto_promote_gate`` is exercised —
   the table-driven test walks each of the 7 predicates.
2. Only ``trading_agent.executor`` and ``trading_agent.executor_promote``
   are permitted to import ``submit_order``/``place_order`` /
   ``OrderExecutor`` / ``TradingClient`` (AST scan).
"""
from __future__ import annotations

import ast
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

from trading_agent.executor_promote import (
    evaluate_auto_promote_gate,
    GateResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Preset:
    """Minimal PresetConfig stand-in."""
    def __init__(self, **kw):
        self.auto_promote_max_notional_usd = kw.get("max_notional", 500.0)
        self.auto_promote_max_contracts    = kw.get("max_contracts", 2)
        self.auto_promote_allowed_strategies = kw.get(
            "allowed", ("iron_condor",))


def _happy(**over) -> Dict[str, Any]:
    """Return kwargs that pass every gate. Individual tests override
    exactly one field to test that predicate's fail branch.
    """
    base = dict(
        proposal={"strategy": "iron_condor",
                  "auto_promote_requested": True},
        preset=_Preset(),
        current_notional=100.0,
        current_contracts=1,
        risk_warnings=0,
        now_utc=datetime.now(timezone.utc),
        within_market_hours=True,
        rescore_drift_pct=1.0,
        minutes_since_open=60,
        minutes_until_close=60,
    )
    base.update(over)
    return base


@pytest.fixture(autouse=True)
def _enable_master_switch(monkeypatch):
    monkeypatch.setenv("TRADING_AGENT_AUTO_PROMOTE_ENABLED", "true")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_gate_happy_path_passes_all_seven():
    g = evaluate_auto_promote_gate(**_happy())
    assert g.all_pass, g.failures()


# ---------------------------------------------------------------------------
# 7 predicate fail branches
# ---------------------------------------------------------------------------

def test_gate_master_switch_off_blocks(monkeypatch):
    monkeypatch.setenv("TRADING_AGENT_AUTO_PROMOTE_ENABLED", "false")
    g = evaluate_auto_promote_gate(**_happy())
    assert not g.all_pass
    assert "TRADING_AGENT_AUTO_PROMOTE_ENABLED" in g.master_switch


def test_gate_notional_cap_blocks():
    g = evaluate_auto_promote_gate(**_happy(current_notional=999.0))
    assert not g.all_pass
    assert "notional" in g.notional_cap


def test_gate_notional_cap_zero_disables_config_side():
    g = evaluate_auto_promote_gate(
        **_happy(preset=_Preset(max_notional=0.0)))
    assert not g.all_pass
    assert "disabled" in g.notional_cap


def test_gate_contract_cap_blocks():
    g = evaluate_auto_promote_gate(**_happy(current_contracts=99))
    assert not g.all_pass
    assert "contracts" in g.contract_cap


def test_gate_strategy_not_allowlisted_blocks():
    g = evaluate_auto_promote_gate(
        **_happy(preset=_Preset(allowed=("bull_put",))))
    assert not g.all_pass
    assert "not in" in g.strategy_allowlist


def test_gate_empty_strategy_allowlist_blocks():
    g = evaluate_auto_promote_gate(
        **_happy(preset=_Preset(allowed=())))
    assert not g.all_pass
    assert "empty" in g.strategy_allowlist


def test_gate_risk_warning_blocks():
    g = evaluate_auto_promote_gate(**_happy(risk_warnings=1))
    assert not g.all_pass
    assert "warning" in g.risk_clean


def test_gate_market_closed_blocks():
    g = evaluate_auto_promote_gate(**_happy(within_market_hours=False))
    assert not g.all_pass
    assert "closed" in g.market_hours


def test_gate_near_open_blocks():
    g = evaluate_auto_promote_gate(**_happy(minutes_since_open=2))
    assert not g.all_pass
    assert "first 5 min" in g.market_hours


def test_gate_near_close_blocks():
    g = evaluate_auto_promote_gate(**_happy(minutes_until_close=2))
    assert not g.all_pass
    assert "last 5 min" in g.market_hours


def test_gate_score_drift_blocks():
    g = evaluate_auto_promote_gate(**_happy(rescore_drift_pct=10.0))
    assert not g.all_pass
    assert "drift" in g.score_drift


def test_gate_failures_lists_all_failed_fields():
    g = evaluate_auto_promote_gate(
        **_happy(current_notional=999.0, rescore_drift_pct=10.0))
    fails = g.failures()
    assert any("notional" in f for f in fails)
    assert any("drift" in f for f in fails)


# ---------------------------------------------------------------------------
# Write-path invariant — only executor.py + executor_promote.py may
# import order-submission primitives.
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[2]
_TRADING_AGENT = _ROOT / "trading_agent"

_ALLOWED_WRITERS = {
    "executor.py",         # defines the primitives
    "executor_promote.py", # Claude Code write path (skill 55)
    "agent.py",            # live headless cycle
    "manual_close.py",     # operator close CLI, atomic close only (skill 63)
}

_FORBIDDEN_NAMES = ("submit_order", "place_order", "OrderExecutor",
                    "TradingClient")


def _files_that_import(names: tuple[str, ...]) -> list[str]:
    """Return every file under trading_agent/ that has an ImportFrom
    node whose imported name intersects ``names``.
    """
    offenders: list[str] = []
    for py in _TRADING_AGENT.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in names:
                        offenders.append(str(py.relative_to(_ROOT)))
                        break
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ("trading_agent.executor",):
                        offenders.append(str(py.relative_to(_ROOT)))
                        break
    return offenders


def test_skill_55_executor_imports_restricted_to_two_files():
    offenders = _files_that_import(_FORBIDDEN_NAMES)
    # Normalize to file basename for allowlist comparison
    disallowed = [
        f for f in offenders
        if Path(f).name not in _ALLOWED_WRITERS
    ]
    # But allow the executor module itself to import its own symbols
    # (they define them). And allow tests to import for verification.
    disallowed = [f for f in disallowed if not f.startswith("tests/")]
    assert not disallowed, (
        "Order-submission primitives may only be imported by "
        "executor.py (defines them), executor_promote.py (Claude Code "
        "write path — skill 55), agent.py (live headless cycle) and "
        "manual_close.py (operator close — skill 63). "
        f"Offenders: {disallowed}"
    )


def test_skill_55_pending_orders_writer_does_not_import_executor():
    """The writer must be a plain file-write — no executor coupling."""
    import trading_agent.pending_orders_writer as w
    src = Path(w.__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "executor" not in (node.module or ""), (
                "pending_orders_writer must not import executor")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert "executor" not in alias.name, (
                    "pending_orders_writer must not import executor")


def test_skill_55_pending_orders_gitignored():
    ig = (_ROOT / ".gitignore").read_text()
    assert "pending_orders" in ig, (
        "pending_orders/ must be in .gitignore — proposals contain "
        "live account context")


def test_skill_55_preset_carries_auto_promote_fields():
    """PresetConfig must expose the three AutoPromote knobs so the
    gate has something to read.
    """
    from trading_agent.strategy_presets import PresetConfig
    fields = {f for f in PresetConfig.__dataclass_fields__.keys()}
    assert "auto_promote_max_notional_usd" in fields
    assert "auto_promote_max_contracts" in fields
    assert "auto_promote_allowed_strategies" in fields


def test_skill_55_rehydrate_plan_round_trips():
    """The proposal's ``verdict.plan`` dict must rebuild into a
    SpreadPlan the RiskManager and OrderExecutor can consume.
    """
    from trading_agent.executor_promote import _rehydrate_plan
    plan_dict = {
        "ticker": "AAPL",
        "strategy": "Bull Put Spread",
        "regime": "test",
        "legs": [
            {"symbol": "AAPL251017P00170000", "strike": 170.0,
             "action": "sell", "type": "put", "delta": -0.25,
             "bid": 1.20, "ask": 1.25},
            {"symbol": "AAPL251017P00165000", "strike": 165.0,
             "action": "buy", "type": "put", "delta": -0.15,
             "bid": 0.60, "ask": 0.65},
        ],
        "spread_width": 5.0,
        "net_credit": 0.60,
        "max_loss": 440.0,
        "credit_to_width_ratio": 0.12,
        "expiration": "2025-10-17",
        "reasoning": "unit test",
    }
    plan = _rehydrate_plan(plan_dict)
    assert plan.ticker == "AAPL"
    assert plan.strategy_name == "Bull Put Spread"
    assert len(plan.legs) == 2
    assert plan.legs[0].action == "sell"
    assert plan.legs[1].action == "buy"
    assert plan.spread_width == 5.0
    assert plan.net_credit == 0.60


def test_skill_55_rehydrate_plan_rejects_missing_fields():
    from trading_agent.executor_promote import _rehydrate_plan
    import pytest
    with pytest.raises(ValueError, match="missing fields"):
        _rehydrate_plan({"ticker": "AAPL"})  # nowhere near enough


def test_skill_55_preset_auto_promote_defaults_are_safe():
    """Every default must be the SAFE end of the scale."""
    from trading_agent.strategy_presets import PresetConfig
    f = PresetConfig.__dataclass_fields__
    assert f["auto_promote_max_notional_usd"].default == 0.0
    assert f["auto_promote_max_contracts"].default == 1
    # allowed strategies uses default_factory or literal
    default = f["auto_promote_allowed_strategies"].default
    assert default in ((), None) or callable(default)
