"""Conformance for skill 56 — daily journal reviewer stays read-only,
LLM output is properly filtered, digest render is deterministic.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict


_ROOT = Path(__file__).resolve().parents[2]
_REVIEWER_FILES = (
    _ROOT / "trading_agent" / "daily_reviewer.py",
    _ROOT / "trading_agent" / "daily_reviewer_main.py",
    _ROOT / "trading_agent" / "apply_preset_update.py",
    _ROOT / "trading_agent" / "pending_preset_updates_writer.py",
)

_FORBIDDEN_IMPORTS = (
    "trading_agent.executor",
    "trading_agent.executor_promote",
    "alpaca.trading",
)
_FORBIDDEN_NAMES = (
    "submit_order",
    "place_order",
    "OrderExecutor",
    "TradingClient",
)


def test_skill_56_reviewer_never_imports_executor():
    """AST-walk every reviewer-side .py and reject any executor import."""
    offenders = []
    for py in _REVIEWER_FILES:
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    for bad in _FORBIDDEN_IMPORTS:
                        if a.name == bad or a.name.startswith(bad + "."):
                            offenders.append((py.name, f"import {a.name}"))
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for bad in _FORBIDDEN_IMPORTS:
                    if mod == bad or mod.startswith(bad + "."):
                        offenders.append((py.name, f"from {mod} import ..."))
                for a in node.names:
                    if a.name in _FORBIDDEN_NAMES:
                        offenders.append(
                            (py.name, f"from {mod} import {a.name}"))
    assert not offenders, (
        "Daily-reviewer files must never touch the executor. "
        f"Offenders: {offenders}"
    )


def test_skill_56_apply_stub_does_not_import_preset_save():
    """The Phase-A apply CLI must NOT import save_active_preset yet.
    Phase B will wire it behind the 3-predicate gate; landing them
    together would be an untested change.
    """
    src = (_ROOT / "trading_agent" / "apply_preset_update.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                assert a.name != "save_active_preset", (
                    "apply_preset_update.py must remain a stub in Phase A"
                )


def test_skill_56_parse_llm_output_filters_forbidden_fields():
    """A hallucinated ``max_risk_pct`` proposal must be dropped even if
    the LLM ignores the system prompt.
    """
    from trading_agent.daily_reviewer import _parse_llm_output
    out = _parse_llm_output({
        "observations": ["test"],
        "preset_proposal": {
            "field": "max_risk_pct",   # forbidden
            "current": 0.02,
            "proposed": 0.05,
            "reason": "let's yolo",
            "confidence": 0.9,
        },
        "digest_lines": [],
        "confidence": 0.5,
    })
    assert out.preset_proposal is None, (
        "forbidden field must not survive parsing")


def test_skill_56_parse_llm_output_accepts_allowed_field():
    from trading_agent.daily_reviewer import _parse_llm_output
    out = _parse_llm_output({
        "observations": ["test"],
        "preset_proposal": {
            "field": "profit_target_pct",
            "current": 0.50,
            "proposed": 0.55,
            "reason": "winners closed at 51%",
            "confidence": 0.72,
        },
        "digest_lines": ["line 1"],
        "confidence": 0.72,
    })
    assert out.preset_proposal is not None
    assert out.preset_proposal["field"] == "profit_target_pct"
    assert out.preset_proposal["proposed"] == 0.55


def test_skill_56_parse_llm_output_survives_malformed():
    from trading_agent.daily_reviewer import _parse_llm_output
    out = _parse_llm_output("not a dict at all")   # noqa: E501
    assert out.observations == []
    assert out.preset_proposal is None
    assert out.raw_llm_text                       # populated so operator can see


def test_skill_56_digest_render_is_deterministic():
    """Same input → same output. Snapshot-testable."""
    from trading_agent.daily_reviewer import (
        render_telegram_digest, ReviewContext, ReviewOutput,
    )
    ctx = ReviewContext(
        review_date="2026-09-28",
        opens=[{"ticker": "AAPL", "strategy": "bull_put", "credit": 0.60}],
        closes=[],
        reject_reasons=[["wide_spread", 4]],
        realized_pl=127.0,
        preset={"name": "custom"},
        watchlist=["SPY", "QQQ"],
        macro={"vix_zone": "normal"},
        recent_alerts=[],
        cold_start=False,
    )
    out = ReviewOutput(
        observations=["Bull-puts on tech hit target within 3 days."],
        preset_proposal={
            "field": "profit_target_pct",
            "current": 0.50,
            "proposed": 0.55,
            "reason": "winners closed at 51%",
            "confidence": 0.72,
        },
        watchlist_proposal=None,
        digest_lines=[],
        confidence=0.72,
    )
    a = render_telegram_digest(ctx, out, proposal_id="abc123")
    b = render_telegram_digest(ctx, out, proposal_id="abc123")
    assert a == b
    assert "Daily Review — 2026-09-28" in a
    assert "profit_target_pct" in a
    assert "abc123" in a


def test_skill_56_digest_cold_start_labeled():
    from trading_agent.daily_reviewer import (
        render_telegram_digest, ReviewContext, ReviewOutput,
    )
    ctx = ReviewContext(
        review_date="2026-09-28",
        opens=[], closes=[], reject_reasons=[], realized_pl=0.0,
        preset={}, watchlist=[], macro={}, recent_alerts=[],
        cold_start=True,
    )
    out = ReviewOutput([], None, None, [], 0.0, "")
    digest = render_telegram_digest(ctx, out)
    assert "cold start" in digest


def test_skill_56_preset_carries_auto_apply_fields():
    from trading_agent.strategy_presets import PresetConfig
    fields = {f for f in PresetConfig.__dataclass_fields__.keys()}
    assert "auto_apply_preset_updates_enabled" in fields
    assert "auto_apply_max_delta_change_pct" in fields
    assert "auto_apply_allowed_fields" in fields


def test_skill_56_preset_auto_apply_defaults_are_safe():
    from trading_agent.strategy_presets import PresetConfig
    f = PresetConfig.__dataclass_fields__
    assert f["auto_apply_preset_updates_enabled"].default is False
    assert f["auto_apply_max_delta_change_pct"].default == 0.0
    d = f["auto_apply_allowed_fields"].default
    assert d in ((), None) or callable(d)


def test_skill_56_pending_writer_round_trips(tmp_path, monkeypatch):
    """The writer + reader agree on a proposal's schema."""
    monkeypatch.setenv("TRADING_AGENT_PENDING_PRESET_UPDATES_DIR",
                       str(tmp_path))
    # Re-import to pick up the new env var
    import importlib
    from trading_agent import pending_preset_updates_writer as w
    importlib.reload(w)
    uid = w.write(
        review_date="2026-09-28",
        preset_snapshot={"profit_target_pct": 0.50},
        preset_diff={"profit_target_pct": 0.55},
        watchlist_diff={},
        observations=["test"],
        llm_reasoning="unit test",
        confidence=0.72,
    )
    got = w.read(uid)
    assert got is not None
    assert got["preset_diff"]["profit_target_pct"] == 0.55
    assert got["confidence"] == 0.72


def test_skill_56_gitignored():
    ig = (_ROOT / ".gitignore").read_text()
    assert "daily_reviews/" in ig
    assert "pending_preset_updates/" in ig
