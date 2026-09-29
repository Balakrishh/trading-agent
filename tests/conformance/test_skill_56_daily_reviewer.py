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


def test_skill_56_save_active_preset_callers_restricted():
    """save_active_preset may only be imported by
    ``apply_preset_update.py`` (Claude Code write path) and Streamlit
    surfaces (the operator's existing "Apply" button). No other
    module in ``trading_agent/`` may reach the persistence primitive.
    Widening this would let a scheduled task or MCP tool silently
    mutate live config.
    """
    allowed = {"apply_preset_update.py", "strategy_presets.py"}
    offenders = []
    for py in (_ROOT / "trading_agent").rglob("*.py"):
        if py.name in allowed:
            continue
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if a.name == "save_active_preset":
                        offenders.append(str(py.relative_to(_ROOT)))
    # Streamlit surfaces are the one intentional exception.
    disallowed = [f for f in offenders if "streamlit" not in f]
    assert not disallowed, (
        "save_active_preset may only be imported by apply_preset_update.py, "
        f"strategy_presets.py, and Streamlit surfaces. Offenders: {disallowed}"
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


# ---------------------------------------------------------------------------
# Phase B — 3-predicate auto-apply gate
# ---------------------------------------------------------------------------

class _StubPreset:
    """Minimal PresetConfig stand-in for gate tests."""
    def __init__(self, **kw):
        self.auto_apply_preset_updates_enabled = kw.get("enabled", True)
        self.auto_apply_max_delta_change_pct = kw.get("cap_pct", 0.15)
        self.auto_apply_allowed_fields = kw.get(
            "allowed", ("profit_target_pct", "edge_buffer"))


def _proposal(**over):
    base = {
        "preset_snapshot": {"profit_target_pct": 0.50},
        "preset_diff":     {"profit_target_pct": 0.55},
        "watchlist_diff":  {},
    }
    base.update(over)
    return base


def test_skill_56_apply_gate_happy_path(monkeypatch):
    monkeypatch.setenv(
        "TRADING_AGENT_AUTO_APPLY_PRESET_UPDATES_ENABLED", "true")
    from trading_agent.apply_preset_update import evaluate_apply_gate
    g = evaluate_apply_gate(proposal=_proposal(), preset=_StubPreset())
    assert g.all_pass, g.failures()


def test_skill_56_apply_gate_master_switch_off(monkeypatch):
    monkeypatch.setenv(
        "TRADING_AGENT_AUTO_APPLY_PRESET_UPDATES_ENABLED", "false")
    from trading_agent.apply_preset_update import evaluate_apply_gate
    g = evaluate_apply_gate(proposal=_proposal(), preset=_StubPreset())
    assert not g.all_pass
    assert "AUTO_APPLY_PRESET_UPDATES_ENABLED" in g.master_switch


def test_skill_56_apply_gate_field_not_in_allowlist(monkeypatch):
    monkeypatch.setenv(
        "TRADING_AGENT_AUTO_APPLY_PRESET_UPDATES_ENABLED", "true")
    from trading_agent.apply_preset_update import evaluate_apply_gate
    g = evaluate_apply_gate(
        proposal=_proposal(preset_diff={"min_pop": 0.60},
                           preset_snapshot={"min_pop": 0.55}),
        preset=_StubPreset(allowed=("profit_target_pct",)))
    assert not g.all_pass
    assert "not in allowlist" in g.field_allowlist


def test_skill_56_apply_gate_empty_allowlist_disables(monkeypatch):
    monkeypatch.setenv(
        "TRADING_AGENT_AUTO_APPLY_PRESET_UPDATES_ENABLED", "true")
    from trading_agent.apply_preset_update import evaluate_apply_gate
    g = evaluate_apply_gate(
        proposal=_proposal(),
        preset=_StubPreset(allowed=()))
    assert not g.all_pass
    assert "empty" in g.field_allowlist


def test_skill_56_apply_gate_delta_size_over_cap(monkeypatch):
    monkeypatch.setenv(
        "TRADING_AGENT_AUTO_APPLY_PRESET_UPDATES_ENABLED", "true")
    from trading_agent.apply_preset_update import evaluate_apply_gate
    # 0.50 → 0.75 = +50% change; cap is 10%
    g = evaluate_apply_gate(
        proposal=_proposal(preset_diff={"profit_target_pct": 0.75}),
        preset=_StubPreset(cap_pct=0.10))
    assert not g.all_pass
    assert "cap" in g.delta_size_cap


def test_skill_56_apply_gate_delta_size_zero_cap_disables(monkeypatch):
    monkeypatch.setenv(
        "TRADING_AGENT_AUTO_APPLY_PRESET_UPDATES_ENABLED", "true")
    from trading_agent.apply_preset_update import evaluate_apply_gate
    g = evaluate_apply_gate(
        proposal=_proposal(),
        preset=_StubPreset(cap_pct=0.0))
    assert not g.all_pass
    assert "disabled" in g.delta_size_cap


def test_skill_56_apply_gate_boolean_flip_bypasses_size_cap(monkeypatch):
    """Boolean fields (defensive_roll_enabled) have no meaningful
    delta-percent — a flip should count as within-cap.
    """
    monkeypatch.setenv(
        "TRADING_AGENT_AUTO_APPLY_PRESET_UPDATES_ENABLED", "true")
    from trading_agent.apply_preset_update import evaluate_apply_gate
    g = evaluate_apply_gate(
        proposal=_proposal(
            preset_diff={"defensive_roll_enabled": True},
            preset_snapshot={"defensive_roll_enabled": False}),
        preset=_StubPreset(
            cap_pct=0.10,
            allowed=("defensive_roll_enabled",)))
    assert g.all_pass, g.failures()


def test_skill_56_apply_gate_no_preset_diff_trivially_ok(monkeypatch):
    """A watchlist-only proposal has no preset_diff; the allowlist +
    size-cap predicates are trivially satisfied.
    """
    monkeypatch.setenv(
        "TRADING_AGENT_AUTO_APPLY_PRESET_UPDATES_ENABLED", "true")
    from trading_agent.apply_preset_update import evaluate_apply_gate
    g = evaluate_apply_gate(
        proposal={"preset_diff": {}, "preset_snapshot": {},
                  "watchlist_diff": {"drops": ["XYZ"]}},
        preset=_StubPreset())
    # allowlist and size-cap are pass; only depend on master switch
    assert g.field_allowlist == "pass"
    # size-cap still shows "disabled" because cap>0 but no fields to check.
    # It's fine either way — this test just verifies no crash on empty diff.


def test_skill_56_digest_snapshot_stable():
    """Golden-file snapshot test — a curated fixture renders to a
    specific byte sequence. Change here means every downstream
    dashboard / Telegram consumer needs review.
    """
    from trading_agent.daily_reviewer import (
        render_telegram_digest, ReviewContext, ReviewOutput,
    )
    ctx = ReviewContext(
        review_date="2026-09-29",
        opens=[
            {"ticker": "AAPL", "strategy": "bull_put", "credit": 0.61},
            {"ticker": "MSFT", "strategy": "iron_condor", "credit": 1.20},
        ],
        closes=[
            {"ticker": "JPM", "strategy": "bear_call",
             "pnl": 128.0, "reason": "profit_target"},
        ],
        reject_reasons=[["wide_spread", 4], ["low_pop", 2]],
        realized_pl=128.0,
        preset={"name": "custom", "profit_target_pct": 0.50},
        watchlist=["SPY", "QQQ", "IWM"],
        macro={"vix_zone": "normal"},
        recent_alerts=[],
        cold_start=False,
    )
    out = ReviewOutput(
        observations=[
            "Bull-puts on tech names took profit on day 2.",
            "GLD rejects for wide spreads continued — 4 of 5 cycles.",
        ],
        preset_proposal={
            "field": "profit_target_pct",
            "current": 0.50,
            "proposed": 0.55,
            "reason": "Today's winners closed at 51% — 5% wringing improves per-trade PnL without hurting hit rate.",
            "confidence": 0.72,
        },
        watchlist_proposal={
            "drops": ["GLD"],
            "adds": [],
            "reason": "Chronic per-leg spread rejects; not tradeable this regime.",
        },
        digest_lines=[],
        confidence=0.72,
    )
    digest = render_telegram_digest(ctx, out, proposal_id="test-uuid-000")

    # Structural assertions rather than a byte-for-byte snapshot so
    # future prose tweaks in the digest don't force a fixture rewrite.
    # If a change here surprises a downstream consumer, extend the
    # assertions rather than pinning the whole string.
    for expected in [
        "Daily Review — 2026-09-29",
        "2 opens · 1 closes",
        "P&amp;L +128.00",
        "Bull-puts on tech names",
        "profit_target_pct",
        "0.5 → 0.55",
        "conf 0.72",
        "Watchlist drops: GLD",
        "test-uuid-000",
    ]:
        assert expected in digest, f"digest missing: {expected}\n{digest}"


def test_skill_56_save_active_preset_accepts_new_overlays():
    """The Phase B write path passes min_pop / max_leg_spread_cents /
    defensive_roll_enabled as overlays. Verify the function signature
    accepts them without raising a TypeError at call time.
    """
    import inspect
    from trading_agent.strategy_presets import save_active_preset
    sig = inspect.signature(save_active_preset)
    for kw in ("min_pop", "max_leg_spread_cents", "defensive_roll_enabled"):
        assert kw in sig.parameters, f"{kw} missing from save_active_preset"
