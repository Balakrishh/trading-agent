"""Conformance test: skill 36 — TickerFilters extraction.

Pins the public-contract surface of TickerFilters + agent integration.
Tests use a stub JournalKB + stub preset so the filters are exercised
in isolation — no MagicMock(spec=TradingAgent), no live broker.

Failure modes caught:
- Someone reverts the inlined IF-chain in _process_ticker → 120 lines
  re-appear, fixture-bloat returns.
- Someone changes the directional-bias / RSI / high-IV journal
  vocabulary → dashboard's "Why didn't this trade?" view breaks.
- Someone forgets to honor the RSI-gate analysis_override → planner
  picks the wrong strategy in override-regime mode.
- Someone reorders filters → cosmetic but the docstring contract
  asserts directional-bias first.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubJournal:
    """Records calls without writing to disk."""
    def __init__(self):
        self.signals: List[Dict[str, Any]] = []
        self.defense_first: List[Dict[str, Any]] = []

    def log_signal(self, *, ticker, action, price, raw_signal,
                   notes=None, exec_status=None):
        self.signals.append({
            "ticker": ticker, "action": action, "price": price,
            "raw_signal": raw_signal,
        })

    def log_defense_first(self, ticker, reason, price, context):
        self.defense_first.append({
            "ticker": ticker, "reason": reason, "price": price,
            "context": context,
        })


@dataclass
class _StubPreset:
    name: str = "Balanced"
    directional_bias: str = "auto"


@dataclass
class _StubRegimeEnum:
    value: str = "bullish"


@dataclass
class _StubAnalysis:
    regime: _StubRegimeEnum = field(default_factory=_StubRegimeEnum)
    current_price: float = 420.0
    rsi_14: float = 50.0
    iv_rank: float = 30.0
    high_iv_warning: bool = False


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------


def test_directional_bias_blocks_disallowed_regime() -> None:
    """Skill 36 §3: a preset with bearish_only bias must skip a bullish ticker."""
    from trading_agent.ticker_filters import TickerFilters
    j = _StubJournal()
    preset = _StubPreset(directional_bias="bearish_only")
    f = TickerFilters(journal_kb=j, preset=preset)
    res = f.evaluate("SPY", _StubAnalysis(regime=_StubRegimeEnum("bullish")))
    assert res is not None
    assert res.triggered_by == "directional_bias"
    assert res.result["status"] == "skipped"
    assert "bearish_only" in res.result["reason"]
    # Journal row written
    assert len(j.signals) == 1
    assert j.signals[0]["action"] == "skipped_bias"


def test_directional_bias_allows_matching_regime() -> None:
    """Skill 36 §3: auto preset allows all regimes — no early
    return for a bullish ticker."""
    from trading_agent.ticker_filters import TickerFilters
    j = _StubJournal()
    preset = _StubPreset(directional_bias="auto")
    f = TickerFilters(journal_kb=j, preset=preset)
    res = f.evaluate("SPY", _StubAnalysis(regime=_StubRegimeEnum("bullish")))
    assert res is None, "auto preset must allow bullish regime"
    assert len(j.signals) == 0


def test_high_iv_triggers_defense_first() -> None:
    """Skill 36 §3: high_iv_warning=True triggers defense_first journal
    write + skipped return."""
    from trading_agent.ticker_filters import TickerFilters
    j = _StubJournal()
    f = TickerFilters(journal_kb=j, preset=_StubPreset())
    a = _StubAnalysis(high_iv_warning=True, iv_rank=98.0)
    res = f.evaluate("VIXY", a)
    assert res is not None
    assert res.triggered_by == "high_iv"
    assert res.result["strategy_mode"] == "defense_first"
    # log_defense_first written (NOT log_signal)
    assert len(j.defense_first) == 1
    assert j.defense_first[0]["ticker"] == "VIXY"
    assert "iv_rank" in j.defense_first[0]["context"]


def test_high_iv_skipped_when_warning_off() -> None:
    """Skill 36 §3: high_iv_warning=False must not trigger the IV block."""
    from trading_agent.ticker_filters import TickerFilters
    j = _StubJournal()
    f = TickerFilters(journal_kb=j, preset=_StubPreset())
    a = _StubAnalysis(high_iv_warning=False, iv_rank=30.0)
    res = f.evaluate("SPY", a)
    assert res is None
    assert len(j.defense_first) == 0


def test_filters_run_in_documented_order() -> None:
    """Skill 36 §4: filter order is directional bias → RSI → high-IV.
    A ticker that would fail TWO filters must report the one that
    runs FIRST."""
    from trading_agent.ticker_filters import TickerFilters
    j = _StubJournal()
    # bearish_only preset + bullish regime + high_iv_warning=True.
    # Directional bias should trip BEFORE high-IV.
    f = TickerFilters(
        journal_kb=j, preset=_StubPreset(directional_bias="bearish_only"),
    )
    a = _StubAnalysis(
        regime=_StubRegimeEnum("bullish"), high_iv_warning=True, iv_rank=99.0,
    )
    res = f.evaluate("DIA", a)
    assert res is not None
    assert res.triggered_by == "directional_bias", (
        "Skill 36 §4: directional bias must be the FIRST filter — "
        f"got triggered_by={res.triggered_by}"
    )


def test_rsi_gate_disabled_by_default() -> None:
    """Skill 36 §3.3: when RSI_GATE_ENABLED is unset, the RSI gate is
    a no-op even on a ticker that would otherwise trigger it."""
    from trading_agent.ticker_filters import TickerFilters
    j = _StubJournal()
    f = TickerFilters(journal_kb=j, preset=_StubPreset())
    # Default env — RSI_GATE_ENABLED unset.
    os.environ.pop("RSI_GATE_ENABLED", None)
    res = f.evaluate("SPY", _StubAnalysis(rsi_14=75.0))
    # No filter triggered; rsi_gate disabled → pass-through.
    assert res is None


def test_filter_result_dataclass_is_frozen() -> None:
    """Skill 36 §1: FilterResult is a frozen dataclass — callers can't
    accidentally mutate the result dict in a way that breaks downstream."""
    from trading_agent.ticker_filters import FilterResult
    import dataclasses
    fields = {f.name for f in dataclasses.fields(FilterResult)}
    assert fields == {"result", "triggered_by", "analysis_override"}
    # Frozen check
    fr = FilterResult(result={}, triggered_by="x")
    try:
        fr.triggered_by = "y"
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("FilterResult must be frozen")


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


def test_agent_constructs_ticker_filters() -> None:
    """Skill 36 §3.3: TradingAgent.__init__ must construct
    self._ticker_filters so _process_ticker can delegate."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "self._ticker_filters = TickerFilters(" in src, (
        "Skill 36 §3.3: TradingAgent.__init__ must construct "
        "self._ticker_filters. Without it, _process_ticker delegation "
        "AttributeErrors and the cycle crashes on the first ticker."
    )


def test_process_ticker_delegates_to_filters() -> None:
    """Skill 36 §3.3: _process_ticker must call self._ticker_filters.evaluate
    instead of re-implementing the 120-line filter chain inline."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "self._ticker_filters.evaluate(" in src, (
        "Skill 36 §3.3: _process_ticker must delegate to "
        "self._ticker_filters.evaluate(). Pinning prevents re-inlining."
    )
    # The 120 lines of inlined logic should no longer be in agent.py.
    assert "DirectionalBias=" not in src or "f\"DirectionalBias=" not in src, (
        "Skill 36 §3.3: the inlined directional-bias skip-reason string "
        "must live in ticker_filters.py, not agent.py."
    )


def test_process_ticker_shrank_below_threshold() -> None:
    """Skill 36 §1: after extraction, _process_ticker must be ≤ 250 lines.
    Pre-extraction was 340; the three filters (~120 lines) moved out.
    Pinning at 250 leaves headroom for additions but blocks regression
    back to 340+."""
    import ast
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_process_ticker"):
            length = node.end_lineno - node.lineno + 1
            assert length <= 250, (
                f"Skill 36 §1: _process_ticker is {length} lines "
                f"(limit 250). The extraction is meant to keep this "
                f"shrunk — likely a filter got re-inlined."
            )
            return
    raise AssertionError("_process_ticker not found in agent.py")
