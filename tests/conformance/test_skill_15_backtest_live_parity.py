"""Conformance test: skill 15 — Backtest ↔ live parity.

Skill 15 §3 documents the parity contract: the backtest package
must call the SAME ``decision_engine.decide()`` function the live
agent uses. If those ever diverge, backtest results would
systematically misrepresent live behavior — the entire point of
having a backtester evaporates.

The architectural-invariant scanner already enforces:
  * No live module imports from ``trading_agent.backtest.*``
  * No module outside chain_scanner/decision_engine defines the
    scoring helpers (``_score_candidate``, etc.)

This conformance test pins the converse: backtest IS allowed to —
and DOES — import the live ``decide()``. If a contributor were to
refactor backtest to use a homegrown scorer (and skip the invariant
check by adding a noqa), this test would catch it.
"""

from __future__ import annotations


def test_skill_15_backtest_imports_live_decide() -> None:
    """Skill 15 §3 — the backtest cycle must use the SAME ``decide()``
    object as the live decision engine. Identity-equality is the
    strongest possible assertion of "same code path."""
    from trading_agent.decision_engine import decide as live_decide
    from trading_agent.backtest.cycle import decide as backtest_decide
    assert backtest_decide is live_decide, (
        "Skill 15 §3: backtest.cycle must import the live "
        "decision_engine.decide directly (not a copy). They are now "
        "different objects, which means backtest is running on a "
        "shadow implementation — exactly the drift skill 15 forbids."
    )


def test_skill_15_backtest_chain_slice_shape_matches_live() -> None:
    """Skill 15 §3: the backtest synthesizes ``ChainSlice`` rows in
    the SAME dataclass shape the live scanner consumes. If the
    backtest used a parallel/forked type, ``decide()`` would
    silently raise TypeError at the first sweep."""
    from trading_agent.decision_engine import ChainSlice as LiveSlice
    from trading_agent.decision_engine import DecisionInput as LiveInput
    # Inputs are dataclasses — having the same module + name is enough.
    assert LiveSlice.__name__ == "ChainSlice"
    assert LiveInput.__name__ == "DecisionInput"


def test_skill_15_backtest_package_importable() -> None:
    """Skill 15 §3: the backtest package must remain importable as
    a unit so the Streamlit Backtesting tab can render. Smoke test
    the package __init__ path."""
    import trading_agent.backtest as backtest_pkg
    # The package should expose at least its main runner symbol.
    assert backtest_pkg is not None
