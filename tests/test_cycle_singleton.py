"""
test_cycle_singleton.py — pin down the cycle-singleton lock.

2026-05-06 regression: concurrent Streamlit watchdog + auto-refresh
+ scheduled-timer triggers caused 9 cycles to start within 38
seconds of each other; two raced past the dedup gate to submit
duplicate SPY Bear Call spreads (orders 462b61ba and 9d271dc7,
identical strikes, 2 seconds apart).  The fix in ``agent.py:run_cycle``
acquires ``self._cycle_lock`` non-blockingly; the second concurrent
caller gets a fast-path skip with status ``skipped_concurrent``.

These tests don't construct a real ``TradingAgent`` (it pulls in
config, broker, LLM, etc).  They use a stub object that mounts
the actual ``run_cycle`` method via ``__get__`` so the lock semantics
are exercised against the production code path with no external deps.
"""

from __future__ import annotations

import threading
import time
import types
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest


def _build_stub_agent():
    """
    Produce a minimal object on which ``TradingAgent.run_cycle`` can
    be invoked.  We only need the attributes ``run_cycle`` reads
    before delegating to ``_run_cycle_with_timeout_guard`` — the lock
    + the inner method — plus a logger that the production code
    references at module scope.
    """
    from trading_agent.agent import TradingAgent

    agent = MagicMock(spec=TradingAgent)
    agent._cycle_lock = threading.Lock()

    # Bind the real run_cycle method to the stub so we exercise the
    # actual lock-and-skip logic.
    agent.run_cycle = types.MethodType(TradingAgent.run_cycle, agent)
    return agent


def test_second_concurrent_cycle_is_skipped():
    """Two threads racing into run_cycle: first runs, second skips."""
    agent = _build_stub_agent()

    # Slow inner method so the second caller hits the lock.
    inner_calls = []

    def slow_inner(self):
        inner_calls.append(time.monotonic())
        time.sleep(0.2)
        return {"status": "ok", "submitted": ["SPY"]}

    agent._run_cycle_with_timeout_guard = types.MethodType(slow_inner, agent)

    results: Dict[str, dict] = {}

    def call(label: str):
        results[label] = agent.run_cycle()

    t1 = threading.Thread(target=call, args=("first",))
    t2 = threading.Thread(target=call, args=("second",))
    t1.start()
    time.sleep(0.05)            # ensure t1 acquires first
    t2.start()
    t1.join()
    t2.join()

    # Exactly one cycle ran; the second was skipped.
    assert len(inner_calls) == 1, (
        f"Expected exactly one inner cycle, got {len(inner_calls)} "
        "— singleton lock didn't fire."
    )
    statuses = sorted(r["status"] for r in results.values())
    assert statuses == ["ok", "skipped_concurrent"], (
        f"Expected one ok + one skipped_concurrent, got {statuses}"
    )


def test_sequential_cycles_both_run():
    """The lock must release after each cycle so subsequent runs proceed."""
    agent = _build_stub_agent()
    inner_calls = []

    def fast_inner(self):
        inner_calls.append(time.monotonic())
        return {"status": "ok"}

    agent._run_cycle_with_timeout_guard = types.MethodType(fast_inner, agent)

    r1 = agent.run_cycle()
    r2 = agent.run_cycle()
    r3 = agent.run_cycle()
    assert len(inner_calls) == 3
    assert r1["status"] == r2["status"] == r3["status"] == "ok"


def test_skipped_result_carries_diagnostic_metadata():
    """The skip result must include status + reason + timestamp so a
    Streamlit caller can render it as 'cycle skipped' in the UI."""
    agent = _build_stub_agent()
    agent._cycle_lock.acquire()      # simulate an in-flight cycle
    try:
        result = agent.run_cycle()
    finally:
        agent._cycle_lock.release()
    assert result["status"] == "skipped_concurrent"
    assert "in progress" in result["reason"].lower()
    assert "timestamp" in result
    # Timestamp is ISO 8601 with timezone — at minimum a non-empty string.
    assert isinstance(result["timestamp"], str) and result["timestamp"]


def test_cycle_lock_is_released_on_inner_exception():
    """If the inner cycle raises, the lock must still release so the
    next trigger can run.  Without this, one bug bricks all future cycles."""
    agent = _build_stub_agent()
    inner_calls = []

    def raising_inner(self):
        inner_calls.append(time.monotonic())
        raise RuntimeError("simulated cycle crash")

    agent._run_cycle_with_timeout_guard = types.MethodType(raising_inner, agent)

    with pytest.raises(RuntimeError, match="simulated cycle crash"):
        agent.run_cycle()

    # Lock must be releasable now — if it weren't, this would block.
    assert agent._cycle_lock.acquire(blocking=False), (
        "Cycle lock leaked after inner exception — future cycles would deadlock"
    )
    agent._cycle_lock.release()


def test_lock_is_non_reentrant():
    """``threading.Lock`` (not RLock).  A cycle that internally calls
    run_cycle must get a skip — not a deadlock or a recursion.

    This guards against a future refactor where someone wires a sub-call
    to ``self.run_cycle`` (e.g. for a 'manual close' button) and creates
    an unintentional re-entry path.
    """
    agent = _build_stub_agent()
    inner_calls = []
    sub_results = []

    def reentering_inner(self):
        inner_calls.append(time.monotonic())
        # Try to call run_cycle from inside a cycle — must skip, not deadlock.
        sub_results.append(self.run_cycle())
        return {"status": "ok"}

    agent._run_cycle_with_timeout_guard = types.MethodType(reentering_inner, agent)
    outer = agent.run_cycle()
    assert outer["status"] == "ok"
    assert len(inner_calls) == 1
    assert sub_results[0]["status"] == "skipped_concurrent"
