"""Conformance for skill 57 — the agent supervisor stays a spawner,
sleep math is bounded, and the launch loop honors its safety knobs.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone, timedelta
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_SUPERVISOR = _ROOT / "trading_agent" / "agent_supervisor.py"

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


def test_skill_57_supervisor_never_imports_executor():
    """AST walker — the supervisor is a spawner, never a trader."""
    tree = ast.parse(_SUPERVISOR.read_text())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                for bad in _FORBIDDEN_IMPORTS:
                    if a.name == bad or a.name.startswith(bad + "."):
                        offenders.append(f"import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for bad in _FORBIDDEN_IMPORTS:
                if mod == bad or mod.startswith(bad + "."):
                    offenders.append(f"from {mod} import ...")
            for a in node.names:
                if a.name in _FORBIDDEN_NAMES:
                    offenders.append(f"from {mod} import {a.name}")
    assert not offenders, (
        "Supervisor must be a pure spawner. Offenders: "
        f"{offenders}"
    )


def test_skill_57_sleep_is_bounded():
    """seconds_until_next_open never returns a value outside its bounds."""
    from trading_agent.agent_supervisor import (
        seconds_until_next_open, _MIN_SLEEP_SEC, _MAX_SLEEP_SEC,
    )
    # Probe five sample moments across the week.
    tz = timezone(timedelta(hours=-5))    # rough US-ET; profile normalizes anyway
    for probe in (
        datetime(2026, 9, 28, 3, 0, tzinfo=tz),    # Sun 3 AM
        datetime(2026, 9, 29, 8, 30, tzinfo=tz),   # Tue 8:30 AM
        datetime(2026, 9, 29, 10, 0, tzinfo=tz),   # Tue 10 AM (in-hours)
        datetime(2026, 9, 29, 17, 0, tzinfo=tz),   # Tue 5 PM
        datetime(2026, 9, 30, 23, 0, tzinfo=tz),   # Wed 11 PM
    ):
        s = seconds_until_next_open(probe)
        assert _MIN_SLEEP_SEC <= s <= _MAX_SLEEP_SEC, (
            f"sleep {s}s at {probe} outside bounds "
            f"[{_MIN_SLEEP_SEC}, {_MAX_SLEEP_SEC}]"
        )


def test_skill_57_supervise_respects_max_iterations():
    """The loop must terminate when max_iterations is set — required
    for tests to run in bounded time.
    """
    from trading_agent import agent_supervisor as sup

    calls = {"agent": 0, "sleep": 0}

    def fake_run_agent_once(env=None):
        calls["agent"] += 1
        return 0    # graceful exit

    def fake_sleep(s):
        calls["sleep"] += 1

    orig = sup._run_agent_once
    sup._run_agent_once = fake_run_agent_once
    try:
        rc = sup.supervise(max_iterations=3, sleep_fn=fake_sleep)
    finally:
        sup._run_agent_once = orig

    assert rc == 0
    assert calls["agent"] == 3
    # After each graceful exit the supervisor sleeps once, but the
    # 4th iteration bails on the max_iterations check before sleeping.
    assert calls["sleep"] == 3


def test_skill_57_crash_triggers_backoff():
    """A non-zero exit should hit the crash-backoff branch, not the
    seconds_until_next_open branch.
    """
    from trading_agent import agent_supervisor as sup

    sleep_calls = []

    def fake_run_agent_once(env=None):
        return 1    # simulate crash

    def fake_sleep(s):
        sleep_calls.append(s)

    orig = sup._run_agent_once
    sup._run_agent_once = fake_run_agent_once
    try:
        sup.supervise(max_iterations=2, sleep_fn=fake_sleep)
    finally:
        sup._run_agent_once = orig

    assert sleep_calls == [sup._CRASH_BACKOFF_SEC,
                            sup._CRASH_BACKOFF_SEC], (
        f"expected two crash-backoff sleeps, got {sleep_calls}")


def test_skill_57_launchd_plist_wires_supervisor():
    """The plist must invoke agent_supervisor, not agent directly —
    otherwise launchd's KeepAlive would fight the agent's graceful
    exit and create a hot loop.
    """
    plist = (_ROOT / "ops" / "launchd"
             / "com.trading-agent.headless.plist").read_text()
    assert "trading_agent.agent_supervisor" in plist
    # KeepAlive is required for the belt-and-suspenders design.
    assert "<key>KeepAlive</key>" in plist
    assert "<true/>" in plist   # KeepAlive true
    # ThrottleInterval must be present so a supervisor-level crash
    # doesn't hot-loop under launchd.
    assert "<key>ThrottleInterval</key>" in plist


def test_skill_57_readonly_in_the_agent_supervisor_docstring():
    """A quick prose-safety check: the module docstring names the
    read-only invariant so future edits can't remove it silently.
    """
    src = _SUPERVISOR.read_text()
    tree = ast.parse(src)
    docstring = ast.get_docstring(tree) or ""
    assert "NEVER imports the executor" in docstring, (
        "supervisor module docstring must state the read-only invariant"
    )


def test_skill_57_in_session_sleep_is_configurable(monkeypatch):
    """The in-session sleep is the gap between cycles (≈75 s cycle by
    default, measured 2026-09-30); AGENT_CYCLE_SLEEP_SEC tunes it."""
    from trading_agent.agent_supervisor import _MIN_SLEEP_SEC, in_session_sleep_sec
    monkeypatch.delenv("AGENT_CYCLE_SLEEP_SEC", raising=False)
    assert in_session_sleep_sec() == _MIN_SLEEP_SEC == 60
    for raw, want in (("285", 285), ("5", 60), ("99999", 900), ("abc", 60)):
        monkeypatch.setenv("AGENT_CYCLE_SLEEP_SEC", raw)
        assert in_session_sleep_sec() == want


def test_skill_57_wall_clock_sleep_survives_system_sleep():
    """2026-10-05: time.sleep pauses while a Mac sleeps, so a 12 h wait
    overran the open. wall_clock_sleep re-checks the wall clock every slice:
    a jump (system sleep) ends the wait at the next slice."""
    from trading_agent.agent_supervisor import wall_clock_sleep
    now = [0.0]
    slept = []

    def fake_sleep(s):
        slept.append(s)
        now[0] += s + (50_000 if len(slept) == 2 else 0)   # lid closed 14 h during slice 2

    wall_clock_sleep(43_200, clock=lambda: now[0], sleep=fake_sleep, slice_sec=300)
    assert slept == [300, 300]          # woke right after the jump, not 12 h later
    slept.clear(); now[0] = 0.0
    wall_clock_sleep(650, clock=lambda: now[0], sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)))
    assert slept == [300, 300, 50]
    wall_clock_sleep(0, clock=lambda: 0.0, sleep=lambda s: (_ for _ in ()).throw(AssertionError))
