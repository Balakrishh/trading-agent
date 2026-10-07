"""2026-10-07 — kill switch + drawdown governor (backlog §9, skill 62)."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from trading_agent import trading_halt as th
from trading_agent.strategy_presets import PRESETS

MON = datetime(2026, 10, 5, 14, 0, tzinfo=timezone.utc)      # 10:00 ET Monday
TUE = datetime(2026, 10, 6, 14, 0, tzinfo=timezone.utc)


def test_baselines_roll_by_day_and_week():
    s, why = th.govern(th.HaltState(), 30_000, MON, 0.02, 0.05)
    assert (s.day, s.day_start_equity, s.week_start_equity, why) == ("2026-10-05", 30_000, 30_000, None)
    s, _ = th.govern(s, 30_500, TUE, 0.02, 0.05)
    assert s.day_start_equity == 30_500 and s.week_start_equity == 30_000   # same ISO week


def test_daily_breach_pauses_and_needs_a_human():
    s, _ = th.govern(th.HaltState(), 30_000, MON, 0.02, 0.05)
    s, why = th.govern(s, 29_390, MON, 0.02, 0.05)                         # −2.03 %
    assert s.paused and s.set_by == "governor" and why.startswith("daily loss 2.0%")
    s, why = th.govern(s, 30_200, TUE, 0.02, 0.05)                         # recovered next day
    assert s.paused and why is None                                         # still paused


def test_weekly_breach():
    s, _ = th.govern(th.HaltState(), 30_000, MON, 0.02, 0.05)
    s, _ = th.govern(s, 29_000, TUE, 0.02, 0.05)                            # new day baseline 29,000
    s, why = th.govern(s, 28_480, TUE, 0.02, 0.05)                          # −1.8 % day, −5.07 % week
    assert s.paused and why.startswith("weekly loss 5.1%")


def test_zero_limits_disable_governor():
    s, _ = th.govern(th.HaltState(), 30_000, MON, 0.0, 0.0)
    s, why = th.govern(s, 10_000, MON, 0.0, 0.0)
    assert not s.paused and why is None


def test_cli_pause_status_resume(capsys, monkeypatch):
    rows = []
    monkeypatch.setattr(th, "_journal", lambda action, st: rows.append((action, st.reason)))
    assert th.main(["pause", "--reason", "FOMC day"]) == 0
    st = th.load()
    assert st.paused and st.set_by == "operator" and st.reason == "FOMC day"
    assert th.main(["status"]) == 0 and '"paused": true' in capsys.readouterr().out
    assert th.main(["resume"]) == 0
    assert not th.load().paused
    assert [a for a, _ in rows] == ["trading_halt_set", "trading_halt_cleared"]


def _agent(**preset_kw):
    from trading_agent.agent import TradingAgent
    a = TradingAgent.__new__(TradingAgent)
    a.preset = replace(PRESETS["balanced"], **preset_kw)
    rows, alerts = [], []
    a.journal_kb = SimpleNamespace(log_signal=lambda **kw: rows.append(kw))
    a.telegram = SimpleNamespace(notify_trading_halt=lambda reason: alerts.append(reason))
    return a, rows, alerts


def test_agent_skips_stage2_while_paused_and_alerts_once():
    a, rows, alerts = _agent()
    assert a._trading_halt_result(30_000, {}) is None                       # sets the baseline
    res = a._trading_halt_result(29_300, {"positions": []})                 # −2.3 % → governor
    assert res["order_summary"]["skipped_reason"] == "trading_halt" and res["new_trades"] == []
    assert [r["action"] for r in rows] == ["trading_halt_set"] and len(alerts) == 1
    assert a._trading_halt_result(29_900, {}) is not None                   # still paused
    assert len(rows) == 1 and len(alerts) == 1                              # no repeat alert


def test_operator_pause_respected_and_resume_reopens(monkeypatch):
    monkeypatch.setattr(th, "_journal", lambda *a: None)
    a, _, _ = _agent()
    th.main(["pause", "--reason", "manual"])
    assert a._trading_halt_result(30_000, {}) is not None
    th.main(["resume"])
    assert a._trading_halt_result(30_000, {}) is None


def test_broken_state_file_never_blocks():
    th.STATE_PATH.write_text("{not json")
    a, _, _ = _agent()
    assert a._trading_halt_result(30_000, {}) is None


def test_mcp_tool_reports_state(monkeypatch):
    monkeypatch.setattr(th, "_journal", lambda *a: None)
    th.main(["pause", "--reason", "x"])
    from trading_agent.mcp.tools.market import get_trading_halt
    out = get_trading_halt()
    assert out["paused"] is True and out["reason"] == "x"
