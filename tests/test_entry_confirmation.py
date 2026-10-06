"""2026-10-05 — wait N cycles before opening; entry window; hourly limit."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from trading_agent.entry_confirmation import (
    EntryConfirmations, before_entry_window, entries_in_last_hour, signature,
)

T0 = datetime(2026, 10, 6, 15, 0, tzinfo=timezone.utc)          # 11:00 ET


def plan(strategy="Put Debit Spread", exp="2026-11-06", far=""):
    return SimpleNamespace(strategy_name=strategy, expiration=exp, far_expiration=far)


def cycle(path, required, observations, at):
    ec = EntryConfirmations(required, path=path, now=at).begin()
    out = {t: ec.observe(t, p) for t, p in observations.items()}
    ec.save()
    return out, ec


def test_confirms_after_three_consecutive_cycles(tmp_path):
    fp = tmp_path / "c.json"
    counts = []
    for i in range(3):
        out, _ = cycle(fp, 3, {"IWM": plan()}, T0 + timedelta(seconds=75 * i))
        counts.append((out["IWM"].count, out["IWM"].confirmed))
    assert counts == [(1, False), (2, False), (3, True)]


def test_regime_flicker_resets(tmp_path):
    """IWM 2026-10-05: put debit one cycle, nothing the next → start over."""
    fp = tmp_path / "c.json"
    cycle(fp, 3, {"IWM": plan()}, T0)
    cycle(fp, 3, {}, T0 + timedelta(seconds=75))                # no approved plan
    out, _ = cycle(fp, 3, {"IWM": plan()}, T0 + timedelta(seconds=150))
    assert out["IWM"].count == 1


def test_changed_strategy_or_expiry_resets(tmp_path):
    fp = tmp_path / "c.json"
    cycle(fp, 3, {"QQQ": plan("Call Debit Spread")}, T0)
    out, _ = cycle(fp, 3, {"QQQ": plan("Call Debit Spread", "2026-11-13")}, T0 + timedelta(seconds=75))
    assert out["QQQ"].count == 1
    assert signature(plan("Calendar Spread", "2026-10-30", "2026-11-20")) == \
        "Calendar Spread|2026-10-30|2026-11-20"


def test_gap_longer_than_a_cycle_resets(tmp_path):
    fp = tmp_path / "c.json"
    cycle(fp, 3, {"SPY": plan()}, T0)
    out, _ = cycle(fp, 3, {"SPY": plan()}, T0 + timedelta(minutes=10))
    assert out["SPY"].count == 1


def test_consumed_starts_over(tmp_path):
    fp = tmp_path / "c.json"
    for i in range(2):
        cycle(fp, 2, {"GLD": plan()}, T0 + timedelta(seconds=75 * i))
    ec = EntryConfirmations(2, path=fp, now=T0 + timedelta(seconds=150)).begin()
    assert ec.observe("GLD", plan()).confirmed
    ec.consumed("GLD")
    ec.save()
    out, _ = cycle(fp, 2, {"GLD": plan()}, T0 + timedelta(seconds=225))
    assert out["GLD"].count == 1


def test_required_one_is_legacy_immediate(tmp_path):
    out, _ = cycle(tmp_path / "c.json", 1, {"SPY": plan()}, T0)
    assert out["SPY"].confirmed


@pytest.mark.parametrize("utc,blocked", [
    (datetime(2026, 10, 6, 13, 40, tzinfo=timezone.utc), True),   # 09:40 ET
    (datetime(2026, 10, 6, 13, 45, tzinfo=timezone.utc), False),  # 09:45 ET
])
def test_entry_window(utc, blocked):
    assert before_entry_window(utc, "09:45") is blocked
    assert before_entry_window(utc, "bad") is False


def test_entries_in_last_hour():
    times = [T0 - timedelta(minutes=m) for m in (5, 59, 61, 120)]
    assert entries_in_last_hour(times, T0) == 2


def _agent(preset_kw, tmp_path):
    from trading_agent.agent import TradingAgent
    from trading_agent.strategy_presets import PRESETS
    from dataclasses import replace
    a = TradingAgent.__new__(TradingAgent)
    a.preset = replace(PRESETS["balanced"], **preset_kw)
    rows = []
    a.journal_kb = SimpleNamespace(log_signal=lambda **kw: rows.append(kw), jsonl_path=None)
    a._cached_price = lambda t: 1.0
    return a, rows


def test_agent_gates(tmp_path, monkeypatch):
    import trading_agent.agent as agent_mod
    a, rows = _agent({"entry_confirm_cycles": 2, "no_entry_before_et": "",
                      "max_new_entries_per_hour": 1}, tmp_path)
    a._begin_entry_gates()
    assert a._entry_confirmation_block("IWM", plan()) == "entry_confirming (1/2)"
    assert a._entry_rate_gate("IWM") is None
    a._entries_last_hour = 1
    skip = a._entry_rate_gate("QQQ")
    assert skip["reason"] == "Entry rate limit" and rows[-1]["action"] == "skipped_entry_rate"
    a._entry_confirm.save()
    a._begin_entry_gates()
    assert a._entry_confirmation_block("IWM", plan()) is None          # 2/2 → go
