"""Stop hygiene (2026-10-16): real consecutive exit debounce, no votes in
the opening window, and a cool-down after a stop (skills 30, 37)."""
import json
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from trading_agent import agent as agent_mod
from trading_agent.agent import EXIT_DEBOUNCE_REQUIRED, TradingAgent
from trading_agent.daily_state import DailyStateStore, reset_exit_vote, tally_exit_vote
from trading_agent.journal_reader import JournalReader
from trading_agent.position_caps import (
    sessions_between, stop_cooldown_failure, trade_direction)
from trading_agent.position_monitor import ExitSignal
from trading_agent.strategy_presets import PRESETS, PresetConfig

STOP, HOLD = ExitSignal.STOP_LOSS, ExitSignal.HOLD


def _agent(tmp_path, monkeypatch, *, before_window=False):
    a = TradingAgent.__new__(TradingAgent)
    a.preset = SimpleNamespace(no_exit_before_et="09:45")
    a.daily_state = DailyStateStore(str(tmp_path))
    monkeypatch.setattr("trading_agent.entry_confirmation.before_entry_window",
                        lambda now, nb: before_window)
    return a


def _pos(sig, exp="2026-11-06", strategy="Call Debit Spread", tk="QQQ"):
    return SimpleNamespace(underlying=tk, expiration=exp, strategy_name=strategy,
                           exit_signal=sig)


# ── Debounce: a HOLD resets the count ─────────────────────────────────

def test_qqq_2026_10_14_replay_never_confirms(tmp_path, monkeypatch):
    """Stop touched at 09:32, 09:37, 09:57 with HOLD cycles between."""
    a = _agent(tmp_path, monkeypatch)
    seq = [STOP, HOLD, HOLD, HOLD, STOP, HOLD, HOLD, HOLD, STOP]
    assert not any(a._should_exit_spread(_pos(s)) for s in seq)


def test_three_consecutive_stops_confirm(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    out = [a._should_exit_spread(_pos(STOP)) for _ in range(EXIT_DEBOUNCE_REQUIRED)]
    assert out == [False] * (EXIT_DEBOUNCE_REQUIRED - 1) + [True]


def test_hold_between_restarts_the_count(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    for s in (STOP, STOP, HOLD, STOP, STOP):
        assert a._should_exit_spread(_pos(s)) is False
    assert a._should_exit_spread(_pos(STOP)) is True


def test_votes_are_per_position(tmp_path, monkeypatch):
    """Two laddered positions on one ticker keep separate counts."""
    a = _agent(tmp_path, monkeypatch)
    for _ in range(2):
        a._should_exit_spread(_pos(STOP, exp="2026-11-06"))
        a._should_exit_spread(_pos(HOLD, exp="2026-11-20"))
    assert a._should_exit_spread(_pos(STOP, exp="2026-11-06")) is True


def test_votes_before_the_window_do_not_count(tmp_path, monkeypatch):
    """TLT 2026-10-15: 11 votes before 09:45, closed on the first cycle."""
    a = _agent(tmp_path, monkeypatch, before_window=True)
    for _ in range(11):
        assert a._should_exit_spread(_pos(STOP, tk="TLT")) is False
    monkeypatch.setattr("trading_agent.entry_confirmation.before_entry_window",
                        lambda now, nb: False)
    out = [a._should_exit_spread(_pos(STOP, tk="TLT")) for _ in range(3)]
    assert out == [False, False, True]


def test_immediate_signal_still_bypasses(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch, before_window=True)
    assert a._should_exit_spread(_pos(ExitSignal.HARD_STOP)) is True


def test_reset_writes_only_when_a_count_exists(tmp_path):
    store = DailyStateStore(str(tmp_path))
    reset_exit_vote(store, "X|e|s")
    assert not (tmp_path / "daily_state.json").exists() or \
        store.load().exit_vote_counts == {}
    tally_exit_vote(store, "X|e|s", "stop_loss")
    reset_exit_vote(store, "X|e|s")
    assert "X|e|s" not in store.load().exit_vote_counts


# ── Cool-down after a stop ────────────────────────────────────────────

THU, FRI, MON = date(2026, 10, 15), date(2026, 10, 16), date(2026, 10, 19)


def test_sessions_between_skips_weekends():
    assert sessions_between(THU, THU) == 0
    assert sessions_between(THU, FRI) == 1
    assert sessions_between(THU, MON) == 2
    assert sessions_between(FRI, MON) == 1


@pytest.mark.parametrize("today,blocked", [(THU, True), (FRI, True), (MON, False)])
def test_tlt_same_direction_reentry(today, blocked):
    stops = [("TLT", "Put Debit Spread", THU)]
    r = stop_cooldown_failure("TLT", "Put Debit Spread", today, stops, 1)
    assert (r is not None) is blocked
    if blocked:
        assert r.startswith("stop_cooldown")


def test_bear_call_shares_the_bear_direction():
    stops = [("TLT", "Put Debit Spread", THU)]
    assert stop_cooldown_failure("TLT", "Bear Call Spread", THU, stops, 1)


def test_opposite_direction_and_other_ticker_allowed():
    stops = [("TLT", "Put Debit Spread", THU)]
    assert stop_cooldown_failure("TLT", "Call Debit Spread", THU, stops, 1) is None
    assert stop_cooldown_failure("IWM", "Put Debit Spread", THU, stops, 1) is None


def test_neutral_structure_cools_only_its_own_kind():
    stops = [("SPY", "Iron Condor", THU)]
    assert stop_cooldown_failure("SPY", "Iron Condor", THU, stops, 1)
    assert stop_cooldown_failure("SPY", "Calendar Spread", THU, stops, 1) is None
    assert trade_direction("Cash-Secured Put") == "bull"


def test_zero_sessions_disables():
    stops = [("TLT", "Put Debit Spread", THU)]
    assert stop_cooldown_failure("TLT", "Put Debit Spread", THU, stops, 0) is None


def _close(ts, tk, strategy, sig):
    return json.dumps({"timestamp": ts, "ticker": tk, "action": "closed",
                       "raw_signal": {"strategy": strategy, "exit_signal": sig,
                                      "net_unrealized_pl": -100.0}})


def test_stops_from_journal_keeps_stop_types_and_et_date(tmp_path, monkeypatch):
    fp = tmp_path / "j.jsonl"
    fp.write_text("\n".join([
        _close("2026-10-15T13:46:10+00:00", "TLT", "Put Debit Spread", "stop_loss"),
        _close("2026-10-15T14:00:00+00:00", "IWM", "Put Debit Spread", "profit_target"),
        # 00:30 UTC Oct 15 = 20:30 ET Oct 14
        _close("2026-10-15T00:30:00+00:00", "VZ", "Cash-Secured Put", "delta_stop"),
    ]) + "\n")
    reader = JournalReader(str(fp), as_of=THU)
    got = TradingAgent._stops_from(reader)
    assert ("TLT", "Put Debit Spread", THU) in got
    assert ("VZ", "Cash-Secured Put", date(2026, 10, 14)) in got
    assert all(t != "IWM" for t, _, _ in got)


def test_agent_gate_uses_preset(monkeypatch):
    a = TradingAgent.__new__(TradingAgent)
    a.preset = SimpleNamespace(stop_cooldown_sessions=1)
    from zoneinfo import ZoneInfo
    a._recent_stops = [("TLT", "Put Debit Spread", datetime.now(ZoneInfo("America/New_York")).date())]
    plan = SimpleNamespace(strategy_name="Put Debit Spread")
    assert a._stop_cooldown_block("TLT", plan)
    a.preset = SimpleNamespace(stop_cooldown_sessions=0)
    assert a._stop_cooldown_block("TLT", plan) is None


def test_preset_default_and_summary():
    assert PresetConfig.__dataclass_fields__["stop_cooldown_sessions"].default == 5
    assert "Stop cool-down 5 sess (10 after a repeat)" in PRESETS["balanced"].to_summary_line()
