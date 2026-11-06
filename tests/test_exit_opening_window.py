"""Skill 30 §4 — debounced exits wait for no_exit_before_et (backlog D1).

2026-10-09: the VZ CSP delta stop fired at 09:36 ET and filled at $1.18
against a $1.045 mark in the open's wide quotes."""
from types import SimpleNamespace

import pytest

from trading_agent import agent as agent_mod
from trading_agent.agent import TradingAgent
from trading_agent.position_monitor import ExitSignal
from trading_agent.strategy_presets import PRESETS


def _agent(monkeypatch, *, votes, before_window, not_before="09:45", before_bell=False):
    a = TradingAgent.__new__(TradingAgent)
    a.preset = SimpleNamespace(no_exit_before_et=not_before)
    a.daily_state = object()
    monkeypatch.setattr(agent_mod, "tally_exit_vote", lambda *a, **k: votes)
    monkeypatch.setattr(agent_mod, "reset_exit_vote", lambda *a, **k: None)
    monkeypatch.setattr("trading_agent.entry_confirmation.before_entry_window",
                        lambda now, nb: before_bell if nb == "09:30" else before_window)
    return a


def spread(sig):
    return SimpleNamespace(underlying="VZ", exit_signal=sig)


@pytest.mark.parametrize("sig", [ExitSignal.DELTA_STOP, ExitSignal.PROFIT_TARGET,
                                 ExitSignal.REGIME_SHIFT, ExitSignal.STOP_LOSS])
def test_confirmed_debounced_exit_waits_before_the_window(monkeypatch, sig):
    assert _agent(monkeypatch, votes=3, before_window=True)._should_exit_spread(spread(sig)) is False


def test_confirmed_exit_goes_out_after_the_window(monkeypatch):
    a = _agent(monkeypatch, votes=3, before_window=False)
    assert a._should_exit_spread(spread(ExitSignal.DELTA_STOP)) is True


@pytest.mark.parametrize("sig", [ExitSignal.HARD_STOP, ExitSignal.STRIKE_PROXIMITY])
def test_mark_emergencies_skip_the_window(monkeypatch, sig):
    assert _agent(monkeypatch, votes=0, before_window=True)._should_exit_spread(spread(sig)) is True


@pytest.mark.parametrize("sig", [ExitSignal.DTE_SAFETY, ExitSignal.EARNINGS_EXIT])
def test_calendar_exits_wait_for_the_window_then_skip_the_debounce(monkeypatch, sig):
    assert _agent(monkeypatch, votes=0, before_window=True)._should_exit_spread(spread(sig)) is False
    assert _agent(monkeypatch, votes=0, before_window=False)._should_exit_spread(spread(sig)) is True


# The agent runs from 09:25 ET; AMZN read -$1,275 on a $460 spread at 09:25 on 10-20.
@pytest.mark.parametrize("sig", [ExitSignal.HARD_STOP, ExitSignal.STRIKE_PROXIMITY,
                                 ExitSignal.DTE_SAFETY, ExitSignal.EARNINGS_EXIT,
                                 ExitSignal.STOP_LOSS, ExitSignal.PROFIT_TARGET])
def test_nothing_closes_before_the_bell(monkeypatch, sig):
    a = _agent(monkeypatch, votes=3, before_window=True, not_before="", before_bell=True)
    assert a._should_exit_spread(spread(sig)) is False


def test_empty_window_disables_the_hold(monkeypatch):
    a = _agent(monkeypatch, votes=3, before_window=True, not_before="")
    assert a._should_exit_spread(spread(ExitSignal.DELTA_STOP)) is True


def test_unconfirmed_exit_still_debounces(monkeypatch):
    a = _agent(monkeypatch, votes=2, before_window=False)
    assert a._should_exit_spread(spread(ExitSignal.DELTA_STOP)) is False


def test_preset_default_and_summary():
    p = PRESETS["balanced"]
    assert p.no_exit_before_et == "09:45"
    assert "Exits from 09:45" in p.to_summary_line()
