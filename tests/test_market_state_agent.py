"""Skill 58 — agent wiring of the market risk state (per-cycle update,
risk multiplier on RiskManager + executor, journal row on change)."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from trading_agent import market_state
from trading_agent.agent import TradingAgent
from trading_agent.strategy_presets import PRESETS


def _agent(enabled=True, closes=None, raise_history=False):
    rows = []
    a = TradingAgent.__new__(TradingAgent)
    a.preset = PRESETS["balanced"].__class__(**{**PRESETS["balanced"].__dict__,
                                               "market_state_enabled": enabled})
    a._base_max_risk_pct = 0.02
    a._market_state = None
    a.risk_manager = SimpleNamespace(max_risk_pct=0.02)
    a.executor = SimpleNamespace(max_risk_pct=0.02)
    a.journal_kb = SimpleNamespace(log_signal=lambda **kw: rows.append(kw))
    a._exception_monitor = SimpleNamespace(record=lambda **kw: rows.append({"exc": kw}))

    def hist(t, period_days=200):
        if raise_history:
            raise RuntimeError("yfinance down")
        return pd.DataFrame({"Close": closes})
    a.data_provider = SimpleNamespace(fetch_historical_prices=hist)
    return a, rows


def test_disabled_keeps_full_size():
    a, rows = _agent(enabled=False)
    a.risk_manager.max_risk_pct = 0.01
    a._update_market_state(["SPY"])
    assert a._market_state is None and a.risk_manager.max_risk_pct == 0.02
    assert rows == [] and market_state.read_state() is None


def test_broad_downtrend_is_defensive_quarter_size_and_journals_once():
    closes = [200.0 - i * 0.3 for i in range(250)]               # SPY below every average
    a, rows = _agent(closes=closes)
    a._update_market_state(["SPY", "QQQ"], 30_000)
    assert a._market_state.state == "DEFENSIVE"          # breadth 0 % below SMA-200
    assert a.risk_manager.max_risk_pct == 0.005 and a.executor.max_risk_pct == 0.005
    snap = market_state.read_state()
    assert snap["state"] == "DEFENSIVE" and snap["account_balance"] == 30_000
    assert [r["ticker"] for r in rows] == ["__market__"]
    a._update_market_state(["SPY", "QQQ"], 30_000)              # unchanged → no new row
    assert len(rows) == 1


def test_history_failure_fails_safe_to_caution():
    a, rows = _agent(raise_history=True)
    a._update_market_state(["SPY"])
    assert a._market_state.state == "CAUTION"
    assert "spy_data_unavailable" in a._market_state.reasons[0]
