"""Backlog §3 / §6.8 — realized-vol POP shadow log."""
from __future__ import annotations

import math

import pytest

from trading_agent.shadow_pop import prob_above, realized_vol, shadow_pop
from trading_agent.strategy import SpreadLeg, SpreadPlan


def leg(action, opt, strike, delta):
    return SpreadLeg("X", strike, action, opt, delta, 0.0, 1.0, 1.1, 1.05)


def plan(legs, net, width=5.0, far=""):
    return SpreadPlan("SPY", "x", "bullish", legs, width, net, 100.0, 0.2, "2026-11-06", "",
                      far_expiration=far)


def test_realized_vol_constant_returns_is_zero_and_needs_history():
    assert realized_vol([100.0] * 25) == 0.0
    assert realized_vol([100.0] * 10) is None
    closes = [100 * math.exp(0.01 * (i % 2)) for i in range(30)]   # ±1 % zig-zag
    assert realized_vol(closes) == pytest.approx(0.01 * 0.5129 * math.sqrt(252) * 2, rel=0.05)


def test_prob_above_symmetry():
    assert prob_above(100, 100, 30, 0.2) == pytest.approx(0.4886, abs=1e-3)    # −σ²t/2 drift
    assert prob_above(100, 0, 30, 0.2) == 1.0


def test_bull_put_uses_breakeven():
    p = plan([leg("sell", "put", 95, -0.25), leg("buy", "put", 90, -0.12)], 1.0)
    r = shadow_pop(p, 100.0, 0.20, 30)
    assert r["pop_delta"] == 0.75 and r["breakeven_low"] == 94.0
    assert r["pop_rv"] == pytest.approx(prob_above(100, 94, 30, 0.2), abs=1e-4)


def test_iron_condor_between_breakevens():
    p = plan([leg("sell", "put", 95, -0.2), leg("buy", "put", 90, -0.1),
              leg("sell", "call", 105, 0.2), leg("buy", "call", 110, 0.1)], 1.0)
    r = shadow_pop(p, 100.0, 0.20, 30)
    assert (r["breakeven_low"], r["breakeven_high"]) == (94.0, 106.0)
    assert r["pop_delta"] == pytest.approx(0.6)
    assert r["pop_rv"] == pytest.approx(prob_above(100, 94, 30, .2) - prob_above(100, 106, 30, .2), abs=1e-4)


def test_debit_verticals_and_calendar():
    call = plan([leg("buy", "call", 100, 0.5), leg("sell", "call", 102, 0.38)], -0.9, width=2.0)
    r = shadow_pop(call, 100.0, 0.20, 30)
    assert r["breakeven_low"] == 100.9 and r["pop_delta"] == pytest.approx(0.5 - 0.12 * 0.45)
    put = plan([leg("buy", "put", 100, -0.5), leg("sell", "put", 98, -0.38)], -0.9, width=2.0)
    assert shadow_pop(put, 100.0, 0.20, 30)["breakeven_high"] == 99.1
    cal = plan([leg("buy", "call", 100, 0.5), leg("sell", "call", 100, 0.5)], -1.0, 0.0, far="2026-12-18")
    assert shadow_pop(cal, 100.0, 0.2, 30) == {"pop_delta": None, "pop_rv": None,
                                                "breakeven_low": None, "breakeven_high": None}


def test_missing_sigma_keeps_delta_pop():
    p = plan([leg("sell", "put", 95, -0.25), leg("buy", "put", 90, -0.12)], 1.0)
    r = shadow_pop(p, 100.0, None, 30)
    assert r["pop_delta"] == 0.75 and r["pop_rv"] is None


def test_agent_shadow_fields_sentinel():
    from types import SimpleNamespace
    import pandas as pd
    from trading_agent.agent import TradingAgent
    a = TradingAgent.__new__(TradingAgent)
    p = plan([leg("sell", "put", 95, -0.25), leg("buy", "put", 90, -0.12)], 1.0)
    an = SimpleNamespace(current_price=100.0)
    a.data_provider = SimpleNamespace(fetch_historical_prices=lambda t, period_days: pd.DataFrame(
        {"Close": [100 * (1.01 if i % 2 else 1.0) for i in range(60)]}))
    ok = a._shadow_pop_fields(p, an)
    assert ok["shadow_pop_available"] is True and ok["rv_20d"] > 0 and ok["pop_rv"] is not None

    def boom(t, period_days):
        raise RuntimeError("yfinance down")
    a.data_provider = SimpleNamespace(fetch_historical_prices=boom)
    bad = a._shadow_pop_fields(p, an)
    assert bad["shadow_pop_available"] is False and bad["pop_rv"] is None and bad["pop_delta"] == 0.75
    assert a._shadow_pop_fields(plan([], 0.0), an) == {}
