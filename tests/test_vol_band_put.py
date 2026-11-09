"""2026-11-09 — vol-band bull put playbook and the optional credit-vertical
strike-proximity exit (docs/plans/research_credit_vol.md §5)."""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from trading_agent.position_monitor import ExitSignal, PositionMonitor, SpreadPosition
from trading_agent.regime import Regime
from trading_agent.strategy import SpreadPlan, StrategyPlanner
from trading_agent.strategy_presets import PRESETS, load_active_preset

VB = replace(PRESETS["balanced"], vol_band_put_enabled=True)


def _chain():
    out = []
    for k in range(85, 101):
        p = 2.0 * 1.25 ** (k - 95)
        out.append(dict(symbol=f"SPY261127P{k:05d}000", strike=float(k), bid=round(p - 0.02, 2),
                        ask=round(p + 0.02, 2), mid=round(p, 2), delta=-max(0.01, 0.25 + 0.05 * (k - 95)), theta=0))
    return out


def _planner(preset=VB, budget=1000.0, rank=60.0, regime=Regime.BEARISH):
    data = MagicMock()
    data.fetch_option_chain.return_value = _chain()
    pl = StrategyPlanner(data, preset=preset)
    pl.max_loss_budget = budget
    a = SimpleNamespace(regime=regime, iv_rank=rank, rsi_14=45, current_price=100.0,
                        mean_reversion_direction="", inter_market_inhibit_bullish=False,
                        leadership_zscore=0.0, leadership_anchor="", vix_zscore=0.0)
    return pl, a


# ── planner ──────────────────────────────────────────────────────────────

def test_sells_quarter_delta_put_at_the_widest_width_inside_two_percent():
    pl, a = _planner()
    p = pl._plan_vol_band_put("SPY", a)
    assert p.valid and p.playbook == "vol_band_put" and p.min_cw == 0.10
    assert p.strategy_name == "Bull Put Spread"
    assert [l.strike for l in p.legs] == [95.0, 93.0] and p.spread_width == 2.0


def test_width_narrows_to_fit_the_risk_budget():
    pl, a = _planner(budget=100.0)                   # $2 wide risks $132 > $100
    p = pl._plan_vol_band_put("SPY", a)
    assert p.valid and p.spread_width == 1.0 and p.max_loss <= 100.0


def test_no_width_fits_gives_an_invalid_plan_with_reason():
    pl, a = _planner(budget=20.0)
    p = pl._plan_vol_band_put("SPY", a)
    assert not p.valid and "risk cap" in p.rejection_reason


def test_min_credit_to_width_filters_thin_credit():
    pl, a = _planner(preset=replace(VB, vol_band_put_min_cw=0.40))
    p = pl._plan_vol_band_put("SPY", a)
    assert not p.valid and "C/W ≥ 0.4" in p.rejection_reason


@pytest.mark.parametrize("preset, ticker, rank", [
    (PRESETS["balanced"], "SPY", 60.0),               # playbook off (default)
    (VB, "XLE", 60.0),                                # ticker not listed
    (VB, "SPY", 49.9),                                # below the band
    (VB, "SPY", 80.0),                                # band is half-open
    (VB, "SPY", None),                                # no vol rank
])
def test_not_applicable_returns_none(preset, ticker, rank):
    pl, a = _planner(preset=preset, rank=rank)
    assert pl._plan_vol_band_put(ticker, a) is None


# ── routing ──────────────────────────────────────────────────────────────

def _invalid(name="Bull Put Spread"):
    return SpreadPlan(ticker="SPY", strategy_name=name, regime="bearish", legs=[], spread_width=0,
                      net_credit=0, max_loss=0, credit_to_width_ratio=0, expiration="",
                      reasoning="", valid=False, rejection_reason="cw below floor")


def test_used_only_when_no_other_plan_is_valid(monkeypatch):
    pl, a = _planner()
    monkeypatch.setattr(pl, "_plan_by_playbook", lambda t, an: _invalid())
    assert pl.plan("SPY", a).playbook == "vol_band_put"
    good = replace(_invalid(), valid=True, rejection_reason="")
    monkeypatch.setattr(pl, "_plan_by_playbook", lambda t, an: good)
    assert pl.plan("SPY", a) is good


def test_deliberate_no_trade_is_respected(monkeypatch):
    pl, a = _planner()
    monkeypatch.setattr(pl, "_plan_by_playbook", lambda t, an: _invalid(name=""))
    assert pl.plan("SPY", a).playbook == ""


def test_failed_vol_band_reason_is_appended(monkeypatch):
    pl, a = _planner(budget=20.0)
    monkeypatch.setattr(pl, "_plan_by_playbook", lambda t, an: _invalid())
    p = pl.plan("SPY", a)
    assert not p.valid and p.rejection_reason.startswith("cw below floor; vol-band put")


def test_plan_dict_carries_the_tag():
    pl, a = _planner()
    d = pl._plan_vol_band_put("SPY", a).to_dict()
    assert d["playbook"] == "vol_band_put" and d["min_cw"] == 0.10


# ── risk manager / executor floor ────────────────────────────────────────

def test_risk_manager_and_executor_use_the_plan_floor():
    from trading_agent.executor import OrderExecutor
    from trading_agent.risk_manager import RiskManager
    pl, a = _planner()
    p = pl._plan_vol_band_put("SPY", a)               # C/W 0.30 < |Δ| 0.25 × 1.10? no — force a low ratio
    rm = RiskManager(max_risk_pct=0.02, delta_aware_floor=True, edge_buffer=0.10, max_delta=0.30)
    v = rm.evaluate(p, account_balance=100_000, market_open=True)
    assert any("vol_band_put floor" in c for c in v.checks_passed)
    ex = object.__new__(OrderExecutor)
    ex.delta_aware_floor, ex.edge_buffer, ex.min_credit_ratio, ex.max_risk_pct = True, 0.10, 0.33, 0.02
    assert ex._recheck_live_economics(p, 0.30, 100_000)[0]           # 0.15 ≥ 0.10
    assert not ex._recheck_live_economics(p, 0.10, 100_000)[0]       # 0.05 < 0.10
    p.min_cw = None
    assert not ex._recheck_live_economics(p, 0.30, 100_000)[0]       # delta floor 0.275


# ── exits ────────────────────────────────────────────────────────────────

def _spread(strategy="Bull Put Spread", playbook="", short=99.5):
    return SpreadPosition(underlying="SPY", strategy_name=strategy, legs=[], original_credit=0.60,
                          max_loss=140.0, spread_width=2.0, net_unrealized_pl=-5.0,
                          expiration="2027-06-18", short_strikes=[short], playbook=playbook)


def _mon(**kw):
    return PositionMonitor(api_key="x", secret_key="x", **kw)


def test_proximity_exit_switch_covers_credit_verticals_only():
    near = 100.0                                       # 0.5 % from the 99.5 short
    assert _mon()._check_exit(_spread(), {}, underlying_price=near)[0] == ExitSignal.STRIKE_PROXIMITY
    off = _mon(credit_strike_proximity_exit=False)
    assert off._check_exit(_spread(), {}, underlying_price=near)[0] == ExitSignal.HOLD
    assert off._check_exit(_spread("Bear Call Spread"), {}, underlying_price=near)[0] == ExitSignal.HOLD
    assert off._check_exit(_spread("Iron Condor"), {}, underlying_price=near)[0] == ExitSignal.STRIKE_PROXIMITY


def test_stops_still_fire_with_proximity_off():
    s = _spread()
    s.net_unrealized_pl = -80.0                        # ≥ 50 % of the $140 max loss
    assert _mon(credit_strike_proximity_exit=False)._check_exit(s, {}, underlying_price=100.0)[0] \
        == ExitSignal.STOP_LOSS


def test_vol_band_put_skips_the_regime_shift_exit():
    bear = {"SPY": Regime.BEARISH}
    assert _mon()._check_exit(_spread(short=90.0), bear, underlying_price=100.0)[0] == ExitSignal.REGIME_SHIFT
    assert _mon()._check_exit(_spread(short=90.0, playbook="vol_band_put"), bear,
                              underlying_price=100.0)[0] == ExitSignal.HOLD


def test_playbook_flows_from_trade_plan_to_position():
    from tests.test_position_monitor import _ENVELOPE_PLAN, _OPEN_LEGS
    plan = json.loads(json.dumps(_ENVELOPE_PLAN))
    plan["trade_plan"]["playbook"] = "vol_band_put"
    s = _mon().group_into_spreads(_OPEN_LEGS, [plan])[0]
    assert s.playbook == "vol_band_put"


# ── agent filter exemption ───────────────────────────────────────────────

def test_trend_filters_are_journaled_but_do_not_block(monkeypatch):
    from trading_agent import entry_filters
    from trading_agent.agent import TradingAgent
    agent = object.__new__(TradingAgent)
    agent.preset = replace(VB, pullback_filter_mode="off")
    agent._trend_rs, agent._trend_eval = {}, {}
    ev = SimpleNamespace(block="RS below p50", to_journal_dict=lambda: {"blocked": True})
    monkeypatch.setattr(entry_filters, "evaluate", lambda *a, **k: ev)
    a = SimpleNamespace(current_price=100.0)
    assert agent._trend_filter_block("SPY", SimpleNamespace(strategy_name="Bull Put Spread", playbook="vol_band_put"), a) is None
    assert agent._trend_eval["SPY"] == {"blocked": True}
    assert agent._trend_filter_block("SPY", SimpleNamespace(strategy_name="Bull Put Spread", playbook=""), a) \
        == "RS below p50"


# ── preset ───────────────────────────────────────────────────────────────

def test_preset_loads_validates_and_summarizes(tmp_path):
    fp = tmp_path / "STRATEGY_PRESET.json"
    fp.write_text(json.dumps({"profile": "custom", "custom": {
        "vol_band_put_enabled": True, "vol_band_put_tickers": ["SPY", "QQQ"],
        "credit_strike_proximity_exit": False}}))
    p = load_active_preset(fp)
    assert p.vol_band_put_tickers == ("SPY", "QQQ") and not p.credit_strike_proximity_exit
    line = p.to_summary_line()
    assert "VolBandPut SPY,QQQ rank 50–80 Δ0.25 21d ≤2% C/W≥0.1" in line and "Credit prox exit off" in line
    assert "VolBandPut" not in PRESETS["balanced"].to_summary_line()
    fp.write_text(json.dumps({"profile": "custom", "custom": {
        "vol_band_put_min_rank": 80, "vol_band_put_max_rank": 50}}))
    p = load_active_preset(fp)
    assert (p.vol_band_put_min_rank, p.vol_band_put_max_rank) == (50.0, 80.0)
