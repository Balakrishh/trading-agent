"""Skill 65 — earnings-aware expiries and trend entry filters."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from trading_agent import entry_filters as ef
from trading_agent.calendar_utils import latest_weekly_on_or_before
from trading_agent.debit_policy import CALENDAR_STRATEGY, CALL_DEBIT_STRATEGY
from trading_agent.earnings_calendar import next_earnings_date
from trading_agent.regime import Regime
from trading_agent.strategy import SpreadLeg, SpreadPlan, StrategyPlanner
from trading_agent.strategy_presets import PRESETS, _make_custom

from tests.test_debit_strategies import PRESET, _calls, _puts

TODAY = datetime.now().date()


# ── calendar helper ───────────────────────────────────────────────────────

def test_latest_weekly_on_or_before():
    wed = date(2026, 10, 14)
    assert latest_weekly_on_or_before(wed, date(2026, 10, 28), 7) == date(2026, 10, 23)
    assert latest_weekly_on_or_before(wed, date(2026, 10, 23), 7) == date(2026, 10, 23)
    assert latest_weekly_on_or_before(wed, date(2026, 10, 20), 7) is None      # < 7 DTE
    # Thanksgiving week: Friday 11/27 is after a Thursday cap → week before
    assert latest_weekly_on_or_before(wed, date(2026, 11, 26), 7) == date(2026, 11, 20)


# ── earnings date cache ──────────────────────────────────────────────────

def test_etf_never_looked_up(tmp_path):
    fetch = MagicMock()
    assert next_earnings_date("SPY", path=tmp_path / "e.json", fetch=fetch) is None
    fetch.assert_not_called()


def test_cache_hit_then_refetch_when_stale_or_passed(tmp_path):
    fp = tmp_path / "e.json"
    d = TODAY + timedelta(days=20)
    fetch = MagicMock(return_value=d)
    assert next_earnings_date("AMZN", path=fp, fetch=fetch) == d
    assert next_earnings_date("AMZN", path=fp, fetch=fetch) == d
    assert fetch.call_count == 1                                       # cached
    old = (datetime.now(timezone.utc) - timedelta(hours=13)).isoformat()
    fp.write_text(json.dumps({"AMZN": {"date": d.isoformat(), "fetched_at": old}}))
    next_earnings_date("AMZN", path=fp, fetch=fetch)
    assert fetch.call_count == 2                                       # stale
    fresh = datetime.now(timezone.utc).isoformat()
    past = (TODAY - timedelta(days=1)).isoformat()
    fp.write_text(json.dumps({"AMZN": {"date": past, "fetched_at": fresh}}))
    next_earnings_date("AMZN", path=fp, fetch=fetch)
    assert fetch.call_count == 3                                       # report passed


def test_failed_lookup_is_unknown_and_cached(tmp_path):
    fp = tmp_path / "e.json"
    fetch = MagicMock(side_effect=RuntimeError("rate limited"))
    assert next_earnings_date("META", path=fp, fetch=fetch) is None
    assert json.loads(fp.read_text())["META"]["date"] is None


# ── planner: expiries end before earnings ────────────────────────────────

def _analysis(regime=Regime.BULLISH, price=100.0):
    return SimpleNamespace(regime=regime, iv_rank=10, rsi_14=55, current_price=price,
                           mean_reversion_direction="", inter_market_inhibit_bullish=False,
                           leadership_zscore=0.0, leadership_anchor="", vix_zscore=0.0)


def _planner(preset=PRESET, earnings_in=None):
    data = MagicMock()
    data.fetch_option_chain.side_effect = lambda t, e, o: _calls() if o == "call" else _puts()
    pl = StrategyPlanner(data, preset=preset)
    if earnings_in is not None:
        pl.earnings_lookup = lambda t: TODAY + timedelta(days=earnings_in)
    return pl, data


def _no_credit(pl, monkeypatch):
    bad = SpreadPlan("AMZN", "Bull Put Spread", "bullish", [], 0, 0, 0, 0, "x", "",
                     valid=False, rejection_reason="No positive-EV candidate")
    monkeypatch.setattr(pl, "_plan_credit_by_regime", lambda t, a: bad)


def test_debit_expiry_moves_before_earnings(monkeypatch):
    pl, _ = _planner(earnings_in=21)
    _no_credit(pl, monkeypatch)
    p = pl.plan("AMZN", _analysis())
    exp = date.fromisoformat(p.expiration)
    assert p.valid and p.strategy_name == CALL_DEBIT_STRATEGY
    assert exp <= TODAY + timedelta(days=20) and (exp - TODAY).days >= 7
    assert pl.earnings_fields("AMZN")["earnings_date"] == (TODAY + timedelta(days=21)).isoformat()
    assert pl.earnings_fields("META") == {}


def test_no_expiry_fits_waits_until_after_report(monkeypatch):
    pl, data = _planner(earnings_in=5)
    _no_credit(pl, monkeypatch)
    p = pl.plan("AMZN", _analysis())
    assert not p.valid and p.strategy_name == ""
    assert "no new AMZN trade until after it" in p.rejection_reason
    data.fetch_option_chain.assert_not_called()


@pytest.mark.parametrize("kw", [{"earnings_in": None}, {"earnings_in": 21, "preset":
                                 replace(PRESET, earnings_guard_enabled=False)}])
def test_no_cap_for_etfs_or_when_disabled(monkeypatch, kw):
    pl, _ = _planner(**kw)
    _no_credit(pl, monkeypatch)
    p = pl.plan("SPY", _analysis())
    assert p.valid and (date.fromisoformat(p.expiration) - TODAY).days >= 21


def test_lookup_error_means_no_cap(monkeypatch):
    pl, _ = _planner()
    pl.earnings_lookup = MagicMock(side_effect=RuntimeError("boom"))
    _no_credit(pl, monkeypatch)
    assert pl.plan("AMZN", _analysis()).valid


def test_calendar_far_leg_must_end_before_earnings():
    pl, _ = _planner(earnings_in=30)
    pl._set_earnings_cap("AMZN")
    p = pl._plan_calendar("AMZN", _analysis(Regime.SIDEWAYS))
    assert not p.valid and p.strategy_name == CALENDAR_STRATEGY
    assert "no far leg ends before earnings" in p.rejection_reason


def test_enforce_rejects_any_leg_past_the_cap():
    pl, _ = _planner(earnings_in=21)
    pl._set_earnings_cap("AMZN")
    far = (TODAY + timedelta(days=40)).strftime("%y%m%d")
    near = (TODAY + timedelta(days=10))
    legs = [SpreadLeg(f"AMZN{near.strftime('%y%m%d')}C00250000", 250, "sell", "call", 0.5, 0, 1, 1.1, 1.05),
            SpreadLeg(f"AMZN{far}C00250000", 250, "buy", "call", 0.5, 0, 2, 2.1, 2.05)]
    plan = SpreadPlan("AMZN", "Calendar Spread", "sideways", legs, 0, -1.0, 100, 0,
                      near.isoformat(), "x")
    out = pl._enforce_earnings_cap("AMZN", _analysis(), plan)
    assert not out.valid and "spans AMZN earnings" in out.rejection_reason


def test_scanner_never_fetches_past_the_cap():
    from trading_agent.chain_scanner import ChainScanner
    data = MagicMock()
    data.fetch_option_chain.return_value = []
    sc = ChainScanner(data, replace(PRESET, dte_grid=(7, 14, 21, 30)), dte_window_days=5)
    today = date(2026, 10, 14)
    sc.max_expiration = date(2026, 10, 28)
    sc.scan("AMZN", "bull_put", today=today)
    exps = {c.args[1] for c in data.fetch_option_chain.call_args_list}
    assert exps and all(e <= "2026-10-28" for e in exps)
    sc.max_expiration = None
    data.fetch_option_chain.reset_mock()
    sc.scan("AMZN", "bull_put", today=today)
    assert max(c.args[1] for c in data.fetch_option_chain.call_args_list) > "2026-10-28"


def test_preset_defaults_and_bad_modes_fall_back():
    p = PRESETS["balanced"]
    assert p.earnings_guard_enabled and p.earnings_buffer_days == 1 and p.earnings_min_dte == 7
    assert p.rs_filter_mode == "shadow" and p.pullback_filter_mode == "shadow"
    c = _make_custom({"rs_filter_mode": "sometimes", "pullback_filter_mode": "live"})
    assert c.rs_filter_mode == "shadow" and c.pullback_filter_mode == "live"
    assert "Earnings exp ≤ report−1d" in p.to_summary_line()


# ── trend filters ─────────────────────────────────────────────────────────

def test_relative_strength_and_percentiles():
    bench = [100.0] * 61
    up = [100.0] * 41 + [105.0] * 19 + [110.0]          # +10 % over 20 d and over 60 d
    assert ef.relative_strength(up, bench) == pytest.approx(0.10)
    assert ef.relative_strength([1.0] * 10, bench) is None
    pct = ef.rs_percentiles({"A": 0.1, "B": -0.1, "C": 0.0, "D": 0.0, "E": None})
    assert pct["A"] == 1.0 and pct["B"] == 0.0 and pct["C"] == pct["D"] == 0.5
    assert "E" not in pct and ef.rs_percentiles({"A": 0.3}) == {"A": 0.5}


def _bars(last, base=100.0, n=60, rng=1.0):
    closes = [base] * (n - 1) + [last]
    return {"close": closes, "high": [c + rng for c in closes], "low": [c - rng for c in closes]}


@pytest.mark.parametrize("direction,last,ok,text", [
    ("bull", 101.0, True, "ATR above"),                # near the 20-day, above the 50-day
    ("bull", 106.0, False, "extended"),                # stretched
    ("bull", 99.0, False, "below 50-day"),             # support broken
    ("bear", 99.0, True, "ATR below"),
    ("bear", 101.0, False, "above 50-day"),
])
def test_pullback_verdicts(direction, last, ok, text):
    b = _bars(last)
    v = ef.pullback_verdict(direction, last, b["close"], b["high"], b["low"], 1.0, "live")
    assert v.ok is ok and text in v.reason


def test_pullback_below_the_20_day_reads_as_pullback():
    closes = [100.0] * 30 + [110.0] * 29 + [105.0]        # 20-day ≈ 109.8, 50-day ≈ 106
    b = {"close": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes]}
    v = ef.pullback_verdict("bull", 107.0, b["close"], b["high"], b["low"], 1.0, "live")
    assert v.ok and "below the 20-day SMA" in v.reason and v.reason.endswith("pullback")


def test_rs_verdict_sides():
    assert ef.rs_verdict("bull", 0.05, 0.8, 0.5, "live").ok
    assert not ef.rs_verdict("bull", -0.05, 0.2, 0.5, "live").ok
    assert ef.rs_verdict("bear", -0.05, 0.2, 0.5, "live").ok
    assert ef.rs_verdict("bull", None, None, 0.5, "live").ok is None


def test_evaluate_blocks_only_in_live_mode_and_only_directional():
    weak = {"score": -0.05, "percentile": 0.1}
    shadow = ef.evaluate(CALL_DEBIT_STRATEGY, PRESETS["balanced"], price=101.0,
                         rs=weak, bars=_bars(101.0))
    assert shadow.block is None and shadow.to_journal_dict()["checks"][0]["ok"] is False
    live = ef.evaluate(CALL_DEBIT_STRATEGY, replace(PRESETS["balanced"], rs_filter_mode="live"),
                       price=101.0, rs=weak, bars=_bars(101.0))
    assert live.block.startswith("trend: relative_strength")
    assert ef.evaluate("Iron Condor", PRESETS["balanced"], price=101.0) is None
    assert ef.evaluate("Bounce Bull Put Spread", PRESETS["balanced"], price=101.0) is None
    off = replace(PRESETS["balanced"], rs_filter_mode="off", pullback_filter_mode="off")
    assert ef.evaluate(CALL_DEBIT_STRATEGY, off, price=101.0) is None
    nodata = ef.evaluate(CALL_DEBIT_STRATEGY, replace(off, pullback_filter_mode="live"), price=101.0)
    assert nodata.block is None                         # missing bars never block


# ── agent wiring ──────────────────────────────────────────────────────────

def _agent(**preset_kw):
    import pandas as pd
    from trading_agent.agent import TradingAgent
    a = TradingAgent.__new__(TradingAgent)
    a.preset = replace(PRESETS["balanced"], **preset_kw)
    series = {"SPY": [100.0] * 61, "XLK": [100.0] * 40 + [110.0] * 21,
              "XLU": [100.0] * 40 + [95.0] * 21}

    def hist(t, period_days=200):
        c = series[t]
        return pd.DataFrame({"Close": c, "High": [x + 1 for x in c], "Low": [x - 1 for x in c]})
    a.data_provider = SimpleNamespace(fetch_historical_prices=hist)
    a._trend_eval = {}
    return a


def _plan(name):
    return SpreadPlan("XLU", name, "bullish", [], 1, -1.0, 100, 0, "2026-11-13", "x")


def test_agent_ranks_and_records_shadow_verdict():
    a = _agent()
    a._trend_rs = a._relative_strength_map(["XLK", "XLU"])
    assert a._trend_rs["XLK"]["percentile"] == 1.0 and a._trend_rs["XLU"]["percentile"] == 0.0
    an = SimpleNamespace(current_price=95.0)
    assert a._trend_filter_block("XLU", _plan(CALL_DEBIT_STRATEGY), an) is None
    rec = a._trend_eval["XLU"]
    assert rec["direction"] == "bull" and not rec["blocked"]
    assert {c["name"] for c in rec["checks"]} == {"relative_strength", "pullback"}


def test_agent_live_mode_blocks_laggard_bullish_entry():
    a = _agent(rs_filter_mode="live")
    a._trend_rs = a._relative_strength_map(["XLK", "XLU"])
    block = a._trend_filter_block("XLU", _plan(CALL_DEBIT_STRATEGY), SimpleNamespace(current_price=95.0))
    assert block.startswith("trend: relative_strength")
    assert a._trend_filter_block("XLU", _plan("Iron Condor"), SimpleNamespace(current_price=95.0)) is None


def test_agent_rs_off_or_no_benchmark_is_empty():
    assert _agent(rs_filter_mode="off")._relative_strength_map(["XLK"]) == {}
    a = _agent()
    a.data_provider = SimpleNamespace(fetch_historical_prices=MagicMock(side_effect=OSError("down")))
    assert a._relative_strength_map(["XLK"]) == {}
