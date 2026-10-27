"""Skill 68 — macro-event guard, implied-vs-realized floor, credit stress,
macro calendar, preset wiring."""
import json
from dataclasses import replace
from datetime import date, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from trading_agent import macro_calendar as mc
from trading_agent import macro_vol_gates as mvg
from trading_agent import market_state as ms
from trading_agent.strategy_presets import PRESETS, PresetConfig, _make_custom

EVENTS = [mc.MacroEvent("NFP", date(2026, 11, 6)), mc.MacroEvent("CPI", date(2026, 11, 10)),
          mc.MacroEvent("FOMC", date(2026, 10, 28))]


# ── calendar ──────────────────────────────────────────────────────────

def test_builtin_tables_sorted_and_cover_the_live_period():
    evs = mc.all_events(path=None)
    for kind in mc.KINDS:
        days = [e.day for e in evs if e.kind == kind]
        assert days == sorted(days) and len(set(days)) == len(days)
    assert mc.MacroEvent("FOMC", date(2026, 10, 28)) in evs
    assert mc.MacroEvent("NFP", date(2026, 11, 6)) in evs
    assert all(e.day.weekday() < 5 for e in evs)


def test_events_between_next_and_coverage():
    got = mc.events_between(date(2026, 10, 28), date(2026, 11, 6), EVENTS)
    assert [e.kind for e in got] == ["FOMC", "NFP"]
    assert mc.events_between(date(2026, 10, 28), date(2026, 11, 30), EVENTS, kinds=("CPI",))[0].kind == "CPI"
    assert mc.next_event(date(2026, 10, 29), EVENTS).kind == "NFP"
    warn = mc.coverage_warnings(date(2026, 10, 27), EVENTS)
    assert len(warn) == 3 and "--refresh" in warn[0]
    assert mc.coverage_warnings(date(2026, 9, 1), EVENTS, lookahead=30) == []


def test_refreshed_json_merges_over_builtins(tmp_path):
    fp = tmp_path / "m.json"
    fp.write_text(json.dumps({"events": {"CPI": ["2027-01-13"], "BOGUS": ["2027-01-01"]}}))
    evs = mc.all_events(fp)
    assert mc.MacroEvent("CPI", date(2027, 1, 13)) in evs
    assert not any(e.kind == "BOGUS" for e in evs)


FED_HTML = """
<h4>2027 FOMC Meetings</h4>
<div class="fomc-meeting__month col"><strong>January</strong></div><div class="fomc-meeting__date col">26-27</div>
<div class="fomc-meeting__month col"><strong>Apr/May</strong></div><div class="fomc-meeting__date col">30-1*</div>
<div class="fomc-meeting__month col"><strong>October</strong></div><div class="fomc-meeting__date col">4 (unscheduled)</div>
<h4>2026 FOMC Meetings</h4>
<div class="fomc-meeting__month col"><strong>October</strong></div><div class="fomc-meeting__date col">27-28</div>
"""


def test_parse_fomc_and_fred_pages():
    assert mc.parse_fomc_page(FED_HTML) == [date(2026, 10, 28), date(2027, 1, 27), date(2027, 5, 1)]
    html = "<span>Wednesday January 13, 2027</span> <span>Friday February 12, 2027</span> junk"
    assert mc.parse_fred_calendar(html) == [date(2027, 1, 13), date(2027, 2, 12)]


def test_refresh_keeps_stored_dates_when_a_source_fails(tmp_path):
    fp = tmp_path / "m.json"
    fp.write_text(json.dumps({"events": {"NFP": ["2026-12-04"]}}))

    def get(url):
        if "federalreserve" in url:
            return FED_HTML
        if "rid=10" in url:
            return "Wednesday January 13, 2027"
        raise RuntimeError("fred down")

    counts = mc.refresh(date(2026, 10, 27), fp, get)
    data = json.loads(fp.read_text())["events"]
    assert data["NFP"] == ["2026-12-04"] and "2027-01-13" in data["CPI"]
    assert counts["FOMC"] == 3


# ── macro event guard ─────────────────────────────────────────────────

NOW = datetime(2026, 11, 2, 10, 30)          # Monday before the Nov 6 jobs report


def test_guard_blocks_short_premium_spanning_nfp():
    r = mvg.macro_event_failure("SPY", "Bull Put Spread", date(2026, 11, 13), NOW, ("NFP",), 14, EVENTS)
    assert r.startswith("macro_event_guard (NFP 2026-11-06 08:30 ET inside a 11-day Bull Put Spread")


@pytest.mark.parametrize("ticker,strategy,exp,kinds", [
    ("SPY", "Call Debit Spread", date(2026, 11, 13), ("NFP",)),     # long premium
    ("SPY", "Calendar Spread", date(2026, 11, 13), ("NFP",)),
    ("TLT", "Bull Put Spread", date(2026, 11, 13), ("NFP",)),       # bonds exempt
    ("SPY", "Bull Put Spread", date(2026, 11, 27), ("NFP",)),       # 25 DTE > 14
    ("SPY", "Bull Put Spread", date(2026, 11, 5), ("NFP",)),        # expires before
    ("SPY", "Bull Put Spread", date(2026, 11, 13), ("FOMC",)),      # kind not chosen
])
def test_guard_passes(ticker, strategy, exp, kinds):
    assert mvg.macro_event_failure(ticker, strategy, exp, NOW, kinds, 14, EVENTS) is None


def test_release_earlier_today_is_already_priced():
    morning = datetime(2026, 11, 6, 8, 0)
    after = datetime(2026, 11, 6, 9, 50)
    assert mvg.macro_event_failure("QQQ", "Iron Condor", date(2026, 11, 13), morning, ("NFP",), 14, EVENTS)
    assert mvg.macro_event_failure("QQQ", "Iron Condor", date(2026, 11, 13), after, ("NFP",), 14, EVENTS) is None
    assert mvg.macro_event_failure("QQQ", "Iron Condor", date(2026, 11, 13), morning, ("NFP",), 0, EVENTS) is None


# ── implied vs realized ───────────────────────────────────────────────

def test_vrp_failure():
    assert mvg.vrp_failure("Bear Call Spread", 0.15, 0.18, 1.0).startswith("vrp_below_min (30d ATM IV 15.0%")
    assert mvg.vrp_failure("Bear Call Spread", 0.20, 0.18, 1.0) is None
    assert mvg.vrp_failure("Put Debit Spread", 0.10, 0.30, 1.0) is None
    assert mvg.vrp_failure("Bear Call Spread", None, 0.18, 1.0) is None        # sentinel, not a block
    assert mvg.vrp_failure("Bear Call Spread", 0.15, 0.18, 0.0) is None        # off


def _vs_row(t, d, iv30=None, term=None):
    exps = [{"target_dte": 7, "atm_iv": 0.5}] + ([{"target_dte": 30, "atm_iv": iv30}] if iv30 else [])
    return json.dumps({"date": d, "ticker": t, "expiries": exps, "term": {"iv30": term}})


def test_latest_atm_iv_newest_row_and_staleness(tmp_path):
    fp = tmp_path / "vs.jsonl"
    fp.write_text("\n".join([_vs_row("SPY", "2026-10-23", 0.14), _vs_row("SPY", "2026-10-26", 0.12),
                             _vs_row("GLD", "2026-10-26", None, 0.21), "garbage"]) + "\n")
    assert mvg.latest_atm_iv("SPY", date(2026, 10, 27), fp) == 0.12
    assert mvg.latest_atm_iv("GLD", date(2026, 10, 27), fp) == 0.21
    assert mvg.latest_atm_iv("SPY", date(2026, 11, 2), fp) is None           # 7 days old
    assert mvg.latest_atm_iv("IWM", date(2026, 10, 27), fp) is None


def test_context_fields():
    f = mvg.context_fields("SPY", date(2026, 11, 13), NOW, 0.15, 0.12, EVENTS)
    assert f["macro_next_event"] == "NFP 2026-11-06 08:30 ET"
    assert f["macro_events_in_trade"] == ["NFP 2026-11-06 08:30 ET", "CPI 2026-11-10 08:30 ET"]
    assert f["iv_rv_ratio"] == 1.25 and f["vrp_available"] is True
    g = mvg.context_fields("SPY", None, datetime(2026, 10, 28, 11, 0), None, 0.12, EVENTS)
    assert g["macro_next_event"] == "FOMC 2026-10-28 14:00 ET" and g["vrp_available"] is False


# ── agent wiring ──────────────────────────────────────────────────────

def _agent(preset, rv=0.20):
    from trading_agent.agent import TradingAgent
    a = TradingAgent.__new__(TradingAgent)
    a.preset = preset
    a._rv20_cache = {"SPY": rv}
    return a


def _plan(strategy, exp="2026-11-13"):
    from trading_agent.strategy import SpreadLeg
    sym = "SPY" + date.fromisoformat(exp).strftime("%y%m%d") + "P00700000"
    leg = SpreadLeg(symbol=sym, strike=700.0, action="sell", option_type="put", delta=-0.2,
                    theta=0.0, bid=1.0, ask=1.1, mid=1.05)
    return SimpleNamespace(strategy_name=strategy, ticker="SPY", expiration=exp, legs=[leg])


def test_agent_block_macro_then_vrp(monkeypatch):
    monkeypatch.setattr("trading_agent.agent.TradingAgent._now_et", staticmethod(lambda: NOW))
    monkeypatch.setattr(mvg, "latest_atm_iv", lambda t, d, path=None: 0.15)
    monkeypatch.setattr(mc, "all_events", lambda path=None: EVENTS)
    a = _agent(PRESETS["balanced"])
    assert a._macro_vol_block("SPY", _plan("Bull Put Spread")).startswith("macro_event_guard")
    # 25 DTE: past the event window → the IV/RV floor decides (0.15 / 0.20 < 1).
    assert a._macro_vol_block("SPY", _plan("Bull Put Spread", "2026-11-27")).startswith("vrp_below_min")
    assert a._macro_vol_block("SPY", _plan("Put Debit Spread")) is None
    off = _agent(replace(PRESETS["balanced"], macro_event_guard_enabled=False, vrp_min_ratio=0.0))
    assert off._macro_vol_block("SPY", _plan("Bull Put Spread")) is None
    fields = a._macro_vol_fields(_plan("Bull Put Spread"))
    assert fields["iv_rv_ratio"] == 0.75 and fields["macro_events_in_trade"]


# ── credit stress / market state ──────────────────────────────────────

def test_credit_stress_pct():
    assert ms.credit_stress_pct([1.0] * 49) is None
    avg = (49 + 0.97) / 50
    assert ms.credit_stress_pct([1.0] * 49 + [0.97]) == pytest.approx((1 - 0.97 / avg) * 100, abs=1e-3)
    assert ms.credit_stress_pct([1.0] * 50 + [1.05]) < 0


def test_aligned_ratio_uses_common_dates():
    idx = pd.to_datetime(["2026-10-21", "2026-10-22", "2026-10-23"])
    a = pd.DataFrame({"Close": [80.0, 81.0, 82.0]}, index=idx)
    b = pd.DataFrame({"close": [100.0, 100.0]}, index=idx[1:])
    assert ms._aligned_ratio(a, b) == [0.81, 0.82]


BASE = dict(spy_price=770.0, spy_sma20=765.0, spy_sma50=760.0, spy_sma200=720.0,
            spy_rsi=55.0, vix=15.0, vix3m=17.0, breadth=0.7, breadth_n=14)


def test_credit_stress_triggers_caution_and_holds_it():
    r = ms.classify_market_state(ms.MarketInputs(**BASE, credit_stress=2.4))
    assert r.state == ms.CAUTION and r.reasons[0].startswith("credit stress: HYG/IEF 2.4%")
    assert ms.classify_market_state(ms.MarketInputs(**BASE, credit_stress=1.5)).state == ms.NORMAL
    held = ms.classify_market_state(ms.MarketInputs(**BASE, credit_stress=1.5), prior_state=ms.CAUTION)
    assert held.state == ms.CAUTION and "credit stress 1.5% not yet < 1%" in held.reasons[0]
    assert ms.classify_market_state(ms.MarketInputs(**BASE, credit_stress=None)).state == ms.NORMAL


def test_compute_inputs_reads_credit_and_survives_its_failure():
    idx = pd.date_range("2026-06-01", periods=60, freq="B")
    spy = pd.DataFrame({"Close": [700.0 + i for i in range(60)] * 1}, index=idx)
    hyg = pd.DataFrame({"Close": [80.0] * 59 + [77.0]}, index=idx)
    ief = pd.DataFrame({"Close": [100.0] * 60}, index=idx)
    frames = {"SPY": spy, "HYG": hyg, "IEF": ief}
    inp = ms.compute_inputs(lambda t: frames[t], [], lambda s: 15.0)
    assert inp.credit_stress > 3.0
    inp2 = ms.compute_inputs(lambda t: frames["SPY"] if t == "SPY" else 1 / 0, [], lambda s: 15.0)
    assert inp2.credit_stress is None and inp2.spy_price is not None
    # The adjusted fetcher, when given, is the one the credit pair uses.
    flat = {"HYG": pd.DataFrame({"Close": [80.0] * 60}, index=idx), "IEF": ief}
    inp3 = ms.compute_inputs(lambda t: frames[t], [], lambda s: 15.0, lambda t: flat[t])
    assert inp3.credit_stress == 0.0


# ── preset ────────────────────────────────────────────────────────────

def test_preset_defaults_summary_and_coercion():
    f = PresetConfig.__dataclass_fields__
    assert f["macro_event_guard_enabled"].default is True
    assert f["macro_event_kinds"].default == ("NFP",)
    assert f["macro_event_max_dte"].default == 14 and f["vrp_min_ratio"].default == 1.0
    line = PRESETS["balanced"].to_summary_line()
    assert "Macro guard NFP ≤14d" in line and "IV/RV ≥1" in line
    c = _make_custom({"macro_event_kinds": ["nfp", "CPI", "GDP"]})
    assert c.macro_event_kinds == ("NFP", "CPI")
