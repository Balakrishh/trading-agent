"""2026-11-06 — exits by direction, the 200-day-trend and volatility-regime
entry filters for debit verticals, bad-quote guards, and the calendar
far-leg snap (skills 59, 65, 30; docs/plans/research_debit_exits.md)."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pytest

from trading_agent import entry_filters as ef
from trading_agent.debit_policy import CALENDAR_STRATEGY, CALL_DEBIT_STRATEGY, PUT_DEBIT_STRATEGY
from trading_agent.position_monitor import ExitSignal, PositionMonitor, SpreadPosition
from trading_agent.strategy_presets import PRESETS, load_active_preset

from tests.test_mid_exit_ladder_fit import _leg


def _spread(name, pl, natural=None, credit=-2.65, width=6.0):
    return SpreadPosition(underlying="XYZ", strategy_name=name,
                          legs=[_leg("S", -1, 5.90), _leg("L", 1, 8.55)],
                          original_credit=credit, max_loss=265.0, spread_width=width,
                          net_unrealized_pl=pl, expiration="2026-12-18", short_strikes=[1.0],
                          net_natural_pl=pl if natural is None else natural)


def _mon(**kw):
    return PositionMonitor("k", "s", post_fill_grace_seconds=0, profit_target_basis="mid", **kw)


# ── exits by direction ────────────────────────────────────────────────────
# Debit $265: 40 % = $106, 50 % = $132.50, 80 % = $212.

def test_put_debit_takes_profit_at_the_bear_target():
    m = _mon(debit_profit_target_pct_bear=0.40)
    sig, why = m._check_debit_exit(_spread(PUT_DEBIT_STRATEGY, 110.0), {})
    assert sig == ExitSignal.PROFIT_TARGET and "40% of debit" in why
    assert m._check_debit_exit(_spread(CALL_DEBIT_STRATEGY, 110.0), {})[0] == ExitSignal.HOLD
    assert m.profit_basis(_spread(PUT_DEBIT_STRATEGY, 0.0))[2] == pytest.approx(106.0)


def test_call_debit_stop_is_wider():
    m = _mon(debit_stop_loss_pct_bull=0.80)
    assert m._check_debit_exit(_spread(CALL_DEBIT_STRATEGY, -150.0), {})[0] == ExitSignal.HOLD
    sig, why = m._check_debit_exit(_spread(CALL_DEBIT_STRATEGY, -213.0), {})
    assert sig == ExitSignal.STOP_LOSS and "80% of debit" in why
    assert m._check_debit_exit(_spread(PUT_DEBIT_STRATEGY, -150.0), {})[0] == ExitSignal.STOP_LOSS


def test_none_keeps_the_shared_values():
    m = _mon()
    assert m._check_debit_exit(_spread(CALL_DEBIT_STRATEGY, -150.0), {})[0] == ExitSignal.STOP_LOSS
    assert m._check_debit_exit(_spread(PUT_DEBIT_STRATEGY, 110.0), {})[0] == ExitSignal.HOLD


def test_calendar_untouched_by_direction_fields():
    m = _mon(debit_profit_target_pct_bear=0.4, debit_stop_loss_pct_bull=0.8)
    cal = _spread(CALENDAR_STRATEGY, -150.0, width=0.0)
    assert m._check_debit_exit(cal, {})[0] == ExitSignal.STOP_LOSS


# ── bad-quote guard ───────────────────────────────────────────────────────

@pytest.mark.parametrize("name,pl,ok", [
    (PUT_DEBIT_STRATEGY, -265.0, True),        # the whole debit — possible
    (PUT_DEBIT_STRATEGY, -555.0, False),       # GLD 10-22 09:25
    (CALL_DEBIT_STRATEGY, 340.0, True),        # max profit 335 + slack
    (CALL_DEBIT_STRATEGY, 600.0, False),
    (CALENDAR_STRATEGY, 900.0, True),          # no upper bound on a calendar
    ("Bull Put Spread", -5000.0, True),        # not a debit structure
])
def test_debit_mark_plausible(name, pl, ok):
    assert PositionMonitor.debit_mark_plausible(_spread(name, pl)) is ok


def test_bad_mark_holds_instead_of_stopping():
    sig, why = _mon()._check_debit_exit(_spread(CALL_DEBIT_STRATEGY, -1275.0), {})
    assert sig == ExitSignal.HOLD and "bad quote" in why


def test_trail_skips_before_the_exit_window(monkeypatch):
    from trading_agent.agent import TradingAgent
    import trading_agent.profit_trail as pt
    a = TradingAgent.__new__(TradingAgent)
    a.preset = SimpleNamespace(profit_trail_mode="live", no_exit_before_et="09:45")
    called = []
    monkeypatch.setattr(pt, "load_states", lambda *a, **k: called.append(1) or {})
    monkeypatch.setattr("trading_agent.entry_confirmation.before_entry_window", lambda now, nb: True)
    spreads = [SimpleNamespace(legs=[1])]
    assert a._apply_profit_trail(spreads) is spreads and called == []


# ── entry filters ─────────────────────────────────────────────────────────

LIVE = replace(PRESETS["balanced"], rs_filter_mode="off", pullback_filter_mode="off",
               trend200_filter_mode="live", vol_regime_filter_mode="live")


@pytest.mark.parametrize("strategy,price,sma200,rank,blocked", [
    (CALL_DEBIT_STRATEGY, 110, 100, 50, None),
    (CALL_DEBIT_STRATEGY, 90, 100, 50, "trend200"),
    (CALL_DEBIT_STRATEGY, 110, 100, 85, "vol_regime"),        # bull cap 80
    (PUT_DEBIT_STRATEGY, 90, 100, 45, None),
    (PUT_DEBIT_STRATEGY, 110, 100, 45, "trend200"),
    (PUT_DEBIT_STRATEGY, 90, 100, 60, "vol_regime"),          # bear cap 50
    (PUT_DEBIT_STRATEGY, 90, None, None, None),               # missing data never blocks
    (PUT_DEBIT_STRATEGY, 90, float("nan"), 0.0, None),
])
def test_debit_filters(strategy, price, sma200, rank, blocked):
    ev = ef.evaluate(strategy, LIVE, price=price, sma200=sma200, iv_rank=rank)
    if blocked is None:
        assert ev.block is None
    else:
        assert ev.block.startswith(f"trend: {blocked}")


def test_credit_spreads_are_not_filtered_by_trend200_or_vol():
    ev = ef.evaluate("Bull Put Spread", LIVE, price=90, sma200=100, iv_rank=99)
    assert ev is None


def test_shadow_records_without_blocking():
    shadow = replace(LIVE, trend200_filter_mode="shadow", vol_regime_filter_mode="shadow")
    ev = ef.evaluate(PUT_DEBIT_STRATEGY, shadow, price=110, sma200=100, iv_rank=90)
    assert ev.block is None
    assert [c["ok"] for c in ev.to_journal_dict()["checks"]] == [False, False]


# ── preset ────────────────────────────────────────────────────────────────

def test_preset_fields_load_validate_and_summarize(tmp_path):
    fp = tmp_path / "STRATEGY_PRESET.json"
    fp.write_text(json.dumps({"profile": "custom", "custom": {
        "debit_profit_target_pct_bear": 0.4, "debit_stop_loss_pct_bull": 0.8,
        "trend200_filter_mode": "live", "vol_regime_filter_mode": "live"}}))
    p = load_active_preset(fp)
    assert (p.debit_profit_target_pct_bear, p.debit_stop_loss_pct_bull) == (0.4, 0.8)
    line = p.to_summary_line()
    assert "bearTP40%" in line and "bullSL80%" in line
    assert "Trend200 live" in line and "VolRegime live bull≤80/bear≤50" in line
    fp.write_text(json.dumps({"profile": "custom", "custom": {
        "debit_stop_loss_pct_bull": 3, "trend200_filter_mode": "sometimes"}}))
    p = load_active_preset(fp)
    assert p.debit_stop_loss_pct_bull is None and p.trend200_filter_mode == "shadow"


# ── calendar far leg ──────────────────────────────────────────────────────

def test_calendar_far_leg_snaps_to_the_holiday_thursday(monkeypatch):
    from trading_agent.strategy import StrategyPlanner
    from trading_agent.decision_engine import ChainSlice
    asked = []
    p = StrategyPlanner(SimpleNamespace(), preset=PRESETS["balanced"])
    monkeypatch.setattr(p, "_expiration_near", lambda d: "2026-11-27")
    monkeypatch.setattr(p, "_typed_slice",
                        lambda t, exp, opt: asked.append(exp) or ChainSlice(exp, 30, [{"strike": 1}]))
    import trading_agent.decision_engine as de
    monkeypatch.setattr(de, "decide_calendar", lambda inp: de.DebitDecisionOutput())
    p._plan_calendar("DIA", SimpleNamespace(regime=SimpleNamespace(value="sideways"),
                                             current_price=500.0))
    assert asked == ["2026-11-27", "2026-12-24"]          # 11-27 + 28 = 12-25 (holiday)
