"""2026-11-01 — fit-to-budget debit widths, earnings exit, market-state
live prices, one gamma for net and flip (skills 59, 65, 58, 68)."""
from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from trading_agent import gamma_exposure as gx
from trading_agent import market_state as ms
from trading_agent import position_monitor as pm
from trading_agent.calendar_utils import sessions_before
from trading_agent.debit_policy import CALENDAR_STRATEGY, CALL_DEBIT_STRATEGY
from trading_agent.decision_engine import ChainSlice, DecisionInput, decide_debit_spread
from trading_agent.position_monitor import (
    IMMEDIATE_EXIT_SIGNALS, ExitSignal, PositionMonitor, PositionSnapshot, SpreadPosition,
)
from trading_agent.strategy_presets import PRESETS

from tests.test_debit_strategies import _calls

GRID = replace(PRESETS["balanced"], max_leg_spread_cents=1.0, max_leg_spread_pct_mid=1.0,
               width_grid_pct=(0.03, 0.05), fit_to_budget_enabled=True)


def _decide(budget=None, preset=GRID, chain=None):
    return decide_debit_spread(DecisionInput(
        side="call_debit", preset=preset, max_debit=budget,
        chain_slices=[ChainSlice("2026-11-06", 30, chain or _calls())]))


def _debit_at(width):
    """Debit of the 100 / 100+width call spread on the test chain."""
    out = decide_debit_spread(DecisionInput(
        side="call_debit", chain_slices=[ChainSlice("2026-11-06", 30, _calls())],
        preset=replace(GRID, width_grid_pct=(width / 100.0,))))
    return out.candidates[0].debit


# ── fit to budget ─────────────────────────────────────────────────────────

def test_no_budget_keeps_the_grid():
    out = _decide()
    assert [c.width for c in out.candidates] == [3.0, 5.0]
    assert out.diagnostics.fit_to_budget is None


def test_budget_the_grid_meets_changes_nothing():
    out = _decide(budget=_debit_at(3) + 0.01)
    assert [c.width for c in out.candidates] == [3.0, 5.0]
    assert out.diagnostics.fit_to_budget is None


def test_over_budget_takes_the_widest_narrower_width_that_fits():
    budget = (_debit_at(1) + _debit_at(2)) / 2          # 1-wide fits, 2-wide does not
    out = _decide(budget=budget)
    assert [(c.long_strike, c.width) for c in out.candidates] == [(100.0, 1.0)]
    fit = out.diagnostics.fit_to_budget
    assert fit["width"] == 1.0 and fit["debit"] <= budget
    assert fit["grid_min_debit"] == pytest.approx(_debit_at(3), abs=1e-4)
    assert _decide(budget=_debit_at(2) + 0.005).candidates[0].width == 2.0


def test_off_or_too_small_leaves_the_over_budget_grid_for_the_risk_check():
    off = _decide(budget=0.01, preset=replace(GRID, fit_to_budget_enabled=False))
    tiny = _decide(budget=0.01)
    for out in (off, tiny):
        assert [c.width for c in out.candidates] == [3.0, 5.0]
        assert out.diagnostics.fit_to_budget is None


def test_narrow_width_still_needs_reward_risk():
    strict = replace(GRID, debit_min_reward_risk=5.0)
    out = _decide(budget=0.01 + _debit_at(1), preset=strict)
    assert out.diagnostics.fit_to_budget is None


def test_planner_passes_the_budget_per_share(monkeypatch):
    from trading_agent import decision_engine
    from trading_agent.strategy import StrategyPlanner
    seen = {}

    def fake(inp, **kw):
        seen["max_debit"] = inp.max_debit
        return decision_engine.DebitDecisionOutput()
    monkeypatch.setattr(decision_engine, "decide_debit_spread", fake)
    p = StrategyPlanner(SimpleNamespace(), preset=GRID)
    monkeypatch.setattr(p, "_expiration_near", lambda d: "2026-11-27")
    monkeypatch.setattr(p, "_typed_slice", lambda *a: ChainSlice("2026-11-27", 30, []))
    p.max_loss_budget = 285.5
    p._plan_debit_spread("QQQ", SimpleNamespace(regime=SimpleNamespace(value="bearish")), "put_debit")
    assert seen["max_debit"] == pytest.approx(2.855)
    p.max_loss_budget = None
    p._plan_debit_spread("QQQ", SimpleNamespace(regime=SimpleNamespace(value="bearish")), "put_debit")
    assert seen["max_debit"] is None


def test_preset_summary_mentions_both_features():
    line = PRESETS["balanced"].to_summary_line()
    assert "Fit to budget" in line and "Earnings exit 1 sess before" in line


# ── earnings exit ─────────────────────────────────────────────────────────

WED, THU, MON = date(2026, 10, 28), date(2026, 10, 29), date(2026, 10, 26)


def _spread(name=CALL_DEBIT_STRATEGY, exp="2026-11-06", legs=None, ticker="AMZN", pl=-40.0):
    return SpreadPosition(underlying=ticker, strategy_name=name, legs=legs or [],
                          original_credit=-4.6, max_loss=460.0, spread_width=10.0,
                          net_unrealized_pl=pl, expiration=exp, short_strikes=[265.0])


def _mon(report=THU, enabled=True, sessions=1):
    m = PositionMonitor("k", "s", post_fill_grace_seconds=0,
                        earnings_exit_enabled=enabled, earnings_exit_sessions=sessions)
    m.earnings_lookup = (report if callable(report) else (lambda t: report))
    return m


def test_sessions_before():
    assert sessions_before(WED, THU) == 1
    assert sessions_before(MON, THU) == 3
    assert sessions_before(THU, THU) == 0
    assert sessions_before(date(2026, 10, 31), date(2026, 11, 2)) == 0     # Sat → Mon


def test_closes_on_the_last_session_before_a_report_it_spans():
    sig, why = _mon()._check_earnings_exit(_spread(), today=WED)
    assert sig == ExitSignal.EARNINGS_EXIT and "2026-10-29" in why
    assert ExitSignal.EARNINGS_EXIT in IMMEDIATE_EXIT_SIGNALS


@pytest.mark.parametrize("mon_kw,spread_kw,today", [
    ({"enabled": False}, {}, WED),
    ({"report": None}, {}, WED),                       # ETF / unknown date
    ({}, {"exp": "2026-10-23"}, WED),                  # expires before the report
    ({}, {}, THU),                                     # report day — already out or pricing in
    ({}, {}, MON),                                     # 3 sessions out, window is 1
])
def test_no_earnings_exit(mon_kw, spread_kw, today):
    assert _mon(**mon_kw)._check_earnings_exit(_spread(**spread_kw), today=today) is None


def test_window_of_three_sessions():
    assert _mon(sessions=3)._check_earnings_exit(_spread(), today=MON)[0] == ExitSignal.EARNINGS_EXIT


def test_failed_lookup_never_exits():
    def boom(t):
        raise RuntimeError("rpc")
    assert _mon(report=boom)._check_earnings_exit(_spread(), today=WED) is None


def test_calendar_far_leg_counts():
    leg = lambda sym: PositionSnapshot(sym, 1, "long", 1.0, 1.0, 100.0, 100.0, 0.0, 0.0, "us_option")
    cal = _spread(name=CALENDAR_STRATEGY, exp="2026-10-23",
                  legs=[leg("AMZN261023C00255000"), leg("AMZN261120C00255000")])
    assert _mon()._check_earnings_exit(cal, today=WED)[0] == ExitSignal.EARNINGS_EXIT


def test_check_exit_runs_it_before_the_pnl_rules_and_skips_wheel(monkeypatch):
    monkeypatch.setattr(pm, "_today_et", lambda: WED)
    m = _mon()
    assert m._check_exit(_spread(pl=-400.0), {}, 255.0)[0] == ExitSignal.EARNINGS_EXIT
    csp = _spread(name="Cash-Secured Put")
    assert m._check_exit(csp, {}, 255.0)[0] != ExitSignal.EARNINGS_EXIT


# ── market state: today's live price ──────────────────────────────────────

def _frame(closes, last_day="2026-10-29"):
    idx = pd.bdate_range(end=last_day, periods=len(closes))
    return pd.DataFrame({"Close": closes}, index=idx)


def test_closes_live_appends_only_when_the_bar_is_old():
    df = _frame([100.0] * 5)
    assert ms._closes_live(df, 99.0, date(2026, 10, 30)) == ([100.0] * 5 + [99.0], True)
    assert ms._closes_live(df, 99.0, date(2026, 10, 29)) == ([100.0] * 5, False)
    assert ms._closes_live(df, None, date(2026, 10, 30)) == ([100.0] * 5, False)
    assert ms._closes_live(df, 99.0, None) == ([100.0] * 5, False)


def test_compute_inputs_reads_today_for_spy_and_breadth():
    up = _frame([100.0 + i * 0.1 for i in range(250)])           # last close 124.9, SMA50 ≈ 122.5

    def hist(t):
        return up
    level = lambda s: 15.0
    stale = ms.compute_inputs(hist, ["SPY", "XLK"], level)
    assert stale.spy_price == pytest.approx(124.9) and not stale.live_overlay
    assert stale.breadth == 1.0
    live = ms.compute_inputs(hist, ["SPY", "XLK"], level,
                             live_spots={"SPY": 118.0, "XLK": 118.0}, today=date(2026, 10, 30))
    assert live.spy_price == 118.0 and live.live_overlay
    assert live.breadth == 0.0                                   # both now under their SMA50


def test_agent_spots_only_in_session(monkeypatch):
    from trading_agent import market_hours
    from trading_agent.agent import TradingAgent
    agent = SimpleNamespace(data_provider=SimpleNamespace(
        fetch_batch_snapshots=lambda syms: {s: 100.0 for s in syms}),
        _now_et=TradingAgent._now_et,
        preset=SimpleNamespace(market_state_live_prices=True))
    monkeypatch.setattr(market_hours, "is_within_market_hours", lambda: False)
    assert TradingAgent._market_state_spots(agent, ["XLK"]) == ({}, None)
    monkeypatch.setattr(market_hours, "is_within_market_hours", lambda: True)
    agent.preset = SimpleNamespace(market_state_live_prices=False)       # default: daily closes
    assert TradingAgent._market_state_spots(agent, ["XLK"]) == ({}, None)
    agent.preset = SimpleNamespace(market_state_live_prices=True)
    spots, today = TradingAgent._market_state_spots(agent, ["XLK", "SPY"])
    assert spots == {"SPY": 100.0, "XLK": 100.0} and today is not None

    def boom(syms):
        raise RuntimeError("rpc")
    agent.data_provider.fetch_batch_snapshots = boom
    assert TradingAgent._market_state_spots(agent, ["XLK"]) == ({}, None)


# ── gamma: one Γ for net and flip ─────────────────────────────────────────

TODAY = date(2026, 10, 28)


def _c(kind, strike, gamma=0.0, oi=1000, iv=0.2, exp="2026-11-20"):
    return {"type": kind, "strike": strike, "gamma": gamma, "open_interest": oi,
            "iv": iv, "expiration": exp}


def test_contract_gamma_is_bs_when_iv_known():
    c = _c("call", 110, gamma=0.0)                 # Schwab rounded a far strike to 0.000
    g = gx.contract_gamma(c, 100.0, TODAY)
    assert g == pytest.approx(gx.bs_gamma(100.0, 110, 23 / 365, 0.2)) and g > 0
    assert gx.contract_gamma({**c, "iv": 0, "gamma": 0.013}, 100.0, TODAY) == 0.013
    assert gx.contract_gamma(c, 100.0) == 0.0      # no date → quoted Γ


def test_summary_net_is_the_flip_curve_at_spot():
    chain = [_c("put", k, oi=4000) for k in (90, 94, 97)] + [_c("call", k, oi=3000) for k in (103, 106, 110)]
    row = gx.summarize("SPY", chain, 100.0, TODAY)
    net_at_spot = gx.net_gex_at(chain, 100.0, TODAY)
    assert row["net_gex"] == round(net_at_spot)
    assert row["regime"] == ("long_gamma" if net_at_spot > 0 else "short_gamma")
    assert row["gamma_source"] == "bs_iv"
