"""Backlog §6.3–6.5 — debit spreads, calendars, bounce bull put (skill 59)."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trading_agent.chain_scanner import (
    _quote_debit, debit_spread_ceiling, debit_spread_fair_value,
)
from trading_agent.debit_policy import (
    BOUNCE_BULL_PUT_STRATEGY, CALENDAR_STRATEGY, CALL_DEBIT_STRATEGY,
    PUT_DEBIT_STRATEGY, build_debit_plan, is_debit_plan,
)
from trading_agent.decision_engine import (
    DEBIT_REJECT_ABOVE_FAIR, DEBIT_REJECT_GE_WIDTH, DEBIT_REJECT_NON_POSITIVE,
    DEBIT_REJECT_REWARD_RISK, CAL_REJECT_IV_MISSING, CAL_REJECT_NEED_TWO_SLICES,
    CAL_REJECT_NO_SPOT, ChainSlice, DecisionInput, _bs_price,
    _score_calendar_with_reason, _score_debit_spread_with_reason,
    decide_calendar, decide_debit_spread,
)
from trading_agent.executor import OrderExecutor, calculate_position_qty
from trading_agent.position_monitor import (
    ExitSignal, PositionMonitor, PositionSnapshot, SpreadPosition,
)
from trading_agent.regime import Regime
from trading_agent.risk_manager import RiskManager
from trading_agent.strategy_presets import PRESETS

PRESET = replace(PRESETS["balanced"], max_leg_spread_cents=1.0, max_leg_spread_pct_mid=1.0)


# ── pricing helpers ────────────────────────────────────────────────────────

def test_quote_debit_natural_and_mid():
    assert _quote_debit(2.00, 2.10, 0.60, 0.70, model="natural") == 1.50   # 2.10 − 0.60
    assert _quote_debit(2.00, 2.10, 0.60, 0.70, model="mid") == 1.42       # 2.05 − 0.65 + 0.02
    assert _quote_debit(0.50, 0.60, 0.80, 0.90, model="natural") == 0.0    # never negative


def test_fair_value_and_ceiling():
    assert debit_spread_fair_value(5.0, 0.60, -0.30) == pytest.approx(2.25)
    assert debit_spread_ceiling(5.0, 0.60, 0.30, 0.05) == pytest.approx(2.3625)


# ── vertical debit scorer ─────────────────────────────────────────────────

def score(debit, width=5.0, **kw):
    args = dict(long_delta=0.60, short_delta=0.30, dte=30, max_overpay=0.05,
                min_reward_risk=1.0)
    args.update(kw)
    return _score_debit_spread_with_reason(debit=debit, width=width, **args)


def test_debit_scorer_accepts_near_fair():
    r = score(2.20)
    assert r["status"] == "accepted"
    assert r["max_profit"] == pytest.approx(2.80)
    assert r["rr"] == pytest.approx(2.80 / 2.20)
    assert r["pop"] == pytest.approx(0.60 - 0.30 * 0.44)       # |Δ| at breakeven
    assert r["ev"] == pytest.approx((2.25 - 2.20) / 2.20)


@pytest.mark.parametrize("debit,kw,reason", [
    (0.0, {}, DEBIT_REJECT_NON_POSITIVE),
    (5.0, {}, DEBIT_REJECT_GE_WIDTH),
    (2.40, {}, DEBIT_REJECT_ABOVE_FAIR),                      # > 2.3625
    (2.30, {"min_reward_risk": 1.5}, DEBIT_REJECT_REWARD_RISK),
])
def test_debit_scorer_rejects(debit, kw, reason):
    assert score(debit, **kw)["reason"] == reason


def _calls(spot=100.0):
    # strike → (delta, bid, ask)
    rows = {95: (0.75, 6.0, 6.1), 98: (0.62, 3.9, 4.0), 100: (0.50, 2.6, 2.7),
            103: (0.33, 1.6, 1.7), 105: (0.25, 0.8, 0.9), 108: (0.15, 0.4, 0.5)}
    return [{"symbol": f"C{k}", "strike": float(k), "delta": d, "bid": b, "ask": a,
             "type": "call"} for k, (d, b, a) in rows.items()]


def _puts():
    rows = {92: (-0.15, 0.4, 0.5), 95: (-0.25, 0.8, 0.9), 97: (-0.33, 1.6, 1.7),
            100: (-0.50, 2.6, 2.7), 102: (-0.62, 3.9, 4.0)}
    return [{"symbol": f"P{k}", "strike": float(k), "delta": d, "bid": b, "ask": a,
             "type": "put"} for k, (d, b, a) in rows.items()]


def test_decide_call_debit_picks_delta_targets():
    out = decide_debit_spread(DecisionInput(
        side="call_debit", preset=PRESET,
        chain_slices=[ChainSlice("2026-11-06", 30, _calls())]))
    c = out.candidates[0]
    assert (c.long_strike, c.short_strike) == (98.0, 103.0)       # Δ .62 / .33
    assert c.debit == pytest.approx(4.0 - 1.6)                   # natural
    assert c.width == 5.0 and c.max_debit == pytest.approx(5 * 0.475 * 1.05, abs=1e-3)


def test_decide_put_debit_sells_below():
    out = decide_debit_spread(DecisionInput(
        side="put_debit", preset=PRESET,
        chain_slices=[ChainSlice("2026-11-06", 30, _puts())]))
    c = out.candidates[0]
    assert (c.long_strike, c.short_strike) == (102.0, 97.0)
    assert c.option_type == "put"


def test_decide_debit_reports_near_miss_when_overpriced():
    rich = [dict(c, ask=c["ask"] + 1.0) if c["strike"] == 98 else c for c in _calls()]
    out = decide_debit_spread(DecisionInput(
        side="call_debit", preset=PRESET, chain_slices=[ChainSlice("2026-11-06", 30, rich)]))
    assert out.candidates == []
    assert out.diagnostics.rejects_by_reason == {DEBIT_REJECT_ABOVE_FAIR: 1}
    assert out.diagnostics.best_near_miss["debit"] == pytest.approx(3.40)


# ── calendars ─────────────────────────────────────────────────────────────

def test_bs_price_atm():
    assert _bs_price(100, 100, 30 / 365, 0.20, "call") == pytest.approx(2.287, abs=1e-3)
    assert _bs_price(100, 90, 0, 0.20, "put") == 0.0


def cal(debit, near_iv=0.20, far_iv=0.20, **kw):
    args = dict(spot=100.0, strike=100.0, option_type="call", near_dte=21, far_dte=49,
                max_overpay=0.05, min_reward_risk=1.0)
    args.update(kw)
    return _score_calendar_with_reason(debit=debit, near_iv=near_iv, far_iv=far_iv, **args)


def test_calendar_flat_vol_model_matches_no_arbitrage():
    r = cal(1.0)
    far, near = _bs_price(100, 100, 49 / 365, .2, "call"), _bs_price(100, 100, 21 / 365, .2, "call")
    assert r["fair"] == pytest.approx(far - near, abs=0.02)
    assert r["status"] == "accepted"


def test_calendar_rich_near_vol_raises_model_value():
    assert cal(1.0, near_iv=0.30)["fair"] < cal(1.0, near_iv=0.30, far_iv=0.30)["fair"]
    assert cal(1.0, far_iv=0.25)["fair"] > cal(1.0)["fair"]


@pytest.mark.parametrize("kw,reason", [
    ({"debit": 0.0}, DEBIT_REJECT_NON_POSITIVE),
    ({"debit": 1.0, "near_iv": 0.0}, CAL_REJECT_IV_MISSING),
    ({"debit": 1.5}, DEBIT_REJECT_ABOVE_FAIR),
])
def test_calendar_rejects(kw, reason):
    assert cal(**kw)["reason"] == reason


def _cal_slice(exp, dte, iv, prices):
    return ChainSlice(exp, dte, [
        {"symbol": f"{exp}C{k}", "strike": float(k), "delta": 0.5, "bid": b, "ask": a,
         "iv": iv, "type": "call"} for k, (b, a) in prices.items()])


def test_decide_calendar_uses_strike_nearest_spot():
    near = _cal_slice("2026-10-26", 21, 0.20, {100: (1.85, 1.90), 105: (0.4, 0.45)})
    far = _cal_slice("2026-11-23", 49, 0.20, {100: (2.85, 2.90), 105: (1.2, 1.3)})
    out = decide_calendar(DecisionInput(side="calendar", chain_slices=[near, far],
                                        preset=PRESET, spot=100.4))
    c = out.candidates[0]
    assert c.long_strike == c.short_strike == 100.0
    assert c.debit == pytest.approx(2.90 - 1.85)
    assert (c.expiration, c.far_expiration) == ("2026-10-26", "2026-11-23")


def test_decide_calendar_needs_two_slices_and_spot():
    near = _cal_slice("2026-10-26", 21, 0.2, {100: (1.85, 1.90)})
    assert decide_calendar(DecisionInput("calendar", [near], PRESET, spot=100)) \
        .diagnostics.rejects_by_reason == {CAL_REJECT_NEED_TWO_SLICES: 1}
    assert decide_calendar(DecisionInput("calendar", [near, near], PRESET)) \
        .diagnostics.rejects_by_reason == {CAL_REJECT_NO_SPOT: 1}


# ── plan, risk, sizing ────────────────────────────────────────────────────

def _call_debit_plan():
    c = decide_debit_spread(DecisionInput(
        side="call_debit", preset=PRESET,
        chain_slices=[ChainSlice("2026-11-06", 30, _calls())])).candidates[0]
    return build_debit_plan(ticker="SPY", regime="bullish", cand=c)


def test_build_debit_plan_sign_convention():
    p = _call_debit_plan()
    assert p.strategy_name == CALL_DEBIT_STRATEGY and is_debit_plan(p)
    assert p.net_credit == pytest.approx(-2.40) and p.max_loss == pytest.approx(240.0)
    assert [(l.action, l.strike) for l in p.legs] == [("buy", 98.0), ("sell", 103.0)]
    assert p.to_dict()["max_debit"] == p.max_debit


def _rm():
    return RiskManager(max_risk_pct=0.02, max_delta=0.25, delta_aware_floor=True)


def test_risk_manager_debit_branch():
    p = _call_debit_plan()
    v = _rm().evaluate(p, 30_000, "paper", market_open=True)
    assert v.approved, v.checks_failed                    # short |Δ| .33 > .25 not checked
    assert any("Debit $2.40 ≤ cap" in c for c in v.checks_passed)
    p.max_debit = 2.30
    v = _rm().evaluate(p, 30_000, "paper", market_open=True)
    assert not v.approved and "outside" in v.checks_failed[0]


def test_sizing_uses_debit_as_max_loss():
    p = _call_debit_plan()
    assert calculate_position_qty(p, 30_000, 0.02) == 2          # 600 // 240
    assert calculate_position_qty(p, 30_000, 0.02, live_credit=-3.10) == 1


def test_executor_debit_recheck_and_positive_limit(tmp_path):
    provider = MagicMock()
    provider.fetch_option_quotes.return_value = {
        "C98": {"bid": 3.9, "ask": 4.0}, "C103": {"bid": 1.6, "ask": 1.7}}
    ex = OrderExecutor("k", "s", trade_plan_dir=str(tmp_path), dry_run=False,
                       data_provider=provider, max_risk_pct=0.02)
    p = _call_debit_plan()
    sent = {}

    def capture(**kw):
        sent.update(kw["order_payload"])
        return {"status": "submitted"}
    with patch.object(ex, "_submit_order_with_idempotency", side_effect=capture):
        ex._submit_order(p, str(tmp_path / "p.json"), "run1", 30_000)
    assert sent["limit_price"] == "2.4" and sent["qty"] == "2"   # debit → positive
    ok, why = ex._recheck_live_economics(p, -2.90, 30_000)
    assert not ok and "cap" in why                                # 2.90 > cap 2.49


# ── monitor exits ─────────────────────────────────────────────────────────

def _pos(name, credit, width, pl, natural=None, contracts=1, exp=None):
    s = SpreadPosition(underlying="SPY", strategy_name=name, legs=[],
                       original_credit=credit, max_loss=-credit * 100, spread_width=width,
                       net_unrealized_pl=pl, expiration=exp or (date.today() + timedelta(days=20)).isoformat(),
                       short_strikes=[103.0], contracts_open=contracts)
    s.net_natural_pl = natural
    return s


def mon():
    return PositionMonitor("k", "s", post_fill_grace_seconds=0)


@pytest.mark.parametrize("name,credit,width,pl,expected", [
    (CALL_DEBIT_STRATEGY, -2.0, 5.0, -100.0, ExitSignal.STOP_LOSS),     # 50 % of $200
    (CALL_DEBIT_STRATEGY, -2.0, 5.0, 150.0, ExitSignal.PROFIT_TARGET),  # 50 % of $300
    (CALL_DEBIT_STRATEGY, -2.0, 5.0, 140.0, ExitSignal.HOLD),
    (CALENDAR_STRATEGY, -1.0, 0.0, 25.0, ExitSignal.PROFIT_TARGET),     # 25 % of $100
    (CALENDAR_STRATEGY, -1.0, 0.0, -45.0, ExitSignal.HOLD),
])
def test_debit_exit_rules(name, credit, width, pl, expected):
    sig, _ = mon()._check_exit(_pos(name, credit, width, pl), {}, underlying_price=103.0)
    assert sig == expected


def test_debit_exit_never_strike_proximity_and_regime_shift():
    sig, _ = mon()._check_exit(_pos(PUT_DEBIT_STRATEGY, -2.0, 5.0, 0.0),
                               {"SPY": Regime.BEARISH}, underlying_price=103.0)
    assert sig == ExitSignal.HOLD                     # price AT the short strike
    sig, _ = mon()._check_exit(_pos(PUT_DEBIT_STRATEGY, -2.0, 5.0, 0.0),
                               {"SPY": Regime.BULLISH}, underlying_price=103.0)
    assert sig == ExitSignal.REGIME_SHIFT


def test_bounce_bull_put_never_regime_shift_closed():
    sig, _ = mon()._check_exit(_pos(BOUNCE_BULL_PUT_STRATEGY, 1.0, 5.0, 0.0),
                               {"SPY": Regime.BEARISH}, underlying_price=150.0)
    assert sig == ExitSignal.HOLD


def _leg(symbol, qty, avg):
    return PositionSnapshot(symbol=symbol, qty=qty, side="short" if qty < 0 else "long",
                            avg_entry_price=avg, current_price=avg, market_value=0.0,
                            cost_basis=0.0, unrealized_pl=0.0, unrealized_plpc=0.0,
                            asset_class="us_option")


def test_inference_debit_vertical_and_calendar():
    legs = [_leg("SPY261106C00098000", 1, 4.0), _leg("SPY261106C00103000", -1, 1.3),
            _leg("QQQ261026C00500000", -1, 5.0), _leg("QQQ261123C00500000", 1, 8.0)]
    spreads = PositionMonitor("k", "s").group_into_spreads(legs, [])
    by = {s.underlying: s for s in spreads}
    assert by["SPY"].strategy_name == CALL_DEBIT_STRATEGY
    assert by["SPY"].max_loss == pytest.approx(270.0)
    assert by["QQQ"].strategy_name == CALENDAR_STRATEGY
    assert by["QQQ"].expiration == "2026-10-26" and by["QQQ"].original_credit == -3.0


# ── planner routing ───────────────────────────────────────────────────────

def _analysis(regime, iv_rank, rsi, price=100.0):
    return SimpleNamespace(regime=regime, iv_rank=iv_rank, rsi_14=rsi, current_price=price,
                           mean_reversion_direction="", inter_market_inhibit_bullish=False,
                           leadership_zscore=0.0, leadership_anchor="", vix_zscore=0.0)


def _planner(preset=PRESET, history=None):
    from trading_agent.strategy import StrategyPlanner
    data = MagicMock()
    data.fetch_option_chain.side_effect = lambda t, e, o: _calls() if o == "call" else _puts()
    data.fetch_historical_prices.return_value = history
    return StrategyPlanner(data, preset=preset)


def test_credit_plan_wins_when_valid(monkeypatch):
    from trading_agent.strategy import SpreadPlan
    pl = _planner()
    good = SpreadPlan("SPY", "Bull Put Spread", "bullish", [], 5, 1.0, 400, 0.2, "x", "")
    monkeypatch.setattr(pl, "_plan_credit_by_regime", lambda t, a: good)
    assert pl.plan("SPY", _analysis(Regime.BULLISH, 10, 55)) is good


def test_low_vol_bullish_falls_back_to_call_debit(monkeypatch):
    from trading_agent.strategy import SpreadPlan
    pl = _planner()
    bad = SpreadPlan("SPY", "Bull Put Spread", "bullish", [], 0, 0, 0, 0, "x", "",
                     valid=False, rejection_reason="No positive-EV candidate")
    monkeypatch.setattr(pl, "_plan_credit_by_regime", lambda t, a: bad)
    p = pl.plan("SPY", _analysis(Regime.BULLISH, 10, 55))
    assert p.strategy_name == CALL_DEBIT_STRATEGY and p.valid


def test_both_fail_keeps_credit_name_with_fallback_reason(monkeypatch):
    from trading_agent.strategy import SpreadPlan
    pl = _planner(preset=replace(PRESET, debit_max_overpay=0.0, debit_min_reward_risk=3.0))
    bad = SpreadPlan("SPY", "Bull Put Spread", "bullish", [], 0, 0, 0, 0, "x", "",
                     valid=False, rejection_reason="No positive-EV candidate")
    monkeypatch.setattr(pl, "_plan_credit_by_regime", lambda t, a: bad)
    p = pl.plan("SPY", _analysis(Regime.BULLISH, 10, 55))
    assert p.strategy_name == "Bull Put Spread"
    assert "fallback Call Debit Spread: no acceptable candidate" in p.rejection_reason


def test_debit_toggle_off_keeps_legacy(monkeypatch):
    from trading_agent.strategy import SpreadPlan
    pl = _planner(preset=replace(PRESET, debit_spreads_enabled=False))
    bad = SpreadPlan("SPY", "Bull Put Spread", "bullish", [], 0, 0, 0, 0, "x", "",
                     valid=False, rejection_reason="No positive-EV candidate")
    monkeypatch.setattr(pl, "_plan_credit_by_regime", lambda t, a: bad)
    assert pl.plan("SPY", _analysis(Regime.BULLISH, 10, 55)) is bad


def _history(last_close):
    closes = [110 - i for i in range(19)] + [last_close]          # falling, then today's
    return pd.DataFrame({"Close": closes, "Low": [c - 1 for c in closes]})


def test_bounce_waits_for_stabilisation_and_never_sells_calls(monkeypatch):
    pl = _planner(history=_history(91.0))
    monkeypatch.setattr(pl, "_plan_credit_by_regime",
                        lambda t, a: pytest.fail("no bear call into an oversold market"))
    p = pl.plan("SPY", _analysis(Regime.BEARISH, 60, 25, price=92.0))
    assert p.strategy_name == "" and "waiting for stabilisation" in p.rejection_reason
    p = pl.plan("SPY", _analysis(Regime.BEARISH, 10, 25, price=92.0))
    assert p.strategy_name == "" and "wait_for_stabilization" in p.rejection_reason


def test_bounce_confirmed_sells_put_below_recent_low():
    pl = _planner(history=_history(91.0))
    pl.preset = replace(PRESET, max_delta=0.30)
    pl.max_delta = 0.30
    high_close, recent_low = pl.bounce_levels("SPY")
    assert high_close == 95.0 and recent_low == 90.0
    low_puts = [{"symbol": f"P{k}", "strike": float(k), "delta": d, "bid": b, "ask": a,
                 "mid": (b + a) / 2, "type": "put"}
                for k, d, b, a in [(80, -0.05, 0.10, 0.15), (85, -0.12, 0.60, 0.65),
                                   (88, -0.22, 1.40, 1.45), (89, -0.26, 1.70, 1.75),
                                   (92, -0.35, 2.60, 2.70)]]
    pl.data.fetch_option_chain.side_effect = lambda t, e, o: low_puts
    p = pl.plan("SPY", _analysis(Regime.BEARISH, 60, 25, price=96.0))
    assert p.strategy_name == BOUNCE_BULL_PUT_STRATEGY
    sold = [l.strike for l in p.legs if l.action == "sell"]
    assert sold and max(sold) < recent_low                      # never the 92 put
    assert p.reasoning.startswith("Bounce: price 96.00 reclaimed")
