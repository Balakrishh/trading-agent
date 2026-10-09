"""Backlog §6.3–6.5 — debit spreads, calendars, bounce bull put (skill 59)."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trading_agent.chain_scanner import (
    _quote_debit, debit_ceiling, debit_mid_value,
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


def test_mid_value_and_ceiling():
    assert debit_mid_value(2.00, 2.10, 0.60, 0.70) == pytest.approx(1.40)
    assert debit_mid_value(0.0, 2.10, 0.60, 0.70) == pytest.approx(1.45)   # no bid → ask
    assert debit_ceiling(2.25, 0.05) == pytest.approx(2.3625)


# ── vertical debit scorer ─────────────────────────────────────────────────

def score(debit, width=5.0, **kw):
    args = dict(mid_value=2.25, long_delta=0.60, short_delta=0.30, dte=30,
                max_overpay=0.05, min_reward_risk=1.0)
    args.update(kw)
    return _score_debit_spread_with_reason(debit=debit, width=width, **args)


def test_debit_scorer_accepts_near_fair():
    r = score(2.20)
    assert r["status"] == "accepted"
    assert r["max_profit"] == pytest.approx(2.80)
    assert r["rr"] == pytest.approx(2.80 / 2.20)
    assert r["pop"] == pytest.approx(0.60 - 0.30 * 0.44)       # |Δ| at breakeven
    assert r["ev"] == pytest.approx((5.0 * 0.45 - 2.20) / 2.20)  # delta model, reported only


@pytest.mark.parametrize("debit,kw,reason", [
    (0.0, {}, DEBIT_REJECT_NON_POSITIVE),
    (5.0, {}, DEBIT_REJECT_GE_WIDTH),
    (2.40, {}, DEBIT_REJECT_ABOVE_FAIR),                      # > 2.3625
    (2.30, {"min_reward_risk": 1.5}, DEBIT_REJECT_REWARD_RISK),
])
def test_debit_scorer_rejects(debit, kw, reason):
    assert score(debit, **kw)["reason"] == reason


def _chain(opt, spot=100.0, dte=30, sigma=0.20, half_spread=0.01):
    """Black-Scholes-consistent chain: strikes 90–110, bid/ask = price ∓ a penny."""
    import math
    t = dte / 365.0
    rows = []
    for k in range(90, 111):
        price = _bs_price(spot, k, t, sigma, opt)
        d1 = (math.log(spot / k) + 0.5 * sigma * sigma * t) / (sigma * math.sqrt(t))
        nd1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        delta = nd1 if opt == "call" else nd1 - 1.0
        rows.append({"symbol": f"{opt[0].upper()}{k}", "strike": float(k), "delta": round(delta, 4),
                     "bid": round(max(0.01, price - half_spread), 2),
                     "ask": round(price + half_spread, 2), "iv": sigma, "type": opt})
    return rows


def _calls():
    return _chain("call")


def _puts():
    return _chain("put")


def _q(chain, k):
    return next(c for c in chain if c["strike"] == k)


WIDE = replace(PRESET, width_grid_pct=(0.02, 0.05))


def test_decide_call_debit_buys_atm_smallest_debit_first():
    chain = _calls()
    out = decide_debit_spread(DecisionInput(
        side="call_debit", preset=WIDE, chain_slices=[ChainSlice("2026-11-06", 30, chain)]))
    assert [(c.long_strike, c.short_strike) for c in out.candidates] == [(100.0, 102.0), (100.0, 105.0)]
    c = out.candidates[0]
    assert c.debit == pytest.approx(_q(chain, 100)["ask"] - _q(chain, 102)["bid"])   # natural
    mid = debit_mid_value(_q(chain, 100)["bid"], _q(chain, 100)["ask"],
                          _q(chain, 102)["bid"], _q(chain, 102)["ask"])
    assert c.fair_value == pytest.approx(mid, abs=1e-4)
    assert c.max_debit == pytest.approx(mid * 1.05, abs=1e-4) and c.width == 2.0


def test_decide_put_debit_sells_below():
    out = decide_debit_spread(DecisionInput(
        side="put_debit", preset=WIDE, chain_slices=[ChainSlice("2026-11-06", 30, _puts())]))
    c = out.candidates[0]
    assert (c.long_strike, c.short_strike) == (100.0, 98.0) and c.option_type == "put"


def test_decide_debit_reports_near_miss_when_overpriced():
    rich = [dict(c, ask=c["ask"] + 0.5) if c["strike"] == 100 else c for c in _calls()]
    out = decide_debit_spread(DecisionInput(
        side="call_debit", preset=WIDE, chain_slices=[ChainSlice("2026-11-06", 30, rich)]))
    assert out.candidates == []
    assert out.diagnostics.rejects_by_reason == {DEBIT_REJECT_ABOVE_FAIR: 2}
    assert out.diagnostics.best_near_miss["width"] == 5.0          # best reward/risk of the misses


# ── calendars ─────────────────────────────────────────────────────────────

def test_bs_price_atm():
    assert _bs_price(100, 100, 30 / 365, 0.20, "call") == pytest.approx(2.287, abs=1e-3)
    assert _bs_price(100, 90, 0, 0.20, "put") == 0.0


def cal(debit, mid_value=1.0, near_iv=0.20, far_iv=0.20, **kw):
    args = dict(spot=100.0, strike=100.0, option_type="call", near_dte=21, far_dte=49,
                max_overpay=0.05, min_reward_risk=1.0)
    args.update(kw)
    return _score_calendar_with_reason(debit=debit, mid_value=mid_value,
                                       near_iv=near_iv, far_iv=far_iv, **args)


def test_calendar_flat_vol_model_matches_no_arbitrage():
    r = cal(1.0)
    far, near = _bs_price(100, 100, 49 / 365, .2, "call"), _bs_price(100, 100, 21 / 365, .2, "call")
    assert r["model"] == pytest.approx(far - near, abs=0.02)
    assert r["status"] == "accepted" and r["fair"] == 1.0         # the market mid sets the level


def test_calendar_model_shape_responds_to_term_structure():
    assert cal(1.0, near_iv=0.30)["model"] < cal(1.0, near_iv=0.30, far_iv=0.30)["model"]
    assert cal(1.0, far_iv=0.25)["model"] > cal(1.0)["model"]


def test_calendar_rescales_to_mid():
    """Zero-rate BS under-prices calendars (carry); the payoff is rescaled
    so max profit is quoted relative to the market's own level."""
    a, b = cal(1.0, mid_value=1.0), cal(2.0, mid_value=2.0)
    assert a["rr"] == pytest.approx(b["rr"], rel=1e-6)


@pytest.mark.parametrize("kw,reason", [
    ({"debit": 0.0}, DEBIT_REJECT_NON_POSITIVE),
    ({"debit": 1.0, "near_iv": 0.0}, CAL_REJECT_IV_MISSING),
    ({"debit": 1.06}, DEBIT_REJECT_ABOVE_FAIR),                   # > mid 1.00 × 1.05
])
def test_calendar_rejects(kw, reason):
    assert cal(**kw)["reason"] == reason


def _cal_slice(exp, dte, iv, prices):
    return ChainSlice(exp, dte, [
        {"symbol": f"{exp}C{k}", "strike": float(k), "delta": 0.5, "bid": b, "ask": a,
         "iv": iv, "type": "call"} for k, (b, a) in prices.items()])


def test_decide_calendar_uses_strike_nearest_spot():
    near = _cal_slice("2026-10-26", 21, 0.20, {100: (1.88, 1.90), 105: (0.4, 0.45)})
    far = _cal_slice("2026-11-23", 49, 0.20, {100: (2.95, 2.97), 105: (1.2, 1.3)})
    out = decide_calendar(DecisionInput(side="calendar", chain_slices=[near, far],
                                        preset=PRESET, spot=100.4))
    c = out.candidates[0]
    assert c.long_strike == c.short_strike == 100.0
    assert c.debit == pytest.approx(2.97 - 1.88) and c.fair_value == pytest.approx(1.07)
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
        side="call_debit", preset=WIDE,
        chain_slices=[ChainSlice("2026-11-06", 30, _calls())])).candidates[0]
    return build_debit_plan(ticker="SPY", regime="bullish", cand=c)


def test_build_debit_plan_sign_convention():
    p = _call_debit_plan()
    debit = round(_q(_calls(), 100)["ask"] - _q(_calls(), 102)["bid"], 2)
    assert p.strategy_name == CALL_DEBIT_STRATEGY and is_debit_plan(p)
    assert p.net_credit == pytest.approx(-debit) and p.max_loss == pytest.approx(debit * 100)
    assert [(l.action, l.strike) for l in p.legs] == [("buy", 100.0), ("sell", 102.0)]
    assert p.to_dict()["max_debit"] == p.max_debit


def _rm():
    return RiskManager(max_risk_pct=0.02, max_delta=0.25, delta_aware_floor=True)


def test_risk_manager_debit_branch():
    p = _call_debit_plan()
    v = _rm().evaluate(p, 30_000, "paper", market_open=True)
    assert v.approved, v.checks_failed                    # short |Δ| .37 > .25 not checked
    assert any(f"Debit ${-p.net_credit:.2f} ≤ cap" in c for c in v.checks_passed)
    p.max_debit = round(-p.net_credit - 0.05, 2)
    v = _rm().evaluate(p, 30_000, "paper", market_open=True)
    assert not v.approved and "outside" in v.checks_failed[0]


def test_sizing_uses_debit_as_max_loss():
    p = _call_debit_plan()
    debit = -p.net_credit
    assert calculate_position_qty(p, 30_000, 0.02) == int(600 // (debit * 100))
    assert calculate_position_qty(p, 30_000, 0.02, live_credit=-4.00) == 1


def test_executor_debit_recheck_and_positive_limit(tmp_path):
    chain = _calls()
    provider = MagicMock()
    provider.fetch_option_quotes.return_value = {
        "C100": {"bid": _q(chain, 100)["bid"], "ask": _q(chain, 100)["ask"]},
        "C102": {"bid": _q(chain, 102)["bid"], "ask": _q(chain, 102)["ask"]}}
    ex = OrderExecutor("k", "s", trade_plan_dir=str(tmp_path), dry_run=False,
                       data_provider=provider, max_risk_pct=0.02)
    p = _call_debit_plan()
    debit = -p.net_credit
    sent = {}

    def capture(**kw):
        sent.update(kw["order_payload"])
        return {"status": "submitted"}
    with patch.object(ex, "_submit_order_with_idempotency", side_effect=capture):
        ex._submit_order(p, str(tmp_path / "p.json"), "run1", 30_000)
    assert float(sent["limit_price"]) == pytest.approx(debit)      # debit → positive
    assert sent["qty"] == str(int(600 // (debit * 100)))
    ok, why = ex._recheck_live_economics(p, -(p.max_debit + 0.05), 30_000)
    assert not ok and "cap" in why


# ── monitor exits ─────────────────────────────────────────────────────────

def _pos(name, credit, width, pl, natural=None, contracts=1, exp=None):
    s = SpreadPosition(underlying="SPY", strategy_name=name, legs=[],
                       original_credit=credit, max_loss=-credit * 100, spread_width=width,
                       net_unrealized_pl=pl, expiration=exp or (date.today() + timedelta(days=20)).isoformat(),
                       short_strikes=[103.0], contracts_open=contracts)
    s.net_natural_pl = natural
    return s


def mon(**kw):
    return PositionMonitor("k", "s", post_fill_grace_seconds=0, **kw)


@pytest.mark.parametrize("name,credit,width,pl,expected", [
    (CALL_DEBIT_STRATEGY, -2.0, 5.0, -100.0, ExitSignal.STOP_LOSS),     # 50 % of $200
    (CALL_DEBIT_STRATEGY, -2.0, 5.0, 100.0, ExitSignal.PROFIT_TARGET),  # 50 % of the $200 debit
    (CALL_DEBIT_STRATEGY, -2.0, 5.0, 95.0, ExitSignal.HOLD),
    (CALENDAR_STRATEGY, -1.0, 0.0, 25.0, ExitSignal.PROFIT_TARGET),     # 25 % of $100
    (CALENDAR_STRATEGY, -1.0, 0.0, -45.0, ExitSignal.HOLD),
])
def test_debit_exit_rules(name, credit, width, pl, expected):
    sig, _ = mon()._check_exit(_pos(name, credit, width, pl), {}, underlying_price=103.0)
    assert sig == expected


@pytest.mark.parametrize("pl,expected", [
    (150.0, ExitSignal.PROFIT_TARGET),                                  # 50 % of $300 max profit
    (140.0, ExitSignal.HOLD),
])
def test_debit_target_on_max_profit_basis(pl, expected):
    m = mon(debit_profit_target_basis="max_profit")
    sig, why = m._check_exit(_pos(CALL_DEBIT_STRATEGY, -2.0, 5.0, pl), {}, underlying_price=103.0)
    assert sig == expected
    if sig == ExitSignal.PROFIT_TARGET:
        assert "of max profit $300.00" in why


def test_debit_target_on_cost_scales_with_contracts_and_labels_the_debit():
    # IWM 2026-10-08: 4 × 1.92 debit = $768 → target $384 (old rule: $616).
    pos = _pos(PUT_DEBIT_STRATEGY, -1.92, 5.0, 444.0, contracts=4)
    kind, basis, target = mon().profit_basis(pos)
    assert kind == "debit_vertical"
    assert basis == pytest.approx(1232.0)       # max profit stays the trail's yardstick
    assert target == pytest.approx(384.0)
    sig, why = mon()._check_exit(pos, {}, underlying_price=273.0)
    assert sig == ExitSignal.PROFIT_TARGET and "of debit $768.00" in why


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


@pytest.mark.parametrize("name,regime,expected", [
    (PUT_DEBIT_STRATEGY, Regime.SIDEWAYS, ExitSignal.HOLD),       # drift, not reversal
    (PUT_DEBIT_STRATEGY, Regime.MEAN_REVERSION, ExitSignal.HOLD),
    (PUT_DEBIT_STRATEGY, Regime.BULLISH, ExitSignal.REGIME_SHIFT),
    (CALL_DEBIT_STRATEGY, Regime.SIDEWAYS, ExitSignal.HOLD),
    (CALL_DEBIT_STRATEGY, Regime.BEARISH, ExitSignal.REGIME_SHIFT),
    (CALENDAR_STRATEGY, Regime.BULLISH, ExitSignal.REGIME_SHIFT),  # a trend threatens a calendar
])
def test_debit_regime_exit_only_on_reversal(name, regime, expected):
    """2026-10-05: IWM put debit voted to close one cycle after filling when
    the regime flickered bearish → sideways."""
    width = 0.0 if name == CALENDAR_STRATEGY else 5.0
    sig, _ = mon()._check_exit(_pos(name, -2.0, width, 0.0), {"SPY": regime},
                               underlying_price=103.0)
    assert sig == expected


# ── 2026-10-06: calendar exits on a real trend / strike drift only ────────

@pytest.mark.parametrize("regime,price,expected", [
    (Regime.MEAN_REVERSION, 103.0, ExitSignal.HOLD),          # one-cycle band touch: hold
    (Regime.SIDEWAYS, 103.0, ExitSignal.HOLD),
    (Regime.BEARISH, 103.0, ExitSignal.REGIME_SHIFT),         # a real trend breaks the range
    (Regime.SIDEWAYS, 106.2, ExitSignal.STRIKE_DRIFT),        # 3.1 % from the 103 strike
    (Regime.SIDEWAYS, 105.9, ExitSignal.HOLD),                # 2.8 %: still inside
])
def test_calendar_exit_rules(regime, price, expected):
    sig, reason = mon()._check_exit(_pos(CALENDAR_STRATEGY, -2.0, 0.0, 0.0), {"SPY": regime},
                                    underlying_price=price)
    assert sig == expected, reason


def test_calendar_drift_disabled_and_stops_still_win():
    m = PositionMonitor("k", "s", post_fill_grace_seconds=0, calendar_max_strike_drift_pct=0.0)
    assert m._check_exit(_pos(CALENDAR_STRATEGY, -2.0, 0.0, 0.0), {}, underlying_price=120.0)[0] \
        == ExitSignal.HOLD
    sig, _ = mon()._check_exit(_pos(CALENDAR_STRATEGY, -2.0, 0.0, -120.0), {}, underlying_price=120.0)
    assert sig == ExitSignal.STOP_LOSS                         # 50 % of the debit first


def test_debit_target_basis_preset_default_summary_and_bad_value():
    from trading_agent.strategy_presets import _make_custom
    assert PRESETS["balanced"].debit_profit_target_basis == "debit"
    assert "TP50% of cost" in PRESETS["balanced"].to_summary_line()
    legacy = replace(PRESETS["balanced"], debit_profit_target_basis="max_profit")
    assert "TP50% of max" in legacy.to_summary_line()
    assert _make_custom({"debit_profit_target_basis": "max_profit"}).debit_profit_target_basis == "max_profit"
    assert _make_custom({"debit_profit_target_basis": "bogus"}).debit_profit_target_basis == "debit"
    assert _make_custom({}).debit_profit_target_basis == "debit"   # older preset files
