"""2026-10-05 — trailing profit-taking (arm at target, ceiling, giveback, time)."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from trading_agent import profit_trail as pt
from trading_agent.position_monitor import ExitSignal, PositionMonitor, PositionSnapshot, SpreadPosition
from trading_agent.strategy_presets import PRESETS

P = pt.TrailParams()


def st(kind="debit_vertical"):
    return pt.TrailState(key="k", ticker="QQQ", strategy="Call Debit Spread", kind=kind,
                         expiration="2026-11-06")


def run(path, kind="debit_vertical", basis=800.0, target=400.0, dte=30):
    s, out = st(kind), []
    for pl in path:
        s, d, _ = pt.evaluate(s, pl, basis=basis, target=target, params=P, dte=dte)
        out.append(d)
    return s, out


def test_arms_at_target_then_trails_up():
    s, out = run([100, 420, 500, 560])
    assert out == [pt.HOLD_UNARMED, pt.HOLD_ARMED, pt.HOLD_ARMED, pt.HOLD_ARMED]
    assert s.peak_pl == 560 and s.basis == 800 and s.target == 400


def test_giveback_closes_after_25_pct_of_peak():
    _, out = run([420, 600, 460, 449])             # lock = 600 × 0.75 = 450
    assert out[-2:] == [pt.HOLD_ARMED, pt.CLOSE_GIVEBACK]


def test_ceiling_banks_it():
    _, out = run([420, 725])                       # 90 % of $800 max profit = 720
    assert out[-1] == pt.CLOSE_CEILING


def test_credit_floor_and_ceiling():
    # credit $200: target 100, floor 80 (40 %), ceiling 150 (75 %)
    _, out = run([105, 110, 84, 82], kind="credit", basis=200, target=100)
    assert out[-2:] == [pt.HOLD_ARMED, pt.CLOSE_GIVEBACK]     # 82.5 lock beats the 80 floor
    _, out = run([105, 151], kind="credit", basis=200, target=100)
    assert out[-1] == pt.CLOSE_CEILING


def test_time_stop_only_once_armed():
    _, out = run([100, 420], dte=5)
    assert out == [pt.HOLD_UNARMED, pt.CLOSE_TIME]


def test_ghost_profit_natural_prices():
    legs = [{"symbol": "L", "qty": 1, "avg_entry": 4.0}, {"symbol": "S", "qty": -1, "avg_entry": 1.6}]
    quotes = {"L": {"bid": 6.0, "ask": 6.2}, "S": {"bid": 2.0, "ask": 2.1}}
    assert pt.ghost_profit(legs, quotes) == pytest.approx((6.0 - 4.0) * 100 + (1.6 - 2.1) * 100)
    assert pt.ghost_profit(legs, {"L": quotes["L"]}) is None


def test_state_roundtrip(tmp_path):
    s, _ = run([420])
    pt.save_states({"k": s}, tmp_path / "t.json")
    assert pt.load_states(tmp_path / "t.json")["k"].peak_pl == 420
    assert pt.load_states(tmp_path / "missing.json") == {}


# ── agent integration ─────────────────────────────────────────────────────

def _leg(sym, qty, avg, bid, ask):
    p = PositionSnapshot(symbol=sym, qty=qty, side="short" if qty < 0 else "long",
                         avg_entry_price=avg, current_price=0, market_value=0, cost_basis=0,
                         unrealized_pl=0, unrealized_plpc=0, asset_class="us_option")
    return p


def _spread(natural_pl, exp=None):
    legs = [_leg("QQQ261106C00756000", 1, 7.91, 0, 0), _leg("QQQ261106C00772000", -1, 0.0, 0, 0)]
    s = SpreadPosition(underlying="QQQ", strategy_name="Call Debit Spread", legs=legs,
                       original_credit=-7.91, max_loss=791.0, spread_width=16.0,
                       net_unrealized_pl=natural_pl, expiration=exp or (date.today() + timedelta(days=30)).isoformat(),
                       short_strikes=[772.0], contracts_open=1)
    s.net_natural_pl = natural_pl
    return s


def _agent(mode, quotes=None):
    from trading_agent.agent import TradingAgent
    a = TradingAgent.__new__(TradingAgent)
    a.preset = replace(PRESETS["balanced"], profit_trail_mode=mode)
    a.position_monitor = PositionMonitor("k", "s", post_fill_grace_seconds=0)
    rows = []
    a.journal_kb = SimpleNamespace(log_signal=lambda **kw: rows.append(kw), jsonl_path=None)
    a.data_provider = SimpleNamespace(fetch_option_quotes=lambda syms: quotes or {})
    return a, rows


def _cycle(a, s):
    a.position_monitor.evaluate([s], {}, {})
    return a._apply_profit_trail([s])[0]


def test_live_mode_holds_past_target_and_closes_on_giveback():
    a, _ = _agent("live")
    # max profit (16 − 7.91) × 100 = 809 → target 404.5, ceiling 728.1
    assert _cycle(a, _spread(450)).exit_signal == ExitSignal.HOLD           # armed, not closed
    assert _cycle(a, _spread(600)).exit_signal == ExitSignal.HOLD
    s = _cycle(a, _spread(440))                                            # lock 450
    assert s.exit_signal == ExitSignal.PROFIT_TARGET and "giveback" in s.exit_reason


def test_shadow_mode_keeps_legacy_close_and_journals_trail_outcome():
    quotes = {"QQQ261106C00756000": {"bid": 13.5, "ask": 13.6},
              "QQQ261106C00772000": {"bid": 0.0, "ask": 0.0}}
    a, rows = _agent("shadow", quotes)
    assert _cycle(a, _spread(450)).exit_signal == ExitSignal.PROFIT_TARGET   # legacy close stays
    # Position closed for real → ghost priced: long 13.5 bid − 7.91 = 559, short ask 0 → None
    a._apply_profit_trail([])
    assert rows == []                                                      # no usable short quote
    a.data_provider = SimpleNamespace(fetch_option_quotes=lambda syms: {
        "QQQ261106C00756000": {"bid": 15.0, "ask": 15.1},
        "QQQ261106C00772000": {"bid": 0.01, "ask": 0.05}})
    a._apply_profit_trail([])                                              # 709 − 5 = 704: armed
    a.data_provider = SimpleNamespace(fetch_option_quotes=lambda syms: {
        "QQQ261106C00756000": {"bid": 16.5, "ask": 16.6},
        "QQQ261106C00772000": {"bid": 0.01, "ask": 0.05}})
    a._apply_profit_trail([])                                              # 854 − 5 ≥ ceiling
    assert rows and rows[-1]["action"] == "profit_trail_shadow"
    rs = rows[-1]["raw_signal"]
    assert rs["actual_exit_pl"] == 450 and rs["trail_exit_pl"] > 728
    assert rs["trail_minus_actual"] == pytest.approx(rs["trail_exit_pl"] - 450)
    assert pt.load_states() == {}


def test_off_mode_untouched_and_stops_win():
    a, _ = _agent("off")
    assert _cycle(a, _spread(450)).exit_signal == ExitSignal.PROFIT_TARGET
    a, _ = _agent("live")
    s = _cycle(a, _spread(-500))                                           # 50 % of debit stop
    assert s.exit_signal == ExitSignal.STOP_LOSS


def test_credit_floor_binds_when_peak_is_low():
    # peak 101 → 25 % giveback lock 75.75, but the 40 % floor (80) is higher
    _, out = run([101, 85, 80], kind="credit", basis=200, target=100)
    assert out == [pt.HOLD_ARMED, pt.HOLD_ARMED, pt.CLOSE_GIVEBACK]
