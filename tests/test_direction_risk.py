"""Week of 2026-10-19 learnings, entry side (skill 37, 59):
correlated-risk groups and direction budgets (E6), the longer stop
cool-down, the put-debit exclusion list (E7) and the preset wiring."""
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pytest

from trading_agent import position_caps as pc
from trading_agent.strategy_presets import PRESETS, PresetConfig, _make_custom

EQ = 28918.0


# ── groups and exposure ───────────────────────────────────────────────

def test_risk_groups():
    assert pc.risk_group_for("TLT") == pc.risk_group_for("XLU") == "rates"
    assert pc.risk_group_for("GDX") == "gold" and pc.risk_group_for("IWM") == "us_equity"
    assert pc.risk_group_for("AMZN") is None


def _book():
    # The book on 2026-10-23 (per-contract max loss × contracts).
    return {"positions": [
        {"underlying": "SPY", "strategy": "Calendar Spread", "max_loss": 579.0, "contracts": 1},
        {"underlying": "AMZN", "strategy": "Call Debit Spread", "max_loss": 460.0, "contracts": 1},
        {"underlying": "TLT", "strategy": "Put Debit Spread", "max_loss": 55.0, "contracts": 5},
        {"underlying": "IWM", "strategy": "Put Debit Spread", "max_loss": 123.0, "contracts": 2},
        {"underlying": "XLU", "strategy": "Put Debit Spread", "max_loss": 56.0, "contracts": 5},
        {"underlying": "GLD", "strategy": "Put Debit Spread", "max_loss": 265.0, "contracts": 1},
        {"underlying": "VZ", "strategy": "Cash-Secured Put", "max_loss": 4000.0, "contracts": 1},
    ]}


def test_direction_exposure_counts_contracts_and_skips_neutral_and_wheel():
    ex = pc.direction_exposure(_book())
    assert ex["book:bear"] == 275 + 246 + 280 + 265
    assert ex["group:rates:bear"] == 555.0
    assert ex["group:us_equity:bear"] == 246.0 and ex["group:gold:bear"] == 265.0
    assert ex["book:bull"] == 460.0                      # CSP excluded, calendar neutral
    assert not any("SPY" in k or "neutral" in k for k in ex)


def test_direction_room_is_the_tighter_of_group_and_book():
    ex = pc.direction_exposure(_book())
    room, label = pc.direction_room("TLT", "Put Debit Spread", EQ, ex, 0.03, 0.06)
    # rates: 867.54 − 555 = 312.54; book: 1735.08 − 1066 = 669.08
    assert room == pytest.approx(0.03 * EQ - 555) and label.startswith("group:rates:bear")
    room, label = pc.direction_room("AMZN", "Put Debit Spread", EQ, ex, 0.03, 0.06)
    assert room == pytest.approx(0.06 * EQ - 1066) and label.startswith("book:bear")
    assert pc.direction_room("SPY", "Iron Condor", EQ, ex, 0.03, 0.06) == (None, "")
    assert pc.direction_room("TLT", "Put Debit Spread", EQ, ex, 0.0, 0.0) == (None, "")
    assert pc.direction_room("TLT", "Put Debit Spread", 0.0, ex, 0.03, 0.06) == (None, "")


def test_direction_risk_failure_blocks_below_one_contract():
    ex = pc.direction_exposure(_book())
    reason, room = pc.direction_risk_failure("XLU", "Put Debit Spread", 400.0, EQ, ex, 0.03, 0.06)
    assert reason.startswith("direction_risk_cap (group:rates:bear $555") and room < 400
    reason, room = pc.direction_risk_failure("XLU", "Put Debit Spread", 60.0, EQ, ex, 0.03, 0.06)
    assert reason is None and room == pytest.approx(0.03 * EQ - 555)
    # Bullish rates trades are not limited by the bearish rates bucket.
    assert pc.direction_risk_failure("TLT", "Call Debit Spread", 400.0, EQ, ex, 0.03, 0.06)[0] is None


def test_add_exposure_tracks_same_cycle_submissions():
    ex = {}
    pc.add_exposure(ex, "TLT", "Put Debit Spread", 300.0)
    pc.add_exposure(ex, "XLU", "Put Debit Spread", 300.0)
    assert ex == {"book:bear": 600.0, "group:rates:bear": 600.0}
    reason, _ = pc.direction_risk_failure("TLT", "Put Debit Spread", 300.0, EQ, ex, 0.03, 0.06)
    assert reason is not None                            # 867 − 600 = 267 < 300


# ── stop cool-down ────────────────────────────────────────────────────

D = date.fromisoformat


def test_five_session_cooldown_blocks_the_tlt_reentry():
    stops = [("TLT", "Put Debit Spread", D("2026-10-15"))]
    # 2026-10-19 is 2 sessions after 10-15: blocked at 5, allowed at the old 1.
    assert pc.stop_cooldown_failure("TLT", "Put Debit Spread", D("2026-10-19"), stops, 1) is None
    assert pc.stop_cooldown_failure("TLT", "Put Debit Spread", D("2026-10-19"), stops, 5)
    assert pc.stop_cooldown_failure("TLT", "Put Debit Spread", D("2026-10-22"), stops, 5)  # 5th
    assert pc.stop_cooldown_failure("TLT", "Put Debit Spread", D("2026-10-23"), stops, 5) is None
    assert pc.stop_cooldown_failure("TLT", "Call Debit Spread", D("2026-10-19"), stops, 5) is None


def test_repeat_stop_extends_the_block():
    stops = [("TLT", "Put Debit Spread", D("2026-10-15")),
             ("TLT", "Put Debit Spread", D("2026-10-22"))]
    r = pc.stop_cooldown_failure("TLT", "Put Debit Spread", D("2026-11-04"), stops, 5,
                                 repeat_sessions=10, repeat_window=20)
    assert r and "repeat stop since 2026-10-15" in r and "10 session(s)" in r
    assert pc.stop_cooldown_failure("TLT", "Put Debit Spread", D("2026-11-06"), stops, 5,
                                    repeat_sessions=10, repeat_window=20) is None
    # Earlier stop outside the window → the normal 5 sessions.
    old = [("TLT", "Put Debit Spread", D("2026-08-03")), stops[1]]
    assert pc.stop_cooldown_failure("TLT", "Put Debit Spread", D("2026-11-04"), old, 5,
                                    repeat_sessions=10, repeat_window=20) is None
    assert pc.stop_cooldown_failure("TLT", "Put Debit Spread", D("2026-11-04"), stops, 0,
                                    repeat_sessions=10) is None             # off


# ── agent wiring ──────────────────────────────────────────────────────

def _agent(preset):
    from trading_agent.agent import TradingAgent
    a = TradingAgent.__new__(TradingAgent)
    a.preset = preset
    a.executor = SimpleNamespace(max_risk_pct=0.02)
    a.risk_manager = SimpleNamespace(max_risk_pct=0.02)
    a._direction_exposure = pc.direction_exposure(_book())
    return a


def test_agent_direction_block_and_sizing():
    a = _agent(PRESETS["balanced"])
    plan = SimpleNamespace(strategy_name="Put Debit Spread", max_loss=60.0)
    assert a._direction_risk_block("XLU", plan, EQ) is None
    # Room 312.5 < 2 % of equity (578) → the trade is sized into the room.
    assert a.executor.max_risk_pct == pytest.approx((0.03 * EQ - 555) / EQ)
    assert a.risk_manager.max_risk_pct == a.executor.max_risk_pct
    big = SimpleNamespace(strategy_name="Put Debit Spread", max_loss=400.0)
    assert a._direction_risk_block("TLT", big, EQ).startswith("direction_risk_cap")


def test_agent_direction_block_never_sizes_up():
    a = _agent(replace(PRESETS["balanced"], max_group_direction_risk_pct=0.0,
                       max_book_direction_risk_pct=0.5))
    plan = SimpleNamespace(strategy_name="Put Debit Spread", max_loss=60.0)
    assert a._direction_risk_block("AMZN", plan, EQ) is None
    assert a.executor.max_risk_pct == 0.02


# ── E7 put-debit exclusion ────────────────────────────────────────────

def _bearish_planner(preset):
    from unittest.mock import MagicMock
    from trading_agent.regime import Regime
    from trading_agent.strategy import StrategyPlanner
    pl = StrategyPlanner(MagicMock(), preset=preset)
    analysis = SimpleNamespace(regime=Regime.BEARISH, iv_rank=60, rsi_14=45, current_price=100.0,
                               mean_reversion_direction="", inter_market_inhibit_bullish=False,
                               leadership_zscore=0.0, leadership_anchor="", vix_zscore=0.0)
    return pl, analysis


def test_excluded_ticker_skips_the_put_debit_fallback(monkeypatch):
    pl, a = _bearish_planner(PRESETS["balanced"])
    monkeypatch.setattr(pl, "_plan_playbook",
                        lambda *x: pytest.fail("put debit must not be tried on XLK"))
    p = pl.plan("XLK", a)
    assert not p.valid and "put_debit_excluded_tickers" in p.rejection_reason


def test_other_tickers_still_get_the_put_debit(monkeypatch):
    pl, a = _bearish_planner(PRESETS["balanced"])
    tried = []
    monkeypatch.setattr(pl, "_plan_playbook", lambda t, an, pb: tried.append((t, pb)))
    pl.plan("XLE", a)
    assert tried == [("XLE", "put_debit")]


# ── preset ────────────────────────────────────────────────────────────

def test_preset_defaults_summary_and_custom_coercion():
    f = PresetConfig.__dataclass_fields__
    assert f["max_group_direction_risk_pct"].default == 0.03
    assert f["max_book_direction_risk_pct"].default == 0.06
    assert f["put_debit_excluded_tickers"].default == ("XLY", "XLK", "XLI", "GDX")
    assert f["net_delta_mode"].default == "shadow"
    line = PRESETS["balanced"].to_summary_line()
    assert "Dir risk ≤3%/group ≤6%/book" in line and "No put debit: XLY,XLK,XLI,GDX" in line
    c = _make_custom({"put_debit_excluded_tickers": ["XLY"], "net_delta_mode": "live"})
    assert c.put_debit_excluded_tickers == ("XLY",) and c.net_delta_mode == "shadow"
