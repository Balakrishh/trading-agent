"""2026-11-06 — no debit spreads on IWM, no put debits on SPY / QQQ (paper
preset), and the mid-first ladder for debit stop closes
(docs/plans/research_debit_stress.md, skill 59 §4)."""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from trading_agent import executor as ex
from trading_agent.position_monitor import ExitSignal
from trading_agent.strategy_presets import PRESETS, load_active_preset

from tests.test_direction_risk import _bearish_planner
from tests.test_mid_exit_ladder_fit import _closer, _closing, _gld

NO_IWM = replace(PRESETS["balanced"], debit_excluded_tickers=("IWM",))


@pytest.mark.parametrize("playbook", ["call_debit", "put_debit"])
def test_debit_excluded_ticker_gets_no_debit_spread(monkeypatch, playbook):
    pl, a = _bearish_planner(NO_IWM)
    monkeypatch.setattr(pl, "_playbook_name", lambda an: playbook)
    monkeypatch.setattr(pl, "_plan_playbook",
                        lambda *x: pytest.fail("no debit spread may be tried on IWM"))
    p = pl.plan("IWM", a)
    assert not p.valid and "debit_excluded_tickers" in p.rejection_reason
    assert playbook.replace("_", " ") in p.rejection_reason


def test_other_tickers_unaffected(monkeypatch):
    pl, a = _bearish_planner(NO_IWM)
    tried = []
    monkeypatch.setattr(pl, "_plan_playbook", lambda t, an, pb: tried.append((t, pb)))
    pl.plan("SPY", a)
    assert tried == [("SPY", "put_debit")]


def test_preset_loads_lists_and_summarizes(tmp_path):
    fp = tmp_path / "STRATEGY_PRESET.json"
    fp.write_text(json.dumps({"profile": "custom", "custom": {
        "debit_excluded_tickers": ["IWM"],
        "put_debit_excluded_tickers": ["XLY", "SPY", "QQQ"]}}))
    p = load_active_preset(fp)
    assert p.debit_excluded_tickers == ("IWM",) and "SPY" in p.put_debit_excluded_tickers
    line = p.to_summary_line()
    assert "No debit: IWM" in line and "No put debit: XLY,SPY,QQQ" in line
    assert "No debit:" not in PRESETS["balanced"].to_summary_line()


# ── stop ladder ───────────────────────────────────────────────────────────

def test_stop_ladder_prices_end_at_natural_rounded_to_fill():
    assert ex.stop_ladder_attempts(-4.70, -4.00) == [
        ("mleg_stop_0", -4.70), ("mleg_stop_1", -4.35), ("mleg_stop_2", -4.00)]
    # natural −0.343 must not become a −0.35 ask (more than the market bids)
    assert ex.stop_ladder_attempts(-0.47, -0.343)[-1] == ("mleg_stop_2", -0.34)


def _stopping(name="Put Debit Spread", signal=ExitSignal.STOP_LOSS):
    s = _gld()
    s.strategy_name, s.exit_signal, s.exit_reason, s.close_min_profit = name, signal, "stop", None
    return s


def test_debit_stop_close_tries_mid_first():
    e, b = _closer(fill_at=2)
    out = e._close_spread_mleg(_stopping())
    assert b.prices == [-4.70, -4.35] and out["close_method"] == "mleg_stop_1"


@pytest.mark.parametrize("spread", [
    _stopping(signal=ExitSignal.HARD_STOP),          # emergencies keep improved → natural
    _stopping(name="Bull Put Spread"),               # credit spreads unchanged
])
def test_other_closes_keep_improved_then_natural(spread):
    e, b = _closer()
    e._close_spread_mleg(spread)
    assert b.prices == [-4.35, -4.0]


def test_profit_ladder_top_rounds_toward_the_fill_without_passing_the_floor():
    e, b = _closer()
    e.data_provider = SimpleNamespace(fetch_option_quotes=lambda s: {
        "GLD261120P00371000": {"bid": 10.207, "ask": 11.00}, "GLD261120P00365000": {"bid": 5.60, "ask": 6.20}})
    e._close_spread_mleg(_closing(50.0))                # natural −4.007 → last ask −4.00
    assert b.prices[-1] == -4.0
