"""2026-10-05 — in-cycle sector cap + total open-risk cap."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from trading_agent.position_caps import (
    compute_position_cap_dedup_set, open_defined_risk, register_open,
)
from trading_agent.strategy_presets import PRESETS, load_active_preset, save_active_preset

SECTORS = {"SPY": "Broad Market", "QQQ": "Broad Market", "IWM": "Broad Market",
           "DIA": "Broad Market", "XLF": "Financials", "XLE": "Energy"}
TICKERS = list(SECTORS)
sector_for = SECTORS.get


def _register(t, per_t, per_s):
    return register_open(t, TICKERS, per_t, per_s, sector_for=sector_for,
                         max_positions_per_ticker=1, max_positions_per_sector=2)


def test_in_cycle_sector_cap_blocks_third_broad_market_trade():
    """The live failure: SPY, QQQ, IWM opened in one cycle with cap 2."""
    blocked, per_t, per_s, _ = compute_position_cap_dedup_set(
        {"positions": []}, TICKERS, sector_for=sector_for,
        max_positions_per_ticker=1, max_positions_per_sector=2)
    assert blocked == set()
    blocked |= _register("SPY", per_t, per_s)
    assert blocked == {"SPY"}
    blocked |= _register("QQQ", per_t, per_s)
    assert {"IWM", "DIA"} <= blocked and "XLF" not in blocked
    assert per_s == {"Broad Market": 2}


def test_pending_order_counts_toward_sector():
    blocked, per_t, per_s, _ = compute_position_cap_dedup_set(
        {"positions": [{"underlying": "SPY"}]}, TICKERS, sector_for=sector_for,
        max_positions_per_ticker=1, max_positions_per_sector=2)
    blocked |= _register("QQQ", per_t, per_s)        # QQQ order pending fill
    assert "IWM" in blocked


def test_open_defined_risk_sums_contracts_and_excludes_wheel():
    mr = {"positions": [
        {"strategy": "Put Debit Spread", "max_loss": 192.0, "contracts": 4},
        {"strategy": "Call Debit Spread", "max_loss": 791.0, "contracts": 1},
        {"strategy": "Calendar Spread", "max_loss": 579.0, "contracts": 1},
        {"strategy": "Cash-Secured Put", "max_loss": 4279.0, "contracts": 1},
        {"strategy": "Iron Condor", "max_loss": 300.0, "contracts": None},   # unknown → 1
    ]}
    assert open_defined_risk(mr) == pytest.approx(768 + 791 + 579 + 300)
    assert open_defined_risk({}) == 0.0


def test_preset_field_default_summary_and_overlay(tmp_path):
    assert PRESETS["balanced"].max_total_risk_pct == 0.10
    assert "(total 10%)" in PRESETS["balanced"].to_summary_line()
    fp = tmp_path / "STRATEGY_PRESET.json"
    save_active_preset("custom", custom={"max_total_risk_pct": 0.06}, path=fp)
    assert load_active_preset(fp).max_total_risk_pct == 0.06


def test_trade_risk_pct_applies_to_validator_and_sizer():
    from trading_agent.agent import TradingAgent
    a = TradingAgent.__new__(TradingAgent)
    a._base_max_risk_pct = 0.03
    a.risk_manager = SimpleNamespace(max_risk_pct=0.03)
    a.executor = SimpleNamespace(max_risk_pct=0.03)
    a._apply_risk_multiplier(0.5)
    assert a._cycle_risk_pct == 0.015
    a._set_trade_risk_pct(min(a._cycle_risk_pct, 300 / 30_000))   # $300 budget left
    assert a.risk_manager.max_risk_pct == a.executor.max_risk_pct == 0.01


def test_total_risk_gate_sizes_into_remaining_then_skips():
    from trading_agent.agent import TradingAgent
    rows = []
    a = TradingAgent.__new__(TradingAgent)
    a._cycle_risk_pct = 0.03
    a.risk_manager = SimpleNamespace(max_risk_pct=0.03)
    a.executor = SimpleNamespace(max_risk_pct=0.03)
    a.journal_kb = SimpleNamespace(log_signal=lambda **kw: rows.append(kw))
    a._cached_price = lambda t: 1.0
    assert a._total_risk_gate("SPY", 2_500, 3_000, 30_000) is None
    assert a.executor.max_risk_pct == pytest.approx(500 / 30_000)     # $500 left
    assert a._total_risk_gate("SPY", 1_000, 3_000, 30_000) is None
    assert a.risk_manager.max_risk_pct == 0.03                        # per-trade cap binds
    skip = a._total_risk_gate("QQQ", 3_000, 3_000, 30_000)
    assert skip["reason"] == "Total risk cap" and rows[0]["action"] == "skipped_total_risk_cap"
    assert TradingAgent._submitted_risk({"max_loss": 192.0, "execution": {"qty": 4}}) == 768.0
    assert TradingAgent._submitted_risk({"max_loss": 300.0, "execution": {}}) == 300.0
