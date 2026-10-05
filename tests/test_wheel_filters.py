"""Backlog §2 (2026-10-05): 200-day CSP filter, one pick per sector,
Wheel tunables wired into PresetConfig."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from trading_agent.decision_engine import (
    _CC_DEFAULT_DTE_BAND, _CC_DEFAULT_MAX_SHORT_DELTA, _CC_DEFAULT_MIN_IV_RANK,
    _CSP_DEFAULT_DTE_BAND, _CSP_DEFAULT_MAX_SHORT_DELTA, _CSP_DEFAULT_MIN_IV_RANK,
    _CSP_DEFAULT_STRIKE_BAND,
)
from trading_agent.long_term_evaluator import LongTermEvaluator
from trading_agent.positions_provider import ManualPositionsProvider
from trading_agent.sector_map import wheel_sector
from trading_agent.strategy_presets import PRESETS, load_active_preset, save_active_preset

GOOD = {"market_cap": 2.5e11, "pe_ratio": 22.0, "eps_ttm": 3.1,
        "net_profit_margin_ttm": 22.0, "roe": 38.0, "beta": 0.6,
        "vol_avg_10d": 14_000_000.0}
PRESET = PRESETS["balanced"]


def _put(sym, strike=95.0, bid=1.95, ask=2.05):
    return {"strike": strike, "delta": -0.25, "bid": bid, "ask": ask, "dte": 35,
            "symbol": sym, "type": "put"}


def _ev(*, sma=None, sectors=None, preset=PRESET, credit=None):
    credit = credit or {}
    return LongTermEvaluator(
        positions_provider=ManualPositionsProvider.from_dicts([]),
        call_chain_fetcher=lambda t: [], preset=preset,
        put_chain_fetcher=lambda t: [_put(f"{t}P95", bid=credit.get(t, 1.95),
                                          ask=credit.get(t, 1.95) + 0.10)],
        fundamentals_fetcher=lambda t: GOOD, spot_fetcher=lambda t: 100.0,
        trend_fetcher=(None if sma is None else (lambda t: sma.get(t) if isinstance(sma, dict) else sma)),
        sector_fetcher=(None if sectors is None else (lambda t: wheel_sector(t))),
    )


# ── 200-day filter ────────────────────────────────────────────────────────

def test_csp_blocked_below_200d_sma():
    ev = _ev(sma={"KO": 105.0})
    assert ev.recommend(["KO"]) == []
    assert ev.last_diagnostics["KO"] == ["below_200d_sma (100.00 < 105.00)"]


def test_csp_allowed_above_200d_sma():
    assert [r.ticker for r in _ev(sma={"KO": 90.0}).recommend(["KO"])] == ["KO"]


def test_csp_fails_closed_when_trend_unknown():
    ev = _ev(sma={})
    assert ev.recommend(["KO"]) == []
    assert ev.last_diagnostics["KO"][0].startswith("trend_unavailable")


def test_trend_filter_toggle_and_legacy_callers():
    off = replace(PRESET, csp_require_above_sma200=False)
    assert _ev(sma={"KO": 105.0}, preset=off).recommend(["KO"])
    assert _ev().recommend(["KO"])                      # no trend_fetcher → not applied


# ── one pick per sector ───────────────────────────────────────────────────

def test_one_ticker_per_sector_best_first():
    ev = _ev(sectors=True, credit={"BAC": 2.40, "WFC": 1.95, "VZ": 1.95})
    recs = ev.recommend(["BAC", "WFC", "VZ"])
    assert sorted(r.ticker for r in recs) == ["BAC", "VZ"]          # WFC: Financials taken
    assert ev.last_diagnostics["WFC"] == ["sector_cap (Financials: BAC ranked higher)"]


def test_sector_cap_two_and_off():
    two = replace(PRESET, csp_max_per_sector=2)
    assert len(_ev(sectors=True, preset=two).recommend(["BAC", "WFC", "C"])) == 2
    off = replace(PRESET, csp_max_per_sector=0)
    assert len(_ev(sectors=True, preset=off).recommend(["BAC", "WFC", "C"])) == 3


def test_wheel_sector_sources():
    assert wheel_sector("VZ") == "Communications"
    assert wheel_sector("ZZZ", lambda t: "Financial Services") == "Financials"
    assert wheel_sector("ZZZ", lambda t: None) == "ZZZ"           # unknown = own sector
    assert wheel_sector("ZZZ", lambda t: (_ for _ in ()).throw(RuntimeError())) == "ZZZ"


# ── preset wiring ─────────────────────────────────────────────────────────

def test_preset_defaults_equal_scorer_fallbacks():
    assert PRESET.csp_max_short_delta == _CSP_DEFAULT_MAX_SHORT_DELTA
    assert PRESET.csp_dte_band == _CSP_DEFAULT_DTE_BAND
    assert PRESET.csp_min_iv_rank == _CSP_DEFAULT_MIN_IV_RANK
    assert PRESET.csp_strike_band == _CSP_DEFAULT_STRIKE_BAND
    assert PRESET.cc_max_short_delta == _CC_DEFAULT_MAX_SHORT_DELTA
    assert PRESET.cc_dte_band == _CC_DEFAULT_DTE_BAND
    assert PRESET.cc_min_iv_rank == _CC_DEFAULT_MIN_IV_RANK
    assert "Wheel CSP Δ≤0.30 21–60d K 85%–97% >200d 1/sector" in PRESET.to_summary_line()


def test_custom_tuple_fields_roundtrip_as_tuples(tmp_path):
    fp = tmp_path / "STRATEGY_PRESET.json"
    save_active_preset("custom", custom={"csp_strike_band": [0.9, 0.95],
                                         "csp_dte_band": [30, 45],
                                         "iron_butterfly_dte_grid": [21, 30]}, path=fp)
    p = load_active_preset(fp)
    assert p.csp_strike_band == (0.9, 0.95) and p.csp_dte_band == (30, 45)
    assert p.iron_butterfly_dte_grid == (21, 30)            # was loaded as a list before
    hash(p)                                                 # frozen dataclass stays hashable
