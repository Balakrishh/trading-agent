"""
Tests for ``trading_agent.backtest.synthetic_chain``.

Pin two structural properties:

  1. The strike grid is wide enough (and aligned to ``STRIKE_STEP``) to
     cover every (Δ × width%) combination the preset's grid asks for.
  2. The chain dict shape is exactly what ``decision_engine.decide()``
     consumes, and ``_quote_credit`` re-extracts the BS mid we
     synthesized when fed back through it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest

from trading_agent.backtest.black_scholes import bs_delta, bs_price
from trading_agent.backtest.synthetic_chain import (
    STRIKE_STEP,
    SYNTHETIC_HALF_SPREAD,
    ChainConfig,
    build_chain_config_from_preset,
    build_chain_slice,
)
from trading_agent.chain_scanner import _quote_credit
from trading_agent.decision_engine import ChainSlice


@dataclass
class _StubPreset:
    """Minimal duck-typed preset for the bridging helper."""
    delta_grid: tuple = (0.20, 0.25, 0.30)
    width_grid_pct: tuple = (0.010, 0.015, 0.020)


# --------------------------------------------------------------------------
# build_chain_slice — shape & content
# --------------------------------------------------------------------------

def test_build_chain_slice_returns_chainslice_with_dict_contracts():
    cfg = ChainConfig(side="bull_put",
                      delta_grid=(0.25,),
                      width_grid_pct=(0.015,))
    slc = build_chain_slice(
        ticker="SPY", side="bull_put", spot=500.0,
        sigma_annual=0.20, now=date(2026, 5, 4),
        expiration=date(2026, 5, 18), cfg=cfg,
    )
    assert isinstance(slc, ChainSlice)
    assert slc.expiration == "2026-05-18"
    assert slc.dte == 14
    assert len(slc.contracts) > 0
    for c in slc.contracts:
        assert {"strike", "delta", "bid", "ask", "symbol"} <= c.keys()
        assert c["bid"] <= c["ask"]
        assert c["strike"] > 0


def test_build_chain_slice_strikes_aligned_to_grid():
    cfg = ChainConfig(side="bear_call",
                      delta_grid=(0.20, 0.30),
                      width_grid_pct=(0.010, 0.020))
    slc = build_chain_slice(
        ticker="QQQ", side="bear_call", spot=400.0,
        sigma_annual=0.22, now=date(2026, 5, 4),
        expiration=date(2026, 5, 25), cfg=cfg,
    )
    for c in slc.contracts:
        # Each strike is on the $1 grid (within float jitter)
        rem = c["strike"] / STRIKE_STEP - round(c["strike"] / STRIKE_STEP)
        assert abs(rem) < 1e-6


def test_build_chain_slice_puts_have_negative_delta():
    cfg = ChainConfig(side="bull_put",
                      delta_grid=(0.25,),
                      width_grid_pct=(0.015,))
    slc = build_chain_slice(
        ticker="SPY", side="bull_put", spot=500.0,
        sigma_annual=0.20, now=date(2026, 5, 4),
        expiration=date(2026, 5, 18), cfg=cfg,
    )
    # Puts: Δ ≤ 0 across the grid.
    assert all(c["delta"] <= 0 for c in slc.contracts)


def test_build_chain_slice_calls_have_non_negative_delta():
    cfg = ChainConfig(side="bear_call",
                      delta_grid=(0.25,),
                      width_grid_pct=(0.015,))
    slc = build_chain_slice(
        ticker="SPY", side="bear_call", spot=500.0,
        sigma_annual=0.20, now=date(2026, 5, 4),
        expiration=date(2026, 5, 18), cfg=cfg,
    )
    assert all(c["delta"] >= 0 for c in slc.contracts)


# --------------------------------------------------------------------------
# Round-trip through _quote_credit — the mid we synthesized comes back
# --------------------------------------------------------------------------

def test_quote_credit_recovers_synthetic_mid_minus_haircut():
    """
    The legacy parity test: build a single put strike at exact BS mid
    with a 5-cent half-spread, feed the bid/ask through the engine's
    ``_quote_credit``, and verify it returns ``mid_short − mid_long −
    fill_haircut`` — i.e., the engine's pricing math sees what we
    intended.
    """
    spot, strike_short, strike_long = 500.0, 490.0, 485.0
    sigma, t_years = 0.20, 14 / 365.0
    short_mid = bs_price(spot, strike_short, t_years, sigma, option_type="put")
    long_mid = bs_price(spot, strike_long, t_years, sigma, option_type="put")
    short_bid = short_mid - SYNTHETIC_HALF_SPREAD
    short_ask = short_mid + SYNTHETIC_HALF_SPREAD
    long_bid = long_mid - SYNTHETIC_HALF_SPREAD
    long_ask = long_mid + SYNTHETIC_HALF_SPREAD
    # _quote_credit defaults to a $0.02 fill haircut.
    expected = round(short_mid - long_mid - 0.02, 2)
    got = _quote_credit(short_bid, short_ask, long_bid, long_ask)
    assert got == pytest.approx(expected, abs=0.02)


# --------------------------------------------------------------------------
# Bridge from a PresetConfig
# --------------------------------------------------------------------------

def test_build_chain_config_from_preset_round_trips_grids():
    preset = _StubPreset(delta_grid=(0.20, 0.30),
                         width_grid_pct=(0.010, 0.020, 0.030))
    cfg = build_chain_config_from_preset("bull_put", preset)
    assert cfg.side == "bull_put"
    assert tuple(cfg.delta_grid) == (0.20, 0.30)
    assert tuple(cfg.width_grid_pct) == (0.010, 0.020, 0.030)
