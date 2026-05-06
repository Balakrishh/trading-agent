"""
synthetic_chain.py — reconstruct an option chain from (spot, σ, strike grid).

The live ``decide()`` consumes a list of dict-shaped contracts with
``{strike, delta, bid, ask, symbol}``. In the live agent these come
from ``MarketDataProvider.fetch_option_chain``. In the backtester there
is no such endpoint for historical data, so we synthesize the chain
analytically:

  * Strike grid: a span around spot covering the preset's
    ``delta_grid × width_grid_pct`` requirements.
  * Per-strike Δ:  ``bs_delta(spot, strike, t_years, σ_t, r=0)``
  * Per-strike mid: ``bs_price(...)`` for the relevant put/call
  * Synthetic bid/ask: ``mid ∓ half_spread`` so ``_quote_credit``
    extracts the exact mid back out (i.e. the engine sees the BS
    mid as if it were the NBBO mid, with a one-tick spread to look
    plausible).

This is the same shape the legacy
``backtest_ui._synth_chain_slice_for_decide`` already used for the
Alpaca-historical bridge — see lines 1665-1736 of the old file. The
difference is that we build it from a *parametric* model (spot + σ +
strike grid) instead of from per-symbol historical bars, which lets
the backtester run on dates older than Alpaca's options-bar window.

Strike grid sizing
------------------
We honour the preset's ``width_grid_pct`` to pick the maximum strike
distance and the preset's ``delta_grid`` to pick the deepest OTM
needed. The grid is then snapped to the project's standard $1 step
(see ``ChainScanner._infer_grid_step``) so the long-leg snap inside
``decide()`` lands on a strike we actually populated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Sequence

from trading_agent.backtest.black_scholes import bs_delta, bs_price
from trading_agent.decision_engine import ChainSlice

# Strikes are placed on a $1 grid for the index ETFs we trade — matches
# what ``ChainScanner._infer_grid_step`` infers from real Alpaca chains
# for SPY/QQQ/IWM and avoids the long-leg-not-found rejection inside
# ``decide()`` when the synthetic grid is too coarse.
STRIKE_STEP = 1.0

# Half of the synthetic NBBO spread, in dollars. ``_quote_credit``
# averages bid+ask back to mid before pricing, so anything > 0 keeps
# the bid/ask gate in ``risk_manager`` happy without changing the
# implied credit.  $0.05 = one option tick.
SYNTHETIC_HALF_SPREAD = 0.05


@dataclass(frozen=True)
class ChainConfig:
    """
    Parameters that govern synthetic-chain construction. Built from the
    PresetConfig grids in ``runner._build_chain_config_for_side`` so
    a single preset hot-reload changes both the planner and the
    synthetic-chain shape simultaneously.
    """
    side:           str               # "bull_put" | "bear_call"
    delta_grid:     Sequence[float]   # |Δ| targets the engine sweeps
    width_grid_pct: Sequence[float]   # width as fraction of spot
    # Multiplicative buffer over the deepest required OTM so the engine
    # can find a long-leg strike one width past the deepest short.
    grid_pad_pct:   float = 0.02


def _strike_span(spot: float, cfg: ChainConfig) -> tuple[float, float]:
    """
    Compute the (lo, hi) strike bracket the synthetic chain must cover
    so every (Δ-target, width%) tuple in the engine's sweep can resolve.

    The bracket extends past the deepest OTM short-strike implied by
    ``max(delta_grid)`` by the largest ``width_grid_pct`` plus a 2 %
    pad — empirically wide enough that ``decide()``'s long-leg snap
    never falls off the edge for the preset grids we ship.
    """
    deepest_pct = max(cfg.width_grid_pct) + cfg.grid_pad_pct
    # |Δ| 0.50 ≈ ATM; deeper-Δ shorts are *closer* to ATM, not further,
    # so the dominant variable for span is the widest spread width.
    span_pct = max(0.05, 2 * deepest_pct + 0.05)
    lo = spot * (1.0 - span_pct)
    hi = spot * (1.0 + span_pct)
    return lo, hi


def _strikes_on_grid(spot: float, cfg: ChainConfig) -> List[float]:
    """
    Return all strikes inside the bracket aligned to ``STRIKE_STEP``,
    inclusive on both ends. Sorted ascending.
    """
    lo, hi = _strike_span(spot, cfg)
    lo_grid = round(lo / STRIKE_STEP) * STRIKE_STEP
    hi_grid = round(hi / STRIKE_STEP) * STRIKE_STEP
    if hi_grid < lo_grid:
        return []
    n = int(round((hi_grid - lo_grid) / STRIKE_STEP)) + 1
    return [round(lo_grid + i * STRIKE_STEP, 4) for i in range(n)]


def _opt_type_for_side(side: str) -> str:
    """Bull-put → puts; bear-call → calls. Same convention as decide()."""
    return "put" if side == "bull_put" else "call"


def _symbol(ticker: str, expiration: date, opt_type: str,
            strike: float) -> str:
    """
    OCC-ish synthetic symbol used as a stable key inside the engine.
    Format: ``TICKER_YYYYMMDD_C_00500000`` (8-digit strike × 1000).
    Not a real OCC symbol — just a unique handle the journal can carry.
    """
    yymmdd = expiration.strftime("%Y%m%d")
    cp = "C" if opt_type == "call" else "P"
    strike_str = f"{int(round(strike * 1000)):08d}"
    return f"{ticker}_{yymmdd}_{cp}_{strike_str}"


def build_chain_slice(*,
                      ticker: str,
                      side: str,
                      spot: float,
                      sigma_annual: float,
                      now: date,
                      expiration: date,
                      cfg: ChainConfig,
                      r: float = 0.0) -> ChainSlice:
    """
    Build one ``ChainSlice`` (one expiration's worth) from spot, σ, and
    the synthetic-chain config. Every strike on the grid gets a row;
    ``decide()`` will then sweep its (Δ × width) grid against this.

    ``sigma_annual`` is the *annualised* implied vol used for both
    pricing and Δ — there is no separate σ_pricing vs σ_delta; matching
    them by construction means the Δ the engine reads agrees with the
    mid the engine prices.
    """
    dte = max(1, (expiration - now).days)
    t_years = dte / 365.0
    opt_type = _opt_type_for_side(side)
    strikes = _strikes_on_grid(spot, cfg)
    contracts = []
    for k in strikes:
        delta = bs_delta(spot, k, t_years, sigma_annual, r=r,
                         option_type=opt_type)
        mid = bs_price(spot, k, t_years, sigma_annual, r=r,
                       option_type=opt_type)
        # Clip the bid floor at 0 so deep-OTM strikes don't go negative
        # after the half-spread subtraction. _quote_credit treats 0 as
        # "missing quote" and falls back to bid/ask conservatively, so
        # this only affects strikes the engine wouldn't consider anyway.
        bid = max(0.0, mid - SYNTHETIC_HALF_SPREAD)
        ask = max(SYNTHETIC_HALF_SPREAD, mid + SYNTHETIC_HALF_SPREAD)
        contracts.append({
            "strike": float(k),
            "delta":  float(delta),
            "bid":    round(float(bid), 4),
            "ask":    round(float(ask), 4),
            "symbol": _symbol(ticker, expiration, opt_type, k),
        })
    return ChainSlice(
        expiration=expiration.isoformat(),
        dte=dte,
        contracts=contracts,
    )


def build_chain_config_from_preset(side: str, preset) -> ChainConfig:
    """
    Bridge from the live ``PresetConfig`` to a ``ChainConfig``. Kept
    thin so the runner can call it once per cycle and reuse the result
    across both decision-engine slices when a preset's ``dte_grid``
    has multiple expirations.
    """
    return ChainConfig(
        side=side,
        delta_grid=tuple(preset.delta_grid),
        width_grid_pct=tuple(preset.width_grid_pct),
    )


__all__ = [
    "ChainConfig",
    "STRIKE_STEP",
    "SYNTHETIC_HALF_SPREAD",
    "build_chain_slice",
    "build_chain_config_from_preset",
]
