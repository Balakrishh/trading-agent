"""
black_scholes.py — pure Black-Scholes option pricing + Greeks.

Why this module exists
----------------------
The backtester rebuilds option chains synthetically because no historical
options data is available in any of the project's data sources (Alpaca's
historical options endpoint returns OHLCV bars per *known* contract
symbol — useless without a full chain — and yfinance only exposes the
*current* chain). Every per-bar mark, every synthetic strike's mid, and
every Greek the engine consumes is derived here from a (spot, strike,
DTE, σ, r) tuple.

Design choices
--------------
* **No carry / dividend term.** We trade ETF index spreads on weekly
  expirations — the dividend impact over a 7-30-day horizon is well
  inside the slippage budget the executor's fill-haircut already pays.
  Simpler closed-form, fewer config knobs to drift.
* **Continuously compounded risk-free rate ``r``.** Defaults to 0.0 to
  match the legacy synthetic chain in ``backtest_ui._synth_chain_slice_for_decide``,
  which the live↔backtest parity tests pin against. Callers can pass
  ``r`` to model term-structure if the parity tests get loosened later.
* **No external numerics.** ``math.erf`` is in the stdlib, accurate to
  ~1 ulp, and avoids the scipy dependency. Vector convenience is
  achieved by callers passing numpy arrays — the math operators
  broadcast naturally.

Why these specific Greeks
-------------------------
* ``delta`` — needed by the synthetic chain so ``decide()`` can run its
  Δ-grid sweep.
* ``theta`` — backtester surfaces it for the daily P&L decomposition.
* ``gamma`` / ``vega`` — exposed for completeness; the runner doesn't
  use them today, but skill 15 requires they exist so the backtester
  can later answer "how much of the loss was σ-shock vs. delta-drift?"
  without a code change.
* ``implied_vol`` — Newton-Raphson with bisection fallback; used by
  ``sim_position`` to recover σ_entry from the market credit at trade
  open, then scaled forward via VIX-proxy on each re-mark.

References
----------
Hull, *Options, Futures, and Other Derivatives* (10e), §15.6 (BS
pricing) and §19.4 (closed-form Greeks). Newton-Raphson for IV is
§19.11; we cap iterations at 50 with a 1e-6 tolerance, falling back to
bisection in the rare case Newton diverges (deep ITM with a quoted
mid below intrinsic).
"""

from __future__ import annotations

import math
from typing import Literal

OptionType = Literal["call", "put"]


# --------------------------------------------------------------------------
# Standard-normal helpers
# --------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    """Standard-normal CDF Φ(x) = ½ (1 + erf(x / √2))."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard-normal PDF φ(x) = e^(-x²/2) / √(2π)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------
# d1 / d2
# --------------------------------------------------------------------------

def _d1_d2(spot: float, strike: float, t_years: float,
           sigma: float, r: float) -> tuple[float, float]:
    """
    Compute the BS d1, d2 ordinates.

    Both spot and strike must be positive; ``sigma * sqrt(t_years)``
    must be positive. Caller responsibility.
    """
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------

def bs_price(spot: float, strike: float, t_years: float,
             sigma: float, r: float = 0.0,
             option_type: OptionType = "call") -> float:
    """
    Black-Scholes price for a European call or put.

    Returns intrinsic value when ``t_years <= 0`` or ``sigma <= 0`` —
    the limiting cases — so callers don't have to special-case
    expiration-day re-marks.
    """
    if spot <= 0 or strike <= 0:
        return 0.0
    # At/after expiry, value is intrinsic. Same for zero vol — the option
    # collapses to a forward.
    if t_years <= 0 or sigma <= 0:
        if option_type == "call":
            return max(0.0, spot - strike)
        return max(0.0, strike - spot)

    d1, d2 = _d1_d2(spot, strike, t_years, sigma, r)
    discount = math.exp(-r * t_years)
    if option_type == "call":
        price = spot * norm_cdf(d1) - strike * discount * norm_cdf(d2)
    else:
        price = strike * discount * norm_cdf(-d2) - spot * norm_cdf(-d1)
    # Clip tiny negatives produced by float rounding deep OTM.
    return max(0.0, price)


# --------------------------------------------------------------------------
# Greeks
# --------------------------------------------------------------------------

def bs_delta(spot: float, strike: float, t_years: float,
             sigma: float, r: float = 0.0,
             option_type: OptionType = "call") -> float:
    """
    Δ_call = Φ(d1)        ∈ [0, 1]
    Δ_put  = Φ(d1) − 1    ∈ [-1, 0]

    Signed delta is what the live ``decide()`` consumes — the C/W
    floor uses ``|Δ|``, so signs cancel inside ``_cw_floor`` and using
    signed values keeps puts and calls symmetric.
    """
    if spot <= 0 or strike <= 0 or t_years <= 0 or sigma <= 0:
        # Step-function delta at expiry / zero vol.
        intrinsic = (spot - strike) if option_type == "call" else (strike - spot)
        if intrinsic > 0:
            return 1.0 if option_type == "call" else -1.0
        return 0.0
    d1, _ = _d1_d2(spot, strike, t_years, sigma, r)
    if option_type == "call":
        return norm_cdf(d1)
    return norm_cdf(d1) - 1.0


def bs_gamma(spot: float, strike: float, t_years: float,
             sigma: float, r: float = 0.0) -> float:
    """Γ = φ(d1) / (S σ √t). Same for call and put."""
    if spot <= 0 or strike <= 0 or t_years <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(spot, strike, t_years, sigma, r)
    return norm_pdf(d1) / (spot * sigma * math.sqrt(t_years))


def bs_vega(spot: float, strike: float, t_years: float,
            sigma: float, r: float = 0.0) -> float:
    """ν = S φ(d1) √t. Same for call and put. Per *whole point* of σ — divide by 100 for "per 1 % vol point"."""
    if spot <= 0 or strike <= 0 or t_years <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(spot, strike, t_years, sigma, r)
    return spot * norm_pdf(d1) * math.sqrt(t_years)


def bs_theta(spot: float, strike: float, t_years: float,
             sigma: float, r: float = 0.0,
             option_type: OptionType = "call") -> float:
    """
    Annualized theta (loss per year of remaining life). Divide by 365
    if you want per-day. We return the annualized number to match
    QuantLib / Hull conventions; the runner converts to per-day for the
    daily P&L decomposition where needed.
    """
    if spot <= 0 or strike <= 0 or t_years <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _d1_d2(spot, strike, t_years, sigma, r)
    sqrt_t = math.sqrt(t_years)
    discount = math.exp(-r * t_years)
    common = -(spot * norm_pdf(d1) * sigma) / (2.0 * sqrt_t)
    if option_type == "call":
        return common - r * strike * discount * norm_cdf(d2)
    return common + r * strike * discount * norm_cdf(-d2)


# --------------------------------------------------------------------------
# Implied volatility (Newton-Raphson with bisection fallback)
# --------------------------------------------------------------------------

def implied_vol(option_price: float, spot: float, strike: float,
                t_years: float, r: float = 0.0,
                option_type: OptionType = "call",
                *,
                tol: float = 1e-6, max_iter: int = 50,
                lo: float = 1e-4, hi: float = 5.0) -> float:
    """
    Solve for σ such that ``bs_price(σ) ≈ option_price``.

    Returns ``0.0`` for degenerate inputs (non-positive price/spot/strike,
    expired option, price below intrinsic). On Newton divergence falls
    back to a bisection over ``[lo, hi]`` so we always return *some*
    sane number. ``hi=5.0`` (= 500 % vol) is wide enough for crisis
    regimes; bisection truncates if the true σ is outside.
    """
    if option_price <= 0 or spot <= 0 or strike <= 0 or t_years <= 0:
        return 0.0
    intrinsic = (
        max(0.0, spot - strike) if option_type == "call"
        else max(0.0, strike - spot)
    )
    if option_price < intrinsic:
        # Quoted price is below intrinsic — usually a stale or crossed
        # quote. No real σ inverts this; return 0 so caller can flag it.
        return 0.0

    # Newton-Raphson seeded at σ ≈ √(2π/T) × P/S — Brenner-Subrahmanyam
    # closed-form for ATM, fine warm-start elsewhere.
    sigma = max(lo, math.sqrt(2.0 * math.pi / t_years) * option_price / spot)
    for _ in range(max_iter):
        price = bs_price(spot, strike, t_years, sigma, r, option_type)
        diff = price - option_price
        if abs(diff) < tol:
            return max(lo, sigma)
        v = bs_vega(spot, strike, t_years, sigma, r)
        if v <= 1e-12:
            break  # Newton step would explode; fall through to bisection.
        sigma -= diff / v
        if sigma <= lo or sigma >= hi:
            break  # Stepped out of bracket; bisect instead.

    # Bisection fallback — guaranteed to converge inside [lo, hi].
    p_lo = bs_price(spot, strike, t_years, lo, r, option_type) - option_price
    p_hi = bs_price(spot, strike, t_years, hi, r, option_type) - option_price
    if p_lo * p_hi > 0:
        # Target price not inside bracket — return whichever endpoint is closer.
        return lo if abs(p_lo) < abs(p_hi) else hi
    a, b = lo, hi
    for _ in range(60):  # 60 bisection halvings → 1e-18 width
        m = 0.5 * (a + b)
        p_m = bs_price(spot, strike, t_years, m, r, option_type) - option_price
        if abs(p_m) < tol:
            return m
        if p_lo * p_m < 0:
            b = m
            p_hi = p_m
        else:
            a = m
            p_lo = p_m
    return 0.5 * (a + b)


__all__ = [
    "norm_cdf",
    "norm_pdf",
    "bs_price",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    "bs_theta",
    "implied_vol",
]
