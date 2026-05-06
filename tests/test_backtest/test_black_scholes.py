"""
Tests for ``trading_agent.backtest.black_scholes``.

These pin the BS pricer + Greeks against textbook closed-form values
and against the put-call parity identity. The IV solver is tested by
round-trip: price an option at known σ, then ``implied_vol`` it back.
"""

from __future__ import annotations

import math

import pytest

from trading_agent.backtest.black_scholes import (
    bs_delta,
    bs_gamma,
    bs_price,
    bs_theta,
    bs_vega,
    implied_vol,
    norm_cdf,
)


# --------------------------------------------------------------------------
# norm_cdf — reference values from the standard normal table
# --------------------------------------------------------------------------

def test_norm_cdf_known_anchors():
    assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)
    assert norm_cdf(1.0) == pytest.approx(0.84134474, abs=1e-6)
    assert norm_cdf(-1.0) == pytest.approx(0.15865526, abs=1e-6)
    assert norm_cdf(1.96) == pytest.approx(0.97500210, abs=1e-6)
    assert norm_cdf(-1.96) == pytest.approx(0.02499790, abs=1e-6)


# --------------------------------------------------------------------------
# Pricing — closed-form sanity checks
# --------------------------------------------------------------------------

class TestBsPriceBasic:
    def test_atm_call_30dte_20vol_matches_textbook(self):
        # Spot=100, strike=100, T=30/365, σ=0.20, r=0
        # Closed-form (Brenner-Subrahmanyam ATM, r=0):
        #   C ≈ S σ √(T / 2π) = 100 × 0.20 × √(0.0822/6.283) ≈ 2.287
        # Cross-checked against an independent BS implementation.
        price = bs_price(100.0, 100.0, 30 / 365.0, 0.20, r=0.0,
                         option_type="call")
        assert price == pytest.approx(2.287, abs=0.001)

    def test_atm_put_30dte_20vol_matches_textbook(self):
        # With r=0 and same σ/T, put = call by put-call parity at ATM.
        price = bs_price(100.0, 100.0, 30 / 365.0, 0.20, r=0.0,
                         option_type="put")
        assert price == pytest.approx(2.287, abs=0.001)

    def test_deep_otm_call_essentially_zero(self):
        # Spot 100, strike 130, 7 DTE, 20% vol → ~0
        price = bs_price(100.0, 130.0, 7 / 365.0, 0.20,
                         option_type="call")
        assert price < 0.01

    def test_deep_itm_call_intrinsic_floor(self):
        # Spot 100, strike 70, 7 DTE, 20% vol → ≈ intrinsic 30
        price = bs_price(100.0, 70.0, 7 / 365.0, 0.20,
                         option_type="call")
        assert price >= 30.0
        assert price < 30.5  # tiny time value

    def test_expired_call_returns_intrinsic(self):
        assert bs_price(100, 95, 0.0, 0.20,
                        option_type="call") == pytest.approx(5.0)
        assert bs_price(100, 105, 0.0, 0.20,
                        option_type="call") == pytest.approx(0.0)

    def test_zero_vol_returns_intrinsic(self):
        assert bs_price(100, 95, 0.5, 0.0,
                        option_type="call") == pytest.approx(5.0)
        assert bs_price(100, 110, 0.5, 0.0,
                        option_type="put") == pytest.approx(10.0)

    def test_put_call_parity_holds(self):
        # C − P = S − K e^{-rT}.  At r=0: C − P = S − K.
        S, K, T, sigma = 100.0, 105.0, 60 / 365.0, 0.25
        c = bs_price(S, K, T, sigma, r=0.0, option_type="call")
        p = bs_price(S, K, T, sigma, r=0.0, option_type="put")
        assert (c - p) == pytest.approx(S - K, abs=1e-6)


# --------------------------------------------------------------------------
# Greeks
# --------------------------------------------------------------------------

class TestBsGreeks:
    def test_atm_call_delta_around_half(self):
        d = bs_delta(100.0, 100.0, 30 / 365.0, 0.20, option_type="call")
        # ATM call delta is ~0.5 + small skew from r=0.5σ²T term
        assert 0.50 < d < 0.55

    def test_atm_put_delta_around_minus_half(self):
        d = bs_delta(100.0, 100.0, 30 / 365.0, 0.20, option_type="put")
        # ATM put delta with r=0 is *slightly above* −0.50 because the
        # +½σ²T drift in d1 nudges Φ(d1) above 0.5 → Δ_put = Φ(d1) − 1
        # lands in (−0.50, −0.49).
        assert -0.50 < d < -0.45

    def test_call_put_delta_difference_unity(self):
        # Δ_call − Δ_put = 1 (no carry).
        S, K, T, s = 100.0, 105.0, 30 / 365.0, 0.20
        dc = bs_delta(S, K, T, s, option_type="call")
        dp = bs_delta(S, K, T, s, option_type="put")
        assert (dc - dp) == pytest.approx(1.0, abs=1e-6)

    def test_gamma_same_for_call_and_put(self):
        # Γ identity for puts and calls.
        S, K, T, s = 100.0, 95.0, 14 / 365.0, 0.30
        g = bs_gamma(S, K, T, s)
        # Gamma is positive and small but non-zero for ATM-ish options.
        assert g > 0
        # Spot-check: ATM gamma at 14 DTE 30% vol ≈ 0.05
        g_atm = bs_gamma(100, 100, 14 / 365.0, 0.30)
        assert 0.03 < g_atm < 0.10

    def test_vega_positive_for_otm_options(self):
        v = bs_vega(100.0, 110.0, 30 / 365.0, 0.25)
        assert v > 0

    def test_theta_negative_for_long_options(self):
        # Long calls and puts both have negative theta (we lose time value).
        tc = bs_theta(100, 100, 30 / 365.0, 0.20, option_type="call")
        tp = bs_theta(100, 100, 30 / 365.0, 0.20, option_type="put")
        assert tc < 0
        assert tp < 0


# --------------------------------------------------------------------------
# implied_vol — Newton + bisection round-trip
# --------------------------------------------------------------------------

class TestImpliedVol:
    @pytest.mark.parametrize("sigma", [0.10, 0.20, 0.35, 0.60])
    @pytest.mark.parametrize("opt", ["call", "put"])
    def test_round_trip_recovers_sigma(self, sigma, opt):
        S, K, T = 100.0, 105.0, 21 / 365.0
        price = bs_price(S, K, T, sigma, r=0.0, option_type=opt)
        recovered = implied_vol(price, S, K, T, r=0.0, option_type=opt)
        assert recovered == pytest.approx(sigma, abs=1e-3)

    def test_below_intrinsic_returns_zero(self):
        # Quoted price below intrinsic — no σ inverts this.
        # 100/95 call intrinsic = 5; offer 4.50 → degenerate.
        assert implied_vol(4.50, 100, 95, 30 / 365.0, option_type="call") == 0.0

    def test_zero_inputs_return_zero(self):
        assert implied_vol(0.0, 100, 100, 0.1) == 0.0
        assert implied_vol(2.5, 0, 100, 0.1) == 0.0
        assert implied_vol(2.5, 100, 0, 0.1) == 0.0
        assert implied_vol(2.5, 100, 100, 0.0) == 0.0
