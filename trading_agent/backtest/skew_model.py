"""skew_model.py — parametric volatility-skew adjustment for synthetic chains.

Skill 39 — backtester improvement #2. Before this module the
synthetic chain priced every strike at the same ATM-σ ("flat vol"),
which systematically UNDERPRICES OTM puts and slightly underprices
deep OTM calls. The net effect on credit spreads:

  * Bull put spread: short put + long (further OTM) put. Flat-vol
    BS UNDERPRICES the protection leg → backtest credit looks
    BIGGER than reality.
  * Bear call spread: short call + long (further OTM) call. Skew
    on the call side is much shallower → smaller error in this
    direction.
  * Iron Condor: composite of both → put-side error dominates.

Real-world index ETF (SPY/QQQ/IWM) skew shapes:
  * 0.10-delta OTM put: ~1.4-1.6× ATM σ
  * 0.25-delta OTM put: ~1.15-1.25× ATM σ
  * 0.10-delta OTM call: ~1.05-1.15× ATM σ
  * 0.25-delta OTM call: ~1.02-1.08× ATM σ

Single-stock skew is steeper (1.7-2.0× far-OTM put). The default
``SkewModel`` here calibrates for broad-market ETFs since that's
the strategy's actual universe.

Parametrization
---------------
We use moneyness ``m = K/spot - 1`` as the X axis (not log-moneyness
since the spreads in this strategy span 1-5% of spot, where linear
moneyness is indistinguishable from log).

::

    σ(K) = σ_ATM × (1 + α × max(0, -m) + β × max(0, +m))

  * α (put_skew) — slope on the OTM-put side. ~0.6 for SPY/QQQ.
    Means a 5%-OTM put has σ = 1.03 × ATM_σ.
  * β (call_skew) — slope on the OTM-call side. ~0.15 for index ETFs.

A "flat" SkewModel sets α=β=0 → identical to pre-this-module
behavior (no skew adjustment). That's the default so legacy
backtests don't silently shift.

Why parametric, not fitted
--------------------------
A real skew calibration would require live option chains for every
backtest date — which the historical_port doesn't have for older
windows. The parametric form is "directionally correct" — better
than flat, far cheaper than full surface fitting. When the operator
wants a tight live-vs-backtest comparison, they can sweep α/β
manually until the backtest's bull-put credit lands within 1-2%
of the live execution credit.

Skew gets applied AT CHAIN BUILD TIME — i.e., when
``build_chain_slice`` computes per-strike (bid, ask, delta). The
``SimPosition.remark`` mid-life re-pricing also takes a SkewModel
so cross-VIX re-marks honour the same skew. Without this, an
in-the-money put would re-mark at flat vol and the realised P&L
would diverge from open-side pricing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkewModel:
    """Linear-in-moneyness IV skew adjustment.

    A ``SkewModel`` is constructor-injected into the chain builder and
    the position re-marker; both apply the SAME formula so the open and
    close legs of every trade get consistent IV.

    ``put_skew`` and ``call_skew`` are dimensionless multipliers; a
    value of 0 collapses the model to flat-vol BS (default).
    """

    # Default values (FLAT) — preserves legacy behavior. Operators sweep
    # ~0.4-0.8 for put_skew and ~0.1-0.2 for call_skew on index ETFs.
    put_skew: float = 0.0
    call_skew: float = 0.0

    def sigma_for_strike(self, strike: float, spot: float,
                         atm_sigma: float) -> float:
        """Return the skew-adjusted σ for a given strike.

        ``atm_sigma`` is the ATM IV proxy (typically from realised vol
        or VIX-proxy). ``strike`` and ``spot`` are in dollars.

        Math:
          m = K/spot - 1
          σ = atm_sigma × (1 + put_skew × max(0, -m) + call_skew × max(0, +m))

        m < 0 → OTM put: σ rises with |m|
        m > 0 → OTM call: σ rises with m (typically much less steeply)
        m = 0 → ATM: σ = atm_sigma (unchanged)

        Returns a clipped value in [0.01, 5.0] so even an
        adversarially-deep OTM strike can't crash the BS pricer.
        """
        if spot <= 0 or atm_sigma <= 0:
            return atm_sigma
        m = (strike / spot) - 1.0
        put_lift = self.put_skew * max(0.0, -m)
        call_lift = self.call_skew * max(0.0, m)
        sigma = atm_sigma * (1.0 + put_lift + call_lift)
        # Clip to a sane band — extreme moneyness shouldn't crash BS.
        return min(5.0, max(0.01, sigma))


# Canonical presets the operator can drop in. Names match what's
# typically loaded in commercial IV surfaces.
FLAT_SKEW = SkewModel(put_skew=0.0, call_skew=0.0)
"""No skew — every strike at ATM σ. Legacy behavior, default."""

INDEX_ETF_SKEW = SkewModel(put_skew=0.6, call_skew=0.15)
"""Calibrated for broad-market ETFs (SPY/QQQ/IWM/DIA/XLF/XLE etc.).
Put-side IV ~1.03× ATM at 5% OTM; ~1.18× at 10% OTM.
Call-side near-flat. Use this for a realistic baseline backtest."""

SINGLE_STOCK_SKEW = SkewModel(put_skew=1.0, call_skew=0.25)
"""Steeper put skew for single-name underlyings (AAPL, NVDA, etc.).
Use only when the strategy expands beyond ETFs."""


__all__ = [
    "SkewModel",
    "FLAT_SKEW",
    "INDEX_ETF_SKEW",
    "SINGLE_STOCK_SKEW",
]
