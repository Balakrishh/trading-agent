"""
rsi_gate.py — refine the strategy choice using RSI alongside the regime.

Why this exists
---------------
The regime classifier emits one of four labels — bullish / bearish /
sideways / mean_reversion — and the planner mechanically maps each to
a single strategy:

    bullish        → Bull Put Spread
    bearish        → Bear Call Spread
    sideways       → Iron Condor
    mean_reversion → Mean Reversion Spread

That works most of the time, but "sideways" is a wide bucket. A truly
balanced market (RSI ≈ 50) and a sideways drift with momentum still
intact (RSI = 60+) both classify as "sideways" — yet the latter has
materially more upside risk for the call wing of an Iron Condor. The
RSI gate adds one extra check between Phase II (CLASSIFY) and Phase III
(PLAN): given the regime AND the current RSI, what's the *best* action
this cycle?

Decision matrix
---------------
| Regime         | RSI band    | Action                        |
|----------------|-------------|-------------------------------|
| sideways       | 45-55       | Iron Condor (full both wings) |
| sideways       | 55-65       | Bull Put only (skip call wing) |
| sideways       | 35-45       | Bear Call only (skip put wing) |
| sideways       | <35 or >65  | SKIP — momentum too active   |
| bullish        | <70         | Bull Put Spread (proceed)     |
| bullish        | ≥70         | SKIP — overbought, reversal risk |
| bearish        | >30         | Bear Call Spread (proceed)    |
| bearish        | ≤30         | SKIP — oversold, reversal risk   |
| mean_reversion | any         | proceed (the 3-σ touch IS the signal — RSI noise irrelevant) |

The "skip" outcomes return an explicit reason so the journal records
*why* the cycle abstained, not just that it did.

How callers use it
------------------
``evaluate_rsi_gate(regime, rsi)`` returns an ``RsiGateDecision`` whose
fields tell ``_process_ticker``:

  * ``allow=False``  → log skipped_rsi_gate, abort this ticker's cycle
  * ``allow=True, override_regime=None``        → no change, proceed
  * ``allow=True, override_regime=Regime.X``    → swap the regime on a
    cloned ``RegimeAnalysis`` before calling the planner; the planner
    then maps it to its single-side strategy.

Pure-function design — no side effects, no I/O, no dataclass mutation.
This makes the gate trivially unit-testable across the full decision
matrix and easy to A/B against the no-gate path in the backtester
(toggle via the ``RSI_GATE_ENABLED`` env var).

Boundary conventions
--------------------
RSI bands are stated in plain English ("45-55") but the implementation
uses inclusive lower / exclusive upper to ensure every real RSI value
falls into exactly one band — no value sits in two bands at once and
none falls through the cracks. The boundary tests in
``tests/test_rsi_gate.py`` lock these conventions down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from trading_agent.regime import Regime


# Band boundaries — inclusive lower, exclusive upper (i.e. [lo, hi))
# so every real RSI ∈ [0, 100] maps to exactly one band.
_SIDEWAYS_NEUTRAL_LO     = 45.0
_SIDEWAYS_NEUTRAL_HI     = 55.0   # [45, 55) → Iron Condor
_SIDEWAYS_LEAN_BULL_HI   = 65.0   # [55, 65) → Bull Put only
_SIDEWAYS_LEAN_BEAR_LO   = 35.0   # [35, 45) → Bear Call only
# Outside [35, 65) on a sideways regime → skip (momentum too active).

_BULLISH_OVERBOUGHT      = 70.0   # bullish + RSI ≥ 70  → skip
_BEARISH_OVERSOLD        = 30.0   # bearish + RSI ≤ 30  → skip


@dataclass(frozen=True)
class RsiGateDecision:
    """The verdict the gate hands back to ``_process_ticker``.

    Attributes
    ----------
    allow : bool
        True → proceed with planning. False → skip this ticker's cycle.
    override_regime : Optional[Regime]
        If set, the planner should be called against this regime
        instead of the classifier's original. Used only to downgrade
        a sideways regime into a single-side vertical. None means
        "leave the regime alone — proceed with the classified value".
    reason : str
        Plain-English description used by the journal logger when
        skip is logged or an override is applied. Always present so
        the journal row has audit context regardless of branch.
    """
    allow: bool
    override_regime: Optional[Regime]
    reason: str


def evaluate_rsi_gate(regime: Regime, rsi: float) -> RsiGateDecision:
    """Evaluate the RSI gate for the given (regime, rsi) pair.

    Pure function — no state, no side effects. See module docstring
    for the decision matrix.

    Parameters
    ----------
    regime : Regime
        The classifier's regime label for this cycle.
    rsi : float
        14-period RSI value, expected in [0, 100]. Out-of-range values
        are tolerated (passed through to the band logic) — the gate
        will still produce a deterministic decision.

    Returns
    -------
    RsiGateDecision
        See class docstring.
    """
    # Mean-reversion bypasses the gate entirely. The 3-σ Bollinger Band
    # touch IS the entry signal for that strategy — it's a fear-spike
    # condition where the *direction* of the bands matters, not RSI.
    # Routing it through the gate would just second-guess a well-defined
    # signal with a noisier one.
    if regime == Regime.MEAN_REVERSION:
        return RsiGateDecision(
            allow=True,
            override_regime=None,
            reason="Mean-reversion bypasses RSI gate (3-σ touch is the signal)",
        )

    # Bullish regime — only enter if RSI hasn't already pushed into
    # overbought territory. RSI ≥ 70 means the bullish move is mature
    # and reversal risk dominates; selling puts (Bull Put = bullish
    # credit play) at that point is fighting the next leg down.
    if regime == Regime.BULLISH:
        if rsi >= _BULLISH_OVERBOUGHT:
            return RsiGateDecision(
                allow=False,
                override_regime=None,
                reason=(
                    f"RSI={rsi:.1f} ≥ {_BULLISH_OVERBOUGHT:.0f} (overbought) — "
                    "skipping Bull Put: too late in the up-move, "
                    "reversal risk outweighs theta"
                ),
            )
        return RsiGateDecision(
            allow=True,
            override_regime=None,
            reason=f"Bullish regime, RSI={rsi:.1f} (< {_BULLISH_OVERBOUGHT:.0f}) — proceed with Bull Put",
        )

    # Bearish regime — symmetric to bullish. RSI ≤ 30 means the
    # down-move is mature; selling calls now invites the bounce.
    if regime == Regime.BEARISH:
        if rsi <= _BEARISH_OVERSOLD:
            return RsiGateDecision(
                allow=False,
                override_regime=None,
                reason=(
                    f"RSI={rsi:.1f} ≤ {_BEARISH_OVERSOLD:.0f} (oversold) — "
                    "skipping Bear Call: too late in the down-move, "
                    "bounce risk outweighs theta"
                ),
            )
        return RsiGateDecision(
            allow=True,
            override_regime=None,
            reason=f"Bearish regime, RSI={rsi:.1f} (> {_BEARISH_OVERSOLD:.0f}) — proceed with Bear Call",
        )

    # Sideways regime — the meat of the gate. Decompose into four bands:
    if regime == Regime.SIDEWAYS:
        if _SIDEWAYS_NEUTRAL_LO <= rsi < _SIDEWAYS_NEUTRAL_HI:
            return RsiGateDecision(
                allow=True,
                override_regime=None,
                reason=(
                    f"Sideways + RSI={rsi:.1f} ∈ [{_SIDEWAYS_NEUTRAL_LO:.0f}, "
                    f"{_SIDEWAYS_NEUTRAL_HI:.0f}) — true neutral, full Iron Condor"
                ),
            )
        if _SIDEWAYS_NEUTRAL_HI <= rsi < _SIDEWAYS_LEAN_BULL_HI:
            return RsiGateDecision(
                allow=True,
                override_regime=Regime.BULLISH,  # → planner picks Bull Put
                reason=(
                    f"Sideways + RSI={rsi:.1f} ∈ [{_SIDEWAYS_NEUTRAL_HI:.0f}, "
                    f"{_SIDEWAYS_LEAN_BULL_HI:.0f}) — lean-bullish, downgrading to "
                    "Bull Put only (skip the call wing — momentum at risk)"
                ),
            )
        if _SIDEWAYS_LEAN_BEAR_LO <= rsi < _SIDEWAYS_NEUTRAL_LO:
            return RsiGateDecision(
                allow=True,
                override_regime=Regime.BEARISH,  # → planner picks Bear Call
                reason=(
                    f"Sideways + RSI={rsi:.1f} ∈ [{_SIDEWAYS_LEAN_BEAR_LO:.0f}, "
                    f"{_SIDEWAYS_NEUTRAL_LO:.0f}) — lean-bearish, downgrading to "
                    "Bear Call only (skip the put wing — momentum at risk)"
                ),
            )
        # rsi < 35 or rsi >= 65 — momentum too active for a sideways play
        return RsiGateDecision(
            allow=False,
            override_regime=None,
            reason=(
                f"Sideways + RSI={rsi:.1f} outside [{_SIDEWAYS_LEAN_BEAR_LO:.0f}, "
                f"{_SIDEWAYS_LEAN_BULL_HI:.0f}) — momentum too active for "
                "an Iron Condor; skipping until RSI normalises"
            ),
        )

    # Unknown regime — fail-safe to allow (gate shouldn't override
    # behaviour for regimes it doesn't understand).
    return RsiGateDecision(
        allow=True,
        override_regime=None,
        reason=f"Unknown regime ({regime}) — gate has no opinion, proceeding",
    )


__all__ = ["RsiGateDecision", "evaluate_rsi_gate"]
