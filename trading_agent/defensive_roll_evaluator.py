"""defensive_roll_evaluator.py — six-predicate gate for defensive rolls.

Skill 31 — when an open spread's short strike approaches spot, the
``STRIKE_PROXIMITY`` exit signal fires and the position is closed at a
loss. The defensive-roll feature offers an alternative: close the
threatened spread AND open a new further-OTM spread in the same atomic
operation, capturing a fresh credit that offsets some or all of the
close debit.

This module is the **pure decision logic** — given an open spread and a
candidate replacement plan, return a ``RollDecision`` describing whether
all six predicates pass. No I/O, no broker calls, no journal writes.
The orchestrator (``agent.py``) owns side-effects.

Six predicates (all must pass for ``RollDecision.ROLL``):

  1. **Trigger band.** Short strike is between ``roll_trigger_min_pct``
     and ``roll_trigger_max_pct`` of spot. Below the lower bound, the
     new strike is also in trouble; above the upper bound the original
     position isn't actually threatened enough to roll.
  2. **DTE viability.** Existing position has ≥ ``min_dte_for_roll``
     days left. Closer than that, gamma is too explosive to roll usefully —
     just close and accept the loss.
  3. **Credit-positive.** Projected credit from the new spread ≥ debit
     to close the old spread. A net-debit roll just doubles down on the
     loss.
  4. **Floor compliance.** The new spread passes the
     ``|Δshort| × (1 + edge_buffer)`` C/W floor used at open (skill 3
     invariant). A roll into a sub-floor spread is a bug.
  5. **Same direction.** New spread is the same strategy type
     (Bull Put → Bull Put). Switching directions mid-defense is a
     regime change, not a roll.
  6. **Roll budget.** ``current_roll_count <
     max_defensive_rolls_per_position`` (default 1). Without a cap,
     a trending market would roll the same position 3–4 times,
     compounding the original loss.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


class RollDecision(str, enum.Enum):
    """Outcome of evaluating a defensive-roll opportunity.

    Inherits from ``str`` so the value round-trips cleanly through the
    journal (which is JSON-serialised).
    """

    ROLL = "roll"
    SKIP_NOT_IN_TRIGGER_BAND = "skip_not_in_trigger_band"
    SKIP_DTE_TOO_SHORT = "skip_dte_too_short"
    SKIP_CREDIT_NEGATIVE = "skip_credit_negative"
    SKIP_BELOW_CW_FLOOR = "skip_below_cw_floor"
    SKIP_DIFFERENT_STRATEGY = "skip_different_strategy"
    SKIP_BUDGET_EXHAUSTED = "skip_budget_exhausted"


@dataclass(frozen=True)
class RollEvalInputs:
    """All inputs the evaluator needs — no I/O hidden inside.

    Frozen so a caller can't mutate the snapshot after the decision is
    rendered (defensive against the orchestrator passing live state by
    reference and mutating it during the journal write).
    """

    # Existing position
    spot:                       float
    short_strike:               float
    dte:                        int
    strategy_name:              str        # "Bull Put", "Bear Call", "Iron Condor"
    debit_to_close:             float      # dollars per spread (positive number)
    current_roll_count:         int        # how many defensive rolls already applied

    # Candidate replacement
    new_short_delta:            float      # |Δ| of the new short leg
    new_strategy_name:          str        # must match strategy_name
    new_projected_credit:       float      # dollars per spread (positive number)
    new_spread_width:            float     # dollars
    new_cw_ratio:               float      # new_projected_credit / new_spread_width

    # Preset thresholds (snapshotted so the evaluator stays pure)
    roll_trigger_min_pct:       float
    roll_trigger_max_pct:       float
    min_dte_for_roll:           int
    max_defensive_rolls_per_position: int
    edge_buffer:                float      # |Δshort|×(1+edge_buffer) floor

    def proximity_pct(self) -> float:
        """Fraction of spot the short strike sits from spot. Always ≥ 0."""
        if self.spot <= 0:
            return 0.0
        return abs(self.spot - self.short_strike) / self.spot


def evaluate_defensive_roll(inp: RollEvalInputs) -> RollDecision:
    """Return ``RollDecision`` describing the six-predicate gate.

    Predicates evaluate in a fixed order so a single failure produces a
    deterministic, named outcome — the journal records which predicate
    blocked the roll so the operator can tune thresholds with data.

    >>> # Trigger band check first — proximity at 1.0% with band 0.5–1.5%
    >>> # would pass that gate but might still fail downstream.
    """
    # ─ Predicate 1: trigger band ───────────────────────────────────────
    prox = inp.proximity_pct()
    if not (inp.roll_trigger_min_pct <= prox <= inp.roll_trigger_max_pct):
        return RollDecision.SKIP_NOT_IN_TRIGGER_BAND

    # ─ Predicate 2: DTE viability ─────────────────────────────────────
    if inp.dte < inp.min_dte_for_roll:
        return RollDecision.SKIP_DTE_TOO_SHORT

    # ─ Predicate 6: roll budget (checked before any candidate math
    #   because a budget-exhausted position can never roll regardless
    #   of the candidate's quality — fail fast).
    if inp.current_roll_count >= inp.max_defensive_rolls_per_position:
        return RollDecision.SKIP_BUDGET_EXHAUSTED

    # ─ Predicate 5: same direction ────────────────────────────────────
    # Compare case-insensitively so "Bull Put" / "bull_put" / "BULL_PUT"
    # all match. The journal historically writes the human form,
    # decision_engine writes the snake form.
    def _norm(s: str) -> str:
        return s.lower().replace(" ", "_").strip()
    if _norm(inp.strategy_name) != _norm(inp.new_strategy_name):
        return RollDecision.SKIP_DIFFERENT_STRATEGY

    # ─ Predicate 3: credit-positive ───────────────────────────────────
    # "Net credit" = new credit collected − debit to close the old.
    # Must be strictly positive; equal-to-close just churns commission.
    if inp.new_projected_credit <= inp.debit_to_close:
        return RollDecision.SKIP_CREDIT_NEGATIVE

    # ─ Predicate 4: floor compliance ──────────────────────────────────
    # Same |Δshort|×(1+edge_buffer) floor used at open (skill 3 invariant).
    # If the new candidate is below the floor, it's not a trade we'd
    # take fresh — refuse to roll into it.
    required_cw = abs(inp.new_short_delta) * (1.0 + inp.edge_buffer)
    if inp.new_cw_ratio < required_cw:
        return RollDecision.SKIP_BELOW_CW_FLOOR

    return RollDecision.ROLL


__all__ = [
    "RollDecision",
    "RollEvalInputs",
    "evaluate_defensive_roll",
]
