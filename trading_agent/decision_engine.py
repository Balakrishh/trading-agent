"""
decision_engine.py — pure spread-scoring engine, shared by live and backtest.

This module owns the *deterministic* part of "given a chain, which spread
should we trade?". No I/O, no calendar lookups, no broker calls — everything
the engine needs is in its ``DecisionInput``. Both the live ``ChainScanner``
and the backtest ``run_one_cycle`` (under ``trading_agent/backtest/``)
construct an input, call ``decide()``, and get back the same shape: a
ranked list of ``SpreadCandidate`` plus a ``ScanDiagnostics`` block
describing why anything was rejected.

Why this lives in its own module:

* It is the **single source of truth** for scoring. ``chain_scanner.py``
  owns the pure pricing/EV helpers (``_quote_credit``, ``_score_candidate``,
  …); ``decision_engine.py`` composes them into the per-(Δ × width) sweep.
  The live scanner and the backtester both call ``decide()`` so the math
  cannot drift between them by construction.

* The ``scan_invariant_check.py`` AST walker can statically guarantee
  that no other module re-implements the score loop — it walks every
  module under ``trading_agent/`` and asserts the only place
  ``_score_candidate_with_reason`` is *called inside a per-grid-point
  loop* is here.

* It composes cleanly with the dataclass types defined in
  ``chain_scanner.py``: there is **no circular import**. The dependency
  arrow points one way: ``decision_engine`` imports from
  ``chain_scanner``; ``chain_scanner.ChainScanner`` (the live wrapper)
  is a *client* of ``decision_engine.decide``.

Public API
----------
``ChainSlice``
    One expiration's worth of contract dicts plus its DTE.

``DecisionInput``
    Everything ``decide()`` needs: side, chain slices, preset.

``DecisionOutput``
    Ranked candidates + diagnostics.

``decide(input, *, max_candidates=10) -> DecisionOutput``
    The pure scoring entrypoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from trading_agent.chain_scanner import (
    REJECT_CW_BELOW_FLOOR,
    REJECT_LEG_SPREAD_WIDE,
    REJECT_NO_CHAIN,
    REJECT_NO_LONG_CONTRACT,
    REJECT_NO_SHORT_CONTRACT,
    REJECT_NON_POSITIVE_WIDTH,
    ChainScanner,
    ScanDiagnostics,
    SpreadCandidate,
    _leg_spread_too_wide,
    _pop_from_delta,
    _quote_credit,
    _quote_credit_single,
    _score_candidate_with_reason,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision input/output types
# ---------------------------------------------------------------------------

@dataclass
class ChainSlice:
    """
    One expiration's worth of chain data, normalised to the dict shape the
    engine expects: ``[{strike, delta, bid, ask, symbol}, ...]``.

    The engine doesn't care where the contracts came from — Alpaca live
    snapshots, Alpaca historical, yfinance, or a hand-rolled fixture in a
    parity test. As long as each contract has ``strike`` (positive),
    ``delta`` (signed), ``bid``/``ask`` (≥ 0), and ``symbol`` (str),
    ``decide()`` will score it.
    """
    expiration: str       # ISO date string, e.g. "2026-05-15"
    dte:        int       # days from "today" to expiration (must be > 0)
    contracts:  List[Dict[str, Any]]


@dataclass
class DecisionInput:
    """
    Bundle every input ``decide()`` needs. Pure data — no callbacks, no
    market-data providers. ``preset`` carries the grids and floors
    (``delta_grid``, ``width_grid_pct``, ``edge_buffer``, ``min_pop``).
    """
    side:          str                   # "bull_put" | "bear_call"
    chain_slices:  List[ChainSlice]
    preset:        Any                   # PresetConfig — duck-typed to avoid an import cycle


@dataclass
class DecisionOutput:
    """Ranked candidates + diagnostics from one ``decide()`` call."""
    candidates:  List[SpreadCandidate] = field(default_factory=list)
    diagnostics: ScanDiagnostics = field(
        default_factory=lambda: ScanDiagnostics(grid_points_total=0)
    )


# ---------------------------------------------------------------------------
# decide() — the pure engine
# ---------------------------------------------------------------------------

def decide(inp: DecisionInput, *, max_candidates: int = 10) -> DecisionOutput:
    """
    Run the (Δ × width) sweep over each ``ChainSlice`` and return ranked
    candidates plus a populated ``ScanDiagnostics``. No I/O.

    The returned ``DecisionOutput.candidates`` list is sorted by
    ``annualized_score`` desc (with absolute credit as tiebreak), then
    truncated to ``max_candidates`` so callers don't pay rendering cost
    on a 50-row list when only the top pick matters.

    The diagnostics block is populated *even when zero candidates pass*:
    ``rejects_by_reason`` enumerates which filter ate each grid point,
    and ``best_near_miss`` quotes the highest-EV candidate that *only*
    failed the C/W floor — actionable signal for tuning ``edge_buffer``.
    """
    side = inp.side
    if side not in ("bull_put", "bear_call"):
        raise ValueError(f"Unsupported side {side!r}")

    preset = inp.preset
    delta_grid    = list(preset.delta_grid)
    width_grid    = list(preset.width_grid_pct)
    edge_buffer   = float(preset.edge_buffer)
    min_pop       = float(preset.min_pop)
    # Per-leg liquidity gate (skill 29). Pulled once outside the inner
    # loop so we don't repeatedly read the same fields per grid point.
    max_leg_spread_cents   = float(getattr(preset, "max_leg_spread_cents", 0.15))
    max_leg_spread_pct_mid = float(getattr(preset, "max_leg_spread_pct_mid", 0.05))

    n_dte   = len(inp.chain_slices)
    n_delta = len(delta_grid)
    n_width = len(width_grid)
    diag = ScanDiagnostics(
        grid_points_total=max(1, len(list(preset.dte_grid))) * n_delta * n_width,
        expirations_resolved=n_dte,
    )
    best_near_miss: Optional[Dict[str, Any]] = None
    candidates: List[SpreadCandidate] = []

    for slc in inp.chain_slices:
        chain = slc.contracts
        if not chain:
            diag.record(REJECT_NO_CHAIN, n_delta * n_width)
            continue

        spot_proxy = ChainScanner._infer_spot_proxy(chain)
        grid_step  = ChainScanner._infer_grid_step(chain)

        for target_delta in delta_grid:
            short_contract = ChainScanner._find_short(chain, float(target_delta))
            if short_contract is None:
                diag.record(REJECT_NO_SHORT_CONTRACT, n_width)
                continue
            short_strike = float(short_contract["strike"])

            for width_pct in width_grid:
                raw_width = float(width_pct) * spot_proxy
                width = ChainScanner._snap_width_to_grid(raw_width, grid_step)
                if width <= 0:
                    diag.record(REJECT_NON_POSITIVE_WIDTH)
                    continue
                long_strike = (short_strike - width if side == "bull_put"
                               else short_strike + width)
                long_contract = ChainScanner._find_strike(chain, long_strike)
                if long_contract is None:
                    diag.record(REJECT_NO_LONG_CONTRACT)
                    continue
                actual_width = abs(short_strike - float(long_contract["strike"]))
                if actual_width <= 0:
                    diag.record(REJECT_NON_POSITIVE_WIDTH)
                    continue

                # ── Per-leg liquidity gate (skill 29) ─────────────────
                # Before estimating credit / scoring EV, verify both
                # legs have tight enough bid-ask that the day-1 mark
                # won't eat half the credit. The 2026-05-15 GLD trade
                # (35¢ spread on $5 short put, 6.6% of mid) opened
                # at -$115 P&L because nothing here rejected it. With
                # this gate it would have been skipped → no panic on
                # the dashboard, no surprise -54% display, and the
                # adaptive scanner moves on to a tighter strike pair.
                # AND-of-two-thresholds means penny-cheap legs still
                # pass; only structurally wide legs are rejected.
                if _leg_spread_too_wide(
                    float(short_contract["bid"]), float(short_contract["ask"]),
                    max_leg_spread_cents, max_leg_spread_pct_mid,
                ) or _leg_spread_too_wide(
                    float(long_contract["bid"]), float(long_contract["ask"]),
                    max_leg_spread_cents, max_leg_spread_pct_mid,
                ):
                    diag.record(REJECT_LEG_SPREAD_WIDE)
                    continue

                # Single source of truth for credit estimation. Mid-mid
                # minus a small fill-haircut, with conservative bid/ask
                # fallback when a leg has a missing or zero quote.
                credit = _quote_credit(
                    short_bid=float(short_contract["bid"]),
                    short_ask=float(short_contract["ask"]),
                    long_bid =float(long_contract["bid"]),
                    long_ask =float(long_contract["ask"]),
                )
                short_delta = float(short_contract["delta"])

                diag.grid_points_priced += 1
                result = _score_candidate_with_reason(
                    credit=credit,
                    width=actual_width,
                    short_delta=short_delta,
                    dte=slc.dte,
                    edge_buffer=edge_buffer,
                    min_pop=min_pop,
                )
                if result["status"] == "rejected":
                    reason = result["reason"]
                    diag.record(reason)
                    if reason == REJECT_CW_BELOW_FLOOR:
                        cand_payload = {
                            "expiration":   slc.expiration,
                            "dte":          slc.dte,
                            "short_strike": short_strike,
                            "long_strike":  float(long_contract["strike"]),
                            "short_delta":  round(short_delta, 4),
                            "credit":       credit,
                            "width":        round(actual_width, 4),
                            "cw_ratio":     round(result.get("cw") or 0.0, 4),
                            "cw_floor":     round(result.get("cw_floor") or 0.0, 4),
                            "pop":          round(result.get("pop") or 0.0, 4),
                            "ev":           round(result.get("ev") or 0.0, 4),
                            "target_delta": float(target_delta),
                            "width_pct":    float(width_pct),
                        }
                        cur_ev = (best_near_miss or {}).get("ev", -1e9)
                        if cand_payload["ev"] > cur_ev:
                            best_near_miss = cand_payload
                    continue

                candidates.append(SpreadCandidate(
                    side=side,
                    expiration=slc.expiration,
                    dte=slc.dte,
                    short_strike=short_strike,
                    long_strike=float(long_contract["strike"]),
                    short_delta=short_delta,
                    short_symbol=str(short_contract.get("symbol", "")),
                    long_symbol=str(long_contract.get("symbol", "")),
                    short_bid=float(short_contract["bid"]),
                    short_ask=float(short_contract["ask"]),
                    long_bid=float(long_contract["bid"]),
                    long_ask=float(long_contract["ask"]),
                    credit=credit,
                    width=actual_width,
                    cw_ratio=result["cw"],
                    pop=result["pop"],
                    cw_floor=result["cw_floor"],
                    ev_per_dollar_risked=result["ev"],
                    annualized_score=result["annualized"],
                    target_delta=float(target_delta),
                    width_pct=float(width_pct),
                ))

    candidates.sort(
        key=lambda c: (c.annualized_score, c.credit),
        reverse=True,
    )
    diag.best_near_miss = best_near_miss
    return DecisionOutput(
        candidates=candidates[:max_candidates],
        diagnostics=diag,
    )


# ---------------------------------------------------------------------------
# Long-term covered-call scoring — skill 40.
# ---------------------------------------------------------------------------
# The CI invariant scanner (scripts/checks/scan_invariant_check.py) blocks
# any module other than chain_scanner.py / decision_engine.py from defining
# a function whose name starts with _score_. The long-term evaluator
# (trading_agent/long_term_evaluator.py) is a pure orchestrator that
# *consumes* the helpers below, never defining its own. This is the same
# discipline that keeps the credit-spread scorer from being shadowed by
# the backtester.

# Conservative defaults; PresetConfig will override via cc_max_short_delta /
# cc_dte_band / cc_min_iv_rank when the preset wiring lands next session.
_CC_DEFAULT_MAX_SHORT_DELTA: float = 0.30
_CC_DEFAULT_DTE_BAND:        Tuple[int, int] = (30, 60)
_CC_DEFAULT_MIN_IV_RANK:     float = 0.25
# Strike must clear cost basis by this multiplier so a wash-sale + locked-in-loss
# is structurally impossible (covered call writer never writes below cost basis).
_CC_COST_BASIS_BUFFER:       float = 1.01

# Stable reject-reason taxonomy for the long-term evaluator. Same pattern as
# the credit-spread side (REJECT_*) so the journal histogram stays grep-able.
LT_REJECT_STRIKE_BELOW_COST_BASIS    = "strike_below_cost_basis"
LT_REJECT_SHORT_DELTA_TOO_HIGH       = "short_delta_too_high"
LT_REJECT_DTE_OUT_OF_BAND            = "dte_out_of_band"
LT_REJECT_CREDIT_NON_POSITIVE_LT     = "credit_non_positive_lt"
LT_REJECT_IV_RANK_TOO_LOW            = "iv_rank_too_low"
LT_REJECT_QTY_BELOW_100              = "qty_below_100"


def _score_covered_call(
    *,
    short_call: Dict[str, Any],
    cost_basis: float,
    preset: Any = None,
) -> Optional[Tuple[float, Dict[str, float], str]]:
    """Score one covered-call candidate. Skill 40 §2.1, §3.2.

    Inputs
    ------
    short_call:
        Normalised contract dict with keys ``strike`` (float),
        ``delta`` (signed float), ``bid`` (float ≥ 0), ``ask`` (float ≥ 0),
        ``dte`` (int > 0), and optionally ``iv_rank`` (float ∈ [0, 1]).
    cost_basis:
        Operator's per-share cost basis for the underlying stock.
    preset:
        Duck-typed PresetConfig. Reads ``cc_max_short_delta``,
        ``cc_dte_band``, ``cc_min_iv_rank``. Missing attrs fall back to the
        ``_CC_DEFAULT_*`` constants above so the function is callable from
        early-wiring sites that haven't yet plumbed the preset.

    Returns
    -------
    ``(score, metrics, "")`` on accept, ``None`` on hard reject. The
    metrics dict mirrors the math in skill 40 §2.1 — credit (dollars per
    contract), capital_at_risk, static_return, annualised_return, pop,
    dte, short_delta_abs. The third tuple element is the reject_reason
    string when the candidate was on the borderline but accepted; empty
    string for clean accepts. Callers that want the reject reason should
    use :func:`_score_covered_call_with_reason` instead.

    Hard reject reasons (returns None):
      * strike below ``cost_basis × 1.01``  — skill 40 §4
      * |Δ_short| > ``preset.cc_max_short_delta``
      * ``dte`` outside ``preset.cc_dte_band``
      * ``credit ≤ 0`` from the single-leg quote helper
      * ``iv_rank < preset.cc_min_iv_rank`` (only when iv_rank is supplied;
        missing iv_rank is treated as fail-open per skill 40 §4)
    """
    result = _score_covered_call_with_reason(
        short_call=short_call, cost_basis=cost_basis, preset=preset,
    )
    if result["status"] != "accepted":
        return None
    return (
        float(result["score"]),
        {k: float(v) for k, v in result["metrics"].items()},
        "",
    )


def _score_covered_call_with_reason(
    *,
    short_call: Dict[str, Any],
    cost_basis: float,
    preset: Any = None,
) -> Dict[str, Any]:
    """Verbose sibling of :func:`_score_covered_call`. Always returns a
    ``{"status": "accepted"|"rejected", ...}`` dict so the caller can log
    a reject taxonomy histogram (same shape as
    ``_score_candidate_with_reason`` for credit spreads).
    """
    # ── Preset reads with safe fallbacks ──────────────────────────────
    max_delta = float(getattr(preset, "cc_max_short_delta", _CC_DEFAULT_MAX_SHORT_DELTA))
    dte_band = tuple(getattr(preset, "cc_dte_band", _CC_DEFAULT_DTE_BAND))
    min_iv_rank = float(getattr(preset, "cc_min_iv_rank", _CC_DEFAULT_MIN_IV_RANK))

    strike = float(short_call["strike"])
    if strike < cost_basis * _CC_COST_BASIS_BUFFER:
        return {"status": "rejected", "reason": LT_REJECT_STRIKE_BELOW_COST_BASIS}

    delta_abs = abs(float(short_call["delta"]))
    if delta_abs > max_delta:
        return {"status": "rejected", "reason": LT_REJECT_SHORT_DELTA_TOO_HIGH}

    dte = int(short_call["dte"])
    if not (int(dte_band[0]) <= dte <= int(dte_band[1])):
        return {"status": "rejected", "reason": LT_REJECT_DTE_OUT_OF_BAND}

    credit = _quote_credit_single(
        bid=float(short_call.get("bid", 0.0)),
        ask=float(short_call.get("ask", 0.0)),
    ) * 100.0
    if credit <= 0:
        return {"status": "rejected", "reason": LT_REJECT_CREDIT_NON_POSITIVE_LT}

    iv_rank = short_call.get("iv_rank")
    if iv_rank is not None and float(iv_rank) < min_iv_rank:
        return {"status": "rejected", "reason": LT_REJECT_IV_RANK_TOO_LOW}

    capital_at_risk = max(0.01, cost_basis * 100.0 - credit)
    static_return = credit / capital_at_risk
    annualised_return = (1.0 + static_return) ** (365.0 / max(1, dte)) - 1.0
    pop = _pop_from_delta(float(short_call["delta"]))   # skill 01
    score = annualised_return * pop

    if_assigned_return = (
        ((strike - cost_basis) * 100.0 + credit) / capital_at_risk
    )

    return {
        "status": "accepted",
        "score": score,
        "metrics": {
            "credit": credit,
            "capital_at_risk": capital_at_risk,
            "static_return": static_return,
            "annualised_return": annualised_return,
            "if_assigned_return": if_assigned_return,
            "pop": pop,
            "dte": float(dte),
            "short_delta_abs": delta_abs,
        },
    }


__all__ = [
    "ChainSlice",
    "DecisionInput",
    "DecisionOutput",
    "decide",
    # Long-term evaluator helpers (skill 40)
    "_score_covered_call",
    "_score_covered_call_with_reason",
    "LT_REJECT_STRIKE_BELOW_COST_BASIS",
    "LT_REJECT_SHORT_DELTA_TOO_HIGH",
    "LT_REJECT_DTE_OUT_OF_BAND",
    "LT_REJECT_CREDIT_NON_POSITIVE_LT",
    "LT_REJECT_IV_RANK_TOO_LOW",
    "LT_REJECT_QTY_BELOW_100",
]
