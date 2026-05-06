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
from typing import Any, Dict, List, Optional

from trading_agent.chain_scanner import (
    REJECT_CW_BELOW_FLOOR,
    REJECT_NO_CHAIN,
    REJECT_NO_LONG_CONTRACT,
    REJECT_NO_SHORT_CONTRACT,
    REJECT_NON_POSITIVE_WIDTH,
    ChainScanner,
    ScanDiagnostics,
    SpreadCandidate,
    _quote_credit,
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


__all__ = [
    "ChainSlice",
    "DecisionInput",
    "DecisionOutput",
    "decide",
]
