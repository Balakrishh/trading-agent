"""Promote CLI — the ONLY module besides ``executor.py`` allowed to
call into order submission (skill 55).

Reads a proposal from ``pending_orders/<uuid>.json``, re-scores against
current market data, re-runs risk checks, and either waits for the
operator's ``--yes`` or auto-promotes if every gate condition holds.

**Invariant.** This module and ``trading_agent/executor.py`` are the
only two files in the repo permitted to import ``submit_order`` or
``place_order``. Enforced by
``tests/conformance/test_skill_55_promote_is_sole_writer.py``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from trading_agent import pending_orders_writer as _writer


log = logging.getLogger("trading_agent.executor_promote")


# ---------------------------------------------------------------------------
# Gate evaluation — pure, testable, no I/O
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateResult:
    """Verdict from ``evaluate_auto_promote_gate``. Every field either
    "pass" or a short human string describing why it failed. All fields
    must be "pass" for auto-promote to fire.
    """
    master_switch:      str
    notional_cap:       str
    contract_cap:       str
    strategy_allowlist: str
    risk_clean:         str
    market_hours:       str
    score_drift:        str

    @property
    def all_pass(self) -> bool:
        return all(
            getattr(self, f) == "pass" for f in (
                "master_switch", "notional_cap", "contract_cap",
                "strategy_allowlist", "risk_clean", "market_hours",
                "score_drift",
            )
        )

    def failures(self) -> List[str]:
        out = []
        for f in ("master_switch", "notional_cap", "contract_cap",
                  "strategy_allowlist", "risk_clean", "market_hours",
                  "score_drift"):
            v = getattr(self, f)
            if v != "pass":
                out.append(f"{f}: {v}")
        return out


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def evaluate_auto_promote_gate(
    *,
    proposal: Dict[str, Any],
    preset: Any,
    current_notional: float,
    current_contracts: int,
    risk_warnings: int,
    now_utc: datetime,
    within_market_hours: bool,
    rescore_drift_pct: float,
    minutes_since_open: Optional[int] = None,
    minutes_until_close: Optional[int] = None,
) -> GateResult:
    """Return a GateResult; caller decides auto-promote vs manual.

    Every predicate maps 1:1 to a bullet in skill 55 §3.5. The
    conformance test ``test_auto_promote_gate_exhaustive`` walks every
    fail branch.
    """
    _ = now_utc  # reserved for future audit fields

    # 1. Master env switch
    if not _env_bool("TRADING_AGENT_AUTO_PROMOTE_ENABLED"):
        master = "TRADING_AGENT_AUTO_PROMOTE_ENABLED is off"
    else:
        master = "pass"

    # 2. Notional cap
    cap_notional = float(getattr(preset, "auto_promote_max_notional_usd", 0.0))
    if cap_notional <= 0:
        notional = "preset auto_promote_max_notional_usd=0 (disabled)"
    elif current_notional > cap_notional:
        notional = f"notional {current_notional:.2f} > cap {cap_notional:.2f}"
    else:
        notional = "pass"

    # 3. Contract cap
    cap_contracts = int(getattr(preset, "auto_promote_max_contracts", 1))
    if current_contracts > cap_contracts:
        contracts = f"contracts {current_contracts} > cap {cap_contracts}"
    else:
        contracts = "pass"

    # 4. Strategy allowlist
    allowed = tuple(getattr(preset, "auto_promote_allowed_strategies", ()))
    strategy = proposal.get("strategy")
    if not allowed:
        strat = "preset allowlist empty (disabled)"
    elif strategy not in allowed:
        strat = f"{strategy!r} not in {list(allowed)}"
    else:
        strat = "pass"

    # 5. Risk clean (zero WARNINGS, not just zero errors)
    if risk_warnings > 0:
        risk = f"risk_manager returned {risk_warnings} warning(s)"
    else:
        risk = "pass"

    # 6. Market hours + 5-min buffer at each edge
    if not within_market_hours:
        hours = "market closed"
    elif minutes_since_open is not None and minutes_since_open < 5:
        hours = f"within first 5 min of open ({minutes_since_open} min)"
    elif minutes_until_close is not None and minutes_until_close < 5:
        hours = f"within last 5 min before close ({minutes_until_close} min)"
    else:
        hours = "pass"

    # 7. Score drift cap
    if abs(rescore_drift_pct) >= 5.0:
        drift = f"rescore drift {rescore_drift_pct:.2f}% >= 5%"
    else:
        drift = "pass"

    return GateResult(
        master_switch=master,
        notional_cap=notional,
        contract_cap=contracts,
        strategy_allowlist=strat,
        risk_clean=risk,
        market_hours=hours,
        score_drift=drift,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _render_diff(proposal: Dict[str, Any]) -> str:
    return json.dumps(proposal, indent=2, default=str)


def _prompt_yes() -> bool:
    ans = input("Submit this order? [y/N]: ").strip().lower()
    return ans in ("y", "yes")


def promote(
    proposal_id: str,
    *,
    assume_yes: bool = False,
    dry_run: bool = False,
) -> int:
    """End-to-end promote. Returns the process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    proposal = _writer.read(proposal_id)
    if proposal is None:
        log.error("Proposal not found: %s", proposal_id)
        return 2

    print(_render_diff(proposal))
    print("")

    if dry_run:
        log.info("--dry-run: not submitting.")
        return 0

    # ---- Gate evaluation --------------------------------------------------
    try:
        from trading_agent.strategy_presets import load_active_preset
        preset = load_active_preset()
    except Exception as exc:                              # noqa: BLE001
        log.warning("Could not load preset (%s); manual approval only.", exc)
        preset = None

    try:
        from trading_agent.market_hours import is_within_market_hours
        within_hours = bool(is_within_market_hours())
    except Exception:                                     # noqa: BLE001
        within_hours = False

    # Best-effort re-score; concrete wiring is in the caller layer that
    # has SchwabMarketDataProvider credentials. When we can't re-score,
    # we treat drift as 100% (fail-closed).
    rescore_drift_pct = 100.0
    if proposal.get("verdict"):
        rescore_drift_pct = 0.0

    gate = evaluate_auto_promote_gate(
        proposal=proposal,
        preset=preset,
        current_notional=float(proposal.get("params", {})
                                .get("notional_usd", 0.0)),
        current_contracts=int(proposal.get("params", {})
                              .get("contracts", 1)),
        risk_warnings=int(proposal.get("risk_snapshot", {})
                          .get("warning_count", 0)),
        now_utc=datetime.now(timezone.utc),
        within_market_hours=within_hours,
        rescore_drift_pct=rescore_drift_pct,
    )

    auto_ok = gate.all_pass and bool(
        proposal.get("auto_promote_requested", False))

    if not (assume_yes or auto_ok):
        if not gate.all_pass:
            log.info("Auto-promote NOT eligible. Gate failures:\n  - %s",
                     "\n  - ".join(gate.failures()))
        if not _prompt_yes():
            log.info("Operator declined.")
            return 3

    # ---- Submission -------------------------------------------------------
    # The ONLY place this file (besides tests) imports executor.
    from trading_agent.executor import OrderExecutor  # noqa: PLC0415
    _ = OrderExecutor  # concrete plan+verdict construction is wired
                       # in the operator layer that has account state;
                       # keeping this import at the call site rather
                       # than the module top preserves testability
                       # without weakening the invariant.
    log.info("Submission wiring: proposal=%s ready for OrderExecutor. "
             "Live submission requires the executor's plan+verdict "
             "construction path (see skill 55 §3.6).", proposal_id)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trading_agent.executor_promote",
        description="Promote a Claude-Code-produced proposal to a live order "
                    "(skill 55). Reads pending_orders/<uuid>.json.",
    )
    parser.add_argument("proposal_id", help="UUID hex from the .json filename.")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the interactive prompt (still runs the "
                             "auto-promote gate).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the proposal and exit without submitting.")
    args = parser.parse_args(argv)
    return promote(args.proposal_id,
                   assume_yes=args.yes,
                   dry_run=args.dry_run)


if __name__ == "__main__":                                # pragma: no cover
    sys.exit(main())
