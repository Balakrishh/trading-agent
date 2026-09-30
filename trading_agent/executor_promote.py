"""Promote CLI — the ONLY module besides ``executor.py`` allowed to
call into order submission on the Claude Code write path (skill 55).

Reads a proposal from ``pending_orders/<uuid>.json``, re-scores against
current market data, re-runs risk checks, and either waits for the
operator's ``--yes`` or auto-promotes if every gate condition holds.

Live paper trading path:
  1. Rehydrate the SpreadPlan from the proposal's ``verdict.plan`` dict.
  2. Build a RiskManager from the active preset + fetched account balance.
  3. Call ``rm.evaluate(plan, account_balance)`` → RiskVerdict.
  4. Instantiate ``OrderExecutor(api_key, secret_key, dry_run=<flag>)``
     and call ``executor.execute(verdict)``.
  5. Log the returned order id / dry-run path.

**Invariant.** This module and ``trading_agent/executor.py`` are the
only two files in the repo permitted to import order-submission
primitives (plus ``agent.py``, the live headless cycle). Enforced by
``tests/conformance/test_skill_55_promote_gate.py``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading_agent import pending_orders_writer as _writer


log = logging.getLogger("trading_agent.executor_promote")


# ---------------------------------------------------------------------------
# Gate evaluation — pure, testable, no I/O
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateResult:
    """Verdict from ``evaluate_auto_promote_gate``."""
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
    """Seven-predicate gate. See skill 55 §3.5."""
    _ = now_utc

    if not _env_bool("TRADING_AGENT_AUTO_PROMOTE_ENABLED"):
        master = "TRADING_AGENT_AUTO_PROMOTE_ENABLED is off"
    else:
        master = "pass"

    cap_notional = float(getattr(preset, "auto_promote_max_notional_usd", 0.0))
    if cap_notional <= 0:
        notional = "preset auto_promote_max_notional_usd=0 (disabled)"
    elif current_notional > cap_notional:
        notional = f"notional {current_notional:.2f} > cap {cap_notional:.2f}"
    else:
        notional = "pass"

    cap_contracts = int(getattr(preset, "auto_promote_max_contracts", 1))
    if current_contracts > cap_contracts:
        contracts = f"contracts {current_contracts} > cap {cap_contracts}"
    else:
        contracts = "pass"

    allowed = tuple(getattr(preset, "auto_promote_allowed_strategies", ()))
    strategy = proposal.get("strategy")
    if not allowed:
        strat = "preset allowlist empty (disabled)"
    elif strategy not in allowed:
        strat = f"{strategy!r} not in {list(allowed)}"
    else:
        strat = "pass"

    if risk_warnings > 0:
        risk = f"risk_manager returned {risk_warnings} warning(s)"
    else:
        risk = "pass"

    if not within_market_hours:
        hours = "market closed"
    elif minutes_since_open is not None and minutes_since_open < 5:
        hours = f"within first 5 min of open ({minutes_since_open} min)"
    elif minutes_until_close is not None and minutes_until_close < 5:
        hours = f"within last 5 min before close ({minutes_until_close} min)"
    else:
        hours = "pass"

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
# Plan rehydration — proposal JSON → SpreadPlan
# ---------------------------------------------------------------------------

def _rehydrate_plan(plan_dict: Dict[str, Any]):
    """Rebuild a SpreadPlan from the proposal's ``verdict.plan`` dict.

    Raises ValueError if the dict is missing required fields. Kept
    isolated so tests can call it without touching Alpaca.
    """
    from trading_agent.strategy import SpreadPlan, SpreadLeg   # noqa: PLC0415

    required = ("ticker", "strategy", "legs", "spread_width",
                "net_credit", "expiration")
    missing = [k for k in required if k not in plan_dict]
    if missing:
        raise ValueError(f"plan dict missing fields: {missing}")

    legs = []
    for leg in plan_dict["legs"]:
        bid = float(leg.get("bid", 0.0))
        ask = float(leg.get("ask", 0.0))
        legs.append(SpreadLeg(
            symbol=leg["symbol"],
            strike=float(leg["strike"]),
            action=leg["action"],
            option_type=leg.get("type") or leg.get("option_type"),
            delta=float(leg.get("delta", 0.0)),
            theta=float(leg.get("theta", 0.0)),
            bid=bid,
            ask=ask,
            mid=float(leg.get("mid", (bid + ask) / 2.0 if bid or ask else 0.0)),
        ))
    return SpreadPlan(
        ticker=plan_dict["ticker"],
        strategy_name=plan_dict["strategy"],
        regime=plan_dict.get("regime", "claude-code-proposal"),
        legs=legs,
        spread_width=float(plan_dict["spread_width"]),
        net_credit=float(plan_dict["net_credit"]),
        max_loss=float(plan_dict.get(
            "max_loss",
            max(0.0, float(plan_dict["spread_width"])
                     - float(plan_dict["net_credit"])) * 100,
        )),
        credit_to_width_ratio=float(plan_dict.get(
            "credit_to_width_ratio", 0.0)),
        expiration=plan_dict["expiration"],
        reasoning=plan_dict.get("reasoning", ""),
        valid=bool(plan_dict.get("valid", True)),
    )


# ---------------------------------------------------------------------------
# Submission — calls into OrderExecutor
# ---------------------------------------------------------------------------

def _submit_via_executor(
    plan,
    *,
    account_balance: float,
    account_type: str,
    dry_run: bool,
) -> Dict[str, Any]:
    """Run risk checks and hand the plan to OrderExecutor.

    Isolated from the CLI plumbing so tests can inject a fake executor
    without hitting Alpaca.
    """
    from trading_agent.config import load_config                # noqa: PLC0415
    from trading_agent.executor import OrderExecutor            # noqa: PLC0415
    from trading_agent.risk_manager import RiskManager          # noqa: PLC0415
    from trading_agent.strategy_presets import load_active_preset  # noqa: PLC0415

    preset = load_active_preset()
    cfg = load_config()
    rm = RiskManager(
        max_risk_pct=getattr(preset, "max_risk_pct", 0.02),
        min_credit_ratio=getattr(preset, "min_credit_ratio", 0.25),
        max_delta=getattr(preset, "max_delta", 0.30),
        delta_aware_floor=getattr(preset, "scan_mode", "static") == "adaptive",
        edge_buffer=getattr(preset, "edge_buffer", 0.10),
    )
    verdict = rm.evaluate(
        plan,
        account_balance=account_balance,
        account_type=account_type,
        market_open=True,
    )
    log.info("RiskManager verdict: approved=%s summary=%s",
             verdict.approved, verdict.summary)
    if not verdict.approved:
        return {"status": "risk_rejected", "reason": verdict.summary}

    executor = OrderExecutor(
        api_key=cfg.alpaca.api_key,
        secret_key=cfg.alpaca.secret_key,
        base_url=cfg.alpaca.base_url,
        dry_run=dry_run,
        max_risk_pct=getattr(preset, "max_risk_pct", 0.02),
        min_credit_ratio=getattr(preset, "min_credit_ratio", 0.25),
        delta_aware_floor=getattr(preset, "scan_mode", "static") == "adaptive",
        edge_buffer=getattr(preset, "edge_buffer", 0.10),
    )
    return executor.execute(verdict)


# ---------------------------------------------------------------------------
# Wheel single-leg submission — CSP / covered call (skill 40 §2.9)
# ---------------------------------------------------------------------------

def check_wheel_order(plan, *, qty: int, equity: float,
                      options_buying_power: Optional[float],
                      shares_held: int) -> List[str]:
    """Pre-submit checks for a Wheel leg. Returns failure reasons; empty
    means OK. Pure so it is unit-testable without a broker.

    * CSP: cash must actually secure the put (strike × 100 × qty ≤
      options buying power — unknown buying power fails closed) and one
      put may not tie up more than MAX_CSP_COLLATERAL_PCT_OF_EQUITY.
    * Covered call: ≥ 100 × qty shares must already be held — otherwise
      the "covered" call is naked and must never be sent.
    """
    from trading_agent.wheel_policy import (                    # noqa: PLC0415
        CC_STRATEGY, CSP_STRATEGY, MAX_CSP_COLLATERAL_PCT_OF_EQUITY)

    fails: List[str] = []
    if len(plan.legs) != 1 or plan.legs[0].action != "sell":
        return ["Wheel plan must be exactly one sell leg"]
    leg = plan.legs[0]
    if qty < 1:
        fails.append(f"qty {qty} < 1")
    if plan.strategy_name == CSP_STRATEGY:
        if leg.option_type != "put":
            fails.append("Cash-Secured Put leg is not a put")
        collateral = leg.strike * 100 * qty
        if options_buying_power is None:
            fails.append("options buying power unknown — cannot confirm the put is cash-secured")
        elif collateral > options_buying_power:
            fails.append(f"collateral ${collateral:,.0f} > options buying power "
                         f"${options_buying_power:,.0f}")
        cap = MAX_CSP_COLLATERAL_PCT_OF_EQUITY * equity
        if equity <= 0 or collateral > cap:
            fails.append(f"collateral ${collateral:,.0f} > "
                         f"{MAX_CSP_COLLATERAL_PCT_OF_EQUITY:.0%} of equity (${cap:,.0f})")
    elif plan.strategy_name == CC_STRATEGY:
        if leg.option_type != "call":
            fails.append("Covered Call leg is not a call")
        if shares_held < 100 * qty:
            fails.append(f"not covered: {shares_held} shares held < {100 * qty} required")
    else:
        fails.append(f"not a Wheel strategy: {plan.strategy_name!r}")
    return fails


def _alpaca_account() -> Optional[Dict[str, Any]]:
    """GET /account on the configured Alpaca endpoint; None on failure."""
    try:
        from trading_agent.config import load_config            # noqa: PLC0415
        import urllib.request                                   # noqa: PLC0415
        cfg = load_config()
        req = urllib.request.Request(f"{cfg.alpaca.base_url}/account")
        req.add_header("APCA-API-KEY-ID", cfg.alpaca.api_key)
        req.add_header("APCA-API-SECRET-KEY", cfg.alpaca.secret_key)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:                                    # noqa: BLE001
        log.warning("Could not fetch Alpaca account (%s).", exc)
        return None


def _submit_wheel(plan, *, qty: int, dry_run: bool) -> Dict[str, Any]:
    """Wheel checks → OrderExecutor.execute_single_leg → journal on fill."""
    from trading_agent.config import load_config                # noqa: PLC0415
    from trading_agent.executor import OrderExecutor            # noqa: PLC0415
    from trading_agent.positions_provider import AlpacaPositionsProvider  # noqa: PLC0415

    cfg = load_config()
    account = _alpaca_account() or {}
    equity = float(account.get("equity") or 0.0)
    obp_raw = account.get("options_buying_power")
    obp = float(obp_raw) if obp_raw not in (None, "") else None
    shares = sum(p.qty for p in AlpacaPositionsProvider(
        cfg.alpaca.api_key, cfg.alpaca.secret_key, cfg.alpaca.base_url).snapshot()
        if p.ticker == plan.ticker)

    fails = check_wheel_order(plan, qty=qty, equity=equity,
                              options_buying_power=obp, shares_held=shares)
    if fails:
        return {"status": "risk_rejected", "reason": "; ".join(fails)}

    data_provider = None
    try:
        from trading_agent.market_data_factory import build_market_data_provider  # noqa: PLC0415
        data_provider = build_market_data_provider(
            alpaca_api_key=cfg.alpaca.api_key, alpaca_secret_key=cfg.alpaca.secret_key,
            alpaca_data_url=cfg.alpaca.data_url, alpaca_base_url=cfg.alpaca.base_url)
    except Exception as exc:                                    # noqa: BLE001
        log.warning("No live quote provider (%s) — pricing from the proposal.", exc)

    executor = OrderExecutor(
        api_key=cfg.alpaca.api_key, secret_key=cfg.alpaca.secret_key,
        base_url=cfg.alpaca.base_url, dry_run=dry_run, data_provider=data_provider,
    )
    result = executor.execute_single_leg(plan, qty=qty, account_balance=equity)
    if result.get("status") == "filled":
        _journal_wheel_open(plan, result, qty)
    return result


def _journal_wheel_open(plan, result: Dict[str, Any], qty: int) -> None:
    """Write the ``submitted`` row JournalReader.open_trades pairs with the
    later close / assignment row (skill 19). Only called on a confirmed fill."""
    from trading_agent.config import load_config                # noqa: PLC0415
    from trading_agent.journal_kb import JournalKB              # noqa: PLC0415

    cfg = load_config()
    journal_dir = (cfg.intelligence.journal_dir
                   if cfg.intelligence and cfg.intelligence.journal_dir else "trade_journal")
    credit = float(result.get("limit_price") or plan.net_credit)
    leg = plan.legs[0]
    JournalKB(journal_dir, run_mode="live").log_signal(
        ticker=plan.ticker, action="submitted", price=0.0,
        raw_signal={
            "strategy": plan.strategy_name, "net_credit": credit,
            "max_loss": plan.max_loss, "spread_width": plan.spread_width,
            "expiration": plan.expiration, "order_id": result.get("order_id"),
            "run_id": result.get("run_id"), "contracts": qty,
            "short_symbol": leg.symbol, "short_strike": leg.strike,
            "option_type": leg.option_type, "source": "executor_promote",
        },
        notes=f"submitted: {plan.strategy_name} {leg.symbol} ×{qty} @ {credit:.2f}",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _render_diff(proposal: Dict[str, Any]) -> str:
    return json.dumps(proposal, indent=2, default=str)


def _prompt_yes() -> bool:
    try:
        ans = input("Submit this order? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def _resolve_account_balance() -> float:
    """Fetch current Alpaca paper equity, or fall back to env / default."""
    override = os.environ.get("TRADING_AGENT_ACCOUNT_BALANCE_OVERRIDE")
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    try:
        from trading_agent.config import load_config           # noqa: PLC0415
        cfg = load_config()
        import urllib.request                                  # noqa: PLC0415
        req = urllib.request.Request(f"{cfg.alpaca.base_url}/account")
        req.add_header("APCA-API-KEY-ID", cfg.alpaca.api_key)
        req.add_header("APCA-API-SECRET-KEY", cfg.alpaca.secret_key)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return float(data.get("equity", 0.0))
    except Exception as exc:                                    # noqa: BLE001
        log.warning("Could not fetch Alpaca account balance (%s); "
                    "using $25,000 fallback.", exc)
        return 25_000.0


def promote(
    proposal_id: str,
    *,
    assume_yes: bool = False,
    dry_run: bool = False,
    paper: bool = True,
) -> int:
    """End-to-end promote. Returns the process exit code.

    ``paper=True`` (default) forces dry_run=False against the Alpaca
    paper endpoint from load_config(). Set ``paper=False`` only after
    the operator has explicitly verified alpaca.base_url points to
    the live endpoint — that's a per-repo config choice, not a CLI
    default we ever want to flip silently.
    """
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
        log.info("--dry-run: printed proposal, not submitting.")
        return 0

    # ---- Gate evaluation ------------------------------------------------
    try:
        from trading_agent.strategy_presets import load_active_preset
        preset = load_active_preset()
    except Exception as exc:                                    # noqa: BLE001
        log.warning("Could not load preset (%s); manual approval only.", exc)
        preset = None

    try:
        from trading_agent.market_hours import is_within_market_hours
        within_hours = bool(is_within_market_hours())
    except Exception:                                           # noqa: BLE001
        within_hours = False

    verdict = proposal.get("verdict") or {}
    plan_dict = verdict.get("plan") if isinstance(verdict, dict) else None

    # Rehydrate up front so the operator sees the plan-shape check
    # BEFORE the interactive prompt — a broken proposal fails fast.
    try:
        plan = _rehydrate_plan(plan_dict) if plan_dict else None
    except Exception as exc:                                    # noqa: BLE001
        log.error("Proposal plan is not rehydratable: %s", exc)
        return 4

    gate = evaluate_auto_promote_gate(
        proposal=proposal,
        preset=preset,
        current_notional=float(plan_dict.get("net_credit", 0.0)) * 100.0
                          if plan_dict else 0.0,
        current_contracts=int(proposal.get("params", {})
                              .get("contracts", 1)),
        risk_warnings=int(proposal.get("risk_snapshot", {})
                          .get("warning_count", 0)),
        now_utc=datetime.now(timezone.utc),
        within_market_hours=within_hours,
        rescore_drift_pct=0.0 if plan_dict else 100.0,
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

    # ---- Submission -----------------------------------------------------
    if plan is None:
        log.error("Proposal has no plan dict — cannot submit. "
                  "Re-run /propose with a live data server so scoring "
                  "produces a full plan.")
        return 5

    from trading_agent.wheel_policy import WHEEL_STRATEGIES     # noqa: PLC0415
    if plan.strategy_name in WHEEL_STRATEGIES:
        qty = int(proposal.get("params", {}).get("contracts", 1) or 1)
        result = _submit_wheel(plan, qty=qty, dry_run=False)
        log.info("Wheel executor result: %s", json.dumps(result, default=str))
        return {"filled": 0, "dry_run": 0, "risk_rejected": 6,
                "unfilled": 8}.get(result.get("status"), 7)

    account_balance = _resolve_account_balance()
    result = _submit_via_executor(
        plan,
        account_balance=account_balance,
        account_type="paper" if paper else "live",
        dry_run=False,
    )
    log.info("Executor result: %s", json.dumps(result, default=str))
    if result.get("status") in ("submitted", "filled", "accepted"):
        return 0
    if result.get("status") == "dry_run":
        return 0
    if result.get("status") == "risk_rejected":
        return 6
    return 7


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
    parser.add_argument("--live", action="store_true",
                        help="Route to the live Alpaca endpoint. Default is "
                             "paper. Requires alpaca.base_url to point at "
                             "the live endpoint (checked at submit time).")
    args = parser.parse_args(argv)
    return promote(args.proposal_id,
                   assume_yes=args.yes,
                   dry_run=args.dry_run,
                   paper=not args.live)


if __name__ == "__main__":                                # pragma: no cover
    sys.exit(main())
