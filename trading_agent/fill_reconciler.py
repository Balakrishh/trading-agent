"""fill_reconciler.py — record actual spread-entry fills (backlog §2, 2026-10-05).

Spread orders are fire-and-forget: the trade plan is saved with the
*estimated* net credit and the order is left working. The position
monitor reads ``net_credit`` from that plan for its profit target and
stops, so an estimate-vs-fill gap (SPY IC: plan 0.49, fill 0.48) skews
every exit. Single-leg Wheel orders already record their fill
(``OrderExecutor._record_fill_credit``); this module does the same for
multi-leg entries once the broker reports them filled.

Each cycle the agent calls :func:`reconcile_fills`. For every recent
``submitted`` run whose plan has no ``estimated_net_credit`` yet:

* order **filled**  → rewrite the plan economics from ``filled_avg_price``
  (signed like the plan: a debit plan stays negative — Alpaca reports
  debits as positive amounts, 2026-10-05: GLD/IWM/QQQ/SPY);
* order **canceled / expired / rejected** with nothing filled → mark the
  run invalid so it cannot claim legs of a later live position;
* otherwise (still working) → leave it for the next cycle.

Pure orchestration: broker access and plan writes are injected.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_TERMINAL_UNFILLED = {"canceled", "expired", "rejected"}


def signed_fill(planned_net: float, filled_avg_price: float) -> float:
    """Fill per share with the plan's sign: negative for a debit plan."""
    fill = abs(float(filled_avg_price))
    return -fill if float(planned_net) < 0 else fill


def _run_time(run_id: str) -> Optional[datetime]:
    try:
        return datetime.strptime(run_id, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def pending_runs(plan_dir: str, *, since_days: int = 3,
                 now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Submitted runs from the last ``since_days`` whose fill is not yet
    recorded: ``[{plan_path, run_id, order_id, net_credit}]``."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=since_days)
    out: List[Dict[str, Any]] = []
    for fp in sorted(Path(plan_dir).glob("trade_plan_*.json")):
        try:
            doc = json.loads(fp.read_text())
        except (OSError, ValueError):
            continue
        for e in doc.get("state_history", []) or []:
            res = e.get("order_result") or {}
            tp = e.get("trade_plan") or {}
            started = _run_time(e.get("run_id", ""))
            if (res.get("status") != "submitted" or not res.get("order_id")
                    or "estimated_net_credit" in tp or tp.get("valid") is False
                    or started is None or started < cutoff):
                continue
            out.append({"plan_path": str(fp), "run_id": e["run_id"],
                        "order_id": res["order_id"],
                        "net_credit": float(tp.get("net_credit") or 0.0)})
    return out


def reconcile_fills(plan_dir: str, *, get_order: Callable[[str], Any],
                    record_fill: Callable[[str, str, float], bool],
                    mark_unfilled: Callable[[str, str], bool],
                    since_days: int = 3) -> Dict[str, int]:
    """Process :func:`pending_runs`; returns counts by outcome."""
    counts = {"recorded": 0, "unfilled": 0, "working": 0, "unknown": 0}
    for run in pending_runs(plan_dir, since_days=since_days):
        try:
            order = get_order(run["order_id"])
        except Exception as exc:                  # noqa: BLE001 — retry next cycle
            logger.warning("Fill lookup for %s failed: %s", run["order_id"], exc)
            order = None
        if order is None:
            counts["unknown"] += 1
            continue
        status = getattr(getattr(order, "status", None), "value", str(getattr(order, "status", "")))
        avg = getattr(order, "filled_avg_price", None)
        if status == "filled" and avg not in (None, ""):
            fill = signed_fill(run["net_credit"], float(avg))
            if record_fill(run["plan_path"], run["run_id"], fill):
                counts["recorded"] += 1
                logger.info("Recorded entry fill %.2f (plan %.2f) for run %s",
                            fill, run["net_credit"], run["run_id"])
        elif status in _TERMINAL_UNFILLED and float(getattr(order, "filled_qty", 0) or 0) == 0:
            if mark_unfilled(run["plan_path"], run["run_id"]):
                counts["unfilled"] += 1
        else:
            counts["working"] += 1
    return counts
