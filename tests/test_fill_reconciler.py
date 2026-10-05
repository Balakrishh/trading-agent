"""Backlog §2 (2026-10-05): record multi-leg entry fills from the broker."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from trading_agent.executor import OrderExecutor
from trading_agent.fill_reconciler import pending_runs, reconcile_fills, signed_fill


_SEQ = [0]


def _run_id(days_ago=0):
    """Distinct second-resolution run ids (the executor's format)."""
    _SEQ[0] += 1
    t = datetime.now(timezone.utc) - timedelta(days=days_ago, seconds=_SEQ[0])
    return t.strftime("%Y%m%d_%H%M%S")


def _plan_file(tmp_path, ticker, entries):
    fp = tmp_path / f"trade_plan_{ticker}.json"
    fp.write_text(json.dumps({"ticker": ticker, "state_history": entries}))
    return fp


def _entry(run_id, order_id, net, *, width=5.0, max_loss=None, status="submitted", **tp):
    return {"run_id": run_id,
            "trade_plan": {"net_credit": net, "spread_width": width,
                           "max_loss": max_loss if max_loss is not None else (width - net) * 100, **tp},
            "order_result": {"status": status, "order_id": order_id}}


def _order(status, avg=None, filled_qty=0):
    return SimpleNamespace(status=SimpleNamespace(value=status), filled_avg_price=avg,
                           filled_qty=str(filled_qty))


def test_signed_fill_follows_plan_sign():
    assert signed_fill(-1.92, 1.92) == -1.92        # debit: Alpaca reports positive
    assert signed_fill(0.49, -0.48) == 0.48         # credit, whichever sign Alpaca uses
    assert signed_fill(0.49, 0.48) == 0.48


def test_reconcile_records_credit_and_debit_fills(tmp_path):
    r1, r2 = _run_id(), _run_id()
    fp = _plan_file(tmp_path, "SPY", [_entry(r1, "o-credit", 0.49)])
    fq = _plan_file(tmp_path, "IWM", [_entry(r2, "o-debit", -1.92, max_loss=192.0)])
    orders = {"o-credit": _order("filled", "-0.48", 16), "o-debit": _order("filled", "1.95", 4)}
    counts = reconcile_fills(str(tmp_path), get_order=orders.get,
                             record_fill=lambda p, r, f: OrderExecutor._record_fill_credit(p, r, None, f),
                             mark_unfilled=OrderExecutor._mark_run_unfilled)
    assert counts["recorded"] == 2
    spy = json.loads(fp.read_text())["state_history"][0]["trade_plan"]
    assert (spy["net_credit"], spy["estimated_net_credit"], spy["max_loss"]) == (0.48, 0.49, 452.0)
    iwm = json.loads(fq.read_text())["state_history"][0]["trade_plan"]
    assert (iwm["net_credit"], iwm["max_loss"]) == (-1.95, 195.0)
    # Already recorded → never looked up again.
    assert pending_runs(str(tmp_path)) == []


def test_unfilled_cancel_invalidates_run_working_left_alone(tmp_path):
    r1, r2 = _run_id(), _run_id()
    fp = _plan_file(tmp_path, "XLF", [_entry(r1, "o-cancel", 0.30), _entry(r2, "o-work", 0.31)])
    orders = {"o-cancel": _order("canceled"), "o-work": _order("new")}
    counts = reconcile_fills(str(tmp_path), get_order=orders.get,
                             record_fill=lambda *a: pytest.fail("nothing filled"),
                             mark_unfilled=OrderExecutor._mark_run_unfilled)
    assert counts == {"recorded": 0, "unfilled": 1, "working": 1, "unknown": 0}
    hist = json.loads(fp.read_text())["state_history"]
    assert hist[0]["trade_plan"]["valid"] is False and "valid" not in hist[1]["trade_plan"]


def test_skips_old_rejected_and_lookup_failures(tmp_path):
    _plan_file(tmp_path, "DIA", [
        _entry(_run_id(days_ago=5), "o-old", 0.4),                 # outside window
        _entry(_run_id(), "o-rej", 0.4, status="rejected"),        # never submitted
        _entry(_run_id(), "o-err", 0.4)])
    assert [r["order_id"] for r in pending_runs(str(tmp_path))] == ["o-err"]

    def boom(_):
        raise RuntimeError("timeout")
    counts = reconcile_fills(str(tmp_path), get_order=boom,
                             record_fill=lambda *a: False, mark_unfilled=lambda *a: False)
    assert counts["unknown"] == 1
