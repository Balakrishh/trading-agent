"""journal_reconcile.py — close out a journal open whose broker close was never recorded.

A ``submitted`` row with no matching ``closed`` row stays in
``JournalReader.open_trades`` forever (``status="expired_unrecorded"`` once
its expiration passes) — e.g. the 2026-07-08 XLE iron condor, opened in a
previous Alpaca paper account whose order history is no longer reachable.

* Realized P&L **known** (``--realized-pl``) → append a normal ``closed`` row
  (``pl_source="operator reconciliation"``); it counts in realized P&L.
* P&L **unknown** → append a ``position_reconciled`` row. ``open_trades`` and
  the playbook scorecard pair it with the open (so it stops showing as open),
  but it is NOT a close: never summed into realized P&L, never a win or loss.

Append-only — existing rows are never edited. Dry-run unless ``--apply``.

    python -m trading_agent.journal_reconcile --ticker XLE --strategy "Iron Condor" \\
        --expiration 2026-07-31 --reason "old paper account; history unavailable" [--apply]
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional

from trading_agent.journal_reader import DEFAULT_LIVE_JOURNAL, RECONCILED_ACTION, JournalReader


def build_row(opened: Any, *, reason: str,
              realized_pl: Optional[float]) -> Dict[str, Any]:
    """(action, raw_signal, notes) for the reconciliation of ``opened``."""
    base = {"strategy": opened.strategy, "expiration": opened.expiration,
            "order_id": opened.order_id, "run_id": opened.run_id,
            "original_credit": opened.credit, "max_loss": opened.max_loss,
            "reconcile_reason": reason, "mode": "LIVE"}
    if realized_pl is not None:
        return {"action": "closed",
                "raw_signal": {**base, "exit_signal": "reconciled",
                               "exit_reason": reason, "net_unrealized_pl": float(realized_pl),
                               "fill_status": "complete",
                               "pl_source": "operator reconciliation"},
                "notes": f"closed (reconciled): {opened.strategy}, P&L=${realized_pl:.2f}"}
    return {"action": RECONCILED_ACTION,
            "raw_signal": {**base, "pl_known": False, "net_unrealized_pl": None},
            "notes": f"reconciled: {opened.strategy} exp {opened.expiration} — P&L unknown ({reason})"}


def find_open(reader: JournalReader, ticker: str, strategy: str, expiration: str):
    for t in reader.open_trades():
        if (t.ticker, t.strategy, t.expiration) == (ticker, strategy, expiration):
            return t
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--expiration", required=True)
    ap.add_argument("--reason", required=True)
    ap.add_argument("--realized-pl", type=float, default=None)
    ap.add_argument("--journal", default=DEFAULT_LIVE_JOURNAL)
    ap.add_argument("--apply", action="store_true", help="write the row (default: dry run)")
    a = ap.parse_args(argv)

    opened = find_open(JournalReader(a.journal), a.ticker.upper(), a.strategy, a.expiration)
    if opened is None:
        print(f"No unmatched open for {a.ticker} {a.strategy} {a.expiration} — nothing to do.")
        return 1
    row = build_row(opened, reason=a.reason, realized_pl=a.realized_pl)
    print(json.dumps({"ticker": opened.ticker, **row}, indent=2, default=str))
    if not a.apply:
        print("\nDry run — re-run with --apply to append this row.")
        return 0
    from pathlib import Path
    from trading_agent.journal_kb import JournalKB
    kb = JournalKB(str(Path(a.journal).parent), run_mode="live")
    kb.log_signal(ticker=opened.ticker, action=row["action"], price=0.0,
                  raw_signal=row["raw_signal"], exec_status=row["action"], notes=row["notes"])
    still_open = find_open(JournalReader(a.journal), opened.ticker, a.strategy, a.expiration)
    print("Appended." if still_open is None else "Appended, but the open still shows — check the journal.")
    return 0 if still_open is None else 2


if __name__ == "__main__":
    sys.exit(main())
