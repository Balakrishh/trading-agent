"""position_snapshot.py — the monitor's per-cycle valuation, for read-only tools (skill 50).

Every cycle the agent values open positions at the bid/ask mid (skill 44)
and evaluates exit rules. Before 2026-10-05 those numbers only reached the
log, so `/triage` re-priced legs from a different quote source (Alpaca's
indicative feed) and disagreed with the agent: −$392 "close now" vs the
agent's −$112 HOLD on the SPY iron condor (2026-09-30).

The agent now writes this snapshot (atomic temp+rename) after evaluation,
and the MCP tool `get_position_valuations` reads it, so triage and the
exit rules use the same numbers.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

SNAPSHOT_PATH = Path(os.environ.get(
    "TRADING_AGENT_POSITION_SNAPSHOT", "trade_journal/position_valuations.json"))


def build_snapshot(spreads: Iterable[Any],
                   underlying_prices: Dict[str, float]) -> Dict[str, Any]:
    """Pure: evaluated SpreadPosition objects → JSON-ready dict."""
    rows = []
    for s in spreads:
        sig = getattr(s, "exit_signal", None)
        rows.append({
            "ticker": s.underlying,
            "strategy": s.strategy_name,
            "expiration": s.expiration,
            "legs": [getattr(leg, "symbol", "") for leg in s.legs],
            "contracts": getattr(s, "contracts_open", 1),
            "original_credit": s.original_credit,
            "max_loss": s.max_loss,
            "short_strikes": list(getattr(s, "short_strikes", []) or []),
            "mid_pl": round(float(s.net_unrealized_pl), 2),
            "mark_sources": sorted({getattr(leg, "mark_source", "broker")
                                    for leg in s.legs}),
            "short_delta": getattr(s, "short_delta", None),
            "underlying_price": underlying_prices.get(s.underlying),
            "exit_signal": getattr(sig, "value", str(sig)) if sig is not None else None,
            "exit_reason": getattr(s, "exit_reason", ""),
        })
    return {"as_of_utc": datetime.now(timezone.utc).isoformat(), "positions": rows}


def write_snapshot(snapshot: Dict[str, Any], path: Path = SNAPSHOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(snapshot, indent=2, default=str))
    tmp.replace(path)


def read_snapshot(path: Path = SNAPSHOT_PATH) -> Optional[Dict[str, Any]]:
    """Snapshot plus ``age_seconds``; None if missing or unreadable."""
    try:
        snap = json.loads(path.read_text())
        as_of = datetime.fromisoformat(snap["as_of_utc"])
        snap["age_seconds"] = round((datetime.now(timezone.utc) - as_of).total_seconds())
        return snap
    except (OSError, ValueError, KeyError, TypeError):
        return None
