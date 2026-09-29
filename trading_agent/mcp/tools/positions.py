"""Read-only wrappers over the position + journal state (skill 48 §2)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _short_strikes(ticker: str, run_id: str) -> Optional[List[float]]:
    """Short-leg strikes from ``trade_plans/trade_plan_{ticker}.json`` for
    the submit ``run_id``. None when the entry has aged out of the
    200-row state_history or the file is unreadable."""
    import json
    from pathlib import Path

    if not run_id:
        return None
    try:
        doc = json.loads(Path(f"trade_plans/trade_plan_{ticker}.json").read_text())
    except (OSError, ValueError):
        return None
    for entry in reversed(doc.get("state_history", []) if isinstance(doc, dict) else []):
        if entry.get("run_id") == run_id:
            legs = (entry.get("trade_plan") or {}).get("legs") or []
            return [float(l["strike"]) for l in legs
                    if l.get("action") == "sell" and "strike" in l] or None
    return None


def _open_row(o: Any) -> Dict[str, Any]:
    return {
        "ticker": o.ticker,
        "strategy": o.strategy,
        "opened_at": o.timestamp_utc or None,
        "credit": o.credit,
        "width": o.spread_width or None,
        "max_loss": o.max_loss or None,
        "short_strikes": _short_strikes(o.ticker, o.run_id),
        "expiration": o.expiration,
        "order_id": o.order_id or None,
        "status": o.status,
    }


def list_positions() -> Dict[str, Any]:
    """Return open positions the trading agent believes it holds.

    Sourced from the live journal + holdings_store — never from the
    executor's in-memory state (that would require importing the
    executor, which is forbidden here).

    ``open_positions`` is every submitted spread without a matching
    close (any open date). Rows with ``status="expired_unrecorded"``
    expired or closed while the agent wasn't journaling — their P&L is
    missing and must be reconciled against the broker. ``opens_today``
    is kept for back-compat (skill 49 §3).
    """
    from trading_agent.journal_reader import JournalReader
    from trading_agent.holdings_store import load_holdings

    reader = JournalReader()
    holdings = load_holdings()
    return {
        "open_positions": [_open_row(o) for o in reader.open_trades()],
        "opens_today": [_open_row(o) for o in reader.opens_today()],
        "holdings_paste_saved_at": holdings.saved_at,
        "holdings_parsed_count": holdings.parsed_count,
    }


def get_position(ticker: str, strategy: Optional[str] = None) -> Dict[str, Any]:
    """Detail for a single open position, keyed by ticker (+ optional strategy)."""
    all_open = list_positions()["open_positions"]
    matches = [
        p for p in all_open
        if (p.get("ticker") or "").upper() == ticker.upper()
        and (strategy is None or p.get("strategy") == strategy)
    ]
    if not matches:
        return {"ticker": ticker, "found": False, "positions": []}
    return {"ticker": ticker, "found": True, "positions": matches}


def list_recent_trades(days: Any = 7) -> Dict[str, Any]:
    """Closes in the last N calendar days (default 7).

    Reads the live journal directly — no executor state involved.

    ``days`` is coerced from string/float since some MCP clients JSON-
    serialize numeric arguments as strings.
    """
    try:
        days = int(days)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"days must be an integer, got {days!r}") from exc
    if days <= 0:
        raise ValueError("days must be positive")
    from trading_agent.journal_reader import JournalReader

    reader = JournalReader()

    def row(c: Any) -> Dict[str, Any]:
        return {
            "ticker": c.ticker,
            "strategy": c.strategy,
            "pnl": c.realized_pl,
            "closed_at": c.timestamp_utc or None,
            "exit_signal": c.exit_signal,
            "reason": c.exit_reason,
            "expiration": c.expiration,
        }

    window = reader.closes_since(days)
    return {
        "window_days": days,
        "closes": [row(c) for c in window],
        "realized_pl_window": round(sum(c.realized_pl for c in window), 2),
        "closes_today": [row(c) for c in reader.closes_today()],
        "realized_pl_today": reader.realized_pl_today(),
    }


def get_journal_summary() -> Dict[str, Any]:
    """Roll-up statistics over today's journal + running counters."""
    from trading_agent.journal_reader import JournalReader

    reader = JournalReader()
    return {
        "cycle_minute_count_today": reader.cycle_minute_count_today(),
        "error_count_today": reader.error_count_today(),
        "opens_today": len(reader.opens_today()),
        "closes_today": len(reader.closes_today()),
        "realized_pl_today": reader.realized_pl_today(),
        "reject_reasons_top5": reader.reject_reasons_today(top_n=5),
    }
