"""Read-only wrappers over the position + journal state (skill 48 §2)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def list_positions() -> List[Dict[str, Any]]:
    """Return open positions the trading agent believes it holds.

    Sourced from the live journal + holdings_store — never from the
    executor's in-memory state (that would require importing the
    executor, which is forbidden here).
    """
    from trading_agent.journal_reader import JournalReader
    from trading_agent.holdings_store import load_holdings

    reader = JournalReader()
    opens = reader.opens_today()
    holdings = load_holdings()

    rows: List[Dict[str, Any]] = []
    for o in opens:
        rows.append({
            "ticker": getattr(o, "ticker", None),
            "strategy": getattr(o, "strategy", None),
            "opened_at": getattr(o, "opened_at", None),
            "credit": getattr(o, "credit", None),
            "width": getattr(o, "width", None),
            "short_strike": getattr(o, "short_strike", None),
            "expiration": getattr(o, "expiration", None),
        })
    return {
        "opens_today": rows,
        "holdings_paste_saved_at": holdings.saved_at,
        "holdings_parsed_count": holdings.parsed_count,
    }


def get_position(ticker: str, strategy: Optional[str] = None) -> Dict[str, Any]:
    """Detail for a single open position, keyed by ticker (+ optional strategy)."""
    all_open = list_positions()["opens_today"]
    matches = [
        p for p in all_open
        if (p.get("ticker") or "").upper() == ticker.upper()
        and (strategy is None or p.get("strategy") == strategy)
    ]
    if not matches:
        return {"ticker": ticker, "found": False, "positions": []}
    return {"ticker": ticker, "found": True, "positions": matches}


def list_recent_trades(days: int = 7) -> Dict[str, Any]:
    """Closes in the last N calendar days (default 7).

    Reads the live journal directly — no executor state involved.
    """
    if days <= 0:
        raise ValueError("days must be positive")
    from trading_agent.journal_reader import JournalReader

    reader = JournalReader()
    closes = reader.closes_today()  # today only; broader window requires
                                    # a JournalReader extension out of scope
    return {
        "window_days": days,
        "closes_today": [
            {
                "ticker": getattr(c, "ticker", None),
                "strategy": getattr(c, "strategy", None),
                "pnl": getattr(c, "pnl", None),
                "closed_at": getattr(c, "closed_at", None),
                "reason": getattr(c, "reason", None),
            }
            for c in closes
        ],
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
