"""book_view.py — the open book with contracts and total risk (F6, backlog 11.3).

``list_positions`` returned the journal's ``max_loss``, which is **per
contract**: the 2026-10-16 and 10-19 briefs showed $1,769 at risk when the
book held $2,974, and the daily reviewer, which only saw the day's opens
and closes, said the GLD Nov 6 spread "never appeared" (opened 10-05).

``open_book()`` joins the journal's open trades with the agent's last
valuation snapshot (``position_snapshot``: broker legs, contracts, mid
P&L) and adds per-row ``contracts`` / ``total_max_loss`` plus book totals:
max loss by direction and by correlated-risk group (E6), mid P&L, and the
E4 net-delta reading. One builder for the MCP tool and the reviewer.

Read-only: imports nothing that can place an order.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

Key = Tuple[str, str, str]


def _key(ticker: Any, strategy: Any, expiration: Any) -> Key:
    return (str(ticker or ""), str(strategy or ""), str(expiration or ""))


def journal_row(o: Any) -> Dict[str, Any]:
    """One journal ``OpenedTrade`` as a dict (``max_loss`` per contract)."""
    return {
        "ticker": o.ticker,
        "strategy": o.strategy,
        "opened_at": o.timestamp_utc or None,
        "credit": o.credit,
        "width": o.spread_width or None,
        "max_loss": o.max_loss or None,
        "expiration": o.expiration,
        "status": o.status,
        "contracts": int(getattr(o, "contracts", 1) or 1),
    }


def merge_rows(journal_rows: List[Dict[str, Any]],
               snapshot: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Journal rows enriched from the snapshot (matched on ticker, strategy,
    expiration, in order). Snapshot positions without a journal row come
    back as ``status="broker_only"``. Contracts: the broker's count when
    the snapshot has the position, else the journal's."""
    snap = list((snapshot or {}).get("positions") or [])
    pool: Dict[Key, List[Dict[str, Any]]] = {}
    for p in snap:
        pool.setdefault(_key(p.get("ticker"), p.get("strategy"), p.get("expiration")), []).append(p)
    out = []
    for r in journal_rows:
        row = dict(r)
        hits = pool.get(_key(r.get("ticker"), r.get("strategy"), r.get("expiration"))) or []
        p = hits.pop(0) if hits else None
        contracts = int((p or {}).get("contracts") or r.get("contracts") or 1)
        per = float(r.get("max_loss") or (p or {}).get("max_loss") or 0.0)
        row.update({
            "contracts": contracts,
            "max_loss_per_contract": per or None,
            "total_max_loss": round(per * contracts, 2) if per else None,
            "mid_pl": (p or {}).get("mid_pl"),
            "broker_legs": (p or {}).get("legs"),
            "underlying_price": (p or {}).get("underlying_price"),
            "exit_signal": (p or {}).get("exit_signal"),
            "in_broker_snapshot": p is not None,
        })
        out.append(row)
    for left in pool.values():
        for p in left:
            per = float(p.get("max_loss") or 0.0)
            n = int(p.get("contracts") or 1)
            out.append({
                "ticker": p.get("ticker"), "strategy": p.get("strategy"),
                "expiration": p.get("expiration"), "status": "broker_only",
                "contracts": n, "max_loss_per_contract": per or None,
                "total_max_loss": round(per * n, 2) if per else None,
                "mid_pl": p.get("mid_pl"), "broker_legs": p.get("legs"),
                "underlying_price": p.get("underlying_price"),
                "exit_signal": p.get("exit_signal"), "in_broker_snapshot": True,
            })
    return out


def totals(rows: List[Dict[str, Any]], equity: Optional[float]) -> Dict[str, Any]:
    """Book totals: max loss (all contracts) overall, by direction and by
    correlated-risk group + direction (same buckets as the E6 cap)."""
    from trading_agent.position_caps import TOTAL_RISK_EXCLUDED, add_exposure, trade_direction
    live = [r for r in rows if r.get("status") != "expired_unrecorded"]
    total = sum(float(r.get("total_max_loss") or 0.0) for r in live
                if r.get("strategy") not in TOTAL_RISK_EXCLUDED)
    by_dir: Dict[str, float] = {}
    exposure: Dict[str, float] = {}
    for r in live:
        d = trade_direction(str(r.get("strategy") or ""))
        d = d if d in ("bull", "bear") else "neutral"
        by_dir[d] = round(by_dir.get(d, 0.0) + float(r.get("total_max_loss") or 0.0), 2)
        add_exposure(exposure, str(r.get("ticker") or ""), str(r.get("strategy") or ""),
                     float(r.get("total_max_loss") or 0.0))
    mids = [r.get("mid_pl") for r in live if r.get("mid_pl") is not None]
    pct = (lambda x: round(x / equity * 100, 2)) if equity else (lambda x: None)
    return {
        "positions": len(live),
        "total_max_loss": round(total, 2), "total_max_loss_pct_equity": pct(total),
        "max_loss_by_direction": by_dir,
        "max_loss_by_group_direction": {k[len("group:"):]: v for k, v in sorted(exposure.items())
                                        if k.startswith("group:")},
        "total_mid_pl": round(sum(float(m) for m in mids), 2) if mids else None,
        "mid_pl_positions": len(mids),
        "equity": equity,
        "note": "max_loss_per_contract × contracts = total_max_loss; Wheel legs excluded from the total",
    }


def _equity() -> Optional[float]:
    """Account balance the agent stored with the last market-state reading."""
    from trading_agent.market_state import read_state
    try:
        v = (read_state() or {}).get("account_balance")
        return float(v) if v else None
    except (TypeError, ValueError):
        return None


def open_book(journal_rows: List[Dict[str, Any]], snapshot: Optional[Dict[str, Any]] = None,
              equity: Optional[float] = None, book_delta: Optional[Dict[str, Any]] = None
              ) -> Dict[str, Any]:
    """Rows + totals + the snapshot's age and the E4 reading."""
    rows = merge_rows(journal_rows, snapshot)
    eq = equity if equity is not None else _equity()
    return {
        "rows": rows,
        "totals": totals(rows, eq),
        "snapshot_as_of_utc": (snapshot or {}).get("as_of_utc"),
        "snapshot_age_seconds": (snapshot or {}).get("age_seconds"),
        "net_delta": (book_delta or {}).get("reading"),
        "net_delta_as_of_utc": (book_delta or {}).get("as_of_utc"),
    }
