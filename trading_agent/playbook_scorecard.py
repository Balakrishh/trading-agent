"""playbook_scorecard.py — per-playbook track record (backlog §6.6, 2026-10-05).

Tags every round trip with the playbook and regime it was opened under,
then reports, per playbook: trades, win rate, average win / loss,
expectancy, return on risk, and entry slippage (estimated vs filled net,
from the fills ``fill_reconciler`` records). After ``min_trades`` round
trips it suggests a per-trade risk (1–3 % of equity) or disabling the
playbook. Advisory only — it never edits the preset; the operator (or a
staged preset update, skill 56) decides.

Pairing mirrors ``JournalReader.open_trades``: ``submitted`` rows and real
``closed`` rows match FIFO on (ticker, strategy, expiration). Opens from
before the playbook tag existed (pre-2026-10-05) fall back to the
playbook their strategy implies.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Strategy → playbook for rows journaled before the playbook tag.
_STRATEGY_PLAYBOOK = {
    "Bull Put Spread": "bull_put", "Bear Call Spread": "bear_call",
    "Iron Condor": "iron_condor", "Iron Butterfly": "iron_condor",
    "Mean Reversion Spread": "mean_reversion",
    "Call Debit Spread": "call_debit", "Put Debit Spread": "put_debit",
    "Calendar Spread": "calendar", "Bounce Bull Put Spread": "bounce_bull_put",
    "Cash-Secured Put": "wheel", "Covered Call": "wheel",
}

MIN_TRADES = 20          # backlog §6.6: judge after 20–30 trades


@dataclass
class RoundTrip:
    ticker: str
    strategy: str
    playbook: str
    regime: str
    opened_utc: str
    closed_utc: str
    realized_pl: float
    risk: Optional[float]        # max loss × contracts at entry ($); None = contracts unknown
    exit_signal: str
    entry_slippage: Optional[float] = None   # filled − estimated net, per share


@dataclass
class PlaybookStats:
    playbook: str
    trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_pl: float = 0.0
    expectancy: float = 0.0
    return_on_risk: Optional[float] = None
    risk_known_trades: int = 0
    avg_entry_slippage: Optional[float] = None
    strategies: List[str] = field(default_factory=list)
    exit_signals: Dict[str, int] = field(default_factory=dict)
    verdict: str = ""
    suggested_risk_pct: Optional[float] = None


def _fills_by_run(plan_dir: Optional[str]) -> Dict[str, float]:
    """run_id → filled − estimated net credit, from trade plans whose fill
    was recorded (``estimated_net_credit`` present)."""
    out: Dict[str, float] = {}
    if not plan_dir:
        return out
    for fp in Path(plan_dir).glob("trade_plan_*.json"):
        try:
            doc = json.loads(fp.read_text())
        except (OSError, ValueError):
            continue
        for e in doc.get("state_history", []) or []:
            tp = e.get("trade_plan") or {}
            est = tp.get("estimated_net_credit")
            if est is not None and tp.get("net_credit") is not None:
                out[str(e.get("run_id", ""))] = round(float(tp["net_credit"]) - float(est), 4)
    return out


def round_trips(rows: Iterable[Dict[str, Any]], *,
                plan_dir: Optional[str] = None) -> List[RoundTrip]:
    """Pair journal rows into closed round trips (oldest first)."""
    slippage = _fills_by_run(plan_dir)
    pending: Dict[tuple, List[Dict[str, Any]]] = {}
    out: List[RoundTrip] = []
    for rec in rows:
        rs = rec.get("raw_signal") or {}
        if not isinstance(rs, dict):
            rs = {}
        action = rec.get("action")
        key = (rec.get("ticker"), rs.get("strategy"), rs.get("expiration"))
        if action == "submitted" and rec.get("ticker"):
            pending.setdefault(key, []).append(rec)
        elif action == "closed" and rs.get("fill_status") != "dry_run":
            queue = pending.get(key)
            if not queue:
                continue
            o = queue.pop(0)
            ors = o.get("raw_signal") or {}
            strategy = str(ors.get("strategy") or rs.get("strategy") or "?")
            # Spread rows journaled before 2026-10-05 carry no contract
            # count: their risk is unknown, not one contract (that made a
            # 2-contract bear call read −704 % return on risk).
            contracts = ors.get("contracts")
            out.append(RoundTrip(
                ticker=str(rec.get("ticker")), strategy=strategy,
                playbook=str(ors.get("playbook") or _STRATEGY_PLAYBOOK.get(strategy, "unknown")),
                regime=str(ors.get("regime") or "?"),
                opened_utc=str(o.get("timestamp", "")), closed_utc=str(rec.get("timestamp", "")),
                realized_pl=float(rs.get("net_unrealized_pl") or 0.0),
                risk=(float(ors.get("max_loss") or 0.0) * int(contracts)
                      if contracts else None),
                exit_signal=str(rs.get("exit_signal") or "?"),
                entry_slippage=slippage.get(str(ors.get("run_id") or "")),
            ))
    return out


def _verdict(s: PlaybookStats, min_trades: int):
    if s.trades < min_trades:
        return f"collecting ({s.trades}/{min_trades} trades)", None
    if s.return_on_risk is None or s.risk_known_trades < min_trades:
        return (f"{'negative' if s.expectancy < 0 else 'positive'} expectancy — "
                f"risk known for only {s.risk_known_trades}/{s.trades} trades, "
                f"no sizing suggestion"), None
    ror = s.return_on_risk
    if s.expectancy < 0 and ror <= -0.10:
        return "disable suggested — losing ≥ 10 % of risk per trade", 0.0
    if s.expectancy < 0:
        return "shrink — negative expectancy", 0.01
    if ror >= 0.05:
        return "size up — positive expectancy, ≥ 5 % return on risk", 0.03
    return "keep — positive expectancy", 0.02


def scorecard(trips: List[RoundTrip], *, min_trades: int = MIN_TRADES) -> List[PlaybookStats]:
    """Per-playbook stats, most trades first."""
    groups: Dict[str, List[RoundTrip]] = {}
    for t in trips:
        groups.setdefault(t.playbook, []).append(t)
    out: List[PlaybookStats] = []
    for pb, ts in groups.items():
        wins = [t.realized_pl for t in ts if t.realized_pl > 0]
        losses = [t.realized_pl for t in ts if t.realized_pl <= 0]
        known = [t for t in ts if t.risk]
        risk = sum(t.risk for t in known)
        slips = [t.entry_slippage for t in ts if t.entry_slippage is not None]
        s = PlaybookStats(
            playbook=pb, trades=len(ts), wins=len(wins),
            win_rate=round(len(wins) / len(ts), 4),
            avg_win=round(sum(wins) / len(wins), 2) if wins else 0.0,
            avg_loss=round(sum(losses) / len(losses), 2) if losses else 0.0,
            total_pl=round(sum(t.realized_pl for t in ts), 2),
            expectancy=round(sum(t.realized_pl for t in ts) / len(ts), 2),
            return_on_risk=(round(sum(t.realized_pl for t in known) / risk, 4)
                            if risk > 0 else None),
            risk_known_trades=len(known),
            avg_entry_slippage=round(sum(slips) / len(slips), 4) if slips else None,
            strategies=sorted({t.strategy for t in ts}),
        )
        for t in ts:
            s.exit_signals[t.exit_signal] = s.exit_signals.get(t.exit_signal, 0) + 1
        s.verdict, s.suggested_risk_pct = _verdict(s, min_trades)
        out.append(s)
    out.sort(key=lambda s: (-s.trades, s.playbook))
    return out


def scorecard_dict(rows: Iterable[Dict[str, Any]], *, plan_dir: Optional[str] = None,
                   min_trades: int = MIN_TRADES) -> Dict[str, Any]:
    trips = round_trips(rows, plan_dir=plan_dir)
    return {"round_trips": len(trips), "min_trades": min_trades,
            "playbooks": [asdict(s) for s in scorecard(trips, min_trades=min_trades)],
            "note": "advisory — suggested_risk_pct is never applied automatically"}
