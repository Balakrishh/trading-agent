"""debit_ladder.py — mid-first entry ladder for debit verticals (backlog C1),
shadow measurement (added 2026-10-16, skill 59 §3). The live ladder
(2026-10-27) is ``OrderExecutor._debit_ladder_probes``: it uses
``ladder_prices`` from here but submits within the cycle, so the shadow
trackers below run only in ``shadow`` mode.

Debit orders go out at the natural price (pay the long ask, take the short
bid) and a plan is rejected when natural > mid × (1 + ``debit_max_overpay``)
(``debit_above_fair``). On 2026-10-15 that rejected 548 put-debit plans,
most of them a few cents over: GLD natural $4.25 vs cap $4.17, TLT 5.7 %
over mid. A ladder would instead offer mid first and step up to the cap,
never past it.

This module only *measures* what the ladder would have done. Prices are
``debit_ladder_steps`` evenly spaced from mid (rounded up to the cent) to
the cap (rounded down), each held for ``debit_ladder_step_cycles`` agent
cycles. Two kinds of tracker:

* ``submitted`` — a debit order the agent really sent at natural. The
  ladder would have saved ``actual − ladder`` per share, or missed the
  trade if it never filled.
* ``unlocked`` — the best pair the scanner rejected only for
  ``debit_above_fair`` whose reward/risk still clears the floor at the
  cap. A fill means the ladder would have opened a trade the agent did
  not. Other gates (risk, market state, ladder, confirmation) might still
  have blocked it, so the count is an upper bound.

Fill proxy: a step fills when the spread's natural debit, re-quoted on a
later cycle inside that step, is ≤ the step price — i.e. the limit became
marketable. Real markets often fill inside the spread, so this is a lower
bound on fills and savings; it is roughly what the paper simulator does.
Never places, changes or cancels an order.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from trading_agent.entry_confirmation import score_legs

LADDER_MODES = ("off", "shadow", "live")  # live: OrderExecutor._debit_ladder_probes
LADDER_STRATEGIES = frozenset({"Call Debit Spread", "Put Debit Spread"})
STATE_PATH = Path(os.environ.get("TRADING_AGENT_DEBIT_LADDER",
                                 "trade_journal/debit_ladder_shadow.json"))
# A tracker whose quotes never come back is dropped after this long.
MAX_TRACK_SECONDS = 30 * 60


@dataclass(frozen=True)
class LadderParams:
    steps: int = 4
    step_cycles: int = 1

    @classmethod
    def from_preset(cls, preset: Any) -> "LadderParams":
        return cls(max(1, int(getattr(preset, "debit_ladder_steps", cls.steps))),
                   max(1, int(getattr(preset, "debit_ladder_step_cycles", cls.step_cycles))))


def ladder_prices(mid: float, cap: float, steps: int) -> List[float]:
    """Ascending limit prices from mid to cap, cent-rounded and deduplicated.
    The first is mid rounded up, the last the cap rounded down; when the
    two round to the same cent (or ``steps`` ≤ 1) there is one price, the
    cap."""
    lo = math.ceil(round(mid * 100, 6)) / 100
    hi = math.floor(round(cap * 100, 6)) / 100
    if hi <= lo or steps <= 1:
        return [round(hi, 2)]
    out: List[float] = []
    for i in range(steps):
        p = round(lo + (hi - lo) * i / (steps - 1), 2)
        if not out or p > out[-1]:
            out.append(p)
    return out


def quote_debit(legs: List[Dict[str, Any]], width: float) -> Optional[Dict[str, float]]:
    """``{"natural", "mid"}`` debit per share from leg quotes (``action``,
    ``bid``, ``ask``), or None without a usable debit quote."""
    s = score_legs(legs, width)
    if s is None or s["net"] >= 0:
        return None
    natural = -s["net"]
    return {"natural": round(natural, 4), "mid": round(natural - s["gap"], 4)}


def ladder_outcome(prices: List[float], naturals: List[Optional[float]],
                   step_cycles: int) -> Optional[Dict[str, Any]]:
    """Resolve a tracker. ``naturals[k]`` is the natural debit on the k-th
    observation (0 = when the plan was made); step i covers observations
    [i·step_cycles, (i+1)·step_cycles). Returns ``{"filled", "step",
    "price"}`` once a step fills or every step has been observed, else
    None (keep watching)."""
    for i, price in enumerate(prices):
        window = naturals[i * step_cycles:(i + 1) * step_cycles]
        if any(n is not None and n <= price + 1e-9 for n in window):
            return {"filled": True, "step": i, "price": price}
        if len(window) < step_cycles:
            return None
    return {"filled": False, "step": None, "price": None}


class LadderShadow:
    """Shadow trackers persisted across cycles in ``STATE_PATH`` (atomic
    temp + rename). ``done_today`` keeps the keys already resolved today so
    a near miss rejected every cycle is tracked once a day."""

    def __init__(self, path: Optional[Path] = None, now: Optional[datetime] = None):
        self.path = Path(path or STATE_PATH)
        self.now = now or datetime.now().astimezone()
        self.trackers: Dict[str, Dict[str, Any]] = {}
        self.done_today: List[str] = []
        self._day = self.now.date().isoformat()
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError):
            raw = {}
        self.trackers = dict(raw.get("trackers") or {})
        if raw.get("day") == self._day:
            self.done_today = list(raw.get("done_today") or [])

    # ── start ─────────────────────────────────────────────────────────
    def _start(self, key: str, tracker: Dict[str, Any]) -> bool:
        if key in self.trackers or key in self.done_today:
            return False
        self.trackers[key] = {**tracker, "started": self.now.isoformat()}
        return True

    def start_submitted(self, ticker: str, plan: Any, qty: int,
                        actual_debit: Optional[float], params: LadderParams) -> bool:
        """Track a debit order sent at natural. Uses the plan's leg quotes
        for the mid; False when the plan is not a debit vertical or has no
        usable quotes."""
        if plan.strategy_name not in LADDER_STRATEGIES or not plan.legs:
            return False
        legs = [{"symbol": l.symbol, "action": l.action, "bid": l.bid, "ask": l.ask}
                for l in plan.legs]
        width = float(plan.spread_width or 0)
        q = quote_debit(legs, width)
        cap = getattr(plan, "max_debit", None)
        if q is None or not cap:
            return False
        actual = float(actual_debit) if actual_debit else q["natural"]
        return self._start(f"{ticker}|{plan.expiration}|" + "|".join(l["symbol"] for l in legs), {
            "kind": "submitted", "ticker": ticker, "strategy": plan.strategy_name,
            "expiration": plan.expiration, "width": width, "qty": int(qty or 1),
            "legs": [{"symbol": l["symbol"], "action": l["action"]} for l in legs],
            "cap": round(float(cap), 4), "mid": q["mid"], "actual_debit": round(actual, 4),
            "prices": ladder_prices(q["mid"], float(cap), params.steps),
            "naturals": [q["natural"]]})

    def start_unlocked(self, ticker: str, strategy: str, cand: Dict[str, Any],
                       params: LadderParams) -> bool:
        """Track the scanner's best ``debit_above_fair`` near miss
        (``ScanDiagnostics.ladder_candidate``)."""
        try:
            legs = [{"symbol": cand["long_symbol"], "action": "buy"},
                    {"symbol": cand["short_symbol"], "action": "sell"}]
            mid, cap, natural = float(cand["mid"]), float(cand["cap"]), float(cand["natural"])
        except (KeyError, TypeError, ValueError):
            return False
        return self._start(f"{ticker}|{cand.get('expiration')}|{legs[0]['symbol']}|{legs[1]['symbol']}", {
            "kind": "unlocked", "ticker": ticker, "strategy": strategy,
            "expiration": cand.get("expiration"), "width": float(cand.get("width") or 0),
            "qty": 1, "legs": legs, "cap": round(cap, 4), "mid": round(mid, 4),
            "actual_debit": None, "reward_risk_at_cap": cand.get("reward_risk_at_cap"),
            "prices": ladder_prices(mid, cap, params.steps), "naturals": [round(natural, 4)]})

    # ── advance ───────────────────────────────────────────────────────
    def advance(self, quote_fn: Callable[[List[str]], Dict[str, Dict[str, float]]],
                params: LadderParams) -> List[Dict[str, Any]]:
        """Re-quote every tracker started on an earlier cycle and return the
        outcomes resolved now (journal rows)."""
        out: List[Dict[str, Any]] = []
        for key, tr in list(self.trackers.items()):
            if tr.get("started") == self.now.isoformat():
                continue
            try:
                age = (self.now - datetime.fromisoformat(tr["started"])).total_seconds()
            except (KeyError, TypeError, ValueError):
                age = MAX_TRACK_SECONDS + 1
            if age > MAX_TRACK_SECONDS:
                del self.trackers[key]
                continue
            quotes = quote_fn([l["symbol"] for l in tr["legs"]]) or {}
            q = quote_debit([{**l, **(quotes.get(l["symbol"]) or {})} for l in tr["legs"]],
                            tr["width"])
            tr["naturals"].append(None if q is None else q["natural"])
            res = ladder_outcome(tr["prices"], tr["naturals"], params.step_cycles)
            if res is None:
                continue
            out.append(self._row(tr, res))
            del self.trackers[key]
            self.done_today.append(key)
        return out

    @staticmethod
    def _row(tr: Dict[str, Any], res: Dict[str, Any]) -> Dict[str, Any]:
        actual = tr.get("actual_debit")
        saving = None
        if tr["kind"] == "submitted" and res["filled"]:
            saving = round((actual - res["price"]) * 100 * tr["qty"], 2)
        return {"ticker": tr["ticker"], "kind": tr["kind"], "strategy": tr["strategy"], "expiration": tr["expiration"],
                "qty": tr["qty"], "mid": tr["mid"], "cap": tr["cap"], "prices": tr["prices"],
                "naturals": tr["naturals"], "actual_debit": actual,
                "filled": res["filled"], "fill_step": res["step"], "ladder_debit": res["price"],
                "saving_usd": saving,
                "missed_trade": tr["kind"] == "submitted" and not res["filled"],
                "would_open": tr["kind"] == "unlocked" and res["filled"],
                "reward_risk_at_cap": tr.get("reward_risk_at_cap")}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"day": self._day, "trackers": self.trackers,
                                   "done_today": self.done_today[-200:]}, indent=2))
        tmp.replace(self.path)
