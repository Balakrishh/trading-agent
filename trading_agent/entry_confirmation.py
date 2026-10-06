"""entry_confirmation.py — wait before opening (2026-10-05).

An approved plan is only *submitted* once the same ticker has produced the
same signal — strategy + expiration — on ``entry_confirm_cycles``
consecutive cycles (≈ 75 s apart). Strikes may drift a grid step between
cycles; the order uses the latest plan and still passes the executor's
live-price recheck. A ticker that produces no approved plan in a cycle, or
a different strategy / expiry, loses its count — only an unbroken streak
confirms. 2026-10-05: IWM was bought as "bearish" and read "sideways" one
cycle later; three cycles of confirmation would have skipped it.

Two more entry gates live here:

* no new entries before ``no_entry_before_et`` (09:45 — the open's quotes
  are the widest of the day);
* at most ``max_new_entries_per_hour`` submissions in any rolling hour
  (2026-10-05: four entries in 8 minutes spent the whole risk budget).

**Entry timing** (``entry_timing_mode``) turns the wait into watching the
price. Every cycle the candidate's quality is scored on a ratio that stays
comparable when strikes drift a grid step (credit ÷ width; reward ÷ risk
for debit verticals; −debit ÷ mid for calendars) together with the
natural-to-mid gap. Once confirmed, the order goes out when the score is
within ``entry_best_tolerance_pct`` of the best seen and the gap is at or
below the window's median; after ``entry_max_wait_cycles`` it enters if the
score is no more than ``entry_chase_limit_pct`` worse than at confirmation,
otherwise it skips (never chase a deteriorating price).

* ``off``    — enter at confirmation.
* ``shadow`` — enter at confirmation, then keep quoting the filled legs for
  the rest of the window and journal ``entry_timing_shadow``: the cycle and
  price the rule would have used and the dollar difference.
* ``live``   — the rule decides when to enter (or skip).

State: ``trade_journal/entry_candidates.json`` (atomic), rebuilt every
cycle because the agent process restarts each cycle.
"""
from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
STATE_PATH = Path(os.environ.get("TRADING_AGENT_ENTRY_CANDIDATES",
                                 "trade_journal/entry_candidates.json"))
# Consecutive means consecutive: cycles run ≈ 75 s apart, so a gap longer
# than this means a cycle was skipped (capitulation, risk cap, restart) and
# the streak starts over.
MAX_GAP_SECONDS = 180
# A shadow timing tracker is dropped after this long (quotes never usable).
MAX_SHADOW_SECONDS = 30 * 60


def signature(plan: Any) -> str:
    """What must stay the same across cycles: strategy + expiration (+ the
    far expiry for a calendar)."""
    return "|".join([str(plan.strategy_name), str(plan.expiration),
                     str(getattr(plan, "far_expiration", "") or "")])


TIMING_MODES = ("off", "shadow", "live")
TIMING_KEY = "__timing_shadow__"          # shadow trackers inside the state file


@dataclass(frozen=True)
class TimingParams:
    max_wait_cycles: int = 8
    best_tolerance_pct: float = 0.01
    chase_limit_pct: float = 0.03

    @classmethod
    def from_preset(cls, preset: Any) -> "TimingParams":
        return cls(int(getattr(preset, "entry_max_wait_cycles", cls.max_wait_cycles)),
                   float(getattr(preset, "entry_best_tolerance_pct", cls.best_tolerance_pct)),
                   float(getattr(preset, "entry_chase_limit_pct", cls.chase_limit_pct)))


def score_legs(legs: List[Dict[str, Any]], width: float) -> Optional[Dict[str, float]]:
    """Score one cycle's price for a structure. ``legs``: dicts with
    ``action`` ("sell"/"buy"), ``bid``, ``ask``. Returns ``score`` (higher
    = better entry), ``gap`` (mid − natural net, ≥ 0 normally) and ``net``
    (natural net credit; negative = debit), or None without usable quotes.

    credit:          score = net ÷ width
    debit vertical:  score = (width − debit) ÷ debit
    calendar:        score = −debit ÷ mid debit
    """
    nat = mid = 0.0
    for leg in legs:
        bid, ask = float(leg.get("bid") or 0), float(leg.get("ask") or 0)
        if ask <= 0:
            return None
        m = (bid + ask) / 2 if bid > 0 else ask
        if leg["action"] == "sell":
            nat += bid
            mid += m
        else:
            nat -= ask
            mid -= m
    gap = mid - nat
    if nat > 0 and width > 0:
        score = nat / width
    elif nat < 0 and width > 0:
        debit = -nat
        score = (width - debit) / debit
    elif nat < 0 and mid < 0:
        score = -(-nat) / (-mid)
    else:
        return None
    return {"score": round(score, 6), "gap": round(gap, 4), "net": round(nat, 4)}


def score_plan(plan: Any) -> Optional[Dict[str, float]]:
    """``score_legs`` on a plan's own leg quotes; None for a plan without legs."""
    legs = getattr(plan, "legs", None) or []
    if not legs:
        return None
    return score_legs([{"action": l.action, "bid": l.bid, "ask": l.ask} for l in legs],
                      float(getattr(plan, "spread_width", 0) or 0))


def timing_decision(history: List[Dict[str, float]], required: int,
                    params: TimingParams) -> Tuple[str, str]:
    """``("enter" | "wait" | "skip", reason)`` for a confirmed candidate.
    ``history`` holds one score dict per consecutive cycle, oldest first;
    the confirmation point is ``history[required − 1]``."""
    n = len(history)
    if n < required:
        return "wait", f"confirming {n}/{required}"
    cur = history[-1]
    best = max(h["score"] for h in history)
    tol = params.best_tolerance_pct * abs(best)
    median_gap = statistics.median(h["gap"] for h in history)
    if cur["score"] >= best - tol and cur["gap"] <= median_gap + 1e-9:
        return "enter", (f"timing: best price in {n} cycles (score {cur['score']:.4f}, "
                         f"gap {cur['gap']:.2f} ≤ median {median_gap:.2f})")
    if n >= max(required, params.max_wait_cycles):
        ref = history[required - 1]["score"]
        if cur["score"] >= ref - params.chase_limit_pct * abs(ref):
            return "enter", (f"timing: max wait {n} cycles, score {cur['score']:.4f} within "
                             f"{params.chase_limit_pct:.0%} of confirmation {ref:.4f}")
        return "skip", (f"timing: price worsened — score {cur['score']:.4f} vs "
                        f"{ref:.4f} at confirmation after {n} cycles")
    return "wait", (f"timing: waiting {n}/{params.max_wait_cycles} (score "
                    f"{cur['score']:.4f}, best {best:.4f}, gap {cur['gap']:.2f})")


@dataclass
class Confirmation:
    count: int
    required: int
    signature: str
    history: List[Dict[str, float]] = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return self.count >= self.required

    @property
    def reason(self) -> str:
        return (f"entry_confirming {self.count}/{self.required} "
                f"({self.signature.replace('|', ' ').strip()})")


class EntryConfirmations:
    """Per-cycle candidate tracker. ``begin()`` loads the previous cycle's
    candidates, ``observe()`` records this cycle's approved plans,
    ``save()`` keeps only what was observed this cycle."""

    def __init__(self, required: int, path: Optional[Path] = None,
                 now: Optional[datetime] = None):
        self.required = max(1, int(required))
        self.path = path or STATE_PATH
        self.now = now or datetime.now(timezone.utc)
        self.previous: Dict[str, Dict[str, Any]] = {}
        self.current: Dict[str, Dict[str, Any]] = {}
        # Shadow trackers for filled entries (entry_timing_mode="shadow"),
        # keyed by ticker; carried across cycles until resolved.
        self.timing: Dict[str, Dict[str, Any]] = {}
        self.resolved: List[Dict[str, Any]] = []

    def begin(self) -> "EntryConfirmations":
        try:
            self.previous = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.previous = {}
        self.timing = dict(self.previous.pop(TIMING_KEY, {}) or {})
        return self

    def observe(self, ticker: str, plan: Any) -> Confirmation:
        sig = signature(plan)
        prev = self.previous.get(ticker) or {}
        count = 1
        try:
            last = datetime.fromisoformat(prev.get("last_seen", ""))
            fresh = (self.now - last).total_seconds() <= MAX_GAP_SECONDS
        except (TypeError, ValueError):
            fresh = False
        history: List[Dict[str, float]] = []
        if prev.get("signature") == sig and fresh:
            count = int(prev.get("count", 0)) + 1
            history = list(prev.get("history") or [])
        scored = score_plan(plan)
        if scored is not None:
            history.append(scored)
        self.current[ticker] = {"signature": sig, "count": count,
                                "first_seen": prev.get("first_seen") if count > 1 else self.now.isoformat(),
                                "last_seen": self.now.isoformat(), "history": history}
        return Confirmation(count, self.required, sig, history)

    def consumed(self, ticker: str, plan: Any = None, qty: int = 0,
                 shadow: bool = False, params: Optional[TimingParams] = None) -> None:
        """The order went out — the next entry on this ticker starts over.
        With ``shadow``, judge the timing rule against this entry: if it
        would also have entered now, record that (``resolved``); otherwise
        keep quoting the filled legs (``advance_shadow``)."""
        cand = self.current.pop(ticker, None)
        if not (shadow and plan is not None and cand and plan.legs):
            return
        entry = score_plan(plan)
        tracker = {
            "strategy": plan.strategy_name, "expiration": plan.expiration,
            "width": float(plan.spread_width or 0), "qty": int(qty or 1),
            "legs": [{"symbol": l.symbol, "action": l.action} for l in plan.legs],
            "actual_net": entry["net"] if entry else float(plan.net_credit),
            "history": list(cand.get("history") or []),
            "started": self.now.isoformat()}
        decision, reason = timing_decision(tracker["history"], self.required,
                                           params or TimingParams())
        if decision == "enter":
            self.resolved.append(self._outcome(ticker, tracker, decision, reason,
                                               tracker["actual_net"]))
        else:
            self.timing[ticker] = tracker

    def _outcome(self, ticker, tr, decision, reason, net):
        improvement = (round((net - tr["actual_net"]) * 100 * tr["qty"], 2)
                       if decision == "enter" else None)
        return {"ticker": ticker, "strategy": tr["strategy"], "expiration": tr["expiration"],
                "decision": decision, "cycle": len(tr["history"]),
                "actual_net": tr["actual_net"],
                "timing_net": net if decision == "enter" else None,
                "improvement_usd": improvement, "reason": reason}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = dict(self.current)
        if self.timing:
            payload[TIMING_KEY] = self.timing
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)

    def advance_shadow(self, quote_fn, params: TimingParams) -> List[Dict[str, Any]]:
        """Re-quote every shadow-tracked entry, apply the timing rule to its
        full history (confirmation cycles + cycles since the real entry) and
        return the outcomes resolved this cycle, including any recorded at
        entry time: ``{ticker, strategy, decision, cycle, actual_net,
        timing_net, improvement_usd, reason}``. ``improvement_usd`` > 0 means
        the rule would have filled better (more credit / less debit)."""
        done = list(self.resolved)
        self.resolved.clear()
        for ticker, tr in list(self.timing.items()):
            if tr.get("started") == self.now.isoformat():
                continue                      # filled this cycle; first re-quote next cycle
            try:
                age = (self.now - datetime.fromisoformat(tr["started"])).total_seconds()
            except (KeyError, TypeError, ValueError):
                age = 0
            if age > MAX_SHADOW_SECONDS:
                del self.timing[ticker]
                continue
            quotes = quote_fn([l["symbol"] for l in tr["legs"]]) or {}
            legs = [{"action": l["action"], **(quotes.get(l["symbol"]) or {})} for l in tr["legs"]]
            scored = score_legs(legs, tr["width"])
            if scored is None:
                continue                      # no usable quote this cycle; retry next
            tr["history"].append(scored)
            decision, reason = timing_decision(tr["history"], self.required, params)
            if decision == "wait":
                continue
            done.append(self._outcome(ticker, tr, decision, reason, scored["net"]))
            del self.timing[ticker]
        return done


def before_entry_window(now_utc: datetime, not_before_et: str) -> bool:
    """True when the ET clock is earlier than ``not_before_et`` ("HH:MM")."""
    try:
        hh, mm = (int(x) for x in str(not_before_et).split(":"))
    except ValueError:
        return False
    return now_utc.astimezone(ET).time() < time(hh, mm)


def entries_in_last_hour(submitted_utc: Iterable[datetime], now_utc: datetime) -> int:
    cutoff = now_utc - timedelta(hours=1)
    return sum(1 for t in submitted_utc if t >= cutoff)
