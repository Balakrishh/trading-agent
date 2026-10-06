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

State: ``trade_journal/entry_candidates.json`` (atomic), rebuilt every
cycle because the agent process restarts each cycle.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
STATE_PATH = Path(os.environ.get("TRADING_AGENT_ENTRY_CANDIDATES",
                                 "trade_journal/entry_candidates.json"))
# Consecutive means consecutive: cycles run ≈ 75 s apart, so a gap longer
# than this means a cycle was skipped (capitulation, risk cap, restart) and
# the streak starts over.
MAX_GAP_SECONDS = 180


def signature(plan: Any) -> str:
    """What must stay the same across cycles: strategy + expiration (+ the
    far expiry for a calendar)."""
    return "|".join([str(plan.strategy_name), str(plan.expiration),
                     str(getattr(plan, "far_expiration", "") or "")])


@dataclass
class Confirmation:
    count: int
    required: int
    signature: str

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

    def begin(self) -> "EntryConfirmations":
        try:
            self.previous = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.previous = {}
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
        if prev.get("signature") == sig and fresh:
            count = int(prev.get("count", 0)) + 1
        self.current[ticker] = {"signature": sig, "count": count,
                                "first_seen": prev.get("first_seen") if count > 1 else self.now.isoformat(),
                                "last_seen": self.now.isoformat()}
        return Confirmation(count, self.required, sig)

    def consumed(self, ticker: str) -> None:
        """The order went out — the next entry on this ticker starts over."""
        self.current.pop(ticker, None)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.current, indent=2))
        tmp.replace(self.path)


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
