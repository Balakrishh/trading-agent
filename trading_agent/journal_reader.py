"""journal_reader.py — single source of truth for journal queries.

Skill 19 §1.2 decoupling pass #2 (2026-05-21). Before this module,
``agent._build_eod_summary``, ``streamlit/live_monitor._render_closed_today``,
and the dashboard realized-P&L tile each opened
``trade_journal/signals_*.jsonl`` independently and applied their own
filters. When the writer's contract changed (the
``dry_run_close`` action label split, the ET-vs-UTC date semantics,
the mode-tag pollution defense), each reader had to be updated in
parallel — and one was always forgotten. The recent
``-$2,976 EOD phantom recap`` was the canonical failure: the
dashboard tile had been fixed (commit ``9cc5636``) but the EOD
builder had not.

This module replaces that pattern with a single
``JournalReader`` class. Every consumer that needs to ask
"what closed today?" / "how much realized P&L?" / "which positions
are stuck?" goes through these named methods. Filter changes happen
in one place, propagate everywhere automatically, and are pinned by
behavioral conformance tests.

Design properties:

  * **Pure read-only.** Never writes. Never mutates the journal file.
    No I/O outside ``open(jsonl_path, "r")``. Trivially mockable for
    tests.

  * **ET-trading-date aware.** All ``*_today`` queries default to
    the current ET calendar date as the boundary. The pre-fix
    behaviour used UTC date which pulled Wed-evening rows into
    Thursday's recap; the new default reflects how operators
    actually think about a trading session.

  * **Mode-aware via file path.** Consumers pick which journal
    to read (``signals_live.jsonl`` vs ``signals_dryrun.jsonl``)
    by passing the path at construction. Defaults to live.

  * **Defense-in-depth filters.** Even with the live/dryrun file
    split (decoupling #1), ``closes_today`` still skips rows
    where ``raw_signal.fill_status == "dry_run"`` so historical
    pre-split journals don't pollute the new view.

  * **Open-once-walk-once.** Each query opens the file, walks all
    lines, returns. No streaming, no caching, no shared state.
    The journal is small enough (~15k rows on a busy week) that
    O(N) per query is fine. Future optimisation (memmap, index)
    would not change the public surface.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone, date
from typing import Iterator, List, Optional, Set
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# All ``*_today`` queries default to ET calendar date. The trading
# session boundary (9:30-16:00 ET on weekdays) lives entirely
# within one ET calendar day, so this matches how operators
# perceive "today's activity".
_ET = ZoneInfo("US/Eastern")

# Path to the canonical live journal. Callers in production code
# pass this explicitly; tests pass a tempdir path.
DEFAULT_LIVE_JOURNAL = "trade_journal/signals_live.jsonl"


# ---------------------------------------------------------------------------
# Result dataclasses — explicit shape so consumers don't depend on dict
# key spelling. A type-check would catch a typo'd key access.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClosedTrade:
    """One complete close from today's ET trading session."""
    ticker: str
    strategy: str
    exit_signal: str
    exit_reason: str
    realized_pl: float
    expiration: str
    timestamp_utc: str  # ISO string, UTC; for display only


@dataclass(frozen=True)
class OpenedTrade:
    """One submitted spread from today's ET trading session."""
    ticker: str
    strategy: str
    credit: float
    expiration: str
    timestamp_utc: str


@dataclass(frozen=True)
class StuckPosition:
    """A position that requires manual operator action right now."""
    ticker: str
    reason: str   # e.g. "PDT block (40310100)" / "Cooldown engaged"


@dataclass(frozen=True)
class SilencedException:
    """One (source, exc_class) group from today's silenced exceptions.

    Skill 34 — operator-visible record of failures the agent caught
    and continued past. Grouped so 50 identical Schwab-token-expired
    warnings show up as one row with ``count=50``, not 50 rows.
    """
    source: str
    exc_class: str
    count: int
    last_message: str
    ticker: str       # "" when the failure isn't ticker-scoped


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------

class JournalReader:
    """Read-only query surface over a single journal file.

    Each public method (``closes_today``, ``realized_pl_today``,
    ``opens_today``, ``stuck_positions``, ``cycle_minute_count_today``,
    ``error_count_today``) returns a fresh answer derived from the
    file's current contents. The methods are intentionally narrow:
    one named method per question an operator asks the dashboard.
    """

    def __init__(self, jsonl_path: str = DEFAULT_LIVE_JOURNAL):
        """``jsonl_path`` is the file to read. Defaults to the live
        journal; tests and the optional dry-run dashboard pass an
        explicit alternate path."""
        self.jsonl_path = jsonl_path

    # ------------------------------------------------------------------
    # Internal walk — every public query goes through this.
    # ------------------------------------------------------------------

    def _iter_rows(self) -> Iterator[dict]:
        """Yield each parseable JSON row. Skips empty lines and
        unparseable lines (those are operator-visible via the
        agent log; the reader silently drops them so a single
        corrupt row doesn't blank the dashboard).
        """
        if not self.jsonl_path or not os.path.isfile(self.jsonl_path):
            return
        try:
            with open(self.jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except OSError as exc:
            logger.warning(
                "JournalReader: failed to read %s: %s",
                self.jsonl_path, exc,
            )
            return

    @staticmethod
    def _today_et() -> date:
        """ET calendar date — the canonical ``today`` boundary."""
        return datetime.now(_ET).date()

    @staticmethod
    def _row_et_date(rec: dict) -> Optional[date]:
        """Convert a journal row's UTC timestamp to its ET calendar date.
        Returns None if the timestamp is missing or malformed."""
        ts_str = rec.get("timestamp", "")
        if not ts_str:
            return None
        try:
            ts_dt = datetime.fromisoformat(ts_str)
        except ValueError:
            return None
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.replace(tzinfo=timezone.utc)
        return ts_dt.astimezone(_ET).date()

    @staticmethod
    def _is_real_close(rec: dict) -> bool:
        """True iff this row represents an actual broker close.

        Defense-in-depth alongside the live/dryrun file split. Even
        if a row sneaks into the live file with ``fill_status="dry_run"``
        (historical pre-split data, manual journal append, etc.),
        the reader excludes it from realized-P&L sums and the
        Closed Today panel."""
        if rec.get("action") != "closed":
            return False
        rs = rec.get("raw_signal") or {}
        if not isinstance(rs, dict):
            return True   # untagged → assume real, matches legacy semantics
        return rs.get("fill_status") != "dry_run"

    # ------------------------------------------------------------------
    # Public query surface
    # ------------------------------------------------------------------

    def closes_today(self) -> List[ClosedTrade]:
        """Every real broker close from today's ET trading session.

        Filters:
          * action == "closed"
          * ET calendar date of timestamp == today (ET)
          * fill_status != "dry_run" (defense; live file shouldn't
            have any after decoupling #1, but historical pi journals
            still might)
        """
        today_et = self._today_et()
        out: List[ClosedTrade] = []
        for rec in self._iter_rows():
            if self._row_et_date(rec) != today_et:
                continue
            if not self._is_real_close(rec):
                continue
            rs = rec.get("raw_signal") or {}
            if not isinstance(rs, dict):
                rs = {}
            out.append(ClosedTrade(
                ticker=rec.get("ticker", "") or "",
                strategy=str(rs.get("strategy", "?")),
                exit_signal=str(rs.get("exit_signal", "?")),
                exit_reason=str(rs.get("exit_reason", "") or ""),
                realized_pl=float(rs.get("net_unrealized_pl") or 0.0),
                expiration=str(rs.get("expiration", "") or ""),
                timestamp_utc=rec.get("timestamp", "") or "",
            ))
        return out

    def realized_pl_today(self) -> float:
        """Sum of realized P&L across ``closes_today``. One-line caller
        for the dashboard tile + EOD recap."""
        return sum(c.realized_pl for c in self.closes_today())

    def opens_today(self) -> List[OpenedTrade]:
        """Every submitted spread from today's ET trading session."""
        today_et = self._today_et()
        out: List[OpenedTrade] = []
        for rec in self._iter_rows():
            if self._row_et_date(rec) != today_et:
                continue
            if rec.get("action") != "submitted":
                continue
            if not rec.get("ticker"):
                continue
            rs = rec.get("raw_signal") or {}
            if not isinstance(rs, dict):
                rs = {}
            out.append(OpenedTrade(
                ticker=rec["ticker"],
                strategy=str(rs.get("strategy", "?")),
                credit=float(rs.get("net_credit") or 0.0),
                expiration=str(rs.get("expiration", "") or ""),
                timestamp_utc=rec.get("timestamp", "") or "",
            ))
        return out

    def stuck_positions(self) -> List[StuckPosition]:
        """Positions currently requiring operator intervention.

        Two stuck states (mutually exclusive per ticker — PDT wins):

          * PDT-blocked today: latest close_failed row carries
            ``pdt_blocked_today=True`` AND
            ``pdt_blocked_date == today UTC``
          * Close cooldown active: latest close_failed row carries
            ``close_cooldown_until`` that hasn't expired yet

        Uses UTC for ``pdt_blocked_date`` comparison because the
        marker is UTC-keyed at write-time (matches
        ``agent._journal_close_event``). The ET trading-date filter
        on the outer row scan is independent.
        """
        today_et = self._today_et()
        today_utc_iso = datetime.now(timezone.utc).date().isoformat()
        now_utc = datetime.now(timezone.utc)
        seen: dict[str, str] = {}
        for rec in self._iter_rows():
            if self._row_et_date(rec) != today_et:
                continue
            if rec.get("action") != "close_failed":
                continue
            ticker = rec.get("ticker", "") or ""
            if not ticker:
                continue
            rs = rec.get("raw_signal") or {}
            if not isinstance(rs, dict):
                rs = {}
            # PDT block (preferred — operator-actionable bigger)
            if (rs.get("pdt_blocked_today")
                    and rs.get("pdt_blocked_date") == today_utc_iso):
                seen[ticker] = (
                    "PDT block (Alpaca 40310100) — close in Alpaca UI "
                    "or wait for next session"
                )
                continue
            # Cooldown still active
            cd = rs.get("close_cooldown_until")
            if cd and ticker not in seen:
                try:
                    cd_dt = datetime.fromisoformat(cd)
                    if cd_dt.tzinfo is None:
                        cd_dt = cd_dt.replace(tzinfo=timezone.utc)
                    if cd_dt > now_utc:
                        seen[ticker] = (
                            "Close cooldown active — manual close "
                            "required to clear zombie state"
                        )
                except ValueError:
                    pass
        return [
            StuckPosition(ticker=t, reason=r)
            for t, r in sorted(seen.items())
        ]

    def cycle_minute_count_today(self) -> int:
        """Distinct minute-bucket count from today's journal writes.
        Health proxy: ``0`` means agent didn't run; ``>200`` means
        active all-day cron cadence."""
        today_et = self._today_et()
        minutes: Set[str] = set()
        for rec in self._iter_rows():
            if self._row_et_date(rec) != today_et:
                continue
            ts_str = rec.get("timestamp", "")
            if len(ts_str) >= 16:
                minutes.add(ts_str[:16])
        return len(minutes)

    def error_count_today(self) -> int:
        """Count of error / warning / cycle_error rows for today's ET date."""
        today_et = self._today_et()
        n = 0
        for rec in self._iter_rows():
            if self._row_et_date(rec) != today_et:
                continue
            if rec.get("action") in ("error", "cycle_error", "warning"):
                n += 1
        return n

    def silenced_exceptions_today(self) -> List["SilencedException"]:
        """Skill 34 — every ``silenced_exception`` row for today's
        ET trading session, grouped per (source, exc_class) with a
        running count + last-seen message.

        Consumed by the EOD recap so the operator sees a tally of
        what failed quietly today even when nothing else broke. The
        first occurrence per group fired a Telegram alert via
        ``ExceptionMonitor`` — the recap is the catch-up view."""
        today_et = self._today_et()
        # Group by (source, exc_class) → (count, last_message, last_ticker)
        groups: dict[tuple[str, str], list] = {}
        for rec in self._iter_rows():
            if rec.get("action") != "silenced_exception":
                continue
            if self._row_et_date(rec) != today_et:
                continue
            rs = rec.get("raw_signal") or {}
            if not isinstance(rs, dict):
                continue
            key = (rs.get("source", "?"), rs.get("exc_class", "?"))
            entry = groups.setdefault(key, [0, "", ""])
            entry[0] += 1
            entry[1] = rs.get("message", "") or entry[1]
            tk = rec.get("ticker", "") or ""
            if tk and tk != "__silenced__":
                entry[2] = tk
        out: List[SilencedException] = []
        for (source, exc_class), (count, msg, ticker) in groups.items():
            out.append(SilencedException(
                source=source, exc_class=exc_class,
                count=count, last_message=msg[:200],
                ticker=ticker,
            ))
        # Highest-count first — operator scans top to bottom
        out.sort(key=lambda s: (-s.count, s.source))
        return out

    def tickers_opened_today_utc(self) -> Set[str]:
        """Return underlyings that submitted a new spread today (UTC).

        Skill 17 §4 — used to suppress same-day REGIME_SHIFT exits
        on PDT-restricted accounts (< $25K equity).

        UTC-keyed (not ET) because the FINRA same-day-open rule is
        defined against the broker's UTC clock, not the operator's
        local trading session. The action filter is ``submitted`` —
        only successful order submissions count. Skipped rows from
        a previous-cycle dedup or a rejected risk check do not.

        Verbatim port of agent._tickers_opened_today.
        """
        today_utc = datetime.now(timezone.utc).date()
        tickers: Set[str] = set()
        for rec in self._iter_rows():
            if rec.get("action") != "submitted":
                continue
            ts_str = rec.get("timestamp", "")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts.astimezone(timezone.utc).date() != today_utc:
                continue
            tk = rec.get("ticker")
            if tk:
                tickers.add(tk)
        return tickers

    def telegram_alert_sent_today_utc(
        self, ticker: str, alert_type: str,
    ) -> bool:
        """True if a ``telegram_alert_sent`` journal row matching
        ``(ticker, alert_type)`` was written earlier the same UTC day.

        Skill 32 §3.4 — dedup gate. The first time the agent detects
        a PDT block on a ticker, ``notify_pdt_block`` fires and a
        ``telegram_alert_sent`` row gets written. Across the next ~78
        cycles of the same trading day, the same DIA detection would
        otherwise re-fire the alert — this helper short-circuits the
        send so the operator sees ONE alert per ticker per day per
        type.

        Date-keyed (UTC) → self-clears at midnight, matching the
        pdt_blocked_today marker's lifetime. The alert_type dimension
        is intentional: PDT-block + close-cooldown + position-closed
        each get separate dedup keys.

        EOD-specific dedup (skill 32 §3.8, 2026-05-23 fix): when
        ``alert_type`` starts with ``"eod_summary:"`` the embedded
        ET trading session date is used as the dedup date instead of
        today's UTC date. This closes the cross-UTC-midnight hole
        that caused two EOD recaps to fire (4 PM ET + 8 PM ET) — the
        4 PM write tagged the row with UTC-today, but the 8 PM lookup
        used the next UTC day's date and missed the row.

        Fail-open: a journal-read failure returns False (prefer one
        extra alert over a missed alert).
        """
        if alert_type.startswith("eod_summary:"):
            dedup_date = alert_type.split(":", 1)[1]
        else:
            dedup_date = datetime.now(timezone.utc).date().isoformat()
        for rec in self._iter_rows():
            if rec.get("action") != "telegram_alert_sent":
                continue
            if rec.get("ticker") != ticker:
                continue
            rs = rec.get("raw_signal") or {}
            if not isinstance(rs, dict):
                continue
            if rs.get("alert_type") != alert_type:
                continue
            # Field is `alert_date` (matches what _send_telegram_alert
            # writes, not the more-natural `sent_date`).
            if rs.get("alert_date") != dedup_date:
                continue
            return True
        return False

    def reject_reasons_today(self, top_n: int = 5) -> List[tuple]:
        """Return today's top rejection reasons sorted by count.

        Skill 32 §3.8.1 (2026-05-28). When the agent runs many cycles
        but opens zero positions, the operator needs to know WHY every
        candidate was rejected. The journal carries each rejection's
        reason in ``raw_signal.rejection_reason`` (free-form text,
        e.g. "No positive-EV candidate", "C/W ratio 0.16 < min 0.25",
        "RSI=72 ≥ 70 overbought", etc.).

        Returns a list of (reason_text, count) tuples, top_n longest
        first, for today's ET trading session. Empty list if no
        rejections (which on a normal day would mean trades opened
        cleanly — also useful information).

        The EOD builder bubbles this into the Telegram alert body so
        the operator gets "12 rejected: top reasons were..." instead
        of silence on a no-trade day.
        """
        from collections import Counter
        today_et = self._today_et()
        c: Counter = Counter()
        for rec in self._iter_rows():
            if self._row_et_date(rec) != today_et:
                continue
            action = rec.get("action", "")
            # Cover the rejection-shaped actions, including
            # skipped_* variants that also explain why no trade.
            if not (action == "rejected"
                    or action.startswith("skipped_")):
                continue
            rs = rec.get("raw_signal") or {}
            if not isinstance(rs, dict):
                continue
            reason = (
                rs.get("rejection_reason")
                or rs.get("reason")
                or "(no reason recorded)"
            )
            # Truncate noisy long reasons but keep them grouping-stable
            # (don't include unique numbers in the key).
            reason_text = str(reason)[:120]
            c[reason_text] += 1
        return c.most_common(top_n)

    def account_balance_today_endpoints(
        self,
    ) -> tuple[Optional[float], Optional[float]]:
        """Return (starting_balance, last_balance) seen in today's
        journal rows. Both ``None`` if no balance snapshots exist.

        Used by the EOD recap to render
        ``Balance: $start → $end (Δ%, Δ$)``.
        """
        today_et = self._today_et()
        first: Optional[float] = None
        last: Optional[float] = None
        for rec in self._iter_rows():
            if self._row_et_date(rec) != today_et:
                continue
            rs = rec.get("raw_signal") or {}
            if not isinstance(rs, dict):
                continue
            bal = rs.get("account_balance")
            if isinstance(bal, (int, float)) and bal > 0:
                if first is None:
                    first = float(bal)
                last = float(bal)
        return first, last


__all__ = [
    "ClosedTrade",
    "OpenedTrade",
    "StuckPosition",
    "SilencedException",
    "JournalReader",
    "DEFAULT_LIVE_JOURNAL",
]
