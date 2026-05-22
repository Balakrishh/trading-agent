"""exception_monitor.py — turn silenced exceptions into operator alerts.

Skill 34 — every ``except Exception`` block in the agent's hot path
is a place a bug can hide. Pre-2026-05-22 the only signal of a
swallowed failure was a single ``logger.warning`` line that nobody
read until the symptoms became visible elsewhere (the 48-hour
Telegram-ticker TypeError; the silently-failed Schwab token refresh).

This module turns those warning lines into a paging system:

  1. **Record** — every silenced exception (or operationally
     significant warning) is journalled as a structured row so
     the count, source, and first/last occurrence are queryable.
  2. **Dedup** — same ``(exc_class, source)`` on the same UTC day
     fires at most ONE Telegram alert. The journal still records
     every occurrence; the operator's phone doesn't.
  3. **Surface** — the EOD recap includes a "Today's silenced
     exceptions" section pulling from the same journal rows. The
     dashboard can display the same breakdown.

Usage from inside an ``except`` block::

    try:
        result = self.executor.close_spread(spread)
    except Exception as exc:                           # noqa: BLE001
        self._exception_monitor.record(
            source="agent.close_spread",
            exc=exc,
            ticker=spread.underlying,
        )
        return None

Or for an operationally-significant warning that isn't an
exception (e.g. Schwab token expired)::

    self._exception_monitor.record(
        source="market_data_schwab.fetch_chain",
        message="Schwab refresh token expired — re-auth required",
        ticker=ticker,
    )

Both forms write a ``silenced_exception`` journal row (de-duped
per ``(source, exc_class)`` per UTC day) and page the operator
once-per-day via the Telegram error channel.
"""

from __future__ import annotations

import logging
import os
import json
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class ExceptionMonitor:
    """Operator-visible record of silenced exceptions.

    Constructed with the agent's ``journal_kb`` writer and (optionally)
    its ``telegram`` notifier. When telegram is ``None`` or inactive,
    the monitor still journals but does not page.
    """

    def __init__(self, journal_kb, telegram=None):
        self.journal_kb = journal_kb
        self.telegram = telegram
        # In-memory dedup for the current process. Cross-process
        # dedup is achieved via the journal scan in
        # ``_already_paged_today``; this set is a fast-path that
        # short-circuits the journal read for repeat hits within
        # one process's lifetime.
        self._paged_this_process: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------
    # Public surface — record + journal + (maybe) page
    # ------------------------------------------------------------------

    def record(self, *,
               source: str,
               exc: Optional[BaseException] = None,
               message: Optional[str] = None,
               ticker: str = "") -> None:
        """Record one silenced exception or operational warning.

        ``source``  — short identifier, e.g. ``"agent.close_spread"``.
                       Combined with ``exc_class`` for dedup.
        ``exc``     — the caught exception (preferred; we extract
                       its class name automatically).
        ``message`` — explicit message override. Defaults to
                       ``str(exc)`` when ``exc`` is provided.
        ``ticker``  — optional ticker the failure pertains to.

        Side effects:
          * One ``silenced_exception`` row in the journal (always).
          * One Telegram alert on the error channel IF this is the
            first ``(exc_class, source)`` combo to fire today.
          * No exception propagates — a failure inside the monitor
            (journal write fail, Telegram down) MUST NOT crash the
            caller's except handler. That defeats the entire purpose.
        """
        try:
            exc_class = (
                type(exc).__name__ if exc is not None else "Warning"
            )
            msg = message or (str(exc) if exc is not None else "")
            msg = msg[:300]   # keep journal rows compact

            # Always journal — every occurrence is recorded so the
            # EOD recap + dashboard can show counts.
            self._journal(
                source=source, exc_class=exc_class,
                message=msg, ticker=ticker,
            )

            # Page once per (source, exc_class, UTC day)
            self._maybe_page(
                source=source, exc_class=exc_class,
                message=msg, ticker=ticker,
            )
        except Exception as monitor_exc:                # noqa: BLE001
            # The monitor itself failed. Log it once and continue —
            # do not propagate, do not loop.
            logger.warning(
                "ExceptionMonitor.record raised %s: %s. "
                "Original event source=%s exc_class=%s msg=%s",
                type(monitor_exc).__name__, monitor_exc,
                source, type(exc).__name__ if exc else "?", message,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _journal(self, *, source: str, exc_class: str,
                 message: str, ticker: str) -> None:
        """Write one ``silenced_exception`` row."""
        if not self.journal_kb:
            return
        try:
            self.journal_kb.log_signal(
                ticker=ticker or "__silenced__",
                action="silenced_exception",
                price=0.0,
                raw_signal={
                    "source": source,
                    "exc_class": exc_class,
                    "message": message,
                },
                notes=f"silenced: {exc_class} @ {source}",
            )
        except Exception as exc:                          # noqa: BLE001
            logger.warning(
                "ExceptionMonitor: journal write failed "
                "for source=%s exc_class=%s: %s",
                source, exc_class, exc,
            )

    def _maybe_page(self, *, source: str, exc_class: str,
                    message: str, ticker: str) -> None:
        """Fire the Telegram alert IF this is today's first occurrence."""
        if not self.telegram or not getattr(self.telegram, "is_active", False):
            return
        key = (source, exc_class)
        # Fast path — already paged this process
        if key in self._paged_this_process:
            return
        # Journal path — already paged earlier today by another
        # process instance (cron-launched cycles re-exec; the
        # in-memory cache resets per process).
        if self._already_paged_today(source=source, exc_class=exc_class):
            self._paged_this_process.add(key)
            return
        # Page
        try:
            sent = self.telegram.notify_silenced_exception(
                source=source,
                exc_class=exc_class,
                message=message,
                ticker=ticker,
            )
        except Exception as exc:                          # noqa: BLE001
            logger.warning(
                "ExceptionMonitor: Telegram notify_silenced_exception "
                "raised: %s", exc,
            )
            return
        if sent:
            self._paged_this_process.add(key)
            # Mark in journal so future processes today honour the dedup
            self._mark_paged(source=source, exc_class=exc_class)

    def _already_paged_today(self, *, source: str, exc_class: str) -> bool:
        """Scan the journal for a ``silenced_exception_paged`` row
        matching today's UTC date + (source, exc_class)."""
        jsonl_path = getattr(self.journal_kb, "jsonl_path", None)
        if not jsonl_path or not os.path.isfile(jsonl_path):
            return False
        today_iso = datetime.now(timezone.utc).date().isoformat()
        try:
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("action") != "silenced_exception_paged":
                        continue
                    rs = rec.get("raw_signal") or {}
                    if not isinstance(rs, dict):
                        continue
                    if (rs.get("source") == source
                            and rs.get("exc_class") == exc_class
                            and rs.get("paged_date") == today_iso):
                        return True
        except Exception as exc:                          # noqa: BLE001
            logger.warning(
                "ExceptionMonitor: dedup scan failed: %s — "
                "fail-open (will page extra rather than skip).", exc,
            )
        return False

    def _mark_paged(self, *, source: str, exc_class: str) -> None:
        """Write the dedup marker so other processes today skip."""
        if not self.journal_kb:
            return
        try:
            today_iso = datetime.now(timezone.utc).date().isoformat()
            self.journal_kb.log_signal(
                ticker="__silenced__",
                action="silenced_exception_paged",
                price=0.0,
                raw_signal={
                    "source": source,
                    "exc_class": exc_class,
                    "paged_date": today_iso,
                },
                notes=f"silenced-paged: {exc_class} @ {source}",
            )
        except Exception as exc:                          # noqa: BLE001
            logger.warning(
                "ExceptionMonitor: paged-marker write failed: %s — "
                "next process today may re-page this combo.", exc,
            )


__all__ = ["ExceptionMonitor"]
