"""close_event_collaborators.py — extracted collaborators for the close path.

Skill 35 — decoupling step #3. Pre-2026-05-22 the ~300-line
``agent._journal_close_event`` method combined:

  * close-row construction (action / notes / payload)
  * cooldown bookkeeping (journal-derived partial-fill streak)
  * reactive PDT detection (Alpaca 40310100 → block marker)
  * three Telegram alert dispatches (cooldown engaged, PDT block, close)
  * journal write

That one method's test fixture needed ``MagicMock(spec=TradingAgent)``
because every collaborator was a private method on the agent. Each
extraction here is its own class with explicit dependencies — testable
in isolation without the agent fixture.

Module surface (all four classes are constructor-injected into
``TradingAgent.__init__``):

  * ``PartialFillCooldown``  — journal-derived streak + cooldown math
  * ``PdtBlockDetector``     — leg-error scan + cross-day block-set
  * ``CloseAlertNotifier``   — three close-event Telegram dispatches
  * ``CloseJournalWriter``   — orchestrates the four above + writes row

``agent._journal_close_event`` shrinks to a one-line delegation:

  self._close_writer.write(spread, ctx, leg_results=..., fill_status=...,
                           dry_run=...)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PartialFillCooldown — journal-derived streak counter + cooldown deadline
# ---------------------------------------------------------------------------


class PartialFillCooldown:
    """Tracks consecutive partial-fill closes via the journal.

    Pre-2026-05-13 streak state was an in-memory dict on TradingAgent;
    every cycle restart blew it away so the cooldown never engaged.
    State is now derived from journal rows on every read — the journal
    persists across cycles by construction.

    Methods are pure functions of the journal + a clock; testable
    end-to-end with a fixture jsonl in tempdir.
    """

    def __init__(self, *, journal_kb, threshold: int, window_min: int):
        """journal_kb only needed for jsonl_path. threshold = streak length
        that engages cooldown. window_min = how far back to count."""
        self.journal_kb = journal_kb
        self.threshold = threshold
        self.window_min = window_min

    # -- core derivation ----------------------------------------------------

    def streak_within_window(
        self, ticker: str,
    ) -> Tuple[int, Optional[datetime]]:
        """Count consecutive ``close_failed`` rows for ticker, in window,
        since the most recent ``closed``. Returns (streak, last_fail_ts).

        Verbatim port of agent._close_failed_streak_within_window —
        same semantics, same fail-open behavior on journal read error."""
        jsonl_path = getattr(self.journal_kb, "jsonl_path", None)
        if not jsonl_path or not os.path.isfile(jsonl_path):
            return 0, None

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=self.window_min)
        events: List[Tuple[datetime, str]] = []
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
                    if rec.get("ticker") != ticker:
                        continue
                    if rec.get("action") not in ("close_failed", "closed"):
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
                    if ts < cutoff:
                        continue
                    events.append((ts, rec.get("action", "")))
        except Exception as exc:                                  # noqa: BLE001, skill-34-exempt — collaborator wrap (skill 35: never propagate)
            logger.warning(
                "Could not read journal for cooldown derivation: %s", exc,
            )
            return 0, None

        events.sort()
        # Find the most recent 'closed' — anything before it doesn't count.
        last_closed_ts: Optional[datetime] = None
        for ts, action in events:
            if action == "closed":
                last_closed_ts = ts

        streak = 0
        last_fail_ts: Optional[datetime] = None
        for ts, action in events:
            if action != "close_failed":
                continue
            if last_closed_ts is not None and ts <= last_closed_ts:
                continue
            streak += 1
            if last_fail_ts is None or ts > last_fail_ts:
                last_fail_ts = ts
        return streak, last_fail_ts

    def minutes_remaining(self, ticker: str) -> int:
        """Cooldown minutes remaining, or 0 if no cooldown active.

        Verbatim port of agent._close_cooldown_minutes_remaining."""
        streak, last_fail_ts = self.streak_within_window(ticker)
        if streak < self.threshold or last_fail_ts is None:
            return 0
        deadline = last_fail_ts + timedelta(minutes=self.window_min)
        now = datetime.now(timezone.utc)
        if now >= deadline:
            return 0
        delta = deadline - now
        return max(
            1,
            int(delta.total_seconds() // 60)
            + (1 if delta.total_seconds() % 60 else 0),
        )

    # -- caller-side hooks (no-op shims kept for log clarity) --------------

    def log_partial_close(self, ticker: str) -> None:
        """Emit a streak-progress log line. State derivation happens at
        read time so there's nothing to record in-memory."""
        streak, _ = self.streak_within_window(ticker)
        upcoming = streak + 1  # +1 for the row about to be written
        if upcoming >= self.threshold:
            logger.warning(
                "[%s] %d consecutive partial closes — entering "
                "%d-min cooldown.  Manually clean up the position on "
                "Alpaca's UI to clear the zombie state; the cooldown "
                "will expire automatically thereafter.",
                ticker, upcoming, self.window_min,
            )
        else:
            logger.info(
                "[%s] Partial-close streak %d/%d.  Will park in "
                "cooldown after %d.",
                ticker, upcoming, self.threshold, self.threshold,
            )

    def log_close_success(self, ticker: str) -> None:
        """Emit a streak-cleared log line if a streak was active."""
        streak, _ = self.streak_within_window(ticker)
        if streak > 0:
            logger.info(
                "[%s] Successful close clears journal-derived "
                "cooldown streak (was %d/%d).",
                ticker, streak, self.threshold,
            )

    # -- payload helper for the writer --------------------------------------

    def build_payload_fields(
        self, ticker: str,
    ) -> Tuple[int, Dict[str, Any]]:
        """For close_failed rows: return (streak_with_pending, fields).

        ``streak_with_pending`` is the existing streak + 1 (the row the
        caller is about to write). ``fields`` is the dict to merge
        into the close-event payload — includes partial_close_streak,
        partial_close_threshold, and (when cooldown engages) the
        deadline + reason.
        """
        existing_streak, _ = self.streak_within_window(ticker)
        streak = existing_streak + 1
        fields: Dict[str, Any] = {
            "partial_close_streak": streak,
            "partial_close_threshold": self.threshold,
        }
        if streak >= self.threshold:
            deadline = (
                datetime.now(timezone.utc)
                + timedelta(minutes=self.window_min)
            )
            fields["close_cooldown_until"] = deadline.isoformat()
            fields["close_cooldown_reason"] = (
                f"{streak} consecutive partial fills "
                f"≥ threshold {self.threshold}; "
                f"auto-close suppressed for "
                f"{self.window_min} min — manual broker "
                f"intervention required to clear zombie state."
            )
        return streak, fields


# ---------------------------------------------------------------------------
# PdtBlockDetector — reactive PDT detection from Alpaca leg-error responses
# ---------------------------------------------------------------------------


class PdtBlockDetector:
    """Detects Alpaca's pattern-day-trading rejection on leg results
    and surfaces journal markers for the close-loop suppression.

    Skill 17 §4 — reactive PDT suppression. The marker is written only
    after Alpaca *actually* responds with code 40310100, never
    speculatively. The cross-day reader (``blocked_tickers_today``)
    is consumed by the close loop to short-circuit retries that are
    doomed for the rest of the UTC day.
    """

    # Strings observed in Alpaca's HTTP response body when the broker
    # blocks an attempt under PDT rules. Pre-2026-05-13 we matched only
    # the code; the phrase catches API responses where the structured
    # code field is missing but the human message is present.
    PDT_SIGNALS: Tuple[str, ...] = ("40310100", "pattern day trading")

    def __init__(self, *, journal_kb):
        self.journal_kb = journal_kb

    @classmethod
    def detect(cls, leg_results: List[Dict]) -> bool:
        """True if any leg's error string carries a PDT signature."""
        return any(
            isinstance(leg, dict)
            and any(
                s in str(leg.get("error", "")).lower()
                for s in cls.PDT_SIGNALS
            )
            for leg in (leg_results or [])
        )

    @staticmethod
    def build_markers() -> Dict[str, Any]:
        """Returns the pdt_blocked_today/_date/_reason payload fields."""
        today_utc = datetime.now(timezone.utc).date().isoformat()
        return {
            "pdt_blocked_today": True,
            "pdt_blocked_date": today_utc,
            "pdt_blocked_reason": (
                "Alpaca returned code 40310100 (pattern day "
                "trading protection) on one or more legs. "
                "Subsequent close attempts on this ticker will "
                "be suppressed until UTC midnight, when the "
                "position is no longer same-day-open."
            ),
        }

    def blocked_tickers_today(self) -> Set[str]:
        """Cross-cycle reader — verbatim port of
        agent._pdt_blocked_today_tickers."""
        try:
            jsonl_path = getattr(self.journal_kb, "jsonl_path", None)
            if not jsonl_path or not os.path.isfile(jsonl_path):
                return set()
            today_iso = datetime.now(timezone.utc).date().isoformat()
            blocked: Set[str] = set()
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("action") != "close_failed":
                        continue
                    rs = rec.get("raw_signal") or {}
                    if not rs.get("pdt_blocked_today"):
                        continue
                    if rs.get("pdt_blocked_date") != today_iso:
                        continue
                    tk = rec.get("ticker")
                    if tk:
                        blocked.add(tk)
            return blocked
        except Exception as exc:                                  # noqa: BLE001, skill-34-exempt — collaborator wrap (skill 35: never propagate)
            logger.warning(
                "Failed to derive PDT-blocked tickers from journal: %s",
                exc,
            )
            return set()


# ---------------------------------------------------------------------------
# CloseAlertNotifier — three Telegram dispatches for close-event side effects
# ---------------------------------------------------------------------------


class CloseAlertNotifier:
    """Typed wrappers around ``_send_telegram_alert`` for every
    operator alert the agent dispatches.

    Historical note: this class was introduced (skill 35) wrapping
    only the three close-event alerts (cooldown engaged / PDT block /
    position closed). In 2026-05-22 (item 6 of the standards
    roadmap) the wrapper surface was extended to cover all six
    alert sites — `roll_open_failed`, EOD summary, and
    `position_opened` — so the agent has ZERO remaining callers of
    the ``_send_telegram_alert(send_fn=...)`` indirection. That
    indirection is what produced the 48-hour ticker-drop TypeError
    bug: ``send_fn(**call_kwargs)`` couldn't validate kwargs
    against the callee signature.

    Each ``notify_*`` method below has a typed signature matching
    the corresponding ``telegram.notify_*`` callee. Mismatches now
    fail at agent construction or first call, not silently for 48
    hours.

    Delegates the actual send + per-day dedup to the agent's existing
    ``_send_telegram_alert`` helper so behavior is identical to pre-
    refactor. Only the call sites move.
    """

    def __init__(self, *, send_alert: Callable, telegram):
        """send_alert: bound method on agent — currently
        ``self._send_telegram_alert``. Wrapping it lets the writer
        dispatch without re-implementing the dedup gate.

        telegram: agent.telegram — needed only to check ``is_active``
        on the position-closed path so we skip the call entirely
        when notifications are disabled.
        """
        self._send_alert = send_alert
        self._telegram = telegram

    def notify_cooldown_engaged(self, *, spread, ctx: Dict, streak: int,
                                threshold: int, deadline_iso: str,
                                failed_legs: str) -> None:
        """Fires the close_cooldown Telegram alert. Dedup'd by
        ``alert_type="close_cooldown"`` per ticker per UTC day."""
        self._send_alert(
            ticker=spread.underlying,
            alert_type="close_cooldown",
            send_fn=self._telegram.notify_close_cooldown,
            strategy=ctx.get("strategy", spread.strategy_name),
            streak=streak,
            threshold=threshold,
            cooldown_until_iso=deadline_iso,
            failed_legs=failed_legs,
        )

    def notify_pdt_block(self, *, spread, ctx: Dict) -> None:
        """Fires the pdt_block Telegram alert. Dedup'd by
        ``alert_type="pdt_block"`` per ticker per UTC day so DIA's
        18 daily detections collapse to 1 message."""
        self._send_alert(
            ticker=spread.underlying,
            alert_type="pdt_block",
            send_fn=self._telegram.notify_pdt_block,
            strategy=ctx.get("strategy", spread.strategy_name),
            exit_signal=ctx.get(
                "exit_signal", spread.exit_signal.value,
            ),
            exit_reason=ctx.get(
                "exit_reason", spread.exit_reason,
            ),
            account_balance=float(
                getattr(spread, "account_balance", 0.0) or 0.0
            ),
        )

    def notify_position_closed(self, *, spread, ctx: Dict) -> None:
        """Fires the position_closed Telegram alert on a fully filled
        close. Dedup'd by ``ticker:expiration:exit_signal:UTC date`` so
        a legitimate same-day re-trade (different expiration) still
        alerts, but the same stuck position can't spam."""
        if not self._telegram.is_active:
            return
        exit_sig_val = ctx.get("exit_signal", spread.exit_signal.value)
        exp_val = ctx.get("expiration", spread.expiration or "")
        dedup_alert_type = (
            f"position_closed:{exp_val}:{exit_sig_val}"
        )
        self._send_alert(
            ticker=spread.underlying,
            alert_type=dedup_alert_type,
            send_fn=self._telegram.notify_position_closed,
            strategy=ctx.get("strategy", spread.strategy_name),
            exit_signal=exit_sig_val,
            exit_reason=ctx.get(
                "exit_reason", spread.exit_reason or "",
            ),
            realized_pl=float(
                ctx.get("net_unrealized_pl", 0.0) or 0.0
            ),
            original_credit=float(
                ctx.get("original_credit", 0.0) or 0.0
            ),
            max_loss=float(ctx.get("max_loss", 0.0) or 0.0),
        )

    # ──────────────────────────────────────────────────────────────────
    # Non-close-event alerts (item 6, 2026-05-22)
    # ──────────────────────────────────────────────────────────────────

    def notify_position_opened(self, *, ticker: str, strategy: str,
                               regime: str, net_credit: float,
                               max_loss: float, spread_width: float,
                               expiration: str, short_strikes: str,
                               thesis: str) -> None:
        """Fires the position_opened Telegram alert on a successful
        broker submission (action='submitted'). Dedup'd by
        ``ticker:expiration`` per UTC day so re-submissions on the
        same ticker+expiration don't re-alert."""
        if not self._telegram.is_active:
            return
        dedup_alert_type = f"position_opened:{expiration}"
        self._send_alert(
            ticker=ticker,
            alert_type=dedup_alert_type,
            send_fn=self._telegram.notify_position_opened,
            strategy=strategy,
            regime=regime,
            net_credit=net_credit,
            max_loss=max_loss,
            spread_width=spread_width,
            expiration=expiration,
            short_strikes=short_strikes,
            thesis=thesis,
        )

    def notify_roll_open_failed(self, *, ticker: str, strategy: str,
                                reason: str) -> None:
        """Fires the roll_open_failed Telegram alert when a defensive
        roll closes successfully but the replacement open crashes.
        Operationally critical — position is FLAT, no replacement."""
        self._send_alert(
            ticker=ticker,
            alert_type="roll_open_failed",
            send_fn=self._telegram.notify_open_failed_after_close,
            strategy=strategy,
            reason=reason,
        )

    def notify_eod_summary(self, *, alert_type: str, date_label: str,
                           account_balance: float,
                           starting_balance,
                           opens_today, closes_today,
                           realized_pl_today: float,
                           unrealized_pl_today: float,
                           cycles_today: int, errors_today: int,
                           stuck_tickers, silenced_exceptions) -> None:
        """Fires the EOD recap Telegram alert. ``alert_type`` carries
        the ET trading-session date for cross-day dedup."""
        self._send_alert(
            ticker="__eod__",
            alert_type=alert_type,
            send_fn=self._telegram.notify_eod_summary,
            date_label=date_label,
            account_balance=account_balance,
            starting_balance=starting_balance,
            opens_today=opens_today,
            closes_today=closes_today,
            realized_pl_today=realized_pl_today,
            unrealized_pl_today=unrealized_pl_today,
            cycles_today=cycles_today,
            errors_today=errors_today,
            stuck_tickers=stuck_tickers,
            silenced_exceptions=silenced_exceptions,
        )


# ---------------------------------------------------------------------------
# CloseJournalWriter — top-level orchestrator
# ---------------------------------------------------------------------------


class CloseJournalWriter:
    """Composes the close-event row, writes it to the journal, fires
    the three close-event Telegram alerts.

    Replaces the 300-line ``agent._journal_close_event`` method.
    Constructor-injected collaborators mean the writer is testable
    end-to-end with stub journal + stub notifier — no MagicMock(spec=
    TradingAgent) needed."""

    def __init__(self, *, journal_kb, cooldown: PartialFillCooldown,
                 pdt_detector: PdtBlockDetector,
                 alerts: CloseAlertNotifier,
                 price_lookup: Callable[[str], float]):
        self.journal_kb = journal_kb
        self.cooldown = cooldown
        self.pdt_detector = pdt_detector
        self.alerts = alerts
        self._price_lookup = price_lookup

    def write(self, spread, ctx: Dict, *, leg_results: List[Dict],
              fill_status: str, dry_run: bool) -> None:
        """Emit a structured close-attempt row + dispatch side-effect
        alerts. Never propagates exceptions — a journal failure must
        not break the cycle (the close itself already happened).

        Skill 19 §2 — three distinct close actions:
          * "closed"        → real broker close, all legs filled
          * "dry_run_close" → synthetic close in dry-run mode
          * "close_failed"  → broker rejected one or more legs
        """
        try:
            self._write_impl(spread, ctx, leg_results=leg_results,
                             fill_status=fill_status, dry_run=dry_run)
        except Exception as exc:                                  # noqa: BLE001, skill-34-exempt — collaborator wrap (skill 35: never propagate)
            logger.warning(
                "[%s] Failed to journal close event: %s",
                getattr(spread, "underlying", "?"), exc,
            )

    def _write_impl(self, spread, ctx: Dict, *, leg_results: List[Dict],
                    fill_status: str, dry_run: bool) -> None:
        pl = ctx["net_unrealized_pl"]
        sign = "+" if pl >= 0 else ""

        # Action + notes branch on fill_status
        if fill_status == "complete":
            action = "closed"
            note = (
                f"closed: {ctx['strategy']}, P&L={sign}${pl:.2f}, "
                f"{ctx['exit_signal']}"
            )
            exec_status = f"closed_{ctx['exit_signal']}"
        elif fill_status == "dry_run":
            action = "dry_run_close"
            note = (
                f"dry_run_close: {ctx['strategy']}, "
                f"would-be P&L={sign}${pl:.2f}, "
                f"{ctx['exit_signal']}"
            )
            exec_status = f"dry_run_close_{ctx['exit_signal']}"
        else:
            action = "close_failed"
            failed_legs = [
                leg.get("symbol", "?")
                for leg in (leg_results or [])
                if isinstance(leg, dict)
                   and leg.get("status") != "closed"
            ]
            failed_str = (
                f" failed_legs={','.join(failed_legs)}"
                if failed_legs else ""
            )
            note = (
                f"close_failed: {ctx['strategy']}, "
                f"{ctx['exit_signal']}, fill_status={fill_status}"
                f"{failed_str}"
            )
            exec_status = f"close_failed_{ctx['exit_signal']}"

        payload: Dict[str, Any] = dict(ctx)
        payload["leg_close_results"] = [
            {"symbol": leg.get("symbol", ""),
             "status": leg.get("status", "unknown")}
            for leg in (leg_results or [])
            if isinstance(leg, dict)
        ]
        payload["fill_status"] = fill_status
        # Mode tag must match _log_signal's casing so the dashboard's
        # current_mode filter keeps close rows in the same view as
        # their corresponding submitted rows.
        payload["mode"] = "DRY-RUN" if dry_run else "LIVE"

        # ── Cooldown surface (close_failed only) ──────────────────
        if action == "close_failed":
            streak, cooldown_fields = (
                self.cooldown.build_payload_fields(spread.underlying)
            )
            payload.update(cooldown_fields)

            if streak >= self.cooldown.threshold:
                # First-engagement Telegram alert
                failed_legs_str = (
                    ",".join(
                        (leg.get("symbol") or "?")
                        for leg in (leg_results or [])
                        if isinstance(leg, dict)
                           and leg.get("status") != "closed"
                    ) or "—"
                )
                self.alerts.notify_cooldown_engaged(
                    spread=spread, ctx=ctx, streak=streak,
                    threshold=self.cooldown.threshold,
                    deadline_iso=cooldown_fields["close_cooldown_until"],
                    failed_legs=failed_legs_str,
                )

            # ── Reactive PDT detection ────────────────────────────
            if self.pdt_detector.detect(leg_results):
                payload.update(self.pdt_detector.build_markers())
                logger.critical(
                    "[%s] PDT block detected from Alpaca response — "
                    "marking ticker pdt_blocked_today=%s. Further "
                    "auto-closes will be suppressed for the rest of "
                    "the trading day. Manual close via Alpaca UI is "
                    "still possible if you accept the day-trade flag.",
                    spread.underlying, payload["pdt_blocked_date"],
                )
                self.alerts.notify_pdt_block(spread=spread, ctx=ctx)

        # ── Persist the row ───────────────────────────────────────
        self.journal_kb.log_signal(
            ticker=spread.underlying,
            action=action,
            price=self._price_lookup(spread.underlying),
            exec_status=exec_status,
            notes=note,
            raw_signal=payload,
        )

        # ── Position-closed alert (action="closed" only) ──────────
        if action == "closed":
            self.alerts.notify_position_closed(spread=spread, ctx=ctx)


__all__ = [
    "PartialFillCooldown",
    "PdtBlockDetector",
    "CloseAlertNotifier",
    "CloseJournalWriter",
]
