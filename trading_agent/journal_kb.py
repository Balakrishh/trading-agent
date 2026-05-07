"""
Journal Knowledge Base
=======================
Structured signal logger for every trade attempt/signal, written
regardless of whether the LLM intelligence layer is enabled.

Two output files maintained in *journal_dir*, basename keyed by
``run_mode``:

  signals_<run_mode>.jsonl  — one JSON object per line (LLM-ready)
  signals_<run_mode>.md     — append-only Markdown table (human-readable)

``run_mode`` is one of ``"live"`` (default — used by the live agent)
or ``"backtest"`` (used by ``backtest_ui._export_to_journal``). Keeping
the streams in separate files prevents backtest signals from polluting
the live RAG corpus or the live diagnostics dashboard, and makes
post-mortem readers obvious about which environment a row came from.

JSONL record schema
-------------------
{
  "timestamp":   ISO-8601 UTC string
  "ticker":      str
  "action":      str   ("dry_run" | "submitted" | "rejected" | "skip" |
                         "error" | "skipped_by_llm" | "skipped_existing")
  "price":       float  (current underlying price at signal time)
  "exec_status": str   (mirrors action or final order status)
  "notes":       str   (brief human-readable summary ≤ 200 chars)
  "raw_signal": {
      "mode":                  str   ("LIVE" | "DRY-RUN") — appended by
                                      agent.py:_log_signal so the dashboard
                                      can scope its guardrail panel to the
                                      currently-selected mode.  Legacy rows
                                      written before this field existed are
                                      treated as LIVE for filtering.
      "regime":                str
      "strategy":              str
      "plan_valid":            bool
      "risk_approved":         bool
      "net_credit":            float
      "max_loss":              float
      "credit_to_width_ratio": float
      "spread_width":          float
      "expiration":            str
      "dte":                   int
      "sma_50":                float
      "sma_200":               float
      "rsi_14":                float
      "account_balance":       float
      "checks_passed":         list[str]
      "checks_failed":         list[str]
      "llm_decision":          str | None
      "llm_confidence":        float | None
      "rejection_reason":      str | None
      "error":                 str | None
  }
}
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from trading_agent.file_locks import locked_append

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rejection-spam dedup (Option A — added 2026-05-06)
# ---------------------------------------------------------------------------
# At a 5-min cycle × ~9 tickers, a $5k account can produce 80–100 redundant
# rejection rows per hour as oversized underlyings (SPY, GLD) trip the
# max_loss > 2% × equity gate every cycle.  This module's dedup
# suppresses identical consecutive entries per ticker, with a periodic
# heartbeat row so the journal isn't completely silent.
#
# Signature = (action, sorted(checks_failed_set), rejection_reason)
# - Match  → increment counter, suppress write unless heartbeat tick.
# - Diff   → write fresh row + reset counter.
# - SUBMITTED / CLOSED actions BYPASS dedup entirely (they're never spam,
#   and operators must always see them at the exact instant they happen).
#
# Configurable: JOURNAL_DEDUP_HEARTBEAT_EVERY env var.
#   - default 12  → ~1 heartbeat per hour at 5-min cycles
#   - 0 disables dedup entirely (fully verbose journal)
#   - 1 also effectively disables (every row is a heartbeat)
_DEFAULT_HEARTBEAT_EVERY = 12

# Actions that ALWAYS write a journal row regardless of dedup state.
# Submissions, closes, close-failures, and dry-run sentinels are
# material events; an operator must see each one at the exact moment
# it happened.  In particular, ``close_failed`` (added 2026-05-06)
# bypasses dedup so successive partial-fill attempts on a zombie
# position remain individually visible — that's the data the operator
# needs to recognise the zombie state and intervene.
_DEDUP_BYPASS_ACTIONS = frozenset({
    "submitted", "dry_run", "closed", "close_failed", "dry_run_close",
    "error",  # errors are rare AND material — dedupe is wrong here
    "warning",  # connectivity / retry-exhausted / OAuth failures —
                # operators must see each occurrence as it happens.
})

_MD_HEADER = (
    "| Timestamp (UTC) | Ticker | Action | Price | Strategy | Regime "
    "| Risk OK | Status | Confidence | Notes |\n"
    "|-----------------|--------|--------|-------|----------|--------"
    "|---------|--------|------------|-------|\n"
)


class JournalKB:
    """
    Append-only signal journal — always active, LLM-independent.

    Usage::
        jkb = JournalKB("journal_kb")
        jkb.log_signal(ticker="AAPL", action="dry_run", price=178.5,
                       raw_signal={...})
    """

    # Allow-list of recognised run-modes — guards against typos that
    # would silently create a new journal file (``signals_lvie.jsonl``…).
    VALID_RUN_MODES = ("live", "backtest")

    def __init__(self, journal_dir: str = "journal_kb",
                 run_mode: str = "live"):
        if run_mode not in self.VALID_RUN_MODES:
            raise ValueError(
                f"run_mode must be one of {self.VALID_RUN_MODES}; got {run_mode!r}"
            )
        self.journal_dir = journal_dir
        self.run_mode    = run_mode
        os.makedirs(journal_dir, exist_ok=True)
        # Path basename keyed by run_mode so live and backtest never
        # interleave bytes inside the same JSONL stream.
        self.jsonl_path = os.path.join(journal_dir, f"signals_{run_mode}.jsonl")
        self.md_path    = os.path.join(journal_dir, f"signals_{run_mode}.md")
        self._ensure_md_header()

        # ── Rejection-spam dedup state (Option A) ────────────────────────
        # In-memory only — not persisted across restarts (intentional: a
        # restart is often the trigger for an operator to re-inspect
        # rejection state, so writing the first post-restart rejection
        # is the right behaviour).
        self._last_signature_by_ticker: Dict[str, Tuple[str, int]] = {}
        try:
            self._heartbeat_every = max(0, int(os.environ.get(
                "JOURNAL_DEDUP_HEARTBEAT_EVERY",
                str(_DEFAULT_HEARTBEAT_EVERY),
            )))
        except (TypeError, ValueError):
            self._heartbeat_every = _DEFAULT_HEARTBEAT_EVERY

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_signal(
        self,
        ticker: str,
        action: str,
        price: float,
        raw_signal: Dict[str, Any],
        exec_status: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        """
        Log one trade signal/attempt.  Called for every ticker every
        cycle regardless of LLM enablement or execution outcome.

        Parameters
        ----------
        ticker      : underlying symbol
        action      : signal disposition (see schema above)
        price       : underlying price at decision time
        raw_signal  : full market + plan + risk context dict
        exec_status : final order status if different from action
        notes       : optional ≤200-char human summary
                      (raised 2026-05-06 from 120 — the rejection-reason
                      strings produced by the chain scanner regularly run
                      150-180 chars and were getting truncated mid-paren
                      in the dashboard's Recent Journal Entries panel,
                      losing the most diagnostic part)
        """
        ts = datetime.now(timezone.utc).isoformat()
        status = exec_status or action

        if notes is None:
            notes = self._auto_notes(raw_signal, action)

        # ── Rejection-spam dedup gate ────────────────────────────────────
        # Compute a per-ticker signature; on consecutive matches, suppress
        # the write entirely OR emit a periodic heartbeat row depending on
        # the cycle counter.  Material events (submitted, closed, errors)
        # bypass the gate so they always appear in the journal at the
        # exact instant they happen.
        is_dedup_skip, dedup_meta = self._dedup_decision(ticker, action, raw_signal)
        if is_dedup_skip:
            return                          # silently suppressed duplicate

        record: Dict[str, Any] = {
            "timestamp": ts,
            "ticker": ticker,
            "action": action,
            "price": round(float(price), 4),
            "exec_status": status,
            "notes": notes[:200],
            "raw_signal": raw_signal,
        }
        if dedup_meta:
            # Annotate heartbeat rows so an operator (and future
            # analytics) can distinguish a heartbeat from a fresh write.
            record["dedup"] = dedup_meta

        self._write_jsonl(record)
        self._write_md_row(ts, ticker, action, price, raw_signal, status, notes)

    # ------------------------------------------------------------------
    # Rejection-spam dedup helper
    # ------------------------------------------------------------------

    def _dedup_decision(self, ticker: str, action: str,
                        raw_signal: Dict[str, Any]) -> Tuple[bool, Optional[Dict]]:
        """
        Decide whether a log_signal call should be suppressed (duplicate)
        or annotated as a heartbeat.

        Returns
        -------
        (is_skip, dedup_meta)
            ``is_skip=True``  → caller must drop the row entirely.
            ``is_skip=False`` → caller writes the row.  ``dedup_meta`` is
            either ``None`` (fresh write — nothing to annotate) or a dict
            ``{"streak": N, "kind": "heartbeat"}`` to tag the row as a
            heartbeat for downstream analytics.

        Bypass conditions
        -----------------
        * ``self._heartbeat_every == 0`` — dedup disabled.
        * ``action`` in :data:`_DEDUP_BYPASS_ACTIONS` — material events
          (submitted, closed, dry_run, error) always pass through.
        """
        # Dedup disabled — write everything.
        if self._heartbeat_every == 0:
            return False, None

        # Material events bypass dedup entirely.
        if action in _DEDUP_BYPASS_ACTIONS:
            # Reset the per-ticker counter so a subsequent rejection
            # (after a real submission/close) writes fresh.
            self._last_signature_by_ticker.pop(ticker, None)
            return False, None

        sig = self._signature(action, raw_signal)
        prev = self._last_signature_by_ticker.get(ticker)

        if prev is None or prev[0] != sig:
            # First occurrence OR signature changed → write fresh, reset count.
            self._last_signature_by_ticker[ticker] = (sig, 1)
            return False, None

        # Identical signature — increment streak.
        prev_sig, prev_count = prev
        new_count = prev_count + 1
        self._last_signature_by_ticker[ticker] = (prev_sig, new_count)

        # Heartbeat if the new count is divisible by the configured
        # interval (e.g. every 12th duplicate at default).
        if new_count % self._heartbeat_every == 0:
            return False, {
                "streak": new_count,
                "kind": "heartbeat",
                "interval": self._heartbeat_every,
            }

        # Otherwise suppress.
        return True, None

    @staticmethod
    def _signature(action: str, raw_signal: Dict[str, Any]) -> str:
        """
        Produce a stable hash for the (action, checks_failed,
        rejection_reason) tuple.  Two calls for the same ticker with
        identical signature → consecutive duplicates.

        Why not a tuple of strings directly?  Python set order is
        non-deterministic, so we sort ``checks_failed`` first to make
        the hash invariant under list-order changes.
        """
        checks_failed = raw_signal.get("checks_failed") or []
        if isinstance(checks_failed, list):
            checks_key = tuple(sorted(str(c) for c in checks_failed))
        else:
            checks_key = (str(checks_failed),)
        rejection_reason = str(raw_signal.get("rejection_reason") or "")
        # Also include the freeform "reason" field that gates like
        # skipped_existing / skipped_bias use, so a transition between
        # two skip-types is captured as a state change.
        reason = str(raw_signal.get("reason") or "")
        material = (action, checks_key, rejection_reason, reason)
        return hashlib.sha256(repr(material).encode("utf-8")).hexdigest()[:16]

    def log_defense_first(
        self,
        ticker: str,
        reason: str,
        price: float,
        extra: Optional[Dict] = None,
    ) -> None:
        """
        Log a capital-retainment skip event (macro guard, high-IV block, etc.).
        Includes strategy_mode: defense_first in the raw_signal for LLM training.
        """
        raw: Dict[str, Any] = {"strategy_mode": "defense_first", "reason": reason}
        if extra:
            raw.update(extra)
        self.log_signal(
            ticker=ticker,
            action="skipped_defense_first",
            price=price,
            raw_signal=raw,
            notes=f"defense_first: {reason[:80]}",
        )

    def log_error(
        self,
        ticker: str,
        error: str,
        price: float = 0.0,
        context: Optional[Dict] = None,
    ) -> None:
        """Log a per-ticker processing failure."""
        self.log_signal(
            ticker=ticker,
            action="error",
            price=price,
            raw_signal={"error": error, **(context or {})},
            exec_status="error",
            notes=error[:200],
        )

    def log_cycle_error(self, error: str, context: Optional[Dict] = None) -> None:
        """Log a full-cycle failure (e.g. account fetch failure, timeout)."""
        ts = datetime.now(timezone.utc).isoformat()
        record: Dict[str, Any] = {
            "timestamp": ts,
            "event": "cycle_error",
            "error": error[:500],
            "context": context or {},
        }
        self._write_jsonl(record)
        logger.error("JournalKB cycle_error: %s", error)

    def log_warning(
        self,
        source: str,
        message: str,
        ticker: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Emit an ``action="warning"`` row to the journal.

        Why this exists
        ---------------
        Pre-2026-05-06 connectivity issues (rate-limits, retry exhaustion,
        Schwab refresh failure surfaced to the operator only via
        ``logs/trading_agent.log`` — invisible to the dashboard
        Recent Journal Entries panel.  An operator who only watches
        the UI could miss real broker issues for hours.

        This helper writes a structured row that:
          * Renders in the Recent Journal Entries panel like any
            other action.
          * Bypasses the rejection-spam dedup gate (warnings are
            material — see ``_DEDUP_BYPASS_ACTIONS``).
          * Carries a ``source`` tag so an operator can grep by
            subsystem (e.g. ``executor``, ``schwab_oauth``,
            ``position_monitor``).

        Parameters
        ----------
        source : a short subsystem tag (e.g. ``"executor"``,
                 ``"schwab_oauth"``, ``"market_data"``).  Surfaced as
                 the ``raw_signal.source`` field.
        message: the human-readable warning copy.  Capped at 500 chars
                 in the journal record (the ``notes`` cell is capped at
                 200 by ``log_signal``; this is the longer ``raw_signal``
                 form for full triage context).
        ticker : optional underlying — when present, the row groups
                 with that ticker in dashboard panels that filter by
                 ticker (Recent Journal Entries does not, so this
                 only matters for future ticker-scoped tools).
        context: arbitrary structured context (HTTP status, attempt
                 count, last error string, etc.).  Surfaced inside
                 ``raw_signal``.
        """
        raw: Dict[str, Any] = {
            "source": source,
            "message": message[:500],
            "context": context or {},
        }
        # Use the same log_signal path so the row carries the full
        # canonical schema (timestamp, ticker, action, exec_status,
        # notes, raw_signal) and the Markdown ledger gets a row too.
        self.log_signal(
            ticker=ticker or "",
            action="warning",
            price=0.0,
            raw_signal=raw,
            exec_status=f"warning_{source}",
            notes=f"[{source}] {message[:180]}",
        )
        logger.warning("JournalKB warning [%s]%s %s",
                       source,
                       f" {ticker}" if ticker else "",
                       message)

    def log_shutdown(self, reason: str, context: Optional[Dict] = None) -> None:
        """
        Log a graceful-shutdown marker to the journal.

        Used by shutdown.graceful_exit() so post-mortem readers can
        correlate gaps in the cycle log with clean exits vs. crashes.
        """
        ts = datetime.now(timezone.utc).isoformat()
        record: Dict[str, Any] = {
            "timestamp": ts,
            "event": "shutdown",
            "reason": reason[:200],
            "context": context or {},
        }
        self._write_jsonl(record)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_md_header(self) -> None:
        """Write the Markdown header if the file doesn't exist yet."""
        if not os.path.exists(self.md_path):
            # Header write is one-shot at init; a locked append is overkill
            # but keeps the concurrency story consistent across this module.
            with locked_append(self.md_path) as fh:
                fh.write("# Trade Signal Journal\n\n")
                fh.write(_MD_HEADER)

    def _write_jsonl(self, record: Dict[str, Any]) -> None:
        """
        Append one JSON record as a single line, under an exclusive lock.

        The lock guarantees concurrent writers (cron cycle + manual run
        + Streamlit backtester) don't interleave bytes inside a line
        and break the JSONL parser.
        """
        try:
            with locked_append(self.jsonl_path) as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.error("JournalKB JSONL write failed: %s", exc)

    def _write_md_row(
        self,
        ts: str,
        ticker: str,
        action: str,
        price: float,
        raw: Dict[str, Any],
        status: str,
        notes: str,
    ) -> None:
        try:
            strategy = raw.get("strategy") or "—"
            regime = raw.get("regime") or "—"
            risk_ok = raw.get("risk_approved", "—")
            conf = raw.get("llm_confidence")
            conf_str = f"{conf:.2f}" if isinstance(conf, float) else "—"
            safe_notes = str(notes).replace("|", "\\|")[:80]
            row = (
                f"| {ts[:19]} | {ticker} | {action} | ${price:.2f} "
                f"| {strategy} | {regime} | {risk_ok} | {status} "
                f"| {conf_str} | {safe_notes} |\n"
            )
            with locked_append(self.md_path) as fh:
                fh.write(row)
        except Exception as exc:
            logger.error("JournalKB Markdown write failed: %s", exc)

    @staticmethod
    def _auto_notes(raw: Dict[str, Any], action: str) -> str:
        """Generate a concise summary string from raw signal data."""
        parts = []
        if raw.get("strategy"):
            parts.append(raw["strategy"])
        if raw.get("net_credit"):
            parts.append(f"cr={raw['net_credit']:.2f}")
        ratio = raw.get("credit_to_width_ratio")
        if ratio:
            parts.append(f"ratio={ratio:.2f}")
        if raw.get("rejection_reason"):
            parts.append(raw["rejection_reason"][:60])
        if raw.get("error"):
            parts.append(raw["error"][:60])
        return (f"{action}: " + ", ".join(parts)) if parts else action
