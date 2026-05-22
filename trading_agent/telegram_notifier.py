"""telegram_notifier.py — opt-in Telegram alerter for operator-actionable events.

The notifier is a *thin, defensive* wrapper around Telegram's Bot API
``sendMessage`` endpoint. Three properties matter most:

  1. **Opt-in via env.** When ``TELEGRAM_BOT_TOKEN`` and
     ``TELEGRAM_CHAT_ID`` are both set, the notifier is active.
     Either missing → the notifier is a silent no-op. This keeps
     existing installs (no env, no token) unchanged.

  2. **Never crashes a cycle.** Every network call is wrapped in a
     try/except + bounded timeout. A failed alert logs a single
     warning and returns — the cycle continues. Alpaca-side actions
     are the source of truth; the alerter is just a paging mechanism.

  3. **Journal-deduped.** The agent passes a ``dedup_key`` derived
     from ``(ticker, alert_type, UTC date)``. The notifier writes a
     ``telegram_alert_sent`` row into the journal on success; before
     re-sending, the agent reads back the journal for matching rows
     and short-circuits if a same-day alert was already delivered.
     Prevents pinging the operator 78 times per day for the same
     stuck DIA position.

Why a separate module and not inline in agent.py
------------------------------------------------
The agent has 11 ``journal_kb.log_signal`` call sites and only some
of them are operator-actionable (close cooldowns, PDT blocks,
zombie positions). Routing every journal write through Telegram
would flood the operator. A separate module with explicit ``notify_*``
methods keeps the alert vocabulary discoverable and reviewable in
one file rather than scattered across the agent.

Skill 32 documents the full contract surface.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Bounded so a slow Telegram API can't tank a 5-minute cycle.
_TELEGRAM_TIMEOUT_SEC = 5

# Public API base — uses the user's bot token from env.
_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """Opt-in Telegram alerter for stuck-position / manual-intervention events.

    Two channels, both env-gated, with fallback:

      * **info channel** — `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`.
        Carries lifecycle events (position open, position close,
        end-of-day summary). The "quiet" channel.

      * **error channel** — `TELEGRAM_ERROR_BOT_TOKEN` /
        `TELEGRAM_ERROR_CHAT_ID`. Carries operator-actionable alerts
        (PDT block, close cooldown engaged, FLAT after close-then-
        open failure). The "act now" channel.

    Fallback rule: if either error env var is unset, the error
    channel reuses the info channel's credentials. This keeps the
    single-bot deployment unchanged for users who haven't created
    a second bot. Set both error env vars to route errors to a
    distinct bot; leave them blank to keep everything on the info
    bot.

    The notifier is stateless beyond the env config — dedup happens
    upstream (in agent.py) via the journal. This module just sends
    the message.
    """

    def __init__(self,
                 token: Optional[str] = None,
                 chat_id: Optional[str] = None,
                 error_token: Optional[str] = None,
                 error_chat_id: Optional[str] = None):
        """Args default to env-var lookup; pass explicit values in tests.

        Error-channel credentials fall back to the info channel when
        not supplied (preserves single-bot behavior for existing
        deployments).
        """
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        # Error-channel credentials. Explicit override → env var →
        # fallback to info channel. Each component falls back
        # independently so partial config (one var set, one blank)
        # still works.
        _env_err_token = os.environ.get("TELEGRAM_ERROR_BOT_TOKEN", "").strip()
        _env_err_chat  = os.environ.get("TELEGRAM_ERROR_CHAT_ID", "").strip()
        self.error_token = error_token or _env_err_token or self.token
        self.error_chat_id = (
            error_chat_id or _env_err_chat or self.chat_id
        )

    @property
    def is_active(self) -> bool:
        """True if AT LEAST ONE channel has both credentials populated.

        The agent's ``_send_telegram_alert`` gates on this — when
        active, individual method calls choose their channel and may
        still no-op if THAT channel is empty (rare partial-config
        edge case)."""
        return bool(
            (self.token and self.chat_id)
            or (self.error_token and self.error_chat_id)
        )

    @property
    def error_channel_distinct(self) -> bool:
        """True iff the error channel uses different credentials from
        the info channel — useful for the dashboard / SDD diagnostics
        to confirm the two-bot setup is actually in effect."""
        return (
            (self.error_token != self.token)
            or (self.error_chat_id != self.chat_id)
        )

    # ------------------------------------------------------------------
    # Internal send (single network call, defensive)
    # ------------------------------------------------------------------

    def _send(self, text: str, *, channel: str = "info") -> bool:
        """POST one message. Returns True on HTTP 200, False otherwise.

        ``channel`` is ``"info"`` (default) or ``"error"`` — routes
        to the matching token/chat_id pair. If the chosen channel's
        credentials aren't populated, returns False without raising
        (the caller treats False as "alert not delivered").

        Any exception is caught — the agent never sees a notifier
        crash. The single WARNING log line lets an operator notice
        consistent delivery failures without flooding the log.
        """
        if channel == "error":
            token, chat_id = self.error_token, self.error_chat_id
        else:
            token, chat_id = self.token, self.chat_id
        if not (token and chat_id):
            return False
        url = _TELEGRAM_API.format(token=token)
        # ``parse_mode=HTML`` so the agent can use <b>, <code>, <pre>
        # for formatting. Telegram's MarkdownV2 has too many escape
        # rules to be safe across arbitrary error strings.
        payload = {
            "chat_id":             chat_id,
            "text":                text,
            "parse_mode":          "HTML",
            "disable_web_page_preview": True,
        }
        try:
            resp = requests.post(
                url, json=payload, timeout=_TELEGRAM_TIMEOUT_SEC,
            )
            if resp.status_code == 200:
                return True
            logger.warning(
                "Telegram sendMessage non-200 (%s, channel=%s): %s",
                resp.status_code, channel, resp.text[:200],
            )
            return False
        except Exception as exc:                                  # noqa: BLE001, skill-34-exempt — Telegram cannot page about itself
            # Bare Exception: network, JSON, timeout, DNS, anything.
            # A flaky Telegram API must not cascade into a cycle abort.
            logger.warning(
                "Telegram send failed (channel=%s): %s", channel, exc,
            )
            return False

    # ------------------------------------------------------------------
    # Operator-actionable alert types
    # ------------------------------------------------------------------
    # Each public ``notify_*`` returns True on successful delivery, False
    # otherwise. The agent uses the return value to decide whether to
    # write a ``telegram_alert_sent`` journal row (which gates future
    # same-day re-sends — see skill 32 §3.4).

    # ── ERROR-class alerts (route to error channel) ─────────────────
    # These wake the operator. Their content is "something needs your
    # attention NOW or before tomorrow's open." Telegram bot:
    # TELEGRAM_ERROR_BOT_TOKEN / TELEGRAM_ERROR_CHAT_ID, fallback to
    # the info bot if those env vars aren't set.

    def notify_pdt_block(self, ticker: str, strategy: str,
                         exit_signal: str, exit_reason: str,
                         account_balance: float) -> bool:
        """Alpaca returned code 40310100 — position is stuck until tomorrow."""
        text = (
            f"🚨 <b>PDT block on {ticker}</b>\n"
            f"Alpaca denied close with code <code>40310100</code> "
            f"(pattern day trading protection).\n\n"
            f"<b>Strategy:</b> {strategy}\n"
            f"<b>Exit signal:</b> {exit_signal}\n"
            f"<b>Reason:</b> {exit_reason}\n"
            f"<b>Equity:</b> ${account_balance:,.2f} (sub-$25K = PDT-restricted)\n\n"
            f"<b>What to do</b>\n"
            f"• Auto-close is suppressed for {ticker} until UTC midnight.\n"
            f"• Position is at risk of assignment if held to expiry ITM.\n"
            f"• To intervene now: close manually in Alpaca's UI (accepts "
            f"a day-trade flag toward your 4-trade weekly limit).\n"
            f"• Otherwise the next session's first cycle will close cleanly."
        )
        return self._send(text, channel="error")

    def notify_close_cooldown(self, ticker: str, strategy: str,
                              streak: int, threshold: int,
                              cooldown_until_iso: str,
                              failed_legs: str) -> bool:
        """Partial-fill streak hit threshold — manual broker action needed."""
        text = (
            f"⚠️ <b>Close cooldown engaged on {ticker}</b>\n"
            f"<b>Strategy:</b> {strategy}\n"
            f"<b>Streak:</b> {streak}/{threshold} consecutive partial fills\n"
            f"<b>Auto-close suppressed until:</b> <code>{cooldown_until_iso}</code>\n"
            f"<b>Failed legs:</b> <code>{failed_legs}</code>\n\n"
            f"<b>What to do</b>\n"
            f"• Open Alpaca UI → Positions → {ticker} → Close manually.\n"
            f"• Or wait for the cooldown to expire and retry will resume.\n"
            f"• If the position is approaching ITM, prioritise manual close — "
            f"the cooldown won't help if the strike is breached."
        )
        return self._send(text, channel="error")

    def notify_open_failed_after_close(self, ticker: str,
                                       strategy: str,
                                       reason: str) -> bool:
        """Defensive-roll: close filled but the replacement open failed."""
        text = (
            f"🆘 <b>FLAT — open failed after close on {ticker}</b>\n"
            f"Defensive roll closed the threatened spread but the "
            f"replacement order failed to submit. The position is "
            f"<b>flat</b> (no exposure), not doubled.\n\n"
            f"<b>Strategy:</b> {strategy}\n"
            f"<b>Open failure reason:</b> <code>{reason[:300]}</code>\n\n"
            f"<b>What to do</b>\n"
            f"• Check Alpaca UI: confirm {ticker} has no legs.\n"
            f"• Re-open manually if desired, or let the next cycle "
            f"plan a fresh entry.\n"
            f"• If you suspect a duplicate-submit bug, freeze the agent "
            f"(⏸ Pause) before investigating to avoid races."
        )
        return self._send(text, channel="error")

    def notify_silenced_exception(self, *, source: str, exc_class: str,
                                  message: str, ticker: str = "") -> bool:
        """Skill 34 — page on the first silent failure per (source,
        exc_class) per UTC day. Fires from ``ExceptionMonitor.record``
        so an operator finds out about a quietly-swallowed exception
        before symptoms surface elsewhere.

        Routes to the error channel — same priority as PDT block /
        close cooldown / FLAT-after-roll. Operator gets one paging
        message per distinct failure mode per day, no spam."""
        ticker_line = f"<b>Ticker:</b> {ticker}\n" if ticker else ""
        text = (
            f"⚠️ <b>Silenced exception: {exc_class}</b>\n"
            f"<b>Source:</b> <code>{source}</code>\n"
            f"{ticker_line}"
            f"<b>Message:</b> <code>{message[:200]}</code>\n\n"
            f"<b>What to do</b>\n"
            f"• Check <code>logs/trading_agent.log</code> for the "
            f"full traceback around the most recent occurrence.\n"
            f"• Look in <code>signals_live.jsonl</code> for "
            f"<code>action=\"silenced_exception\"</code> rows to see "
            f"every occurrence today.\n"
            f"• This is the first {exc_class} from {source} today; "
            f"subsequent occurrences are journalled but not re-paged."
        )
        return self._send(text, channel="error")

    # ── INFO-class alerts (route to info channel) ───────────────────
    # These are passive notifications — what the bot did and how it
    # turned out. Default Telegram bot: TELEGRAM_BOT_TOKEN /
    # TELEGRAM_CHAT_ID. Operators don't need to act on these
    # immediately; they're for awareness + EOD recap.
    #
    # Position lifecycle (skill 32 §3.6): fire once per real position
    # event (open submission, close completion). Each event is
    # uniquely identifiable in the journal (run_id on open,
    # timestamp+exit_signal on close), so no per-day dedup is needed
    # — the agent calls these inline with the journal write that
    # records the event, so the same event can't fire twice within
    # one cycle.

    def notify_position_opened(self, ticker: str, strategy: str,
                               regime: str, net_credit: float,
                               max_loss: float, spread_width: float,
                               expiration: str, short_strikes: str,
                               thesis: str) -> bool:
        """A new spread was just submitted to Alpaca. Tells the operator
        what was opened, why, and the immediate risk profile."""
        cw_ratio = (net_credit / spread_width) if spread_width else 0.0
        text = (
            f"✅ <b>{ticker} {strategy} OPENED</b>\n"
            f"<b>Regime:</b> {regime}\n"
            f"<b>Credit:</b> ${net_credit:.2f}  ·  "
            f"<b>Max loss:</b> ${max_loss:.2f}  ·  "
            f"<b>C/W:</b> {cw_ratio:.0%}\n"
            f"<b>Short strikes:</b> <code>{short_strikes}</code>\n"
            f"<b>Expiration:</b> {expiration}\n\n"
            f"<b>Why</b>\n"
            f"<i>{thesis[:400]}</i>"
        )
        return self._send(text, channel="info")

    def notify_eod_summary(self, *,
                           date_label: str,
                           account_balance: float,
                           starting_balance: Optional[float],
                           opens_today: list,
                           closes_today: list,
                           realized_pl_today: float,
                           unrealized_pl_today: float,
                           cycles_today: int,
                           errors_today: int,
                           stuck_tickers: list,
                           silenced_exceptions: Optional[list] = None) -> bool:
        """End-of-day recap (skill 32 §3.8).

        Fired once after market close. Aggregates today's trading
        activity into a single Telegram message: opens, closes,
        realized + unrealized P&L, cycle count, error count, account
        balance + intraday delta, and any tickers currently stuck
        in PDT-block or cooldown that need manual attention before
        tomorrow.

        ``opens_today``: list of dicts with keys ticker, strategy, credit
        ``closes_today``: list of dicts with keys ticker, strategy,
                          exit_signal, realized_pl
        ``stuck_tickers``: list of dicts with keys ticker, reason
        """
        # ── Header line ─────────────────────────────────────────────
        if starting_balance is not None and starting_balance > 0:
            day_change = account_balance - starting_balance
            day_change_pct = (day_change / starting_balance) * 100
            balance_str = (
                f"<b>Balance:</b> ${starting_balance:,.2f} → "
                f"${account_balance:,.2f} "
                f"({day_change_pct:+.2f}%, "
                f"{'+' if day_change >= 0 else '-'}${abs(day_change):,.2f})"
            )
        else:
            balance_str = f"<b>Balance:</b> ${account_balance:,.2f}"

        total_pl = realized_pl_today + unrealized_pl_today
        if total_pl > 0:
            pl_emoji = "📈"
        elif total_pl < 0:
            pl_emoji = "📉"
        else:
            pl_emoji = "➖"

        lines = [
            f"📊 <b>End-of-Day Summary — {date_label}</b>",
            "",
            balance_str,
            (f"{pl_emoji} <b>Day P&amp;L:</b> "
             f"${total_pl:+,.2f}  "
             f"(realized ${realized_pl_today:+,.2f}, "
             f"unrealized ${unrealized_pl_today:+,.2f})"),
            "",
        ]

        # ── Opens ───────────────────────────────────────────────────
        if opens_today:
            lines.append(f"<b>🟢 Opens ({len(opens_today)})</b>")
            for o in opens_today[:10]:
                lines.append(
                    f"• {o.get('ticker','?')} {o.get('strategy','?')} "
                    f"@ ${float(o.get('credit',0)):.2f}"
                )
            if len(opens_today) > 10:
                lines.append(f"<i>… and {len(opens_today) - 10} more</i>")
            lines.append("")
        else:
            lines.append("<b>🟢 Opens:</b> none today")
            lines.append("")

        # ── Closes ──────────────────────────────────────────────────
        if closes_today:
            lines.append(f"<b>🔴 Closes ({len(closes_today)})</b>")
            for c in closes_today[:10]:
                pl = float(c.get('realized_pl', 0))
                sign = "+" if pl >= 0 else ""
                lines.append(
                    f"• {c.get('ticker','?')} {c.get('strategy','?')} "
                    f"→ {c.get('exit_signal','?')}  "
                    f"<b>{sign}${pl:.2f}</b>"
                )
            if len(closes_today) > 10:
                lines.append(f"<i>… and {len(closes_today) - 10} more</i>")
            lines.append("")
        else:
            lines.append("<b>🔴 Closes:</b> none today")
            lines.append("")

        # ── Cycle health ────────────────────────────────────────────
        # ``cycles_today`` is journal-minute-distinct (one bucket per
        # minute with any journal activity). Each agent cycle writes
        # rows in ~1-2 minute buckets, so this is "agent active for
        # ~X minutes today" — not a strict cycle count. Still a useful
        # health signal (zero = agent didn't run; thousands = agent
        # ran hard all day).
        lines.append(
            f"<b>⚙️ Agent active:</b> {cycles_today} minute(s), "
            f"{errors_today} error(s) today"
        )

        # ── Stuck positions (surface so operator can act overnight) ─
        if stuck_tickers:
            lines.append("")
            lines.append(
                f"<b>⚠️ Needs manual attention ({len(stuck_tickers)})</b>"
            )
            for s in stuck_tickers:
                lines.append(
                    f"• {s.get('ticker','?')} — {s.get('reason','?')}"
                )
            lines.append(
                "<i>Close in Alpaca UI tonight or accept day-trade flag.</i>"
            )

        # ── Silenced exceptions (skill 34) — surface what failed quietly today ─
        if silenced_exceptions:
            total = sum(s.get("count", 1) for s in silenced_exceptions)
            lines.append("")
            lines.append(
                f"<b>🤫 Silenced exceptions ({len(silenced_exceptions)} "
                f"group{'s' if len(silenced_exceptions) != 1 else ''}, "
                f"{total} total)</b>"
            )
            for s in silenced_exceptions[:5]:
                ticker_tag = (
                    f" [{s.get('ticker','')}]"
                    if s.get("ticker") else ""
                )
                lines.append(
                    f"• <code>{s.get('exc_class','?')}</code> × "
                    f"{s.get('count',1)}{ticker_tag}\n"
                    f"  └ {s.get('source','?')}: "
                    f"<i>{(s.get('last_message','') or '')[:80]}</i>"
                )
            if len(silenced_exceptions) > 5:
                lines.append(
                    f"<i>… and {len(silenced_exceptions) - 5} more "
                    f"groups in signals_live.jsonl</i>"
                )

        return self._send("\n".join(lines), channel="info")

    def notify_position_closed(self, ticker: str, strategy: str,
                               exit_signal: str, exit_reason: str,
                               realized_pl: float, original_credit: float,
                               max_loss: float) -> bool:
        """A spread just closed (all legs filled). Tells the operator
        which exit fired, the P&L outcome, and how it compares to the
        original credit collected at open."""
        # Compute percent-of-credit captured so the operator can see
        # at a glance whether this hit the 50% profit target, ran to
        # near-max-loss, or finished somewhere in between.
        pct_of_credit = (
            (realized_pl / original_credit * 100)
            if original_credit > 0 else 0.0
        )
        pct_of_max_loss = (
            (abs(realized_pl) / max_loss * 100)
            if (realized_pl < 0 and max_loss > 0) else 0.0
        )
        if realized_pl > 0:
            emoji = "💰"
            outcome = f"<b>+${realized_pl:.2f}</b> ({pct_of_credit:+.0f}% of credit)"
        elif realized_pl < 0:
            emoji = "❌"
            outcome = (
                f"<b>-${abs(realized_pl):.2f}</b> "
                f"({pct_of_max_loss:.0f}% of max loss)"
            )
        else:
            emoji = "➖"
            outcome = "<b>break-even</b>"
        text = (
            f"{emoji} <b>{ticker} {strategy} CLOSED</b>\n"
            f"<b>Result:</b> {outcome}\n"
            f"<b>Exit signal:</b> {exit_signal}\n"
            f"<b>Reason:</b> <i>{exit_reason[:200]}</i>\n\n"
            f"<b>Original credit:</b> ${original_credit:.2f}  ·  "
            f"<b>Max loss:</b> ${max_loss:.2f}"
        )
        return self._send(text, channel="info")


__all__ = [
    "TelegramNotifier",
]
