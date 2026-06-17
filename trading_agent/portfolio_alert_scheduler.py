"""portfolio_alert_scheduler.py — hourly Telegram digest of the long-term evaluator.

Skill: ``docs/skills/42_portfolio_alert_scheduler.md``.

Wakes up on a schedule (Cowork scheduled task, or cron / launchd on the
operator's machine), runs the Long-Term Evaluator against the persisted
holdings + watchlist, and posts an actionable digest to the Telegram
info channel.

The scheduler is a thin orchestrator — all scoring math lives in
``decision_engine.py`` and the recommendation assembly lives in
``long_term_evaluator.py``. This module is just:

  1. Gate on env opt-out + market hours.
  2. Load holdings (from ``holdings_store``) + watchlist (from
     ``watchlist_store``).
  3. Run ``LongTermEvaluator.recommend(...)``.
  4. Format a Telegram message body.
  5. Dedup on body-hash via the journal so identical digests in the
     same UTC day don't re-send.
  6. Send via ``TelegramNotifier.notify_portfolio_review(...)``.

CLI usage::

    python -m trading_agent.portfolio_alert_scheduler             # normal run
    python -m trading_agent.portfolio_alert_scheduler --dry-run   # print, no send
    python -m trading_agent.portfolio_alert_scheduler --force     # ignore market-hours gate

Cowork wires this as a scheduled task at 09:30/10:30/.../15:30 ET Mon-Fri.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from trading_agent.holdings_store import HoldingsSnapshot, load_holdings
from trading_agent.long_term_evaluator import (
    EvaluatorConfig,
    LongTermEvaluator,
    Recommendation,
)
from trading_agent.market_hours import is_within_market_hours
from trading_agent.positions_provider import (
    ManualPositionsProvider,
    Position,
    aggregate_snapshot,
)
from trading_agent.sector_map import sector_for
from trading_agent.watchlist_store import load_watchlist

logger = logging.getLogger(__name__)

ALERT_DEDUP_KEY_PREFIX = "lt_portfolio_review"

# Operator opt-out — env-gated so a noisy day can be silenced from the
# pi without redeploying code.
_ENV_ENABLED = "PORTFOLIO_ALERTS_ENABLED"


@dataclass
class AlertResult:
    """What the scheduler did this run. Returned to the CLI."""

    sent: bool = False
    skipped_reason: str = ""
    body_chars: int = 0
    body_hash: str = ""
    rec_count: int = 0
    holdings_count: int = 0
    debug: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Composition — turn evaluator output into a Telegram body
# ---------------------------------------------------------------------------

def _format_money(value: float) -> str:
    """`$1,234.56` formatter for digest bodies."""
    return f"${value:,.2f}"


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def compose_digest_body(
    *,
    positions: List[Position],
    watchlist_symbols: List[str],
    recommendations: List[Recommendation],
    now: datetime,
    cc_top_n: int = 3,
) -> str:
    """Format the operator-facing digest body. Pure function, testable.

    Sections:
      * Header — date + UTC stamp.
      * Holdings one-liner — total cost basis + position count.
      * Sector breakdown — top sectors by cost basis.
      * Income overlay — top-N covered-call recommendations with
        bracket sketches.
      * Manage existing — TP/SL anchor surveillance (option positions,
        next-session-stub today).
      * Skipped — held tickers below the 100-share floor with a
        one-line "why" each, capped to keep the digest readable.

    The body is plain text (no HTML) so the same body can be sent to a
    Telegram channel, journalled, hashed for dedup, or printed to stdout
    without escape-rule juggling.
    """
    snap = aggregate_snapshot(positions, sector_for=sector_for)
    held_qty = {p.ticker: p.qty for p in positions if p.kind == "stock"}
    held_set = set(held_qty.keys())
    wl_set = {t.strip().upper() for t in watchlist_symbols if t and t.strip()}

    lines: List[str] = []
    lines.append(f"📈 Portfolio Review — {now:%Y-%m-%d %H:%M} UTC")
    lines.append("")
    lines.append(
        f"Holdings: {_format_money(snap.total_stock_market_value)} "
        f"cost basis · {snap.position_count_by_kind.get('stock', 0)} "
        f"stock positions"
    )
    if snap.by_sector:
        top_sectors = sorted(
            snap.by_sector.items(), key=lambda x: -x[1],
        )[:4]
        sector_strs = [
            f"{name} "
            f"{_format_pct(value / max(1.0, snap.total_stock_market_value))}"
            for name, value in top_sectors
        ]
        lines.append("Sector: " + " · ".join(sector_strs))
    lines.append("")

    # ── Income overlay ───────────────────────────────────────────────
    cc_recs = [r for r in recommendations if r.strategy == "covered_call"]
    lines.append(f"🎯 Income overlay — {len(cc_recs)} candidate(s)")
    if cc_recs:
        for i, rec in enumerate(cc_recs[:cc_top_n], start=1):
            ann = rec.metrics.get("annualised_return", 0.0) * 100
            pop = rec.metrics.get("pop", 0.0) * 100
            dte = int(rec.metrics.get("dte", 0))
            strike = float(
                rec.legs[0].limit_price * 100  # not actual strike, placeholder
            )
            # Pull strike from the rationale string instead — the
            # Recommendation dataclass doesn't expose strike directly.
            lines.append(
                f"{i}. {rec.ticker}: {rec.rationale}"
            )
            lines.append(
                f"   Bracket: STO @ ${rec.entry_limit:.2f} → "
                f"BTC @ ${rec.take_profit_limit:.2f} + "
                f"stop if underlying < ${rec.stop_trigger:.2f}"
            )
        if len(cc_recs) > cc_top_n:
            lines.append(f"   _…and {len(cc_recs) - cc_top_n} more")
    else:
        lines.append("(none today — see skipped section below)")
    lines.append("")

    # ── Skipped reasons ──────────────────────────────────────────────
    intersect = sorted(held_set & wl_set)
    under_floor = [
        (t, held_qty[t]) for t in intersect if held_qty[t] < 100
    ]
    if under_floor:
        lines.append(
            f"⏭ Below CC floor — {len(under_floor)} ticker(s) need 100 shares:"
        )
        for tk, qty in under_floor[:6]:
            lines.append(f"   {tk}: held {qty} (need {100 - qty} more)")
        if len(under_floor) > 6:
            lines.append(f"   _…and {len(under_floor) - 6} more")
        lines.append("")

    not_on_watchlist = sorted(held_set - wl_set)
    if not_on_watchlist:
        lines.append(
            "🔍 Held but not on watchlist (no recommendations until "
            f"added): {', '.join(not_on_watchlist[:8])}"
        )
        lines.append("")

    # ── Watchlist signals — entry/exit consolidation ─────────────────
    # Watchlist tickers NOT held are entry-vehicle candidates (CSP /
    # LEAPS). The scoring functions land in skill 40 §2.2/§2.3 — once
    # there, this section renders concrete recommendations the same
    # shape as the income overlay above. Until then we surface the
    # universe so the operator sees their watchlist in every digest.
    watchlist_only = sorted(wl_set - held_set)
    if watchlist_only:
        lines.append(
            f"🔭 Watchlist entry candidates — {len(watchlist_only)} ticker(s)"
        )
        # Render up to 8 tickers in one line; the per-ticker entry-vehicle
        # scoring is next-session work, so a one-liner suffices today.
        head = watchlist_only[:8]
        tail_n = len(watchlist_only) - len(head)
        lines.append(
            f"   {', '.join(head)}"
            + (f"  (+{tail_n} more)" if tail_n else "")
        )
        lines.append(
            "   CSP / LEAPS entry signals land next session (skill 40 §2.2/§2.3)."
        )
        lines.append("")

    lines.append("— Long-Term Evaluator · skill 40 · read-only")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scheduler — orchestration + dedup + send
# ---------------------------------------------------------------------------

@dataclass
class SchedulerDeps:
    """Injection point for the scheduler's collaborators.

    Tests pass fixtures; production uses :func:`build_default_deps`.
    Keeping the collaborators behind a dataclass lets a future
    "manual run with paste-from-clipboard" CLI mode inject a different
    holdings provider without touching the scheduler internals.
    """

    holdings_loader: Callable[[], HoldingsSnapshot]
    watchlist_loader: Callable[[], List[str]]
    evaluator_factory: Callable[[List[Position]], LongTermEvaluator]
    telegram_send: Callable[[str, str], bool]   # (body, dedup_key) -> sent?
    market_hours_check: Callable[[datetime], bool]
    journal_dedup_check: Callable[[str], bool]  # key -> already_sent_today?
    journal_dedup_write: Callable[[str, str], None]  # key, body_hash


def run_scheduler(
    *,
    deps: SchedulerDeps,
    now: Optional[datetime] = None,
    force: bool = False,
    dry_run: bool = False,
) -> AlertResult:
    """Single-shot scheduler invocation. Returns ``AlertResult``."""
    now = now or datetime.now(timezone.utc)
    result = AlertResult()

    # Env opt-out gate (operator's silent kill switch).
    if os.environ.get(_ENV_ENABLED, "true").lower() in (
        "false", "0", "no", "off",
    ):
        result.skipped_reason = "PORTFOLIO_ALERTS_ENABLED=false"
        return result

    if not force and not deps.market_hours_check(now):
        result.skipped_reason = "outside market hours"
        return result

    # Load holdings.
    holdings_snap = deps.holdings_loader()
    if holdings_snap.is_empty:
        result.skipped_reason = "holdings file empty"
        return result

    try:
        positions_provider = ManualPositionsProvider.from_json_text(
            holdings_snap.raw_paste,
        )
    except ValueError as exc:
        result.skipped_reason = f"holdings parse failed: {exc!s}"
        return result

    positions = positions_provider.snapshot()
    result.holdings_count = len(positions)
    if not positions:
        result.skipped_reason = "0 positions after parse"
        return result

    # Load watchlist.
    watchlist = deps.watchlist_loader()
    if not watchlist:
        result.skipped_reason = "watchlist empty"
        return result

    # Score.
    evaluator = deps.evaluator_factory(positions)
    recommendations = evaluator.recommend(watchlist)
    result.rec_count = len(recommendations)

    # Compose + hash.
    body = compose_digest_body(
        positions=positions,
        watchlist_symbols=watchlist,
        recommendations=recommendations,
        now=now,
    )
    result.body_chars = len(body)
    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    result.body_hash = body_hash
    result.debug["body"] = body

    # Journal dedup — same body within the same UTC day stays silent.
    # The hash key includes only the date so an hourly run that produces
    # an identical digest doesn't ping the operator twice in 60 minutes.
    dedup_key = f"{ALERT_DEDUP_KEY_PREFIX}:{now:%Y-%m-%d}:{body_hash}"
    if deps.journal_dedup_check(dedup_key):
        result.skipped_reason = "dedup: same body already sent today"
        return result

    if dry_run:
        result.skipped_reason = "dry-run"
        return result

    # Send + journal.
    sent_ok = deps.telegram_send(body, dedup_key)
    if sent_ok:
        deps.journal_dedup_write(dedup_key, body_hash)
        result.sent = True
    else:
        result.skipped_reason = "telegram send returned False"
    return result


# ---------------------------------------------------------------------------
# Production deps — wired against the real modules
# ---------------------------------------------------------------------------

def build_default_deps() -> SchedulerDeps:
    """Wire ``SchedulerDeps`` against the real modules. Used by the CLI."""

    from trading_agent.config import load_config
    from trading_agent.journal_kb import JournalKB
    from trading_agent.market_data_factory import build_market_data_provider
    from trading_agent.telegram_notifier import TelegramNotifier

    config = load_config()
    journal = JournalKB()

    # Telegram — info channel. The notifier returns False silently when
    # creds are absent (same fail-quiet contract as the rest of the agent).
    notifier = TelegramNotifier()

    def _holdings_loader() -> HoldingsSnapshot:
        return load_holdings()

    def _watchlist_loader() -> List[str]:
        return load_watchlist().symbols()

    def _evaluator_factory(positions: List[Position]) -> LongTermEvaluator:
        provider = ManualPositionsProvider(positions=positions)
        # Chain fetcher — reuse the watchlist surface routing.
        # surface="long_term" + default_provider="schwab" — see skill 42.
        # Schwab's options-chain coverage is materially better than
        # Alpaca's `indicative` feed for the income-overlay scorer.
        # MARKET_DATA_PROVIDER_LONG_TERM in .env overrides per-tab.
        market_data = build_market_data_provider(
            alpaca_api_key=config.alpaca.api_key,
            alpaca_secret_key=config.alpaca.secret_key,
            alpaca_data_url=config.alpaca.data_url,
            alpaca_base_url=config.alpaca.base_url,
            surface="long_term",
            default_provider="schwab",
        )

        def _fetch(ticker: str) -> List[Dict[str, Any]]:
            from datetime import date

            from trading_agent.calendar_utils import next_weekly_expiration
            today = date.today()
            try:
                exp = next_weekly_expiration(
                    today, target_dte=45, dte_min=30, dte_max=60,
                )
                raw = market_data.fetch_option_chain(
                    underlying=ticker,
                    expiration_date=(
                        exp.isoformat() if hasattr(exp, "isoformat") else str(exp)
                    ),
                    option_type="call",
                ) or []
            except Exception as exc:  # noqa: BLE001 — fail-open per skill 40 §4
                logger.warning("Chain fetch for %s failed: %s", ticker, exc)
                return []
            normalised: List[Dict[str, Any]] = []
            for c in raw:
                try:
                    dte = int(c.get("dte") or (exp - today).days)
                except Exception:
                    dte = 45
                normalised.append({
                    "strike": float(c.get("strike", 0.0)),
                    "delta": float(c.get("delta", 0.0)),
                    "bid": float(c.get("bid", 0.0)),
                    "ask": float(c.get("ask", 0.0)),
                    "dte": dte,
                    "symbol": str(c.get("symbol", "")),
                    **(
                        {"iv_rank": float(c["iv_rank"])}
                        if "iv_rank" in c else {}
                    ),
                })
            return normalised

        return LongTermEvaluator(
            positions_provider=provider,
            call_chain_fetcher=_fetch,
            preset=None,
            config=EvaluatorConfig(),
        )

    def _telegram_send(body: str, dedup_key: str) -> bool:
        if not notifier.is_active:
            logger.warning(
                "TelegramNotifier not configured — alert not sent. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable."
            )
            return False
        return notifier.notify_portfolio_review(
            body=body, dedup_key=dedup_key,
        )

    def _market_hours_check(now: datetime) -> bool:
        return is_within_market_hours(now)

    def _journal_dedup_check(dedup_key: str) -> bool:
        # Re-uses the same JournalReader.telegram_alert_sent_today_utc
        # pattern the agent uses for trade alerts (skill 32 §3.4).
        # ticker == "_portfolio_" (sentinel, same as the EOD recap),
        # alert_type carries the body hash so per-body dedup is exact.
        try:
            from trading_agent.journal_reader import JournalReader
            jsonl_path = getattr(journal, "jsonl_path", None)
            if not jsonl_path:
                return False
            return JournalReader(jsonl_path).telegram_alert_sent_today_utc(
                ticker="_portfolio_",
                alert_type=dedup_key,
            )
        except Exception:  # noqa: BLE001 — fail-open (better dup than miss)
            return False

    def _journal_dedup_write(dedup_key: str, body_hash: str) -> None:
        try:
            journal.log_signal(
                ticker="_portfolio_",
                action="telegram_alert_sent",
                notes=f"portfolio_review {body_hash}",
                payload={
                    "alert_type": dedup_key,
                    "body_hash": body_hash,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not write dedup journal row: %s", exc)

    return SchedulerDeps(
        holdings_loader=_holdings_loader,
        watchlist_loader=_watchlist_loader,
        evaluator_factory=_evaluator_factory,
        telegram_send=_telegram_send,
        market_hours_check=_market_hours_check,
        journal_dedup_check=_journal_dedup_check,
        journal_dedup_write=_journal_dedup_write,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one cycle of the long-term portfolio review and post "
            "the digest to Telegram. Intended to be invoked hourly by "
            "a scheduled task or cron."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Render + hash the digest but skip the Telegram send.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore the market-hours gate (testing only).",
    )
    parser.add_argument(
        "--print-body", action="store_true",
        help="Echo the digest body to stdout.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    deps = build_default_deps()
    result = run_scheduler(
        deps=deps, force=args.force, dry_run=args.dry_run,
    )

    if args.print_body and result.debug.get("body"):
        print()
        print(result.debug["body"])
        print()

    print(
        f"sent={result.sent} reason={result.skipped_reason!r} "
        f"recs={result.rec_count} holdings={result.holdings_count} "
        f"body_hash={result.body_hash} chars={result.body_chars}"
    )
    return 0 if result.sent or result.skipped_reason == "dry-run" else 1


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(main())
