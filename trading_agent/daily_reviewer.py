"""End-of-Day Trade Journal Reviewer — skill 56.

Once a day (via launchd at 4:15 PM ET), this module reads today's
full trade journal + current PresetConfig + watchlist + macro context,
asks the LLM for a structured review, writes:

- an audit JSON to ``daily_reviews/YYYY-MM-DD.json`` (permanent record)
- an optional pending preset-update proposal (only when the LLM
  produces a concrete diff)
- a Markdown digest to Telegram

**Does not trade.** Numeric-threshold auto-closes (skills 30, 31)
keep firing at cycle time — this reviewer is retrospective.

Read/write invariant: this module NEVER imports the executor or any
order-submission primitive. Verified by
``tests/conformance/test_skill_56_daily_reviewer.py`` (AST walker).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from trading_agent import pending_preset_updates_writer as _pwriter


log = logging.getLogger("trading_agent.daily_reviewer")


REVIEWS_DIR = Path(os.environ.get(
    "TRADING_AGENT_REVIEWS_DIR",
    "daily_reviews",
)).resolve()

# LLM fields the reviewer is permitted to propose changes to. Anything
# outside this list is ignored at parse time. Fields like max_risk_pct
# and max_delta are deliberately excluded — they require human account-
# level judgment that a per-day journal doesn't capture.
_ALLOWED_PROPOSAL_FIELDS: tuple[str, ...] = (
    "profit_target_pct",
    "edge_buffer",
    "min_pop",
    "max_leg_spread_cents",
    "defensive_roll_enabled",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReviewContext:
    """Everything assembled BEFORE the LLM call. Pure data."""
    review_date:     str
    opens:           List[Dict[str, Any]]
    closes:          List[Dict[str, Any]]
    reject_reasons:  List[tuple]
    realized_pl:     float
    preset:          Dict[str, Any]
    watchlist:       List[str]
    macro:           Dict[str, Any]
    recent_alerts:   List[str]
    cold_start:      bool = False
    # Append-only (2026-10-16): the tickers the agent actually scans
    # (TICKERS) — ``watchlist`` is the dashboard Watchlist tab, the list
    # a watchlist_proposal edits — and the shadow trend-filter tally.
    trading_universe: List[str] = field(default_factory=list)
    trend_filters:   Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReviewOutput:
    """LLM verdict + parsed diffs + digest lines."""
    observations:      List[str]
    preset_proposal:   Optional[Dict[str, Any]]
    watchlist_proposal: Optional[Dict[str, Any]]
    digest_lines:      List[str]
    confidence:        float = 0.0
    raw_llm_text:      str = ""
    # claude CLI path (skill 56 §3): session id, context / output tokens.
    llm_usage:         Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def _load_watchlist() -> List[str]:
    try:
        from trading_agent.watchlist_store import load_watchlist   # noqa: PLC0415
        wl = load_watchlist()
        # watchlist_store returns various shapes across surfaces; normalize.
        if isinstance(wl, list):
            return [str(t).upper() for t in wl]
        if hasattr(wl, "tickers"):
            return [str(t).upper() for t in wl.tickers]
        if isinstance(wl, dict) and "tickers" in wl:
            return [str(t).upper() for t in wl["tickers"]]
        return []
    except Exception as exc:                                       # noqa: BLE001
        log.warning("Could not load watchlist (%s); using empty list.", exc)
        return []


def _load_trading_universe() -> List[str]:
    """The agent's scan list (``TICKERS`` in .env)."""
    try:
        from trading_agent.config import load_config               # noqa: PLC0415
        return list(load_config().trading.tickers)
    except Exception as exc:                                       # noqa: BLE001
        log.warning("Could not load TICKERS (%s); using empty list.", exc)
        return []


def _load_macro_snapshot() -> Dict[str, Any]:
    """Best-effort macro read from the agent's last market-state snapshot
    (skill 58) — never blocks the review on a data outage. Before
    2026-10-16 this called a ``current_vix_zone`` that never existed, so
    ``vix_zone`` was always null."""
    out: Dict[str, Any] = {"vix_zone": None}
    try:
        from trading_agent.market_state import read_state          # noqa: PLC0415
        from trading_agent.vix_regime_monitor import classify_vix  # noqa: PLC0415
        snap = read_state()
        if not snap:
            return out
        inputs = snap.get("inputs") or {}
        gate = snap.get("gate") or {}
        vix = inputs.get("vix")
        out.update({
            "market_state":    snap.get("state"),
            "state_reasons":   snap.get("reasons") or [],
            "size_multiplier": gate.get("size_multiplier"),
            "allow_new_csp":   gate.get("allow_new_csp"),
            "vix":             vix,
            "vix_zone":        classify_vix(float(vix)).name if vix is not None else None,
            "vix3m":           inputs.get("vix3m"),
            "term_ratio":      inputs.get("term_ratio"),
            "breadth":         inputs.get("breadth"),
            "breadth_n":       inputs.get("breadth_n"),
            "spy_price":       inputs.get("spy_price"),
            "spy_sma20":       inputs.get("spy_sma20"),
            "spy_sma50":       inputs.get("spy_sma50"),
            "spy_sma200":      inputs.get("spy_sma200"),
            "spy_rsi":         inputs.get("spy_rsi"),
            "account_balance": snap.get("account_balance"),
            "snapshot_age_s":  snap.get("age_seconds"),
        })
    except Exception as exc:                                       # noqa: BLE001
        log.warning("Could not read market state (%s).", exc)
    return out


def _trend_filter_tally(reader) -> Dict[str, Any]:
    """Today's shadow trend-filter verdicts (skill 65): how many plans
    were checked and how many each filter would have blocked in live
    mode, with the tickers involved."""
    from collections import Counter                                # noqa: PLC0415
    today = reader._today_et()
    checked = 0
    fails: Counter = Counter()
    tickers: Dict[str, set] = {}
    for rec in reader._iter_rows():
        if reader._row_et_date(rec) != today:
            continue
        rs = rec.get("raw_signal") or {}
        tf = rs.get("trend_filters") if isinstance(rs, dict) else None
        if not isinstance(tf, dict):
            continue
        checked += 1
        for chk in tf.get("checks") or []:
            if chk.get("ok") is False:
                name = str(chk.get("name"))
                fails[name] += 1
                tickers.setdefault(name, set()).add(rec.get("ticker"))
    return {"plans_checked": checked,
            "would_block": dict(fails),
            "would_block_tickers": {k: sorted(t for t in v if t)
                                    for k, v in tickers.items()}}


def assemble_context(review_date: Optional[str] = None) -> ReviewContext:
    """Gather every field the LLM needs. No LLM call yet."""
    from trading_agent.journal_reader import JournalReader        # noqa: PLC0415
    from trading_agent.strategy_presets import load_active_preset # noqa: PLC0415
    from dataclasses import asdict, is_dataclass

    reader = JournalReader(
        as_of=date.fromisoformat(review_date) if review_date else None)
    preset = load_active_preset()
    preset_dict = asdict(preset) if is_dataclass(preset) else dict(vars(preset))

    opens = [
        {
            "ticker":   getattr(o, "ticker", None),
            "strategy": getattr(o, "strategy", None),
            "credit":   getattr(o, "credit", None),
            "max_loss": getattr(o, "max_loss", None),
            "expiration": getattr(o, "expiration", None),
            "opened_at_utc": getattr(o, "timestamp_utc", None),
        }
        for o in reader.opens_today()
    ]
    closes = [
        {
            "ticker":   getattr(c, "ticker", None),
            "strategy": getattr(c, "strategy", None),
            # ClosedTrade has realized_pl / exit_signal / exit_reason;
            # reading "pnl" / "reason" sent null to the LLM until 2026-10-16.
            "pnl":         getattr(c, "realized_pl", None),
            "exit_signal": getattr(c, "exit_signal", None),
            "reason":      getattr(c, "exit_reason", None),
            "expiration":  getattr(c, "expiration", None),
            "closed_at_utc": getattr(c, "timestamp_utc", None),
        }
        for c in reader.closes_today()
    ]
    cold_start = not (REVIEWS_DIR.exists()
                       and any(REVIEWS_DIR.glob("*.json")))

    return ReviewContext(
        review_date=review_date or date.today().isoformat(),
        opens=opens,
        closes=closes,
        reject_reasons=[list(r) for r in reader.reject_reasons_today(top_n=5)],
        realized_pl=float(reader.realized_pl_today() or 0.0),
        preset=preset_dict,
        watchlist=_load_watchlist(),
        macro=_load_macro_snapshot(),
        recent_alerts=[str(e) for e in reader.silenced_exceptions_today()],
        cold_start=cold_start,
        trading_universe=_load_trading_universe(),
        trend_filters=_safe_trend_tally(reader),
    )


def _safe_trend_tally(reader) -> Dict[str, Any]:
    try:
        return _trend_filter_tally(reader)
    except Exception as exc:                                       # noqa: BLE001
        log.warning("Could not tally trend filters (%s).", exc)
        return {}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a trading-strategy reviewer. Once per day, at market close, "
    "you read one day's worth of defined-risk options trade activity "
    "(credit and debit spreads, calendars, Wheel puts) plus the "
    "current strategy preset, and produce a short structured report. "
    "You NEVER trade — your only job is proposal-writing. "
    "Return valid JSON with these top-level keys: observations "
    "(list of strings), preset_proposal (object or null), "
    "watchlist_proposal (object or null), digest_lines (list of strings), "
    "confidence (float 0.0-1.0). trading_universe is the list of tickers "
    "the agent scans and trades; watchlist is the separate dashboard "
    "Watchlist tab, which is what a watchlist_proposal edits — a ticker "
    "in trading_universe but not in watchlist is expected. macro is the "
    "agent's last whole-market risk snapshot. trend_filters counts the "
    "shadow relative-strength / pullback verdicts. When you propose a preset change, use "
    "one of these fields only: profit_target_pct, edge_buffer, min_pop, "
    "max_leg_spread_cents, defensive_roll_enabled. Never propose changes "
    "to max_risk_pct, max_delta, or dte_* fields."
)


def _build_user_prompt(ctx: ReviewContext) -> str:
    body = {
        "review_date":    ctx.review_date,
        "cold_start":     ctx.cold_start,
        "opens":          ctx.opens,
        "closes":         ctx.closes,
        "reject_reasons_top5": ctx.reject_reasons,
        "realized_pl_today":   ctx.realized_pl,
        "current_preset": ctx.preset,
        "trading_universe": ctx.trading_universe,
        "watchlist":      ctx.watchlist,
        "macro":          ctx.macro,
        "trend_filters":  ctx.trend_filters,
        "recent_alerts":  ctx.recent_alerts,
    }
    return (
        "Review this day's activity and return the JSON structure "
        "described in the system prompt.\n\n"
        + json.dumps(body, indent=2, default=str)
    )


def _parse_llm_output(raw: Dict[str, Any]) -> ReviewOutput:
    """Validate + filter the LLM's response.

    Rejects proposal fields outside the allowlist so a hallucinated
    ``max_risk_pct`` proposal cannot land even if the LLM ignores the
    system prompt. Malformed → return an all-empty ReviewOutput with
    raw_llm_text populated so the operator can see what came back.
    """
    if not isinstance(raw, dict):
        return ReviewOutput(observations=[], preset_proposal=None,
                             watchlist_proposal=None, digest_lines=[],
                             raw_llm_text=str(raw))

    observations = [str(o) for o in raw.get("observations") or []]
    digest_lines = [str(d) for d in raw.get("digest_lines") or []]
    confidence   = float(raw.get("confidence") or 0.0)

    pp = raw.get("preset_proposal")
    if isinstance(pp, dict) and pp.get("field") in _ALLOWED_PROPOSAL_FIELDS:
        preset_proposal = {
            "field":      pp.get("field"),
            "current":    pp.get("current"),
            "proposed":   pp.get("proposed"),
            "reason":     str(pp.get("reason") or "")[:280],
            "confidence": float(pp.get("confidence") or 0.0),
        }
    else:
        preset_proposal = None

    wp = raw.get("watchlist_proposal")
    if isinstance(wp, dict) and (wp.get("drops") or wp.get("adds")):
        watchlist_proposal = {
            "drops":  [str(t).upper() for t in wp.get("drops") or []],
            "adds":   [str(t).upper() for t in wp.get("adds") or []],
            "reason": str(wp.get("reason") or "")[:280],
        }
    else:
        watchlist_proposal = None

    return ReviewOutput(
        observations=observations,
        preset_proposal=preset_proposal,
        watchlist_proposal=watchlist_proposal,
        digest_lines=digest_lines,
        confidence=confidence,
        raw_llm_text="",
    )


def _empty(raw: str, usage: Optional[Dict[str, Any]] = None) -> ReviewOutput:
    return ReviewOutput(observations=[], preset_proposal=None,
                        watchlist_proposal=None, digest_lines=[],
                        raw_llm_text=raw, llm_usage=usage or {})


def _call_claude_cli(ctx: ReviewContext) -> ReviewOutput:
    """``TRADING_AGENT_REVIEWER_LLM=claude_cli``: one persistent ``claude -p``
    session across days (claude_reviewer.py)."""
    from dataclasses import replace                              # noqa: PLC0415
    from trading_agent import claude_reviewer                    # noqa: PLC0415
    parsed, usage, err = claude_reviewer.review(
        _SYSTEM_PROMPT, _build_user_prompt(ctx), ctx.review_date, REVIEWS_DIR)
    if parsed is None:
        log.warning("claude CLI review failed: %s", err)
        return _empty(f"<error: {err}>", usage)
    return replace(_parse_llm_output(parsed), llm_usage=usage)


def call_llm(ctx: ReviewContext) -> ReviewOutput:
    """Send the prompt to the configured LLM. Returns ReviewOutput even
    on failure (with empty proposals + raw_llm_text populated).

    ``TRADING_AGENT_REVIEWER_LLM=claude_cli`` uses the logged-in Claude
    Code CLI (the Pi's default since 2026-10-14); anything else uses
    ``LLMClient`` (Ollama / OpenAI-compatible).
    """
    if os.environ.get("TRADING_AGENT_REVIEWER_LLM", "").strip().lower() == "claude_cli":
        try:
            return _call_claude_cli(ctx)
        except Exception as exc:                                    # noqa: BLE001
            log.warning("claude CLI review crashed (%s); returning empty review.", exc)
            return _empty(f"<error: {exc}>")
    try:
        from trading_agent.llm_client import LLMClient             # noqa: PLC0415
        client = LLMClient()
        result = client.chat_json(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(ctx)},
            ],
            temperature=0.2,
        )
        if not result:
            return ReviewOutput(observations=[], preset_proposal=None,
                                 watchlist_proposal=None, digest_lines=[],
                                 raw_llm_text="<empty response>")
        return _parse_llm_output(result)
    except Exception as exc:                                        # noqa: BLE001
        log.warning("LLM call failed (%s); returning empty review.", exc)
        return ReviewOutput(observations=[], preset_proposal=None,
                             watchlist_proposal=None, digest_lines=[],
                             raw_llm_text=f"<error: {exc}>")


# ---------------------------------------------------------------------------
# Digest rendering (deterministic — snapshot-testable)
# ---------------------------------------------------------------------------

def render_telegram_digest(
    ctx: ReviewContext,
    output: ReviewOutput,
    proposal_id: Optional[str] = None,
) -> str:
    """Render the Telegram-flavoured Markdown digest. Deterministic.

    Uses HTML tags because the notifier configures parse_mode=HTML
    (see telegram_notifier.py::_send).
    """
    header = f"<b>📓 Daily Review — {ctx.review_date}</b>"
    if ctx.cold_start:
        header += " <i>(cold start — no prior reviews)</i>"

    summary = (
        f"Today: {len(ctx.opens)} opens · {len(ctx.closes)} closes · "
        f"{len(ctx.reject_reasons)} reject reasons · "
        f"P&amp;L {ctx.realized_pl:+.2f}"
    )

    parts = [header, "", summary, ""]

    if output.observations:
        parts.append("<b>Observations</b>")
        for o in output.observations:
            parts.append(f"  • {o}")
        parts.append("")

    proposals: List[str] = []
    if output.preset_proposal:
        pp = output.preset_proposal
        proposals.append(
            f"  • Preset: <code>{pp['field']}</code> "
            f"{pp['current']!r} → {pp['proposed']!r} "
            f"(conf {pp['confidence']:.2f})"
        )
        if pp.get("reason"):
            proposals.append(f"    <i>{pp['reason']}</i>")
    if output.watchlist_proposal:
        wp = output.watchlist_proposal
        if wp.get("drops"):
            proposals.append(f"  • Watchlist drops: {', '.join(wp['drops'])}")
        if wp.get("adds"):
            proposals.append(f"  • Watchlist adds:  {', '.join(wp['adds'])}")
        if wp.get("reason"):
            proposals.append(f"    <i>{wp['reason']}</i>")

    if proposals:
        parts.append("<b>Proposals</b>")
        parts.extend(proposals)
        parts.append("")
        if proposal_id:
            parts.append(
                "Apply with:\n"
                f"  <code>python -m trading_agent.apply_preset_update "
                f"{proposal_id}</code>"
            )
    else:
        parts.append("<i>No preset or watchlist changes proposed today.</i>")

    if output.raw_llm_text:
        parts.append("")
        parts.append(f"<i>LLM note: {output.raw_llm_text[:200]}</i>")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _write_audit(ctx: ReviewContext, output: ReviewOutput) -> Path:
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    fp = REVIEWS_DIR / f"{ctx.review_date}.json"
    payload = {
        "review_date":     ctx.review_date,
        "written_at_utc":  datetime.now(timezone.utc).isoformat(),
        "context": {
            "opens":          ctx.opens,
            "closes":         ctx.closes,
            "reject_reasons": ctx.reject_reasons,
            "realized_pl":    ctx.realized_pl,
            "preset":         ctx.preset,
            "trading_universe": ctx.trading_universe,
            "watchlist":      ctx.watchlist,
            "macro":          ctx.macro,
            "trend_filters":  ctx.trend_filters,
            "recent_alerts":  ctx.recent_alerts,
            "cold_start":     ctx.cold_start,
        },
        "output": {
            "observations":       output.observations,
            "preset_proposal":    output.preset_proposal,
            "watchlist_proposal": output.watchlist_proposal,
            "digest_lines":       output.digest_lines,
            "confidence":         output.confidence,
            "raw_llm_text":       output.raw_llm_text,
            "llm_usage":          output.llm_usage,
        },
    }
    tmp = fp.with_suffix(fp.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(fp)
    return fp


def _maybe_stage_proposal(ctx: ReviewContext,
                            output: ReviewOutput) -> Optional[str]:
    """Write a pending preset-update proposal iff the LLM produced a
    concrete diff. Returns the proposal UUID, or None when there's
    nothing to stage.
    """
    diff: Dict[str, Any] = {}
    if output.preset_proposal:
        pp = output.preset_proposal
        diff[pp["field"]] = pp["proposed"]
    if not diff and not output.watchlist_proposal:
        return None
    return _pwriter.write(
        review_date=ctx.review_date,
        preset_snapshot=ctx.preset,
        preset_diff=diff,
        watchlist_diff=output.watchlist_proposal or {},
        observations=output.observations,
        llm_reasoning=(output.preset_proposal or {}).get("reason", ""),
        confidence=output.confidence,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(review_date: Optional[str] = None,
        *,
        send_telegram: bool = True) -> Dict[str, Any]:
    """End-to-end. Returns a dict summarizing what was written."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ctx = assemble_context(review_date)
    output = call_llm(ctx)

    audit_path   = _write_audit(ctx, output)
    proposal_id  = _maybe_stage_proposal(ctx, output)
    digest       = render_telegram_digest(ctx, output, proposal_id)

    sent = False
    if send_telegram:
        try:
            from trading_agent.telegram_notifier import TelegramNotifier
            notifier = TelegramNotifier()
            if notifier.is_active():
                sent = notifier._send(digest, channel="info")   # noqa: SLF001
        except Exception as exc:                                # noqa: BLE001
            log.warning("Telegram digest send failed (%s).", exc)

    return {
        "review_date":  ctx.review_date,
        "audit_path":   str(audit_path),
        "proposal_id":  proposal_id,
        "digest":       digest,
        "telegram_sent": sent,
    }
