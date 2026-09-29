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


@dataclass(frozen=True)
class ReviewOutput:
    """LLM verdict + parsed diffs + digest lines."""
    observations:      List[str]
    preset_proposal:   Optional[Dict[str, Any]]
    watchlist_proposal: Optional[Dict[str, Any]]
    digest_lines:      List[str]
    confidence:        float = 0.0
    raw_llm_text:      str = ""


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


def _load_macro_snapshot() -> Dict[str, Any]:
    """Best-effort macro read — never blocks the review on a data outage."""
    out: Dict[str, Any] = {}
    try:
        from trading_agent.vix_regime_monitor import current_vix_zone
        out["vix_zone"] = current_vix_zone()
    except Exception:                                              # noqa: BLE001
        out["vix_zone"] = None
    return out


def assemble_context(review_date: Optional[str] = None) -> ReviewContext:
    """Gather every field the LLM needs. No LLM call yet."""
    from trading_agent.journal_reader import JournalReader        # noqa: PLC0415
    from trading_agent.strategy_presets import load_active_preset # noqa: PLC0415
    from dataclasses import asdict, is_dataclass

    reader = JournalReader()
    preset = load_active_preset()
    preset_dict = asdict(preset) if is_dataclass(preset) else dict(vars(preset))

    opens = [
        {
            "ticker":   getattr(o, "ticker", None),
            "strategy": getattr(o, "strategy", None),
            "credit":   getattr(o, "credit", None),
        }
        for o in reader.opens_today()
    ]
    closes = [
        {
            "ticker":   getattr(c, "ticker", None),
            "strategy": getattr(c, "strategy", None),
            "pnl":      getattr(c, "pnl", None),
            "reason":   getattr(c, "reason", None),
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
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a trading-strategy reviewer. Once per day, at market close, "
    "you read one day's worth of credit-spread trade activity plus the "
    "current strategy preset, and produce a short structured report. "
    "You NEVER trade — your only job is proposal-writing. "
    "Return valid JSON with these top-level keys: observations "
    "(list of strings), preset_proposal (object or null), "
    "watchlist_proposal (object or null), digest_lines (list of strings), "
    "confidence (float 0.0-1.0). When you propose a preset change, use "
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
        "watchlist":      ctx.watchlist,
        "macro":          ctx.macro,
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


def call_llm(ctx: ReviewContext) -> ReviewOutput:
    """Send the prompt to the configured LLM. Returns ReviewOutput even
    on failure (with empty proposals + raw_llm_text populated).
    """
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
            "watchlist":      ctx.watchlist,
            "macro":          ctx.macro,
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
