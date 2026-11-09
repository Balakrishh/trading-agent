"""Pre-market brief and plan-for-tomorrow (skill 71) — Telegram, read-only.

Two messages a trading day, each started by a systemd timer:

* ``premarket`` (09:00 ET) — the market risk state the open starts
  from, overnight / pre-market moves of the watchlist, today's macro
  events and earnings, where each open position stands against its
  exits, and what new entries the state allows.
* ``tomorrow`` (16:20 ET, after the end-of-day recap) — today's
  activity and top reject reasons, tomorrow's events, positions to
  watch tomorrow (near a target / stop, a DTE-safety close, ≤ 2
  sessions to expiry) and the closing market state.

Nothing here places, changes or stages an order. Quotes come from the
local data server (``/session-quotes``) — no second Schwab session.
Pre-market prices are an overnight read, not tradable marks (the agent
itself never acts before 09:30). Each message is sent at most once per
date (sentinel file) and only inside its window on an NYSE trading day.

    python -m trading_agent.session_brief premarket|tomorrow [--dry-run] [--force]
"""
from __future__ import annotations

import argparse
import html
import logging
import os
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

GAP_ALERT_PCT = 0.015                     # list watchlist movers at or beyond ±1.5 %
NEAR_EXIT_PTS = 0.10                      # "near" = within 10 points of a target / stop
INDEXES = ("SPY", "QQQ", "IWM", "DIA")
WINDOWS = {"premarket": (dtime(6, 0), dtime(9, 30)), "tomorrow": (dtime(16, 0), dtime(23, 59))}
SENTINEL_DIR = Path(os.environ.get("SESSION_BRIEF_DIR", "trade_journal"))


def _e(x: Any) -> str:
    return html.escape(str(x))


def _money(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{'+' if x >= 0 else '−'}${abs(x):,.0f}"


# ── positions vs exits ──────────────────────────────────────────────────

def exit_view(pos: Dict[str, Any], preset, today: date) -> Dict[str, Any]:
    """P&L, target and stop of one open position (``list_positions`` row),
    in dollars, plus flags for the day ``today``."""
    from trading_agent.calendar_utils import is_last_trading_day_before, sessions_before
    strat, n = pos.get("strategy", ""), max(1, int(pos.get("contracts") or 1))
    pl = pos.get("mid_pl")
    credit = float(pos.get("credit") or 0.0)
    if credit < 0:                                   # debit structure: risk is the debit
        basis = -credit * 100 * n
        tp = (preset.debit_profit_target_pct_bear if strat == "Put Debit Spread"
              and preset.debit_profit_target_pct_bear else preset.debit_profit_target_pct)
        sl = (preset.debit_stop_loss_pct_bull if strat == "Call Debit Spread"
              and preset.debit_stop_loss_pct_bull else preset.debit_stop_loss_pct)
        target, stop = tp * basis, -sl * basis
    else:                                            # credit: target on credit, stop on max loss
        basis = credit * 100 * n
        target = preset.profit_target_pct * basis
        stop = -0.5 * float(pos.get("total_max_loss") or pos.get("max_loss") or 0.0)
    flags: List[str] = []
    exp = None
    try:
        exp = date.fromisoformat(str(pos.get("expiration", "")))
    except ValueError:
        pass
    sessions = sessions_before(today, exp) if exp else None
    if exp and is_last_trading_day_before(today, exp):
        flags.append("DTE-safety close this session")
    elif sessions is not None and sessions <= 2:
        flags.append(f"{sessions} session(s) to expiry")
    if pl is not None and basis > 0:
        if target > 0 and pl >= target - NEAR_EXIT_PTS * basis:
            flags.append("near profit target")
        if stop < 0 and pl <= stop + NEAR_EXIT_PTS * basis:
            flags.append("near stop")
    if exp and (exp - today).days <= getattr(preset, "trail_max_hold_dte", 7):
        flags.append(f"≤{preset.trail_max_hold_dte} DTE: trail no longer holds")
    return dict(ticker=pos.get("ticker"), strategy=strat, pl=pl, basis=basis, target=target,
                stop=stop, expiration=pos.get("expiration"), sessions=sessions, flags=flags,
                playbook=pos.get("playbook", ""))


def _position_line(v: Dict[str, Any]) -> str:
    pct = f" ({v['pl'] / v['basis']:+.0%})" if v["pl"] is not None and v["basis"] else ""
    line = (f"• <b>{_e(v['ticker'])}</b> {_e(v['strategy'])} exp {_e(v['expiration'])}: "
            f"{_money(v['pl'])}{pct} · target {_money(v['target'])} · stop {_money(v['stop'])}")
    if v["flags"]:
        line += " — <i>" + _e("; ".join(v["flags"])) + "</i>"
    return line


# ── message builders (pure) ─────────────────────────────────────────────

def _state_lines(ms: Optional[Dict[str, Any]], preset=None) -> List[str]:
    if not ms or not ms.get("state"):
        return ["<b>Market state</b>: unavailable (no snapshot yet)"]
    gate = ms.get("gate") or {}
    asof = ms.get("as_of_utc", "")
    try:
        from trading_agent.market_hours import EASTERN
        asof = datetime.fromisoformat(asof.replace("Z", "+00:00")).astimezone(EASTERN).strftime("%a %H:%M ET")
    except (ValueError, TypeError):
        pass
    out = [f"<b>Market state</b>: {_e(ms['state'])} (last cycle {_e(asof)}) — size "
           f"×{gate.get('size_multiplier', 1):g}"
           + ("" if gate.get("allow_new_csp", True) else ", new CSPs paused")]
    if ms.get("reasons"):
        out.append("  " + _e("; ".join(ms["reasons"][:4])))
    if gate.get("allowed_strategies"):
        names = [f"{n} (off)" if n == "Bear Call Spread" and preset is not None
                 and not getattr(preset, "bear_call_enabled", True) else n
                 for n in gate["allowed_strategies"]]
        out.append("  Allowed: " + _e(", ".join(names)))
    return out


def _events_lines(events: List[Dict[str, Any]], earnings: Dict[str, date], day: date,
                  label: str) -> List[str]:
    todays = [e for e in events if e.get("date") == day.isoformat()]
    soon = {t: d for t, d in earnings.items() if d is not None and 0 <= (d - day).days <= 14}
    out = [f"<b>{label}</b>: " + (", ".join(f"{_e(e['kind'])} {_e(e.get('release_et', ''))} ET"
                                            for e in todays) or "no FOMC / CPI / jobs report")]
    if soon:
        out.append("  Earnings ≤ 14 d: " + _e(", ".join(f"{t} {d:%m-%d}" for t, d in sorted(
            soon.items(), key=lambda kv: kv[1]))))
    nxt = next((e for e in events if e.get("date", "") > day.isoformat()), None)
    if nxt:
        out.append(f"  Next macro: {_e(nxt['kind'])} {_e(nxt['date'])}")
    return out


def build_premarket(ctx: Dict[str, Any]) -> str:
    """``ctx``: today, market_state, quotes {tk: {close, extended, last,
    change_pct}}, events, earnings {tk: date}, views (exit_view rows),
    held (tickers with open positions), preset."""
    today: date = ctx["today"]
    lines = [f"<b>☀️ Pre-market brief — {today:%a %Y-%m-%d}</b>", ""]
    lines += _state_lines(ctx.get("market_state"), ctx.get("preset"))
    q = ctx.get("quotes") or {}
    idx = []
    for t in INDEXES:
        r = q.get(t) or {}
        ref = r.get("extended") or r.get("last")
        if r.get("close") and ref:
            idx.append(f"{t} {r['close']:.2f}→{ref:.2f} ({r['change_pct']:+.1%})")
    lines += ["", "<b>Overnight</b>: " + (_e(" · ".join(idx)) if idx else "no pre-market quotes")]
    movers = sorted(((t, r["change_pct"]) for t, r in q.items()
                     if t not in INDEXES and r.get("change_pct") is not None
                     and abs(r["change_pct"]) >= GAP_ALERT_PCT), key=lambda kv: -abs(kv[1]))
    if movers:
        lines.append("  Movers ≥1.5%: " + _e(", ".join(f"{t} {c:+.1%}" for t, c in movers)))
    lines += [""] + _events_lines(ctx.get("events") or [], ctx.get("earnings") or {}, today, "Today")
    views = ctx.get("views") or []
    lines += ["", f"<b>Open positions</b> ({len(views)})"] + ([_position_line(v) for v in views] or ["  none"])
    lines += ["", "<b>Entries today</b>"] + _entry_lines(ctx)
    lines += ["", "<i>Read-only. Pre-market quotes are indicative; the agent acts from 09:45 ET.</i>"]
    return "\n".join(lines)


def _entry_lines(ctx: Dict[str, Any]) -> List[str]:
    pre, out = ctx["preset"], []
    held = sorted(ctx.get("held") or [])
    if held:
        out.append("  Held (a 2nd only if winning, ≥7 d apart): " + _e(", ".join(held)))
    excl = list(getattr(pre, "put_debit_excluded_tickers", ()) or ())
    if excl:
        out.append("  No put debits: " + _e(", ".join(excl)))
    if getattr(pre, "debit_excluded_tickers", ()):
        out.append("  No debit spreads: " + _e(", ".join(pre.debit_excluded_tickers)))
    if getattr(pre, "vol_band_put_enabled", False):
        allowed = (ctx.get("market_state") or {}).get("gate", {}).get("allowed_strategies")
        blocked = allowed is not None and "Bull Put Spread" not in allowed
        out.append(f"  Vol-band bull put ({_e(','.join(pre.vol_band_put_tickers))}, rank "
                   f"{pre.vol_band_put_min_rank:g}–{pre.vol_band_put_max_rank:g})"
                   + (": blocked by the market state" if blocked else ": eligible"))
    return out or ["  no ticker-level blocks"]


def build_tomorrow(ctx: Dict[str, Any]) -> str:
    """``ctx``: today, next_day, journal (get_journal_summary), events,
    earnings, views (exit_view rows judged for next_day), market_state."""
    nd: date = ctx["next_day"]
    j = ctx.get("journal") or {}
    lines = [f"<b>🌙 Plan for tomorrow — {nd:%a %Y-%m-%d}</b>", "",
             f"<b>Today</b>: {j.get('opens_today', 0)} open(s), {j.get('closes_today', 0)} close(s), "
             f"realized {_money(j.get('realized_pl_today') or 0.0)}"]
    rej = j.get("reject_reasons_top5") or []
    if rej:
        def fmt(r):
            return f"{r[0]} ×{r[1]}" if isinstance(r, (list, tuple)) else str(r)
        lines.append("  Top rejects: " + _e("; ".join(fmt(r) for r in rej[:3])))
    lines += [""] + _events_lines(ctx.get("events") or [], ctx.get("earnings") or {}, nd, "Tomorrow")
    watch = [v for v in ctx.get("views") or [] if v["flags"]]
    lines += ["", f"<b>Positions to watch</b> ({len(watch)} of {len(ctx.get('views') or [])})"]
    lines += [_position_line(v) for v in watch] or ["  none near an exit"]
    lines += [""] + _state_lines(ctx.get("market_state"), ctx.get("preset"))
    lines += ["", "<i>Read-only. Exits still follow the agent's rules from 09:45 ET.</i>"]
    return "\n".join(lines)


# ── gathering (I/O) ─────────────────────────────────────────────────────

def gather(kind: str, today: date, tickers: Iterable[str], preset, *,
           get: Callable[[str], Optional[Dict[str, Any]]],
           positions: Callable[[], Dict[str, Any]],
           market_state: Callable[[], Dict[str, Any]],
           journal: Callable[[], Dict[str, Any]],
           earnings: Callable[[str], Optional[date]]) -> Dict[str, Any]:
    from trading_agent.calendar_utils import next_sessions
    tickers = list(tickers)
    day = today if kind == "premarket" else next_sessions(today, 2)[-1]
    ctx: Dict[str, Any] = {"today": today, "next_day": day, "preset": preset}
    for key, fn in (("market_state", market_state), ("journal", journal)):
        try:
            ctx[key] = fn()
        except Exception as exc:                  # noqa: BLE001, skill-34-exempt — one missing section must not stop the brief
            logger.warning("session brief: %s unavailable: %s", key, exc)
    try:
        ev = get("macro-events") or {}
        ctx["events"] = ev.get("events") or []
    except Exception as exc:                      # noqa: BLE001, skill-34-exempt — one missing section must not stop the brief
        logger.warning("session brief: macro events unavailable: %s", exc)
    if kind == "premarket":
        try:
            ctx["quotes"] = (get("session-quotes?tickers=" + ",".join(tickers)) or {}).get("quotes") or {}
        except Exception as exc:                  # noqa: BLE001, skill-34-exempt — one missing section must not stop the brief
            logger.warning("session brief: session quotes unavailable: %s", exc)
    ern = {}
    for t in tickers:
        try:
            ern[t] = earnings(t)
        except Exception as exc:                  # noqa: BLE001, skill-34-exempt — one missing section must not stop the brief
            logger.debug("session brief: earnings for %s unavailable: %s", t, exc)
    ctx["earnings"] = ern
    try:
        rows = (positions() or {}).get("open_positions") or []
    except Exception as exc:                      # noqa: BLE001, skill-34-exempt — one missing section must not stop the brief
        logger.warning("session brief: positions unavailable: %s", exc)
        rows = []
    ctx["views"] = [exit_view(r, preset, day) for r in rows]
    ctx["held"] = {r.get("ticker") for r in rows if r.get("ticker")}
    return ctx


def due(kind: str, now_et: datetime, sentinel_dir: Path = SENTINEL_DIR) -> Optional[str]:
    """None when the brief should go out now, else the reason it should not."""
    from trading_agent.calendar_utils import is_trading_day
    if not is_trading_day(now_et.date()):
        return f"{now_et.date()} is not a trading day"
    lo, hi = WINDOWS[kind]
    if not (lo <= now_et.time() <= hi):
        return f"outside the {kind} window {lo:%H:%M}–{hi:%H:%M} ET"
    if (sentinel_dir / f".session_brief_{kind}_{now_et.date()}").exists():
        return "already sent today"
    return None


def mark_sent(kind: str, day: date, sentinel_dir: Path = SENTINEL_DIR) -> None:
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    fp = sentinel_dir / f".session_brief_{kind}_{day}"
    tmp = fp.with_suffix(fp.suffix + ".tmp")
    tmp.write_text(datetime.now().isoformat())
    tmp.replace(fp)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pre-market brief / plan for tomorrow (skill 71)")
    ap.add_argument("kind", choices=sorted(WINDOWS))
    ap.add_argument("--dry-run", action="store_true", help="print, do not send")
    ap.add_argument("--force", action="store_true", help="ignore the window / once-a-day check")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))
    from trading_agent.config import load_config
    from trading_agent.earnings_calendar import next_earnings_date
    from trading_agent.market_hours import EASTERN
    from trading_agent.mcp.tools.market import _data_server_get, get_market_state
    from trading_agent.mcp.tools.positions import get_journal_summary, list_positions
    from trading_agent.strategy_presets import load_active_preset
    from trading_agent.telegram_notifier import TelegramNotifier

    now_et = datetime.now(EASTERN)
    why_not = None if args.force else due(args.kind, now_et)
    if why_not:
        logger.info("session brief %s: skipped — %s", args.kind, why_not)
        return 0
    tickers = list(load_config(os.path.join(os.getcwd(), ".env")).trading.tickers)
    preset = load_active_preset()
    ctx = gather(args.kind, now_et.date(), tickers, preset, get=_data_server_get,
                 positions=list_positions, market_state=get_market_state,
                 journal=get_journal_summary, earnings=next_earnings_date)
    text = build_premarket(ctx) if args.kind == "premarket" else build_tomorrow(ctx)
    if args.dry_run:
        print(text)
        return 0
    if TelegramNotifier().notify_session_brief(text):
        mark_sent(args.kind, now_et.date())
        logger.info("session brief %s: sent", args.kind)
    else:
        logger.warning("session brief %s: Telegram send failed (or not configured)", args.kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
