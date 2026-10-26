"""position_caps.py — per-ticker + per-sector position-count caps.

Extracted from the inline block in ``agent._run_cycle_impl`` (item 5,
2026-05-22). Returns the dedup set that Stage 2 (OPEN new positions)
uses to skip tickers that have already hit their per-ticker or
per-sector ceiling.

The two caps:

  * **Per-ticker cap** — ``MAX_POSITIONS_PER_TICKER``. Default 1.
    Pre-2026-05-13 this gate only triggered on ``signal=HOLD``
    positions, which let GLD slip through with a regime_shift exit
    pending and a fresh open get planned on top. Now every reported
    position counts regardless of signal.
  * **Per-sector cap** — ``MAX_POSITIONS_PER_SECTOR``. Default 2.
    Prevents concentration risk when several ETFs in the same
    sector (e.g. XLF + KRE both financials) would otherwise pile
    on at once. Sector lookup via ``sector_for(ticker)``.

The function is pure: same monitor_results + ticker universe in →
same dedup set out. Trivially testable.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def compute_position_cap_dedup_set(
    monitor_results: Dict,
    tickers: Iterable[str],
    *,
    sector_for,
    max_positions_per_ticker: int,
    max_positions_per_sector: int,
) -> Tuple[Set[str], Dict[str, int], Dict[str, int], Set[str]]:
    """Walk the monitor stage's reported positions and compute the
    union of tickers blocked by per-ticker OR per-sector caps.

    Returns:
        ``(blocked_set, positions_per_ticker, positions_per_sector,
        sectors_at_cap)``

        * ``blocked_set`` — Stage 2 union: skip these tickers when
          opening new positions.
        * ``positions_per_ticker`` — count of open positions per
          underlying. Used by the log line and dashboards.
        * ``positions_per_sector`` — count per sector.
        * ``sectors_at_cap`` — sectors that hit the ceiling. Logged
          separately so an operator can see "no new XLF because
          financials are full" without grepping per-ticker counts.

    Parameters:
        monitor_results — dict from ``_stage_monitor`` carrying a
            ``"positions"`` list. Each entry must have ``underlying``.
        tickers — the agent's configured universe.
        sector_for — ``Callable[[str], str]`` mapping ticker → sector
            label. Pre-extraction this was ``utils.sector_for``.
        max_positions_per_ticker — cap; default 1 in production.
        max_positions_per_sector — cap; default 2 in production.
    """
    positions_per_ticker: Dict[str, int] = {}
    positions_per_sector: Dict[str, int] = {}
    for sr in monitor_results.get("positions", []) or []:
        underlying = sr.get("underlying", "")
        if not underlying:
            continue
        positions_per_ticker[underlying] = (
            positions_per_ticker.get(underlying, 0) + 1
        )
        sec = sector_for(underlying)
        positions_per_sector[sec] = (
            positions_per_sector.get(sec, 0) + 1
        )

    # Per-ticker cap: block tickers at the ceiling.
    blocked: Set[str] = {
        t for t, n in positions_per_ticker.items()
        if n >= max_positions_per_ticker
    }
    # Per-sector cap: block tickers in sectors at the ceiling, EXCEPT
    # ones already caught by the per-ticker cap (keeps log noise
    # focused on sector-as-additional-gate signal).
    sectors_at_cap: Set[str] = {
        s for s, n in positions_per_sector.items()
        if n >= max_positions_per_sector
    }
    if sectors_at_cap:
        blocked |= {
            t for t in tickers
            if sector_for(t) in sectors_at_cap
            and t not in blocked
        }
    return blocked, positions_per_ticker, positions_per_sector, sectors_at_cap


def register_open(
    ticker: str,
    tickers: Iterable[str],
    positions_per_ticker: Dict[str, int],
    positions_per_sector: Dict[str, int],
    *,
    sector_for,
    max_positions_per_ticker: int,
    max_positions_per_sector: int,
) -> Set[str]:
    """Count one more open position (or pending order) on ``ticker`` and
    return the tickers that are now blocked by its per-ticker or
    per-sector cap. Mutates the two count dicts in place.

    2026-10-05: the caps were computed once from Stage 1's FILLED
    positions, so SPY, QQQ and IWM (all Broad Market, cap 2) were opened
    in one cycle. The agent now calls this for every ticker with an
    order still pending fill and after every submission in the loop."""
    positions_per_ticker[ticker] = positions_per_ticker.get(ticker, 0) + 1
    sec = sector_for(ticker)
    positions_per_sector[sec] = positions_per_sector.get(sec, 0) + 1
    blocked: Set[str] = set()
    if positions_per_ticker[ticker] >= max_positions_per_ticker:
        blocked.add(ticker)
    if positions_per_sector[sec] >= max_positions_per_sector:
        blocked |= {t for t in tickers if sector_for(t) == sec}
    return blocked


def ladder_failure(ticker: str, expiration: str,
                   open_expirations: Dict[str, List[str]],
                   opened_today: Set[str], min_gap_days: int,
                   open_pls: Optional[Dict[str, List[float]]] = None,
                   require_profit: bool = False) -> Optional[str]:
    """Backlog §6.7 laddering: a second position on a ticker is allowed
    only on a later day than the first entry and only when its expiration
    is ≥ ``min_gap_days`` from every open position's on that ticker.
    With ``require_profit`` (2026-10-06, "add to winners only") every open
    position on the ticker must also be at or above breakeven — the GLD
    put debit was losing while the planner kept proposing a second one.
    Returns a ``ladder_*`` reason, or None when the entry is allowed."""
    existing = open_expirations.get(ticker) or []
    if not existing:
        return None
    if ticker in opened_today:
        return "ladder_same_day_entry"
    pls = (open_pls or {}).get(ticker) or []
    if require_profit and pls and min(pls) < 0:
        return f"ladder_existing_losing (open P&L ${min(pls):.0f})"
    try:
        new = date.fromisoformat(expiration)
    except (TypeError, ValueError):
        return "ladder_expiration_unknown"
    for exp in existing:
        try:
            gap = abs((new - date.fromisoformat(exp)).days)
        except (TypeError, ValueError):
            return "ladder_expiration_unknown"
        if gap < min_gap_days:
            return f"ladder_gap_{gap}d_lt_{min_gap_days}d (open {exp})"
    return None


def open_pls(monitor_results: Dict) -> Dict[str, List[float]]:
    """underlying → mid P&L ($) of each open position (monitor summary)."""
    out: Dict[str, List[float]] = {}
    for sr in monitor_results.get("positions", []) or []:
        if sr.get("underlying"):
            out.setdefault(sr["underlying"], []).append(float(sr.get("pl") or 0.0))
    return out


def open_expirations(monitor_results: Dict) -> Dict[str, List[str]]:
    """underlying → expirations of its open positions (monitor summary)."""
    out: Dict[str, List[str]] = {}
    for sr in monitor_results.get("positions", []) or []:
        if sr.get("underlying") and sr.get("expiration"):
            out.setdefault(sr["underlying"], []).append(str(sr["expiration"]))
    return out


# Strategies excluded from the total-risk cap: Wheel legs carry their own
# collateral limit (≤ 40 % of equity per CSP, skill 40 §2.9) and a CSP's
# "max loss" is the whole strike, which would crowd out every spread.
TOTAL_RISK_EXCLUDED = frozenset({"Cash-Secured Put", "Covered Call"})


def open_defined_risk(monitor_results: Dict) -> float:
    """Σ max loss (dollars, all contracts) of open defined-risk positions
    from the monitor summary; Wheel legs excluded."""
    total = 0.0
    for sr in monitor_results.get("positions", []) or []:
        if sr.get("strategy") in TOTAL_RISK_EXCLUDED:
            continue
        total += float(sr.get("max_loss") or 0.0) * max(1, int(sr.get("contracts") or 1))
    return round(total, 2)


__all__ = ["compute_position_cap_dedup_set", "register_open",
           "open_defined_risk", "TOTAL_RISK_EXCLUDED", "ladder_failure",
           "open_expirations", "open_pls", "stop_cooldown_failure", "trade_direction",
           "RISK_GROUPS", "risk_group_for", "add_exposure", "direction_exposure",
           "direction_room", "direction_risk_failure"]


# ── Cool-down after a stop (added 2026-10-16) ───────────────────────────
# 2026-10-15: the TLT put debit stopped at 09:46 ET and the agent bought
# a new TLT put debit on the same bearish view at 09:50. The ladder rules
# only see OPEN positions, so a full close cleared every block.

STOP_EXIT_SIGNALS = frozenset({"stop_loss", "hard_stop", "delta_stop", "strike_proximity"})
_BULL = frozenset({"Bull Put Spread", "Call Debit Spread", "Bounce Bull Put Spread",
                   "Cash-Secured Put"})
_BEAR = frozenset({"Bear Call Spread", "Put Debit Spread"})


def trade_direction(strategy: str) -> str:
    """"bull" / "bear" for directional structures, else the strategy name
    itself (a neutral structure only cools down its own kind)."""
    if strategy in _BULL:
        return "bull"
    if strategy in _BEAR:
        return "bear"
    return str(strategy)


def sessions_between(start: date, end: date) -> int:
    """Weekdays after ``start`` up to and including ``end`` (0 on the same
    day). Exchange holidays are not excluded — a holiday only shortens
    the cool-down by one session."""
    if end <= start:
        return 0
    n, d = 0, start
    while d < end:
        d = date.fromordinal(d.toordinal() + 1)
        if d.weekday() < 5:
            n += 1
    return n


def stop_cooldown_failure(ticker: str, strategy: str, today: date,
                          recent_stops: Iterable[Tuple[str, str, date]],
                          sessions: int, repeat_sessions: int = 0,
                          repeat_window: int = 20) -> Optional[str]:
    """Reason a new entry on ``ticker`` is blocked because a position on
    it in the same direction was stopped out within the cool-down, else
    None. ``recent_stops`` holds (ticker, strategy, ET close date) of
    stop-type closes. The stop's own session and the next ``sessions``
    sessions are blocked; ``sessions`` ≤ 0 disables the check.

    Repeat stops (2026-10-26): when the latest same-direction stop is the
    second within ``repeat_window`` sessions, the block after it lasts
    ``repeat_sessions`` (if longer). TLT was stopped on 10-15, re-entered
    on 10-19 when the 1-session cool-down ended and stopped again 10-22."""
    if sessions <= 0:
        return None
    want = trade_direction(strategy)
    same = sorted((closed_on, strat) for tk, strat, closed_on in recent_stops
                  if tk == ticker and trade_direction(strat) == want)
    if not same:
        return None
    last_on, last_strat = same[-1]
    block = sessions
    earlier = [d for d, _ in same[:-1] if sessions_between(d, last_on) <= repeat_window]
    if earlier and repeat_sessions > block:
        block = repeat_sessions
    if sessions_between(last_on, today) <= block:
        tag = f"; repeat stop since {earlier[-1].isoformat()}" if block != sessions else ""
        return (f"stop_cooldown ({last_strat} stopped {last_on.isoformat()}{tag}; "
                f"no new {want} entry for {block} session(s) after)")
    return None


# ── Correlated-risk groups and direction budgets (E6, backlog 11.2) ────
# 2026-10-19: TLT ×2 + XLU put debits (≈ $839) were one bet on higher
# yields, and every entry since 10-14 was bearish; sector caps saw
# "Treasuries" and "Utilities" as unrelated. A group's same-direction
# max loss, and the book's, are capped as a share of equity.

RISK_GROUPS: Dict[str, frozenset] = {
    "rates": frozenset({"TLT", "IEF", "TLH", "XLU"}),
    "gold": frozenset({"GLD", "IAU", "GDX", "GDXJ", "SLV"}),
    "us_equity": frozenset({"SPY", "QQQ", "DIA", "IWM", "XLK", "SMH"}),
}


def risk_group_for(ticker: str) -> Optional[str]:
    """The ticker's correlated-risk group, or None when it has none."""
    for name, members in RISK_GROUPS.items():
        if ticker in members:
            return name
    return None


def _exposure_keys(ticker: str, strategy: str) -> List[str]:
    """Exposure buckets a position counts toward: ``book:<dir>`` and, for
    grouped tickers, ``group:<name>:<dir>``. Neutral structures and Wheel
    legs (collateral-based, TOTAL_RISK_EXCLUDED) count toward none."""
    d = trade_direction(strategy)
    if d not in ("bull", "bear") or strategy in TOTAL_RISK_EXCLUDED:
        return []
    keys = [f"book:{d}"]
    g = risk_group_for(ticker)
    if g:
        keys.append(f"group:{g}:{d}")
    return keys


def add_exposure(exposure: Dict[str, float], ticker: str, strategy: str,
                 dollars: float) -> None:
    """Add ``dollars`` of max loss to every bucket the position counts toward."""
    for k in _exposure_keys(ticker, strategy):
        exposure[k] = round(exposure.get(k, 0.0) + float(dollars), 2)


def direction_exposure(monitor_results: Dict) -> Dict[str, float]:
    """Max loss ($, all contracts) of open positions per direction bucket."""
    out: Dict[str, float] = {}
    for sr in monitor_results.get("positions", []) or []:
        add_exposure(out, sr.get("underlying", ""), sr.get("strategy", ""),
                     float(sr.get("max_loss") or 0.0) * max(1, int(sr.get("contracts") or 1)))
    return out


def direction_room(ticker: str, strategy: str, equity: float,
                   exposure: Dict[str, float], group_pct: float,
                   book_pct: float) -> Tuple[Optional[float], str]:
    """(dollars of max loss still allowed in this trade's direction, the
    bucket that limits it). (None, "") when no cap applies (neutral
    structure, caps off, or no equity)."""
    if equity <= 0:
        return None, ""
    room, label = None, ""
    for k in _exposure_keys(ticker, strategy):
        pct = group_pct if k.startswith("group:") else book_pct
        if pct <= 0:
            continue
        left = pct * equity - exposure.get(k, 0.0)
        if room is None or left < room:
            room, label = left, f"{k} ${exposure.get(k, 0.0):.0f} of ${pct * equity:.0f}"
    return room, label


def direction_risk_failure(ticker: str, strategy: str, per_contract: float,
                           equity: float, exposure: Dict[str, float],
                           group_pct: float, book_pct: float
                           ) -> Tuple[Optional[str], Optional[float]]:
    """(block reason, dollars allowed). Blocked when the room left in the
    trade's group / book direction is below one contract's max loss;
    otherwise the room caps the trade's size."""
    room, label = direction_room(ticker, strategy, equity, exposure, group_pct, book_pct)
    if room is None:
        return None, None
    if room < max(per_contract, 0.01):
        return f"direction_risk_cap ({label}; one contract ${per_contract:.0f})", room
    return None, room
