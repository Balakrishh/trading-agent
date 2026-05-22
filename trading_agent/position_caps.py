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
from typing import Dict, Iterable, Set, Tuple

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


__all__ = ["compute_position_cap_dedup_set"]
