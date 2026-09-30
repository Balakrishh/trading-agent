"""wheel_lifecycle.py — reconcile expired Wheel legs (skill 40 §2.9).

A Wheel leg that reaches expiration never produces a close order: it
either expires worthless or is exercised, and the broker simply swaps the
option for shares (or removes shares). Without this module those trades
stay "open" in the journal forever and their P&L is never recorded (the
2026-07 XLE problem, but by design rather than by outage).

After expiration, compare the Wheel trade against current share holdings:

| Leg | Shares held ≥ 100 × contracts? | Outcome |
|---|---|---|
| Cash-Secured Put | yes | ``assigned`` — shares delivered; next leg is a covered call |
| Cash-Secured Put | no  | ``expired_worthless`` |
| Covered Call     | no  | ``called_away`` — shares delivered away |
| Covered Call     | yes | ``expired_worthless`` |

Each outcome is journalled as an ``action="closed"`` row keyed on the same
(ticker, strategy, expiration) so ``JournalReader.open_trades`` pairs it
and the trade leaves the open list. Realized P&L recorded is the premium
kept (credit × 100 × contracts); share P&L after assignment belongs to
the share position, not to the option leg.

Limitation: shares already held before the put was sold are
indistinguishable from assigned shares. Wheel puts are only recommended
for tickers held < 100 shares (skill 40 §3.5), which keeps this rare.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from trading_agent.wheel_policy import (
    CC_STRATEGY,
    CSP_STRATEGY,
    EXIT_ASSIGNED,
    EXIT_CALLED_AWAY,
    EXIT_EXPIRED_WORTHLESS,
    WHEEL_STRATEGIES,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WheelResolution:
    ticker: str
    strategy: str
    expiration: str
    contracts: int
    credit: float
    exit_signal: str
    realized_pl: float
    reason: str

    def to_raw_signal(self) -> Dict[str, Any]:
        """Journal payload shaped like the agent's own close rows."""
        return {
            "strategy": self.strategy,
            "expiration": self.expiration,
            "exit_signal": self.exit_signal,
            "exit_reason": self.reason,
            "net_unrealized_pl": self.realized_pl,
            "fill_status": "complete",
            "contracts": self.contracts,
            "source": "wheel_lifecycle",
        }


def resolve_expired_wheel_trades(open_trades: List[Any],
                                 shares_by_ticker: Dict[str, int],
                                 today: date) -> List[WheelResolution]:
    """Pure decision step. ``open_trades`` are ``JournalReader.OpenedTrade``
    rows; only Wheel legs whose expiration is strictly before ``today``
    are resolved (on expiration day the broker has not settled yet)."""
    out: List[WheelResolution] = []
    for t in open_trades:
        if t.strategy not in WHEEL_STRATEGIES:
            continue
        try:
            expired = date.fromisoformat(t.expiration) < today
        except ValueError:
            continue
        if not expired:
            continue
        contracts = max(1, int(getattr(t, "contracts", 1) or 1))
        held = int(shares_by_ticker.get(t.ticker, 0))
        covered = held >= 100 * contracts
        if t.strategy == CSP_STRATEGY:
            signal = EXIT_ASSIGNED if covered else EXIT_EXPIRED_WORTHLESS
        elif t.strategy == CC_STRATEGY:
            signal = EXIT_EXPIRED_WORTHLESS if covered else EXIT_CALLED_AWAY
        else:                                           # pragma: no cover — guarded above
            continue
        premium = round(t.credit * 100 * contracts, 2)
        out.append(WheelResolution(
            ticker=t.ticker, strategy=t.strategy, expiration=t.expiration,
            contracts=contracts, credit=t.credit, exit_signal=signal,
            realized_pl=premium,
            reason=(f"{t.strategy} expired {t.expiration}: {signal} "
                    f"({held} shares held); premium kept ${premium:.2f}"),
        ))
    return out


def reconcile(*, journal_reader: Any, positions_provider: Any, journal_kb: Any,
              today: date, sentinel: Optional[Path] = None) -> List[WheelResolution]:
    """Run at most once per calendar day (``sentinel`` holds the last run
    date; atomic temp+rename). Reads open trades, current share holdings,
    journals one ``closed`` row per resolution, returns them."""
    if sentinel is not None:
        try:
            if sentinel.read_text().strip() == today.isoformat():
                return []
        except OSError:
            pass
    holdings = positions_provider.snapshot()
    if getattr(positions_provider, "last_fetch_ok", True) is False:
        # Unknown holdings would book every expired put as worthless.
        # Leave the sentinel unwritten so the next cycle retries.
        logger.warning("Wheel reconcile skipped — broker holdings unavailable.")
        return []
    shares: Dict[str, int] = {}
    for p in holdings:
        if getattr(p, "kind", "") == "stock":
            shares[p.ticker] = shares.get(p.ticker, 0) + int(p.qty)
    resolutions = resolve_expired_wheel_trades(journal_reader.open_trades(), shares, today)
    for r in resolutions:
        journal_kb.log_signal(ticker=r.ticker, action="closed", price=0.0,
                              raw_signal=r.to_raw_signal(),
                              notes=f"closed: {r.strategy}, P&L=${r.realized_pl:.2f}, "
                                    f"{r.exit_signal}")
        logger.info("[%s] Wheel reconcile: %s", r.ticker, r.reason)
    if sentinel is not None:
        tmp = sentinel.with_suffix(sentinel.suffix + ".tmp")
        tmp.write_text(today.isoformat())
        tmp.replace(sentinel)
    return resolutions
