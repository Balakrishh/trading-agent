"""profit_trail.py — trailing profit-taking (2026-10-05).

Instead of closing the first cycle a position reaches its profit target,
the trail *arms* there, remembers the best profit seen, and closes when

* profit reaches the **ceiling** (bank it — little reward left for the risk),
* profit gives back ``trail_giveback_pct`` of the peak, never below the
  **floor** (credit spreads only — the 40 %-of-credit lock-in), or
* the position is within ``trail_max_hold_dte`` days of (near) expiry.

Profit is the natural-price P&L the monitor already uses for its target
(``PositionMonitor._profit_pl``); the agent's 3-cycle exit debounce then
applies to any close the trail asks for, so one bad quote cannot end a
winning trade. Stops are never touched — a stop signal always wins.

``PresetConfig.profit_trail_mode``:

* ``"off"``    — legacy: close at the target.
* ``"shadow"`` — legacy closes stay; the trail is evaluated alongside and,
  once a position closes for real, its legs keep being priced until the
  trail would have closed. Each such outcome is journaled as
  ``profit_trail_shadow`` (actual vs trail P&L) so the rule can be judged
  on paper data before it goes live.
* ``"live"``   — the trail decides profit-taking closes.

Wheel legs are excluded (they keep the 50 %-of-credit rule).
State: ``trade_journal/profit_trail.json`` (atomic temp + rename), keyed by
the position's sorted leg symbols, because the agent restarts every cycle.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TRAIL_MODES = ("off", "shadow", "live")
STATE_PATH = Path(os.environ.get("TRADING_AGENT_PROFIT_TRAIL",
                                 "trade_journal/profit_trail.json"))

# Decisions
HOLD_UNARMED = "hold_unarmed"
HOLD_ARMED = "hold_armed"
CLOSE_CEILING = "close_ceiling"
CLOSE_GIVEBACK = "close_giveback"
CLOSE_TIME = "close_time"
CLOSE_DECISIONS = (CLOSE_CEILING, CLOSE_GIVEBACK, CLOSE_TIME)


@dataclass
class TrailState:
    key: str
    ticker: str
    strategy: str
    kind: str
    expiration: str
    armed: bool = False
    basis: float = 0.0          # profit yardstick at arming ($, position scale)
    target: float = 0.0         # the legacy profit target at arming ($)
    peak_pl: float = 0.0
    armed_at: str = ""
    last_pl: float = 0.0
    # Shadow-mode bookkeeping once the real position has closed.
    legs: List[Dict[str, Any]] = field(default_factory=list)   # symbol, qty, avg_entry
    ghost: bool = False
    actual_exit_pl: Optional[float] = None
    shadow_exit_pl: Optional[float] = None
    shadow_reason: str = ""
    done: bool = False


@dataclass(frozen=True)
class TrailParams:
    giveback_pct: float = 0.25
    credit_floor_pct: float = 0.40
    credit_ceiling_pct: float = 0.75
    debit_ceiling_pct: float = 0.90
    calendar_ceiling_pct: float = 0.40
    max_hold_dte: int = 7

    @classmethod
    def from_preset(cls, preset: Any) -> "TrailParams":
        return cls(
            giveback_pct=float(getattr(preset, "trail_giveback_pct", cls.giveback_pct)),
            credit_floor_pct=float(getattr(preset, "trail_credit_floor_pct", cls.credit_floor_pct)),
            credit_ceiling_pct=float(getattr(preset, "trail_credit_ceiling_pct", cls.credit_ceiling_pct)),
            debit_ceiling_pct=float(getattr(preset, "trail_debit_ceiling_pct", cls.debit_ceiling_pct)),
            calendar_ceiling_pct=float(getattr(preset, "trail_calendar_ceiling_pct",
                                               cls.calendar_ceiling_pct)),
            max_hold_dte=int(getattr(preset, "trail_max_hold_dte", cls.max_hold_dte)),
        )

    def ceiling(self, kind: str, basis: float) -> float:
        pct = {"credit": self.credit_ceiling_pct, "debit_vertical": self.debit_ceiling_pct,
               "calendar": self.calendar_ceiling_pct}.get(kind, 1.0)
        return basis * pct

    def floor(self, kind: str, basis: float) -> float:
        return basis * self.credit_floor_pct if kind == "credit" else 0.0


def position_key(symbols) -> str:
    return "|".join(sorted(str(s) for s in symbols))


def days_to(expiration: str, today: Optional[date] = None) -> Optional[int]:
    try:
        return (date.fromisoformat(expiration) - (today or date.today())).days
    except (TypeError, ValueError):
        return None


def evaluate(state: TrailState, profit: float, *, basis: float, target: float,
             params: TrailParams, dte: Optional[int]) -> Tuple[TrailState, str, str]:
    """Advance ``state`` with this cycle's ``profit`` ($, position scale).
    Returns (state, decision, reason). Pure — no I/O."""
    state.last_pl = round(profit, 2)
    if not state.armed:
        if target <= 0 or profit < target:
            return state, HOLD_UNARMED, ""
        state.armed = True
        state.basis, state.target = basis, target
        state.armed_at = datetime.now(timezone.utc).isoformat()
        state.peak_pl = profit
    state.peak_pl = max(state.peak_pl, profit)
    ceiling = params.ceiling(state.kind, basis)
    if ceiling > 0 and profit >= ceiling:
        return state, CLOSE_CEILING, (f"trail ceiling: profit ${profit:.2f} ≥ "
                                      f"${ceiling:.2f}")
    lock = max(params.floor(state.kind, basis), state.peak_pl * (1.0 - params.giveback_pct))
    if profit <= lock:
        return state, CLOSE_GIVEBACK, (f"trail giveback: profit ${profit:.2f} ≤ lock "
                                       f"${lock:.2f} (peak ${state.peak_pl:.2f})")
    if dte is not None and dte <= params.max_hold_dte:
        return state, CLOSE_TIME, f"trail time stop: {dte} DTE ≤ {params.max_hold_dte}"
    return state, HOLD_ARMED, (f"trail armed: profit ${profit:.2f}, peak "
                               f"${state.peak_pl:.2f}, lock ${lock:.2f}, ceiling ${ceiling:.2f}")


def ghost_profit(legs: List[Dict[str, Any]], quotes: Dict[str, Dict[str, float]]) -> Optional[float]:
    """Natural-price P&L ($) of a closed position's legs at current quotes:
    shorts bought back at the ask, longs sold at the bid. None when a leg
    has no usable quote."""
    total = 0.0
    for leg in legs:
        q = quotes.get(leg["symbol"]) or {}
        bid, ask = float(q.get("bid") or 0), float(q.get("ask") or 0)
        qty, entry = int(leg["qty"]), float(leg["avg_entry"])
        if qty < 0:
            if ask <= 0:
                return None
            total += (entry - ask) * 100 * abs(qty)
        else:
            if bid <= 0 and ask <= 0:
                return None
            total += (bid - entry) * 100 * qty
    return round(total, 2)


def load_states(path: Optional[Path] = None) -> Dict[str, TrailState]:
    try:
        raw = json.loads((path or STATE_PATH).read_text())
    except (OSError, ValueError):
        return {}
    out: Dict[str, TrailState] = {}
    for k, v in raw.items():
        try:
            out[k] = TrailState(**v)
        except TypeError:
            continue
    return out


def save_states(states: Dict[str, TrailState], path: Optional[Path] = None) -> None:
    p = path or STATE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps({k: asdict(v) for k, v in states.items()}, indent=2))
    tmp.replace(p)
