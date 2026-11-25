"""Same-day momentum debit playbook (skill 72, 2026-11-24).

Research: docs/plans/research_credit_vol.md §11 — on the agent's own
intraday 0DTE SPY quotes (2026-11-09 … 11-23) a ≈ATM $5 debit vertical
bought in the direction of the move since the open made +$1,051 over 42
trades while the same trade against the move lost $1,272; on two years of
hourly bars (model-priced) it paid +0.19 per $ on days with VIX ≥ 25 and
lost on calm days. Opened and closed within the session, so it runs
outside the market-state gate and the HighIV block; the trading halt
still applies.

Pure helpers here; ``TradingAgent._stage_same_day`` wires them to the
broker, ``PositionMonitor`` owns the exits. State (the 09:35 reference
price and the entry slots already used today) lives in a small JSON
sentinel written with temp + rename.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

PLAYBOOK = "same_day_momentum"
REF_TIME_ET = dtime(9, 35)            # the move is measured from the first price at/after this
STATE_PATH = Path(os.environ.get("TRADING_AGENT_SAME_DAY", "trade_journal/same_day_state.json"))
_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def valid_hhmm(text) -> bool:
    return isinstance(text, str) and bool(_HHMM.match(text))


def parse_hhmm(text: str) -> dtime:
    h, m = text.split(":")
    return dtime(int(h), int(m))


@dataclass
class SameDayState:
    day: str                                   # ET date, ISO
    open_ref: Optional[float] = None           # first price at/after 09:35 ET
    slots_used: List[str] = field(default_factory=list)


def load_state(today: date, path: Optional[Path] = None) -> SameDayState:
    """Today's state; a missing, unreadable or stale file starts fresh."""
    p = path or STATE_PATH
    try:
        raw = json.loads(p.read_text())
        if raw.get("day") == today.isoformat():
            return SameDayState(day=raw["day"], open_ref=raw.get("open_ref"),
                                slots_used=list(raw.get("slots_used") or []))
    except FileNotFoundError:
        pass
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("same-day state unreadable (%s) — starting fresh", exc)
    return SameDayState(day=today.isoformat())


def save_state(state: SameDayState, path: Optional[Path] = None) -> None:
    p = path or STATE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(state)))
    tmp.replace(p)


def due_slot(now_et: datetime, entry_times: Sequence[str], window_min: int,
             used: Sequence[str]) -> Optional[str]:
    """The entry time whose window [t, t + window_min) contains ``now_et``
    and that has not been used today, else None."""
    t = now_et.time()
    for slot in entry_times:
        start = parse_hhmm(slot)
        end = (datetime.combine(now_et.date(), start) + timedelta(minutes=int(window_min))).time()
        if slot not in used and start <= t < end:
            return slot
    return None


def direction(spot: float, ref: Optional[float]) -> Optional[str]:
    """"call" when price is above the 09:35 reference, "put" when below,
    None without a reference or on an exactly flat tape."""
    if not ref or not spot or spot == ref:
        return None
    return "call" if spot > ref else "put"


def prior_vix_close(today: date,
                    history: Optional[Callable[[], Dict[date, float]]] = None) -> Optional[float]:
    """Last VIX close strictly before ``today`` (ET). ``history`` returns
    date → close (defaults to yfinance); any failure returns None, which
    keeps the playbook flat for the slot."""
    try:
        closes = (history or _yf_vix_closes)()
        prior = [d for d in closes if d < today]
        return float(closes[max(prior)]) if prior else None
    except Exception as exc:                                  # noqa: BLE001, skill-34-exempt — no VIX reading = no entry this slot
        logger.warning("same-day: prior VIX close unavailable (%s)", exc)
        return None


def _yf_vix_closes() -> Dict[date, float]:
    import yfinance as yf
    h = yf.Ticker("^VIX").history(period="10d", auto_adjust=False)
    return {ts.date(): float(v) for ts, v in h["Close"].dropna().items()}


def implied_spot(calls: List[Dict], puts: List[Dict]) -> Optional[float]:
    """Median K + C − P (mids) over strikes where the call |Δ| is in
    (0.1, 0.9). The quote feed's last price can be the prior close in the
    first minutes after a gap (2026-11-23 09:32: 689.81 logged, options
    priced ~721); put–call parity on the live option quotes cannot."""
    by_k = {float(p["strike"]): p for p in puts if p.get("strike") is not None}
    xs = []
    for c in calls:
        p = by_k.get(float(c.get("strike") or -1))
        d = abs(float(c.get("delta") or 0.0))
        if p is None or not 0.1 < d < 0.9:
            continue
        if min(float(c.get("bid") or 0), float(p.get("bid") or 0)) <= 0:
            continue
        xs.append(float(c["strike"]) + (float(c["bid"]) + float(c["ask"])) / 2
                  - (float(p["bid"]) + float(p["ask"])) / 2)
    xs.sort()
    return xs[len(xs) // 2] if xs else None


def _leg_ok(c: Dict, max_cents: float, max_pct: float) -> bool:
    """Liquidity gate as the scanner's: reject only when the bid/ask is
    wider than BOTH ``max_cents`` and ``max_pct`` of mid."""
    bid, ask = float(c.get("bid") or 0), float(c.get("ask") or 0)
    if bid <= 0 or ask <= 0 or ask < bid:
        return False
    w, mid = ask - bid, (ask + bid) / 2
    return not (w > max_cents and w > max_pct * mid)


def build_plan(*, ticker: str, expiration: str, side: str, spot: float,
               contracts: List[Dict], width: float, regime: str = "",
               max_leg_cents: float = 0.15, max_leg_pct: float = 0.05):
    """0DTE debit vertical: buy the strike nearest ``spot``, sell the strike
    nearest ``width`` dollars further out. ``max_debit`` is the natural
    price at planning (the executor's live recheck keeps the fill inside
    it). Returns a SpreadPlan, ``valid=False`` with a reason when the
    chain cannot supply it."""
    from trading_agent.debit_policy import KIND_TO_STRATEGY
    from trading_agent.strategy import SpreadLeg, SpreadPlan
    name = KIND_TO_STRATEGY["call_debit" if side == "call" else "put_debit"]
    sign = 1.0 if side == "call" else -1.0

    def empty(reason: str):
        return SpreadPlan(ticker=ticker, strategy_name=name, regime=regime, legs=[],
                          spread_width=0, net_credit=0, max_loss=0, credit_to_width_ratio=0,
                          expiration=expiration, reasoning=reason, valid=False,
                          rejection_reason=reason, playbook=PLAYBOOK)

    chain = [c for c in contracts if c.get("strike") is not None and c.get("symbol")]
    if not chain or spot <= 0:
        return empty("same-day: no 0DTE chain")
    long_c = min(chain, key=lambda c: abs(float(c["strike"]) - spot))
    k_long = float(long_c["strike"])
    outs = [c for c in chain if sign * (float(c["strike"]) - k_long) > 0]
    if not outs:
        return empty("same-day: no strike beyond the long leg")
    short_c = min(outs, key=lambda c: abs(float(c["strike"]) - (k_long + sign * width)))
    if not (_leg_ok(long_c, max_leg_cents, max_leg_pct) and _leg_ok(short_c, max_leg_cents, max_leg_pct)):
        return empty("same-day: leg_spread_wide")
    w = abs(float(short_c["strike"]) - k_long)
    natural = round(float(long_c["ask"]) - float(short_c["bid"]), 2)
    mid = round((float(long_c["bid"]) + float(long_c["ask"])) / 2
                - (float(short_c["bid"]) + float(short_c["ask"])) / 2, 2)
    if natural <= 0 or mid <= 0 or natural >= w:
        return empty(f"same-day: unusable debit (natural {natural:.2f}, width {w:g})")

    def leg(c, action):
        bid, ask = float(c["bid"]), float(c["ask"])
        return SpreadLeg(symbol=c["symbol"], strike=float(c["strike"]), action=action,
                         option_type=side, delta=float(c.get("delta") or 0.0), theta=0.0,
                         bid=bid, ask=ask, mid=round((bid + ask) / 2, 4))

    reasoning = (f"Same-day momentum {name} on {ticker}: price {spot:.2f} "
                 f"{'above' if side == 'call' else 'below'} the 09:35 reference; buy {k_long:g} / "
                 f"sell {float(short_c['strike']):g} {side}s exp {expiration}, debit "
                 f"${natural:.2f} (mid ${mid:.2f}), width ${w:g}.")
    return SpreadPlan(
        ticker=ticker, strategy_name=name, regime=regime,
        legs=[leg(long_c, "buy"), leg(short_c, "sell")],
        spread_width=w, net_credit=-natural, max_loss=round(natural * 100, 2),
        credit_to_width_ratio=round(-natural / w, 4), expiration=expiration,
        reasoning=reasoning, max_debit=natural, playbook=PLAYBOOK)


def trade_risk_pct(plan, balance: float, max_risk_pct: float, max_contracts: int) -> float:
    """Per-trade risk fraction that sizes ``plan`` to at most
    ``max_contracts`` (10 % headroom for the live re-quote) and never
    above ``max_risk_pct``."""
    if balance <= 0 or not plan.max_loss:
        return 0.0
    cap = plan.max_loss * max(1, int(max_contracts)) * 1.10 / balance
    return min(float(max_risk_pct), cap)


CLOSE_MARGIN_MIN = 35                 # 15:25 before a 16:00 close; 12:25 before a 13:00 one


def effective_close(day: date, close_et: str) -> Optional[dtime]:
    """The forced-close time for ``day``: ``close_et``, or
    ``CLOSE_MARGIN_MIN`` before an earlier NYSE close (the Friday after
    Thanksgiving closes at 13:00). A calendar failure keeps ``close_et``."""
    if not valid_hhmm(close_et):
        return None
    cfg = parse_hhmm(close_et)
    try:
        from trading_agent.calendar_utils import session_close_et
        sess = session_close_et(day)
    except Exception as exc:                                  # noqa: BLE001, skill-34-exempt — fall back to the configured time
        logger.warning("same-day: session close unavailable (%s) — using %s", exc, close_et)
        return cfg
    if sess is None:
        return cfg
    early = (datetime.combine(day, sess) - timedelta(minutes=CLOSE_MARGIN_MIN)).time()
    return min(cfg, early)


def past_close(now_et: datetime, close_et: str) -> bool:
    cut = effective_close(now_et.date(), close_et)
    return cut is not None and now_et.time() >= cut
