"""entry_filters.py — trend entry filters for directional trades (skill 65).

Added 2026-10-15. Two checks on a directional plan before it can enter:

* **Relative strength** — the ticker's 20- and 60-day returns minus SPY's,
  averaged, then ranked across the watchlist (percentile 0 = weakest,
  1 = strongest). A bullish trade needs percentile ≥ ``rs_min_percentile``;
  a bearish trade needs ≤ 1 − ``rs_min_percentile``. This replaces "is the
  last 5-minute bar stronger than its anchor" with what traders mean by
  relative strength.
* **Pullback** — a bullish trade needs price above its 50-day SMA (the
  trend's support holds) and no more than ``pullback_max_atr`` × ATR(14)
  above its 20-day SMA (not chasing an extended move). A bearish trade is
  the mirror. The QQQ call debit bought on 2026-10-05 after a run-up and
  stopped on the first dip is the case this targets.

Each filter has a preset mode: "off" (not computed), "shadow" (verdict
journaled, never blocks — the default) or "live" (a failed check blocks
the entry with reason ``trend: …``). Missing data never blocks.

Neutral structures (condors, butterflies, calendars), the mean-reversion
spread and the oversold bounce bull put (counter-trend by design, with its
own price levels) are not filtered. Pure functions — no I/O.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from trading_agent.debit_policy import CALL_DEBIT_STRATEGY, PUT_DEBIT_STRATEGY

BULLISH_STRATEGIES = frozenset({"Bull Put Spread", CALL_DEBIT_STRATEGY})
BEARISH_STRATEGIES = frozenset({"Bear Call Spread", PUT_DEBIT_STRATEGY})
RS_WINDOWS = (20, 60)
RS_BENCHMARK = "SPY"


def direction_of(strategy_name: str) -> Optional[str]:
    """"bull" / "bear" for filtered directional strategies, else None."""
    if strategy_name in BULLISH_STRATEGIES:
        return "bull"
    if strategy_name in BEARISH_STRATEGIES:
        return "bear"
    return None


def _ret(closes: Sequence[float], n: int) -> Optional[float]:
    if len(closes) <= n or not closes[-n - 1]:
        return None
    return closes[-1] / closes[-n - 1] - 1.0


def relative_strength(closes: Sequence[float], bench: Sequence[float],
                      windows: Sequence[int] = RS_WINDOWS) -> Optional[float]:
    """Mean over ``windows`` of (ticker return − benchmark return); None
    when either series is too short for every window."""
    diffs = []
    for n in windows:
        a, b = _ret(closes, n), _ret(bench, n)
        if a is not None and b is not None:
            diffs.append(a - b)
    return sum(diffs) / len(diffs) if diffs else None


def rs_percentiles(scores: Dict[str, Optional[float]]) -> Dict[str, float]:
    """Percentile rank of each known score in [0, 1] (ties share the mean
    rank; a single ticker is 0.5). Unknown scores are left out."""
    known = {t: s for t, s in scores.items() if s is not None}
    if not known:
        return {}
    if len(known) == 1:
        return {t: 0.5 for t in known}
    ordered = sorted(known.values())
    n = len(ordered)
    out = {}
    for t, s in known.items():
        lo = ordered.index(s)
        hi = n - 1 - ordered[::-1].index(s)
        out[t] = ((lo + hi) / 2) / (n - 1)
    return out


def sma(values: Sequence[float], n: int) -> Optional[float]:
    return sum(values[-n:]) / n if len(values) >= n else None


def atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
        n: int = 14) -> Optional[float]:
    """Simple average true range over the last ``n`` bars."""
    m = min(len(highs), len(lows), len(closes))
    if m < n + 1:
        return None
    h, l, c = highs[-m:], lows[-m:], closes[-m:]
    trs = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
           for i in range(m - n, m)]
    return sum(trs) / n


@dataclass(frozen=True)
class FilterVerdict:
    name: str                     # "relative_strength" | "pullback"
    mode: str                     # off | shadow | live
    ok: Optional[bool]            # None = not enough data (never blocks)
    reason: str
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)


def rs_verdict(direction: str, score: Optional[float], pct: Optional[float],
               min_pct: float, mode: str) -> FilterVerdict:
    m = {"rs_score": None if score is None else round(score, 4),
         "rs_percentile": None if pct is None else round(pct, 3)}
    if pct is None:
        return FilterVerdict("relative_strength", mode, None, "no relative-strength data", m)
    if direction == "bull":
        ok = pct >= min_pct
        need = f"≥ {min_pct:.0%}"
    else:
        ok = pct <= 1 - min_pct
        need = f"≤ {1 - min_pct:.0%}"
    word = "strong" if score is not None and score >= 0 else "weak"
    reason = (f"relative strength percentile {pct:.0%} ({word} vs SPY, "
              f"{score:+.1%}) — {direction}ish trade needs {need}")
    return FilterVerdict("relative_strength", mode, ok, reason, m)


def pullback_verdict(direction: str, price: float, closes: Sequence[float],
                     highs: Sequence[float], lows: Sequence[float],
                     max_atr: float, mode: str) -> FilterVerdict:
    s20, s50, a = sma(closes, 20), sma(closes, 50), atr(highs, lows, closes)
    m = {"sma20": None if s20 is None else round(s20, 2),
         "sma50": None if s50 is None else round(s50, 2),
         "atr14": None if a is None else round(a, 2)}
    if s20 is None or s50 is None or not a or not price:
        return FilterVerdict("pullback", mode, None, "not enough daily bars", m)
    stretch = (price - s20) / a if direction == "bull" else (s20 - price) / a
    m["stretch_atr"] = round(stretch, 2)
    if direction == "bull":
        if price <= s50:
            return FilterVerdict("pullback", mode, False,
                                 f"price {price:.2f} below 50-day SMA {s50:.2f} — support broken", m)
        ok = stretch <= max_atr
        side = "above"
    else:
        if price >= s50:
            return FilterVerdict("pullback", mode, False,
                                 f"price {price:.2f} above 50-day SMA {s50:.2f} — resistance broken", m)
        ok = stretch <= max_atr
        side = "below"
    if stretch < 0:
        other = "below" if direction == "bull" else "above"
        reason = f"price {-stretch:.1f} ATR {other} the 20-day SMA {s20:.2f} — pullback"
    else:
        reason = (f"price {stretch:.1f} ATR {side} the 20-day SMA {s20:.2f}"
                  + ("" if ok else f" — extended (max {max_atr:g})"))
    return FilterVerdict("pullback", mode, ok, reason, m)


@dataclass(frozen=True)
class TrendEvaluation:
    direction: str
    verdicts: List[FilterVerdict]

    @property
    def block(self) -> Optional[str]:
        """Reason for the first failed check in live mode, else None."""
        for v in self.verdicts:
            if v.mode == "live" and v.ok is False:
                return f"trend: {v.name} — {v.reason}"
        return None

    def to_journal_dict(self) -> Dict:
        return {"direction": self.direction,
                "blocked": self.block is not None,
                "checks": [asdict(v) for v in self.verdicts]}


def evaluate(strategy_name: str, preset, *, price: float,
             rs: Optional[Dict[str, Optional[float]]] = None,
             bars: Optional[Dict[str, Sequence[float]]] = None) -> Optional[TrendEvaluation]:
    """Run the enabled filters for a plan. ``rs`` = {"score", "percentile"};
    ``bars`` = {"close", "high", "low"} daily lists. None when the strategy
    is not directional or both filters are off."""
    direction = direction_of(strategy_name)
    if direction is None:
        return None
    verdicts: List[FilterVerdict] = []
    rs_mode = str(getattr(preset, "rs_filter_mode", "shadow"))
    if rs_mode != "off":
        rs = rs or {}
        verdicts.append(rs_verdict(direction, rs.get("score"), rs.get("percentile"),
                                   float(getattr(preset, "rs_min_percentile", 0.5)), rs_mode))
    pb_mode = str(getattr(preset, "pullback_filter_mode", "shadow"))
    if pb_mode != "off":
        bars = bars or {}
        verdicts.append(pullback_verdict(
            direction, price, bars.get("close") or [], bars.get("high") or [],
            bars.get("low") or [], float(getattr(preset, "pullback_max_atr", 1.0)), pb_mode))
    return TrendEvaluation(direction, verdicts) if verdicts else None
