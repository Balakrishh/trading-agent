"""fundamentals_screen.py — quality gate for Wheel candidates (skill 40 §2.7).

The Wheel sells cash-secured puts on stocks the operator is willing to
own. This module decides "willing to own" from the skill-47
``/fundamentals/{ticker}`` block before any option chain is fetched.

It is a pass/fail filter, not a scorer: ranking stays in
``decision_engine._score_cash_secured_put`` (CI invariant 2).

Missing-data discipline (sentinel pattern): a field that is absent,
``None`` or non-numeric is *unknown*, and unknown fails the screen with a
``missing:<field>`` reason. Volume fields reading ``0.0`` are also treated
as unknown — Schwab returned 0.0 for unmapped volume on 2026-09-29, and a
liquid large cap never truly averages zero shares.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional


@dataclass(frozen=True)
class WheelScreenConfig:
    """Thresholds for the quality screen. Percent fields use Schwab's
    units (27.6 means 27.6 %)."""
    min_market_cap: float = 10e9          # large caps only
    max_pe_ratio: float = 40.0            # 0 < P/E ≤ this (negative P/E = losses)
    min_eps_ttm: float = 0.0              # strictly profitable
    min_net_margin_pct: float = 8.0
    min_roe_pct: float = 10.0
    max_beta: float = 1.6
    min_avg_volume_10d: float = 1_000_000  # options liquidity proxy


@dataclass(frozen=True)
class ScreenResult:
    ticker: str
    passed: bool
    reasons: List[str] = field(default_factory=list)   # empty when passed


def _num(block: Mapping[str, Any], key: str, *, zero_is_missing: bool = False) -> Optional[float]:
    v = block.get(key)
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if zero_is_missing and f == 0.0:
        return None
    return f


def screen_fundamentals(ticker: str, block: Optional[Mapping[str, Any]],
                        cfg: WheelScreenConfig = WheelScreenConfig()) -> ScreenResult:
    """Return pass/fail plus every failing reason (not just the first), so
    the operator sees the whole picture for a rejected ticker."""
    if not block:
        return ScreenResult(ticker, False, ["missing:fundamentals"])

    checks = (
        ("market_cap", _num(block, "market_cap"),
         lambda v: v >= cfg.min_market_cap, f"market_cap<{cfg.min_market_cap:.0e}"),
        ("pe_ratio", _num(block, "pe_ratio"),
         lambda v: 0 < v <= cfg.max_pe_ratio, f"pe_ratio∉(0,{cfg.max_pe_ratio:g}]"),
        ("eps_ttm", _num(block, "eps_ttm"),
         lambda v: v > cfg.min_eps_ttm, f"eps_ttm≤{cfg.min_eps_ttm:g}"),
        ("net_profit_margin_ttm", _num(block, "net_profit_margin_ttm"),
         lambda v: v >= cfg.min_net_margin_pct, f"net_margin<{cfg.min_net_margin_pct:g}%"),
        ("roe", _num(block, "roe"),
         lambda v: v >= cfg.min_roe_pct, f"roe<{cfg.min_roe_pct:g}%"),
        ("beta", _num(block, "beta"),
         lambda v: v <= cfg.max_beta, f"beta>{cfg.max_beta:g}"),
        ("vol_avg_10d", _num(block, "vol_avg_10d", zero_is_missing=True),
         lambda v: v >= cfg.min_avg_volume_10d, f"vol_avg_10d<{cfg.min_avg_volume_10d:.0e}"),
    )
    reasons = []
    for name, value, ok, label in checks:
        if value is None:
            reasons.append(f"missing:{name}")
        elif not ok(value):
            reasons.append(label)
    return ScreenResult(ticker, not reasons, reasons)
