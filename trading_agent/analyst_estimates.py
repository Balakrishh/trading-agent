"""analyst_estimates.py — analyst forecasts and estimate revisions (skill 67).

Added 2026-10-26 for the quality-at-lows sleeve. Schwab's fundamentals
carry trailing revenue growth but no forecasts, and estimate revisions
are the clearest sign that a sell-off is mood rather than business:
analysts who still raise next year's EPS after a 40 % drop think the
story is intact. Yahoo Finance publishes these per ticker with no API
key (``yfinance`` is already a dependency for daily bars):

* ``eps_trend`` — consensus EPS now and 7 / 30 / 60 / 90 days ago, for
  the current quarter (0q), next quarter (+1q), current year (0y) and
  next year (+1y);
* ``eps_revisions`` — analysts revising up / down in the last 7 / 30 days;
* ``revenue_estimate`` / ``earnings_estimate`` — consensus and growth
  vs the year before;
* ``analyst_price_targets`` and ``recommendations`` (strong buy … strong
  sell counts, now and 1–3 months ago).

Yahoo's endpoints are unofficial and can change shape or rate-limit:
every field is optional, a failed table is skipped, and a ticker with
nothing usable returns None. Results are cached in
``trade_journal/analyst_estimates.json`` for ``CACHE_MAX_AGE_H`` hours —
the numbers move at most daily.

    python -m trading_agent.analyst_estimates NFLX ORCL   # print summaries
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

CACHE_PATH = Path(os.environ.get("TRADING_AGENT_ANALYST_CACHE",
                                 "trade_journal/analyst_estimates.json"))
CACHE_MAX_AGE_H = 20
# |30-day change in consensus EPS| below this counts as flat.
REVISION_FLAT_PCT = 1.0
# Recommendation weights: 1 = strong buy … 5 = strong sell.
_REC_WEIGHTS = {"strongBuy": 1, "buy": 2, "hold": 3, "sell": 4, "strongSell": 5}


def _f(v: Any) -> Optional[float]:
    """Float or None (NaN, missing, non-numeric → None)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x


def _cell(table: Any, row: str, col: str) -> Optional[float]:
    """``table.loc[row, col]`` for a pandas DataFrame, or None."""
    try:
        return _f(table.loc[row, col])
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _pct_change(now: Optional[float], then: Optional[float]) -> Optional[float]:
    if now is None or then is None or then == 0:
        return None
    return round((now / then - 1.0) * 100.0, 2)


def _rec_score(table: Any, period: str) -> Optional[float]:
    """Weighted mean recommendation (1 strong buy … 5 strong sell) for one
    ``period`` row ("0m", "-3m") of Yahoo's recommendations table."""
    try:
        row = table[table["period"] == period].iloc[0]
    except (AttributeError, KeyError, IndexError, TypeError):
        return None
    n = s = 0.0
    for col, w in _REC_WEIGHTS.items():
        c = _f(row.get(col)) or 0.0
        n += c
        s += c * w
    return round(s / n, 3) if n else None


def summarize(tables: Dict[str, Any], spot: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Normalise Yahoo's tables into one flat dict; None when nothing is usable.

    ``tables`` maps ``eps_trend``, ``eps_revisions``, ``revenue_estimate``,
    ``earnings_estimate``, ``analyst_price_targets``, ``recommendations``
    to whatever yfinance returned (DataFrame / dict / None).
    Percent fields are percents (12.5 = 12.5 %)."""
    trend = tables.get("eps_trend")
    revs = tables.get("eps_revisions")
    rev_est = tables.get("revenue_estimate")
    earn_est = tables.get("earnings_estimate")
    targets = tables.get("analyst_price_targets") or {}
    recs = tables.get("recommendations")

    out: Dict[str, Any] = {}
    for period, tag in (("0y", "cur_year"), ("+1y", "next_year")):
        now = _cell(trend, period, "current")
        out[f"eps_{tag}"] = now
        out[f"eps_{tag}_rev_30d_pct"] = _pct_change(now, _cell(trend, period, "30daysAgo"))
        out[f"eps_{tag}_rev_90d_pct"] = _pct_change(now, _cell(trend, period, "90daysAgo"))
        out[f"eps_growth_{tag}_pct"] = (None if _cell(earn_est, period, "growth") is None
                                        else round(_cell(earn_est, period, "growth") * 100, 2))
        out[f"rev_growth_{tag}_pct"] = (None if _cell(rev_est, period, "growth") is None
                                        else round(_cell(rev_est, period, "growth") * 100, 2))
    up = down = None
    for period in ("0y", "+1y"):
        u, d = _cell(revs, period, "upLast30days"), _cell(revs, period, "downLast30days")
        if u is not None:
            up = (up or 0) + u
        if d is not None:
            down = (down or 0) + d
    out["revisions_up_30d"] = None if up is None else int(up)
    out["revisions_down_30d"] = None if down is None else int(down)
    out["analysts_cur_year"] = (None if _cell(earn_est, "0y", "numberOfAnalysts") is None
                                else int(_cell(earn_est, "0y", "numberOfAnalysts")))

    mean_t = _f(targets.get("mean")) if isinstance(targets, dict) else None
    cur = spot if spot is not None else (_f(targets.get("current")) if isinstance(targets, dict) else None)
    out["target_mean"] = mean_t
    out["target_upside_pct"] = _pct_change(mean_t, cur)
    out["rec_score_now"] = _rec_score(recs, "0m")
    out["rec_score_3m_ago"] = _rec_score(recs, "-3m")

    if all(v is None for v in out.values()):
        return None
    out["revision_trend"] = revision_trend(out)
    return out


def revision_trend(s: Dict[str, Any]) -> str:
    """"up" | "down" | "flat" | "unknown" from the 30-day change in
    current- and next-year consensus EPS (their mean), with the up/down
    analyst counts as the tie-break when only counts exist."""
    vals = [v for v in (s.get("eps_cur_year_rev_30d_pct"), s.get("eps_next_year_rev_30d_pct"))
            if v is not None]
    if vals:
        m = sum(vals) / len(vals)
        if m >= REVISION_FLAT_PCT:
            return "up"
        if m <= -REVISION_FLAT_PCT:
            return "down"
        return "flat"
    up, down = s.get("revisions_up_30d"), s.get("revisions_down_30d")
    if up is None or down is None:
        return "unknown"
    return "up" if up > down else "down" if down > up else "flat"


_TABLES = ("eps_trend", "eps_revisions", "revenue_estimate", "earnings_estimate",
           "analyst_price_targets", "recommendations")


def fetch_tables(ticker: str, ticker_factory: Optional[Callable[[str], Any]] = None) -> Dict[str, Any]:
    """Yahoo's tables for ``ticker``; a table that fails is None."""
    if ticker_factory is None:
        import yfinance as yf
        ticker_factory = yf.Ticker
    tk = ticker_factory(ticker)
    out: Dict[str, Any] = {}
    for name in _TABLES:
        try:
            out[name] = getattr(tk, name)
        except Exception as exc:                       # noqa: BLE001 — unofficial endpoint; one table must not sink the rest
            logger.debug("[%s] analyst table %s failed: %s", ticker, name, exc)
            out[name] = None
    return out


def get_estimates(ticker: str, *, spot: Optional[float] = None,
                  path: Optional[Path] = None, max_age_hours: float = CACHE_MAX_AGE_H,
                  fetch: Optional[Callable[[str], Dict[str, Any]]] = None,
                  now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """Cached summary for ``ticker`` (refetched when older than
    ``max_age_hours``); None when Yahoo has nothing. Never raises."""
    fp = Path(path or CACHE_PATH)
    now = now or datetime.now(timezone.utc)
    t = ticker.upper()
    try:
        cache = json.loads(fp.read_text())
        if not isinstance(cache, dict):
            cache = {}
    except (OSError, ValueError):
        cache = {}
    entry = cache.get(t) or {}
    try:
        age_h = (now - datetime.fromisoformat(entry["fetched_at"])).total_seconds() / 3600
        if age_h < max_age_hours:
            return entry.get("summary")
    except (KeyError, TypeError, ValueError):
        pass
    try:
        summary = summarize((fetch or fetch_tables)(t), spot)
    except Exception as exc:                           # noqa: BLE001 — unknown estimates are a missing input, not an error
        logger.warning("[%s] analyst estimates lookup failed: %s", t, exc)
        summary = None
    cache[t] = {"summary": summary, "fetched_at": now.isoformat()}
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        tmp = fp.with_suffix(fp.suffix + ".tmp")
        tmp.write_text(json.dumps(cache, indent=2, sort_keys=True))
        tmp.replace(fp)
    except OSError as exc:
        logger.warning("analyst estimates cache write failed: %s", exc)
    return summary


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for t in args or ["SPY"]:
        print(t, json.dumps(summarize(fetch_tables(t.upper())), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
