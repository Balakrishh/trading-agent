"""News, analyst research and sentiment for the data server. Skill 47 §3.5.

Added 2026-10-26 so every data point an agent, the MCP surface or a
script needs comes from one local server: Schwab market data (price,
chains, quotes, fundamentals) *and* the research inputs that until now
lived only inside the quality-at-lows scan (skill 67):

* ``news(ticker)``      — Alpaca news + Yahoo headline RSS, de-duplicated,
  spread across days (``news_aggregator.ticker_news``);
* ``research(ticker)``  — analyst estimates and revisions (Yahoo, cached
  ``CACHE_MAX_AGE_H`` hours in ``analyst_estimates.json``) plus the
  ticker's latest quality-at-lows row (stage, drop cause, Claude label);
* ``sentiment(ticker)`` — one roll-up of the stored readings: Claude's
  drop-cause label, analyst stance (recommendation drift, revision
  trend, target upside) and news flow (last 7 days vs the 23 before).

No LLM runs here. The Claude label is the one the 15:40 ET scan wrote;
an on-demand call would take ~1 minute and spend Claude usage per
request. Every source is optional: a failed source comes back as None
with a note, never as an error for the whole response.

Read-only: imports nothing that writes the journal or places orders.
The analyst cache file is the only thing written (by
``analyst_estimates.get_estimates``, as in the scan).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

NEWS_DAYS = 30
NEWS_MAX_ITEMS = 60
NEWS_TTL_SEC = 600               # Alpaca + Yahoo per ticker; 10 min is fresh enough
SUMMARY_CHARS = 400
RECENT_NEWS_DAYS = 7
# Recommendation score (1 strong buy … 5 strong sell) change over 3 months
# that counts as a real shift rather than noise.
REC_SHIFT = 0.15


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def news_rows(items: List[Any]) -> List[Dict[str, Any]]:
    """``NewsItem`` objects as JSON rows, newest first."""
    return [{
        "published_at": _iso(i.published_at),
        "source": i.source,
        "publisher": i.author or None,
        "title": i.title,
        "summary": (i.body or "")[:SUMMARY_CHARS] or None,
        "url": i.url or None,
    } for i in items]


def latest_quality_row(ticker: str, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The newest quality-at-lows row for ``ticker`` (None when the scan
    has never recorded it or the file is missing)."""
    if path is None:
        from trading_agent.quality_lows import STORE_PATH
        path = STORE_PATH
    best = None
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if f'"{ticker}"' not in line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("ticker") == ticker and (
                        best is None or str(row.get("as_of_utc", "")) >= str(best.get("as_of_utc", ""))):
                    best = row
    except OSError:
        return None
    return best


def news_flow(rows: List[Dict[str, Any]], now: datetime,
              recent_days: int = RECENT_NEWS_DAYS, window_days: int = NEWS_DAYS) -> Dict[str, Any]:
    """Items in the last ``recent_days`` vs the rest of the window, as a
    daily rate ratio (> 1 = news is picking up)."""
    recent = older = 0
    cut = now - timedelta(days=recent_days)
    for r in rows:
        try:
            ts = datetime.fromisoformat(r["published_at"])
        except (KeyError, TypeError, ValueError):
            continue
        if ts >= cut:
            recent += 1
        else:
            older += 1
    older_days = max(window_days - recent_days, 1)
    ratio = None
    if older:
        ratio = round((recent / recent_days) / (older / older_days), 2)
    return {"items_last_7d": recent, "items_prior_23d": older, "rate_ratio": ratio,
            "capped": len(rows) >= NEWS_MAX_ITEMS}


def analyst_stance(est: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Direction of the analyst signals in ``est`` (an
    ``analyst_estimates.summarize`` dict)."""
    if not est:
        return None
    now, ago = est.get("rec_score_now"), est.get("rec_score_3m_ago")
    rec = None
    if now is not None and ago is not None:
        # Lower score = more bullish, so a falling score is an upgrade.
        rec = ("upgrading" if ago - now >= REC_SHIFT
               else "downgrading" if now - ago >= REC_SHIFT else "steady")
    return {
        "revision_trend": est.get("revision_trend"),
        "revisions_up_30d": est.get("revisions_up_30d"),
        "revisions_down_30d": est.get("revisions_down_30d"),
        "rec_score_now": now,
        "rec_score_3m_ago": ago,
        "recommendation_drift": rec,
        "target_upside_pct": est.get("target_upside_pct"),
    }


@dataclass
class ResearchSources:
    """Injectable fetchers (tests hand in stubs)."""
    fetch_news: Callable[[str, int, int], List[Any]]
    fetch_estimates: Callable[[str], Optional[Dict[str, Any]]]
    quality_row: Callable[[str], Optional[Dict[str, Any]]]
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)


def default_sources() -> ResearchSources:
    def _news(t: str, days: int, limit: int) -> List[Any]:
        from trading_agent.news_aggregator import ticker_news
        return ticker_news(t, days=days, max_items=limit)

    def _est(t: str) -> Optional[Dict[str, Any]]:
        from trading_agent.analyst_estimates import get_estimates
        return get_estimates(t)

    return ResearchSources(fetch_news=_news, fetch_estimates=_est,
                           quality_row=latest_quality_row)


def _safe(label: str, fn: Callable[[], Any], notes: List[str]) -> Any:
    try:
        return fn()
    except Exception as exc:                            # noqa: BLE001 — one source must not sink the rest
        logger.warning("research source %s failed: %s", label, exc)
        notes.append(f"{label} unavailable: {str(exc)[:120]}")
        return None


def research(ticker: str, src: ResearchSources) -> Dict[str, Any]:
    notes: List[str] = []
    est = _safe("analyst_estimates", lambda: src.fetch_estimates(ticker), notes)
    row = _safe("quality_lows", lambda: src.quality_row(ticker), notes)
    return {
        "ticker": ticker,
        "analyst_estimates": est,
        "quality_lows": row,
        "notes": notes,
        "as_of": src.now().isoformat(),
    }


def sentiment(ticker: str, news: Optional[List[Dict[str, Any]]], src: ResearchSources,
              notes: Optional[List[str]] = None) -> Dict[str, Any]:
    """Roll-up of the stored readings; ``news`` is the /news rows (None
    when the fetch failed)."""
    notes = list(notes or [])
    est = _safe("analyst_estimates", lambda: src.fetch_estimates(ticker), notes)
    row = _safe("quality_lows", lambda: src.quality_row(ticker), notes) or {}
    llm = row.get("llm_cause")
    label = None
    if isinstance(llm, dict) and llm.get("label"):
        label = {k: llm.get(k) for k in ("label", "confidence", "summary", "model", "news_from")}
        label["scan_date"] = row.get("date")
    elif row:
        notes.append("no Claude label: the scan only labels names at watch or later")
    now = src.now()
    return {
        "ticker": ticker,
        "drop_cause_llm": label,
        "drop_cause_rule": row.get("cause") if row else None,
        "quality_lows_stage": row.get("stage") if row else None,
        "analyst": analyst_stance(est),
        "news_flow": news_flow(news, now) if news is not None else None,
        "latest_headlines": [n["title"] for n in (news or [])[:5]],
        "notes": notes,
        "as_of": now.isoformat(),
    }
