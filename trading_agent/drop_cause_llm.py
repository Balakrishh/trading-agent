"""drop_cause_llm.py — "why did it drop" label from news, via the ``claude`` CLI.

Skill 67 §2.4 (backlog 12.3 / I3, 2026-10-26). The quality-at-lows shadow
scan already assigns a rule-based ``cause`` from numbers (growth intact?
sector ETF down as much?). This adds the reading of the story: for every
name at ``watch`` or later, 30 days of headlines and summaries (Alpaca
news + Yahoo RSS, ``news_aggregator.ticker_news``) plus the row's own
numbers go to Claude, which answers ``sentiment`` | ``sector`` |
``fundamental`` | ``unclear`` and cites the evidence it used.

* Runs through the operator's logged-in ``claude`` CLI like the daily
  reviewer (``claude_reviewer``): no API key, no tools, no MCP servers, no
  settings files, no saved session — one stateless call per ticker.
* **Made-up claims are checked in code**, not by a second model: every
  news item has an id (``N1`` …) and the row's numbers are ``DATA``.
  Citations of ids that were not sent are dropped and listed under
  ``unsupported_citations``; a non-``unclear`` label left with no valid
  citation becomes ``unclear``.
* The label is journalled next to the rule-based cause (``llm_cause`` on
  the row). It does not change the stage: the two are compared over the
  shadow weeks first (skill 67 §5).

Never trades: imports nothing that can place an order.

    python -m trading_agent.drop_cause_llm NFLX      # needs a quality-lows row for NFLX
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from trading_agent.claude_reviewer import claude_bin, parse_result_json

log = logging.getLogger(__name__)

LABELS = ("sentiment", "sector", "fundamental", "unclear")
WORKDIR = Path(os.environ.get("TRADING_AGENT_DROP_CAUSE_WORKDIR",
                              str(Path.home() / ".trading-agent-drop-cause")))
# Sonnet by default: a short classification, and it keeps the operator's
# subscription quota for the reviewer. "" = the CLI's default model.
MODEL = os.environ.get("TRADING_AGENT_DROP_CAUSE_MODEL", "sonnet")
TIMEOUT_S = 180
NEWS_DAYS = 30
MAX_NEWS = 60
SUMMARY_CHARS = 300
MAX_EVIDENCE = 12

SYSTEM_PROMPT = (
    "You classify why a stock trades near its 52-week low, for a long-term "
    "investor who buys strong companies that were sold on sentiment. You get "
    "JSON: the stock's price position, the sector ETF's drawdown, revenue "
    "growth and margins, analyst estimate revisions and price targets "
    "(together id DATA), a rule-based label computed from those numbers, "
    "and news items with ids N1, N2, ….\n\n"
    "Labels:\n"
    "- fundamental: the business got worse — guidance or estimates cut, "
    "revenue or margins falling, lost customers or market share, a "
    "structural threat, accounting or legal damage.\n"
    "- sector: the whole sector or market fell for a shared reason "
    "(rates, regulation, a sector-wide demand cycle) and this name with it.\n"
    "- sentiment: the business is intact (results and estimates holding) "
    "but the price fell on fear, positioning, a narrative, a one-off, "
    "valuation de-rating or a reaction out of proportion to the news.\n"
    "- unclear: the evidence is thin or points both ways.\n\n"
    "Rules: use only the JSON you are given — no outside knowledge of "
    "events. Cite every claim with ids from the input (N… or DATA); never "
    "invent an id or a fact. Prefer unclear over a guess. Answer with one "
    "JSON object and nothing else:\n"
    '{"label": "sentiment|sector|fundamental|unclear", "confidence": 0.0-1.0, '
    '"summary": "at most two sentences", '
    '"evidence": [{"id": "N3", "claim": "what this item shows"}], '
    '"risks": ["what would make it fundamental"], '
    '"catalysts": ["what could end the drop"]}'
)


def _news_payload(items) -> List[Dict[str, Any]]:
    out = []
    for n, it in enumerate(items, 1):
        out.append({
            "id": f"N{n}",
            "date": it.published_at.date().isoformat() if it.published_at else None,
            "source": it.author or it.source,
            "headline": it.title,
            "summary": (it.body or "")[:SUMMARY_CHARS] or None,
        })
    return out


def build_prompt(row: Dict[str, Any], news: List[Dict[str, Any]]) -> str:
    """The user prompt: the row's numbers (DATA) and the numbered news."""
    p = row.get("price") or {}
    data = {
        "ticker": row.get("ticker"), "date": row.get("date"),
        "price": {k: p.get(k) for k in ("close", "high_52w", "low_52w", "drawdown_pct",
                                        "range_pos", "sessions_since_low", "rsi14")},
        "sector_etf": row.get("sector_etf"),
        "sector_etf_drawdown_pct": row.get("sector_drawdown_pct"),
        "fundamentals": row.get("fundamentals"),
        "analyst_estimates": row.get("estimates"),
        "rule_based_label": row.get("cause"),
        "rule_reasons": [r for r in (row.get("reasons") or []) if not str(r).startswith("missing:")],
    }
    return json.dumps({"DATA": data, "news": news}, default=str)


def build_cmd(prompt: str, model: str = MODEL) -> List[str]:
    cmd = [claude_bin(), "-p", prompt, "--output-format", "json",
           "--tools", "", "--strict-mcp-config", "--setting-sources", "",
           "--no-session-persistence", "--system-prompt", SYSTEM_PROMPT]
    if model:
        cmd += ["--model", model]
    return cmd


def verify(answer: Dict[str, Any], ids: set) -> Dict[str, Any]:
    """Normalise Claude's answer and drop citations of ids it was not given.
    A label other than ``unclear`` needs at least one valid citation."""
    label = str(answer.get("label") or "").strip().lower()
    flags: List[str] = []
    if label not in LABELS:
        flags.append(f"bad_label:{label or '-'}")
        label = "unclear"
    evidence, unsupported = [], []
    for e in (answer.get("evidence") or [])[:MAX_EVIDENCE]:
        if not isinstance(e, dict):
            continue
        eid = str(e.get("id") or "").strip().upper()
        (evidence if eid in ids else unsupported).append(
            {"id": eid, "claim": str(e.get("claim") or "")[:300]})
    if label != "unclear" and not evidence:
        flags.append(f"no_valid_evidence_for:{label}")
        label = "unclear"
    try:
        conf = max(0.0, min(1.0, float(answer.get("confidence"))))
    except (TypeError, ValueError):
        conf = None
    return {
        "label": label, "confidence": conf,
        "summary": str(answer.get("summary") or "")[:600],
        "evidence": evidence, "unsupported_citations": unsupported,
        "risks": [str(x)[:200] for x in (answer.get("risks") or [])][:5],
        "catalysts": [str(x)[:200] for x in (answer.get("catalysts") or [])][:5],
        "flags": flags,
    }


def classify(row: Dict[str, Any], news_items=None,
             news_fn: Optional[Callable[[str], list]] = None,
             runner: Callable[..., Any] = subprocess.run,
             model: str = MODEL) -> Dict[str, Any]:
    """One ticker's label. Never raises: failures come back as
    ``{"label": None, "error": …}`` so the scan row still gets written."""
    ticker = row.get("ticker")
    if news_items is None:
        if news_fn is None:
            from trading_agent.news_aggregator import ticker_news
            news_fn = lambda t: ticker_news(t, days=NEWS_DAYS, max_items=MAX_NEWS)  # noqa: E731
        try:
            news_items = news_fn(ticker)
        except Exception as exc:                       # noqa: BLE001 — classify on numbers alone
            log.warning("drop cause: news for %s failed (%s)", ticker, exc)
            news_items = []
    news = _news_payload(news_items)
    base = {"news_count": len(news), "model": model or "default",
            "news_from": news[-1]["date"] if news else None}
    WORKDIR.mkdir(parents=True, exist_ok=True)
    try:
        proc = runner(build_cmd(build_prompt(row, news), model), cwd=str(WORKDIR),
                      capture_output=True, text=True, timeout=TIMEOUT_S)
        out = json.loads(proc.stdout or "")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {**base, "label": None, "error": f"claude CLI failed: {exc}"[:300]}
    if out.get("is_error"):
        return {**base, "label": None, "error": f"claude CLI error: {str(out.get('result'))[:300]}"}
    parsed = parse_result_json(str(out.get("result") or ""))
    if parsed is None:
        return {**base, "label": None, "error": f"unparseable answer: {str(out.get('result'))[:300]}"}
    ids = {n["id"] for n in news} | {"DATA"}
    usage = out.get("usage") or {}
    return {**base, **verify(parsed, ids),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cost_usd_equivalent": out.get("total_cost_usd")}


def main(argv=None) -> int:
    """Label one ticker from its latest quality-lows row; print, write nothing."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m trading_agent.drop_cause_llm TICKER")
        return 2
    from dotenv import load_dotenv
    from trading_agent.quality_lows import STORE_PATH
    load_dotenv(os.path.join(os.getcwd(), ".env"))
    t = argv[0].upper()
    row = None
    try:
        with open(STORE_PATH, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("ticker") == t and r.get("stage") != "not_near_low":
                    row = r
    except (OSError, ValueError):
        pass
    if row is None:
        print(f"no quality-lows row past not_near_low for {t}")
        return 1
    print(json.dumps(classify(row), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
