"""News-reading drop-cause label (skill 67 §2.4) and its news sources."""
import ast
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from trading_agent import drop_cause_llm as dc
from trading_agent import news_aggregator as na
from trading_agent import quality_lows as ql

NOW = datetime.now(timezone.utc)


def _item(title, days_ago=0, body="", source="alpaca_news"):
    return na.NewsItem(source=source, ticker="NFLX", title=title, body=body,
                       published_at=NOW - timedelta(days=days_ago), author="benzinga")


def _row(**kw):
    row = {"ticker": "NFLX", "date": "2026-10-26", "stage": "watch", "quality_ok": True,
           "cause": "sentiment", "sector_etf": "XLC", "sector_drawdown_pct": -12.4,
           "price": {"close": 90.0, "high_52w": 160.0, "low_52w": 85.0, "drawdown_pct": -43.6,
                     "range_pos": 0.07, "sessions_since_low": 4, "rsi14": 38.0},
           "fundamentals": {"rev_change_ttm_pct": 14.65}, "estimates": {"revisions_down_30d": 10},
           "reasons": ["missing:eps", "net margin drop"]}
    row.update(kw)
    return row


def _runner(result, is_error=False, usage=None, exc=None, stdout=None):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        if exc:
            raise exc
        out = stdout if stdout is not None else json.dumps(
            {"result": result, "is_error": is_error, "usage": usage or {"output_tokens": 42},
             "total_cost_usd": 0.01})
        return SimpleNamespace(stdout=out, stderr="", returncode=0)
    run.calls = calls
    return run


@pytest.fixture(autouse=True)
def _workdir(tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "WORKDIR", tmp_path / "wd")


# ── verify: the made-up-claim check ───────────────────────────────────

def test_verify_keeps_given_ids_and_drops_invented_ones():
    out = dc.verify({"label": "Sentiment", "confidence": 0.7, "summary": "s",
                     "evidence": [{"id": "n1", "claim": "a"}, {"id": "N9", "claim": "made up"},
                                  {"id": "DATA", "claim": "numbers"}, "junk"]},
                    {"N1", "N2", "DATA"})
    assert out["label"] == "sentiment"
    assert [e["id"] for e in out["evidence"]] == ["N1", "DATA"]
    assert out["unsupported_citations"] == [{"id": "N9", "claim": "made up"}]
    assert out["flags"] == []


def test_verify_label_without_valid_evidence_becomes_unclear():
    out = dc.verify({"label": "fundamental", "evidence": [{"id": "N7", "claim": "x"}]}, {"N1", "DATA"})
    assert out["label"] == "unclear"
    assert out["flags"] == ["no_valid_evidence_for:fundamental"]


def test_verify_unknown_label_and_bad_confidence():
    out = dc.verify({"label": "bullish", "confidence": "high"}, {"DATA"})
    assert out["label"] == "unclear" and out["confidence"] is None
    assert out["flags"] == ["bad_label:bullish"]
    assert dc.verify({"label": "unclear", "confidence": 3}, {"DATA"})["confidence"] == 1.0


def test_verify_caps_evidence_and_lists():
    ev = [{"id": "DATA", "claim": str(i)} for i in range(30)]
    out = dc.verify({"label": "sector", "evidence": ev, "risks": list("abcdefg")}, {"DATA"})
    assert len(out["evidence"]) == dc.MAX_EVIDENCE and len(out["risks"]) == 5


# ── prompt and command ────────────────────────────────────────────────

def test_build_prompt_numbers_news_and_drops_missing_reasons():
    news = dc._news_payload([_item("Beat", 1, body="x" * 500), _item("Downgrade", 3)])
    assert [n["id"] for n in news] == ["N1", "N2"]
    assert len(news[0]["summary"]) == dc.SUMMARY_CHARS and news[1]["summary"] is None
    payload = json.loads(dc.build_prompt(_row(), news))
    assert payload["DATA"]["rule_based_label"] == "sentiment"
    assert payload["DATA"]["rule_reasons"] == ["net margin drop"]
    assert payload["DATA"]["sector_etf_drawdown_pct"] == -12.4
    assert payload["news"][1]["headline"] == "Downgrade"


def test_build_cmd_is_toolless_and_stateless():
    cmd = dc.build_cmd("p", model="sonnet")
    for flag in ("--no-session-persistence", "--strict-mcp-config", "--system-prompt"):
        assert flag in cmd
    assert cmd[cmd.index("--tools") + 1] == "" and cmd[cmd.index("--model") + 1] == "sonnet"
    assert "--model" not in dc.build_cmd("p", model="")


# ── classify ──────────────────────────────────────────────────────────

def test_classify_success_records_counts_and_usage():
    ans = {"label": "sentiment", "confidence": 0.6, "summary": "intact",
           "evidence": [{"id": "N2", "claim": "beat"}, {"id": "N5", "claim": "invented"}]}
    run = _runner(json.dumps(ans))
    out = dc.classify(_row(), news_items=[_item("a", 0), _item("b", 20)], runner=run)
    assert out["label"] == "sentiment" and out["news_count"] == 2
    assert out["news_from"] == (NOW - timedelta(days=20)).date().isoformat()
    assert out["unsupported_citations"][0]["id"] == "N5"
    assert out["output_tokens"] == 42 and out["cost_usd_equivalent"] == 0.01
    assert run.calls[0][1]["cwd"] == str(dc.WORKDIR)


def test_classify_fenced_json_answer():
    run = _runner('Here:\n```json\n{"label": "unclear", "evidence": []}\n```')
    assert dc.classify(_row(), news_items=[], runner=run)["label"] == "unclear"


@pytest.mark.parametrize("kw,needle", [
    ({"result": "boom", "is_error": True}, "claude CLI error"),
    ({"result": "no json here"}, "unparseable"),
    ({"result": "", "stdout": "not json"}, "claude CLI failed"),
    ({"result": "", "exc": subprocess.TimeoutExpired("claude", 1)}, "claude CLI failed"),
    ({"result": "", "exc": FileNotFoundError("claude")}, "claude CLI failed"),
])
def test_classify_failures_return_error_not_raise(kw, needle):
    out = dc.classify(_row(), news_items=[], runner=_runner(**kw))
    assert out["label"] is None and needle in out["error"]


def test_classify_news_failure_falls_back_to_numbers():
    def bad(_t):
        raise RuntimeError("down")
    run = _runner(json.dumps({"label": "unclear"}))
    out = dc.classify(_row(), news_fn=bad, runner=run)
    assert out["news_count"] == 0 and out["label"] == "unclear"


def test_module_never_imports_order_paths():
    tree = ast.parse(Path(dc.__file__).read_text())
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    for bad in ("executor", "pending_orders", "alpaca_client", "order"):
        assert not any(bad in m for m in names), m


# ── news sources ──────────────────────────────────────────────────────

class _Resp:
    def __init__(self, body=None, text="", status=200):
        self._body, self.text, self.status = body, text, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(self.status)

    def json(self):
        return self._body


def test_alpaca_news_pages_filters_roundups_and_parses(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    pages = [
        {"news": [{"headline": "NFLX beat", "summary": "sum", "created_at": "2026-10-20T20:01:00Z",
                   "symbols": ["NFLX"], "source": "benzinga"},
                  {"headline": "5 stocks buzzing", "symbols": list("ABCDE"),
                   "created_at": "2026-10-20T20:01:00Z"}],
         "next_page_token": "t2"},
        {"news": [{"headline": "Downgrade", "created_at": "bad", "symbols": ["NFLX", "WBD"]}],
         "next_page_token": None},
    ]
    seen = []

    class S:
        def get(self, url, params, headers, timeout):
            seen.append(params.get("page_token"))
            return _Resp(pages[len(seen) - 1])
    items = na.fetch_alpaca_news("NFLX", session=S())
    assert seen == [None, "t2"]
    assert [i.title for i in items] == ["NFLX beat", "Downgrade"]
    assert items[0].body == "sum" and items[0].author == "benzinga"
    assert items[0].published_at.tzinfo is not None and items[1].published_at is None


def test_alpaca_news_without_keys_or_on_error(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    assert na.fetch_alpaca_news("NFLX") == []
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")

    class S:
        def get(self, *a, **k):
            return _Resp(status=500)
    assert na.fetch_alpaca_news("NFLX", session=S()) == []


def test_yahoo_rss_parses_and_drops_old_items():
    recent = NOW.strftime("%a, %d %b %Y %H:%M:%S +0000")
    old = (NOW - timedelta(days=60)).strftime("%a, %d %b %Y %H:%M:%S +0000")
    xml = (f"<rss><channel><item><title>New</title><description>d</description><pubDate>{recent}</pubDate>"
           f"<link>u</link></item><item><title>Old</title><pubDate>{old}</pubDate></item>"
           "<item><title></title></item></channel></rss>")

    class S:
        def get(self, *a, **k):
            return _Resp(text=xml)
    items = na.fetch_yahoo_rss("NFLX", session=S())
    assert [i.title for i in items] == ["New"] and items[0].body == "d"

    class Bad:
        def get(self, *a, **k):
            return _Resp(text="<not xml")
    assert na.fetch_yahoo_rss("NFLX", session=Bad()) == []


def test_spread_by_day_keeps_old_days_when_capped():
    items = [_item(f"today {i}", 0) for i in range(10)] + [_item("earnings", 6), _item("layoffs", 12)]
    picked = na.spread_by_day(items, 4)
    titles = [i.title for i in picked]
    assert len(picked) == 4 and "earnings" in titles and "layoffs" in titles
    assert picked[0].published_at >= picked[-1].published_at
    assert na.spread_by_day(items[:3], 10) == sorted(items[:3], key=lambda i: i.published_at, reverse=True)


def test_ticker_news_dedupes_across_sources(monkeypatch):
    monkeypatch.setattr(na, "fetch_alpaca_news", lambda t, days: [_item("Same Title", 1)])
    monkeypatch.setattr(na, "fetch_yahoo_rss", lambda t, days: [_item("same title!", 1, source="yahoo_rss"),
                                                                _item("Other", 2, source="yahoo_rss")])
    assert [i.title for i in na.ticker_news("NFLX")] == ["Same Title", "Other"]


# ── wiring into the scan ──────────────────────────────────────────────

def test_add_llm_labels_order_cap_and_errors():
    rows = [_row(ticker="A", stage="watch"), _row(ticker="B", stage="turn"),
            _row(ticker="C", stage="excluded"), _row(ticker="D", stage="not_near_low"),
            _row(ticker="E", stage="watch", quality_ok=False), _row(ticker="F", stage="basing")]
    asked = []

    def classify(r):
        asked.append(r["ticker"])
        if r["ticker"] == "F":
            raise RuntimeError("cli down")
        return {"label": "sentiment"}
    from dataclasses import replace
    ql.add_llm_labels(rows, classify, replace(ql.DEFAULT_CONFIG, llm_max_names=3))
    assert asked == ["B", "F", "A"]                 # most advanced stage first
    by = {r["ticker"]: r for r in rows}
    assert by["B"]["llm_cause"]["label"] == "sentiment"
    assert by["F"]["llm_cause"] == {"label": None, "error": "cli down"}
    assert by["C"]["llm_cause"]["error"] == "skipped: llm_max_names"   # excluded by rule still eligible
    assert "llm_cause" not in by["D"] and "llm_cause" not in by["E"]
    assert by["B"]["stage"] == "turn"               # journal-only: the stage is untouched


def test_report_shows_llm_label_and_why(tmp_path):
    fp = tmp_path / "ql.jsonl"
    r = _row(llm_cause={"label": "unclear", "confidence": 0.55, "summary": "guidance cut"})
    r2 = _row(ticker="ORCL", llm_cause={"label": None, "error": "x"})
    fp.write_text(json.dumps(r) + "\n" + json.dumps(r2) + "\n")
    out = "\n".join(ql.report(fp))
    assert "llm=unclear(0.55)" in out and "why: guidance cut" in out and "llm=error" in out
