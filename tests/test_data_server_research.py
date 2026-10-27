"""Skill 47 §3.5 — /news, /research and /sentiment on the data server,
and the MCP passthroughs (skill 48)."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from trading_agent.data_server import research as R
from trading_agent.news_aggregator import NewsItem

NOW = datetime(2026, 10, 26, 20, 0, tzinfo=timezone.utc)

EST = {"revision_trend": "flat", "revisions_up_30d": 58, "revisions_down_30d": 11,
       "rec_score_now": 1.9, "rec_score_3m_ago": 2.2, "target_upside_pct": 41.0}
ROW = {"date": "2026-10-26", "ticker": "ORCL", "as_of_utc": "2026-10-26T19:41:00+00:00",
       "stage": "watch", "cause": "sentiment",
       "llm_cause": {"label": "sentiment", "confidence": 0.6, "summary": "AI financing fears",
                     "model": "sonnet", "news_from": "2026-09-28", "evidence": ["N1"]}}


def _items(n_recent=3, n_old=2):
    out = [NewsItem(source="alpaca_news", ticker="ORCL", title=f"recent {i}", body="x" * 900,
                    url="u", author="Benzinga", published_at=NOW - timedelta(days=1, hours=i))
           for i in range(n_recent)]
    out += [NewsItem(source="yahoo_rss", ticker="ORCL", title=f"old {i}",
                     published_at=NOW - timedelta(days=15 + i)) for i in range(n_old)]
    return out


class Calls:
    def __init__(self, news=None, est=EST, row=ROW, news_exc=None, est_exc=None):
        self.n = 0
        self._news, self._est, self._row = news, est, row
        self._news_exc, self._est_exc = news_exc, est_exc

    def sources(self):
        def news(t, days, limit):
            self.n += 1
            if self._news_exc:
                raise self._news_exc
            return _items() if self._news is None else self._news

        def est(t):
            if self._est_exc:
                raise self._est_exc
            return self._est

        return R.ResearchSources(fetch_news=news, fetch_estimates=est,
                                 quality_row=lambda t: self._row, now=lambda: NOW)


def _client(calls, api_key=None):
    from fastapi.testclient import TestClient
    from trading_agent.data_server.app import build_app
    from trading_agent.data_server.config import ServerConfig
    return TestClient(build_app(provider=object(), config=ServerConfig(api_key=api_key),
                                research_sources=calls.sources()))


# ── pure helpers ──────────────────────────────────────────────────────

def test_news_rows_truncate_and_keep_order():
    rows = R.news_rows(_items())
    assert rows[0]["title"] == "recent 0" and rows[0]["publisher"] == "Benzinga"
    assert len(rows[0]["summary"]) == R.SUMMARY_CHARS and rows[-1]["summary"] is None


def test_news_flow_rate_ratio():
    flow = R.news_flow(R.news_rows(_items(3, 2)), NOW)
    # 3/7 per day vs 2/23 per day
    assert flow["items_last_7d"] == 3 and flow["items_prior_23d"] == 2
    assert flow["rate_ratio"] == round((3 / 7) / (2 / 23), 2) and flow["capped"] is False
    assert R.news_flow(R.news_rows(_items(2, 0)), NOW)["rate_ratio"] is None


def test_analyst_stance_drift():
    assert R.analyst_stance(EST)["recommendation_drift"] == "upgrading"
    assert R.analyst_stance({**EST, "rec_score_now": 2.4})["recommendation_drift"] == "downgrading"
    assert R.analyst_stance({**EST, "rec_score_now": 2.15})["recommendation_drift"] == "steady"
    assert R.analyst_stance({**EST, "rec_score_3m_ago": None})["recommendation_drift"] is None
    assert R.analyst_stance(None) is None


def test_latest_quality_row_picks_newest(tmp_path):
    fp = tmp_path / "q.jsonl"
    rows = [{"ticker": "ORCL", "as_of_utc": "2026-10-23T19:40:00+00:00", "stage": "basing"},
            {"ticker": "ORCL", "as_of_utc": "2026-10-26T19:40:00+00:00", "stage": "watch"},
            {"ticker": "ORCLX", "as_of_utc": "2026-10-27T19:40:00+00:00", "stage": "turn"}]
    fp.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json \"ORCL\"\n")
    assert R.latest_quality_row("ORCL", fp)["stage"] == "watch"
    assert R.latest_quality_row("NFLX", fp) is None
    assert R.latest_quality_row("ORCL", tmp_path / "missing.jsonl") is None


# ── routes ────────────────────────────────────────────────────────────

def test_news_route_caches_and_validates():
    calls = Calls()
    c = _client(calls)
    r = c.get("/news/orcl?days=30&limit=60")
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "ORCL" and body["count"] == 5 and body["items"][0]["title"] == "recent 0"
    c.get("/news/ORCL?days=30&limit=60")
    assert calls.n == 1                                     # second call served from cache
    assert c.get("/news/ORCL?days=0").status_code == 400
    assert c.get("/news/ORCL?limit=500").status_code == 400
    assert c.get("/news/bad$").status_code == 400


def test_news_route_upstream_error_is_502():
    r = _client(Calls(news_exc=RuntimeError("alpaca 500"))).get("/news/ORCL")
    assert r.status_code == 502 and r.json()["detail"]["error"] == "news_upstream"


def test_research_route():
    body = _client(Calls()).get("/research/ORCL").json()
    assert body["analyst_estimates"]["revisions_up_30d"] == 58
    assert body["quality_lows"]["stage"] == "watch" and body["notes"] == []


def test_research_route_survives_a_failed_source():
    body = _client(Calls(est_exc=RuntimeError("yahoo 429"))).get("/research/ORCL").json()
    assert body["analyst_estimates"] is None and body["quality_lows"]["stage"] == "watch"
    assert body["notes"] and "analyst_estimates unavailable" in body["notes"][0]


def test_sentiment_route_rolls_up():
    body = _client(Calls()).get("/sentiment/ORCL").json()
    assert body["drop_cause_llm"]["label"] == "sentiment" and body["drop_cause_llm"]["scan_date"] == "2026-10-26"
    assert "evidence" not in body["drop_cause_llm"]
    assert body["drop_cause_rule"] == "sentiment" and body["quality_lows_stage"] == "watch"
    assert body["analyst"]["recommendation_drift"] == "upgrading"
    assert body["news_flow"]["items_last_7d"] == 3 and len(body["latest_headlines"]) == 5


def test_sentiment_without_label_or_news():
    calls = Calls(row={"ticker": "JPM", "stage": "not_near_low"}, news_exc=RuntimeError("down"))
    body = _client(calls).get("/sentiment/JPM").json()
    assert body["drop_cause_llm"] is None and body["news_flow"] is None
    assert any("news unavailable" in n for n in body["notes"])
    assert any("watch or later" in n for n in body["notes"])
    empty = _client(Calls(row=None)).get("/sentiment/ZZZ").json()
    assert empty["quality_lows_stage"] is None and empty["drop_cause_llm"] is None


def test_research_routes_require_the_key():
    c = _client(Calls(), api_key="k")
    for path in ("/news/ORCL", "/research/ORCL", "/sentiment/ORCL"):
        assert c.get(path).status_code == 401
        assert c.get(path, headers={"Authorization": "Bearer k"}).status_code == 200


def test_openapi_lists_the_research_routes():
    paths = _client(Calls()).get("/openapi.json").json()["paths"]
    assert {"/news/{ticker}", "/research/{ticker}", "/sentiment/{ticker}"} <= set(paths)


# ── MCP passthroughs ──────────────────────────────────────────────────

def test_mcp_tools_pass_through_with_long_timeout(monkeypatch):
    from trading_agent.mcp.tools import market
    seen = []
    monkeypatch.setattr(market, "_data_server_get",
                        lambda path, timeout=5: seen.append((path, timeout)) or {"ok": path})
    assert market.get_news("orcl", days=7, limit=10) == {"ok": "news/ORCL?days=7&limit=10"}
    market.get_research("orcl")
    market.get_sentiment("orcl")
    assert [p for p, _ in seen][1:] == ["research/ORCL", "sentiment/ORCL"]
    assert all(t == market.RESEARCH_TIMEOUT_S for _, t in seen)


def test_mcp_tools_when_server_unreachable(monkeypatch):
    from trading_agent.mcp.tools import market
    monkeypatch.setattr(market, "_data_server_get", lambda path, timeout=5: None)
    assert market.get_sentiment("NFLX")["source"] == "unavailable"
    with pytest.raises(ValueError):
        market.get_research("")
