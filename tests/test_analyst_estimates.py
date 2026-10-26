"""Analyst estimates summary + cache (skill 67)."""
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from trading_agent import analyst_estimates as ae


def _tables(cur_now=3.58, cur_30=3.58, nxt_now=3.76, nxt_30=3.81, up=3, down=10):
    trend = pd.DataFrame({"current": [cur_now, nxt_now], "30daysAgo": [cur_30, nxt_30],
                          "90daysAgo": [3.58, 3.81]}, index=["0y", "+1y"])
    revs = pd.DataFrame({"upLast30days": [up, 0], "downLast30days": [down, 0]}, index=["0y", "+1y"])
    rev_est = pd.DataFrame({"growth": [0.1315, 0.1077]}, index=["0y", "+1y"])
    earn_est = pd.DataFrame({"growth": [0.4157, 0.0511], "numberOfAnalysts": [35, 44]},
                            index=["0y", "+1y"])
    recs = pd.DataFrame({"period": ["0m", "-3m"], "strongBuy": [7, 7], "buy": [26, 29],
                         "hold": [16, 15], "sell": [2, 0], "strongSell": [0, 0]})
    return {"eps_trend": trend, "eps_revisions": revs, "revenue_estimate": rev_est,
            "earnings_estimate": earn_est, "recommendations": recs,
            "analyst_price_targets": {"current": 65.85, "mean": 84.83}}


def test_summarize_fields_in_percent():
    s = ae.summarize(_tables())
    assert s["eps_next_year_rev_30d_pct"] == round((3.76 / 3.81 - 1) * 100, 2)
    assert s["rev_growth_cur_year_pct"] == 13.15
    assert s["eps_growth_next_year_pct"] == 5.11
    assert (s["revisions_up_30d"], s["revisions_down_30d"]) == (3, 10)
    assert s["analysts_cur_year"] == 35
    assert s["target_upside_pct"] == round((84.83 / 65.85 - 1) * 100, 2)
    assert 2.0 < s["rec_score_now"] < 2.5


def test_spot_overrides_target_current():
    s = ae.summarize(_tables(), spot=80.0)
    assert s["target_upside_pct"] == round((84.83 / 80.0 - 1) * 100, 2)


def test_revision_trend_from_size_then_counts():
    assert ae.summarize(_tables(cur_now=3.70, cur_30=3.58, nxt_now=3.90, nxt_30=3.81))["revision_trend"] == "up"
    assert ae.summarize(_tables(cur_now=3.30, cur_30=3.58, nxt_now=3.50, nxt_30=3.81))["revision_trend"] == "down"
    assert ae.revision_trend({"revisions_up_30d": 1, "revisions_down_30d": 4}) == "down"
    assert ae.revision_trend({}) == "unknown"


def test_failed_tables_are_skipped_and_empty_is_none():
    assert ae.summarize({k: None for k in ae._TABLES}) is None
    part = ae.summarize({"eps_trend": None, "analyst_price_targets": {"current": 10, "mean": 12}})
    assert part["target_upside_pct"] == 20.0 and part["eps_cur_year"] is None


def test_nan_cells_are_none():
    t = _tables()
    t["eps_trend"].loc["0y", "30daysAgo"] = float("nan")
    assert ae.summarize(t)["eps_cur_year_rev_30d_pct"] is None


def test_fetch_tables_isolates_a_failing_attribute():
    class Tk:
        eps_trend = "ok"

        @property
        def eps_revisions(self):
            raise RuntimeError("yahoo 429")
    out = ae.fetch_tables("X", ticker_factory=lambda t: Tk())
    assert out["eps_trend"] == "ok" and out["eps_revisions"] is None


def test_cache_hit_miss_and_failure(tmp_path):
    fp = tmp_path / "ae.json"
    calls = []

    def fetch(t):
        calls.append(t)
        return _tables()
    now = datetime(2026, 10, 26, 20, tzinfo=timezone.utc)
    a = ae.get_estimates("nflx", path=fp, fetch=fetch, now=now)
    b = ae.get_estimates("NFLX", path=fp, fetch=fetch, now=now + timedelta(hours=1))
    assert a == b and calls == ["NFLX"]
    ae.get_estimates("NFLX", path=fp, fetch=fetch, now=now + timedelta(hours=25))
    assert calls == ["NFLX", "NFLX"]

    def boom(t):
        raise RuntimeError("down")
    assert ae.get_estimates("ORCL", path=fp, fetch=boom, now=now) is None
    assert json.loads(fp.read_text())["ORCL"]["summary"] is None
