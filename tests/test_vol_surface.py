"""Skill 66 phase 2 — daily skew / term-structure snapshot."""
from __future__ import annotations

import json
from datetime import date

from trading_agent import vol_surface as vs


def _c(delta, iv, bid=1.0, ask=1.02, typ="call"):
    return {"delta": delta, "iv": iv, "bid": bid, "ask": ask, "type": typ}


CALLS = [_c(0.75, 0.10), _c(0.52, 0.12), _c(0.26, 0.10), _c(0.10, 0.11)]
PUTS = [_c(-0.48, 0.13, typ="put"), _c(-0.24, 0.16, typ="put"), _c(-0.10, 0.20, typ="put")]


def test_nearest_by_delta_skips_contracts_without_iv():
    assert vs.nearest_by_delta([_c(0.5, 0.0), _c(0.4, 0.2)], 0.5)["delta"] == 0.4
    assert vs.nearest_by_delta([], 0.5) is None


def test_summarize_expiry_skew_numbers():
    s = vs.summarize_expiry(CALLS, PUTS)
    assert s["atm_iv"] == 0.125                      # (0.12 + 0.13) / 2
    assert (s["put25_iv"], s["call25_iv"]) == (0.16, 0.10)
    assert s["rr25"] == -0.06 and s["put_skew"] == 0.035
    assert s["fly25"] == 0.005
    assert s["n_contracts"] == 7 and s["atm_spread_pct"] < 0.03


def test_summarize_expiry_needs_both_sides():
    assert vs.summarize_expiry(CALLS, []) is None


def test_term_structure_ratios():
    t = vs.term_structure([{"target_dte": 7, "atm_iv": 0.10}, {"target_dte": 30, "atm_iv": 0.125},
                           {"target_dte": 60, "atm_iv": 0.15}])
    assert t["ratio_7_30"] == 0.8 and t["ratio_30_60"] == 0.8333
    assert vs.term_structure([{"target_dte": 30, "atm_iv": 0.1}])["ratio_7_30"] is None


def _server(missing=()):
    def get(path):
        if path.startswith("/price/"):
            return {"price": 100.0}
        t = path.split("/")[2].split("?")[0]
        if t in missing:
            return None
        return {"contracts": CALLS if "option_type=call" in path else PUTS}
    return get


def test_collect_appends_once_per_ticker_per_day(tmp_path):
    fp = tmp_path / "vs.jsonl"
    day = date(2026, 10, 16)
    rows = vs.collect(["SPY", "XLU"], _server(), day, fp, pause_s=0)
    assert [r["ticker"] for r in rows] == ["SPY", "XLU"]
    r = json.loads(fp.read_text().splitlines()[0])
    assert r["date"] == "2026-10-16" and r["spot"] == 100.0
    assert [e["target_dte"] for e in r["expiries"]] == [7, 30, 60]
    assert vs.collect(["SPY", "XLU"], _server(), day, fp, pause_s=0) == []   # already recorded
    assert len(fp.read_text().splitlines()) == 2


def test_collect_skips_a_failing_ticker_and_dry_run_writes_nothing(tmp_path):
    fp = tmp_path / "vs.jsonl"
    rows = vs.collect(["SPY", "BAD"], _server(missing={"BAD"}), date(2026, 10, 16), fp,
                      dry_run=True, pause_s=0)
    assert [r["ticker"] for r in rows] == ["SPY"] and not fp.exists()


def test_candidate_expirations_start_near_target():
    c = vs.candidate_expirations(date(2026, 10, 16), 30)
    assert 24 <= (c[0] - date(2026, 10, 16)).days <= 38
    assert all(d > date(2026, 10, 16) for d in c)


def test_candidate_expirations_holiday_week_neighbours_are_fridays():
    """2026-10-16 run: the 60-day pick was Thu Dec 24 (Christmas Friday);
    shifting it a week gave Thu Dec 17 instead of the Dec 18 monthly, so
    names without weeklies that far out recorded no 60-day IV."""
    c = vs.candidate_expirations(date(2026, 10, 16), 60)
    assert c[0] == date(2026, 12, 24)
    assert date(2026, 12, 18) in c and date(2026, 12, 17) not in c
    assert date(2026, 12, 31) in c          # Thu: New Year's Day is a Friday


def test_collector_never_opens_a_schwab_session():
    """Skill 16 §4: one token holder — the collector goes through the data server."""
    import ast, inspect
    tree = ast.parse(inspect.getsource(vs))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any(m and ("schwab" in m or "market_data" in m) for m in mods)
