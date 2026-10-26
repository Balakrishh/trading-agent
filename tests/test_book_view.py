"""Week of 2026-10-19 learnings, monitoring side: positions with contracts
and total risk (F6), book net delta in shadow (E4), Schwab auth paging
(A8) and the early-session close note (skill 63)."""
import json
import urllib.error
from datetime import datetime, timezone
from io import BytesIO
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from trading_agent import book_delta as bd
from trading_agent import book_view as bv

ET = ZoneInfo("America/New_York")

SNAP = {"as_of_utc": "2026-10-23T20:04:05+00:00", "age_seconds": 60, "positions": [
    {"ticker": "TLT", "strategy": "Put Debit Spread", "expiration": "2026-11-13", "contracts": 5,
     "max_loss": 55.0, "mid_pl": -102.5, "legs": ["TLT261113P00078500", "TLT261113P00080000"],
     "underlying_price": 87.1, "exit_signal": "hold"},
    {"ticker": "XLU", "strategy": "Put Debit Spread", "expiration": "2026-11-20", "contracts": 5,
     "max_loss": 56.0, "mid_pl": -60.0, "legs": ["a", "b"]},
    {"ticker": "QQQ", "strategy": "Call Debit Spread", "expiration": "2026-11-06", "contracts": 1,
     "max_loss": 300.0, "mid_pl": 20.0, "legs": ["c", "d"]},
]}
JOURNAL = [
    {"ticker": "TLT", "strategy": "Put Debit Spread", "expiration": "2026-11-13", "max_loss": 55.0,
     "contracts": 6, "status": "open"},
    {"ticker": "XLU", "strategy": "Put Debit Spread", "expiration": "2026-11-20", "max_loss": 56.0,
     "contracts": 5, "status": "open"},
    {"ticker": "GLD", "strategy": "Put Debit Spread", "expiration": "2026-11-06", "max_loss": 425.0,
     "contracts": 2, "status": "open"},
    {"ticker": "VZ", "strategy": "Cash-Secured Put", "expiration": "2026-10-09", "max_loss": 4000.0,
     "contracts": 1, "status": "expired_unrecorded"},
]


# ── F6 open book ──────────────────────────────────────────────────────

def test_merge_prefers_broker_contracts_and_adds_totals():
    rows = {r["ticker"]: r for r in bv.merge_rows(JOURNAL, SNAP)}
    tlt = rows["TLT"]
    assert tlt["contracts"] == 5                 # broker count beats the journal's 6
    assert tlt["max_loss_per_contract"] == 55.0 and tlt["total_max_loss"] == 275.0
    assert tlt["mid_pl"] == -102.5 and tlt["broker_legs"][0] == "TLT261113P00078500"
    assert tlt["in_broker_snapshot"] is True
    gld = rows["GLD"]
    assert gld["contracts"] == 2 and gld["total_max_loss"] == 850.0
    assert gld["in_broker_snapshot"] is False and gld["mid_pl"] is None
    assert rows["QQQ"]["status"] == "broker_only" and rows["QQQ"]["total_max_loss"] == 300.0


def test_totals_by_direction_and_group():
    t = bv.totals(bv.merge_rows(JOURNAL, SNAP), 28918.0)
    assert t["positions"] == 4                   # expired_unrecorded VZ left out
    assert t["total_max_loss"] == 275 + 280 + 850 + 300
    assert t["max_loss_by_direction"] == {"bear": 1405.0, "bull": 300.0}
    assert t["max_loss_by_group_direction"] == {"rates:bear": 555.0, "gold:bear": 850.0,
                                                 "us_equity:bull": 300.0}
    assert t["total_mid_pl"] == -142.5 and t["mid_pl_positions"] == 3
    assert t["total_max_loss_pct_equity"] == round(1705 / 28918 * 100, 2)


def test_open_book_without_snapshot_or_equity(monkeypatch):
    monkeypatch.setattr(bv, "_equity", lambda: None)
    book = bv.open_book(JOURNAL[:1], None)
    assert book["rows"][0]["contracts"] == 6 and book["rows"][0]["total_max_loss"] == 330.0
    assert book["totals"]["total_max_loss_pct_equity"] is None
    assert book["snapshot_as_of_utc"] is None and book["net_delta"] is None


def test_journal_row_carries_contracts():
    o = SimpleNamespace(ticker="TLT", strategy="Put Debit Spread", timestamp_utc="t", credit=-0.55,
                        spread_width=1.5, max_loss=55.0, expiration="2026-11-13", status="open",
                        contracts=5)
    assert bv.journal_row(o)["contracts"] == 5 and bv.journal_row(o)["max_loss"] == 55.0


def test_list_positions_returns_book(monkeypatch):
    from trading_agent.mcp.tools import positions as mp
    import trading_agent.journal_reader as jr
    import trading_agent.holdings_store as hs
    import trading_agent.position_snapshot as ps
    o = SimpleNamespace(ticker="TLT", strategy="Put Debit Spread", timestamp_utc="t", credit=-0.55,
                        spread_width=1.5, max_loss=55.0, expiration="2026-11-13", status="open",
                        contracts=6, run_id="", order_id="x")
    monkeypatch.setattr(jr, "JournalReader", lambda *a, **k: SimpleNamespace(
        open_trades=lambda: [o], opens_today=lambda: []))
    monkeypatch.setattr(hs, "load_holdings", lambda: SimpleNamespace(saved_at=None, parsed_count=0))
    monkeypatch.setattr(ps, "read_snapshot", lambda: SNAP)
    monkeypatch.setattr(bd, "load_state", lambda *a: {})
    monkeypatch.setattr(bv, "_equity", lambda: 28918.0)
    out = mp.list_positions()
    row = out["open_positions"][0]
    assert row["contracts"] == 5 and row["total_max_loss"] == 275.0 and row["order_id"] == "x"
    assert out["book"]["totals"]["total_max_loss"] > 0
    assert {r["status"] for r in out["open_positions"][1:]} == {"broker_only"}


def test_reviewer_prompt_includes_the_open_book():
    from trading_agent import daily_reviewer as dr
    ctx = dr.ReviewContext(review_date="2026-10-26", opens=[], closes=[], reject_reasons=[],
                           realized_pl=0.0, preset={}, watchlist=[], macro={}, recent_alerts=[],
                           open_book={"totals": {"total_max_loss": 1705.0}})
    body = json.loads(dr._build_user_prompt(ctx).split("\n\n", 1)[1])
    assert body["open_book"]["totals"]["total_max_loss"] == 1705.0
    assert "total_max_loss" in dr._SYSTEM_PROMPT


# ── E4 book delta ─────────────────────────────────────────────────────

def _leg(sym, qty):
    return SimpleNamespace(symbol=sym, qty=qty)


def test_share_deltas_sum_signed_legs_and_flag_unknown():
    legs = [_leg("TLT261113P00080000", 5), _leg("TLT261113P00078500", -5),
            _leg("XLU261120P00042000", 5), _leg("XLU261120P00040000", -5)]
    quotes = {"TLT261113P00080000": {"delta": -0.45}, "TLT261113P00078500": {"delta": -0.30},
              "XLU261120P00042000": {"delta": -0.5}, "XLU261120P00040000": {}}
    out = bd.share_deltas(legs, quotes)
    assert out["TLT"] == pytest.approx((-0.45 * 5 + 0.30 * 5) * 100)
    assert out["XLU"] is None
    assert bd.underlying_of("bad") is None


def test_beta_from_closes():
    spy = [100.0]
    for i in range(70):
        spy.append(spy[-1] * (1 + (0.01 if i % 2 else -0.008)))
    two_x = [100.0]
    for i in range(70):
        two_x.append(two_x[-1] * (1 + 2 * (0.01 if i % 2 else -0.008)))
    assert bd.beta_from_closes(two_x, spy) == pytest.approx(2.0, abs=0.05)
    assert bd.beta_from_closes(two_x[:30], spy) is None
    assert bd.beta_from_closes([1.0] * 71, [1.0] * 71) is None          # no SPY variance


def test_book_reading_and_shadow_verdict():
    r = bd.book_reading({"IWM": -40.0, "TLT": -37.5, "XLU": None},
                        {"IWM": 270.0, "TLT": 87.0, "XLU": 41.0},
                        {"IWM": 1.2, "TLT": -0.3, "XLU": 0.5}, 28918.0)
    want = -40 * 270 * 1.2 + -37.5 * 87 * -0.3
    assert r["spy_equiv_dollars"] == pytest.approx(want, abs=0.01)
    assert r["pct_of_equity"] == pytest.approx(want / 28918, abs=1e-4) and r["unknown"] == ["XLU"]
    assert bd.shadow_verdict(-0.40, -30.0, 0.30).startswith("net_delta_budget (book -40%")
    assert bd.shadow_verdict(-0.40, +30.0, 0.30) is None                  # reduces the tilt
    assert bd.shadow_verdict(-0.20, -30.0, 0.30) is None                  # inside the budget
    assert bd.shadow_verdict(-0.40, 0.4, 0.30) is None                    # neutral structure
    assert bd.shadow_verdict(None, -30.0, 0.30) is None


def test_plan_share_delta():
    plan = SimpleNamespace(legs=[SimpleNamespace(delta=-0.50, action="buy"),
                                 SimpleNamespace(delta=-0.25, action="sell")])
    assert bd.plan_share_delta(plan) == -25.0
    assert bd.plan_share_delta(SimpleNamespace(legs=[SimpleNamespace(delta=None, action="buy")])) is None


def test_state_keeps_todays_shadow_keys_only(tmp_path):
    fp = tmp_path / "bd.json"
    assert bd.first_shadow_today("IWM|Put Debit Spread", "2026-10-26", fp) is True
    assert bd.first_shadow_today("IWM|Put Debit Spread", "2026-10-26", fp) is False
    bd.record_reading({"pct_of_equity": -0.4}, "2026-10-26", fp)
    assert bd.load_state(fp)["shadow_keys"] == ["IWM|Put Debit Spread"]
    bd.record_reading({"pct_of_equity": -0.4}, "2026-10-27", fp)
    assert bd.load_state(fp)["shadow_keys"] == []
    assert bd.first_shadow_today("IWM|Put Debit Spread", "2026-10-27", fp) is True


def test_agent_net_delta_shadow_journals_once(tmp_path, monkeypatch):
    from trading_agent.agent import TradingAgent
    from trading_agent.strategy_presets import PRESETS
    monkeypatch.setattr(bd, "STATE_PATH", tmp_path / "bd.json")
    monkeypatch.setattr(bd.first_shadow_today, "__defaults__", (tmp_path / "bd.json",))
    a = TradingAgent.__new__(TradingAgent)
    a.preset = PRESETS["balanced"]
    a._book_delta_pct = -0.45
    rows = []
    a.journal_kb = SimpleNamespace(log_signal=lambda **kw: rows.append(kw))
    a._cached_price = lambda t: 1.0
    plan = SimpleNamespace(strategy_name="Put Debit Spread",
                           legs=[SimpleNamespace(delta=-0.5, action="buy"),
                                 SimpleNamespace(delta=-0.25, action="sell")])
    a._net_delta_shadow("IWM", plan)
    a._net_delta_shadow("IWM", plan)
    assert len(rows) == 1 and rows[0]["action"] == "net_delta_shadow"
    assert rows[0]["raw_signal"]["candidate_share_delta"] == -25.0


def test_schwab_quotes_pass_delta_through(monkeypatch):
    from unittest.mock import MagicMock
    from trading_agent.market_data_schwab import SchwabMarketDataProvider
    from trading_agent.schwab_oauth import SchwabOAuth
    p = SchwabMarketDataProvider(schwab_oauth=MagicMock(spec=SchwabOAuth))
    monkeypatch.setattr(p, "_get", lambda *a, **k: {
        "XLF   260529C00053000": {"quote": {"bidPrice": 0.36, "askPrice": 0.37, "delta": 0.275}},
        "XLF   260529P00050000": {"quote": {"bidPrice": 0.1, "askPrice": 0.12}}})
    out = p.fetch_option_quotes(["XLF260529C00053000", "XLF260529P00050000"])
    assert out["XLF260529C00053000"]["delta"] == 0.275
    assert out["XLF260529P00050000"]["delta"] is None


# ── A8 Schwab auth paging ─────────────────────────────────────────────

def _provider(token_exc=None, body=None):
    from unittest.mock import MagicMock
    from trading_agent.market_data_schwab import SchwabMarketDataProvider
    from trading_agent.schwab_oauth import SchwabOAuth
    oauth = MagicMock(spec=SchwabOAuth)
    if token_exc:
        oauth.get_access_token.side_effect = token_exc
    p = SchwabMarketDataProvider(schwab_oauth=oauth)
    p._get = lambda *a, **k: body
    return p


def test_auth_probe():
    assert _provider(body={"equity": {}}).auth_probe() == (True, "")
    ok, detail = _provider(body=None).auth_probe()
    assert not ok and "no data" in detail
    ok, detail = _provider(token_exc=RuntimeError("refresh permanently failed")).auth_probe()
    assert not ok and "token unavailable" in detail


def test_ready_route_uses_auth_probe():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from trading_agent.data_server.app import build_app
    bad = SimpleNamespace(auth_probe=lambda: (False, "Schwab token unavailable: x"),
                          is_market_open=lambda: False)
    r = TestClient(build_app(provider=bad)).get("/ready")
    assert r.status_code == 503 and "token unavailable" in r.text
    good = SimpleNamespace(auth_probe=lambda: (True, ""))
    assert TestClient(build_app(provider=good)).get("/ready").status_code == 200


def et(h, m):
    return datetime(2026, 10, 20, h, m, tzinfo=ET).astimezone(timezone.utc)


def test_liveness_pages_on_schwab_failure():
    from trading_agent import liveness as lv
    status, reason = lv.assess(et(10, 20), et(10, 19), True, True, False, "Schwab token unavailable")
    assert status == "down" and "Schwab not returning data" in reason
    # Server down: only that problem, not a second Schwab line.
    _, reason = lv.assess(et(10, 20), et(10, 19), True, False, False, "x")
    assert "Schwab" not in reason
    assert lv.assess(et(17, 0), et(16, 0), False, True, False, "x")[0] == "idle"


def test_schwab_ready_reads_503_detail(monkeypatch):
    from trading_agent import liveness as lv
    body = json.dumps({"detail": {"error": "not_ready", "detail": "Schwab token unavailable"}})

    def boom(url, timeout):
        raise urllib.error.HTTPError(url, 503, "x", {}, BytesIO(body.encode()))
    monkeypatch.setattr(lv.urllib.request, "urlopen", boom)
    assert lv.schwab_ready() == (False, "Schwab token unavailable")

    def down(url, timeout):
        raise urllib.error.URLError("refused")
    monkeypatch.setattr(lv.urllib.request, "urlopen", down)
    assert lv.schwab_ready() == (True, "")                # /health reports an unreachable server


# ── skill 63 early-session note ───────────────────────────────────────

def test_manual_close_early_session_note():
    from trading_agent.manual_close import early_session_note
    assert "Consider waiting" in early_session_note(datetime(2026, 10, 23, 10, 15, tzinfo=ET))
    assert early_session_note(datetime(2026, 10, 23, 10, 30, tzinfo=ET)) is None
    assert early_session_note(datetime(2026, 10, 23, 8, 0, tzinfo=ET)) is None
