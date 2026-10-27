"""Skill 68 dealer-gamma recorder and the skill-47 §3.6 market-context
routes + MCP passthroughs."""
import json
from datetime import date, datetime

import pandas as pd
import pytest

from trading_agent import gamma_exposure as gx
from trading_agent import macro_calendar as mc

TODAY = date(2026, 10, 27)


def _c(kind, strike, gamma=0.02, oi=1000, iv=0.2, exp="2026-11-20"):
    return {"type": kind, "strike": strike, "gamma": gamma, "open_interest": oi,
            "iv": iv, "expiration": exp}


def test_contract_gex_sign_and_missing_inputs():
    assert gx.contract_gex(_c("call", 100), 100.0) == pytest.approx(0.02 * 1000 * 100 * 100 * 100 * 0.01)
    assert gx.contract_gex(_c("put", 100), 100.0) < 0
    assert gx.contract_gex(_c("call", 100, oi=0), 100.0) is None
    assert gx.contract_gex({**_c("call", 100), "open_interest": None}, 100.0) is None


def test_bs_gamma_peaks_at_the_money():
    atm = gx.bs_gamma(100, 100, 30 / 365, 0.2)
    assert atm > gx.bs_gamma(100, 110, 30 / 365, 0.2) > 0
    assert gx.bs_gamma(100, 100, 0, 0.2) == 0.0


def test_flip_between_put_and_call_heavy_strikes():
    # Puts dominate below 100, calls above: net GEX turns positive somewhere near 100.
    chain = [_c("put", 95, oi=5000), _c("call", 105, oi=5000)]
    flip = gx.flip_level(chain, 100.0, TODAY)
    assert flip is not None and 97 < flip < 103
    assert gx.flip_level([_c("call", 100)], 100.0, TODAY) is None          # never negative


def test_summarize_walls_and_regime():
    chain = [_c("call", 105, oi=3000), _c("call", 110, oi=500), _c("put", 95, oi=2000),
             _c("put", 90, oi=100), _c("call", 120, gamma=0.0)]
    row = gx.summarize("SPY", chain, 100.0, TODAY)
    assert row["contracts"] == 4 and row["call_wall"] == 105 and row["put_wall"] == 95
    assert row["net_gex"] == row["call_gex"] + row["put_gex"]
    assert row["regime"] == ("long_gamma" if row["net_gex"] > 0 else "short_gamma")
    assert gx.summarize("SPY", [], 100.0, TODAY) is None


def test_snapshot_ticker_and_collect(tmp_path, monkeypatch):
    monkeypatch.setattr(gx, "expirations", lambda today, max_dte=35: [date(2026, 11, 20)])
    calls = []

    def get(path):
        calls.append(path)
        if path.startswith("/price"):
            return {"price": 100.0}
        side = "call" if "option_type=call" in path else "put"
        return {"contracts": [{k: v for k, v in _c(side, 105 if side == "call" else 95).items()
                               if k != "type"}]}

    fp = tmp_path / "g.jsonl"
    rows = gx.collect(["SPY"], get, TODAY, fp, pause_s=0)
    assert rows[0]["ticker"] == "SPY" and len(calls) == 3
    assert gx.collect(["SPY"], get, TODAY, fp, pause_s=0) == []           # once a day
    assert gx.latest_rows(fp)["SPY"]["spot"] == 100.0


def test_snapshot_without_open_interest_is_skipped(monkeypatch):
    monkeypatch.setattr(gx, "expirations", lambda today, max_dte=35: [date(2026, 11, 20)])

    def get(path):
        if path.startswith("/price"):
            return {"price": 100.0}
        return {"contracts": [{"strike": 100, "gamma": 0.02, "iv": 0.2, "expiration": "2026-11-20"}]}

    assert gx.snapshot_ticker("SPY", TODAY, get, pause_s=0) is None


def test_schwab_contract_carries_open_interest():
    from trading_agent.market_data_schwab import SchwabMarketDataProvider
    c = SchwabMarketDataProvider._normalize_contract(
        {"symbol": "SPY   261120C00700000", "bid": 1, "ask": 1.2, "mark": 1.1, "delta": 0.5,
         "gamma": 0.01, "volatility": 20, "strikePrice": 700, "openInterest": 1234}, "2026-11-20", "call")
    assert c["open_interest"] == 1234
    c2 = SchwabMarketDataProvider._normalize_contract(
        {"symbol": "SPY   261120C00700000", "strikePrice": 700}, "2026-11-20", "call")
    assert c2["open_interest"] is None


# ── data server routes ────────────────────────────────────────────────

EVENTS = [mc.MacroEvent("FOMC", date(2026, 10, 28)), mc.MacroEvent("NFP", date(2026, 11, 6))]


def _frame(vals):
    return pd.DataFrame({"Close": vals}, index=pd.date_range("2026-06-01", periods=len(vals), freq="B"))


def _sources(**over):
    from trading_agent.data_server.market_context import ContextSources
    hist = {"HYG": _frame([80.0] * 59 + [78.0]), "IEF": _frame([100.0] * 60),
            "SPY": _frame([700 * (1.01 if i % 2 else 0.99) for i in range(60)])}
    base = dict(history=lambda t: hist[t], atm_iv=lambda t, d: 0.15,
                gamma_rows=lambda: {"SPY": {"ticker": "SPY", "net_gex": 1}},
                market_state=lambda: {"state": "CAUTION"}, events=lambda: EVENTS,
                now_et=lambda: datetime(2026, 10, 27, 17, 0))
    base.update(over)
    return ContextSources(**base)


def _client(src):
    from fastapi.testclient import TestClient
    from trading_agent.data_server.app import build_app
    return TestClient(build_app(provider=object(), context_sources=src))


def test_macro_events_route():
    c = _client(_sources())
    body = c.get("/macro-events?days=15").json()
    assert [e["kind"] for e in body["events"]] == ["FOMC", "NFP"] and body["next"].startswith("FOMC")
    assert body["events"][0]["release_et"] == "14:00" and body["events"][0]["days_away"] == 1
    assert c.get("/macro-events?days=0").status_code == 400


def test_credit_stress_route():
    body = _client(_sources()).get("/credit-stress").json()
    assert body["stress_pct"] > 2 and body["triggers_caution"] is True
    assert body["pair"] == "HYG/IEF" and len(body["recent"]) >= 1


def test_credit_stress_upstream_error():
    def boom(t):
        raise RuntimeError("yahoo down")
    r = _client(_sources(history=boom)).get("/credit-stress")
    assert r.status_code == 502 and r.json()["detail"]["error"] == "context_upstream"


def test_vol_route(monkeypatch):
    body = _client(_sources()).get("/vol/spy").json()
    assert body["ticker"] == "SPY" and body["iv_atm_30d"] == 0.15 and body["rv_20d"] > 0.1
    assert body["iv_rv_ratio"] < 1 and body["short_premium_allowed"] is False
    nodata = _client(_sources(atm_iv=lambda t, d: None)).get("/vol/SPY").json()
    assert nodata["iv_rv_ratio"] is None and nodata["short_premium_allowed"] is None and nodata["notes"]


def test_gamma_and_market_context_routes():
    c = _client(_sources())
    assert c.get("/gamma/SPY").json()["gamma"]["net_gex"] == 1
    assert c.get("/gamma/QQQ").json()["gamma"] is None
    ctx = c.get("/market-context").json()
    assert ctx["market_state"]["state"] == "CAUTION" and ctx["macro_events"]["next"]
    assert ctx["credit_stress"]["triggers_caution"] and ctx["gamma"]["SPY"]["net_gex"] == 1


def test_market_context_survives_a_failed_source():
    def boom(t):
        raise RuntimeError("down")
    ctx = _client(_sources(history=boom)).get("/market-context").json()
    assert ctx["credit_stress"] is None and ctx["market_state"]["state"] == "CAUTION"
    assert any("credit_stress unavailable" in n for n in ctx["notes"])


def test_openapi_lists_context_routes():
    paths = _client(_sources()).get("/openapi.json").json()["paths"]
    assert {"/macro-events", "/credit-stress", "/vol/{ticker}", "/gamma/{ticker}",
            "/market-context"} <= set(paths)


def test_mcp_context_tools(monkeypatch):
    from trading_agent.mcp.tools import market
    seen = []
    monkeypatch.setattr(market, "_data_server_get", lambda p, timeout=5: seen.append(p) or {"ok": 1})
    market.get_macro_events(30)
    market.get_credit_stress()
    market.get_vol("gld")
    market.get_gamma("qqq")
    market.get_market_context()
    assert seen == ["macro-events?days=30", "credit-stress", "vol/GLD", "gamma/QQQ", "market-context"]
    monkeypatch.setattr(market, "_data_server_get", lambda p, timeout=5: None)
    assert market.get_market_context()["source"] == "unavailable"
