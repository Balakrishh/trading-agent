"""Skill 48 — get_quote handles option symbols via /quotes (2026-10-13:
option symbols went to /price, got HTTP 400, and read as
"data server unreachable")."""
from trading_agent.mcp.tools import market

OPT = "GLD261106P00379000"


def _route(monkeypatch, responses):
    calls = []

    def fake(path, body=None, timeout=5):
        calls.append((path, body))
        return responses.get(path)
    monkeypatch.setattr(market, "_data_server_request", fake)
    return calls


def test_option_symbol_uses_quotes_endpoint(monkeypatch):
    calls = _route(monkeypatch, {"quotes": {"count": 1, "quotes": {
        OPT: {"bid": 6.2, "ask": 6.4, "mid": 6.3}}}})
    out = market.get_quote(OPT.lower())
    assert calls == [("quotes", {"symbols": [OPT]})]
    assert out == {"symbol": OPT, "price": 6.3, "bid": 6.2, "ask": 6.4,
                   "mid": 6.3, "source": "data_server"}


def test_option_without_a_quote_says_so(monkeypatch):
    _route(monkeypatch, {"quotes": {"count": 0, "quotes": {}}})
    out = market.get_quote(OPT)
    assert out["price"] is None and out["note"] == "no quote for this contract"


def test_equity_still_uses_price(monkeypatch):
    calls = _route(monkeypatch, {"price/SPY": {"ticker": "SPY", "price": 777.5}})
    assert market.get_quote("spy") == {"ticker": "SPY", "price": 777.5}
    assert calls == [("price/SPY", None)]


def test_unreachable_server(monkeypatch):
    _route(monkeypatch, {})
    assert market.get_quote(OPT)["source"] == "unavailable"
