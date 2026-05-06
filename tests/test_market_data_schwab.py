"""
test_market_data_schwab.py — pin down the Schwab adapter contract.

These tests run with no Schwab credentials and no network access:

  * Symbol translators are pure functions (no I/O).
  * The HTTP-bound methods (`fetch_option_chain`, `fetch_option_quotes`,
    `fetch_batch_snapshots`, `fetch_intraday_bars`) are exercised by
    monkey-patching `_get` so we can feed canonical Schwab JSON shapes
    captured from the official OpenAPI spec and assert the adapter
    returns the agent's canonical contract dict shape.

The point of these tests is to lock in field-mapping decisions made
in the adapter so a future schema drift on Schwab's side fails loudly
instead of silently feeding the chain scanner zeros.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest

from trading_agent.market_data_schwab import (
    SchwabMarketDataProvider,
    from_schwab_symbol,
    to_schwab_symbol,
)
from trading_agent.schwab_oauth import SchwabOAuth, TokenSet


# ── Symbol translators ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "compact,padded",
    [
        ("AMZN220617C03170000", "AMZN  220617C03170000"),  # 4-char root → 2 spaces
        ("XLF260529C00053000",  "XLF   260529C00053000"),  # 3-char root → 3 spaces
        ("AAPL250101P00150000", "AAPL  250101P00150000"),
        ("DJX231215C00290000",  "DJX   231215C00290000"),
        ("SPXW260620P05000000", "SPXW  260620P05000000"),  # weekly root, 4 char
    ],
)
def test_to_schwab_symbol_pads_root_to_six(compact, padded):
    assert to_schwab_symbol(compact) == padded
    assert from_schwab_symbol(padded) == compact


def test_symbol_translators_passthrough_non_options():
    """Equity / index symbols don't match the OCC pattern → passthrough."""
    for s in ["AAPL", "$SPX", "$DJI", "/ESZ24", "EUR/USD"]:
        assert to_schwab_symbol(s) == s
        assert from_schwab_symbol(s) == s


def test_symbol_translators_handle_empty():
    assert to_schwab_symbol("") == ""
    assert from_schwab_symbol("") == ""


def test_round_trip_symbol_preserves_shape():
    """For valid OCC symbols compact → padded → compact must be identity."""
    for s in ["AAPL250101P00150000", "XLF260529C00053000",
              "QQQM260619P00400000"]:
        assert from_schwab_symbol(to_schwab_symbol(s)) == s


# ── Adapter fixture (no creds, no network) ──────────────────────────


@pytest.fixture
def adapter() -> SchwabMarketDataProvider:
    """
    Build an adapter with a stubbed OAuth client and a `_get` we can
    monkey-patch per test.  No network calls are ever made.
    """
    oauth = MagicMock(spec=SchwabOAuth)
    oauth.get_access_token.return_value = "DUMMY_ACCESS_TOKEN"
    oauth._tokens = TokenSet(
        access_token="DUMMY_ACCESS_TOKEN",
        refresh_token="DUMMY_REFRESH_TOKEN",
        expires_at=time.time() + 1800,
        refresh_expires_at=time.time() + 7 * 86400,
    )
    a = SchwabMarketDataProvider(
        schwab_oauth=oauth,
        alpaca_api_key="dummy", alpaca_secret_key="dummy",
    )
    return a


# ── /chains response normalisation ──────────────────────────────────

# Captured shape from the official Schwab OpenAPI spec — one CALL contract.
SAMPLE_CHAIN_RESPONSE: Dict[str, Any] = {
    "symbol": "XLF",
    "status": "SUCCESS",
    "underlying": {"symbol": "XLF", "last": 51.49, "mark": 51.495},
    "callExpDateMap": {
        "2026-05-29:24": {
            "53.0": [
                {
                    "putCall": "CALL",
                    "symbol": "XLF   260529C00053000",
                    "description": "XLF 05/29/2026 $53 Call",
                    "bidPrice": 0.36, "askPrice": 0.37,
                    "lastPrice": 0.36, "markPrice": 0.365,
                    "bidSize": 100, "askSize": 100, "lastSize": 0,
                    "highPrice": 0.40, "lowPrice": 0.34,
                    "openPrice": 0.36, "closePrice": 0.38,
                    "totalVolume": 234,
                    "tradeDate": 1716739200000,
                    "quoteTimeInLong": 1716739200000,
                    "tradeTimeInLong": 1716739200000,
                    "netChange": -0.02,
                    "volatility": 22.31,        # PERCENT, not decimal
                    "delta": 0.275, "gamma": 0.08,
                    "theta": -0.012, "vega": 0.04, "rho": 0.02,
                    "timeValue": 0.36,
                    "openInterest": 5421,
                    "isInTheMoney": False,
                    "isPennyPilot": True,
                    "isMini": False,
                    "isNonStandard": False,
                    "strikePrice": 53.0,
                    "expirationDate": "2026-05-29",
                    "daysToExpiration": 24,
                    "expirationType": "S",
                    "lastTradingDay": 1716998400000,
                    "multiplier": 100,
                    "settlementType": "P",
                    "isIndexOption": False,
                    "percentChange": -5.26,
                    "markChange": -0.005,
                    "markPercentChange": -1.35,
                    "intrinsicValue": 0,
                    "optionRoot": "XLF",
                }
            ]
        }
    },
    "putExpDateMap": {},
}


def test_fetch_option_chain_normalises_to_agent_shape(adapter, monkeypatch):
    """The adapter must return the same dict shape Alpaca's adapter returns."""
    monkeypatch.setattr(adapter, "_get", lambda *a, **kw: SAMPLE_CHAIN_RESPONSE)
    contracts = adapter.fetch_option_chain("XLF", "2026-05-29", "call")

    assert contracts is not None
    assert len(contracts) == 1
    c = contracts[0]
    # Compact OCC, not Schwab-padded
    assert c["symbol"] == "XLF260529C00053000"
    assert c["bid"] == 0.36
    assert c["ask"] == 0.37
    assert c["mid"] == 0.365
    assert c["delta"] == 0.275
    assert c["theta"] == -0.012
    assert c["vega"] == 0.04
    assert c["gamma"] == 0.08
    # Critical: IV must be converted percent → decimal
    assert c["iv"] == pytest.approx(0.2231, rel=1e-3)
    assert c["strike"] == 53.0
    assert c["expiration"] == "2026-05-29"
    assert c["type"] == "call"


def test_fetch_option_chain_caches(adapter, monkeypatch):
    """Second call within OPTION_CHAIN_TTL must reuse the cached list."""
    calls = {"n": 0}

    def fake_get(*_a, **_kw):
        calls["n"] += 1
        return SAMPLE_CHAIN_RESPONSE

    monkeypatch.setattr(adapter, "_get", fake_get)
    adapter.fetch_option_chain("XLF", "2026-05-29", "call")
    adapter.fetch_option_chain("XLF", "2026-05-29", "call")
    assert calls["n"] == 1, "second call should hit the cache"


def test_fetch_option_chain_returns_empty_list_when_side_missing(adapter, monkeypatch):
    """If putExpDateMap is empty, adapter returns [] (not None)."""
    monkeypatch.setattr(adapter, "_get", lambda *a, **kw: SAMPLE_CHAIN_RESPONSE)
    contracts = adapter.fetch_option_chain("XLF", "2026-05-29", "put")
    assert contracts == []


def test_fetch_option_chain_returns_none_on_http_failure(adapter, monkeypatch):
    """`_get` returning None must propagate as None — caller skips ticker."""
    monkeypatch.setattr(adapter, "_get", lambda *a, **kw: None)
    assert adapter.fetch_option_chain("XLF", "2026-05-29", "call") is None


# Production-shape fixture captured 2026-05-06 from the live Schwab
# /chains response.  Critical: production uses `bid`/`ask`/`mark` —
# NOT `bidPrice`/`askPrice`/`markPrice` as the swagger spec declares.
# Greeks ARE present and correct.
SCHWAB_PROD_CHAIN: Dict[str, Any] = {
    "symbol": "SPY",
    "status": "SUCCESS",
    "callExpDateMap": {},
    "putExpDateMap": {
        "2026-05-15:9": {
            "718.0": [
                {
                    "putCall": "PUT",
                    "symbol": "SPY   260515P00718000",
                    # ── Production short-form price fields ─────────────
                    "bid": 4.20,
                    "ask": 4.30,
                    "mark": 4.25,
                    # ── Greeks (correctly populated) ───────────────────
                    "delta": -0.202,
                    "gamma": 0.016,
                    "theta": -0.271,
                    "vega": 0.328,
                    "rho": -0.038,
                    "volatility": 15.249,           # PERCENT
                    "openInterest": 3706,
                    "strikePrice": 718.0,
                    "expirationDate": "2026-05-15",
                    "daysToExpiration": 9,
                    "expirationType": "W",
                    "isInTheMoney": False,
                    "isPennyPilot": True,
                }
            ]
        }
    },
}


def test_chain_parses_production_short_field_names(adapter, monkeypatch):
    """Production Schwab uses bid/ask/mark, not bidPrice/askPrice/markPrice.

    Regression for 2026-05-06 — the chain scanner was rejecting every
    candidate as `no_short_contract` because `_normalize_contract` was
    reading None from non-existent `bidPrice`/`askPrice`/`markPrice`
    fields, then `_find_short` filtered out every contract with bid=0.
    """
    monkeypatch.setattr(adapter, "_get", lambda *a, **kw: SCHWAB_PROD_CHAIN)
    contracts = adapter.fetch_option_chain("SPY", "2026-05-15", "put")
    assert contracts is not None
    assert len(contracts) == 1
    c = contracts[0]
    # Critical: the production short-form fields must populate the canonical
    # bid/ask/mid keys non-zero so the chain scanner doesn't filter them out.
    assert c["bid"] == 4.20, f"bid must come from `bid` (not bidPrice), got {c['bid']}"
    assert c["ask"] == 4.30, f"ask must come from `ask` (not askPrice), got {c['ask']}"
    assert c["mid"] == 4.25, f"mid must come from `mark` (not markPrice), got {c['mid']}"
    # Greeks unchanged (production uses the spec field names for these).
    assert c["delta"] == -0.202
    assert c["iv"] == pytest.approx(0.15249, rel=1e-3)


def test_chain_also_accepts_swagger_long_field_names(adapter, monkeypatch):
    """Forward compat — if Schwab eventually ships the swagger-shape, the
    long field names (bidPrice/askPrice/markPrice) should still parse."""
    swagger_shape = {
        "symbol": "SPY",
        "callExpDateMap": {
            "2026-05-15:9": {
                "718.0": [
                    {
                        "putCall": "CALL",
                        "symbol": "SPY   260515C00718000",
                        "bidPrice": 5.10,
                        "askPrice": 5.20,
                        "markPrice": 5.15,
                        "delta": 0.55,
                        "strikePrice": 718.0,
                        "expirationDate": "2026-05-15",
                        "volatility": 18.0,
                    }
                ]
            }
        },
        "putExpDateMap": {},
    }
    monkeypatch.setattr(adapter, "_get", lambda *a, **kw: swagger_shape)
    contracts = adapter.fetch_option_chain("SPY", "2026-05-15", "call")
    assert contracts is not None
    assert contracts[0]["bid"] == 5.10
    assert contracts[0]["ask"] == 5.20
    assert contracts[0]["mid"] == 5.15


# ── /quotes response normalisation ──────────────────────────────────


def test_fetch_option_quotes_round_trips_compact_symbols(adapter, monkeypatch):
    """
    Caller passes compact OCC; adapter must query Schwab with padded
    OCC and return results keyed by the original compact form.
    """
    captured: Dict[str, Any] = {}

    def fake_get(path, params=None, **kw):
        captured["path"] = path
        captured["params"] = params
        # Schwab keys responses by the symbol you sent — including spaces.
        return {
            "XLF   260529C00053000": {
                "assetMainType": "OPTION",
                "symbol": "XLF   260529C00053000",
                "quote": {
                    "bidPrice": 0.36, "askPrice": 0.37, "mark": 0.365,
                    "delta": 0.275, "gamma": 0.08,
                },
            }
        }

    monkeypatch.setattr(adapter, "_get", fake_get)
    out = adapter.fetch_option_quotes(["XLF260529C00053000"])

    assert captured["path"] == "/quotes"
    # Adapter must request the padded form
    assert "XLF   260529C00053000" in captured["params"]["symbols"]

    # …but return results keyed by the compact form the caller sent in
    assert "XLF260529C00053000" in out
    assert out["XLF260529C00053000"]["bid"] == 0.36
    assert out["XLF260529C00053000"]["ask"] == 0.37
    assert out["XLF260529C00053000"]["mid"] == 0.365


def test_fetch_option_quotes_handles_missing_symbol(adapter, monkeypatch):
    """If Schwab omits a symbol from the response, adapter just skips it."""
    monkeypatch.setattr(adapter, "_get", lambda *a, **kw: {})  # empty body
    out = adapter.fetch_option_quotes(["XLF260529C00053000"])
    assert out == {}


def test_fetch_option_quotes_empty_input_short_circuits(adapter, monkeypatch):
    monkeypatch.setattr(adapter, "_get",
                        lambda *a, **kw: pytest.fail("should not be called"))
    assert adapter.fetch_option_quotes([]) == {}


# ── /quotes for equities (snapshots) ────────────────────────────────


SAMPLE_EQUITY_QUOTE_RESPONSE: Dict[str, Any] = {
    "AAPL": {
        "assetMainType": "EQUITY",
        "symbol": "AAPL",
        "quote": {
            "bidPrice": 168.40, "askPrice": 168.41,
            "lastPrice": 168.405, "mark": 168.405,
            "totalVolume": 22361159,
        },
    },
    "XLF": {
        "assetMainType": "EQUITY", "assetSubType": "ETF",
        "symbol": "XLF",
        "quote": {
            "bidPrice": 51.49, "askPrice": 51.50,
            "lastPrice": 51.495, "mark": 51.495,
        },
    },
}


def test_fetch_batch_snapshots_uses_lastprice(adapter, monkeypatch):
    monkeypatch.setattr(adapter, "_get",
                        lambda *a, **kw: SAMPLE_EQUITY_QUOTE_RESPONSE)
    prices = adapter.fetch_batch_snapshots(["AAPL", "XLF"])
    assert prices["AAPL"] == 168.405
    assert prices["XLF"] == 51.495


def test_fetch_batch_snapshots_falls_back_to_mid_when_no_last(adapter, monkeypatch):
    """When lastPrice and mark are both missing, adapter uses (bid+ask)/2."""
    monkeypatch.setattr(adapter, "_get", lambda *a, **kw: {
        "FOO": {"symbol": "FOO", "quote": {"bidPrice": 10.0, "askPrice": 10.10}}
    })
    prices = adapter.fetch_batch_snapshots(["FOO"])
    assert prices["FOO"] == pytest.approx(10.05)


def test_fetch_batch_snapshots_caches(adapter, monkeypatch):
    """Subsequent calls within SNAPSHOT_TTL hit the cache."""
    calls = {"n": 0}

    def fake_get(*_a, **_kw):
        calls["n"] += 1
        return SAMPLE_EQUITY_QUOTE_RESPONSE

    monkeypatch.setattr(adapter, "_get", fake_get)
    adapter.fetch_batch_snapshots(["AAPL", "XLF"])
    adapter.fetch_batch_snapshots(["AAPL", "XLF"])
    assert calls["n"] == 1


# ── /pricehistory parsing ───────────────────────────────────────────


def test_fetch_intraday_bars_parses_candles(adapter, monkeypatch):
    """Schwab returns epoch-ms `datetime`; adapter must coerce to UTC ts."""
    body = {
        "symbol": "AAPL",
        "candles": [
            {"datetime": 1639137600000, "open": 175.01, "high": 175.15,
             "low": 175.01, "close": 175.04, "volume": 10719},
            {"datetime": 1639137900000, "open": 175.04, "high": 175.10,
             "low": 175.00, "close": 175.05, "volume": 5000},
        ],
        "previousClose": 174.56,
        "empty": False,
    }
    monkeypatch.setattr(adapter, "_get", lambda *a, **kw: body)
    df = adapter.fetch_intraday_bars("AAPL", "5m")
    assert not df.empty
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    # First row datetime → 2021-12-10 12:00:00 UTC
    assert df.index[0].tzinfo is not None
    assert float(df["Close"].iloc[0]) == 175.04


def test_fetch_intraday_bars_returns_empty_df_on_failure(adapter, monkeypatch):
    monkeypatch.setattr(adapter, "_get", lambda *a, **kw: None)
    df = adapter.fetch_intraday_bars("AAPL", "5m")
    assert df.empty


# ── 401 retry path ───────────────────────────────────────────────────


def test_get_retries_once_on_401(adapter, monkeypatch):
    """A 401 response forces a token refresh and retries the request once."""

    class Resp:
        def __init__(self, status, body=None, headers=None):
            self.status_code = status
            self._body = body or {}
            self.headers = headers or {}
            self.text = ""

        def json(self):
            return self._body

    sequence = [Resp(401, headers={"Schwab-Client-CorrelId": "abc"}),
                Resp(200, body={"symbol": "AAPL"})]

    def fake_session_get(*_a, **_kw):
        return sequence.pop(0)

    adapter._session = MagicMock()
    adapter._session.get.side_effect = fake_session_get
    body = adapter._get("/quotes", params={"symbols": "AAPL"})
    assert body == {"symbol": "AAPL"}
    # Ensure tokens were cleared once so refresh path was triggered
    # (oauth.get_access_token gets called twice — once before and once
    # after we cleared _tokens).
    assert adapter.oauth.get_access_token.call_count >= 2


# ── Factory / env routing ────────────────────────────────────────────


def test_factory_returns_alpaca_by_default(monkeypatch):
    monkeypatch.delenv("MARKET_DATA_PROVIDER", raising=False)
    from trading_agent.market_data import MarketDataProvider as Alpaca
    from trading_agent.market_data_factory import build_market_data_provider
    p = build_market_data_provider(
        alpaca_api_key="dummy", alpaca_secret_key="dummy",
    )
    # Strict equality — Schwab subclass would also pass `isinstance(...,
    # Alpaca)`, so we need to assert it's exactly the parent.
    assert type(p) is Alpaca


def test_factory_returns_schwab_when_env_set(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "schwab")
    monkeypatch.setenv("SCHWAB_CLIENT_ID", "test_id")
    monkeypatch.setenv("SCHWAB_CLIENT_SECRET", "test_secret")
    monkeypatch.setenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1:8182")
    monkeypatch.setenv("SCHWAB_TOKEN_PATH", "/tmp/nonexistent_token.json")
    from trading_agent.market_data_factory import build_market_data_provider
    from trading_agent.market_data_schwab import SchwabMarketDataProvider
    p = build_market_data_provider(
        alpaca_api_key="dummy", alpaca_secret_key="dummy",
    )
    assert isinstance(p, SchwabMarketDataProvider)


def test_factory_legacy_re_export_still_works():
    """Old import path `from trading_agent.market_data_schwab import
    build_market_data_provider` must keep working — external scripts
    may still depend on it."""
    from trading_agent.market_data_factory import (
        build_market_data_provider as canonical,
    )
    from trading_agent.market_data_schwab import (
        build_market_data_provider as legacy,
    )
    assert legacy is canonical
