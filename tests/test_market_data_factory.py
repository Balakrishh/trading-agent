"""
test_market_data_factory.py — pin down per-surface provider routing.

The factory's job is to walk the env-var priority chain
(MARKET_DATA_PROVIDER_<SURFACE> → MARKET_DATA_PROVIDER → alpaca) and
return the right concrete provider class.  These tests lock down that
routing so a future change to the priority chain or a typo in a known-
provider name fails loudly instead of silently falling back to Alpaca.

No network calls or secret material involved.
"""

from __future__ import annotations

import pytest

from trading_agent.market_data import MarketDataProvider
from trading_agent.market_data_factory import (
    _resolve_provider_name,
    build_market_data_provider,
)
from trading_agent.market_data_yahoo import YahooMarketDataProvider


# ── _resolve_provider_name() — pure logic ────────────────────────────


def _clear_env(monkeypatch):
    """Strip every market-data env var so tests don't leak."""
    for k in [
        "MARKET_DATA_PROVIDER",
        "MARKET_DATA_PROVIDER_LIVE",
        "MARKET_DATA_PROVIDER_WATCHLIST",
        "MARKET_DATA_PROVIDER_BACKTEST",
        "MARKET_DATA_PROVIDER_DEMO",
    ]:
        monkeypatch.delenv(k, raising=False)


def test_resolve_defaults_to_alpaca(monkeypatch):
    _clear_env(monkeypatch)
    assert _resolve_provider_name(None) == "alpaca"
    assert _resolve_provider_name("live") == "alpaca"
    assert _resolve_provider_name("watchlist") == "alpaca"


def test_resolve_uses_global_when_set(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "schwab")
    assert _resolve_provider_name(None) == "schwab"
    assert _resolve_provider_name("live") == "schwab"
    assert _resolve_provider_name("watchlist") == "schwab"


def test_resolve_per_surface_overrides_global(monkeypatch):
    """Per-surface env var beats the global."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "alpaca")
    monkeypatch.setenv("MARKET_DATA_PROVIDER_LIVE", "schwab")
    monkeypatch.setenv("MARKET_DATA_PROVIDER_WATCHLIST", "yahoo")
    assert _resolve_provider_name("live") == "schwab"
    assert _resolve_provider_name("watchlist") == "yahoo"
    # An unset surface falls through to global
    assert _resolve_provider_name("backtest") == "alpaca"


def test_resolve_unknown_surface_still_works(monkeypatch):
    """Unknown surface strings still consult MARKET_DATA_PROVIDER_<X>."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("MARKET_DATA_PROVIDER_DEMO", "yahoo")
    assert _resolve_provider_name("demo") == "yahoo"


def test_resolve_unknown_provider_falls_back(monkeypatch):
    """Typo in MARKET_DATA_PROVIDER_LIVE → fall through to global."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "schwab")
    monkeypatch.setenv("MARKET_DATA_PROVIDER_LIVE", "alpacaa")  # typo
    assert _resolve_provider_name("live") == "schwab"


def test_resolve_unknown_global_provider_falls_back_to_alpaca(monkeypatch):
    """Typo in MARKET_DATA_PROVIDER → fall back to alpaca."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "yyyahoo")
    assert _resolve_provider_name(None) == "alpaca"


def test_resolve_case_insensitive(monkeypatch):
    """Provider name lookup is case-insensitive."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("MARKET_DATA_PROVIDER_LIVE", "SCHWAB")
    assert _resolve_provider_name("live") == "schwab"
    monkeypatch.setenv("MARKET_DATA_PROVIDER_LIVE", "Yahoo")
    assert _resolve_provider_name("live") == "yahoo"


def test_resolve_empty_string_treated_as_unset(monkeypatch):
    """Empty MARKET_DATA_PROVIDER_LIVE='' shouldn't shadow the global."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "schwab")
    monkeypatch.setenv("MARKET_DATA_PROVIDER_LIVE", "")
    assert _resolve_provider_name("live") == "schwab"


# ── build_market_data_provider() — routes to the right class ────────


def test_factory_returns_alpaca_by_default(monkeypatch):
    _clear_env(monkeypatch)
    p = build_market_data_provider(
        alpaca_api_key="dummy", alpaca_secret_key="dummy",
    )
    # Use type() so the Schwab/Yahoo subclasses don't pass via isinstance().
    assert type(p) is MarketDataProvider


def test_factory_returns_yahoo_when_selected(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "yahoo")
    p = build_market_data_provider(
        alpaca_api_key="", alpaca_secret_key="",
    )
    assert isinstance(p, YahooMarketDataProvider)


def test_factory_returns_schwab_when_selected(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "schwab")
    monkeypatch.setenv("SCHWAB_CLIENT_ID", "id")
    monkeypatch.setenv("SCHWAB_CLIENT_SECRET", "secret")
    monkeypatch.setenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1:8182")
    monkeypatch.setenv("SCHWAB_TOKEN_PATH", "/tmp/nonexistent.json")
    from trading_agent.market_data_schwab import SchwabMarketDataProvider
    p = build_market_data_provider(
        alpaca_api_key="dummy", alpaca_secret_key="dummy",
    )
    assert isinstance(p, SchwabMarketDataProvider)


def test_factory_routes_per_surface(monkeypatch):
    """Mixed config: live=schwab, watchlist=alpaca, backtest=yahoo."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "alpaca")
    monkeypatch.setenv("MARKET_DATA_PROVIDER_LIVE", "schwab")
    monkeypatch.setenv("MARKET_DATA_PROVIDER_WATCHLIST", "alpaca")
    monkeypatch.setenv("MARKET_DATA_PROVIDER_BACKTEST", "yahoo")
    monkeypatch.setenv("SCHWAB_CLIENT_ID", "id")
    monkeypatch.setenv("SCHWAB_CLIENT_SECRET", "secret")
    monkeypatch.setenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1:8182")
    monkeypatch.setenv("SCHWAB_TOKEN_PATH", "/tmp/nonexistent.json")

    from trading_agent.market_data_schwab import SchwabMarketDataProvider

    live = build_market_data_provider(
        alpaca_api_key="d", alpaca_secret_key="d", surface="live")
    watch = build_market_data_provider(
        alpaca_api_key="d", alpaca_secret_key="d", surface="watchlist")
    back = build_market_data_provider(
        alpaca_api_key="d", alpaca_secret_key="d", surface="backtest")

    assert isinstance(live, SchwabMarketDataProvider)
    assert type(watch) is MarketDataProvider          # plain Alpaca
    assert isinstance(back, YahooMarketDataProvider)


# ── YahooMarketDataProvider — option methods are no-ops ──────────────


def test_yahoo_provider_options_return_none(monkeypatch):
    """Options aren't supported on Yahoo — fetch_option_chain returns None."""
    p = YahooMarketDataProvider(alpaca_api_key="", alpaca_secret_key="")
    assert p.fetch_option_chain("AAPL", "2026-05-29", "call") is None
    assert p.fetch_option_chain("XLF", "2026-05-29", "put") is None


def test_yahoo_provider_option_quotes_returns_empty():
    p = YahooMarketDataProvider(alpaca_api_key="", alpaca_secret_key="")
    assert p.fetch_option_quotes(["XLF260529C00053000"]) == {}
    assert p.fetch_option_quotes([]) == {}


def test_yahoo_provider_underlying_bid_ask_returns_none():
    """Yahoo doesn't expose live NBBO; adapter returns None."""
    p = YahooMarketDataProvider(alpaca_api_key="", alpaca_secret_key="")
    assert p.get_underlying_bid_ask("AAPL") is None


def test_yahoo_provider_accepts_empty_alpaca_creds():
    """Constructor must work with empty Alpaca creds — Yahoo path
    never calls Alpaca, and a no-creds dashboard mode should be valid."""
    p = YahooMarketDataProvider()
    assert isinstance(p, MarketDataProvider)


def test_yahoo_provider_subclasses_market_data_provider():
    """So the chain scanner / regime classifier / agent core all
    accept it through their existing MarketDataProvider type hints."""
    p = YahooMarketDataProvider(alpaca_api_key="", alpaca_secret_key="")
    assert isinstance(p, MarketDataProvider)


# ── Smoke check: the legacy import path still resolves ──────────────


def test_legacy_market_data_schwab_re_export():
    from trading_agent.market_data_factory import (
        build_market_data_provider as canonical,
    )
    from trading_agent.market_data_schwab import (
        build_market_data_provider as legacy,
    )
    assert legacy is canonical
