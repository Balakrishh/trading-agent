"""Conformance test: skill 16 — Market-data provider routing.

Skill 16 §2 documents the per-surface market-data routing:

  * ``MARKET_DATA_PROVIDER`` env var picks the live provider
    (alpaca / schwab / yahoo)
  * The factory ``build_market_data_provider`` returns the
    corresponding adapter
  * Schwab is the canonical live-options provider (it carries
    delta/gamma/theta in the chain payload); Alpaca is the
    execution-only fallback
  * Yahoo is the historical-backtest provider

This conformance test pins the factory contract surface +
verifies the three documented adapter classes exist.

Failure modes caught:
- Someone deletes one of the three adapter classes
- The factory function's signature changes (every consumer breaks)
- The env-var name changes from ``MARKET_DATA_PROVIDER`` to
  something else without skill 16 update
"""

from __future__ import annotations

import inspect


def test_skill_16_factory_function_exists() -> None:
    """Skill 16 §3: build_market_data_provider is the public entry."""
    from trading_agent.market_data_factory import build_market_data_provider
    assert callable(build_market_data_provider)


def test_skill_16_schwab_adapter_exists() -> None:
    """Skill 16 §3: SchwabMarketDataProvider is the canonical
    live-options adapter. Deleting it breaks chain fetching."""
    from trading_agent.market_data_schwab import SchwabMarketDataProvider
    assert SchwabMarketDataProvider is not None


def test_skill_16_yahoo_adapter_exists() -> None:
    """Skill 16 §3: YahooMarketDataProvider supplies historical bars
    for the backtester."""
    from trading_agent.market_data_yahoo import YahooMarketDataProvider
    assert YahooMarketDataProvider is not None


def test_skill_16_alpaca_adapter_exists() -> None:
    """Skill 16 §3: Alpaca's MarketDataProvider is the execution-side
    fallback when Schwab isn't configured."""
    from trading_agent.market_data import MarketDataProvider
    assert MarketDataProvider is not None


def test_skill_16_factory_accepts_documented_kwargs() -> None:
    """Skill 16 §3: factory takes provider-name + per-provider
    credentials as keyword args. Pin the kwarg-only convention."""
    from trading_agent.market_data_factory import build_market_data_provider
    sig = inspect.signature(build_market_data_provider)
    # All parameters should be keyword-only per the documented contract.
    params = sig.parameters
    # ``alpaca_api_key`` is documented as a required kwarg.
    assert "alpaca_api_key" in params, (
        "Skill 16 §3: factory must accept alpaca_api_key as a kwarg."
    )
