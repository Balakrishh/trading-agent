"""
market_data_factory.py — surface-aware MarketDataPort factory
==============================================================

Single entry point that every surface of the app uses to obtain a
:class:`MarketDataPort`-conforming provider.  The concrete adapter
(Alpaca, Schwab, Yahoo) is chosen via environment variables — no
provider class is imported until it's actually selected, so the
adapters' optional dependencies (Schwab OAuth, requests Session
pooling) don't load on surfaces that don't need them.

Per-surface routing
-------------------
Each surface of the app is given a ``surface`` string when it calls
the factory.  The factory looks up environment variables in this
priority order:

  1. ``MARKET_DATA_PROVIDER_<SURFACE>``  (most specific — e.g.
                                          ``MARKET_DATA_PROVIDER_LIVE``,
                                          ``MARKET_DATA_PROVIDER_WATCHLIST``,
                                          ``MARKET_DATA_PROVIDER_BACKTEST``)
  2. ``MARKET_DATA_PROVIDER``            (global default)
  3. ``"alpaca"``                        (hardcoded fallback so existing
                                          deployments keep working)

This lets an operator do exactly what the user asked for::

    MARKET_DATA_PROVIDER=alpaca               # global default
    MARKET_DATA_PROVIDER_LIVE=schwab          # agent uses Schwab real-time
    MARKET_DATA_PROVIDER_WATCHLIST=alpaca     # dashboard uses Alpaca
    MARKET_DATA_PROVIDER_BACKTEST=yahoo       # backtester uses yfinance

Adding a new surface
--------------------
Pass any new ``surface=...`` string from your callsite; the factory
will look for ``MARKET_DATA_PROVIDER_<SURFACE.upper()>`` automatically
without any code change here.

Adding a new provider
---------------------
Add a branch to :func:`build_market_data_provider`.  The provider
class must satisfy :class:`MarketDataPort` (declared in
``trading_agent/ports.py``).  Document the provider's required env
vars in the ``.env.example`` block alongside the existing entries.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from trading_agent.market_data import MarketDataProvider

logger = logging.getLogger(__name__)

# Recognised provider identifiers — case-insensitive lookup.
_KNOWN_PROVIDERS = {"alpaca", "schwab", "yahoo"}

# Recognised surface identifiers — included for log clarity only; the
# factory accepts any surface string (env-var lookup is generated from
# the string itself), but logging an unknown surface helps catch typos.
# "long_term" (skill 42 hourly portfolio digest) defaults to Schwab via
# the `default_provider` kwarg on `build_market_data_provider` — Schwab's
# options-chain coverage is better than Alpaca's indicative feed for
# the covered-call income-overlay scoring.
_KNOWN_SURFACES = {"live", "watchlist", "backtest", "long_term"}


def _resolve_provider_name(
    surface: Optional[str],
    *,
    default_provider: str = "alpaca",
) -> str:
    """
    Walk the env-var priority chain and return the resolved provider name.

    Priority:
      1. ``MARKET_DATA_PROVIDER_<SURFACE>`` (per-surface override).
      2. ``MARKET_DATA_PROVIDER`` (global override).
      3. ``default_provider`` kwarg — surface-aware fallback the call
         site supplies. Lets the long-term consumers default to
         Schwab without forcing every other surface to switch.
      4. Hardcoded ``"alpaca"`` final fallback.
    """
    surface_norm = (surface or "").strip().upper()
    if surface_norm:
        per_surface_var = f"MARKET_DATA_PROVIDER_{surface_norm}"
        per_surface = os.environ.get(per_surface_var, "").strip().lower()
        if per_surface:
            if per_surface not in _KNOWN_PROVIDERS:
                logger.warning(
                    "Unknown provider %r for %s — falling back to "
                    "global MARKET_DATA_PROVIDER.",
                    per_surface, per_surface_var,
                )
            else:
                return per_surface

    global_var = os.environ.get("MARKET_DATA_PROVIDER", "").strip().lower()
    if global_var:
        if global_var in _KNOWN_PROVIDERS:
            return global_var
        logger.warning(
            "Unknown provider %r in MARKET_DATA_PROVIDER — falling "
            "back to %s.", global_var, default_provider,
        )

    # Surface-aware default. Long-term tab + scheduler call with
    # default_provider="schwab" so the operator's preferred broker for
    # options-chain coverage wins when no env override is set.
    default = (default_provider or "alpaca").strip().lower()
    return default if default in _KNOWN_PROVIDERS else "alpaca"


def build_market_data_provider(
    *,
    alpaca_api_key: str,
    alpaca_secret_key: str,
    alpaca_data_url: str = "https://data.alpaca.markets/v2",
    alpaca_base_url: str = "https://paper-api.alpaca.markets/v2",
    surface: Optional[str] = None,
    default_provider: str = "alpaca",
) -> MarketDataProvider:
    """
    Return a concrete :class:`MarketDataPort` provider for *surface*.

    Parameters
    ----------
    alpaca_api_key, alpaca_secret_key, alpaca_data_url, alpaca_base_url
        Alpaca creds.  Required regardless of provider because:
          * The Schwab adapter still uses Alpaca for non-market-data
            primitives (backwards compat).
          * The Yahoo adapter accepts them for constructor-shape parity
            but never calls Alpaca.
    surface
        Surface identifier (e.g. ``"live"``, ``"watchlist"``,
        ``"backtest"``).  Maps to ``MARKET_DATA_PROVIDER_<SURFACE>``
        env var.  ``None`` skips the surface-specific lookup and uses
        only the global ``MARKET_DATA_PROVIDER``.

    Returns
    -------
    MarketDataProvider (or one of its subclasses)
        The concrete provider that satisfies the
        :class:`MarketDataPort` Protocol.

    Logging
    -------
    Every call logs the resolved provider at INFO so operators can see
    which adapter is wired into each surface.  Per-cycle data calls
    stay at DEBUG inside the adapters themselves.
    """
    provider_name = _resolve_provider_name(
        surface, default_provider=default_provider,
    )
    label = f"surface={surface!r}" if surface else "(no surface)"
    logger.info("MarketData factory: %s → provider=%s", label, provider_name)

    if surface and surface.lower() not in _KNOWN_SURFACES:
        logger.debug(
            "Unknown surface %r — env-var lookup still works (looks up "
            "MARKET_DATA_PROVIDER_%s), but consider standardising on "
            "live/watchlist/backtest for log clarity.",
            surface, surface.upper(),
        )

    if provider_name == "schwab":
        # Lazy import: requests Session + OAuth helper not loaded on
        # Alpaca-only deployments.
        from trading_agent.market_data_schwab import SchwabMarketDataProvider
        return SchwabMarketDataProvider(
            alpaca_api_key=alpaca_api_key,
            alpaca_secret_key=alpaca_secret_key,
            alpaca_data_url=alpaca_data_url,
            alpaca_base_url=alpaca_base_url,
        )

    if provider_name == "yahoo":
        from trading_agent.market_data_yahoo import YahooMarketDataProvider
        return YahooMarketDataProvider(
            alpaca_api_key=alpaca_api_key,
            alpaca_secret_key=alpaca_secret_key,
            alpaca_data_url=alpaca_data_url,
            alpaca_base_url=alpaca_base_url,
        )

    # Default: Alpaca
    return MarketDataProvider(
        alpaca_api_key=alpaca_api_key,
        alpaca_secret_key=alpaca_secret_key,
        alpaca_data_url=alpaca_data_url,
        alpaca_base_url=alpaca_base_url,
    )
