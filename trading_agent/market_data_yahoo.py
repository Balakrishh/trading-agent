"""
market_data_yahoo.py — Yahoo-only MarketDataPort adapter
=========================================================

Why this exists
---------------
Some surfaces of the app don't need real-time options data — only
historical bars and intraday OHLCV.  The Watchlist tab is the
canonical example: it draws charts and computes regime indicators,
but never fetches an option chain or sizes a trade.  For those
surfaces, hammering the Alpaca/Schwab data plane is gratuitous,
and the operator may want a no-credentials fallback so the dashboard
keeps working when the broker is unreachable.

``YahooMarketDataProvider`` subclasses :class:`MarketDataProvider` so
it inherits the parts that are already vendor-neutral:

  * ``fetch_historical_prices`` — yfinance-backed daily OHLCV.
  * ``fetch_intraday_bars`` — yfinance-backed 5m/15m/30m/60m/1h/4h bars.
  * All ``compute_*`` indicator helpers (pure pandas math).

It overrides:

  * ``fetch_batch_snapshots`` / ``get_current_price`` — fall back to
    yfinance's ``fast_info`` so spot prices come from the same place
    as the bars (no Alpaca call required).
  * ``get_underlying_bid_ask`` — Yahoo's free feed does not publish
    a real-time NBBO, so this returns ``None`` and the live trading
    layer treats the underlying as "no quote".  Use this provider
    only on surfaces that don't need the bid/ask gate.
  * ``fetch_option_chain`` / ``fetch_option_quotes`` — Yahoo's option
    chain is unreliable (no Greeks on free tier; yfinance scrapes
    HTML which breaks frequently), so these return ``None`` / ``{}``
    and log a warning.  Routing the *agent's live loop* through
    Yahoo is therefore unsupported — pick Alpaca or Schwab for
    surfaces that trade options.

Crash safety
------------
Subclassing keeps the constructor signature identical so the factory
can swap providers without changing callsites.  Empty Alpaca creds
are accepted because the Yahoo overrides never call Alpaca.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from trading_agent.market_data import (
    SNAPSHOT_TTL,
    MarketDataProvider,
)

try:
    import yfinance as yf
except ImportError:
    yf = None  # Tests use mocked data

logger = logging.getLogger(__name__)


class YahooMarketDataProvider(MarketDataProvider):
    """
    Yahoo-only MarketDataPort.  Suitable for surfaces that need
    historical bars + indicators but not real-time options data
    (Watchlist, Backtest UI charts, demo mode without broker creds).
    """

    def __init__(self, *,
                 alpaca_api_key: str = "",
                 alpaca_secret_key: str = "",
                 alpaca_data_url: str = "https://data.alpaca.markets/v2",
                 alpaca_base_url: str = "https://paper-api.alpaca.markets/v2"):
        # Alpaca creds are accepted for constructor-shape parity but
        # the Yahoo overrides never call Alpaca — empty values are fine.
        super().__init__(
            alpaca_api_key=alpaca_api_key,
            alpaca_secret_key=alpaca_secret_key,
            alpaca_data_url=alpaca_data_url,
            alpaca_base_url=alpaca_base_url,
        )

    # ------------------------------------------------------------------
    # Spot prices via yfinance — no Alpaca round-trip
    # ------------------------------------------------------------------
    def fetch_batch_snapshots(self, tickers: List[str]) -> Dict[str, float]:
        """Last regular-market price per ticker via yfinance ``fast_info``."""
        if not tickers:
            return {}
        if yf is None:
            logger.warning("yfinance not installed — Yahoo snapshot skipped.")
            return {}
        out: Dict[str, float] = {}
        now = time.monotonic()
        for t in tickers:
            cached = self._snapshot_cache.get(t)
            if cached and (now - cached[1]) < SNAPSHOT_TTL:
                out[t] = cached[0]
                continue
            try:
                tk = yf.Ticker(t)
                fi = tk.fast_info
                price = (
                    fi.get("last_price")
                    or fi.get("regular_market_price")
                    or fi.get("previous_close")
                    or 0.0
                )
                if price:
                    out[t] = float(price)
                    self._snapshot_cache[t] = (float(price), now)
            except Exception as exc:                       # noqa: BLE001
                logger.debug("[%s] Yahoo snapshot failed: %s", t, exc)
        return out

    def get_current_price(self, ticker: str) -> float:
        return float(self.fetch_batch_snapshots([ticker]).get(ticker, 0.0))

    # ------------------------------------------------------------------
    # Live bid/ask: not available on Yahoo's free feed
    # ------------------------------------------------------------------
    def get_underlying_bid_ask(self, ticker: str) -> Optional[Tuple[float, float]]:
        """
        Yahoo doesn't publish a real-time NBBO; return ``None`` so the
        liquidity guardrail can short-circuit gracefully.  The agent's
        risk manager treats ``None`` as "no quote" — callers should
        pick Alpaca / Schwab if they need this gate to fire.
        """
        return None

    # ------------------------------------------------------------------
    # Options: unsupported on Yahoo
    # ------------------------------------------------------------------
    def fetch_option_chain(self, underlying: str,
                           expiration_date: str,
                           option_type: str = "put") -> Optional[list]:
        logger.warning(
            "[%s] Option chain requested while MARKET_DATA_PROVIDER=yahoo — "
            "Yahoo does not provide reliable options data with Greeks. "
            "Use MARKET_DATA_PROVIDER=alpaca or schwab for surfaces that "
            "trade options.", underlying,
        )
        return None

    def fetch_option_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        if symbols:
            logger.warning(
                "Option quote refresh requested while "
                "MARKET_DATA_PROVIDER=yahoo — returning {} for %d symbols.",
                len(symbols),
            )
        return {}

    # ------------------------------------------------------------------
    # Market hours: lightweight US-equity heuristic — yfinance has no
    # dedicated market-hours endpoint, but the watchlist surface only
    # needs a coarse "is the cash session open?" answer.
    # ------------------------------------------------------------------
    def is_market_open(self) -> bool:                      # type: ignore[override]
        """
        Coarse US-equity cash-session check (Mon–Fri, 09:30–16:00 ET).
        Doesn't account for holidays — surfaces that need exact
        market-hours data should use Alpaca/Schwab instead.
        """
        try:
            from zoneinfo import ZoneInfo
            now_et = datetime.now(ZoneInfo("America/New_York"))
        except Exception:                                  # noqa: BLE001
            return False
        if now_et.weekday() >= 5:    # Sat/Sun
            return False
        open_min = 9 * 60 + 30
        close_min = 16 * 60
        cur_min = now_et.hour * 60 + now_et.minute
        return open_min <= cur_min < close_min
