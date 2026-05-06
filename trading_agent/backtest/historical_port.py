"""
historical_port.py — historical bar feed with hard look-ahead protection.

Wraps yfinance for daily / intraday OHLCV and exposes a *cursor-bound*
view: every fetch is filtered against ``cursor_t`` and the call raises
``LookaheadError`` if the caller asks for a window that contains
timestamps in the future relative to the simulator's clock. This is the
single most important invariant for backtest credibility — without it,
a stray ``df.iloc[-1]`` somewhere in the regime classifier would silently
peek tomorrow's close while pretending to decide on today's data.

Two distinct datasources, one cursor
------------------------------------
* ``fetch_underlying_daily(ticker, lookback_days)`` — long-window daily
  OHLCV used for SMA-200 / regime classification. Calls yfinance once
  per ticker per backtest, caches in-memory.
* ``fetch_underlying_intraday(ticker, interval)`` — 5-min bars for the
  intraday-decision cadence. Yahoo only carries 5m back ~30 days
  (see ``market_data.SUPPORTED_INTRADAY_INTERVALS``); the runner
  decides intraday-vs-daily based on the request range and only calls
  this method when the range fits.
* ``fetch_vix_daily(lookback_days)`` — ^VIX close series used by
  ``sim_position`` for VIX-proxy IV scaling on re-marks. Same yfinance
  source the live agent uses, just historically.

Why we *don't* fetch options data here
--------------------------------------
There is no historical option-chain endpoint in any project data
source (see PROJECT_MANIFEST.md and ``market_data.fetch_option_chain``).
Synthetic chains are constructed in ``synthetic_chain.py`` from
(spot, IV, strike grid) — never fetched. Treating chains as
*reconstructed* rather than *queried* is what makes the Option-E
hybrid cadence feasible at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LookaheadError(RuntimeError):
    """
    Raised when a historical fetch would expose data the simulator
    isn't allowed to see at its current cursor position.

    A single LookaheadError that escapes a backtest run is grounds to
    treat *every* result of that run as suspect — by definition the
    simulator just behaved better than the live agent could have. Tests
    in ``tests/test_backtest/`` deliberately try to provoke this
    exception to prove the cursor seal is honest.
    """


@dataclass
class _CachedFrame:
    """Single yfinance pull memoised in-process for the run's lifetime."""
    df: pd.DataFrame
    fetched_at: datetime


class HistoricalPort:
    """
    Cursor-bound historical data adapter.

    The runner calls :meth:`set_cursor` at the top of each event;
    every subsequent ``fetch_*`` call is filtered to ``df.index <=
    cursor_t``. If the caller would receive zero rows, an
    ``InsufficientDataError``-style ``ValueError`` is raised so the
    runner can skip the cycle (mirrors the live agent's behaviour
    when ``MarketDataProvider`` returns < 200 bars).

    Note on test pluggability: the runner can pass a ``yf_loader``
    callable so unit tests inject a deterministic OHLCV frame without
    hitting Yahoo. Default loader uses ``yfinance.download`` with the
    same ``auto_adjust=False`` the live agent uses (see
    ``backtest_ui.py:21`` for the parity-pinning comment).
    """

    DEFAULT_DAILY_LOOKBACK = 365  # ≥ 200 trading days for SMA-200 warmup

    def __init__(self, *, yf_loader=None):
        self._cursor_t: Optional[datetime] = None
        self._daily_cache: Dict[str, _CachedFrame] = {}
        self._intraday_cache: Dict[tuple[str, str], _CachedFrame] = {}
        self._vix_cache: Optional[_CachedFrame] = None
        self._yf_loader = yf_loader

    # ------------------------------------------------------------------
    # Cursor mechanics
    # ------------------------------------------------------------------

    def set_cursor(self, t: datetime) -> None:
        """
        Move the cursor to ``t`` (naive datetime in US/Eastern, matching
        ``clock.iter_events``). All subsequent fetches see only data
        with timestamps ``<= t``.
        """
        self._cursor_t = t

    @property
    def cursor(self) -> Optional[datetime]:
        return self._cursor_t

    def _require_cursor(self) -> datetime:
        if self._cursor_t is None:
            raise RuntimeError(
                "HistoricalPort: cursor not set. Call set_cursor() before "
                "any fetch_* method — this is the lookahead guard."
            )
        return self._cursor_t

    # ------------------------------------------------------------------
    # yfinance loader (test-overridable)
    # ------------------------------------------------------------------

    def _load_yf(self, ticker: str, *,
                 start: date, end: date, interval: str) -> pd.DataFrame:
        """
        Single-ticker historical fetch.  Wrapped so ``yf_loader`` can be
        stubbed in tests. ``auto_adjust=False`` keeps OHLCV un-adjusted —
        live agent does the same so daily SMAs match.
        """
        if self._yf_loader is not None:
            return self._yf_loader(ticker, start=start, end=end, interval=interval)
        # Defer import so the package can be imported in test environments
        # without yfinance installed (skipping any tests that would
        # actually network).
        import yfinance as yf  # type: ignore[import]
        df = yf.download(
            ticker, start=start, end=end + timedelta(days=1),
            interval=interval, auto_adjust=False, progress=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        # yfinance occasionally returns column-multi-index (one level per
        # ticker) even for single-ticker pulls. Flatten to plain columns.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Strip tz so cursor comparisons against naive datetimes work
        # uniformly. Yahoo intraday is US/Eastern; daily has no tz.
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    # ------------------------------------------------------------------
    # Daily underlying OHLCV — the SMA-200 / regime feed
    # ------------------------------------------------------------------

    def fetch_underlying_daily(self, ticker: str,
                               lookback_days: int = DEFAULT_DAILY_LOOKBACK
                               ) -> pd.DataFrame:
        """
        Return daily OHLCV up to and including the cursor date. The
        returned frame is a *copy* — caller can mutate without
        polluting the cache.
        """
        cursor_t = self._require_cursor()
        cache_key = ticker
        cached = self._daily_cache.get(cache_key)
        if cached is None:
            # One-shot fetch covers the whole backtest window. We pull
            # ``lookback_days`` before the cursor's date and 1 year after,
            # which lets the cursor advance through the run without
            # re-fetching. The +1y slack handles any window the runner is
            # likely to use without re-network.
            start = cursor_t.date() - timedelta(days=lookback_days)
            end = cursor_t.date() + timedelta(days=400)
            df = self._load_yf(ticker, start=start, end=end, interval="1d")
            cached = _CachedFrame(df=df, fetched_at=datetime.utcnow())
            self._daily_cache[cache_key] = cached
        df = cached.df
        if df is None or df.empty:
            return pd.DataFrame()
        # The look-ahead seal — never return rows past the cursor's date.
        # Use date-only comparison since daily bars are EOD timestamps.
        cutoff = pd.Timestamp(cursor_t.date())
        sliced = df.loc[df.index <= cutoff]
        return sliced.copy()

    # ------------------------------------------------------------------
    # Intraday underlying OHLCV — only when the runner is in 5m mode
    # ------------------------------------------------------------------

    def fetch_underlying_intraday(self, ticker: str,
                                  interval: str = "5m") -> pd.DataFrame:
        """
        Return intraday OHLCV up to and including the cursor *minute*.
        Caller is responsible for choosing an interval Yahoo supports
        for the requested range — see ``trading_agent.market_data
        .SUPPORTED_INTRADAY_INTERVALS``.
        """
        cursor_t = self._require_cursor()
        key = (ticker, interval)
        cached = self._intraday_cache.get(key)
        if cached is None:
            # 5m has a 30-day max; pull the maximum window the runner
            # could possibly need (60 days back from cursor, capped) so
            # walk-forward never re-fetches.
            window_days = 30 if interval == "5m" else 60
            start = cursor_t.date() - timedelta(days=window_days)
            end = cursor_t.date() + timedelta(days=1)
            df = self._load_yf(ticker, start=start, end=end, interval=interval)
            cached = _CachedFrame(df=df, fetched_at=datetime.utcnow())
            self._intraday_cache[key] = cached
        df = cached.df
        if df is None or df.empty:
            return pd.DataFrame()
        cutoff = pd.Timestamp(cursor_t)
        sliced = df.loc[df.index <= cutoff]
        return sliced.copy()

    # ------------------------------------------------------------------
    # ^VIX daily close — drives the VIX-proxy IV scaling in sim_position
    # ------------------------------------------------------------------

    def fetch_vix_daily(self, lookback_days: int = DEFAULT_DAILY_LOOKBACK
                        ) -> pd.Series:
        """
        Return ^VIX daily close as a Series indexed by date. Used by
        ``sim_position.SimPosition.remark`` to scale IV between
        ``vix_entry`` and ``vix_t`` — the cheap, observable, dependency-
        free σ-regime signal we agreed to use after rejecting the
        scipy-heavy alternatives.
        """
        cursor_t = self._require_cursor()
        if self._vix_cache is None:
            start = cursor_t.date() - timedelta(days=lookback_days)
            end = cursor_t.date() + timedelta(days=400)
            df = self._load_yf("^VIX", start=start, end=end, interval="1d")
            self._vix_cache = _CachedFrame(df=df, fetched_at=datetime.utcnow())
        df = self._vix_cache.df
        if df is None or df.empty:
            return pd.Series(dtype=float)
        cutoff = pd.Timestamp(cursor_t.date())
        return df.loc[df.index <= cutoff, "Close"].copy()

    def vix_at(self, t: datetime) -> Optional[float]:
        """
        Look up the most recent VIX close at-or-before ``t``. Returns
        None if VIX history isn't loaded yet at the cursor — caller
        treats this as "no scaling, use σ_entry as-is".
        """
        series = self.fetch_vix_daily()
        if series.empty:
            return None
        cutoff = pd.Timestamp(t.date())
        eligible = series.loc[series.index <= cutoff]
        if eligible.empty:
            return None
        return float(eligible.iloc[-1])


__all__ = [
    "HistoricalPort",
    "LookaheadError",
]
