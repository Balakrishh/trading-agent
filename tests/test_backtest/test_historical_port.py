"""
Tests for ``trading_agent.backtest.historical_port``.

Pin the cursor seal: data past the cursor is never visible. We use a
stub yf_loader so the test is hermetic.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from trading_agent.backtest.historical_port import HistoricalPort


def _make_daily_frame(start: date, n_days: int) -> pd.DataFrame:
    """Generate a dummy OHLCV frame for ``n_days`` consecutive days."""
    idx = pd.date_range(start=start, periods=n_days, freq="D")
    rng = np.random.default_rng(42)
    close = 100 + rng.standard_normal(n_days).cumsum()
    return pd.DataFrame({
        "Open":   close + 0.5,
        "High":   close + 1.0,
        "Low":    close - 1.0,
        "Close":  close,
        "Volume": (1_000_000 + rng.integers(0, 500_000, n_days)),
    }, index=idx)


def _stub_loader(df: pd.DataFrame):
    """Return a yf_loader stub that yields the same df for any request."""
    def _load(ticker, *, start, end, interval):
        # Honour the [start, end] bracket so the cache is realistic.
        return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    return _load


# --------------------------------------------------------------------------
# Cursor enforcement
# --------------------------------------------------------------------------

class TestCursor:
    def test_fetch_without_cursor_raises(self):
        port = HistoricalPort()
        with pytest.raises(RuntimeError, match="cursor not set"):
            port.fetch_underlying_daily("SPY")

    def test_cursor_filters_future_rows(self):
        df = _make_daily_frame(date(2025, 1, 1), n_days=200)
        port = HistoricalPort(yf_loader=_stub_loader(df))
        # Cursor at day 50 — should see 50 rows max
        port.set_cursor(datetime(2025, 2, 19, 16, 0))  # ~day 50
        result = port.fetch_underlying_daily("SPY")
        assert not result.empty
        # Last row's timestamp must be ≤ cursor's date
        assert result.index.max().date() <= date(2025, 2, 19)
        # And rows past cursor are absent.
        assert (result.index.max().date() < date(2025, 12, 31))

    def test_advancing_cursor_widens_window(self):
        df = _make_daily_frame(date(2025, 1, 1), n_days=300)
        port = HistoricalPort(yf_loader=_stub_loader(df))
        port.set_cursor(datetime(2025, 3, 1, 16, 0))
        first = port.fetch_underlying_daily("SPY")
        port.set_cursor(datetime(2025, 6, 1, 16, 0))
        second = port.fetch_underlying_daily("SPY")
        assert len(second) > len(first)


# --------------------------------------------------------------------------
# VIX lookup
# --------------------------------------------------------------------------

class TestVixLookup:
    def test_vix_at_returns_most_recent_close(self):
        df = _make_daily_frame(date(2025, 1, 1), n_days=100)
        port = HistoricalPort(yf_loader=_stub_loader(df))
        port.set_cursor(datetime(2025, 3, 1, 16, 0))
        vix = port.vix_at(datetime(2025, 3, 1, 16, 0))
        assert vix is not None
        assert vix > 0

    def test_vix_at_returns_none_when_no_data(self):
        port = HistoricalPort(yf_loader=lambda *a, **kw: pd.DataFrame())
        port.set_cursor(datetime(2025, 3, 1, 16, 0))
        assert port.vix_at(datetime(2025, 3, 1, 16, 0)) is None


# --------------------------------------------------------------------------
# Caching — second fetch reuses the first
# --------------------------------------------------------------------------

def test_daily_cache_hits_reuse_first_fetch():
    df = _make_daily_frame(date(2025, 1, 1), n_days=100)
    call_count = {"n": 0}

    def counting_loader(ticker, *, start, end, interval):
        call_count["n"] += 1
        return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

    port = HistoricalPort(yf_loader=counting_loader)
    port.set_cursor(datetime(2025, 2, 1, 16, 0))
    port.fetch_underlying_daily("SPY")
    port.fetch_underlying_daily("SPY")
    port.fetch_underlying_daily("SPY")
    # First call hits network; subsequent calls hit cache.
    assert call_count["n"] == 1
