"""
Phase II — CLASSIFY
Determines the current Market Regime (Bullish, Bearish, Sideways)
based on SMA positioning and slope analysis.
"""

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from trading_agent.market_data import MarketDataProvider

logger = logging.getLogger(__name__)


class Regime(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    MEAN_REVERSION = "mean_reversion"   # 3-std Bollinger Band touch


@dataclass
class RegimeAnalysis:
    """Structured output of the regime classification."""
    regime: Regime
    current_price: float
    sma_50: float
    sma_200: float
    sma_50_slope: float
    rsi_14: float
    bollinger_width: float
    reasoning: str
    # Mean reversion signal — set when price touches a 3-std Bollinger Band
    mean_reversion_signal: bool = False
    mean_reversion_direction: str = ""    # "upper" or "lower"
    # Relative strength vs benchmarks (5-min return differential)
    relative_strength_vs_spy: float = 0.0
    relative_strength_vs_qqq: float = 0.0


class RegimeClassifier:
    """
    Classifies market regime using the following rules:

    BULLISH  — Price > SMA-200  AND  SMA-50 slope > 0
    BEARISH  — Price < SMA-200  AND  SMA-50 slope < 0
    SIDEWAYS — Everything else  (price between SMAs or narrow Bollinger Bands)
    """

    BOLLINGER_NARROW_THRESHOLD = 0.04  # 4 % width considered "narrow"

    def __init__(self, data_provider: MarketDataProvider):
        self.data = data_provider

    def classify(self, ticker: str) -> RegimeAnalysis:
        """Run full regime analysis for *ticker*.

        Data sources:
          - yfinance: 200-day historical OHLCV for SMA, RSI, Bollinger Bands
          - Alpaca snapshot: real-time current price for regime decision
          - Alpaca 5-min bars: relative strength vs SPY/QQQ
        """
        df = self.data.fetch_historical_prices(ticker, period_days=200)
        close = df["Close"]

        sma_50 = self.data.compute_sma(close, 50)
        sma_200 = self.data.compute_sma(close, 200)
        rsi = self.data.compute_rsi(close, 14)
        upper, middle, lower = self.data.compute_bollinger_bands(close, 20, 2.0)
        sma_50_slope = self.data.sma_slope(sma_50, lookback=5)

        # 3-standard-deviation Bollinger Bands for mean-reversion detection
        upper_3std, _, lower_3std = self.data.compute_bollinger_bands(close, 20, 3.0)

        # Use Alpaca real-time price for current price (falls back to
        # historical close if the snapshot API is unavailable)
        current_price = self.data.get_current_price(ticker)
        current_sma_50 = float(sma_50.iloc[-1])
        current_sma_200 = float(sma_200.iloc[-1])
        current_rsi = float(rsi.iloc[-1])

        # Bollinger band width as percentage of price
        bb_width = float((upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1])

        # Mean reversion: check if price has touched the 3-std band
        u3 = float(upper_3std.iloc[-1])
        l3 = float(lower_3std.iloc[-1])
        mr_upper = current_price >= u3
        mr_lower = current_price <= l3
        mean_reversion_signal = mr_upper or mr_lower
        mean_reversion_direction = ("upper" if mr_upper
                                    else "lower" if mr_lower else "")

        # Relative strength: 5-min return vs SPY and QQQ
        rs_vs_spy = 0.0
        rs_vs_qqq = 0.0
        if ticker not in ("SPY", "QQQ") and hasattr(self.data, "get_5min_return"):
            ticker_ret = self.data.get_5min_return(ticker)
            spy_ret = self.data.get_5min_return("SPY")
            qqq_ret = self.data.get_5min_return("QQQ")
            if ticker_ret is not None and spy_ret is not None:
                rs_vs_spy = ticker_ret - spy_ret
            if ticker_ret is not None and qqq_ret is not None:
                rs_vs_qqq = ticker_ret - qqq_ret

        regime, reasoning = self._determine_regime(
            current_price, current_sma_50, current_sma_200,
            sma_50_slope, bb_width, mean_reversion_signal,
            mean_reversion_direction, u3, l3,
        )

        analysis = RegimeAnalysis(
            regime=regime,
            current_price=current_price,
            sma_50=current_sma_50,
            sma_200=current_sma_200,
            sma_50_slope=sma_50_slope,
            rsi_14=current_rsi,
            bollinger_width=bb_width,
            reasoning=reasoning,
            mean_reversion_signal=mean_reversion_signal,
            mean_reversion_direction=mean_reversion_direction,
            relative_strength_vs_spy=rs_vs_spy,
            relative_strength_vs_qqq=rs_vs_qqq,
        )
        logger.info(
            "[%s] Regime → %s | Price=%.2f SMA50=%.2f SMA200=%.2f "
            "Slope=%.4f BB=%.4f RSI=%.1f MR=%s RS_SPY=%.4f",
            ticker, regime.value, current_price, current_sma_50,
            current_sma_200, sma_50_slope, bb_width, current_rsi,
            mean_reversion_direction or "none", rs_vs_spy,
        )
        return analysis

    def _determine_regime(self, price: float, sma50: float, sma200: float,
                          slope_50: float, bb_width: float,
                          mean_reversion_signal: bool = False,
                          mean_reversion_direction: str = "",
                          upper_3std: float = 0.0,
                          lower_3std: float = 0.0):
        """Core classification logic.

        Mean reversion takes highest priority — a 3-std Bollinger Band
        touch overrides any trend signal.
        """
        # Highest priority: 3-std Bollinger Band touch → Mean Reversion
        if mean_reversion_signal:
            dir_label = ("above upper" if mean_reversion_direction == "upper"
                         else "below lower")
            return (Regime.MEAN_REVERSION,
                    f"Price ({price:.2f}) is {dir_label} 3-std Bollinger Band "
                    f"(upper={upper_3std:.2f}, lower={lower_3std:.2f}). "
                    f"Expect mean reversion.")

        # Sideways check: narrow Bollinger Bands
        if bb_width < self.BOLLINGER_NARROW_THRESHOLD:
            return (Regime.SIDEWAYS,
                    f"Bollinger Band width ({bb_width:.4f}) is below "
                    f"threshold ({self.BOLLINGER_NARROW_THRESHOLD}), "
                    f"indicating low volatility / sideways movement.")

        # Bullish: price above SMA-200 and SMA-50 trending up
        if price > sma200 and slope_50 > 0:
            return (Regime.BULLISH,
                    f"Price ({price:.2f}) > SMA-200 ({sma200:.2f}) and "
                    f"SMA-50 slope is positive ({slope_50:.4f}).")

        # Bearish: price below SMA-200 and SMA-50 trending down
        if price < sma200 and slope_50 < 0:
            return (Regime.BEARISH,
                    f"Price ({price:.2f}) < SMA-200 ({sma200:.2f}) and "
                    f"SMA-50 slope is negative ({slope_50:.4f}).")

        # Default: sideways
        return (Regime.SIDEWAYS,
                f"Price ({price:.2f}) is between SMAs or slope direction "
                f"doesn't confirm a trend. SMA-50={sma50:.2f}, "
                f"SMA-200={sma200:.2f}, slope={slope_50:.4f}.")
