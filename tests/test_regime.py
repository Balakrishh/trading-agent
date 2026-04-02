"""Tests for regime classification logic."""

import pytest
from unittest.mock import MagicMock, patch

from trading_agent.regime import RegimeClassifier, Regime
from trading_agent.market_data import MarketDataProvider


class TestRegimeClassification:
    """Test that regime is correctly identified from price data."""

    def _make_classifier(self, price_data):
        """Helper: build a classifier with mocked data provider."""
        provider = MagicMock(spec=MarketDataProvider)
        provider.fetch_historical_prices.return_value = price_data
        # get_current_price must return a float — MagicMock causes TypeError
        # in comparisons inside the classifier
        provider.get_current_price.return_value = float(
            price_data["Close"].iloc[-1])
        # Static methods must be assigned directly so they behave as callables
        provider.compute_sma = MarketDataProvider.compute_sma
        provider.compute_rsi = MarketDataProvider.compute_rsi
        provider.compute_bollinger_bands = MarketDataProvider.compute_bollinger_bands
        provider.sma_slope = MarketDataProvider.sma_slope
        return RegimeClassifier(provider)

    def test_bullish_regime(self, bullish_prices):
        classifier = self._make_classifier(bullish_prices)
        result = classifier.classify("SPY")
        assert result.regime == Regime.BULLISH
        assert result.current_price > result.sma_200
        assert result.sma_50_slope > 0

    def test_bearish_regime(self, bearish_prices):
        classifier = self._make_classifier(bearish_prices)
        result = classifier.classify("SPY")
        assert result.regime == Regime.BEARISH
        assert result.current_price < result.sma_200
        assert result.sma_50_slope < 0

    def test_sideways_regime(self, sideways_prices):
        classifier = self._make_classifier(sideways_prices)
        result = classifier.classify("SPY")
        assert result.regime == Regime.SIDEWAYS

    def test_analysis_contains_reasoning(self, bullish_prices):
        classifier = self._make_classifier(bullish_prices)
        result = classifier.classify("SPY")
        assert len(result.reasoning) > 0
        assert isinstance(result.rsi_14, float)
        assert isinstance(result.bollinger_width, float)

    def test_new_signal_fields_default(self, bullish_prices):
        """New signal fields are always present with reasonable defaults."""
        classifier = self._make_classifier(bullish_prices)
        result = classifier.classify("SPY")
        assert isinstance(result.mean_reversion_signal, bool)
        assert result.mean_reversion_direction in ("", "upper", "lower")
        assert isinstance(result.relative_strength_vs_spy, float)
        assert isinstance(result.relative_strength_vs_qqq, float)


class TestMeanReversionDetection:
    """Mean reversion regime fires when price touches the 3-std Bollinger Band."""

    def _make_classifier_at_price(self, prices, current_price):
        """Build a classifier that returns *current_price* as get_current_price."""
        provider = MagicMock(spec=MarketDataProvider)
        provider.fetch_historical_prices.return_value = prices
        provider.get_current_price.return_value = float(current_price)
        provider.compute_sma = MarketDataProvider.compute_sma
        provider.compute_rsi = MarketDataProvider.compute_rsi
        provider.compute_bollinger_bands = MarketDataProvider.compute_bollinger_bands
        provider.sma_slope = MarketDataProvider.sma_slope
        # No 5-min return support so RS stays 0
        del provider.get_5min_return
        return RegimeClassifier(provider)

    def test_upper_3std_touch_gives_mean_reversion(self, bullish_prices):
        from trading_agent.regime import Regime
        import numpy as np
        # Push price well above the 3-std upper band
        close = bullish_prices["Close"]
        mean = float(close.rolling(20).mean().iloc[-1])
        std = float(close.rolling(20).std().iloc[-1])
        price_above_3std = mean + 3.5 * std

        classifier = self._make_classifier_at_price(bullish_prices, price_above_3std)
        result = classifier.classify("SPY")
        assert result.regime == Regime.MEAN_REVERSION
        assert result.mean_reversion_signal is True
        assert result.mean_reversion_direction == "upper"

    def test_lower_3std_touch_gives_mean_reversion(self, bullish_prices):
        from trading_agent.regime import Regime
        close = bullish_prices["Close"]
        mean = float(close.rolling(20).mean().iloc[-1])
        std = float(close.rolling(20).std().iloc[-1])
        price_below_3std = mean - 3.5 * std

        classifier = self._make_classifier_at_price(bullish_prices, price_below_3std)
        result = classifier.classify("SPY")
        assert result.regime == Regime.MEAN_REVERSION
        assert result.mean_reversion_signal is True
        assert result.mean_reversion_direction == "lower"

    def test_normal_price_no_mean_reversion(self, bullish_prices):
        from trading_agent.regime import Regime
        close = bullish_prices["Close"]
        mean_price = float(close.iloc[-1])  # last close is within bands
        classifier = self._make_classifier_at_price(bullish_prices, mean_price)
        result = classifier.classify("SPY")
        assert result.regime != Regime.MEAN_REVERSION
        assert result.mean_reversion_signal is False
