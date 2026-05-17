"""Conformance test: skill 10 — ADX with Wilder smoothing.

Skill 10 §2 documents the ADX-strength function used by the
watchlist regime-strength badge. Three banded thresholds:

  * ADX < 20  → weak / chop
  * 20-39     → developing trend
  * ADX ≥ 40  → strong trend

This conformance test pins:
  * The function exists at ``multi_tf_regime.adx_strength``
  * It returns None for too-short input series (skill 10 §4)
  * It produces values in the [0, 100] ADX range on synthetic data

Full numerical validation against Wilder's reference implementation
is covered by ``tests/test_multi_tf_regime.py``.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from trading_agent.multi_tf_regime import adx_strength


def test_skill_10_function_exists() -> None:
    """Skill 10 §3: ``adx_strength(bars, window=14)`` is the public entry."""
    assert callable(adx_strength)


def _flat_bars(n: int) -> pd.DataFrame:
    """Build a flat OHLCV series — no trend → ADX should be low."""
    return pd.DataFrame({
        "Open": [100.0] * n,
        "High": [100.1] * n,
        "Low":  [99.9] * n,
        "Close": [100.0] * n,
        "Volume": [1000] * n,
    })


def _trending_bars(n: int) -> pd.DataFrame:
    """Build a monotonically-rising series — strong trend → high ADX."""
    closes = np.linspace(100.0, 130.0, n)
    return pd.DataFrame({
        "Open":  closes - 0.05,
        "High":  closes + 0.10,
        "Low":   closes - 0.10,
        "Close": closes,
        "Volume": [1000] * n,
    })


def test_skill_10_returns_none_for_short_series() -> None:
    """Skill 10 §4: need at least 2 × window bars to compute ADX.
    Below that, return None rather than a degenerate value."""
    short = _flat_bars(10)
    assert adx_strength(short, window=14) is None


def test_skill_10_value_in_documented_range() -> None:
    """Skill 10 §2: ADX is a [0, 100] index. Any value outside that
    range indicates a math error."""
    bars = _trending_bars(60)
    adx = adx_strength(bars, window=14)
    assert adx is not None
    assert 0.0 <= adx <= 100.0, f"ADX {adx} outside [0, 100]"


def test_skill_10_trending_market_has_higher_adx_than_flat() -> None:
    """Skill 10 §1: stronger directional move → higher ADX. The
    documented bands rely on this monotonicity."""
    flat = adx_strength(_flat_bars(60), window=14)
    trend = adx_strength(_trending_bars(60), window=14)
    # Allow None on flat (no movement → no trend signal computable)
    assert trend is not None
    if flat is not None:
        assert trend > flat, (
            f"Skill 10: a trending series ({trend:.1f}) should produce "
            f"a higher ADX than a flat one ({flat:.1f})."
        )
