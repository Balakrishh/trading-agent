"""Conformance test: skill 08 — Leadership Z-score.

Skill 08 documents the per-ticker leadership signal: how strongly a
ticker is leading or lagging its anchor benchmark, expressed as a
z-score over the past N 5-min bars (default 20).

The full behavior requires market-data fixtures. This conformance
test pins the *contract surface* — function exists, signature is
stable, return type is a tuple or None. The integration behavior
is covered by ``tests/test_market_data.py``.

Failure modes caught:
- Someone changes the return type from Optional[Tuple[z, sigma]] to
  just float — silent None becomes 0.0 downstream and watchlist
  shows zeros instead of em-dashes
- Someone renames the function — every consumer breaks
- The LEADERSHIP_WINDOW_BARS constant gets removed
"""

from __future__ import annotations

import inspect


def test_skill_08_function_exists_and_is_callable() -> None:
    """Skill 08 §3: the public entry-point lives on MarketDataProvider."""
    from trading_agent.market_data import MarketDataProvider
    assert hasattr(MarketDataProvider, "get_leadership_zscore")
    assert callable(MarketDataProvider.get_leadership_zscore)


def test_skill_08_signature_is_stable() -> None:
    """The (ticker, anchor, window) signature is what every caller
    builds against — `regime.py`, the watchlist UI, conformance
    test for skill 07. Changing it requires updating skill 08 §3."""
    from trading_agent.market_data import MarketDataProvider
    sig = inspect.signature(MarketDataProvider.get_leadership_zscore)
    params = list(sig.parameters.keys())
    # First is self; subsequent params are ticker, anchor, window.
    assert params[0] == "self"
    assert "ticker" in params
    assert "anchor" in params
    assert "window" in params


def test_skill_08_window_default_is_documented_value() -> None:
    """Skill 08 §1: ``LEADERSHIP_WINDOW_BARS = 21`` — 20 returns + 1
    anchor close, ~105 minutes of intraday data. Changing the default
    silently shifts the z-score's sensitivity across all consumers.

    Note the constant lives as a CLASS attribute on
    ``MarketDataProvider``, not at module level — accessing it via
    the module would AttributeError."""
    from trading_agent.market_data import MarketDataProvider
    assert MarketDataProvider.LEADERSHIP_WINDOW_BARS == 21, (
        f"Skill 08 §1: LEADERSHIP_WINDOW_BARS must be 21 "
        f"(20 returns + 1 anchor close); got "
        f"{MarketDataProvider.LEADERSHIP_WINDOW_BARS}."
    )


def test_skill_08_returns_optional_tuple() -> None:
    """Skill 08 §3: return type is ``Optional[Tuple[float, float]]``
    (z-score, sigma) — Tuple shape lets the watchlist render the raw
    σ alongside z. Returning just float breaks the watchlist signature."""
    from trading_agent.market_data import MarketDataProvider
    import typing
    sig = inspect.signature(MarketDataProvider.get_leadership_zscore)
    ret = sig.return_annotation
    # Annotation may be a string under PEP 563 or the actual generic alias
    ret_str = str(ret)
    assert "Tuple" in ret_str or "tuple" in ret_str, (
        f"Skill 08 §3: get_leadership_zscore must return "
        f"Optional[Tuple[float, float]]; annotation is {ret_str!r}"
    )
