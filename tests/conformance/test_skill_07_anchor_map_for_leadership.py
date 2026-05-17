"""Conformance test: skill 07 — Anchor map for leadership Z-score.

Skill 07 §3 documents the lookup contract for ``leadership_anchor_for``:

  * Each known ticker maps to a sibling benchmark.
  * Sector ETFs (XL*) all anchor to SPY.
  * Broad-market ETFs (SPY/QQQ/IWM/DIA) cross-anchor (SPY↔QQQ).
  * Large-cap single names anchor to their sector ETF.
  * Unknown tickers fall back to SPY (skill 07 §4 fallback rule).

A missing or wrongly-mapped entry would produce silently-zero
leadership z-scores in the regime classifier — a real production
risk because there'd be no log line announcing the fallback.

Failure modes caught:
- Someone deletes a sector ETF entry and the regime code starts
  using SPY-vs-SPY (always zero) for that ticker
- The fallback path changes from "SPY" to None — downstream
  ``get_leadership_zscore`` would crash on the None receiver
"""

from __future__ import annotations

import pytest

from trading_agent.regime import LEADERSHIP_ANCHORS, leadership_anchor_for


class TestSkill07SectorEtfsAnchorToSpy:
    """Skill 07 §3: the 11 Select Sector SPDRs all anchor to SPY."""

    @pytest.mark.parametrize("sector_etf", [
        "XLB", "XLC", "XLE", "XLF", "XLI", "XLK",
        "XLP", "XLRE", "XLU", "XLV", "XLY",
    ])
    def test_sector_etf_anchors_to_spy(self, sector_etf: str) -> None:
        assert leadership_anchor_for(sector_etf) == "SPY", (
            f"Skill 07 §3: {sector_etf} must anchor to SPY. "
            f"If you changed this, update the skill 07 anchor table."
        )


class TestSkill07BroadMarketCrossAnchors:
    """Skill 07 §3: broad-market ETFs anchor against each other so
    there's always a meaningful z-score (SPY-vs-SPY would be zero)."""

    def test_spy_anchors_to_qqq(self) -> None:
        """SPY's leadership signal is vs the growth proxy."""
        assert leadership_anchor_for("SPY") == "QQQ"

    def test_qqq_anchors_to_spy(self) -> None:
        """QQQ's leadership signal is vs broad market."""
        assert leadership_anchor_for("QQQ") == "SPY"

    @pytest.mark.parametrize("broad_etf", ["IWM", "DIA"])
    def test_other_broad_etfs_anchor_to_spy(self, broad_etf: str) -> None:
        assert leadership_anchor_for(broad_etf) == "SPY"


class TestSkill07LargeCapsAnchorToSector:
    """Skill 07 §3: large-cap single names anchor to their sector ETF
    so the z-score measures 'leading peers,' not just 'riding the
    market.'"""

    @pytest.mark.parametrize("ticker,expected_sector", [
        ("JPM", "XLF"), ("BAC", "XLF"), ("GS", "XLF"), ("MS", "XLF"),
        ("AAPL", "XLK"), ("MSFT", "XLK"), ("NVDA", "XLK"),
    ])
    def test_known_large_cap_anchors_to_sector(
        self, ticker: str, expected_sector: str,
    ) -> None:
        assert leadership_anchor_for(ticker) == expected_sector


class TestSkill07FallbackToSpy:
    """Skill 07 §4: unknown tickers must fall back to SPY, NOT None.
    The fallback prevents the watchlist UI from silently emitting
    +0.00 z-scores when an unmapped ticker appears."""

    @pytest.mark.parametrize("unknown", [
        "FAKEXYZ", "NEW_IPO_TICKER", "SOMETHING_UNMAPPED",
    ])
    def test_unknown_ticker_falls_back_to_spy(self, unknown: str) -> None:
        result = leadership_anchor_for(unknown)
        assert result == "SPY", (
            f"Skill 07 §4: unknown ticker {unknown!r} must fall back to "
            f"SPY (got {result!r}). Returning None would crash the "
            f"watchlist UI's get_leadership_zscore call."
        )

    def test_empty_ticker_falls_back_to_spy(self) -> None:
        """Defensive: empty string shouldn't crash the lookup."""
        assert leadership_anchor_for("") == "SPY"


def test_skill_07_anchors_dict_is_non_empty() -> None:
    """Sanity: the lookup table itself is populated. If LEADERSHIP_ANCHORS
    ever ends up empty (file corruption, accidental deletion), every
    ticker would fall back to SPY and all leadership signals would
    collapse — a silent regression that would slip past unit tests
    but immediately break the watchlist UI in production."""
    assert len(LEADERSHIP_ANCHORS) >= 11, (
        f"Skill 07: anchor table has only {len(LEADERSHIP_ANCHORS)} "
        f"entries. Expected ≥11 (at minimum the 11 sector SPDRs)."
    )
