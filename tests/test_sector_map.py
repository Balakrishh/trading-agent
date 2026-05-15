"""test_sector_map.py — pins the ticker → sector taxonomy + the sector-
cap behavior consumed by ``agent.py:run_cycle``.

The sector cap is enforced via a Set-union into ``tickers_with_positions``;
we test the predicate logic directly here. The full integration with
Stage 1.5's position fetch is exercised by ``test_agent_integration.py``.
"""
from __future__ import annotations

from trading_agent.sector_map import (
    MAX_POSITIONS_PER_SECTOR,
    TICKER_SECTOR_MAP,
    sector_for,
)


class TestSectorForLookup:
    """Direct contract on the lookup primitive."""

    def test_canonical_sector_spdr(self):
        assert sector_for("XLF") == "Financials"
        assert sector_for("XLE") == "Energy"
        assert sector_for("XLU") == "Utilities"
        assert sector_for("XLP") == "Consumer Staples"
        assert sector_for("XLB") == "Materials"

    def test_subsector_classified_under_parent(self):
        """KRE is a financial sub-sector → groups under Financials so
        the per-sector cap blocks XLF+KRE stacking."""
        assert sector_for("KRE") == "Financials"
        assert sector_for("KBE") == "Financials"
        assert sector_for("SMH") == "Technology"
        assert sector_for("SOXX") == "Technology"
        assert sector_for("IBB") == "Healthcare"

    def test_broad_market_indices_have_own_bucket(self):
        """SPY/QQQ/IWM aren't a sector — but they each get their own
        bucket called 'Broad Market' so the cap still applies."""
        for t in ("SPY", "QQQ", "IWM", "DIA"):
            assert sector_for(t) == "Broad Market"

    def test_bond_etfs_split_by_credit_quality(self):
        """Investment-grade Treasuries grouped together; high-yield is
        its own bucket because it correlates with equity risk."""
        assert sector_for("TLT") == "Treasuries"
        assert sector_for("IEF") == "Treasuries"
        assert sector_for("SHY") == "Treasuries"
        assert sector_for("HYG") == "High-Yield Bond"
        assert sector_for("LQD") == "Corp Bond"

    def test_commodities_each_get_own_bucket(self):
        """Gold and silver behave correlatedly but stacking two metals
        positions should still be prevented — each gets its own bucket."""
        assert sector_for("GLD") == "Gold"
        assert sector_for("SLV") == "Silver"

    def test_unknown_ticker_returns_other(self):
        """Unmapped tickers go to 'Other' so the per-sector cap still
        applies (defaulting to 'unknown' means an unclassified ticker
        can't accidentally bypass the cap)."""
        assert sector_for("UNKNOWN_TICKER") == "Other"
        assert sector_for("FOOBAR") == "Other"

    def test_empty_or_none_input(self):
        assert sector_for("") == "Other"

    def test_case_insensitive(self):
        """Real-world ticker symbols come uppercase from Alpaca but
        defensive: lowercase input still resolves correctly."""
        assert sector_for("xlf") == "Financials"
        assert sector_for("Xlf") == "Financials"


class TestSectorCapPolicy:
    """Pins the per-sector cap default + the data-shape invariants the
    cap-check in agent.py:run_cycle depends on."""

    def test_default_cap_is_two(self):
        """The default of 2 matches the design intent: prevent stacking
        more than two same-sector trades. Raising this requires a
        deliberate edit, not an accidental config drift."""
        assert MAX_POSITIONS_PER_SECTOR == 2

    def test_every_known_ticker_has_a_sector(self):
        """Defensive: every ticker explicitly in the map must have a
        non-empty sector string. Catches a future contributor who
        accidentally adds a ticker with a None or empty-string value."""
        for ticker, sector in TICKER_SECTOR_MAP.items():
            assert isinstance(sector, str) and sector, (
                f"{ticker} maps to invalid sector {sector!r}"
            )
            assert ticker.isupper(), (
                f"{ticker} should be uppercase (Alpaca convention)"
            )

    def test_select_sector_spdrs_all_covered(self):
        """The 11 official Select Sector SPDR ETFs all need explicit
        classification — they're the canonical sector pool."""
        spdrs = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK",
                 "XLP", "XLRE", "XLU", "XLV", "XLY"]
        for t in spdrs:
            assert t in TICKER_SECTOR_MAP, f"{t} missing from sector map"


class TestSectorCapEnforcement:
    """End-to-end logic check on the union-into-tickers_with_positions
    pattern that ``agent.py:run_cycle`` uses for the sector cap.

    We don't import agent.py here (it would pull the full dep graph);
    we just re-implement the predicate and verify it matches the
    documented contract.
    """

    @staticmethod
    def _sectors_at_cap(positions_per_sector, cap):
        return {s for s, n in positions_per_sector.items() if n >= cap}

    @staticmethod
    def _tickers_blocked(universe, sectors_blocked):
        return {t for t in universe if sector_for(t) in sectors_blocked}

    def test_two_financials_blocks_third(self):
        """User has XLF and KRE open (both Financials, count=2). The
        sector cap is 2 → Financials is at cap → KBE (also Financials)
        should be blocked from new entry."""
        universe = ["XLF", "KRE", "KBE", "XLE", "XLU"]
        positions = {"Financials": 2, "Energy": 1}
        sectors_blocked = self._sectors_at_cap(positions, 2)
        blocked = self._tickers_blocked(universe, sectors_blocked)
        assert "KBE" in blocked
        assert "XLE" not in blocked   # Energy only has 1 open
        assert "XLU" not in blocked   # Utilities has 0

    def test_one_financial_one_energy_blocks_neither(self):
        """Single position in two different sectors → cap not reached."""
        universe = ["XLF", "KRE", "XLE", "XLU"]
        positions = {"Financials": 1, "Energy": 1}
        sectors_blocked = self._sectors_at_cap(positions, 2)
        assert sectors_blocked == set()
        assert self._tickers_blocked(universe, sectors_blocked) == set()

    def test_unknown_ticker_classified_other(self):
        """A ticker not in the sector map should still participate in
        the cap via the 'Other' bucket — prevents stacking arbitrarily
        many unclassified tickers."""
        universe = ["FAKE1", "FAKE2", "FAKE3"]
        positions = {"Other": 2}
        sectors_blocked = self._sectors_at_cap(positions, 2)
        assert sectors_blocked == {"Other"}
        # Future "FAKE3" attempts would be blocked because it's Other
        assert "FAKE3" in self._tickers_blocked(universe, sectors_blocked)

    def test_recommended_universe_diversifies(self):
        """The 6-ticker recommended universe should NOT trigger the
        sector cap with one position open per ticker (every one is its
        own sector → no doubling)."""
        universe = ["XLF", "XLE", "XLU", "XLP", "XLB", "HYG"]
        # Every ticker has its own sector
        sectors = {sector_for(t) for t in universe}
        assert len(sectors) == 6   # all distinct
        # Even with all 6 open simultaneously, no sector is at cap
        positions = {s: 1 for s in sectors}
        assert self._sectors_at_cap(positions, 2) == set()
