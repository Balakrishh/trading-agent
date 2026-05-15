"""sector_map.py — ticker → sector classification + per-sector position cap.

Single source of truth for sector classification. Used by:

  * ``agent.py`` to enforce a per-sector position cap alongside the
    per-ticker cap (``MAX_POSITIONS_PER_TICKER``), preventing
    over-concentration when multiple tickers in the same sector are
    in the universe (e.g., XLF + KRE both Financials).
  * ``streamlit/components.py`` to annotate the guardrail grid's
    ticker cell and add a ``Sector`` column to the Open Positions
    table.
  * Future sector-rotation diagnostics and per-sector P&L attribution.

Kept as a tiny standalone module rather than embedded in ``agent.py``
so the UI layer (``components.py``) can import the map without
pulling in the entire agent dependency graph. Match the Select Sector
SPDR taxonomy when adding new tickers — those classifications are the
de facto market standard and align with how Morningstar / Bloomberg /
S&P aggregate sector exposure.

Added 2026-05-15 in response to the GLD wide-spread incident and the
subsequent ticker-list refresh to sector-balanced ETFs.
"""

from __future__ import annotations

from typing import Dict


# Canonical sector taxonomy. Keep alphabetised within each block for
# diff-friendliness. Sector names match the Select Sector SPDR fund
# descriptions so a future contributor adding a new ticker can look up
# the official classification at sectorspdr.com without ambiguity.
TICKER_SECTOR_MAP: Dict[str, str] = {
    # ── Broad-market indices ─────────────────────────────────────────
    # These aren't "a sector" per se but each gets its own bucket so a
    # double-up of SPY + QQQ counts as two broad-market exposures
    # rather than colliding under one cap.
    "DIA":  "Broad Market",
    "IWM":  "Broad Market",
    "QQQ":  "Broad Market",
    "SPY":  "Broad Market",

    # ── Select Sector SPDRs ──────────────────────────────────────────
    "XLB":  "Materials",
    "XLC":  "Communications",
    "XLE":  "Energy",
    "XLF":  "Financials",
    "XLI":  "Industrials",
    "XLK":  "Technology",
    "XLP":  "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU":  "Utilities",
    "XLV":  "Healthcare",
    "XLY":  "Consumer Discretionary",

    # ── Sub-sector / themed ETFs ─────────────────────────────────────
    # Classified under the parent sector they correlate with — so the
    # per-sector cap blocks "two flavors of financials" or
    # "two flavors of semis" from stacking simultaneously.
    "KBE":  "Financials",     # KBW Bank
    "KRE":  "Financials",     # Regional Banks
    "SMH":  "Technology",     # Semiconductors
    "SOXX": "Technology",     # Semiconductors (alternate)
    "IBB":  "Healthcare",     # Biotech
    "XBI":  "Healthcare",     # Biotech (smaller-cap)
    "ITA":  "Industrials",    # Aerospace & Defense

    # ── Bond ETFs ────────────────────────────────────────────────────
    "HYG":  "High-Yield Bond",
    "IEF":  "Treasuries",
    "LQD":  "Corp Bond",
    "SHY":  "Treasuries",
    "TLT":  "Treasuries",

    # ── Commodity ETFs ───────────────────────────────────────────────
    # Each gets its own bucket — gold and silver behave correlatedly
    # but a portfolio still shouldn't stack two metals positions.
    "GLD":  "Gold",
    "SLV":  "Silver",
    "USO":  "Energy Commodity",
}


# Maximum simultaneous positions per sector. The default of 2 prevents
# over-concentration when multiple tickers in the same sector are in
# the universe (e.g., XLF + KRE both classified Financials). Tune via
# direct edit — kept as a module-level constant rather than a
# ``PresetConfig`` field because sector grouping is a global property
# of the trading universe, not a per-strategy tunable.
MAX_POSITIONS_PER_SECTOR: int = 2


def sector_for(ticker: str) -> str:
    """Return the canonical sector name for ``ticker``.

    Returns ``"Other"`` for unknown tickers so the caller never has to
    handle a ``None`` and the per-sector cap still applies to anything
    not explicitly mapped (defaulting to a generic bucket means an
    unclassified ticker can't accidentally bypass the cap).

    >>> sector_for("XLF")
    'Financials'
    >>> sector_for("KRE")
    'Financials'
    >>> sector_for("UNKNOWN_TICKER")
    'Other'
    >>> sector_for("")
    'Other'
    """
    if not ticker:
        return "Other"
    return TICKER_SECTOR_MAP.get(ticker.upper(), "Other")
