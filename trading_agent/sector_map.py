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
    "XME":  "Metals & Mining",  # Broad metals & mining (copper, steel, etc.)

    # ── International / regional ETFs ────────────────────────────────
    # Geographic buckets rather than sector buckets — a US Financials
    # cap shouldn't restrict an EAFE-wide exposure, and vice versa.
    "EFA":  "International Developed",   # iShares MSCI EAFE
    "EWJ":  "International Developed",   # iShares MSCI Japan (future-proof)
    "EEM":  "Emerging Markets",          # iShares MSCI Emerging Markets
    "FXI":  "Emerging Markets",          # iShares China Large-Cap (future-proof)

    # ── Bond ETFs ────────────────────────────────────────────────────
    "HYG":  "High-Yield Bond",
    "IEF":  "Treasuries",
    "LQD":  "Corp Bond",
    "SHY":  "Treasuries",
    "TLT":  "Treasuries",

    # ── Commodity ETFs ───────────────────────────────────────────────
    # Each gets its own bucket — gold and silver behave correlatedly
    # but a portfolio still shouldn't stack two metals positions.
    # GDX / GDXJ (gold miners) share Gold's bucket because they
    # correlate ~0.85+ with spot gold and stacking gold + miners would
    # double-up an effectively-single bet.
    "GDX":  "Gold",           # VanEck Gold Miners
    "GDXJ": "Gold",           # VanEck Junior Gold Miners (future-proof)
    "GLD":  "Gold",
    "SLV":  "Silver",
    "USO":  "Energy Commodity",

    # ── Single-name equities — long-term evaluator (skill 40) ────────
    # Added 2026-06-16 so the long-term evaluator's portfolio snapshot
    # can break a single-name book down by sector. Same Select Sector
    # SPDR taxonomy as the ETFs above. Add new tickers alphabetically
    # within their sector block for diff-friendliness.
    #
    # The per-sector cap (MAX_POSITIONS_PER_SECTOR) does NOT apply to
    # single-name long-term positions — it gates the credit-spread
    # agent's per-cycle ETF picks only. The classification is purely
    # for display + diversification analysis.
    "AAPL":  "Technology",
    "AMZN":  "Consumer Discretionary",   # AMZN sits in XLY per S&P GICS
    "AVGO":  "Technology",
    "GOOG":  "Communications",            # GOOG/GOOGL in XLC per S&P GICS
    "GOOGL": "Communications",
    "INTC":  "Technology",
    "JPM":   "Financials",
    "META":  "Communications",
    "MSFT":  "Technology",
    "NFLX":  "Communications",
    "NOK":   "Technology",                # Nokia: telecom equipment
    "NVDA":  "Technology",
    "PLTR":  "Technology",
    "SOFI":  "Financials",
    "TSLA":  "Consumer Discretionary",
    "ZS":    "Technology",

    # Common Wheel names (2026-10-05, backlog §2 "one pick per sector":
    # the screen clustered banks and telecom). Unmapped names fall back
    # to the yfinance sector via ``wheel_sector``.
    "BAC":   "Financials",
    "C":     "Financials",
    "GS":    "Financials",
    "MS":    "Financials",
    "PNC":   "Financials",
    "SCHW":  "Financials",
    "TFC":   "Financials",
    "USB":   "Financials",
    "WFC":   "Financials",
    "CMCSA": "Communications",
    "T":     "Communications",
    "TMUS":  "Communications",
    "VZ":    "Communications",
    "CL":    "Consumer Staples",
    "GIS":   "Consumer Staples",
    "KHC":   "Consumer Staples",
    "KMB":   "Consumer Staples",
    "KO":    "Consumer Staples",
    "MO":    "Consumer Staples",
    "PEP":   "Consumer Staples",
    "PG":    "Consumer Staples",
    "PM":    "Consumer Staples",
    "WMT":   "Consumer Staples",
    "ABBV":  "Healthcare",
    "BMY":   "Healthcare",
    "CVS":   "Healthcare",
    "GILD":  "Healthcare",
    "JNJ":   "Healthcare",
    "MRK":   "Healthcare",
    "PFE":   "Healthcare",
    "COP":   "Energy",
    "CVX":   "Energy",
    "OXY":   "Energy",
    "SLB":   "Energy",
    "XOM":   "Energy",
    "D":     "Utilities",
    "DUK":   "Utilities",
    "NEE":   "Utilities",
    "SO":    "Utilities",
    "CSCO":  "Technology",
    "IBM":   "Technology",
    "ORCL":  "Technology",
    "F":     "Consumer Discretionary",
    "GM":    "Consumer Discretionary",
    "NKE":   "Consumer Discretionary",
    "SBUX":  "Consumer Discretionary",

    # Themed equity ETFs landed alongside single-names.
    "NASA":  "Industrials",               # TEMA Space Innovators: aerospace + defense lean
}


# Maximum simultaneous positions per sector. The default of 2 prevents
# over-concentration when multiple tickers in the same sector are in
# the universe (e.g., XLF + KRE both classified Financials). Tune via
# direct edit — kept as a module-level constant rather than a
# ``PresetConfig`` field because sector grouping is a global property
# of the trading universe, not a per-strategy tunable.
MAX_POSITIONS_PER_SECTOR: int = 2


# yfinance ``info["sector"]`` → the SPDR names above.
_YF_SECTOR = {
    "Communication Services": "Communications",
    "Financial Services": "Financials",
    "Consumer Defensive": "Consumer Staples",
    "Consumer Cyclical": "Consumer Discretionary",
    "Basic Materials": "Materials",
}


def wheel_sector(ticker: str, info_sector=None) -> str:
    """Sector for the Wheel's one-pick-per-sector rule: the map above,
    else ``info_sector(ticker)`` (e.g. yfinance ``info["sector"]``,
    translated to SPDR names), else the ticker itself — an unknown name
    is its own sector, never lumped with other unknowns under "Other"."""
    mapped = TICKER_SECTOR_MAP.get((ticker or "").upper())
    if mapped:
        return mapped
    if info_sector is not None:
        try:
            raw = info_sector(ticker)
        except Exception:                       # noqa: BLE001 — lookup is best-effort
            raw = None
        if raw:
            return _YF_SECTOR.get(raw, raw)
    return (ticker or "").upper()


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
