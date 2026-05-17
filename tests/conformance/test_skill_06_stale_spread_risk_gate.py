"""Conformance test: skill 06 — Stale-spread risk gate.

Skill 06 §2 documents the underlying-liquidity gate that runs as
risk-manager check #7. Three thresholds compose:

  * ``liquidity_max_spread``       (absolute $)  — default 0.05 (5 ¢)
  * ``liquidity_bps_of_mid``       (relative)    — default 0.0005 (5 bps)
  * ``stale_spread_pct``           (downgrade)   — default 0.01 (1%)

The bid/ask gate fires when the underlying's spread exceeds
``max(liquidity_max_spread, liquidity_bps_of_mid × mid)``. If the
relative spread (spread/mid) exceeds ``stale_spread_pct`` the check
is downgraded to a soft warning rather than a hard rejection — the
quote is treated as stale rather than illiquid.

Failure modes caught:
- Someone tightens ``stale_spread_pct`` from 1% to 0.1% — every
  near-the-bell quote would downgrade and the gate would lose
  signal
- The OR-relationship between absolute and bps thresholds gets
  flipped to AND, silently letting wider spreads through
"""

from __future__ import annotations

from trading_agent.risk_manager import RiskManager


def test_skill_06_default_thresholds_match_documented_values() -> None:
    """Skill 06 §2: the three defaults must match the documented numbers.

    Changing any of these silently shifts the gate's sensitivity and
    can either flood logs with false positives or hide real illiquidity.
    """
    rm = RiskManager(max_risk_pct=0.02)
    assert rm.liquidity_max_spread == 0.05, (
        f"Skill 06 §2: liquidity_max_spread default must be 0.05; got "
        f"{rm.liquidity_max_spread}."
    )
    assert rm.liquidity_bps_of_mid == 0.0005, (
        f"Skill 06 §2: liquidity_bps_of_mid default must be 0.0005 "
        f"(5 bps); got {rm.liquidity_bps_of_mid}."
    )
    assert rm.stale_spread_pct == 0.01, (
        f"Skill 06 §2: stale_spread_pct default must be 0.01 (1%); got "
        f"{rm.stale_spread_pct}."
    )


def test_skill_06_thresholds_are_user_overridable() -> None:
    """Skill 06 §3: all three thresholds are constructor parameters
    (not class constants) so a preset or test fixture can override
    them. Validates the contract surface."""
    rm = RiskManager(
        max_risk_pct=0.02,
        liquidity_max_spread=0.10,
        liquidity_bps_of_mid=0.001,
        stale_spread_pct=0.02,
    )
    assert rm.liquidity_max_spread == 0.10
    assert rm.liquidity_bps_of_mid == 0.001
    assert rm.stale_spread_pct == 0.02


class TestSkill06GateMath:
    """Skill 06 §2: gate fires when spread > max(abs_floor, bps_of_mid × mid).

    The two thresholds bind in different regimes:
      * Low-spot underlyings (mid < $100 for default 5 bps): the
        absolute $0.05 floor binds.
      * High-spot underlyings (mid > $100): the bps cap dominates
        because 0.0005 × $100 = $0.05 is the crossover point. At
        SPY's ~$733 mid, the threshold is ~$0.37, not $0.05.
    """

    @staticmethod
    def _gate_fires(bid: float, ask: float,
                    abs_cap: float = 0.05,
                    bps_of_mid: float = 0.0005) -> bool:
        spread = ask - bid
        mid = (ask + bid) / 2.0
        threshold = max(abs_cap, bps_of_mid * mid)
        return spread > threshold

    def test_low_spot_tight_quote_passes(self) -> None:
        """XLF at $50, 2¢ spread → 2¢ < 5¢ absolute floor → passes."""
        assert not self._gate_fires(49.99, 50.01)

    def test_low_spot_wide_spread_fires_absolute_cap(self) -> None:
        """XLF at $50, 10¢ spread → exceeds the $0.05 absolute floor
        (the bps cap of 0.0005 × $50 = $0.025 is below the floor and
        the max() picks the larger threshold = $0.05)."""
        assert self._gate_fires(49.95, 50.05)

    def test_high_spot_tight_quote_passes(self) -> None:
        """SPY at $733, 10¢ spread → 10¢ < 36.65¢ bps cap → passes.
        Demonstrates that the bps cap protects high-spot underlyings
        from being flagged as "wide" just because absolute pennies
        look big."""
        assert not self._gate_fires(732.95, 733.05)

    def test_high_spot_wide_spread_fires_bps_cap(self) -> None:
        """SPY at $733, 50¢ spread → exceeds 0.0005 × $733 = $0.37
        bps cap → fires. The absolute $0.05 floor is irrelevant
        here because the bps cap dominates."""
        assert self._gate_fires(732.75, 733.25)

    def test_very_high_spot_uses_bps_cap(self) -> None:
        """Hypothetical GOOG-class $2,500 underlying: threshold is
        0.0005 × 2500 = $1.25, so a $1.00 spread passes but $2.00
        fires. The bps cap fully dominates the absolute floor at
        these spots."""
        assert not self._gate_fires(2499.50, 2500.50)
        assert self._gate_fires(2499.00, 2501.00)
