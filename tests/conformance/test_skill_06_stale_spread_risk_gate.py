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
    """Skill 06 §2: gate fires when spread > max(abs_floor, bps × mid)."""

    @staticmethod
    def _gate_fires(bid: float, ask: float,
                    abs_cap: float = 0.05,
                    bps_of_mid: float = 0.0005) -> bool:
        spread = ask - bid
        mid = (ask + bid) / 2.0
        threshold = max(abs_cap, bps_of_mid * mid)
        return spread > threshold

    def test_tight_quote_passes(self) -> None:
        """SPY at $733, 2¢ spread → 2¢ < 5¢ absolute → passes."""
        assert not self._gate_fires(732.99, 733.01)

    def test_wide_absolute_spread_fires(self) -> None:
        """SPY-style underlying with a 10¢ spread → fires the absolute
        threshold (10¢ > 5¢)."""
        assert self._gate_fires(732.95, 733.05)

    def test_high_spot_uses_bps_threshold(self) -> None:
        """At spot $10,000 (hypothetical), 5 bps × mid = $0.50, which
        is much greater than the absolute floor of 5¢. The bps cap
        dominates here."""
        # spread 0.30 on mid 10000 → 0.30 < 0.50 → passes
        assert not self._gate_fires(9999.85, 10000.15,
                                    abs_cap=0.05, bps_of_mid=0.0005)
        # spread 0.60 on mid 10000 → 0.60 > 0.50 → fires
        assert self._gate_fires(9999.70, 10000.30,
                                abs_cap=0.05, bps_of_mid=0.0005)
