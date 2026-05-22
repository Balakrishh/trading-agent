"""Conformance test: skill 37 — position-cap dedup extraction.

Pins compute_position_cap_dedup_set behavior + the agent integration
that uses it. Pure function, no fixtures needed — easy to test
exhaustively.

Failure modes caught:
- Someone removes the per-sector cap → concentration risk returns.
- Someone reverts the GLD/2026-05-12 fix (counting only HOLD signals)
  → exit-pending positions slip past dedup.
- Someone re-inlines the logic in _run_cycle_impl → method-length
  ratchet test catches it.
"""

from __future__ import annotations


def _sector_table():
    """Static fixture sector map used across the tests."""
    return {
        "SPY": "broad_market", "QQQ": "broad_market", "DIA": "broad_market",
        "XLF": "financials", "KRE": "financials",
        "GLD": "metals", "SLV": "metals",
        "TLT": "bonds",
    }


def _sector_for(t):
    return _sector_table().get(t, "unknown")


def test_per_ticker_cap_blocks_at_threshold() -> None:
    """Skill 37 §3.1: a ticker with N ≥ max_positions_per_ticker is
    in the blocked set."""
    from trading_agent.position_caps import compute_position_cap_dedup_set
    monitor = {"positions": [{"underlying": "SPY"}]}
    blocked, ppt, pps, sec_at_cap = compute_position_cap_dedup_set(
        monitor, ["SPY", "QQQ", "DIA"],
        sector_for=_sector_for,
        max_positions_per_ticker=1, max_positions_per_sector=99,
    )
    assert "SPY" in blocked
    assert "QQQ" not in blocked
    assert ppt == {"SPY": 1}


def test_per_sector_cap_blocks_other_tickers_in_sector() -> None:
    """Skill 37 §3.1: a sector at cap blocks ALL tickers in that
    sector from the universe (not just the ones with open positions)."""
    from trading_agent.position_caps import compute_position_cap_dedup_set
    # 2 financials open → cap=2 → KRE + XLF both blocked.
    monitor = {"positions": [
        {"underlying": "XLF"}, {"underlying": "KRE"},
    ]}
    blocked, _, pps, sectors_at_cap = compute_position_cap_dedup_set(
        monitor, ["XLF", "KRE", "SPY", "GLD"],
        sector_for=_sector_for,
        max_positions_per_ticker=99,  # per-ticker disabled
        max_positions_per_sector=2,
    )
    assert "XLF" in blocked
    assert "KRE" in blocked, (
        "Skill 37 §3.1: a sector at cap must block every ticker in "
        "that sector — concentration risk would otherwise allow "
        "a third financial to slip through."
    )
    assert "SPY" not in blocked, "broad_market != financials"
    assert "GLD" not in blocked, "metals != financials"
    assert sectors_at_cap == {"financials"}
    assert pps == {"financials": 2}


def test_sector_block_does_not_double_count_ticker_block() -> None:
    """Skill 37 §3.1: when a ticker is already blocked by the per-ticker
    cap, the sector logic shouldn't re-add it. Keeps log noise focused
    on the sector signal."""
    from trading_agent.position_caps import compute_position_cap_dedup_set
    # XLF has 1 position (per-ticker cap=1 → XLF blocked).
    # Financials sector has 1 position (cap=2 → not blocked).
    monitor = {"positions": [{"underlying": "XLF"}]}
    blocked, _, _, sectors_at_cap = compute_position_cap_dedup_set(
        monitor, ["XLF", "KRE"],
        sector_for=_sector_for,
        max_positions_per_ticker=1, max_positions_per_sector=2,
    )
    assert "XLF" in blocked  # per-ticker
    assert "KRE" not in blocked  # sector still has capacity
    assert sectors_at_cap == set()


def test_no_positions_returns_empty_dedup_set() -> None:
    """Skill 37 §3.1: empty monitor results → empty dedup, no logs."""
    from trading_agent.position_caps import compute_position_cap_dedup_set
    blocked, ppt, pps, sectors = compute_position_cap_dedup_set(
        {"positions": []}, ["SPY"],
        sector_for=_sector_for,
        max_positions_per_ticker=1, max_positions_per_sector=2,
    )
    assert blocked == set()
    assert ppt == {}
    assert pps == {}
    assert sectors == set()


def test_missing_underlying_is_skipped() -> None:
    """Skill 37 §3.1: a malformed position entry without 'underlying'
    must not crash the cap computation."""
    from trading_agent.position_caps import compute_position_cap_dedup_set
    monitor = {"positions": [
        {"underlying": "SPY"},
        {},  # malformed
        {"underlying": ""},  # empty string
    ]}
    blocked, ppt, _, _ = compute_position_cap_dedup_set(
        monitor, ["SPY"],
        sector_for=_sector_for,
        max_positions_per_ticker=1, max_positions_per_sector=99,
    )
    assert "SPY" in blocked
    assert ppt == {"SPY": 1}


def test_position_count_aggregates_across_signals() -> None:
    """Skill 37 §3.1 / GLD 2026-05-12 post-mortem: every reported
    position counts toward the cap regardless of signal — exit-pending
    positions must NOT slip past dedup."""
    from trading_agent.position_caps import compute_position_cap_dedup_set
    monitor = {"positions": [
        # Same ticker, different signals — all 3 count
        {"underlying": "GLD", "signal": "HOLD"},
        {"underlying": "GLD", "signal": "regime_shift"},
        {"underlying": "GLD", "signal": "STRIKE_PROXIMITY"},
    ]}
    blocked, ppt, _, _ = compute_position_cap_dedup_set(
        monitor, ["GLD"],
        sector_for=_sector_for,
        max_positions_per_ticker=2, max_positions_per_sector=99,
    )
    assert "GLD" in blocked
    assert ppt["GLD"] == 3, (
        "Skill 37 §3.1: pre-2026-05-13 only HOLD-signal positions "
        "counted. The fix is to count EVERY reported position. "
        f"Got count={ppt['GLD']}, expected 3."
    )


def test_agent_uses_position_caps_module() -> None:
    """Skill 37 §3.2: _run_cycle_impl must delegate cap derivation
    to compute_position_cap_dedup_set rather than re-implementing
    the loop inline. Pinning this prevents the inlined version
    from creeping back."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "compute_position_cap_dedup_set" in src, (
        "Skill 37 §3.2: _run_cycle_impl must call "
        "compute_position_cap_dedup_set. Without it the inline loop "
        "and the dedup-set construction would be re-implemented "
        "locally — exactly the duplication this extraction prevents."
    )
    # The inline mutations of positions_per_ticker should be gone.
    assert "positions_per_ticker[underlying] = (" not in src, (
        "Skill 37 §3.2: the inline `positions_per_ticker[underlying] "
        "= (...) + 1` loop must live in position_caps.py, not "
        "agent.py — that's the regression guard."
    )
