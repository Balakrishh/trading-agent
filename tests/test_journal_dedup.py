"""
test_journal_dedup.py — pin down the Option A rejection-spam dedup.

At a 5-min cycle × ~9 tickers × steady-state rejection a $5K paper
account can produce 80–100 redundant journal rows per hour.  The
dedup added 2026-05-06 suppresses identical consecutive
``(action, checks_failed, reason)`` signatures per ticker, with a
periodic heartbeat row so the journal isn't completely silent.

These tests pin every branch of the decision matrix so a future
refactor can't silently re-spam the journal:

  * 1st rejection writes
  * 2nd identical rejection is suppressed
  * Nth duplicate emits a heartbeat row tagged with `dedup.streak=N`
  * Different signature → fresh row + counter reset
  * SUBMITTED / CLOSED / dry_run / error always bypass dedup
  * Setting JOURNAL_DEDUP_HEARTBEAT_EVERY=0 disables dedup entirely
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_agent.journal_kb import JournalKB


@pytest.fixture
def jkb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> JournalKB:
    monkeypatch.setenv("JOURNAL_DEDUP_HEARTBEAT_EVERY", "12")
    return JournalKB(journal_dir=str(tmp_path), run_mode="live")


def _read_rows(jkb: JournalKB) -> list[dict]:
    p = Path(jkb.jsonl_path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# ── Basic dedup behaviour ────────────────────────────────────────────


def test_first_rejection_is_written(jkb: JournalKB):
    jkb.log_signal(
        ticker="SPY", action="rejected", price=731.0,
        raw_signal={"checks_failed": ["Max loss $400 > 2% × $5,000 (=$100)"]},
    )
    rows = _read_rows(jkb)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "SPY"
    assert rows[0]["action"] == "rejected"


def test_second_identical_rejection_is_suppressed(jkb: JournalKB):
    payload = {"checks_failed": ["Max loss $400 > 2% × $5,000 (=$100)"]}
    jkb.log_signal(ticker="SPY", action="rejected", price=731.0, raw_signal=payload)
    jkb.log_signal(ticker="SPY", action="rejected", price=731.5, raw_signal=payload)
    rows = _read_rows(jkb)
    assert len(rows) == 1, "Identical consecutive rejection must be suppressed"


def test_heartbeat_emits_at_interval(jkb: JournalKB):
    """The 12th duplicate writes a heartbeat row (default interval=12)."""
    payload = {"checks_failed": ["Max loss $400 > 2% × $5,000 (=$100)"]}
    for _ in range(13):                  # cycles 1..13
        jkb.log_signal(ticker="SPY", action="rejected",
                       price=731.0, raw_signal=payload)
    rows = _read_rows(jkb)
    # Cycle 1 → write.  Cycles 2..11 → suppress.  Cycle 12 → heartbeat.
    # Cycle 13 → suppress again (next heartbeat at 24).
    assert len(rows) == 2, f"Expected 2 rows (1 fresh + 1 heartbeat), got {len(rows)}"
    assert "dedup" not in rows[0]
    assert rows[1].get("dedup", {}).get("kind") == "heartbeat"
    assert rows[1]["dedup"]["streak"] == 12


def test_changed_signature_writes_fresh_and_resets_counter(jkb: JournalKB):
    p1 = {"checks_failed": ["Max loss $400 > $100"]}
    p2 = {"checks_failed": ["No positive-EV candidate"]}
    # Two of A, then one of B (different reason), then two more of A.
    jkb.log_signal(ticker="SPY", action="rejected", price=731.0, raw_signal=p1)
    jkb.log_signal(ticker="SPY", action="rejected", price=731.5, raw_signal=p1)  # suppressed
    jkb.log_signal(ticker="SPY", action="rejected", price=732.0, raw_signal=p2)  # fresh
    jkb.log_signal(ticker="SPY", action="rejected", price=732.5, raw_signal=p1)  # fresh (signature flipped back)
    jkb.log_signal(ticker="SPY", action="rejected", price=733.0, raw_signal=p1)  # suppressed
    rows = _read_rows(jkb)
    assert len(rows) == 3, f"Expected 3 fresh writes, got {len(rows)}"


# ── Per-ticker isolation ─────────────────────────────────────────────


def test_dedup_is_per_ticker(jkb: JournalKB):
    """SPY and QQQ should not share dedup state."""
    payload = {"checks_failed": ["Max loss > budget"]}
    jkb.log_signal(ticker="SPY", action="rejected", price=731.0, raw_signal=payload)
    jkb.log_signal(ticker="QQQ", action="rejected", price=640.0, raw_signal=payload)
    jkb.log_signal(ticker="SPY", action="rejected", price=731.5, raw_signal=payload)  # suppress
    jkb.log_signal(ticker="QQQ", action="rejected", price=640.5, raw_signal=payload)  # suppress
    rows = _read_rows(jkb)
    assert len(rows) == 2
    assert {r["ticker"] for r in rows} == {"SPY", "QQQ"}


# ── Material events bypass dedup ─────────────────────────────────────


@pytest.mark.parametrize("action", ["submitted", "closed", "dry_run", "error"])
def test_material_actions_bypass_dedup(jkb: JournalKB, action):
    """Submitted, closed, dry_run, and error rows ALWAYS write."""
    payload = {"strategy": "Iron Condor"}
    for _ in range(5):
        jkb.log_signal(ticker="SPY", action=action, price=731.0, raw_signal=payload)
    rows = _read_rows(jkb)
    assert len(rows) == 5, f"All {action} rows must be written, got {len(rows)}"


def test_submitted_resets_dedup_counter_for_subsequent_rejections(jkb: JournalKB):
    """A SUBMITTED clears the per-ticker counter, so a *next* rejection
    writes fresh even if it has the same signature as before."""
    p_reject = {"checks_failed": ["Max loss > budget"]}
    # Establish a 2-cycle rejection streak.
    jkb.log_signal(ticker="SPY", action="rejected", price=731.0, raw_signal=p_reject)
    jkb.log_signal(ticker="SPY", action="rejected", price=731.5, raw_signal=p_reject)  # suppress
    # Now SPY actually fills.
    jkb.log_signal(ticker="SPY", action="submitted", price=732.0,
                   raw_signal={"strategy": "Bear Call Spread"})
    # Later, rejection signature recurs — must write fresh because the
    # SUBMITTED reset the counter.
    jkb.log_signal(ticker="SPY", action="rejected", price=733.0, raw_signal=p_reject)
    rows = _read_rows(jkb)
    assert len(rows) == 3, (
        f"Expected fresh-reject + submitted + fresh-reject (3 rows), got {len(rows)}"
    )
    assert rows[0]["action"] == "rejected"
    assert rows[1]["action"] == "submitted"
    assert rows[2]["action"] == "rejected"


# ── Disable knob ─────────────────────────────────────────────────────


def test_heartbeat_every_zero_disables_dedup(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch):
    """JOURNAL_DEDUP_HEARTBEAT_EVERY=0 → write every row (legacy behaviour)."""
    monkeypatch.setenv("JOURNAL_DEDUP_HEARTBEAT_EVERY", "0")
    jkb = JournalKB(journal_dir=str(tmp_path), run_mode="live")
    payload = {"checks_failed": ["Max loss > budget"]}
    for _ in range(20):
        jkb.log_signal(ticker="SPY", action="rejected", price=731.0, raw_signal=payload)
    rows = _read_rows(jkb)
    assert len(rows) == 20, f"Dedup off — every row should write; got {len(rows)}"


def test_invalid_env_falls_back_to_default(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JOURNAL_DEDUP_HEARTBEAT_EVERY", "not-an-int")
    jkb = JournalKB(journal_dir=str(tmp_path), run_mode="live")
    assert jkb._heartbeat_every == 12         # default


# ── Signature stability ──────────────────────────────────────────────


def test_signature_invariant_to_checks_failed_order():
    """Reordering ``checks_failed`` must NOT change the signature
    (otherwise dedup would silently fail when the agent reorders the
    failure list)."""
    sig_a = JournalKB._signature(
        "rejected",
        {"checks_failed": ["A", "B", "C"]},
    )
    sig_b = JournalKB._signature(
        "rejected",
        {"checks_failed": ["C", "A", "B"]},
    )
    assert sig_a == sig_b


def test_signature_distinguishes_rejection_reasons():
    sig_max_loss = JournalKB._signature(
        "rejected", {"checks_failed": ["Max loss > budget"]},
    )
    sig_no_ev = JournalKB._signature(
        "rejected", {"checks_failed": ["No positive-EV candidate"]},
    )
    assert sig_max_loss != sig_no_ev
