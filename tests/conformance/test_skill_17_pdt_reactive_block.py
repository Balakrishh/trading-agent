"""Conformance test: skill 17 §4 — Reactive PDT-block suppression.

Skill 17 §4 documents that after Alpaca returns code 40310100 (or the
literal "pattern day trading" phrase) on any close leg, the agent writes
``pdt_blocked_today=True`` + ``pdt_blocked_date=<UTC today>`` onto the
close_failed journal row, and that subsequent same-UTC-day close
attempts get suppressed with ``action="skipped_pdt_blocked"``.

Pins the contract surface:

  1. ``_journal_close_event`` inspects leg_results for the PDT signal
  2. ``_pdt_blocked_today_tickers`` helper exists with the documented
     name and signature
  3. The detection logic uses BOTH the literal Alpaca code and the
     phrase (substring match, case-insensitive)
  4. The state is UTC-date-keyed so it self-clears at midnight

Failure modes caught:
- Someone renames the helper → close loop's import breaks silently
- Someone narrows the detection to only-the-code → real-world responses
  that carry the phrase but a different code slip through
- Someone broadens the date check (e.g. forgets to compare dates) →
  state never clears, ticker stuck for life
- Someone forgets to register the new ``skipped_pdt_blocked`` action
  string → the dashboard's action vocabulary drifts
"""

from __future__ import annotations


def test_skill_17_pdt_helper_exists() -> None:
    """Skill 17 §4: ``_pdt_blocked_today_tickers`` is the journal-derived
    entry point. Read agent.py source directly so the test doesn't need
    pandas_market_calendars / scipy to be installed in the conformance
    runner."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(encoding="utf-8")
    assert "def _pdt_blocked_today_tickers" in src, (
        "Skill 17 §4: TradingAgent must define _pdt_blocked_today_tickers. "
        "Renaming breaks the close-loop suppression gate."
    )


def test_skill_17_pdt_detection_inspects_leg_results() -> None:
    """Skill 17 §4: PDT detection must scan leg_results for both Alpaca
    signal forms (numeric code + human phrase) and tag the row before
    writing it. Skill 35 (2026-05-22) moved the detection from
    agent._journal_close_event into close_event_collaborators.
    PdtBlockDetector — assertions span both files."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    detector_src = (
        repo_root / "trading_agent" / "close_event_collaborators.py"
    ).read_text(encoding="utf-8")
    agent_src = (
        repo_root / "trading_agent" / "agent.py"
    ).read_text(encoding="utf-8")
    # Both signals must be present in the detector — narrowing to one
    # would silently miss real-world responses.
    assert '"40310100"' in detector_src, (
        "Skill 17 §4: PdtBlockDetector.PDT_SIGNALS must include the "
        "literal Alpaca code 40310100. Without it, responses that "
        "carry only the phrase would not be detected."
    )
    assert '"pattern day trading"' in detector_src, (
        "Skill 17 §4: PdtBlockDetector.PDT_SIGNALS must include the "
        "phrase 'pattern day trading'. Without it, responses that "
        "carry only a different code/phrasing would not be detected."
    )
    # The marker field names appear in BOTH files: emitted by the
    # detector + consumed by the close-loop reader on TradingAgent.
    assert "pdt_blocked_today" in detector_src
    assert "pdt_blocked_today" in agent_src, (
        "Skill 17 §4: the close-loop must read pdt_blocked_today via "
        "the journal scan. Renaming the field breaks self-clearing."
    )
    assert "pdt_blocked_date" in detector_src


def test_skill_17_close_loop_uses_blocked_set() -> None:
    """Skill 17 §4: the Stage-1 close loop must check pdt_blocked_today
    set after the existing cooldown / REGIME_SHIFT gates."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(encoding="utf-8")
    assert "pdt_blocked_today = self._pdt_blocked_today_tickers()" in src, (
        "Skill 17 §4: the cycle must compute the blocked-tickers set "
        "once per cycle and feed it to the close loop. Recomputing in "
        "the loop would multiply journal I/O by ticker count."
    )
    assert 'action="skipped_pdt_blocked"' in src, (
        "Skill 17 §4: the suppressed close must journal a "
        "skipped_pdt_blocked action so the dashboard / replay tooling "
        "can attribute the skip to a real broker response (vs. the "
        "speculative same-day-REGIME_SHIFT suppression)."
    )


def test_skill_17_helper_is_date_keyed_for_self_clear() -> None:
    """Skill 17 §4: the marker's date field must be compared to today's
    UTC date so it self-clears at midnight. A missing date check would
    leave the marker active forever — bug worse than the original noise.

    Skill 35 (2026-05-22) moved the helper body into
    PdtBlockDetector.blocked_tickers_today() in
    close_event_collaborators.py. The agent.py method is now a
    one-line shim, so the date-keyed comparison lives in the
    collaborator module."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (
        repo_root / "trading_agent" / "close_event_collaborators.py"
    ).read_text(encoding="utf-8")
    helper_start = src.find("def blocked_tickers_today")
    assert helper_start > 0
    helper_end = src.find("\n    def ", helper_start + 1)
    helper_body = src[helper_start:helper_end if helper_end > 0 else None]
    assert "today_iso" in helper_body, (
        "Skill 17 §4: blocked_tickers_today must compare against "
        "today's UTC date. Without the date filter the marker never "
        "expires — a position blocked once would be permanently "
        "untouchable until the agent restarts."
    )
    assert 'pdt_blocked_date' in helper_body, (
        "Skill 17 §4: helper must filter rows by pdt_blocked_date to "
        "implement self-clearing. Missing this means yesterday's "
        "markers would leak into today's blocked set."
    )


def test_skill_17_detection_is_reactive_only() -> None:
    """Skill 17 §4: the gate must engage ONLY after a journal-confirmed
    PDT response. Speculative gating (PDT-restricted + same-day-open ⇒
    pre-emptive skip) would defeat the purpose: it would suppress
    legitimate closes that Alpaca might have accepted. Pin that the
    suppression condition reads the journal-derived set, not the
    speculative same_day_tickers set."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(encoding="utf-8")
    # Find the new reactive-PDT block in the close loop.
    needle = "Reactive PDT-block suppression"
    block_start = src.find(needle)
    assert block_start > 0, (
        "Skill 17 §4: close loop must carry a 'Reactive PDT-block "
        "suppression' block. Renaming silently removes the gate's "
        "documentation anchor."
    )
    # The suppression condition right under that comment must read
    # the journal-derived set, not the speculative same_day_tickers.
    block = src[block_start:block_start + 1200]
    assert "if spread.underlying in pdt_blocked_today:" in block, (
        "Skill 17 §4: the suppression must gate on pdt_blocked_today "
        "(journal-derived from real broker responses). Gating on "
        "same_day_tickers / pdt_restricted alone would make this a "
        "speculative gate, defeating the design."
    )
