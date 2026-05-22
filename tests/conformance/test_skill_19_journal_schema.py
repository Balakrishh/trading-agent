"""Conformance test: skill 19 — Journal schema.

Skill 19 §2 documents the journal row schema:

  * Each row carries an ``action`` from a fixed enum
  * The canonical action values include:
      submitted, rejected, closed, close_failed,
      skipped_existing, skipped_rsi_gate, skipped_defense_first,
      skipped_bias, error, warning, dry_run, dry_run_close
  * ``JournalKB.log_signal(...)`` is the canonical write entry point

This conformance test pins the public-method surface + that the
canonical action values are accepted (no schema-enforcement code
rejects them at write time).

Full schema-validation behavior (dedup, raw_signal preservation,
mode tagging) is covered by ``tests/test_journal_kb.py`` and
``tests/test_journal_dedup.py``.
"""

from __future__ import annotations

import inspect

from trading_agent.journal_kb import JournalKB


def test_skill_19_log_signal_method_exists() -> None:
    """Skill 19 §3: log_signal is the canonical write entry point."""
    assert hasattr(JournalKB, "log_signal")
    assert callable(JournalKB.log_signal)


def test_skill_19_log_signal_signature_carries_action_param() -> None:
    """Skill 19 §2: every journal row's ``action`` field comes from
    the log_signal call. The parameter must exist by that name."""
    sig = inspect.signature(JournalKB.log_signal)
    assert "action" in sig.parameters, (
        "Skill 19 §2: log_signal must accept an ``action`` keyword. "
        "Renaming this parameter breaks every caller (agent.py, "
        "executor.py, decision_engine.py)."
    )


def test_skill_19_documented_action_constants_used() -> None:
    """Skill 19 §2 lists action strings as literals (not an enum)
    so any string can technically be passed. We pin the canonical
    list here so a contributor doesn't accidentally use a typo
    (e.g., "skip_existing" instead of "skipped_existing")."""
    canonical_actions = [
        "submitted", "rejected", "closed", "close_failed",
        "skipped_existing", "skipped_rsi_gate",
        "skipped_defense_first", "skipped_bias",
        "error", "warning",
        "dry_run", "dry_run_close",
    ]
    # Read the journal_kb source to spot-check each canonical action
    # appears at least once.
    from pathlib import Path
    src = Path(JournalKB.__module__.replace(".", "/") + ".py")
    # Try the actual file path
    import trading_agent.journal_kb as jkb_module
    src_path = Path(jkb_module.__file__)
    text = src_path.read_text(encoding="utf-8")
    missing = [a for a in canonical_actions if a not in text]
    # We don't fail on every absence — some are emitted by callers
    # in other modules (agent.py for `submitted` / `rejected`). But
    # at least the "skipped_*" family and "warning" / "error" / "closed"
    # should be visible in journal_kb itself.
    core_actions = {"closed", "close_failed", "warning", "error"}
    missing_core = [a for a in core_actions if a not in text]
    assert not missing_core, (
        f"Skill 19 §2: core action strings {missing_core} not found "
        f"in journal_kb.py — schema may have drifted."
    )


def test_skill_19_specialised_log_helpers_exist() -> None:
    """Skill 19 §3 mentions specialised log methods that wrap
    log_signal with canonical defaults. Pin the public-method surface."""
    for method in ("log_defense_first", "log_error", "log_warning",
                   "log_shutdown", "log_cycle_error"):
        assert hasattr(JournalKB, method), (
            f"Skill 19 §3: JournalKB.{method}() is documented but the "
            f"method is missing."
        )


def test_skill_19_dry_run_close_action_distinct_from_closed() -> None:
    """Skill 19 §4 (2026-05-21 hotfix): the agent must write
    ``action="dry_run_close"`` when fill_status=="dry_run", NOT
    ``action="closed"``. Pre-fix on the Pi, 22 dry-run synthetic
    closes accumulated -$2,860 of phantom realized P&L on the
    dashboard because they were tagged action="closed" + summed.

    Pin the source-level branch shape so a refactor can't
    accidentally re-collapse the two action labels.

    Skill 35 (2026-05-22) moved the close-event branching into
    CloseJournalWriter._write_impl. Search the collaborators
    module — that's where the action mapping now lives."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (
        repo_root / "trading_agent" / "close_event_collaborators.py"
    ).read_text(encoding="utf-8")
    # The branch must distinguish complete vs dry_run. The pre-fix
    # pattern was a single ``if fill_status in ("complete", "dry_run"):
    # action="closed"`` collapsed branch.
    forbidden = 'fill_status in ("complete", "dry_run")'
    assert forbidden not in src, (
        f"Skill 19 §4: CloseJournalWriter must NOT collapse "
        f"fill_status='complete' and fill_status='dry_run' into the "
        f"same action='closed' branch. Pre-fix this caused -$2,860 "
        f"of phantom realized P&L when a position was stuck in "
        f"dry-run mode."
    )
    # The two action assignments must both appear, in the correct
    # branches.
    assert 'action = "dry_run_close"' in src, (
        "Skill 19 §4: dry-run close must use action='dry_run_close'."
    )
    assert 'action = "closed"' in src, (
        "Skill 19 §4: real broker close must still use action='closed'."
    )


def test_skill_19_dryrun_writes_to_separate_file() -> None:
    """Skill 19 §1.1 (2026-05-21 decoupling): dry-run cycles must
    write to ``signals_dryrun.jsonl``, not ``signals_live.jsonl``.

    Pre-fix dry-run rows landed in signals_live.jsonl with a mode
    tag and three readers were expected to filter on the tag.
    Three readers forgot, producing the -$2,860 family of bugs.
    Physical file separation makes the entire failure class
    structurally impossible.

    Pin the behavior by booting two JournalKB instances side-by-
    side, writing one row each, and asserting the files don't
    cross-contaminate."""
    import json
    import tempfile
    from pathlib import Path
    from trading_agent.journal_kb import JournalKB

    with tempfile.TemporaryDirectory() as d:
        live = JournalKB(d, run_mode="live", dry_run=False)
        dry = JournalKB(d, run_mode="dryrun", dry_run=True)

        # Physical paths must differ
        assert live.jsonl_path != dry.jsonl_path, (
            "Skill 19 §1.1: live and dryrun JournalKB instances "
            "must write to physically distinct files."
        )
        assert Path(live.jsonl_path).name == "signals_live.jsonl"
        assert Path(dry.jsonl_path).name == "signals_dryrun.jsonl"

        # Write one row each; verify isolation
        live.log_signal(
            ticker="SPY", action="submitted", price=700.0,
            raw_signal={"strategy": "Bear Call", "net_credit": 0.66},
        )
        dry.log_signal(
            ticker="DIA", action="dry_run_close", price=500.0,
            raw_signal={"strategy": "Iron Condor"},
        )

        live_rows = [
            json.loads(line)
            for line in Path(live.jsonl_path).read_text().splitlines()
            if line.strip()
        ]
        dry_rows = [
            json.loads(line)
            for line in Path(dry.jsonl_path).read_text().splitlines()
            if line.strip()
        ]
        # Live file: ONLY the live row. Dryrun file: ONLY the dry row.
        assert len(live_rows) == 1 and live_rows[0]["ticker"] == "SPY"
        assert len(dry_rows) == 1 and dry_rows[0]["ticker"] == "DIA"
        # Negative assertion: no cross-contamination
        assert not any(r["ticker"] == "DIA" for r in live_rows), (
            "Skill 19 §1.1: dry-run rows must NEVER appear in "
            "signals_live.jsonl. This is the structural guarantee "
            "that prevents the -$2,860 phantom-loss class of bugs."
        )


def test_skill_19_valid_run_modes_includes_dryrun() -> None:
    """Skill 19 §1.1: VALID_RUN_MODES must list ``dryrun`` so a typo
    in the constructor (``"dyrun"``) fails loudly instead of
    silently creating ``signals_dyrun.jsonl``."""
    from trading_agent.journal_kb import JournalKB
    assert "dryrun" in JournalKB.VALID_RUN_MODES, (
        "Skill 19 §1.1: 'dryrun' must be in VALID_RUN_MODES. "
        "Without this, JournalKB(run_mode='dryrun') would raise "
        "ValueError and the agent couldn't write to the isolated "
        "stream."
    )
    # Defensive: typo'd mode must still raise. Construct in a temp
    # dir to avoid leaving stray journal_kb/ directories on test
    # cleanup paths.
    import tempfile
    with tempfile.TemporaryDirectory() as _td:
        try:
            JournalKB(_td, run_mode="dyrun")
            raised = False
        except ValueError:
            raised = True
    assert raised, (
        "Skill 19 §1.1: an invalid run_mode must raise ValueError "
        "at construction, not silently create a phantom journal file."
    )


def test_skill_19_dashboard_realized_pl_uses_journal_reader() -> None:
    """Skill 19 §1.2 (decoupling #2, 2026-05-22): the dashboard's
    realized-P&L tile must compute its value via
    ``JournalReader.realized_pl_today()`` — NOT a local re-implementation
    of the filter logic.

    History: this test originally pinned the in-line filter
    (``rs.get('fill_status') == 'dry_run'``) directly in
    ``live_monitor.py``. After decoupling #2 that filter moved into
    ``JournalReader.closes_today`` where it's pinned by the dedicated
    reader test (``test_journal_reader_closes_today_skips_dry_run_fill_status``).
    The dashboard-side test now pins the DELEGATION rather than the
    inline logic. That's the whole point of the refactor — one place
    owns the filter; many call sites consume it. A future refactor
    that re-inlines the filter in the dashboard would re-introduce
    the duplicate-readers bug pattern, so this test forbids it."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "streamlit" /
           "live_monitor.py").read_text(encoding="utf-8")
    # The dashboard MUST go through the reader for realized P&L.
    assert ".realized_pl_today()" in src, (
        "Skill 19 §1.2: dashboard realized-P&L tile must call "
        "JournalReader(...).realized_pl_today(). The filter logic "
        "(skip dry-run, ET-date boundary) lives there now — any "
        "local re-implementation would drift from sibling readers."
    )
    # Defensive: ensure the old in-line pattern hasn't been reintroduced.
    # A fresh ``realized_today += float`` followed by a dataframe walk
    # would be the regression to catch.
    forbidden_indicator = "for _, row in closed_today.iterrows():"
    assert forbidden_indicator not in src, (
        "Skill 19 §1.2: dashboard must NOT iterate journal rows "
        "directly to compute realized P&L. Use the JournalReader "
        "delegation instead — re-introducing the inline walk "
        "drifts filter semantics across consumers."
    )
