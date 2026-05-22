"""Conformance test: skill 35 — close-event collaborators.

Pins the public-contract surface of the four extracted classes:
PartialFillCooldown, PdtBlockDetector, CloseAlertNotifier,
CloseJournalWriter. Behavior tests use a JournalKB in tempdir —
no real network, no real disk pollution.

Failure modes caught:
- Someone reverts the streak derivation to in-memory state
  → cooldown silently never engages across cycle restarts (the
    2026-05-13 XLF/GLD post-mortem failure mode).
- Someone removes either PDT signature (40310100 / "pattern day
  trading") → reactive suppression misses legitimate Alpaca
  responses.
- Someone re-inlines the close-event logic in agent.py
  → _journal_close_event regresses to ~300 lines and the test
    fixtures need MagicMock(spec=TradingAgent) again.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# PartialFillCooldown
# ---------------------------------------------------------------------------


def _make_cooldown(tmpdir: str):
    from trading_agent.journal_kb import JournalKB
    from trading_agent.close_event_collaborators import PartialFillCooldown
    jkb = JournalKB(tmpdir, run_mode="live", dry_run=False)
    return PartialFillCooldown(
        journal_kb=jkb, threshold=3, window_min=60,
    ), jkb


def test_skill_35_cooldown_streak_increments_on_repeated_close_failed():
    """Skill 35 §3.1: streak_within_window counts unsuperseded
    close_failed rows for the ticker."""
    with tempfile.TemporaryDirectory() as d:
        cd, jkb = _make_cooldown(d)
        for _ in range(2):
            jkb.log_signal(
                ticker="DIA", action="close_failed", price=420.0,
                raw_signal={}, notes="partial",
            )
        streak, last_ts = cd.streak_within_window("DIA")
        assert streak == 2, f"expected 2, got {streak}"
        assert last_ts is not None


def test_skill_35_cooldown_streak_resets_after_closed_row():
    """Skill 35 §3.1: a `closed` row supersedes prior close_failed
    rows — streak after a real close starts back at 0."""
    with tempfile.TemporaryDirectory() as d:
        cd, jkb = _make_cooldown(d)
        for _ in range(2):
            jkb.log_signal(
                ticker="DIA", action="close_failed", price=420.0,
                raw_signal={}, notes="partial",
            )
        jkb.log_signal(
            ticker="DIA", action="closed", price=420.0,
            raw_signal={}, notes="real close",
        )
        streak, _ = cd.streak_within_window("DIA")
        assert streak == 0, (
            f"Skill 35 §3.1: a successful close must reset the "
            f"streak counter; got {streak}, expected 0."
        )


def test_skill_35_cooldown_build_payload_engages_at_threshold():
    """Skill 35 §3.1: build_payload_fields returns the cooldown
    deadline + reason iff the streak (including the pending row)
    reaches the threshold."""
    with tempfile.TemporaryDirectory() as d:
        cd, jkb = _make_cooldown(d)
        # 2 existing failures → +1 pending = 3 = threshold → engage
        for _ in range(2):
            jkb.log_signal(
                ticker="DIA", action="close_failed", price=420.0,
                raw_signal={}, notes="partial",
            )
        streak, fields = cd.build_payload_fields("DIA")
        assert streak == 3
        assert "close_cooldown_until" in fields, (
            "Skill 35 §3.1: cooldown deadline must be embedded once "
            "the streak hits the threshold so the dashboard can show "
            "the engaged state without re-deriving from the journal."
        )
        assert "close_cooldown_reason" in fields


def test_skill_35_cooldown_below_threshold_omits_deadline():
    """Skill 35 §3.1: pre-engagement rows carry the streak counter
    but not the deadline (operator sees '2/3' but no cooldown banner)."""
    with tempfile.TemporaryDirectory() as d:
        cd, jkb = _make_cooldown(d)
        jkb.log_signal(
            ticker="DIA", action="close_failed", price=420.0,
            raw_signal={}, notes="partial",
        )
        streak, fields = cd.build_payload_fields("DIA")
        assert streak == 2
        assert fields["partial_close_streak"] == 2
        assert fields["partial_close_threshold"] == 3
        assert "close_cooldown_until" not in fields


# ---------------------------------------------------------------------------
# PdtBlockDetector
# ---------------------------------------------------------------------------


def test_skill_35_pdt_detector_matches_alpaca_error_code():
    """Skill 35 §3.2: detect() must match Alpaca's structured PDT
    error code 40310100 anywhere in a leg's error message."""
    from trading_agent.close_event_collaborators import PdtBlockDetector
    legs = [{"symbol": "SPY", "status": "rejected",
             "error": "Code 40310100: pattern day trading protection"}]
    assert PdtBlockDetector.detect(legs) is True


def test_skill_35_pdt_detector_matches_human_phrase():
    """Skill 35 §3.2: detect() must also match the human-readable
    'pattern day trading' phrase in case the structured code is
    missing from the response body."""
    from trading_agent.close_event_collaborators import PdtBlockDetector
    legs = [{"symbol": "SPY", "status": "rejected",
             "error": "Pattern Day Trading violation"}]
    assert PdtBlockDetector.detect(legs) is True


def test_skill_35_pdt_detector_no_false_positive_on_clean_errors():
    """Skill 35 §3.2: an ordinary rejection (e.g. buying power)
    must NOT trigger PDT detection — false positives would
    suppress all close attempts for the day."""
    from trading_agent.close_event_collaborators import PdtBlockDetector
    legs = [{"symbol": "SPY", "status": "rejected",
             "error": "insufficient buying power"}]
    assert PdtBlockDetector.detect(legs) is False


def test_skill_35_pdt_blocked_tickers_today_reads_journal():
    """Skill 35 §3.2: blocked_tickers_today returns underlyings
    with an unexpired pdt_blocked_today=True marker in today's
    journal rows."""
    from trading_agent.journal_kb import JournalKB
    from trading_agent.close_event_collaborators import PdtBlockDetector
    import datetime as _dt
    today_iso = _dt.datetime.utcnow().date().isoformat()
    with tempfile.TemporaryDirectory() as d:
        jkb = JournalKB(d, run_mode="live", dry_run=False)
        # Two PDT-blocked rows for DIA, one for SPY, one stale.
        jkb.log_signal(
            ticker="DIA", action="close_failed", price=420.0,
            raw_signal={"pdt_blocked_today": True,
                        "pdt_blocked_date": today_iso},
            notes="close_failed: PDT",
        )
        jkb.log_signal(
            ticker="SPY", action="close_failed", price=500.0,
            raw_signal={"pdt_blocked_today": True,
                        "pdt_blocked_date": today_iso},
            notes="close_failed: PDT",
        )
        jkb.log_signal(
            ticker="XLF", action="close_failed", price=40.0,
            raw_signal={"pdt_blocked_today": True,
                        "pdt_blocked_date": "2024-01-01"},
            notes="stale (different date)",
        )
        det = PdtBlockDetector(journal_kb=jkb)
        blocked = det.blocked_tickers_today()
        assert "DIA" in blocked
        assert "SPY" in blocked
        assert "XLF" not in blocked, (
            "Skill 35 §3.2: stale (different-date) markers must be "
            "ignored — the suppression self-clears at UTC midnight."
        )


# ---------------------------------------------------------------------------
# CloseJournalWriter — end-to-end orchestration
# ---------------------------------------------------------------------------


class _FakeSpread:
    underlying = "DIA"
    strategy_name = "Bear Call Spread"
    expiration = "2026-06-20"
    exit_reason = "STRIKE_PROXIMITY"

    class _ExitSignal:
        value = "STRIKE_PROXIMITY"
    exit_signal = _ExitSignal()
    account_balance = 5000.0


class _FakeTelegram:
    is_active = True
    notify_close_cooldown = staticmethod(lambda **kw: True)
    notify_pdt_block = staticmethod(lambda **kw: True)
    notify_position_closed = staticmethod(lambda **kw: True)


def _make_writer(tmpdir: str):
    """Construct CloseJournalWriter with stub collaborators."""
    from trading_agent.journal_kb import JournalKB
    from trading_agent.close_event_collaborators import (
        PartialFillCooldown, PdtBlockDetector,
        CloseAlertNotifier, CloseJournalWriter,
    )
    jkb = JournalKB(tmpdir, run_mode="live", dry_run=False)
    cd = PartialFillCooldown(
        journal_kb=jkb, threshold=3, window_min=60,
    )
    pd = PdtBlockDetector(journal_kb=jkb)
    sent_alerts: list = []
    def fake_send(**kw):
        sent_alerts.append(kw)
        return True
    alerts = CloseAlertNotifier(
        send_alert=fake_send, telegram=_FakeTelegram(),
    )
    writer = CloseJournalWriter(
        journal_kb=jkb, cooldown=cd, pdt_detector=pd, alerts=alerts,
        price_lookup=lambda t: 420.0,
    )
    return writer, jkb, sent_alerts


def _make_ctx():
    return {
        "strategy": "Bear Call Spread",
        "exit_signal": "STRIKE_PROXIMITY",
        "exit_reason": "Short leg pierced",
        "net_unrealized_pl": -50.0,
        "expiration": "2026-06-20",
        "original_credit": 1.20,
        "max_loss": 380.0,
    }


def test_skill_35_writer_writes_closed_action_on_complete_fill():
    """Skill 35 §3.4: fill_status='complete' produces action='closed'
    and triggers the position_closed alert."""
    import json
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        writer, jkb, alerts = _make_writer(d)
        writer.write(
            _FakeSpread(), _make_ctx(),
            leg_results=[{"symbol": "DIA250620C440",
                          "status": "closed"}],
            fill_status="complete", dry_run=False,
        )
        rows = [json.loads(l) for l in
                Path(jkb.jsonl_path).read_text().splitlines() if l]
        actions = [r["action"] for r in rows]
        assert "closed" in actions
        # position_closed alert fired
        alert_types = [a["alert_type"] for a in alerts]
        assert any("position_closed" in t for t in alert_types), (
            "Skill 35 §3.4: a complete fill must dispatch the "
            "position_closed Telegram alert."
        )


def test_skill_35_writer_writes_dry_run_close_distinct_from_closed():
    """Skill 35 §3.4: fill_status='dry_run' produces a DISTINCT
    action='dry_run_close' so realized-P&L queries don't sum the
    synthetic close as a real loss. This is the bug fix from
    commit 9cc5636 (-$2K phantom-loss recap)."""
    import json
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        writer, jkb, _ = _make_writer(d)
        writer.write(
            _FakeSpread(), _make_ctx(),
            leg_results=[],
            fill_status="dry_run", dry_run=True,
        )
        rows = [json.loads(l) for l in
                Path(jkb.jsonl_path).read_text().splitlines() if l]
        actions = [r["action"] for r in rows]
        assert "dry_run_close" in actions
        assert "closed" not in actions, (
            "Skill 35 §3.4: dry-run closes MUST NOT be tagged "
            "action='closed' — that's the bug pattern that produced "
            "the -$2,976 phantom-loss EOD recap."
        )


def test_skill_35_writer_engages_cooldown_at_threshold():
    """Skill 35 §3.4: three close_failed rows in a row engage the
    cooldown — the third row carries the deadline + reason, and the
    Telegram cooldown alert fires."""
    import json
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        writer, jkb, alerts = _make_writer(d)
        # Three close_failed in succession.
        for i in range(3):
            writer.write(
                _FakeSpread(), _make_ctx(),
                leg_results=[{"symbol": "DIA250620C440",
                              "status": "rejected",
                              "error": "limit not honored"}],
                fill_status="partial", dry_run=False,
            )
        rows = [json.loads(l) for l in
                Path(jkb.jsonl_path).read_text().splitlines() if l]
        # Last close_failed row carries the cooldown deadline
        close_failed_rows = [
            r for r in rows if r["action"] == "close_failed"
        ]
        assert len(close_failed_rows) == 3
        last = close_failed_rows[-1]["raw_signal"]
        assert "close_cooldown_until" in last, (
            "Skill 35 §3.4: the row that engages the cooldown must "
            "embed the deadline so the dashboard banner can render "
            "without re-deriving state."
        )
        # Cooldown alert dispatched
        alert_types = [a["alert_type"] for a in alerts]
        assert "close_cooldown" in alert_types


def test_skill_35_writer_marks_pdt_when_alpaca_responds_with_code():
    """Skill 35 §3.4: a close_failed row whose leg errors contain
    40310100 must carry pdt_blocked_today=True so the next cycle's
    close loop suppresses retries — and the pdt_block Telegram alert
    must fire."""
    import json
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        writer, jkb, alerts = _make_writer(d)
        writer.write(
            _FakeSpread(), _make_ctx(),
            leg_results=[{"symbol": "DIA250620C440",
                          "status": "rejected",
                          "error": "Code 40310100 pattern day trading"}],
            fill_status="partial", dry_run=False,
        )
        rows = [json.loads(l) for l in
                Path(jkb.jsonl_path).read_text().splitlines() if l]
        cf = [r for r in rows if r["action"] == "close_failed"][0]
        assert cf["raw_signal"].get("pdt_blocked_today") is True, (
            "Skill 35 §3.4: PDT detection must mark the journal row "
            "so the next cycle's close loop short-circuits hopeless "
            "retries (skill 17 §4 reactive suppression)."
        )
        alert_types = [a["alert_type"] for a in alerts]
        assert "pdt_block" in alert_types


def test_skill_35_writer_never_propagates_exceptions():
    """Skill 35 §3.4: a journal-write failure must NOT propagate
    out of write() — the close itself already happened at the
    broker; raising here would crash the cycle and leave the
    position state diverged from broker truth."""
    from trading_agent.close_event_collaborators import (
        PartialFillCooldown, PdtBlockDetector,
        CloseAlertNotifier, CloseJournalWriter,
    )
    class BrokenJournal:
        jsonl_path = "/tmp/nonexistent"
        def log_signal(self, **kw):
            raise RuntimeError("journal disk full")
    cd = PartialFillCooldown(
        journal_kb=BrokenJournal(), threshold=3, window_min=60,
    )
    pd = PdtBlockDetector(journal_kb=BrokenJournal())
    alerts = CloseAlertNotifier(
        send_alert=lambda **kw: True, telegram=_FakeTelegram(),
    )
    writer = CloseJournalWriter(
        journal_kb=BrokenJournal(), cooldown=cd, pdt_detector=pd,
        alerts=alerts, price_lookup=lambda t: 0.0,
    )
    # If write() raises, this test errors loudly
    writer.write(
        _FakeSpread(), _make_ctx(),
        leg_results=[], fill_status="complete", dry_run=False,
    )
    # Reached this line → write did NOT propagate the journal failure


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


def test_skill_35_agent_constructs_close_collaborators():
    """Skill 35 §3.3: TradingAgent.__init__ must construct the four
    collaborators and store them on self so call sites can delegate."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    for ctor in (
        "self._cooldown = PartialFillCooldown(",
        "self._pdt_detector = PdtBlockDetector(",
        "self._close_alerts = CloseAlertNotifier(",
        "self._close_writer = CloseJournalWriter(",
    ):
        assert ctor in src, (
            f"Skill 35 §3.3: TradingAgent.__init__ must construct "
            f"{ctor.strip()} — without it the extracted writer can't "
            f"find its collaborators and the close path AttributeErrors."
        )


def test_skill_35_journal_close_event_delegates_to_writer():
    """Skill 35 §3.3: _journal_close_event MUST be a thin delegation
    to self._close_writer.write — not a 300-line re-implementation.
    Pinning this in the test prevents accidental re-inlining."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "self._close_writer.write(" in src, (
        "Skill 35 §3.3: agent._journal_close_event must delegate "
        "via self._close_writer.write — the 300-line inline body "
        "was the test-fixture nightmare we just got rid of."
    )
