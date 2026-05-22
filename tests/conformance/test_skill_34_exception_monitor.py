"""Conformance test: skill 34 — ExceptionMonitor.

Pins the public-contract surface of ExceptionMonitor and its
agent-side integration. Behavior tests use a fixture telegram +
JournalKB in a temp dir — no real network, no real disk pollution.

Failure modes caught:
- Someone removes the per-day Telegram dedup → operator spam
- Someone changes the journal action vocabulary →
  silenced_exceptions_today() can't find rows
- Someone reverts a key agent call site to bare logger.warning →
  silent failures hide for days again
- Someone makes record() propagate exceptions → caller's recovery
  path breaks
"""

from __future__ import annotations

import tempfile
from pathlib import Path


class _FakeTelegram:
    """Stub Telegram notifier that records calls without networking."""
    def __init__(self, is_active: bool = True):
        self.is_active = is_active
        self.calls: list = []
    def notify_silenced_exception(self, **kwargs) -> bool:
        self.calls.append(kwargs)
        return True


def _make_monitor(tmpdir: str, telegram_active: bool = True):
    """Helper: fresh JournalKB + monitor in a temp dir."""
    from trading_agent.journal_kb import JournalKB
    from trading_agent.exception_monitor import ExceptionMonitor
    jkb = JournalKB(tmpdir, run_mode="live", dry_run=False)
    tg = _FakeTelegram(is_active=telegram_active)
    return ExceptionMonitor(journal_kb=jkb, telegram=tg), jkb, tg


def test_skill_34_exception_monitor_class_exists() -> None:
    """Skill 34 §3.1: ExceptionMonitor is the documented entry point."""
    from trading_agent.exception_monitor import ExceptionMonitor
    assert callable(ExceptionMonitor)
    n = ExceptionMonitor(journal_kb=None, telegram=None)
    assert hasattr(n, "record"), (
        "Skill 34 §3.1: .record(...) is the documented public method."
    )


def test_skill_34_dedupe_pages_only_once_per_day() -> None:
    """Skill 34 §2: same (source, exc_class) on the same UTC day fires
    AT MOST one Telegram alert. Repeat calls still journal but don't
    re-page."""
    with tempfile.TemporaryDirectory() as d:
        mon, _, tg = _make_monitor(d)
        try:
            raise ValueError("Schwab token expired")
        except Exception as exc:
            mon.record(source="X.fetch", exc=exc, ticker="DIA")
            mon.record(source="X.fetch", exc=exc, ticker="DIA")
            mon.record(source="X.fetch", exc=exc, ticker="XLF")
        assert len(tg.calls) == 1, (
            f"Skill 34 §2: expected 1 Telegram page for repeated "
            f"(source, exc_class); got {len(tg.calls)}. The operator "
            f"would be spammed if dedup were missing."
        )


def test_skill_34_distinct_groups_each_page_once() -> None:
    """Skill 34 §2: different (source, exc_class) groups each get ONE
    page on their first occurrence today."""
    with tempfile.TemporaryDirectory() as d:
        mon, _, tg = _make_monitor(d)
        try: raise ValueError("a")
        except Exception as e:
            mon.record(source="src1", exc=e)
        try: raise TypeError("b")
        except Exception as e:
            mon.record(source="src2", exc=e)
        assert len(tg.calls) == 2, (
            f"Skill 34 §2: distinct (source, exc_class) groups must "
            f"each fire once. Got {len(tg.calls)}, expected 2."
        )


def test_skill_34_inactive_telegram_still_journals() -> None:
    """Skill 34 §4: when telegram is inactive (env unset), record
    must still write the journal row so the EOD recap and dashboard
    can show the count later."""
    with tempfile.TemporaryDirectory() as d:
        mon, jkb, tg = _make_monitor(d, telegram_active=False)
        try: raise ValueError("test")
        except Exception as exc:
            mon.record(source="quiet_path", exc=exc)
        assert len(tg.calls) == 0   # silent
        from trading_agent.journal_reader import JournalReader
        import datetime as _dt
        from zoneinfo import ZoneInfo
        JournalReader._today_et = staticmethod(
            lambda: _dt.datetime.now(ZoneInfo("US/Eastern")).date()
        )
        silenced = JournalReader(jkb.jsonl_path).silenced_exceptions_today()
        assert len(silenced) == 1, (
            "Skill 34 §4: journal row must be written even when "
            "telegram is inactive — operator can still inspect via "
            "the EOD recap or dashboard."
        )


def test_skill_34_record_never_propagates() -> None:
    """Skill 34 §4: record MUST swallow its own exceptions so the
    caller's except handler can complete. A monitor failure must
    not cascade into a cycle abort."""
    from trading_agent.exception_monitor import ExceptionMonitor
    # Pass a deliberately broken journal_kb that raises on log_signal
    class BrokenJournal:
        jsonl_path = "/tmp/nonexistent"
        def log_signal(self, **kw):
            raise RuntimeError("journal disk full")
    mon = ExceptionMonitor(journal_kb=BrokenJournal(), telegram=None)
    try: raise ValueError("test")
    except Exception as exc:
        # If record raises, this test errors loudly
        mon.record(source="caller", exc=exc)
    # Reached this line → record did NOT propagate the journal failure


def test_skill_34_journal_reader_groups_by_source_and_exc_class() -> None:
    """Skill 34 §3.5: silenced_exceptions_today returns one row per
    (source, exc_class), counting repeats."""
    with tempfile.TemporaryDirectory() as d:
        mon, jkb, _ = _make_monitor(d)
        try: raise ValueError("e")
        except Exception as exc:
            mon.record(source="s1", exc=exc)
            mon.record(source="s1", exc=exc)
            mon.record(source="s1", exc=exc)
        try: raise TypeError("e")
        except Exception as exc:
            mon.record(source="s2", exc=exc)
        from trading_agent.journal_reader import JournalReader
        import datetime as _dt
        from zoneinfo import ZoneInfo
        JournalReader._today_et = staticmethod(
            lambda: _dt.datetime.now(ZoneInfo("US/Eastern")).date()
        )
        groups = JournalReader(jkb.jsonl_path).silenced_exceptions_today()
        assert len(groups) == 2, f"got {len(groups)} groups, expected 2"
        # Sorted by count desc
        assert groups[0].count == 3
        assert groups[0].exc_class == "ValueError"
        assert groups[1].count == 1
        assert groups[1].exc_class == "TypeError"


def test_skill_34_agent_constructs_monitor() -> None:
    """Skill 34 §3.3: TradingAgent must construct an _exception_monitor
    instance during __init__ so every except block can call it."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "trading_agent" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "self._exception_monitor = ExceptionMonitor(" in src, (
        "Skill 34 §3.3: TradingAgent.__init__ must construct "
        "self._exception_monitor. Without it, every record() call "
        "would AttributeError and the catch path would crash."
    )
    # At least the four documented critical paths must call record
    critical_call_sites = [
        "agent._send_telegram_alert",
        "agent._build_eod_summary",
        "agent._maybe_defensive_roll/plan",
        "agent._maybe_defensive_roll/risk_check",
    ]
    for site in critical_call_sites:
        assert site in src, (
            f"Skill 34 §3.3: critical path {site} must call "
            f"self._exception_monitor.record. Documented in §3.3 "
            f"as one of the four MVP-instrumented sites."
        )


def test_skill_34_journal_kb_bypasses_dedup_for_silenced_exceptions() -> None:
    """Skill 34 §3.4: silenced_exception + silenced_exception_paged
    must be in JournalKB's _DEDUP_BYPASS_ACTIONS so successive
    occurrences are all journalled (counter accuracy in the EOD recap)."""
    from trading_agent.journal_kb import _DEDUP_BYPASS_ACTIONS
    assert "silenced_exception" in _DEDUP_BYPASS_ACTIONS, (
        "Skill 34 §3.4: 'silenced_exception' must bypass the journal "
        "dedup gate. Without bypass, successive identical occurrences "
        "are suppressed and the count in the EOD recap underreports."
    )
    assert "silenced_exception_paged" in _DEDUP_BYPASS_ACTIONS, (
        "Skill 34 §3.4: 'silenced_exception_paged' must bypass dedup "
        "so the cross-process paging marker is always written."
    )


def test_skill_34_eod_recap_includes_silenced_exceptions() -> None:
    """Skill 34 §3.5: notify_eod_summary must accept silenced_exceptions
    and render them in the alert body."""
    from trading_agent.telegram_notifier import TelegramNotifier
    n = TelegramNotifier(token="t", chat_id="c")
    captured: list = []
    n._send = lambda text, *, channel="info": (
        captured.append(text), True
    )[1]
    n.notify_eod_summary(
        date_label="Test", account_balance=4500, starting_balance=4700,
        opens_today=[], closes_today=[], realized_pl_today=0.0,
        unrealized_pl_today=0.0, cycles_today=0, errors_today=0,
        stuck_tickers=[],
        silenced_exceptions=[
            {"source": "src1", "exc_class": "ValueError",
             "count": 3, "last_message": "Schwab token expired",
             "ticker": "DIA"},
            {"source": "src2", "exc_class": "TypeError",
             "count": 1, "last_message": "ticker missing",
             "ticker": "SPY"},
        ],
    )
    body = captured[-1]
    assert "Silenced exceptions" in body, (
        "Skill 34 §3.5: EOD body must include 'Silenced exceptions' "
        "section when silenced_exceptions is non-empty."
    )
    assert "ValueError" in body
    assert "TypeError" in body
    assert "Schwab token expired" in body
