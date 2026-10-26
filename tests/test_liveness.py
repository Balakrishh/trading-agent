"""Skill 64 — liveness watchdog."""
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from trading_agent import liveness as lv

ET = ZoneInfo("America/New_York")


def et(h, m, day=13):
    return datetime(2026, 10, day, h, m, tzinfo=ET).astimezone(timezone.utc)


@pytest.mark.parametrize("now, last, in_session, server_ok, status", [
    (et(9, 35), et(9, 0), True, True, "idle"),          # inside the open grace
    (et(9, 41), None, True, True, "down"),              # never ran today
    (et(11, 0), et(10, 58), True, True, "ok"),
    (et(11, 0), et(10, 49), True, True, "down"),        # 11 min stale
    (et(11, 0), et(10, 59), True, False, "down"),       # data server down
    (et(17, 0), et(16, 5), False, False, "idle"),       # after hours
])
def test_assess(now, last, in_session, server_ok, status):
    assert lv.assess(now, last, in_session, server_ok)[0] == status


def test_assess_reason_names_both_problems():
    _, reason = lv.assess(et(11, 0), et(10, 30), True, False)
    assert "30 min" in reason and "10:30 ET" in reason and "data server" in reason


def test_one_page_per_outage_then_recovery():
    s = lv.LivenessState()
    now = et(11, 0)
    s, alert = lv.step(s, "down", "x", now)
    assert alert == "down" and s.alerted and s.since == now.isoformat()
    s2, alert = lv.step(s, "down", "x", now + timedelta(minutes=5))
    assert alert is None and s2 == s
    s3, alert = lv.step(s2, "ok", "", now + timedelta(minutes=10))
    assert alert == "recovered" and not s3.alerted
    assert lv.step(s3, "ok", "", now)[1] is None
    assert lv.step(s3, "idle", "", now)[1] is None


def test_last_activity_reads_the_newest_timestamp(tmp_path):
    p = tmp_path / "signals.jsonl"
    rows = [{"timestamp": "2026-10-13T14:00:00+00:00"},
            {"timestamp": "2026-10-13T14:05:00+00:00"}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n{broken\n")
    assert lv.last_activity(p) == datetime(2026, 10, 13, 14, 5, tzinfo=timezone.utc)
    assert lv.last_activity(tmp_path / "missing.jsonl") is None


def test_state_round_trip_and_corrupt_file(tmp_path):
    p = tmp_path / "liveness_state.json"
    lv.save_state(lv.LivenessState(True, "2026-10-13T15:00:00+00:00", "r"), p)
    assert lv.load_state(p).alerted is True
    p.write_text("{not json")
    assert lv.load_state(p) == lv.LivenessState()


def test_main_pages_once_and_keeps_state_when_send_fails(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "trade_journal").mkdir()
    monkeypatch.setattr(lv, "assess", lambda *a: ("down", "no agent cycle for 30 min"))
    monkeypatch.setattr(lv, "server_healthy", lambda: True)
    monkeypatch.setattr(lv, "schwab_ready", lambda: (True, ""))
    monkeypatch.setattr(lv, "is_within_market_hours", lambda now: True)
    sent = []
    from trading_agent import telegram_notifier as tn

    class FakeTG:
        is_active = True

        def notify_agent_down(self, reason):
            sent.append(reason)
            return ok[0]
    monkeypatch.setattr(tn, "TelegramNotifier", FakeTG)
    ok = [False]
    lv.main([])
    assert sent == ["no agent cycle for 30 min"]
    assert not lv.load_state().alerted          # unsent → retried next run
    ok[0] = True
    lv.main([])
    lv.main([])
    assert len(sent) == 2 and lv.load_state().alerted


def test_alert_texts():
    from trading_agent.telegram_notifier import TelegramNotifier
    n = TelegramNotifier(token="t", chat_id="c")
    out = []
    n._send = lambda text, *, channel="info": (out.append((text, channel)), True)[1]
    n.notify_agent_down("no agent cycle for 12 min (last row 10:48 ET) < x")
    assert "restart trading-agent" in out[-1][0] and "&lt; x" in out[-1][0]
    assert out[-1][1] == "error"
    n.notify_agent_recovered("2026-10-13T15:00:00+00:00")
    assert "2026-10-13T15:00Z" in out[-1][0]
