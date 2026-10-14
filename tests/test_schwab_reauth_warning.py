"""Skill 16 §4 — the agent pages when the weekly Schwab login is due."""
from types import SimpleNamespace

import pytest

from trading_agent.agent import TradingAgent


def _agent(monkeypatch, days_left):
    a = TradingAgent.__new__(TradingAgent)
    sent = []
    a.telegram = SimpleNamespace(notify_schwab_reauth_due=lambda days_left: True)
    a._send_telegram_alert = lambda ticker, alert_type, fn, **kw: sent.append((ticker, alert_type, kw))
    oauth = SimpleNamespace(refresh_days_left=lambda: days_left)
    monkeypatch.setattr("trading_agent.schwab_oauth.SchwabOAuth.from_env",
                        classmethod(lambda cls: oauth))
    return a, sent


@pytest.mark.parametrize("days_left, fires", [
    (6.5, False), (2.01, False), (1.8, True), (0.2, True), (-1.0, True), (None, False),
])
def test_warns_only_inside_the_window(monkeypatch, days_left, fires):
    a, sent = _agent(monkeypatch, days_left)
    a._maybe_warn_schwab_reauth()
    assert bool(sent) is fires
    if fires:
        assert sent[0][:2] == ("__schwab__", "schwab_reauth_due")


def test_silent_when_schwab_is_not_configured(monkeypatch):
    a, sent = _agent(monkeypatch, 0.5)

    def boom(cls):
        raise ValueError("SCHWAB_CLIENT_ID missing")
    monkeypatch.setattr("trading_agent.schwab_oauth.SchwabOAuth.from_env", classmethod(boom))
    a._maybe_warn_schwab_reauth()
    assert sent == []


def test_alert_text_says_expired_or_hours_left():
    from trading_agent.telegram_notifier import TelegramNotifier
    n = TelegramNotifier(token="t", chat_id="c")
    out = []
    n._send = lambda text, *, channel="info": (out.append((text, channel)), True)[1]
    n.notify_schwab_reauth_due(1.5)
    assert "36 h" in out[-1][0] and out[-1][1] == "error"
    n.notify_schwab_reauth_due(-0.1)
    assert "expired" in out[-1][0]
