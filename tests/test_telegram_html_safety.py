"""Skill 32 §4 — free text never breaks a Telegram HTML message.

2026-10-13: the EOD summary was rejected twice ("can't parse entities:
Unsupported start tag") because a reject reason carried a raw '<'
("Credit-to-width ratio 0.2333 < minimum 0.25").
"""
from types import SimpleNamespace

from trading_agent import telegram_notifier as tn
from trading_agent.telegram_notifier import TelegramNotifier, _html_to_plain

REASON = "Credit-to-width ratio 0.2333 < minimum 0.25; fallback Calendar Spread"


def _capture():
    n = TelegramNotifier(token="t", chat_id="c")
    sent = []
    n._send = lambda text, *, channel="info": (sent.append(text), True)[1]
    return n, sent


def test_eod_summary_escapes_reject_reasons_and_silenced_messages():
    n, sent = _capture()
    n.notify_eod_summary(
        date_label="2026-10-13", account_balance=29430.52, starting_balance=29560.52,
        opens_today=[], closes_today=[], realized_pl_today=0.0, unrealized_pl_today=-130.0,
        cycles_today=372, errors_today=0, stuck_tickers=[{"ticker": "GLD", "reason": "x < y"}],
        silenced_exceptions=[{"source": "a<b", "exc_class": "RuntimeError", "count": 2,
                              "last_message": "ratio < floor"}],
        reject_reasons=[(REASON, 13)])
    body = sent[-1]
    assert "0.2333 &lt; minimum" in body and "ratio &lt; floor" in body
    assert "x &lt; y" in body and "a&lt;b" in body
    assert "< m" not in body


def test_close_and_error_alerts_escape_free_text():
    n, sent = _capture()
    n.notify_position_closed("XLF", "Iron Condor", exit_signal="stop_loss",
                             exit_reason="C/W 0.26 < floor 0.53", realized_pl=-10.0,
                             original_credit=0.5, max_loss=100.0)
    assert "0.26 &lt; floor" in sent[-1]
    n.notify_silenced_exception(source="s", exc_class="E", message="a < b")
    assert "a &lt; b" in sent[-1]
    n.notify_trading_halt("loss < limit")
    assert "loss &lt; limit" in sent[-1]


def test_html_to_plain_strips_tags_and_unescapes():
    assert _html_to_plain("<b>Day P&amp;L:</b> <code>3×</code> a &lt; b") == "Day P&L: 3× a < b"


def test_send_resends_plain_text_when_telegram_cannot_parse(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append(json)
        if "parse_mode" in json:
            return SimpleNamespace(status_code=400, text='{"description":"can\'t parse entities"}')
        return SimpleNamespace(status_code=200, text="ok")

    monkeypatch.setattr(tn.requests, "post", fake_post)
    n = TelegramNotifier(token="t", chat_id="c")
    assert n._send("<b>x</b> a < b") is True
    assert len(calls) == 2 and "parse_mode" not in calls[1]
    assert calls[1]["text"] == "x a < b"


def test_send_does_not_resend_on_other_errors(monkeypatch):
    calls = []
    monkeypatch.setattr(tn.requests, "post", lambda url, json, timeout: (
        calls.append(json), SimpleNamespace(status_code=403, text="forbidden"))[1])
    assert TelegramNotifier(token="t", chat_id="c")._send("hi") is False
    assert len(calls) == 1
