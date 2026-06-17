"""Conformance tests for skill 42 — portfolio alert scheduler.

Pinned behaviors:

- Env opt-out short-circuits before any I/O — §4
- Market-hours gate skips outside the window unless force=True — §4
- Empty holdings file = silent skip — §4
- Dedup short-circuits same body within UTC day — §1.6, §4
- Dedup row only written on confirmed Telegram delivery — §4
- Failed Telegram send leaves the dedup slot free for retry — §4
- Body composition includes documented sections — §3.2
- Scheduler is read-only — does not import executor.*, does not
  redefine _score_* — §4
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from trading_agent.holdings_store import HoldingsSnapshot
from trading_agent.long_term_evaluator import (
    LongTermEvaluator,
    Recommendation,
    RecommendationLeg,
)
from trading_agent.portfolio_alert_scheduler import (
    ALERT_DEDUP_KEY_PREFIX,
    AlertResult,
    SchedulerDeps,
    compose_digest_body,
    run_scheduler,
)
from trading_agent.positions_provider import (
    ManualPositionsProvider,
    Position,
)


# ---------------------------------------------------------------------------
# Test fixtures — minimal but realistic
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 6, 16, 14, 30, tzinfo=timezone.utc)  # 10:30 ET (open)

_HOLDINGS_BLOB = (
    '[{"ticker": "NOK", "qty": 100, "avg_cost": 4.76, "kind": "stock"},'
    ' {"ticker": "MSFT", "qty": 9, "avg_cost": 414.66, "kind": "stock"}]'
)


def _fixture_recommendation() -> Recommendation:
    return Recommendation(
        ticker="NOK",
        strategy="covered_call",
        legs=[RecommendationLeg(
            action="STO", occ_symbol="NOK   270115C00015000",
            qty=1, side="short", limit_price=0.18,
        )],
        entry_limit=0.18,
        entry_kind="credit",
        take_profit_limit=0.09,
        stop_trigger=4.38,
        stop_kind="underlying_price",
        stop_limit_offset_pct=0.05,
        score=0.123,
        rationale="Sell 1× CC @ $15 (45d), |Δ|=0.22, ann. yield 8.7%, "
                  "POP 78%. Credit $18.00/contract.",
        metrics={
            "annualised_return": 0.087, "pop": 0.78, "dte": 45.0,
            "short_delta_abs": 0.22, "credit": 18.0,
            "capital_at_risk": 458.0, "static_return": 0.039,
            "if_assigned_return": 1.27,
        },
    )


class _FakeEvaluator:
    def __init__(self, recs):
        self._recs = recs

    def recommend(self, _watchlist):
        return list(self._recs)


def _build_deps(
    *,
    holdings_blob: str = _HOLDINGS_BLOB,
    watchlist: List[str] = None,
    recs=None,
    market_open: bool = True,
    dedup_hit: bool = False,
    telegram_returns: bool = True,
) -> tuple[SchedulerDeps, dict]:
    """Return SchedulerDeps + a shared `events` dict tests can inspect."""
    events: Dict[str, Any] = {
        "telegram_sent": False, "dedup_written": False,
        "body_seen": None, "dedup_key_seen": None,
    }
    watchlist = watchlist if watchlist is not None else ["NOK", "MSFT"]
    recs = recs if recs is not None else [_fixture_recommendation()]

    def _send(body, dedup_key):
        events["body_seen"] = body
        events["dedup_key_seen"] = dedup_key
        events["telegram_sent"] = telegram_returns
        return telegram_returns

    def _dedup_write(key, body_hash):
        events["dedup_written"] = True

    deps = SchedulerDeps(
        holdings_loader=lambda: HoldingsSnapshot(
            raw_paste=holdings_blob, parsed_count=2,
        ) if holdings_blob else HoldingsSnapshot(),
        watchlist_loader=lambda: list(watchlist),
        evaluator_factory=lambda _positions: _FakeEvaluator(recs),
        telegram_send=_send,
        market_hours_check=lambda _now: market_open,
        journal_dedup_check=lambda _key: dedup_hit,
        journal_dedup_write=_dedup_write,
    )
    return deps, events


# ---------------------------------------------------------------------------
# §4 — Env opt-out
# ---------------------------------------------------------------------------

def test_env_opt_out_skips_before_io(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_ALERTS_ENABLED", "false")
    called = {"holdings": 0}
    deps, _ = _build_deps()
    # Wrap holdings_loader so we'd notice if it ran.
    original = deps.holdings_loader

    def _spy():
        called["holdings"] += 1
        return original()

    deps.holdings_loader = _spy
    result = run_scheduler(deps=deps, now=_NOW)
    assert not result.sent
    assert "PORTFOLIO_ALERTS_ENABLED" in result.skipped_reason
    assert called["holdings"] == 0   # short-circuited before I/O


@pytest.mark.parametrize("value", ["false", "0", "no", "off", "FALSE"])
def test_env_opt_out_accepts_common_falsey_values(monkeypatch, value):
    monkeypatch.setenv("PORTFOLIO_ALERTS_ENABLED", value)
    deps, _ = _build_deps()
    result = run_scheduler(deps=deps, now=_NOW)
    assert not result.sent


# ---------------------------------------------------------------------------
# §4 — Market-hours gate
# ---------------------------------------------------------------------------

def test_market_hours_gate_skips_outside_window(monkeypatch):
    monkeypatch.delenv("PORTFOLIO_ALERTS_ENABLED", raising=False)
    deps, _ = _build_deps(market_open=False)
    result = run_scheduler(deps=deps, now=_NOW)
    assert not result.sent
    assert "outside market hours" in result.skipped_reason


def test_force_bypasses_market_hours_gate(monkeypatch):
    monkeypatch.delenv("PORTFOLIO_ALERTS_ENABLED", raising=False)
    deps, events = _build_deps(market_open=False)
    result = run_scheduler(deps=deps, now=_NOW, force=True)
    assert result.sent
    assert events["telegram_sent"] is True


# ---------------------------------------------------------------------------
# §4 — Empty holdings / watchlist
# ---------------------------------------------------------------------------

def test_empty_holdings_file_skips(monkeypatch):
    monkeypatch.delenv("PORTFOLIO_ALERTS_ENABLED", raising=False)
    deps, _ = _build_deps(holdings_blob="")
    result = run_scheduler(deps=deps, now=_NOW)
    assert not result.sent
    assert "holdings file empty" in result.skipped_reason


def test_empty_watchlist_skips(monkeypatch):
    monkeypatch.delenv("PORTFOLIO_ALERTS_ENABLED", raising=False)
    deps, _ = _build_deps(watchlist=[])
    result = run_scheduler(deps=deps, now=_NOW)
    assert not result.sent
    assert "watchlist empty" in result.skipped_reason


def test_unparseable_holdings_blob_skips(monkeypatch):
    monkeypatch.delenv("PORTFOLIO_ALERTS_ENABLED", raising=False)
    deps, _ = _build_deps(holdings_blob="not json at all")
    result = run_scheduler(deps=deps, now=_NOW)
    assert not result.sent
    assert "holdings parse failed" in result.skipped_reason


# ---------------------------------------------------------------------------
# §4 — Dedup behavior
# ---------------------------------------------------------------------------

def test_dedup_short_circuits_same_body_within_utc_day(monkeypatch):
    monkeypatch.delenv("PORTFOLIO_ALERTS_ENABLED", raising=False)
    deps, events = _build_deps(dedup_hit=True)
    result = run_scheduler(deps=deps, now=_NOW)
    assert not result.sent
    assert "dedup" in result.skipped_reason
    assert events["telegram_sent"] is False


def test_dedup_row_only_written_on_confirmed_delivery(monkeypatch):
    monkeypatch.delenv("PORTFOLIO_ALERTS_ENABLED", raising=False)
    deps, events = _build_deps(telegram_returns=False)
    result = run_scheduler(deps=deps, now=_NOW)
    assert not result.sent
    assert "telegram" in result.skipped_reason.lower()
    assert events["dedup_written"] is False    # no dedup burn on failure


def test_successful_send_writes_dedup_row(monkeypatch):
    monkeypatch.delenv("PORTFOLIO_ALERTS_ENABLED", raising=False)
    deps, events = _build_deps()
    result = run_scheduler(deps=deps, now=_NOW)
    assert result.sent
    assert events["dedup_written"] is True
    assert events["dedup_key_seen"].startswith(ALERT_DEDUP_KEY_PREFIX)
    assert "2026-06-16" in events["dedup_key_seen"]


def test_dry_run_renders_but_does_not_send(monkeypatch):
    monkeypatch.delenv("PORTFOLIO_ALERTS_ENABLED", raising=False)
    deps, events = _build_deps()
    result = run_scheduler(deps=deps, now=_NOW, dry_run=True)
    assert not result.sent
    assert result.skipped_reason == "dry-run"
    assert events["telegram_sent"] is False
    assert events["dedup_written"] is False
    # Body should still be computed so --print-body works.
    assert result.body_chars > 0
    assert "Portfolio Review" in result.debug.get("body", "")


# ---------------------------------------------------------------------------
# §3.2 — Body composition
# ---------------------------------------------------------------------------

def test_compose_digest_includes_documented_sections():
    positions = ManualPositionsProvider.from_json_text(_HOLDINGS_BLOB).snapshot()
    body = compose_digest_body(
        positions=positions,
        watchlist_symbols=["NOK", "MSFT"],
        recommendations=[_fixture_recommendation()],
        now=_NOW,
    )
    # Header with date + UTC stamp
    assert "Portfolio Review" in body
    assert "2026-06-16" in body
    assert "UTC" in body
    # Holdings one-liner
    assert "Holdings:" in body
    # Sector line (NOK is Technology in the sector map)
    assert "Sector:" in body
    # Income overlay section + the recommendation
    assert "Income overlay" in body
    assert "NOK" in body
    # Bracket sketch
    assert "Bracket:" in body
    assert "STO @" in body
    assert "BTC @" in body
    # Skipped section — MSFT has 9 shares < 100
    assert "MSFT" in body and "9" in body
    # Footer
    assert "skill 40" in body


def test_compose_digest_handles_zero_recommendations():
    positions = ManualPositionsProvider.from_json_text(_HOLDINGS_BLOB).snapshot()
    body = compose_digest_body(
        positions=positions, watchlist_symbols=["NOK"],
        recommendations=[], now=_NOW,
    )
    assert "Income overlay" in body
    assert "none today" in body


def test_compose_digest_truncates_top_n():
    positions = ManualPositionsProvider.from_json_text(_HOLDINGS_BLOB).snapshot()
    recs = [_fixture_recommendation() for _ in range(8)]
    body = compose_digest_body(
        positions=positions, watchlist_symbols=["NOK"],
        recommendations=recs, now=_NOW, cc_top_n=3,
    )
    assert "5 more" in body   # 8 recs - 3 shown = 5 more


# ---------------------------------------------------------------------------
# §4 — Read-only contract
# ---------------------------------------------------------------------------

def test_scheduler_does_not_import_executor():
    """Skill 42 §4 — scheduler is read-only by design."""
    import trading_agent.portfolio_alert_scheduler as mod
    src = open(mod.__file__).read()
    assert "from trading_agent.executor" not in src
    assert "submit_order" not in src
    assert "place_order" not in src


# ---------------------------------------------------------------------------
# §3.3 — notify_portfolio_review channel routing
# ---------------------------------------------------------------------------

def test_notify_portfolio_review_routes_to_long_term_channel(monkeypatch):
    """When TELEGRAM_LONG_TERM_BOT_TOKEN is set, the digest uses it."""
    from trading_agent.telegram_notifier import TelegramNotifier
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "info-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "info-chat")
    monkeypatch.setenv("TELEGRAM_LONG_TERM_BOT_TOKEN", "lt-token")
    monkeypatch.setenv("TELEGRAM_LONG_TERM_CHAT_ID", "lt-chat")
    n = TelegramNotifier()
    assert n.long_term_token == "lt-token"
    assert n.long_term_chat_id == "lt-chat"
    assert n.long_term_channel_distinct is True


def test_long_term_channel_falls_back_to_info(monkeypatch):
    """Single-bot deployments stay unchanged — no LT env vars set →
    long_term channel reuses info channel credentials."""
    from trading_agent.telegram_notifier import TelegramNotifier
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "info-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "info-chat")
    monkeypatch.delenv("TELEGRAM_LONG_TERM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_LONG_TERM_CHAT_ID", raising=False)
    n = TelegramNotifier()
    assert n.long_term_token == "info-token"
    assert n.long_term_chat_id == "info-chat"
    assert n.long_term_channel_distinct is False


def test_send_routes_long_term_channel_credentials(monkeypatch):
    """_send(channel='long_term') uses LT creds, not info or error."""
    from trading_agent.telegram_notifier import TelegramNotifier
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "info-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "info-chat")
    monkeypatch.setenv("TELEGRAM_ERROR_BOT_TOKEN", "err-token")
    monkeypatch.setenv("TELEGRAM_ERROR_CHAT_ID", "err-chat")
    monkeypatch.setenv("TELEGRAM_LONG_TERM_BOT_TOKEN", "lt-token")
    monkeypatch.setenv("TELEGRAM_LONG_TERM_CHAT_ID", "lt-chat")
    n = TelegramNotifier()

    captured = {}

    def _fake_post(url, json, timeout):
        captured["url"] = url
        captured["chat_id"] = json["chat_id"]
        class _Resp: status_code = 200
        return _Resp()

    monkeypatch.setattr(
        "trading_agent.telegram_notifier.requests.post", _fake_post,
    )
    assert n._send("body", channel="long_term") is True
    assert "lt-token" in captured["url"]
    assert captured["chat_id"] == "lt-chat"


def test_scheduler_does_not_define_score_helpers():
    """CI invariant 2 — scoring helpers may only be defined in
    chain_scanner.py / decision_engine.py."""
    import trading_agent.portfolio_alert_scheduler as mod
    src = open(mod.__file__).read()
    # The scheduler may CALL scoring helpers via the evaluator but must
    # never DEFINE one. Any `def _score_` definition would trip the
    # invariant scanner anyway; this is a belt-and-suspenders check.
    assert "def _score_" not in src
