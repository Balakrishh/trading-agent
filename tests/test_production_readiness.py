"""
test_production_readiness.py — pin down the 2026-05-06 production-
readiness bundle:

  1. ``client_order_id`` UUID + 2-attempt retry on order submission.
  2. Schwab OAuth refresh: 3-attempt retry + permanent vs transient
     failure distinction.
  3. Position fetch + batch snapshot retry.
  4. Cooldown timestamp embedded in close_failed journal row.
  5. ``log_warning`` helper writes ``action="warning"`` rows that
     bypass the dedup gate.

Tests use stubs / MagicMock to avoid pulling in heavy vendor SDKs.
"""
from __future__ import annotations

import time
import types
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch


# ───────────────────────────────────────────────────────────────────
#  client_order_id idempotency + retry
# ───────────────────────────────────────────────────────────────────

def _make_executor_stub():
    """Bind real OrderExecutor methods to a MagicMock so we can
    exercise _submit_order_with_idempotency without instantiating
    the broker-facing constructor (which requires real Alpaca creds)."""
    from trading_agent.executor import OrderExecutor

    ex = MagicMock(spec=OrderExecutor)
    ex.base_url = "https://paper-api.alpaca.markets/v2"
    ex._headers = MagicMock(return_value={})
    ex._append_to_plan = MagicMock()
    ex._submit_order_with_idempotency = types.MethodType(
        OrderExecutor._submit_order_with_idempotency, ex
    )
    return ex


def test_order_submission_succeeds_first_try_no_retry():
    ex = _make_executor_stub()
    plan = MagicMock(ticker="SPY", strategy_name="Bear Call")

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"id": "ord-abc-123"}
    fake_response.raise_for_status = MagicMock()

    with patch("trading_agent.executor.requests.post",
               return_value=fake_response) as mock_post:
        result = ex._submit_order_with_idempotency(
            plan=plan,
            plan_path="/tmp/plan.json",
            run_id="run-1",
            order_payload={"client_order_id": "test-uuid"},
            client_order_id="test-uuid",
        )

    assert mock_post.call_count == 1
    assert result["status"] == "submitted"
    assert result["order_id"] == "ord-abc-123"
    assert result["client_order_id"] == "test-uuid"
    assert result["retry_attempts"] == 1


def test_order_submission_retries_on_transient_then_succeeds():
    """Connection timeout on attempt 1, success on attempt 2 with
    the SAME client_order_id (so any duplicate the broker accepted
    on attempt 1 collapses server-side)."""
    import requests

    ex = _make_executor_stub()
    plan = MagicMock(ticker="SPY", strategy_name="Bear Call")

    successful_response = MagicMock()
    successful_response.status_code = 200
    successful_response.json.return_value = {"id": "ord-xyz-456"}
    successful_response.raise_for_status = MagicMock()

    side_effects = [
        requests.ConnectTimeout("simulated timeout"),
        successful_response,
    ]

    with patch("trading_agent.executor.requests.post",
               side_effect=side_effects) as mock_post:
        with patch("trading_agent.executor.time.sleep"):  # skip backoff sleep
            result = ex._submit_order_with_idempotency(
                plan=plan,
                plan_path="/tmp/plan.json",
                run_id="run-1",
                order_payload={"client_order_id": "stable-uuid"},
                client_order_id="stable-uuid",
            )

    assert mock_post.call_count == 2
    # Both POSTs must use the SAME payload (same client_order_id).
    payloads = [call.kwargs.get("json") for call in mock_post.call_args_list]
    assert payloads[0] == payloads[1]
    assert payloads[0]["client_order_id"] == "stable-uuid"
    assert result["status"] == "submitted"
    assert result["retry_attempts"] == 2


def test_order_submission_does_not_retry_on_4xx():
    """4xx (e.g. insufficient buying power) is permanent — the broker
    has evaluated and refused.  Retrying with the same payload is
    pointless."""
    import requests

    ex = _make_executor_stub()
    plan = MagicMock(ticker="SPY", strategy_name="Bear Call")

    error_response = MagicMock()
    error_response.status_code = 403
    error_response.json.return_value = {"message": "insufficient buying power"}
    error_response.text = '{"message": "insufficient buying power"}'

    http_err = requests.HTTPError("403 Forbidden")
    http_err.response = error_response
    error_response.raise_for_status = MagicMock(side_effect=http_err)

    with patch("trading_agent.executor.requests.post",
               return_value=error_response) as mock_post:
        result = ex._submit_order_with_idempotency(
            plan=plan,
            plan_path="/tmp/plan.json",
            run_id="run-1",
            order_payload={"client_order_id": "test-uuid"},
            client_order_id="test-uuid",
        )

    assert mock_post.call_count == 1, "4xx must not retry"
    assert result["status"] == "error"
    assert result["client_order_id"] == "test-uuid"


def test_order_submission_exhausts_retries_returns_error():
    import requests

    ex = _make_executor_stub()
    plan = MagicMock(ticker="SPY", strategy_name="Bear Call")

    timeout = requests.ReadTimeout("simulated read timeout")

    with patch("trading_agent.executor.requests.post",
               side_effect=timeout) as mock_post:
        with patch("trading_agent.executor.time.sleep"):
            result = ex._submit_order_with_idempotency(
                plan=plan,
                plan_path="/tmp/plan.json",
                run_id="run-1",
                order_payload={"client_order_id": "test-uuid"},
                client_order_id="test-uuid",
            )

    from trading_agent.executor import ORDER_RETRY_ATTEMPTS
    assert mock_post.call_count == ORDER_RETRY_ATTEMPTS
    assert result["status"] == "error"
    assert result["client_order_id"] == "test-uuid"
    assert result["retry_attempts"] == ORDER_RETRY_ATTEMPTS


# ───────────────────────────────────────────────────────────────────
#  Schwab OAuth refresh retry
# ───────────────────────────────────────────────────────────────────

def _make_oauth_stub():
    """Construct a SchwabOAuth instance with the env-required attrs
    pre-populated.  We bypass the constructor's loaders because we
    only test the _refresh logic here."""
    from trading_agent.schwab_oauth import SchwabOAuth, TokenSet

    oauth = SchwabOAuth.__new__(SchwabOAuth)
    oauth.client_id = "fake-id"
    oauth.client_secret = "fake-secret"
    oauth.redirect_uri = "https://example.test/cb"
    oauth.token_path = MagicMock()
    oauth._tokens = None
    return oauth, TokenSet


def test_oauth_refresh_succeeds_first_try():
    oauth, TokenSet = _make_oauth_stub()
    ts = TokenSet(
        access_token="old", refresh_token="rt-1",
        expires_at=time.time() - 1,                # already expired
        refresh_expires_at=time.time() + 86400,    # 1 day left
    )

    new_token = {
        "access_token": "new", "refresh_token": "rt-2",
        "expires_in": 1800, "token_type": "Bearer",
    }
    response = MagicMock(status_code=200)
    response.json.return_value = new_token
    response.text = ""

    oauth._ingest_token_response = MagicMock(return_value=TokenSet(
        access_token="new", refresh_token="rt-2",
        expires_at=time.time() + 1800,
        refresh_expires_at=time.time() + 86400 * 7,
    ))

    with patch("trading_agent.schwab_oauth.requests.post",
               return_value=response) as mock_post:
        result = oauth._refresh(ts)

    assert mock_post.call_count == 1
    assert result.access_token == "new"


def test_oauth_refresh_retries_on_5xx_then_succeeds():
    oauth, TokenSet = _make_oauth_stub()
    ts = TokenSet(
        access_token="old", refresh_token="rt-1",
        expires_at=time.time() - 1,
        refresh_expires_at=time.time() + 86400,
    )

    bad = MagicMock(status_code=503)
    bad.text = "service unavailable"
    bad.json = MagicMock(side_effect=ValueError())
    good = MagicMock(status_code=200)
    good.json.return_value = {
        "access_token": "new", "refresh_token": "rt-2",
        "expires_in": 1800, "token_type": "Bearer",
    }
    good.text = ""

    oauth._ingest_token_response = MagicMock(return_value=TokenSet(
        access_token="new", refresh_token="rt-2",
        expires_at=time.time() + 1800,
        refresh_expires_at=time.time() + 86400 * 7,
    ))

    with patch("trading_agent.schwab_oauth.requests.post",
               side_effect=[bad, bad, good]) as mock_post:
        with patch("trading_agent.schwab_oauth.time.sleep"):
            result = oauth._refresh(ts)

    assert mock_post.call_count == 3
    assert result.access_token == "new"


def test_oauth_refresh_does_not_retry_on_4xx():
    """401 / 400 = refresh token revoked or wrong client creds.
    Permanent — the operator must re-auth."""
    oauth, TokenSet = _make_oauth_stub()
    ts = TokenSet(
        access_token="old", refresh_token="rt-revoked",
        expires_at=time.time() - 1,
        refresh_expires_at=time.time() + 86400,
    )

    bad = MagicMock(status_code=401)
    bad.text = "invalid_grant"
    bad.json = MagicMock(side_effect=ValueError())

    with patch("trading_agent.schwab_oauth.requests.post",
               return_value=bad) as mock_post:
        with patch("trading_agent.schwab_oauth.time.sleep"):
            try:
                oauth._refresh(ts)
            except RuntimeError as exc:
                assert "permanently failed" in str(exc).lower()
            else:
                raise AssertionError(
                    "Expected RuntimeError on 401 but none raised."
                )

    assert mock_post.call_count == 1, "4xx must not retry"


def test_oauth_refresh_raises_after_all_attempts_exhausted():
    oauth, TokenSet = _make_oauth_stub()
    ts = TokenSet(
        access_token="old", refresh_token="rt-1",
        expires_at=time.time() - 1,
        refresh_expires_at=time.time() + 86400,
    )

    bad = MagicMock(status_code=500)
    bad.text = "server error"
    bad.json = MagicMock(side_effect=ValueError())

    with patch("trading_agent.schwab_oauth.requests.post",
               return_value=bad) as mock_post:
        with patch("trading_agent.schwab_oauth.time.sleep"):
            try:
                oauth._refresh(ts)
            except RuntimeError as exc:
                assert "after" in str(exc).lower()
            else:
                raise AssertionError(
                    "Expected RuntimeError after exhaustion but none raised."
                )

    from trading_agent.schwab_oauth import SchwabOAuth
    assert mock_post.call_count == SchwabOAuth.REFRESH_MAX_ATTEMPTS


# ───────────────────────────────────────────────────────────────────
#  Position-fetch retry
# ───────────────────────────────────────────────────────────────────

def test_position_fetch_retries_on_transient_then_succeeds():
    import requests
    from trading_agent.position_monitor import (
        PositionMonitor, POSITION_FETCH_RETRY_ATTEMPTS,
    )

    pm = PositionMonitor.__new__(PositionMonitor)
    pm.base_url = "https://example.test/v2"
    pm._headers = MagicMock(return_value={})

    success_resp = MagicMock(status_code=200)
    success_resp.json.return_value = []
    success_resp.raise_for_status = MagicMock()

    side_effects = [requests.ConnectTimeout("blip"), success_resp]

    with patch("trading_agent.position_monitor.requests.get",
               side_effect=side_effects) as mock_get:
        with patch("trading_agent.position_monitor.time.sleep"):
            result = pm.fetch_open_positions()

    assert mock_get.call_count == 2
    # Empty broker (success) — distinguishable from None (failure).
    assert result == []


def test_position_fetch_returns_none_when_all_retries_fail():
    import requests
    from trading_agent.position_monitor import (
        PositionMonitor, POSITION_FETCH_RETRY_ATTEMPTS,
    )

    pm = PositionMonitor.__new__(PositionMonitor)
    pm.base_url = "https://example.test/v2"
    pm._headers = MagicMock(return_value={})

    with patch("trading_agent.position_monitor.requests.get",
               side_effect=requests.ConnectTimeout("permanent")) as mock_get:
        with patch("trading_agent.position_monitor.time.sleep"):
            result = pm.fetch_open_positions()

    assert mock_get.call_count == POSITION_FETCH_RETRY_ATTEMPTS
    assert result is None, "Must return None so dedup gate fails closed"


# ───────────────────────────────────────────────────────────────────
#  Cooldown surface in close_failed journal row
# ───────────────────────────────────────────────────────────────────

def test_close_failed_row_carries_streak_pre_cooldown():
    """A close_failed row before the threshold should carry
    partial_close_streak / threshold but NOT close_cooldown_until."""
    from trading_agent.agent import TradingAgent

    agent = MagicMock(spec=TradingAgent)
    agent.journal_kb = MagicMock()
    agent._cached_price = MagicMock(return_value=100.0)
    agent._partial_close_count = {"SPY": 1}  # 1/3 — pre-lockout
    agent._close_cooldown_until = {}
    agent._journal_close_event = types.MethodType(
        TradingAgent._journal_close_event, agent
    )

    spread = MagicMock(underlying="SPY")
    ctx = {
        "strategy": "Bear Call",
        "exit_signal": "regime_shift",
        "exit_reason": "regime flipped",
        "exit_immediate": False,
        "net_unrealized_pl": -10.0,
        "original_credit": 1.0,
        "max_loss": 4.0,
        "spread_width": 5.0,
        "expiration": "2026-05-30",
        "short_strikes": [550],
        "regime_at_close": "BEAR",
        "origin": "trade_plan",
    }
    agent._journal_close_event(
        spread, ctx, leg_results=[], fill_status="partial", dry_run=False,
    )

    call_kwargs = agent.journal_kb.log_signal.call_args.kwargs
    assert call_kwargs["action"] == "close_failed"
    rs = call_kwargs["raw_signal"]
    assert rs["partial_close_streak"] == 1
    assert "partial_close_threshold" in rs
    assert "close_cooldown_until" not in rs


def test_close_failed_row_carries_cooldown_when_locked():
    """When the cooldown_until dict is populated, the row must
    embed the ISO timestamp + reason for dashboard rendering."""
    from trading_agent.agent import TradingAgent

    agent = MagicMock(spec=TradingAgent)
    agent.journal_kb = MagicMock()
    agent._cached_price = MagicMock(return_value=100.0)
    cooldown_deadline = datetime.now(timezone.utc) + timedelta(minutes=60)
    agent._partial_close_count = {"SPY": 3}
    agent._close_cooldown_until = {"SPY": cooldown_deadline}
    agent._journal_close_event = types.MethodType(
        TradingAgent._journal_close_event, agent
    )

    spread = MagicMock(underlying="SPY")
    ctx = {
        "strategy": "Bear Call",
        "exit_signal": "regime_shift",
        "exit_reason": "regime flipped",
        "exit_immediate": False,
        "net_unrealized_pl": -10.0,
        "original_credit": 1.0,
        "max_loss": 4.0,
        "spread_width": 5.0,
        "expiration": "2026-05-30",
        "short_strikes": [550],
        "regime_at_close": "BEAR",
        "origin": "trade_plan",
    }
    agent._journal_close_event(
        spread, ctx, leg_results=[], fill_status="partial", dry_run=False,
    )

    call_kwargs = agent.journal_kb.log_signal.call_args.kwargs
    rs = call_kwargs["raw_signal"]
    assert rs["partial_close_streak"] == 3
    assert "close_cooldown_until" in rs
    # ISO format: starts with the year.
    assert rs["close_cooldown_until"].startswith(
        str(cooldown_deadline.year)
    )
    assert "manual broker intervention" in rs["close_cooldown_reason"]


def test_closed_row_does_not_carry_cooldown_fields():
    """A successful close (action='closed') must NOT include the
    cooldown surface — it's exclusive to close_failed rows."""
    from trading_agent.agent import TradingAgent

    agent = MagicMock(spec=TradingAgent)
    agent.journal_kb = MagicMock()
    agent._cached_price = MagicMock(return_value=100.0)
    agent._partial_close_count = {}
    agent._close_cooldown_until = {}
    agent._journal_close_event = types.MethodType(
        TradingAgent._journal_close_event, agent
    )

    spread = MagicMock(underlying="QQQ")
    ctx = {
        "strategy": "Iron Condor",
        "exit_signal": "profit_target",
        "exit_reason": "75% captured",
        "exit_immediate": True,
        "net_unrealized_pl": 30.0,
        "original_credit": 0.5,
        "max_loss": 1.5,
        "spread_width": 2.0,
        "expiration": "2026-05-30",
        "short_strikes": [400, 420],
        "regime_at_close": "RANGE",
        "origin": "trade_plan",
    }
    agent._journal_close_event(
        spread, ctx, leg_results=[], fill_status="complete", dry_run=False,
    )

    rs = agent.journal_kb.log_signal.call_args.kwargs["raw_signal"]
    assert "partial_close_streak" not in rs
    assert "close_cooldown_until" not in rs


# ───────────────────────────────────────────────────────────────────
#  log_warning helper
# ───────────────────────────────────────────────────────────────────

def test_log_warning_writes_action_warning_row(tmp_path):
    from trading_agent.journal_kb import JournalKB

    jkb = JournalKB(str(tmp_path), run_mode="live")
    jkb.log_warning(
        source="executor",
        ticker="SPY",
        message="Order POST failed after 2 attempts",
        context={"client_order_id": "abc-123", "retry_attempts": 2},
    )

    import json
    contents = open(jkb.jsonl_path).read().strip().splitlines()
    assert len(contents) == 1
    rec = json.loads(contents[0])
    assert rec["action"] == "warning"
    assert rec["ticker"] == "SPY"
    assert rec["raw_signal"]["source"] == "executor"
    assert rec["raw_signal"]["context"]["client_order_id"] == "abc-123"


def test_log_warning_bypasses_dedup(tmp_path):
    """Two identical warnings must produce two journal rows (warning
    is in _DEDUP_BYPASS_ACTIONS)."""
    from trading_agent.journal_kb import JournalKB

    jkb = JournalKB(str(tmp_path), run_mode="live")
    for _ in range(3):
        jkb.log_warning(
            source="schwab_oauth",
            message="refresh failed",
        )

    import json
    contents = open(jkb.jsonl_path).read().strip().splitlines()
    # All 3 must have made it through; dedup must NOT have suppressed.
    assert len(contents) == 3
    for line in contents:
        rec = json.loads(line)
        assert rec["action"] == "warning"
