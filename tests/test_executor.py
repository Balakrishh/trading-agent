"""Tests for the order executor — dry-run, plan persistence, live quote refresh."""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from trading_agent.executor import OrderExecutor, MAX_HISTORY
from trading_agent.risk_manager import RiskVerdict
from trading_agent.strategy import SpreadPlan, SpreadLeg


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_plan(ticker="SPY", valid=True):
    return SpreadPlan(
        ticker=ticker, strategy_name="Bull Put Spread", regime="bullish",
        legs=[
            SpreadLeg("SPY250425P00480000", 480.0, "sell", "put",
                      -0.15, -0.05, 1.20, 1.40, 1.30),
            SpreadLeg("SPY250425P00475000", 475.0, "buy", "put",
                      -0.10, -0.03, 0.80, 1.00, 0.90),
        ],
        spread_width=5.0, net_credit=0.40, max_loss=460.0,
        credit_to_width_ratio=0.08, expiration="2025-04-25",
        reasoning="Test", valid=valid,
    )


def _make_verdict(approved=True, plan=None) -> RiskVerdict:
    if plan is None:
        plan = _make_plan(valid=approved)
    return RiskVerdict(
        approved=approved, plan=plan, account_balance=100_000,
        max_allowed_loss=2_000,
        checks_passed=["check1"] if approved else [],
        checks_failed=[] if approved else ["failed_check"],
        summary="APPROVED" if approved else "REJECTED",
    )


def _make_executor(tmp_path, dry_run=True, data_provider=None):
    return OrderExecutor(
        api_key="test", secret_key="test",
        trade_plan_dir=str(tmp_path), dry_run=dry_run,
        data_provider=data_provider,
    )


# ------------------------------------------------------------------
# Dry-run execution
# ------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_writes_plan_file(self, tmp_path):
        executor = _make_executor(tmp_path)
        result = executor.execute(_make_verdict(approved=True))

        assert result["status"] == "dry_run"
        assert os.path.exists(result["plan_file"])

    def test_rejected_trade_not_executed(self, tmp_path):
        executor = _make_executor(tmp_path)
        result = executor.execute(_make_verdict(approved=False))
        assert result["status"] == "rejected"

    def test_result_contains_run_id(self, tmp_path):
        executor = _make_executor(tmp_path)
        result = executor.execute(_make_verdict(approved=True))
        assert "run_id" in result
        assert len(result["run_id"]) == 15  # YYYYMMDD_HHMMSS


# ------------------------------------------------------------------
# Single-file state_history persistence (new format)
# ------------------------------------------------------------------

class TestPlanPersistence:
    def test_file_uses_ticker_name_not_timestamp(self, tmp_path):
        """Plan file is trade_plan_{TICKER}.json, not timestamped."""
        executor = _make_executor(tmp_path)
        executor.execute(_make_verdict(approved=True))
        files = os.listdir(tmp_path)
        assert any(f == "trade_plan_SPY.json" for f in files)
        # No old-style timestamped files
        assert not any(f.count("_") >= 3 for f in files)

    def test_state_history_exists_in_file(self, tmp_path):
        executor = _make_executor(tmp_path)
        result = executor.execute(_make_verdict(approved=True))

        with open(result["plan_file"]) as f:
            data = json.load(f)

        assert "state_history" in data
        assert "ticker" in data
        assert "last_updated" in data
        assert len(data["state_history"]) == 1

    def test_state_history_entry_has_required_fields(self, tmp_path):
        executor = _make_executor(tmp_path)
        result = executor.execute(_make_verdict(approved=True))

        with open(result["plan_file"]) as f:
            entry = json.load(f)["state_history"][0]

        assert entry["trade_plan"]["ticker"] == "SPY"
        assert entry["risk_verdict"]["approved"] is True
        assert entry["risk_verdict"]["account_balance"] == 100_000
        assert entry["mode"] == "dry_run"
        assert "run_id" in entry
        assert "timestamp" in entry

    def test_multiple_runs_accumulate_in_state_history(self, tmp_path):
        executor = _make_executor(tmp_path)
        executor.execute(_make_verdict(approved=True))
        executor.execute(_make_verdict(approved=True))
        executor.execute(_make_verdict(approved=False))

        plan_file = os.path.join(tmp_path, "trade_plan_SPY.json")
        with open(plan_file) as f:
            data = json.load(f)

        assert len(data["state_history"]) == 3

    def test_different_tickers_use_separate_files(self, tmp_path):
        executor = _make_executor(tmp_path)
        executor.execute(_make_verdict(plan=_make_plan(ticker="SPY")))
        executor.execute(_make_verdict(plan=_make_plan(ticker="QQQ")))

        assert os.path.exists(os.path.join(tmp_path, "trade_plan_SPY.json"))
        assert os.path.exists(os.path.join(tmp_path, "trade_plan_QQQ.json"))

    def test_max_history_trimming(self, tmp_path):
        executor = _make_executor(tmp_path)
        for _ in range(MAX_HISTORY + 5):
            executor.execute(_make_verdict(approved=True))

        plan_file = os.path.join(tmp_path, "trade_plan_SPY.json")
        with open(plan_file) as f:
            data = json.load(f)

        assert len(data["state_history"]) == MAX_HISTORY

    def test_append_to_plan_updates_latest_entry(self, tmp_path):
        executor = _make_executor(tmp_path)
        result = executor.execute(_make_verdict(approved=True))
        run_id = result["run_id"]

        executor._append_to_plan(result["plan_file"], run_id,
                                  {"order_result": {"status": "submitted"}})

        with open(result["plan_file"]) as f:
            data = json.load(f)

        latest = data["state_history"][-1]
        assert latest["order_result"]["status"] == "submitted"


# ------------------------------------------------------------------
# Live quote refresh before execution
# ------------------------------------------------------------------

class TestLiveQuoteRefresh:
    def _make_provider_with_quotes(self, sold_bid, bought_ask):
        provider = MagicMock()
        provider.fetch_option_quotes.return_value = {
            "SPY250425P00480000": {"bid": sold_bid, "ask": sold_bid + 0.10, "mid": sold_bid + 0.05},
            "SPY250425P00475000": {"bid": bought_ask - 0.10, "ask": bought_ask, "mid": bought_ask - 0.05},
        }
        return provider

    def test_refresh_recalculates_limit_price(self, tmp_path):
        """Live credit from quotes is used as limit_price, not stale plan credit."""
        provider = self._make_provider_with_quotes(sold_bid=1.50, bought_ask=0.70)
        executor = _make_executor(tmp_path, dry_run=False, data_provider=provider)
        plan = _make_plan()

        live_credit = executor._refresh_limit_price(plan)
        # sold.bid=1.50 - bought.ask=0.70 = 0.80
        assert live_credit == pytest.approx(0.80)

    def test_refresh_returns_none_when_no_provider(self, tmp_path):
        executor = _make_executor(tmp_path, data_provider=None)
        assert executor._refresh_limit_price(_make_plan()) is None

    def test_refresh_returns_none_when_symbol_missing(self, tmp_path):
        provider = MagicMock()
        provider.fetch_option_quotes.return_value = {}  # empty — no quotes
        executor = _make_executor(tmp_path, data_provider=provider)
        assert executor._refresh_limit_price(_make_plan()) is None

    def test_fallback_to_plan_credit_when_refresh_fails(self, tmp_path):
        """When refresh returns None, executor still submits using plan credit."""
        provider = MagicMock()
        provider.fetch_option_quotes.return_value = {}

        executor = _make_executor(tmp_path, dry_run=False, data_provider=provider)

        with patch.object(executor, "_submit_order") as mock_submit:
            mock_submit.return_value = {"status": "submitted", "order_id": "123"}
            executor.execute(_make_verdict(approved=True))
            # submit was still called despite refresh failure
            assert mock_submit.called


# ------------------------------------------------------------------
# Position sizing — unified MAX_RISK_PCT, no silent qty floor
# ------------------------------------------------------------------

def _make_sized_plan(spread_width=5.0, net_credit=0.40):
    """Helper for _calculate_qty tests — only width/credit matter for sizing."""
    plan = _make_plan()
    plan.spread_width = spread_width
    plan.net_credit = net_credit
    return plan


class TestPositionSizing:
    def test_default_max_risk_pct_is_two_percent(self, tmp_path):
        """Constructor default matches the RiskManager guardrail #4 default."""
        executor = _make_executor(tmp_path)
        assert executor.max_risk_pct == 0.02

    def test_qty_uses_configured_max_risk_pct(self, tmp_path):
        """Sizing budget = max_risk_pct × equity, NOT a hardcoded 1%."""
        executor = OrderExecutor(
            api_key="k", secret_key="s",
            trade_plan_dir=str(tmp_path), dry_run=True,
            max_risk_pct=0.02,
        )
        # width=5, credit=0.40 → max_loss_per_contract = $460
        # budget = 0.02 × 100_000 = $2,000 → qty = floor(2000/460) = 4
        plan = _make_sized_plan(spread_width=5.0, net_credit=0.40)
        assert executor._calculate_qty(plan, account_balance=100_000) == 4

    def test_qty_changes_with_max_risk_pct_setting(self, tmp_path):
        """Same plan, different ceilings → proportionally different qty."""
        plan = _make_sized_plan(spread_width=5.0, net_credit=0.40)
        for pct, expected in [(0.01, 2), (0.02, 4), (0.03, 6)]:
            ex = OrderExecutor(
                api_key="k", secret_key="s",
                trade_plan_dir=str(tmp_path), dry_run=True,
                max_risk_pct=pct,
            )
            assert ex._calculate_qty(plan, 100_000) == expected, (
                f"Expected qty={expected} for max_risk_pct={pct}")

    def test_qty_returns_zero_when_one_contract_exceeds_budget(self, tmp_path):
        """No silent floor-to-1: a contract over-budget yields qty=0."""
        executor = OrderExecutor(
            api_key="k", secret_key="s",
            trade_plan_dir=str(tmp_path), dry_run=True,
            max_risk_pct=0.02,
        )
        # width=5, credit=0.10 → max_loss_per_contract = $490
        # budget on tiny account = 0.02 × $20_000 = $400 → 490 > 400 → qty=0
        plan = _make_sized_plan(spread_width=5.0, net_credit=0.10)
        assert executor._calculate_qty(plan, account_balance=20_000) == 0

    def test_qty_returns_zero_for_invalid_inputs(self, tmp_path):
        """Non-positive max_loss_per_contract or equity yields qty=0."""
        executor = _make_executor(tmp_path)
        # net_credit ≥ width → max_loss_per_contract ≤ 0
        bad_plan = _make_sized_plan(spread_width=5.0, net_credit=5.0)
        assert executor._calculate_qty(bad_plan, 100_000) == 0
        # zero equity
        good_plan = _make_sized_plan()
        assert executor._calculate_qty(good_plan, 0.0) == 0

    def test_submit_order_aborts_when_qty_is_zero(self, tmp_path):
        """If sizing returns 0, the order is rejected — never submitted."""
        executor = OrderExecutor(
            api_key="k", secret_key="s",
            trade_plan_dir=str(tmp_path), dry_run=False,
            max_risk_pct=0.02,
        )
        # Plan whose single-contract loss exceeds the sizing budget
        plan = _make_sized_plan(spread_width=5.0, net_credit=0.10)
        verdict = _make_verdict(approved=True, plan=plan)
        verdict.account_balance = 20_000  # tiny equity → qty=0

        with patch("trading_agent.executor.requests.post") as mock_post:
            result = executor.execute(verdict)
            assert mock_post.called is False, (
                "requests.post must NOT be called when qty=0 — "
                "the floor-to-1 bypass is gone")

        assert result["status"] == "rejected"
        assert "qty=0" in result["reason"]
