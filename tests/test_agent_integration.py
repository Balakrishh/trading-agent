"""
Integration test — runs the full agent pipeline with mocked external APIs.
Verifies that the two-stage cycle (Monitor → Open) wires together correctly.
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from trading_agent.agent import TradingAgent
from trading_agent.config import AppConfig, AlpacaConfig, TradingConfig, LoggingConfig


def _make_config(tmp_path, tickers=None):
    return AppConfig(
        alpaca=AlpacaConfig(
            api_key="INT_TEST_KEY",
            secret_key="INT_TEST_SECRET",
            base_url="https://paper-api.alpaca.markets/v2",
            data_url="https://data.alpaca.markets/v2",
        ),
        trading=TradingConfig(
            tickers=tickers or ["SPY"],
            mode="dry_run",
            max_risk_pct=0.02,
            min_credit_ratio=0.33,
            max_delta=0.20,
            dry_run=True,
            force_market_open=True,   # bypass market-hours check in tests
        ),
        logging=LoggingConfig(
            log_level="DEBUG",
            log_dir=str(tmp_path / "logs"),
            trade_plan_dir=str(tmp_path / "plans"),
        ),
    )


def _mock_agent(agent, prices, option_chain):
    """Apply standard API mocks to an agent instance."""
    dp = agent.data_provider
    dp.fetch_historical_prices = MagicMock(return_value=prices)
    dp.fetch_option_chain = MagicMock(return_value=option_chain)
    dp.get_account_info = MagicMock(return_value={"equity": "100000"})
    dp.is_market_open = MagicMock(return_value=True)
    # get_current_price is separate from fetch_historical_prices after the
    # TTL cache split — mock it explicitly so classify() gets a real float
    dp.get_current_price = MagicMock(
        return_value=float(prices["Close"].iloc[-1]))
    # New methods added for 5-min optimisation
    dp.fetch_batch_snapshots = MagicMock(return_value={"SPY": 500.0})
    dp.prefetch_historical_parallel = MagicMock()
    # Stage 1: no open positions so we skip straight to Stage 2
    agent.position_monitor.fetch_open_positions = MagicMock(return_value=[])


class TestFullPipeline:

    def test_full_cycle_bullish(self, tmp_path, bullish_prices, sample_put_contracts):
        """End-to-end: bullish regime → bull put spread → dry-run plan file."""
        agent = TradingAgent(_make_config(tmp_path))
        _mock_agent(agent, bullish_prices, sample_put_contracts)

        results = agent.run_cycle()

        trades = results["new_trades"]
        assert len(trades) == 1
        r = trades[0]
        assert r["ticker"] == "SPY"
        assert r["regime"] == "bullish"
        assert r["strategy"] == "Bull Put Spread"
        assert r["execution"]["status"] in ("dry_run", "rejected")

        # Plan file persisted using new single-file format
        plan_dir = tmp_path / "plans"
        assert os.path.exists(plan_dir / "trade_plan_SPY.json")
        with open(plan_dir / "trade_plan_SPY.json") as f:
            data = json.load(f)
        assert "state_history" in data
        assert data["ticker"] == "SPY"

    def test_full_cycle_no_account(self, tmp_path):
        """Agent aborts gracefully when account info is unavailable."""
        agent = TradingAgent(_make_config(tmp_path))
        agent.data_provider.get_account_info = MagicMock(return_value=None)

        results = agent.run_cycle()
        # Returns a top-level error dict, not a list
        assert results["status"] == "error"
        assert "account" in results["reason"].lower()

    def test_full_cycle_handles_per_ticker_exception(self, tmp_path, bullish_prices):
        """Agent catches unhandled errors per-ticker without crashing."""
        agent = TradingAgent(_make_config(tmp_path))
        agent.data_provider.get_account_info = MagicMock(return_value={"equity": "100000"})
        agent.data_provider.is_market_open = MagicMock(return_value=True)
        agent.data_provider.fetch_batch_snapshots = MagicMock(return_value={})
        agent.data_provider.prefetch_historical_parallel = MagicMock()
        agent.position_monitor.fetch_open_positions = MagicMock(return_value=[])
        # Make classify throw per ticker
        agent.regime_classifier.classify = MagicMock(
            side_effect=Exception("API timeout"))

        results = agent.run_cycle()
        trades = results["new_trades"]
        assert len(trades) == 1
        assert trades[0]["status"] == "error"
        assert "timeout" in trades[0]["reason"].lower()

    def test_prefetch_called_for_all_tickers(self, tmp_path, bullish_prices,
                                              sample_put_contracts):
        """prefetch_historical_parallel is called with all configured tickers."""
        cfg = _make_config(tmp_path, tickers=["SPY", "QQQ"])
        agent = TradingAgent(cfg)
        _mock_agent(agent, bullish_prices, sample_put_contracts)
        agent.data_provider.fetch_batch_snapshots = MagicMock(return_value={
            "SPY": 500.0, "QQQ": 450.0})

        agent.run_cycle()

        agent.data_provider.prefetch_historical_parallel.assert_called_once_with(
            ["SPY", "QQQ"])

    def test_batch_snapshot_called_for_all_tickers(self, tmp_path, bullish_prices,
                                                    sample_put_contracts):
        """fetch_batch_snapshots is called with all configured tickers."""
        cfg = _make_config(tmp_path, tickers=["SPY", "QQQ"])
        agent = TradingAgent(cfg)
        _mock_agent(agent, bullish_prices, sample_put_contracts)
        agent.data_provider.fetch_batch_snapshots = MagicMock(return_value={
            "SPY": 500.0, "QQQ": 450.0})

        agent.run_cycle()

        agent.data_provider.fetch_batch_snapshots.assert_called_once_with(
            ["SPY", "QQQ"])

    def test_journal_kb_logs_signal_on_dry_run(self, tmp_path, bullish_prices,
                                                sample_put_contracts):
        """JournalKB signals.jsonl is written after each cycle."""
        agent = TradingAgent(_make_config(tmp_path))
        _mock_agent(agent, bullish_prices, sample_put_contracts)

        agent.run_cycle()

        journal_dir = agent.journal_kb.journal_dir
        jsonl_path = os.path.join(journal_dir, "signals.jsonl")
        assert os.path.exists(jsonl_path)
        lines = open(jsonl_path).readlines()
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert record["ticker"] == "SPY"
        assert "action" in record
        assert "raw_signal" in record
