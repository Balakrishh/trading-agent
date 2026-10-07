"""Backlog §6.7 — laddering: ≤ N positions per ticker, later day, expiry gap."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from trading_agent.position_caps import (
    compute_position_cap_dedup_set, ladder_failure, open_expirations,
)
from trading_agent.strategy_presets import PRESETS


OPEN = {"SPY": ["2026-10-30"]}


@pytest.mark.parametrize("ticker,exp,today,reason", [
    ("QQQ", "2026-10-30", set(), None),                                # no open position
    ("SPY", "2026-11-06", set(), None),                                # 7 days apart
    ("SPY", "2026-11-03", set(), "ladder_gap_4d_lt_7d (open 2026-10-30)"),
    ("SPY", "2026-10-30", set(), "ladder_gap_0d_lt_7d (open 2026-10-30)"),
    ("SPY", "2026-11-20", {"SPY"}, "ladder_same_day_entry"),          # opened today
    ("SPY", "", set(), "ladder_expiration_unknown"),
])
def test_ladder_failure(ticker, exp, today, reason):
    assert ladder_failure(ticker, exp, OPEN, today, 7) == reason


def test_ladder_checks_every_open_expiry():
    two = {"SPY": ["2026-10-30", "2026-11-13"]}
    assert ladder_failure("SPY", "2026-11-06", two, set(), 7) is None
    assert ladder_failure("SPY", "2026-11-10", two, set(), 7).startswith("ladder_gap_3d")


def test_open_expirations_and_cap_two():
    mr = {"positions": [{"underlying": "SPY", "expiration": "2026-10-30"},
                        {"underlying": "IWM", "expiration": "2026-11-06"}]}
    assert open_expirations(mr) == {"SPY": ["2026-10-30"], "IWM": ["2026-11-06"]}
    blocked, *_ = compute_position_cap_dedup_set(
        mr, ["SPY", "IWM", "XLF"], sector_for=lambda t: t,
        max_positions_per_ticker=2, max_positions_per_sector=5)
    assert blocked == set()                     # one each → still eligible to ladder


def test_preset_defaults_and_summary():
    p = PRESETS["balanced"]
    assert (p.max_positions_per_ticker, p.ladder_min_gap_days) == (2, 7)
    assert "Ladder 2/ticker ≥7d" in p.to_summary_line()


def test_agent_ladder_block_and_per_ticker_cap():
    from trading_agent.agent import TradingAgent
    a = TradingAgent.__new__(TradingAgent)
    a.preset = PRESETS["balanced"]
    a._open_expirations = {"SPY": ["2026-10-30"]}
    a._opened_today = set()
    assert a._max_per_ticker() == 2
    assert a._ladder_block("SPY", "2026-11-02").startswith("ladder_gap_3d")
    assert a._ladder_block("SPY", "2026-11-13") is None
    a.preset = SimpleNamespace(ladder_min_gap_days=7)        # legacy preset → constant 1
    assert a._max_per_ticker() == 1


def test_opened_today_ignores_non_path_journal():
    """A MagicMock journal_kb must never reach open(): its __index__ (1)
    made open() close stdout and crash pytest -v (CI exit code 3)."""
    from unittest.mock import MagicMock
    from trading_agent.agent import TradingAgent
    import os
    from trading_agent.journal_reader import JournalReader
    a = TradingAgent.__new__(TradingAgent)
    a.journal_kb = MagicMock()
    assert a._tickers_opened_today() == set()
    assert list(JournalReader(MagicMock())._iter_rows()) == []
    os.fstat(1)                                   # stdout still open


# ── 2026-10-06: add to winners only ───────────────────────────────────────

@pytest.mark.parametrize("pls,require,reason", [
    ([-135.0], True, "ladder_existing_losing (open P&L $-135)"),   # GLD: losing → no second
    ([0.0], True, None),                                            # at breakeven: allowed
    ([40.0], True, None),
    ([-135.0], False, None),                                        # rule off: legacy ladder
])
def test_add_to_winners_only(pls, require, reason):
    assert ladder_failure("GLD", "2026-11-20", {"GLD": ["2026-11-06"]}, set(), 7,
                          open_pls={"GLD": pls}, require_profit=require) == reason


def test_open_pls_from_monitor_summary():
    from trading_agent.position_caps import open_pls
    mr = {"positions": [{"underlying": "GLD", "pl": -135.0}, {"underlying": "QQQ", "pl": 91.0}]}
    assert open_pls(mr) == {"GLD": [-135.0], "QQQ": [91.0]}


def test_agent_ladder_block_uses_profit_rule():
    from trading_agent.agent import TradingAgent
    a = TradingAgent.__new__(TradingAgent)
    a.preset = PRESETS["balanced"]
    a._open_expirations, a._opened_today = {"GLD": ["2026-11-06"]}, set()
    a._open_pls = {"GLD": [-160.0]}
    assert a._ladder_block("GLD", "2026-11-20").startswith("ladder_existing_losing")
    a._open_pls = {"GLD": [25.0]}
    assert a._ladder_block("GLD", "2026-11-20") is None
