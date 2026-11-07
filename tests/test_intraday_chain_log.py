"""2026-11-06 — intraday option-chain log (skill 69): research data only."""
from __future__ import annotations

import gzip
import json
from dataclasses import replace
from datetime import datetime

import pytest

from trading_agent import intraday_chain_log as icl
from trading_agent.calendar_utils import next_sessions
from trading_agent.market_hours import EASTERN
from trading_agent.strategy_presets import PRESETS, load_active_preset

ON = replace(PRESETS["balanced"], intraday_chain_log_mode="on", intraday_chain_log_tickers=("SPY", "QQQ"))
MON_11 = datetime(2026, 11, 9, 11, 0, tzinfo=EASTERN)


class Provider:
    def __init__(self, fail=()):
        self.fail, self.calls = set(fail), []

    def get_cached_price(self, tk):
        return 700.0

    def get_current_price(self, tk):
        return 700.0

    def fetch_option_chain(self, tk, exp, side):
        self.calls.append((tk, exp, side))
        if tk in self.fail:
            raise RuntimeError("schwab down")
        return [{"type": side, "strike": k, "bid": 1.0, "ask": 1.1, "delta": 0.3, "iv": 0.2}
                for k in (600.0, 690.0, 700.0, 710.0, 800.0)]


def _rows(path):
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh]


def test_writes_one_snapshot_per_ticker_with_banded_strikes(tmp_path):
    p = Provider()
    assert icl.maybe_log(ON, p, MON_11, tmp_path) == 2
    rows = _rows(tmp_path / "SPY_2026-11-09.jsonl.gz")
    assert len(rows) == 1 and rows[0]["spot"] == 700.0
    assert sorted(rows[0]["chains"]) == ["2026-11-09", "2026-11-10"]   # 0DTE + 1DTE
    strikes = {r[1] for r in rows[0]["chains"]["2026-11-09"]}
    assert strikes == {690.0, 700.0, 710.0}                           # ±4 % band
    assert {r[0] for r in rows[0]["chains"]["2026-11-09"]} == {"P", "C"}


@pytest.mark.parametrize("preset, when", [
    (PRESETS["balanced"], MON_11),                                     # mode off (default)
    (ON, datetime(2026, 11, 9, 9, 25, tzinfo=EASTERN)),                # before the bell
    (ON, datetime(2026, 11, 9, 16, 0, tzinfo=EASTERN)),                # after the close
])
def test_no_snapshot_when_off_or_outside_session(tmp_path, preset, when):
    p = Provider()
    assert icl.maybe_log(preset, p, when, tmp_path) == 0 and not p.calls


def test_interval_throttle_survives_a_new_process(tmp_path):
    assert icl.maybe_log(ON, Provider(), MON_11, tmp_path) == 2
    p = Provider()                                                     # next cycle, fresh provider
    assert icl.maybe_log(ON, p, MON_11, tmp_path) == 0 and not p.calls
    assert icl.maybe_log(replace(ON, intraday_chain_log_interval_s=0), p, MON_11, tmp_path) == 2


def test_one_ticker_failing_does_not_stop_the_others(tmp_path):
    assert icl.maybe_log(ON, Provider(fail={"SPY"}), MON_11, tmp_path) == 1
    assert not (tmp_path / "SPY_2026-11-09.jsonl.gz").exists()
    assert len(_rows(tmp_path / "QQQ_2026-11-09.jsonl.gz")) == 1


def test_appends_across_cycles(tmp_path):
    fast = replace(ON, intraday_chain_log_interval_s=0)
    icl.maybe_log(fast, Provider(), MON_11, tmp_path)
    icl.maybe_log(fast, Provider(), MON_11, tmp_path)
    assert len(_rows(tmp_path / "SPY_2026-11-09.jsonl.gz")) == 2


def test_next_sessions_skips_weekends_and_holidays():
    assert [d.isoformat() for d in next_sessions(datetime(2026, 11, 25).date(), 2)] == \
        ["2026-11-25", "2026-11-27"]                                   # Thanksgiving 11-26
    assert [d.isoformat() for d in next_sessions(datetime(2026, 11, 7).date(), 1)] == ["2026-11-09"]


def test_preset_loads_validates_and_summarizes(tmp_path):
    fp = tmp_path / "STRATEGY_PRESET.json"
    fp.write_text(json.dumps({"profile": "custom", "custom": {
        "intraday_chain_log_mode": "on", "intraday_chain_log_tickers": ["SPY", "QQQ"]}}))
    p = load_active_preset(fp)
    assert p.intraday_chain_log_tickers == ("SPY", "QQQ")
    assert "Intraday chain log SPY,QQQ ≤1DTE ±4%" in p.to_summary_line()
    assert "Intraday chain log" not in PRESETS["balanced"].to_summary_line()
    fp.write_text(json.dumps({"profile": "custom", "custom": {"intraday_chain_log_mode": "live"}}))
    assert load_active_preset(fp).intraday_chain_log_mode == "off"


def test_agent_hook_swallows_failures(monkeypatch):
    from trading_agent.agent import TradingAgent
    agent = object.__new__(TradingAgent)
    agent.preset, agent.data_provider = ON, Provider()
    monkeypatch.setattr(icl, "maybe_log", lambda *a, **k: 1 / 0)
    agent._maybe_log_intraday_chains()                                  # no exception
