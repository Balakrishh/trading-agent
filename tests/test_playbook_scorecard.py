"""Backlog §6.6 — per-playbook scorecard."""
from __future__ import annotations

import json

import pytest

from trading_agent.playbook_scorecard import round_trips, scorecard, scorecard_dict


def opened(ticker, strategy, exp, *, playbook=None, max_loss=100.0, contracts=1,
           run_id="", regime="bullish", ts="2026-10-05T14:00:00+00:00"):
    rs = {"strategy": strategy, "expiration": exp, "max_loss": max_loss,
          "run_id": run_id, "regime": regime}
    if contracts is not None:
        rs["contracts"] = contracts
    if playbook:
        rs["playbook"] = playbook
    return {"timestamp": ts, "ticker": ticker, "action": "submitted", "raw_signal": rs}


def closed(ticker, strategy, exp, pl, *, signal="profit_target", dry=False,
           ts="2026-10-06T14:00:00+00:00"):
    rs = {"strategy": strategy, "expiration": exp, "net_unrealized_pl": pl,
          "exit_signal": signal, "fill_status": "dry_run" if dry else "complete"}
    return {"timestamp": ts, "ticker": ticker, "action": "closed", "raw_signal": rs}


def test_round_trips_pair_fifo_and_tag_playbook():
    rows = [opened("IWM", "Put Debit Spread", "2026-11-06", playbook="put_debit",
                   max_loss=192.0, contracts=4),
            opened("SPY", "Iron Condor", "2026-10-23", contracts=None),       # pre-tag row
            closed("SPY", "Iron Condor", "2026-10-23", -176.0, signal="strike_proximity"),
            closed("IWM", "Put Debit Spread", "2026-11-06", 120.0),
            closed("QQQ", "Bull Put Spread", "2026-11-06", 50.0),             # no open → ignored
            closed("IWM", "Put Debit Spread", "2026-11-06", 9.0, dry=True)]   # dry run ignored
    trips = round_trips(rows)
    assert [(t.ticker, t.playbook, t.realized_pl, t.risk) for t in trips] == [
        ("SPY", "iron_condor", -176.0, None), ("IWM", "put_debit", 120.0, 768.0)]


def _trips(pls, playbook="bull_put", risk=100.0):
    rows = []
    for i, pl in enumerate(pls):
        exp = f"2026-11-{i + 1:02d}"
        rows += [opened("SPY", "Bull Put Spread", exp, playbook=playbook, max_loss=risk),
                 closed("SPY", "Bull Put Spread", exp, pl)]
    return round_trips(rows)


def test_stats_and_collecting_verdict():
    s = scorecard(_trips([40, 40, -60]))[0]
    assert (s.trades, s.wins, s.win_rate) == (3, 2, pytest.approx(0.6667, abs=1e-4))
    assert (s.avg_win, s.avg_loss, s.total_pl, s.expectancy) == (40.0, -60.0, 20.0, 6.67)
    assert s.return_on_risk == pytest.approx(20 / 300, abs=1e-4)
    assert s.verdict == "collecting (3/20 trades)" and s.suggested_risk_pct is None


@pytest.mark.parametrize("pls,verdict,pct", [
    ([10] * 20, "size up", 0.03),               # +10 % return on risk
    ([3] * 20, "keep", 0.02),                   # +3 %
    ([-5] * 20, "shrink", 0.01),                # −5 %
    ([-20] * 20, "disable suggested", 0.0),     # −20 %
])
def test_verdict_after_min_trades(pls, verdict, pct):
    s = scorecard(_trips(pls))[0]
    assert s.verdict.startswith(verdict) and s.suggested_risk_pct == pct


def test_entry_slippage_from_recorded_fills(tmp_path):
    (tmp_path / "trade_plan_SPY.json").write_text(json.dumps({"state_history": [
        {"run_id": "r1", "trade_plan": {"net_credit": 0.48, "estimated_net_credit": 0.49}}]}))
    rows = [opened("SPY", "Iron Condor", "2026-10-23", run_id="r1"),
            closed("SPY", "Iron Condor", "2026-10-23", -176.0)]
    out = scorecard_dict(rows, plan_dir=str(tmp_path))
    assert out["round_trips"] == 1
    assert out["playbooks"][0]["avg_entry_slippage"] == pytest.approx(-0.01)
    assert "advisory" in out["note"]


def test_unknown_contracts_never_drive_sizing():
    """Live journal 2026-10-05: legacy spread rows without a contract count
    produced −704 % return on risk; now risk is unknown → no suggestion."""
    rows = []
    for i in range(20):
        exp = f"2026-12-{i + 1:02d}"
        rows += [opened("SPY", "Bear Call Spread", exp, max_loss=138.0, contracts=None),
                 closed("SPY", "Bear Call Spread", exp, -486.0, signal="hard_stop")]
    s = scorecard(round_trips(rows))[0]
    assert s.return_on_risk is None and s.risk_known_trades == 0
    assert s.suggested_risk_pct is None and "no sizing suggestion" in s.verdict
