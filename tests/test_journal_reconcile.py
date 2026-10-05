"""Reconciling a journal open whose close was never recorded (2026-10-05, XLE)."""
from __future__ import annotations

import json

import pytest

from trading_agent import journal_reconcile as jr
from trading_agent.journal_reader import JournalReader
from trading_agent.playbook_scorecard import round_trips


def _write(fp, rows):
    fp.write_text("".join(json.dumps(r) + "\n" for r in rows))


OPEN = {"timestamp": "2026-07-08T15:51:49+00:00", "ticker": "XLE", "action": "submitted",
        "raw_signal": {"strategy": "Iron Condor", "expiration": "2026-07-31",
                       "net_credit": 0.58, "max_loss": 142.0, "order_id": "e9", "run_id": "r1"}}
ARGS = ["--ticker", "XLE", "--strategy", "Iron Condor", "--expiration", "2026-07-31",
        "--reason", "old paper account; history unavailable"]


def test_dry_run_writes_nothing(tmp_path, capsys):
    fp = tmp_path / "signals_live.jsonl"
    _write(fp, [OPEN])
    assert jr.main(ARGS + ["--journal", str(fp)]) == 0
    assert "Dry run" in capsys.readouterr().out
    assert len(fp.read_text().splitlines()) == 1


def test_unknown_pl_reconciles_without_touching_realized_pl(tmp_path):
    fp = tmp_path / "signals_live.jsonl"
    _write(fp, [OPEN])
    assert jr.main(ARGS + ["--journal", str(fp), "--apply"]) == 0
    reader = JournalReader(str(fp))
    assert reader.open_trades() == []                      # no longer "open"
    assert reader.closes_since(400) == []                  # not a close
    rows = [json.loads(l) for l in fp.read_text().splitlines()]
    assert rows[0] == OPEN                                 # append-only
    assert rows[-1]["action"] == "position_reconciled"
    assert rows[-1]["raw_signal"]["pl_known"] is False
    assert round_trips(rows) == []                         # no scorecard trade


def test_known_pl_writes_a_real_close(tmp_path):
    fp = tmp_path / "signals_live.jsonl"
    _write(fp, [OPEN])
    assert jr.main(ARGS + ["--journal", str(fp), "--realized-pl", "58", "--apply"]) == 0
    reader = JournalReader(str(fp))
    assert reader.open_trades() == []
    (close,) = reader.closes_since(400)
    assert close.realized_pl == 58.0 and close.exit_signal == "reconciled"
    assert round_trips([json.loads(l) for l in fp.read_text().splitlines()])[0].realized_pl == 58.0


def test_no_matching_open(tmp_path):
    fp = tmp_path / "signals_live.jsonl"
    _write(fp, [])
    assert jr.main(ARGS + ["--journal", str(fp), "--apply"]) == 1
