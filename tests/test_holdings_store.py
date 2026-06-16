"""Unit tests for trading_agent/holdings_store.py.

Mirrors the test surface for ``watchlist_store`` — atomic write,
schema-versioned read, idempotent clear, round-trip equivalence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_agent.holdings_store import (
    SCHEMA_VERSION,
    HoldingsSnapshot,
    clear_holdings,
    load_holdings,
    save_holdings,
    update_paste,
)


# ---------------------------------------------------------------------------
# load / save round-trip
# ---------------------------------------------------------------------------

def test_load_returns_empty_snapshot_when_file_missing(tmp_path: Path):
    p = tmp_path / "absent.json"
    snap = load_holdings(p)
    assert isinstance(snap, HoldingsSnapshot)
    assert snap.is_empty
    assert snap.raw_paste == ""
    assert snap.parsed_count == 0


def test_save_then_load_round_trips(tmp_path: Path):
    p = tmp_path / "holdings.json"
    snap = HoldingsSnapshot(
        raw_paste='[{"ticker":"AAPL","qty":100,"avg_cost":215.4,"kind":"stock"}]',
        parsed_count=1,
        notes="initial",
    )
    saved = save_holdings(snap, p)
    assert saved.saved_at != ""           # stamped on write
    assert p.exists()
    reloaded = load_holdings(p)
    assert reloaded.raw_paste == snap.raw_paste
    assert reloaded.parsed_count == 1
    assert reloaded.saved_at == saved.saved_at


def test_save_uses_atomic_tmp_then_rename(tmp_path: Path):
    """No leftover .tmp file should remain after a successful save."""
    p = tmp_path / "holdings.json"
    save_holdings(HoldingsSnapshot(raw_paste="[]", parsed_count=0), p)
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_save_creates_parent_dir(tmp_path: Path):
    p = tmp_path / "nested" / "dir" / "holdings.json"
    save_holdings(HoldingsSnapshot(raw_paste="[]", parsed_count=0), p)
    assert p.exists()


# ---------------------------------------------------------------------------
# Schema versioning
# ---------------------------------------------------------------------------

def test_writes_current_schema_version(tmp_path: Path):
    p = tmp_path / "holdings.json"
    save_holdings(HoldingsSnapshot(raw_paste="[]", parsed_count=0), p)
    raw = json.loads(p.read_text())
    assert raw["schema_version"] == SCHEMA_VERSION


def test_unknown_future_version_still_loads_best_effort(
    tmp_path: Path, caplog
):
    p = tmp_path / "holdings.json"
    p.write_text(json.dumps({
        "schema_version": 99,
        "raw_paste": '[{"ticker":"X","qty":1,"avg_cost":1.0,"kind":"stock"}]',
        "saved_at": "2099-01-01T00:00:00Z",
        "parsed_count": 1,
    }))
    snap = load_holdings(p)
    assert snap.raw_paste.startswith("[")
    # File-side schema_version is normalised to current on the snapshot.
    assert snap.schema_version == SCHEMA_VERSION


def test_malformed_json_returns_empty_snapshot(tmp_path: Path):
    p = tmp_path / "holdings.json"
    p.write_text("not json at all")
    snap = load_holdings(p)
    assert snap.is_empty
    assert snap.raw_paste == ""


# ---------------------------------------------------------------------------
# clear / update helpers
# ---------------------------------------------------------------------------

def test_clear_holdings_deletes_file(tmp_path: Path):
    p = tmp_path / "holdings.json"
    save_holdings(HoldingsSnapshot(raw_paste="[]", parsed_count=0), p)
    assert p.exists()
    clear_holdings(p)
    assert not p.exists()


def test_clear_holdings_is_idempotent(tmp_path: Path):
    p = tmp_path / "absent.json"
    clear_holdings(p)   # no-op on a missing file
    clear_holdings(p)   # still no-op


def test_update_paste_persists_and_returns_snapshot(tmp_path: Path):
    p = tmp_path / "holdings.json"
    blob = '[{"ticker":"AAPL","qty":100,"avg_cost":215.4,"kind":"stock"}]'
    saved = update_paste(blob, parsed_count=1, path=p)
    assert saved.raw_paste == blob
    assert saved.parsed_count == 1
    assert saved.saved_at != ""
    reloaded = load_holdings(p)
    assert reloaded.raw_paste == blob
    assert reloaded.parsed_count == 1


def test_update_paste_overwrites_previous(tmp_path: Path):
    p = tmp_path / "holdings.json"
    update_paste('[{"ticker":"A","qty":1,"avg_cost":1.0,"kind":"stock"}]',
                 parsed_count=1, path=p)
    update_paste('[{"ticker":"B","qty":2,"avg_cost":2.0,"kind":"stock"}]',
                 parsed_count=1, path=p)
    snap = load_holdings(p)
    assert '"B"' in snap.raw_paste
    assert '"A"' not in snap.raw_paste


def test_snapshot_is_empty_detects_whitespace(tmp_path: Path):
    snap = HoldingsSnapshot(raw_paste="   \n\t  ")
    assert snap.is_empty
    snap2 = HoldingsSnapshot(raw_paste="[]")
    assert not snap2.is_empty
