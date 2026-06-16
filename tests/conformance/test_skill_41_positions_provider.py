"""Conformance tests for skill 41 — positions provider.

Pinned behaviors:

- Position invariants (positive qty, valid kind/side, 21-char OCC) — §4
- ManualPositionsProvider round-trip from JSON text — §3.3
- snapshot() returns a fresh copy (defensive copy) — §4
- snapshot() is idempotent — §4
- aggregate_snapshot rolls stock + options correctly — §3 helpers

Quote the same fields and shapes the skill cites; if those change, the
test fails and forces a skill update.
"""

from __future__ import annotations

import json

import pytest

from trading_agent.positions_provider import (
    AlpacaPositionsProvider,
    ManualPositionsProvider,
    PortfolioSnapshot,
    Position,
    PositionsProvider,
    SchwabPositionsProvider,
    aggregate_snapshot,
)


# ---------------------------------------------------------------------------
# §3.1 — Position dataclass invariants
# ---------------------------------------------------------------------------

def test_position_stock_happy_path():
    p = Position(ticker="aapl", qty=100, avg_cost=215.40, kind="stock")
    assert p.ticker == "AAPL"          # uppercased
    assert p.qty == 100
    assert p.kind == "stock"
    assert p.side == "long"            # default
    assert p.occ_symbol == ""


def test_position_option_requires_21_char_occ():
    occ = "MSFT  270115C00400000"     # 21 chars
    assert len(occ) == 21
    p = Position(ticker="MSFT", qty=1, avg_cost=18.50,
                 kind="option", occ_symbol=occ, side="long")
    assert p.occ_symbol == occ


def test_position_qty_must_be_positive():
    # Skill 41 §4: qty is always positive; side carries the sign.
    with pytest.raises(ValueError, match="positive int"):
        Position(ticker="AAPL", qty=0, avg_cost=215.40, kind="stock")
    with pytest.raises(ValueError, match="positive int"):
        Position(ticker="AAPL", qty=-50, avg_cost=215.40, kind="stock")


def test_position_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        Position(ticker="AAPL", qty=100, avg_cost=215.40, kind="bond")


def test_position_rejects_unknown_side():
    with pytest.raises(ValueError, match="side"):
        Position(ticker="MSFT", qty=1, avg_cost=18.50,
                 kind="option", occ_symbol="M" * 21, side="straddle")


def test_position_option_requires_occ_symbol():
    with pytest.raises(ValueError, match="OCC symbol"):
        Position(ticker="MSFT", qty=1, avg_cost=18.50, kind="option")


def test_position_stock_rejects_occ_symbol():
    # Skill 41 §4: stock positions must have empty occ_symbol.
    with pytest.raises(ValueError, match="empty occ_symbol"):
        Position(ticker="AAPL", qty=100, avg_cost=215.40,
                 kind="stock", occ_symbol="A" * 21)


def test_position_stock_rejects_short_side():
    # Skill 41 §4: short-stock positions are out of scope.
    with pytest.raises(ValueError, match="Short-stock"):
        Position(ticker="AAPL", qty=100, avg_cost=215.40,
                 kind="stock", side="short")


# ---------------------------------------------------------------------------
# §3.3 — ManualPositionsProvider
# ---------------------------------------------------------------------------

def test_manual_provider_from_json_text_happy():
    blob = json.dumps([
        {"ticker": "AAPL", "qty": 100, "avg_cost": 215.40, "kind": "stock"},
        {"ticker": "NVDA", "qty": 200, "avg_cost": 132.10, "kind": "stock"},
    ])
    prov = ManualPositionsProvider.from_json_text(blob)
    snap = prov.snapshot()
    assert [p.ticker for p in snap] == ["AAPL", "NVDA"]
    assert prov.source_label == "Manual"


def test_manual_provider_empty_input_returns_empty_list():
    assert ManualPositionsProvider.from_json_text("").snapshot() == []
    assert ManualPositionsProvider.from_json_text("[]").snapshot() == []


def test_manual_provider_rejects_non_array_root():
    with pytest.raises(ValueError, match="array at root"):
        ManualPositionsProvider.from_json_text(json.dumps({"x": 1}))


def test_manual_provider_rejects_malformed_json():
    with pytest.raises(ValueError, match="not parseable"):
        ManualPositionsProvider.from_json_text("not-json-at-all")


def test_manual_provider_includes_row_index_in_error():
    # Malformed row at index 1 should surface its index in the error,
    # so the Streamlit panel can highlight the right line.
    blob = json.dumps([
        {"ticker": "AAPL", "qty": 100, "avg_cost": 215.40, "kind": "stock"},
        {"ticker": "BAD",  "qty": -1,  "avg_cost": 50.0,   "kind": "stock"},
    ])
    with pytest.raises(ValueError, match="Row 1"):
        ManualPositionsProvider.from_json_text(blob)


def test_manual_provider_rejects_fractional_qty():
    blob = json.dumps([
        {"ticker": "AAPL", "qty": 100.5, "avg_cost": 215.40, "kind": "stock"},
    ])
    with pytest.raises(ValueError, match="whole number"):
        ManualPositionsProvider.from_json_text(blob)


def test_manual_provider_qty_as_float_integer_is_accepted():
    # JSON often serialises 100 as 100.0; that should round-trip.
    blob = json.dumps([
        {"ticker": "AAPL", "qty": 100.0, "avg_cost": 215.40, "kind": "stock"},
    ])
    prov = ManualPositionsProvider.from_json_text(blob)
    assert prov.snapshot()[0].qty == 100


# ---------------------------------------------------------------------------
# §4 — Defensive copy + idempotence
# ---------------------------------------------------------------------------

def test_snapshot_returns_defensive_copy():
    p = Position(ticker="AAPL", qty=100, avg_cost=215.40, kind="stock")
    prov = ManualPositionsProvider(positions=[p])
    snap_a = prov.snapshot()
    snap_b = prov.snapshot()
    assert snap_a == snap_b
    assert snap_a is not snap_b           # different list objects
    snap_a.clear()                        # mutating the returned list
    assert len(prov.snapshot()) == 1      # ...does not affect the provider


def test_snapshot_is_idempotent():
    """Skill 41 §4: repeated calls must return equal lists."""
    rows = [
        {"ticker": "AAPL", "qty": 100, "avg_cost": 215.40, "kind": "stock"},
        {"ticker": "NVDA", "qty": 200, "avg_cost": 132.10, "kind": "stock"},
    ]
    prov = ManualPositionsProvider.from_dicts(rows)
    assert prov.snapshot() == prov.snapshot()


# ---------------------------------------------------------------------------
# §3 helpers — aggregate_snapshot
# ---------------------------------------------------------------------------

def test_aggregate_snapshot_rolls_stock_and_options():
    occ = "M" * 21
    positions = [
        Position(ticker="AAPL", qty=100, avg_cost=200.0, kind="stock"),
        Position(ticker="NVDA", qty=200, avg_cost=100.0, kind="stock"),
        Position(ticker="MSFT", qty=2, avg_cost=15.0,
                 kind="option", occ_symbol=occ),
    ]
    sectors = {"AAPL": "Technology", "NVDA": "Technology", "MSFT": "Technology"}
    snap = aggregate_snapshot(positions, sector_for=lambda t: sectors.get(t, "Other"))

    # 100×200 + 200×100 = 40000
    assert snap.total_stock_market_value == pytest.approx(40_000.0)
    # 2 contracts × $15 × 100 multiplier = $3000
    assert snap.total_option_premium_paid == pytest.approx(3_000.0)
    assert snap.by_sector == {"Technology": pytest.approx(40_000.0)}
    assert snap.held_stock_tickers == ["AAPL", "NVDA"]
    assert snap.held_option_tickers == ["MSFT"]
    assert snap.position_count_by_kind == {"stock": 2, "option": 1}


def test_aggregate_snapshot_handles_empty_list():
    snap = aggregate_snapshot([], sector_for=lambda t: "Other")
    assert isinstance(snap, PortfolioSnapshot)
    assert snap.total_stock_market_value == 0.0
    assert snap.by_sector == {}
    assert snap.position_count_by_kind == {"stock": 0, "option": 0}


# ---------------------------------------------------------------------------
# §3.4, §3.5 — Next-session stubs are present but raise on use
# ---------------------------------------------------------------------------

def test_alpaca_stub_raises_not_implemented():
    prov = AlpacaPositionsProvider(api_key="k", secret_key="s", base_url="b")
    assert prov.source_label == "Alpaca (paper)"
    with pytest.raises(NotImplementedError, match="next session"):
        prov.snapshot()


def test_schwab_stub_raises_not_implemented():
    prov = SchwabPositionsProvider(account_id="123", oauth_session=None)
    assert prov.source_label == "Schwab (live)"
    with pytest.raises(NotImplementedError, match="next session"):
        prov.snapshot()


def test_abc_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        PositionsProvider()  # type: ignore[abstract]
