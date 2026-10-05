"""Backlog §6.1 — fill-realistic pricing (PresetConfig.fill_model, 2026-10-05).

Week 1 on the paper account: 3 of 3 fills at natural (sell at the bid,
buy at the ask); no mid-priced attempt ever filled. Scoring and the profit
target now default to natural; "mid" keeps the legacy behaviour.
"""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from trading_agent.chain_scanner import _quote_credit, _quote_credit_single
from trading_agent.position_monitor import (
    ExitSignal, PositionMonitor, PositionSnapshot, SpreadPosition, remark_positions_at_mid,
)
from trading_agent.strategy_presets import (
    FILL_MODELS, PRESETS, load_active_preset, save_active_preset,
)


# ── credit helpers ────────────────────────────────────────────────────────

def test_spread_credit_natural_vs_mid():
    q = dict(short_bid=1.20, short_ask=1.30, long_bid=0.50, long_ask=0.60)
    assert _quote_credit(**q, model="natural") == 0.60            # 1.20 − 0.60
    assert _quote_credit(**q, model="mid") == 0.68                # 1.25 − 0.55 − 0.02
    assert _quote_credit(**q) == 0.68                             # helper default unchanged


def test_single_credit_natural_is_the_bid():
    assert _quote_credit_single(1.20, 1.30, model="natural") == 1.20
    assert _quote_credit_single(0.0, 1.30, model="natural") == 0.0


def test_natural_never_negative():
    assert _quote_credit(0.10, 0.20, 0.30, 0.40, model="natural") == 0.0


# ── decide() honours the preset ───────────────────────────────────────────

def _chain():
    # Short put 95 (Δ −0.25) and long put 90, priced so natural vs mid differ.
    return [
        {"symbol": "X95P", "strike": 95.0, "delta": -0.25, "bid": 2.00, "ask": 2.10, "type": "put"},
        {"symbol": "X90P", "strike": 90.0, "delta": -0.12, "bid": 0.70, "ask": 0.80, "type": "put"},
    ]


def test_decide_credit_follows_fill_model():
    """Same chain: at mid the spread clears the C/W floor (1.28 / 5 = 0.256
    ≥ |Δ| 0.25); at natural it does not (1.20 / 5 = 0.24, EV < 0) — the
    optimism §6.1 removes."""
    from trading_agent.decision_engine import ChainSlice, DecisionInput, decide
    base = replace(PRESETS["balanced"], delta_grid=(0.25,), dte_grid=(30,),
                   width_grid_pct=(0.05,), min_pop=0.5, edge_buffer=0.0,
                   max_leg_spread_cents=1.0, max_leg_spread_pct_mid=1.0)

    def run(model):
        return decide(DecisionInput(
            side="bull_put", preset=replace(base, fill_model=model),
            chain_slices=[ChainSlice(expiration="2026-11-06", dte=30, contracts=_chain())]))

    mid, natural = run("mid"), run("natural")
    assert [round(c.credit, 2) for c in mid.candidates] == [1.28]
    assert natural.candidates == []
    assert natural.diagnostics.rejects_by_reason == {"cw_below_floor": 1}
    assert natural.diagnostics.best_near_miss["credit"] == 1.20


# ── monitor: profit target at the cost to close ───────────────────────────

def _leg(symbol, qty, avg, bid, ask):
    return PositionSnapshot(symbol=symbol, qty=qty, side="short" if qty < 0 else "long",
                            avg_entry_price=avg, current_price=0.0, market_value=0.0,
                            cost_basis=avg * qty * 100, unrealized_pl=0.0,
                            unrealized_plpc=0.0, asset_class="us_option"), {"bid": bid, "ask": ask}


def test_remark_records_natural_pl():
    (short, q1), (long_, q2) = _leg("S", -1, 2.00, 0.90, 1.10), _leg("L", 1, 1.00, 0.40, 0.60)
    out = remark_positions_at_mid([short, long_], {"S": q1, "L": q2})
    assert out[0].unrealized_pl == pytest.approx((1.00 - 2.00) * -100)       # mid
    assert out[0].natural_unrealized_pl == pytest.approx((1.10 - 2.00) * -100)  # buy back at ask
    assert out[1].natural_unrealized_pl == pytest.approx((0.40 - 1.00) * 100)   # sell at bid


def _spread(mid_pl, natural_pl):
    s = SpreadPosition(underlying="SPY", strategy_name="Bull Put Spread", legs=[],
                       original_credit=1.00, max_loss=400.0, spread_width=5.0,
                       net_unrealized_pl=mid_pl, expiration="2026-11-06",
                       short_strikes=[500.0], contracts_open=1)
    s.net_natural_pl = natural_pl
    return s


@pytest.mark.parametrize("basis,expected", [("natural", ExitSignal.HOLD),
                                            ("mid", ExitSignal.PROFIT_TARGET)])
def test_profit_target_basis(basis, expected):
    """Mid says +$55 (≥ 50 % of $100); buying back at the ask only nets
    +$40 — natural basis holds until the target is really capturable."""
    mon = PositionMonitor("k", "s", post_fill_grace_seconds=0, profit_target_basis=basis)
    sig, _ = mon._check_exit(_spread(55.0, 40.0), {}, underlying_price=520.0)
    assert sig == expected


def test_profit_target_falls_back_to_mid_without_quotes():
    mon = PositionMonitor("k", "s", post_fill_grace_seconds=0, profit_target_basis="natural")
    sig, _ = mon._check_exit(_spread(55.0, None), {}, underlying_price=520.0)
    assert sig == ExitSignal.PROFIT_TARGET


# ── preset wiring ─────────────────────────────────────────────────────────

def test_preset_default_and_summary():
    assert PRESETS["balanced"].fill_model == "natural"
    assert "Fills @ natural" in PRESETS["balanced"].to_summary_line()


def test_preset_overlay_roundtrip_and_validation(tmp_path):
    fp = tmp_path / "STRATEGY_PRESET.json"
    save_active_preset("balanced", fill_model="mid", path=fp)
    assert load_active_preset(fp).fill_model == "mid"
    fp.write_text(json.dumps({"profile": "balanced", "fill_model": "bogus"}))
    assert load_active_preset(fp).fill_model == "natural"          # invalid → default
    with pytest.raises(ValueError):
        save_active_preset("balanced", fill_model="bogus", path=fp)
