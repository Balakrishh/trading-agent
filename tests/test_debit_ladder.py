"""Backlog C1 — mid-first debit ladder, shadow mode (skill 59 §3)."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from trading_agent import debit_ladder as dl
from trading_agent.agent import TradingAgent
from trading_agent.decision_engine import ChainSlice, DecisionInput, decide_debit_spread
from trading_agent.strategy_presets import PRESETS, PresetConfig, _coerce_overrides

T0 = datetime(2026, 10, 16, 10, 0, tzinfo=timezone.utc)
P = dl.LadderParams(steps=4, step_cycles=1)


# ── prices + outcome ──────────────────────────────────────────────────

def test_gld_2026_10_15_ladder_prices():
    """GLD put debit: mid 3.975, cap 4.1738 (natural 4.25 was rejected)."""
    assert dl.ladder_prices(3.975, 4.1738, 4) == [3.98, 4.04, 4.11, 4.17]


def test_ladder_collapses_to_the_cap_when_mid_and_cap_share_a_cent():
    assert dl.ladder_prices(0.452, 0.4589, 4) == [0.45]
    assert dl.ladder_prices(1.00, 1.20, 1) == [1.20]


def test_quote_debit_natural_and_mid():
    legs = [{"action": "buy", "bid": 4.00, "ask": 4.20}, {"action": "sell", "bid": 0.10, "ask": 0.20}]
    assert dl.quote_debit(legs, 10.0) == {"natural": 4.10, "mid": 3.95}
    assert dl.quote_debit([{"action": "sell", "bid": 1, "ask": 1.1}], 5.0) is None   # a credit


@pytest.mark.parametrize("naturals,step_cycles,expect", [
    ([4.25, 4.20, 4.10], 1, {"filled": True, "step": 2, "price": 4.11}),
    ([4.25, 4.25, 4.25, 4.25], 1, {"filled": False, "step": None, "price": None}),
    ([4.25, 4.25], 1, None),                                     # still watching
    ([4.25, None, 4.03], 1, {"filled": True, "step": 2, "price": 4.11}),
    ([4.25, 4.00], 2, None),                                     # step 0 full (no fill), step 1 not observed yet
])
def test_ladder_outcome(naturals, step_cycles, expect):
    assert dl.ladder_outcome([3.98, 4.04, 4.11, 4.17], naturals, step_cycles) == expect


def test_two_cycles_per_step():
    prices = [3.98, 4.04, 4.11, 4.17]
    assert dl.ladder_outcome(prices, [4.25, 4.20, 4.04], 2) == {"filled": True, "step": 1, "price": 4.04}


# ── tracker ───────────────────────────────────────────────────────────

def _leg(symbol, action, bid, ask):
    return SimpleNamespace(symbol=symbol, action=action, bid=bid, ask=ask)


def _plan(strategy="Put Debit Spread"):
    return SimpleNamespace(strategy_name=strategy, expiration="2026-11-13", spread_width=10.0,
                           max_debit=4.1738, valid=True,
                           legs=[_leg("GLD_L", "buy", 3.95, 4.35), _leg("GLD_S", "sell", 0.10, 0.25)])  # mid 3.975, natural 4.25


def _quotes(long_ask, short_bid):
    return lambda syms: {"GLD_L": {"bid": long_ask - 0.2, "ask": long_ask},
                         "GLD_S": {"bid": short_bid, "ask": short_bid + 0.1}}


def test_submitted_ladder_saving(tmp_path):
    fp = tmp_path / "l.json"
    lad = dl.LadderShadow(fp, now=T0)
    assert lad.start_submitted("GLD", _plan(), qty=2, actual_debit=4.25, params=P)
    assert lad.advance(_quotes(4.35, 0.10), P) == []             # started this cycle
    lad.save()
    rows = []
    for i, (ask, bid) in enumerate([(4.35, 0.10), (4.20, 0.12), (4.15, 0.11)], start=1):
        lad = dl.LadderShadow(fp, now=T0 + timedelta(minutes=i))
        rows += lad.advance(_quotes(ask, bid), P)
        lad.save()
    assert len(rows) == 1
    r = rows[0]
    assert r["kind"] == "submitted" and r["ticker"] == "GLD" and r["filled"]
    assert r["fill_step"] == 2 and r["ladder_debit"] == 4.11      # natural 4.08 ≤ 4.11
    assert r["saving_usd"] == pytest.approx((4.25 - 4.11) * 100 * 2)
    assert r["missed_trade"] is False


def test_submitted_ladder_missed(tmp_path):
    lad = dl.LadderShadow(tmp_path / "l.json", now=T0)
    lad.start_submitted("GLD", _plan(), 1, 4.25, P)
    rows = []
    for i in range(1, 5):
        lad.now = T0 + timedelta(minutes=i)
        rows += lad.advance(_quotes(4.40, 0.10), P)
    assert rows[0]["filled"] is False and rows[0]["missed_trade"] is True
    assert rows[0]["saving_usd"] is None


def test_unlocked_near_miss_would_open_and_is_tracked_once_a_day(tmp_path):
    cand = {"expiration": "2026-11-13", "width": 10.0, "long_symbol": "GLD_L",
            "short_symbol": "GLD_S", "natural": 4.25, "mid": 3.975, "cap": 4.1738,
            "reward_risk_at_cap": 1.396}
    lad = dl.LadderShadow(tmp_path / "l.json", now=T0)
    assert lad.start_unlocked("GLD", "Put Debit Spread", cand, P)
    assert not lad.start_unlocked("GLD", "Put Debit Spread", cand, P)       # already active
    lad.now = T0 + timedelta(minutes=1)
    rows = lad.advance(_quotes(4.05, 0.10), P)                               # natural 3.95
    assert rows[0]["would_open"] is True and rows[0]["ladder_debit"] == 4.04
    assert not lad.start_unlocked("GLD", "Put Debit Spread", cand, P)       # done today
    lad.save()
    tomorrow = dl.LadderShadow(tmp_path / "l.json", now=T0 + timedelta(days=1))
    assert tomorrow.start_unlocked("GLD", "Put Debit Spread", cand, P)


def test_non_debit_plan_and_stale_tracker(tmp_path):
    lad = dl.LadderShadow(tmp_path / "l.json", now=T0)
    assert not lad.start_submitted("SPY", _plan("Bull Put Spread"), 1, None, P)
    lad.start_submitted("GLD", _plan(), 1, None, P)
    lad.now = T0 + timedelta(seconds=dl.MAX_TRACK_SECONDS + 1)
    assert lad.advance(lambda s: {}, P) == [] and lad.trackers == {}


def test_missing_quotes_count_as_no_fill(tmp_path):
    lad = dl.LadderShadow(tmp_path / "l.json", now=T0)
    lad.start_submitted("GLD", _plan(), 1, 4.25, P)
    for i in range(1, 3):
        lad.now = T0 + timedelta(minutes=i)
        assert lad.advance(lambda s: {}, P) == []
    assert list(lad.trackers.values())[0]["naturals"][1:] == [None, None]
    lad.now = T0 + timedelta(minutes=3)
    assert lad.advance(lambda s: {}, P)[0]["filled"] is False


# ── scanner: ladder candidate ─────────────────────────────────────────

def _chain():
    import math
    from trading_agent.decision_engine import _bs_price
    rows = []
    for k in range(90, 111):
        price = _bs_price(100.0, k, 30 / 365, 0.2, "put")
        d1 = (math.log(100.0 / k) + 0.02 * 30 / 365) / (0.2 * math.sqrt(30 / 365))
        delta = 0.5 * (1 + math.erf(d1 / math.sqrt(2))) - 1.0
        half = 0.20 if k == 100 else 0.01          # the long leg's quote is wide
        rows.append({"symbol": f"P{k}", "strike": float(k), "delta": round(delta, 4),
                     "bid": round(max(0.01, price - half), 2), "ask": round(price + half, 2),
                     "type": "put"})
    return rows


@pytest.mark.parametrize("mode,expect", [("shadow", True), ("off", False)])
def test_scanner_offers_ladder_candidate(mode, expect):
    pre = replace(PRESETS["balanced"], max_leg_spread_cents=1.0, max_leg_spread_pct_mid=1.0,
                  width_grid_pct=(0.05,), debit_ladder_mode=mode)
    out = decide_debit_spread(DecisionInput(side="put_debit", preset=pre,
                                            chain_slices=[ChainSlice("2026-11-13", 30, _chain())]))
    assert out.candidates == []
    lc = out.diagnostics.ladder_candidate
    assert (lc is not None) is expect
    if expect:
        assert (lc["long_symbol"], lc["short_symbol"]) == ("P100", "P95")
        assert lc["strategy"] == "Put Debit Spread"
        assert lc["natural"] > lc["cap"] > lc["mid"]
        assert lc["reward_risk_at_cap"] >= pre.debit_min_reward_risk
        assert out.diagnostics.to_journal_dict()["ladder_candidate"] == lc


def test_scanner_skips_candidate_failing_rr_at_cap():
    pre = replace(PRESETS["balanced"], max_leg_spread_cents=1.0, max_leg_spread_pct_mid=1.0,
                  width_grid_pct=(0.05,), debit_min_reward_risk=3.0)
    out = decide_debit_spread(DecisionInput(side="put_debit", preset=pre,
                                            chain_slices=[ChainSlice("2026-11-13", 30, _chain())]))
    assert out.diagnostics.ladder_candidate is None


# ── preset ────────────────────────────────────────────────────────────

def test_preset_defaults_summary_and_modes():
    f = PresetConfig.__dataclass_fields__
    assert (f["debit_ladder_mode"].default, f["debit_ladder_steps"].default,
            f["debit_ladder_step_cycles"].default) == ("shadow", 4, 1)
    assert "Ladder shadow 4×1c" in PRESETS["balanced"].to_summary_line()
    assert _coerce_overrides({"debit_ladder_mode": "live"})["debit_ladder_mode"] == "live"
    assert "debit_ladder_mode" not in _coerce_overrides({"debit_ladder_mode": "turbo"})


# ── agent hooks ───────────────────────────────────────────────────────

def _agent(tmp_path):
    a = TradingAgent.__new__(TradingAgent)
    a.preset = PRESETS["balanced"]
    a._ladder = dl.LadderShadow(tmp_path / "l.json", now=T0)
    a.journal_kb = SimpleNamespace(rows=[])
    a.journal_kb.log_signal = lambda **kw: a.journal_kb.rows.append(kw)
    return a


def test_agent_starts_submitted_and_unlocked(tmp_path):
    a = _agent(tmp_path)
    a._debit_ladder_observe("GLD", _plan(), {"status": "submitted", "qty": 1, "limit_price": 4.25}, None)
    cand = {"expiration": "2026-11-13", "width": 1.5, "long_symbol": "XLU_L", "short_symbol": "XLU_S",
            "natural": 0.57, "mid": 0.45, "cap": 0.4725, "reward_risk_at_cap": 2.17}
    a._debit_ladder_observe("XLU", SimpleNamespace(valid=False, strategy_name="Put Debit Spread"),
                            {"status": "rejected"}, {"diagnostics": {"ladder_candidate": cand}})
    kinds = sorted(t["kind"] for t in a._ladder.trackers.values())
    assert kinds == ["submitted", "unlocked"]


def test_unlocked_tracker_named_after_the_debit_spread(tmp_path):
    """2026-10-16: a bear-call plan's put-debit fallback was journaled as
    "Bear Call Spread"; the candidate's own strategy wins."""
    a = _agent(tmp_path)
    cand = {"strategy": "Put Debit Spread", "expiration": "2026-11-20", "width": 2.0,
            "long_symbol": "GDX_L", "short_symbol": "GDX_S",
            "natural": 1.10, "mid": 0.95, "cap": 0.9975, "reward_risk_at_cap": 1.005}
    a._debit_ladder_observe("GDX", SimpleNamespace(valid=False, strategy_name="Bear Call Spread"),
                            {"status": "rejected"}, {"diagnostics": {"ladder_candidate": cand}})
    assert [t["strategy"] for t in a._ladder.trackers.values()] == ["Put Debit Spread"]


def test_agent_advance_journals_rows(tmp_path):
    a = _agent(tmp_path)
    a._ladder.start_submitted("GLD", _plan(), 1, 4.25, dl.LadderParams())
    a._ladder.now = T0 + timedelta(minutes=1)
    a.data_provider = SimpleNamespace(fetch_option_quotes=_quotes(4.05, 0.10))
    a._advance_debit_ladder_shadow()
    row = a.journal_kb.rows[0]
    assert row["action"] == "debit_ladder_shadow" and row["ticker"] == "GLD"
    assert row["raw_signal"]["ladder_debit"] == 4.04
    assert json.loads((tmp_path / "l.json").read_text())["done_today"]


def test_agent_off_mode_does_nothing(tmp_path):
    a = _agent(tmp_path)
    a._ladder = None
    a._debit_ladder_observe("GLD", _plan(), {"status": "submitted"}, None)
    a._advance_debit_ladder_shadow()
