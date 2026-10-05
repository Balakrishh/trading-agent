"""Backlog §1 + §6.2 — market risk state and playbook table (skill 58)."""
from __future__ import annotations

import pandas as pd
import pytest

from trading_agent import market_state as ms
from trading_agent.market_state import (
    BEAR_CALL, BULL_PUT, CAPITULATION, CAUTION, DEFENSIVE, NORMAL, RECOVERY,
    MarketInputs, classify_market_state, compute_inputs, hedge_suggestion,
    playbook_for, read_state, write_state,
)

CALM = dict(spy_price=110.0, spy_sma20=108.0, spy_sma50=105.0, spy_sma200=100.0,
            spy_rsi=55.0, vix=14.0, vix3m=17.0, breadth=0.7)


def st(prior=None, **kw):
    if kw.get("vix") and "vix3m" not in kw:
        kw["vix3m"] = kw["vix"] * 1.2          # contango unless a test sets the term structure
    return classify_market_state(MarketInputs(**{**CALM, **kw}), prior)


def test_normal():
    r = st()
    assert r.state == NORMAL and r.gate.size_multiplier == 1.0 and r.gate.allow_new_csp


@pytest.mark.parametrize("kw", [dict(vix=36.0), dict(vix=30.0, vix3m=26.0),
                                dict(spy_rsi=22.0, vix=29.0)])
def test_capitulation(kw):
    r = st(**kw)
    assert r.state == CAPITULATION and r.gate.size_multiplier == 0.0
    assert not r.gate.allowed_strategies


@pytest.mark.parametrize("kw", [dict(vix=23.0), dict(vix=19.0, vix3m=18.5), dict(breadth=0.3)])
def test_defensive_needs_below_sma200(kw):
    assert st(spy_price=95.0, spy_sma50=105.0, **kw).state == DEFENSIVE
    r = st(spy_price=95.0, spy_sma50=105.0, **kw)
    assert r.gate.allowed_strategies == {BEAR_CALL, ms.PUT_DEBIT} and not r.gate.allow_new_csp


@pytest.mark.parametrize("kw", [dict(spy_price=104.0), dict(vix=21.0),
                                dict(vix=16.0, vix3m=16.5), dict(breadth=0.45)])
def test_caution(kw):
    r = st(**kw)
    assert r.state == CAUTION and r.gate.size_multiplier == 0.5
    assert BULL_PUT not in r.gate.allowed_strategies and not r.gate.allow_new_csp


def test_missing_spy_fails_safe():
    r = classify_market_state(MarketInputs(vix=12.0))
    assert r.state == CAUTION and "spy_data_unavailable" in r.reasons[0]


def test_missing_optional_inputs_are_skipped_not_zero():
    r = st(vix=None, vix3m=None, breadth=None)
    assert r.state == NORMAL and r.inputs.term_ratio is None


def test_recovery_after_defensive():
    # SPY back above SMA-20 but below SMA-50 → RECOVERY (not CAUTION).
    r = st(prior=DEFENSIVE, spy_price=104.0, spy_sma20=102.0, vix=19.0)
    assert r.state == RECOVERY and r.gate.allow_new_csp and BULL_PUT in r.gate.allowed_strategies


def test_recovery_exits_to_normal():
    assert st(prior=RECOVERY, vix=18.0).state == NORMAL


def test_recovery_holds_while_vix_elevated():
    assert st(prior=RECOVERY, vix=20.5).state == RECOVERY


def test_no_recovery_without_prior_stress():
    assert st(prior=NORMAL, spy_price=104.0, spy_sma20=102.0).state == CAUTION


@pytest.mark.parametrize("regime,vol,rsi,name,impl", [
    ("bullish", 45, 55, "bull_put", True),
    ("bullish", 20, 55, "call_debit", True),
    ("bearish", 60, 45, "bear_call", True),
    ("bearish", 20, 45, "put_debit", True),
    ("bearish", 40, 25, "bounce_bull_put", True),
    ("bearish", 20, 25, "wait_for_stabilization", False),
    ("sideways", 35, 50, "iron_condor", True),
    ("sideways", 10, 50, "calendar", True),
    ("mean_reversion", 10, 80, "mean_reversion", True),
    ("bullish", None, None, "call_debit", True),
])
def test_playbook(regime, vol, rsi, name, impl):
    pb = playbook_for(regime, vol, rsi)
    assert (pb.name, pb.implemented) == (name, impl)


def test_compute_inputs_per_input_isolation():
    up = pd.DataFrame({"Close": [100.0 + i * 0.1 for i in range(250)]})
    down = pd.DataFrame({"Close": [200.0 - i * 0.1 for i in range(250)]})

    def hist(t):
        if t == "BAD":
            raise RuntimeError("rpc")
        return down if t == "DN" else up

    def level(sym):
        if sym == "^VIX3M":
            raise RuntimeError("rpc")
        return 15.0

    inp = compute_inputs(hist, ["SPY", "DN", "BAD"], level)
    assert inp.spy_price == pytest.approx(124.9)
    assert inp.spy_sma200 is not None and inp.spy_rsi == 100.0
    assert inp.vix == 15.0 and inp.vix3m is None
    assert inp.breadth == 0.5 and inp.breadth_n == 2


def test_state_roundtrip(tmp_path):
    fp = tmp_path / "market_state.json"
    write_state(st(vix=21.0), fp)
    d = read_state(fp)
    assert d["state"] == CAUTION and d["gate"]["size_multiplier"] == 0.5
    assert d["age_seconds"] >= 0 and not fp.with_suffix(".json.tmp").exists()
    assert read_state(tmp_path / "missing.json") is None


def test_hedge_suggestion():
    h = hedge_suggestion(30_000, 600.0)
    assert h["contracts"] == 1 and h["long_strike"] == 582 and h["short_strike"] == 540
    assert hedge_suggestion(0, 600.0) is None


def test_gate_failure_and_mean_reversion_direction():
    from trading_agent.market_state import IRON_CONDOR, MEAN_REVERSION, gate_failure
    caution = st(vix=21.0)
    assert gate_failure(None, BULL_PUT) is None
    assert gate_failure(caution, IRON_CONDOR) is None
    assert gate_failure(caution, BULL_PUT) == "market_state_CAUTION_blocks_bull_put_spread"
    assert gate_failure(caution, MEAN_REVERSION, ["put"]) == "market_state_CAUTION_blocks_bull_put_spread"
    assert gate_failure(caution, MEAN_REVERSION, ["call"]) is None
    assert gate_failure(st(vix=40.0), BEAR_CALL) == "market_state_CAPITULATION_blocks_bear_call_spread"


def test_breadth_universe_excludes_spy_and_non_equity():
    assert ms.breadth_universe(["SPY", "QQQ", "TLT", "GLD", "XLF", "GDX"]) == ["QQQ", "XLF"]


def test_caution_hysteresis():
    # VIX 19.5 clears the 20 entry level but not the 19 exit level.
    assert st(vix=19.5).state == NORMAL
    r = st(prior=CAUTION, vix=19.5)
    assert r.state == CAUTION and r.reasons[0].startswith("hysteresis")
    assert st(prior=CAUTION, spy_price=105.5).state == CAUTION      # < 1 % above SMA-50
    assert st(prior=CAUTION, vix=15.0).state == NORMAL


def test_skill_59_names_match_debit_policy():
    from trading_agent import debit_policy as dp
    assert (ms.CALL_DEBIT, ms.PUT_DEBIT, ms.CALENDAR, ms.BOUNCE_BULL_PUT) == (
        dp.CALL_DEBIT_STRATEGY, dp.PUT_DEBIT_STRATEGY, dp.CALENDAR_STRATEGY,
        dp.BOUNCE_BULL_PUT_STRATEGY)


@pytest.mark.parametrize("state,allowed,blocked", [
    (CAUTION, ["Put Debit Spread", "Calendar Spread", "Bounce Bull Put Spread"],
     ["Call Debit Spread"]),
    (DEFENSIVE, ["Put Debit Spread"], ["Calendar Spread", "Bounce Bull Put Spread"]),
    (RECOVERY, ["Call Debit Spread", "Calendar Spread"], ["Put Debit Spread"]),
])
def test_skill_59_gates(state, allowed, blocked):
    g = ms.GATES[state]
    assert all(a in g.allowed_strategies for a in allowed)
    assert not any(b in g.allowed_strategies for b in blocked)
