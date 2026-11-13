"""2026-11-12 — vol-band bull put allowed in CAUTION above SMA-200 (research_credit_vol.md §8)."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from trading_agent import market_state as ms
from trading_agent import session_brief as sb
from trading_agent.agent import TradingAgent
from trading_agent.strategy_presets import PRESETS

CAUTION_GATE = ms.StateGate(size_multiplier=0.5, allowed_strategies=frozenset({"Put Debit Spread"}),
                            allow_new_csp=False)


def _state(name="CAUTION"):
    return ms.MarketStateResult(state=name, reasons=("SPY < SMA-50",), inputs=ms.MarketInputs(),
                                gate=CAUTION_GATE)


def _me(on=True, state=None):
    return SimpleNamespace(preset=replace(PRESETS["balanced"], vol_band_put_in_caution=on),
                           _market_state=state or _state())


def _plan(playbook="vol_band_put"):
    return SimpleNamespace(ticker="SPY", strategy_name="Bull Put Spread", playbook=playbook,
                           legs=[SimpleNamespace(option_type="put", action="sell"),
                                 SimpleNamespace(option_type="put", action="buy")])


def _an(price, sma200):
    return SimpleNamespace(current_price=price, sma_200=sma200)


@pytest.mark.parametrize("on, playbook, price, state, blocked", [
    (True, "vol_band_put", 757.0, "CAUTION", False),     # above SMA-200: allowed
    (True, "vol_band_put", 720.0, "CAUTION", True),      # below SMA-200: still blocked
    (True, "vol_band_put", 0.0, "CAUTION", True),        # no price: fail closed
    (False, "vol_band_put", 757.0, "CAUTION", True),     # flag off
    (True, "", 757.0, "CAUTION", True),                  # an ordinary bull put
    (True, "vol_band_put", 757.0, "RISK_OFF", True),     # only CAUTION is relaxed
])
def test_caution_exception(on, playbook, price, state, blocked):
    got = TradingAgent._market_gate_block(_me(on, _state(state)), _plan(playbook), _an(price, 731.6))
    assert (got is not None) is blocked
    if blocked:
        assert got.startswith(f"market_state_{state}_blocks_bull_put_spread")


def test_allowed_strategy_is_untouched():
    plan = _plan()
    plan.strategy_name = "Put Debit Spread"
    assert TradingAgent._market_gate_block(_me(False), plan, _an(700.0, 731.6)) is None


def test_no_market_state_allows_everything():
    me = _me(False)
    me._market_state = None
    assert TradingAgent._market_gate_block(me, _plan(""), _an(700.0, 731.6)) is None


def test_summary_and_brief_show_the_exception():
    pre = replace(PRESETS["balanced"], vol_band_put_enabled=True, vol_band_put_in_caution=True)
    assert "CAUTION>SMA200" in pre.to_summary_line()
    ctx = dict(today=__import__("datetime").date(2026, 11, 12), preset=pre,
               market_state={"state": "CAUTION", "reasons": [], "gate": {
                   "size_multiplier": 0.5, "allow_new_csp": False,
                   "allowed_strategies": ["Put Debit Spread"]}})
    assert "only above SMA-200, half size" in sb.build_premarket(ctx)


def test_preset_override_round_trips():
    from trading_agent.strategy_presets import _coerce_overrides
    assert _coerce_overrides({"vol_band_put_in_caution": True}).get("vol_band_put_in_caution") is True
