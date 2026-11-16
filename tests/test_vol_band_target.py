"""2026-11-15 — vol-band put profit target of its own (research_credit_vol.md §9)."""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from trading_agent import profit_trail as pt
from trading_agent import session_brief as sb
from trading_agent.book_view import merge_rows
from trading_agent.position_monitor import ExitSignal, PositionMonitor, SpreadPosition
from trading_agent.position_snapshot import build_snapshot
from trading_agent.strategy_presets import PRESETS, _coerce_overrides


def _spread(playbook="vol_band_put", pl=20.0):
    return SpreadPosition(underlying="SPY", strategy_name="Bull Put Spread", legs=[], original_credit=0.60,
                          max_loss=140.0, spread_width=2.0, net_unrealized_pl=pl,
                          expiration="2027-06-18", short_strikes=[90.0], playbook=playbook)


def _mon(tp=0.30):
    return PositionMonitor(api_key="x", secret_key="x", post_fill_grace_seconds=0,
                           profit_target_basis="mid", credit_strike_proximity_exit=False,
                           vol_band_profit_target_pct=tp)


def test_profit_basis_uses_the_vol_band_target():
    assert _mon().profit_basis(_spread())[2] == pytest.approx(18.0)            # 30 % of $60
    assert _mon().profit_basis(_spread(playbook=""))[2] == pytest.approx(30.0)  # shared 50 %
    assert _mon(None).profit_basis(_spread())[2] == pytest.approx(30.0)         # unset → shared


def test_exit_fires_at_30_percent_for_vol_band_only():
    sig, why = _mon()._check_exit(_spread(pl=20.0), {}, underlying_price=100.0)
    assert sig == ExitSignal.PROFIT_TARGET and "30% of credit" in why
    assert _mon()._check_exit(_spread(playbook="", pl=20.0), {}, underlying_price=100.0)[0] == ExitSignal.HOLD


def test_trail_closes_at_once_when_target_is_below_the_credit_floor():
    """30 % target < the 40 %-of-credit trail floor → the trail arms and closes on the same cycle."""
    st = pt.TrailState(key="k", ticker="SPY", strategy="Bull Put Spread", kind="credit",
                       expiration="2027-06-18")
    _, decision, _ = pt.evaluate(st, 20.0, basis=60.0, target=18.0, params=pt.TrailParams(), dte=10)
    assert decision == pt.CLOSE_GIVEBACK


def test_playbook_reaches_the_book_and_the_brief():
    snap = build_snapshot([_spread()], {})
    row = merge_rows([], snap)[0]
    assert row["playbook"] == "vol_band_put"
    pre = replace(PRESETS["balanced"], vol_band_put_profit_target_pct=0.30)
    pos = dict(row, credit=0.60, contracts=1, total_max_loss=140.0, mid_pl=10.0)
    assert sb.exit_view(pos, pre, date(2026, 11, 16))["target"] == pytest.approx(18.0)
    assert sb.exit_view(dict(pos, playbook=""), pre, date(2026, 11, 16))["target"] == pytest.approx(30.0)


@pytest.mark.parametrize("value, kept", [(0.3, True), (0.0, False), (1.2, False)])
def test_preset_validation(value, kept):
    assert ("vol_band_put_profit_target_pct" in _coerce_overrides(
        {"vol_band_put_profit_target_pct": value})) is kept


def test_summary_tag():
    pre = replace(PRESETS["balanced"], vol_band_put_enabled=True, vol_band_put_profit_target_pct=0.3)
    assert "TP30%" in pre.to_summary_line()
