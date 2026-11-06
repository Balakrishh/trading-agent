"""2026-11-06 — debit exits judged at mid with a mid-first close ladder,
and the ladder fit for in-budget widths blocked only by the fair-price
check (skill 59 §3/§4)."""
from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from trading_agent import executor as ex
from trading_agent.debit_policy import CALENDAR_STRATEGY, PUT_DEBIT_STRATEGY, build_debit_plan
from trading_agent.decision_engine import ChainSlice, DecisionInput, decide_debit_spread
from trading_agent.position_monitor import (
    ExitSignal, PositionMonitor, PositionSnapshot, SpreadPosition,
)
from trading_agent.strategy_presets import PRESETS

from tests.test_debit_strategies import _chain

# ── debit exits at mid ────────────────────────────────────────────────────
# GLD 371/365 put debit, $2.65 debit: on 2026-10-30 +$210 at mid, +$130 at
# natural; target 50 % of the debit = $132.50.

SHORT, LONG = "GLD261120P00365000", "GLD261120P00371000"


def _leg(sym, qty, entry, mark="mid"):
    return PositionSnapshot(sym, qty, "short" if qty < 0 else "long", entry, entry,
                            0.0, 0.0, 0.0, 0.0, "us_option", mark_source=mark)


def _gld(mid_pl=210.0, natural_pl=130.0, mark="mid", name=PUT_DEBIT_STRATEGY):
    return SpreadPosition(underlying="GLD", strategy_name=name,
                          legs=[_leg(SHORT, -1, 5.90, mark), _leg(LONG, 1, 8.55, mark)],
                          original_credit=-2.65, max_loss=265.0, spread_width=6.0,
                          net_unrealized_pl=mid_pl, expiration="2026-12-18",
                          short_strikes=[365.0], net_natural_pl=natural_pl)


def _mon(price="mid", giveback=0.5):
    return PositionMonitor("k", "s", post_fill_grace_seconds=0,
                           debit_exit_price=price, debit_exit_max_giveback=giveback)


def test_natural_misses_the_target_mid_hits_it():
    assert _mon("natural")._check_debit_exit(_gld(), {})[0] == ExitSignal.HOLD
    sig, why = _mon("mid")._check_debit_exit(_gld(), {})
    assert sig == ExitSignal.PROFIT_TARGET and "(mid)" in why


@pytest.mark.parametrize("spread,price,expected", [
    (_gld(), "mid", 210.0),
    (_gld(), "natural", 130.0),
    (_gld(mark="broker"), "mid", 130.0),           # a last-trade mark is not a mid
    (_gld(name="Bull Put Spread"), "mid", 130.0),  # credit spreads keep their basis
    (_gld(name=CALENDAR_STRATEGY), "mid", 210.0),
])
def test_profit_measure(spread, price, expected):
    assert _mon(price)._profit_pl(spread) == expected


def test_close_floor_gives_back_at_most_the_share():
    assert _mon("mid").debit_close_floor(_gld()) == 105.0
    assert _mon("mid", giveback=0.25).debit_close_floor(_gld()) == 157.5
    assert _mon("natural").debit_close_floor(_gld()) is None
    assert _mon("mid").debit_close_floor(_gld(mid_pl=-20.0)) is None


# ── the close ladder ──────────────────────────────────────────────────────
# Quotes: long 371 10.20/11.00, short 365 5.60/6.20 → close mid −4.70
# (receive 4.70), natural −4.00. Entry credit −2.65: mid profit $205,
# natural $135.

class FakeCloser:
    def __init__(self, fill_at=None):
        self.prices, self.fill_at = [], fill_at

    def submit(self, spread, payload, qty, price):
        self.prices.append(price)
        return {"id": f"c{len(self.prices)}"}

    def await_fill(self, oid, wait_s=None):
        return "filled" if self.fill_at is not None and len(self.prices) == self.fill_at else "canceled"


def _closer(fill_at=None):
    e = ex.OrderExecutor.__new__(ex.OrderExecutor)
    e.debit_ladder_steps = 4
    e.data_provider = SimpleNamespace(fetch_option_quotes=lambda s: {
        LONG: {"bid": 10.20, "ask": 11.00}, SHORT: {"bid": 5.60, "ask": 6.20}})
    b = FakeCloser(fill_at)
    e._submit_close_order = b.submit
    e._await_fill = b.await_fill
    e._order_fill_price = lambda oid: b.prices[-1]
    return e, b


def _closing(floor):
    s = _gld()
    s.exit_signal, s.exit_reason, s.close_min_profit = ExitSignal.PROFIT_TARGET, "tp", floor
    return s


def test_ladder_walks_from_mid_to_natural_and_books_the_fill():
    e, b = _closer(fill_at=3)
    out = e._close_spread_mleg(_closing(102.5))
    assert b.prices == [-4.70, -4.47, -4.23]
    assert out["all_closed"] and out["close_method"] == "mleg_ladder_2"
    assert out["realized_pl"] == pytest.approx(158.0)


def test_ladder_stops_at_the_floor_and_defers():
    e, b = _closer()
    out = e._close_spread_mleg(_closing(150.0))           # worst close −4.15
    assert b.prices[0] == -4.70 and b.prices[-1] == -4.15 and max(b.prices) <= -4.15
    assert out["deferred"] and not out["all_closed"]


def test_floor_above_mid_sends_nothing():
    e, b = _closer()
    out = e._close_spread_mleg(_closing(250.0))
    assert b.prices == [] and out["deferred"]


def test_without_floor_the_usual_improved_then_natural():
    e, b = _closer()
    e._close_spread_mleg(_closing(None))
    assert b.prices == [-4.35, -4.0]


def test_close_spread_returns_the_deferral_without_per_leg_fallback(monkeypatch):
    e, b = _closer()
    monkeypatch.setattr(e, "_close_spread_mleg", lambda s: {"deferred": True, "all_closed": False})
    assert e.close_spread(_closing(150.0))["deferred"]


# ── ladder fit ────────────────────────────────────────────────────────────
# Legs 5¢ either side of fair: every width is >5 % over mid at natural, so
# the grid finds nothing; the ladder fit prices the pair at its cap.

WIDE_CHAIN = _chain("call", half_spread=0.05)
LADDER = replace(PRESETS["balanced"], max_leg_spread_cents=1.0, max_leg_spread_pct_mid=1.0,
                 width_grid_pct=(0.03, 0.05), fit_to_budget_enabled=True,
                 debit_ladder_mode="live", fit_to_budget_ladder=True)


def _decide(budget, preset=LADDER, chain=WIDE_CHAIN):
    return decide_debit_spread(DecisionInput(
        side="call_debit", preset=preset, max_debit=budget,
        chain_slices=[ChainSlice("2026-11-06", 30, chain)]))


def _cap(width):
    q = {c["strike"]: c for c in WIDE_CHAIN}
    mid = (q[100.0]["bid"] + q[100.0]["ask"]) / 2 - (q[100.0 + width]["bid"] + q[100.0 + width]["ask"]) / 2
    return math.floor(round(mid * 1.05 * 100, 6)) / 100


def test_ladder_fit_picks_the_widest_in_budget_width_at_its_cap():
    out = _decide(budget=_cap(3) + 0.01)
    [c] = out.candidates
    assert c.ladder_only and c.width == 3.0 and c.debit == pytest.approx(_cap(3))
    fit = out.diagnostics.fit_to_budget
    assert fit["ladder"] and fit["width"] == 3.0 and fit["mid"] < c.debit


@pytest.mark.parametrize("preset,budget", [
    (replace(LADDER, debit_ladder_mode="shadow"), 5.0),
    (replace(LADDER, fit_to_budget_ladder=False), 5.0),
    (replace(LADDER, fit_to_budget_enabled=False), 5.0),
    (LADDER, None),                                       # no budget
    (LADDER, 0.05),                                       # nothing fits even at the cap
])
def test_no_ladder_fit(preset, budget):
    assert _decide(budget, preset=preset).candidates == []


def test_a_natural_fit_beats_the_ladder():
    out = _decide(budget=5.0, chain=_chain("call"))      # penny-wide legs pass at natural
    assert out.candidates and not any(c.ladder_only for c in out.candidates)


def test_plan_carries_ladder_only():
    c = _decide(budget=_cap(3) + 0.01).candidates[0]
    plan = build_debit_plan(ticker="XYZ", regime="bullish", cand=c, reasoning="r")
    assert plan.ladder_only and plan.max_debit == round(c.max_debit, 2)
    assert plan.to_dict()["ladder_only"] is True


# ── ladder-only submission ────────────────────────────────────────────────

from tests.test_debit_ladder_live import FakeBroker, _executor, _plan  # noqa: E402


def _ladder_only_executor(broker):
    e = _executor(broker)                                 # mid 1.10, natural 1.20
    e._recheck_live_economics = lambda plan, credit, bal: (True, "")
    e._calculate_qty = lambda plan, bal, live_credit=None: 1
    e._append_to_plan = lambda *a: None
    return e


def _lplan(cap=1.15):
    return SimpleNamespace(**vars(_plan()), max_debit=cap, ladder_only=True)


def test_ladder_only_includes_the_cap_and_never_natural():
    b = FakeBroker([("canceled", 0, None)] * 4)
    out = _ladder_only_executor(b)._submit_ladder_only(_lplan(), "p", "run12345", [], 30000.0,
                                                       natural=1.20)
    assert [float(p["limit_price"]) for p in b.submitted] == [1.10, 1.12, 1.13, 1.15]
    assert out["status"] == "rejected" and out["reason"].startswith("ladder_unfilled")


def test_ladder_only_fill_at_the_cap():
    b = FakeBroker([("canceled", 0, None)] * 3 + [("filled", 1, 1.15)])
    out = _ladder_only_executor(b)._submit_ladder_only(_lplan(), "p", "run12345", [], 30000.0,
                                                       natural=1.20)
    assert out["status"] == "submitted" and out["limit_price"] == 1.15 and out["qty"] == 1


def test_ladder_only_rejected_when_the_cap_breaks_the_risk_check():
    b = FakeBroker([])
    e = _ladder_only_executor(b)
    e._recheck_live_economics = lambda plan, credit, bal: (False, "live_debit_risk: over budget")
    out = e._submit_ladder_only(_lplan(), "p", "run12345", [], 30000.0, natural=1.20)
    assert out["status"] == "rejected" and b.submitted == []


# ── preset ────────────────────────────────────────────────────────────────

def test_summary_and_validation():
    line = replace(PRESETS["balanced"], debit_exit_price="mid", debit_ladder_mode="live").to_summary_line()
    assert "Exit@mid (give back ≤50%)" in line and "Fit to budget +ladder" in line
    assert "Exit@mid" not in PRESETS["balanced"].to_summary_line()


def test_invalid_exit_price_falls_back_to_natural(tmp_path):
    import json
    from trading_agent.strategy_presets import load_active_preset
    fp = tmp_path / "STRATEGY_PRESET.json"
    fp.write_text(json.dumps({"profile": "custom", "custom": {"debit_exit_price": "bid"}}))
    assert load_active_preset(fp).debit_exit_price == "natural"
    fp.write_text(json.dumps({"profile": "custom", "custom": {"debit_exit_price": "mid"}}))
    assert load_active_preset(fp).debit_exit_price == "mid"


def test_excursion_keeps_the_natural_peak_beside_the_mid_peak():
    from trading_agent import profit_trail as pt
    st = pt.TrailState(key="k", ticker="GLD", strategy=PUT_DEBIT_STRATEGY, kind="debit_vertical",
                       expiration="2026-11-20")
    pt.track_excursion(st, 210.0, now_iso="t1", natural=130.0)
    pt.track_excursion(st, 150.0, now_iso="t2", natural=140.0)
    pt.track_excursion(st, 120.0, now_iso="t3")
    row = pt.excursion_row(st, 100.0)
    assert row["max_pl"] == 210.0 and row["max_natural_pl"] == 140.0
