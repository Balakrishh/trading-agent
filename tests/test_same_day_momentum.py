"""2026-11-24 — same-day momentum debit playbook (skill 72, research_credit_vol.md §11)."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from trading_agent import same_day_momentum as sdm
from trading_agent.agent import TradingAgent
from trading_agent.executor import calculate_position_qty
from trading_agent.journal_reader import ClosedTrade
from trading_agent.position_monitor import ExitSignal, PositionMonitor, SpreadPosition
from trading_agent.regime import Regime
from trading_agent.risk_manager import RiskManager
from trading_agent.strategy_presets import PRESETS, _coerce_overrides

PRE = replace(PRESETS["balanced"], same_day_mode="live")


def _chain(opt, spot=720.0, half=0.01):
    """A synthetic 0DTE chain: intrinsic + a little time value, 1c wide."""
    out = []
    for k in range(700, 741):
        intr = max(0.0, (spot - k) if opt == "call" else (k - spot))
        mid = intr + max(0.05, 2.0 - 0.15 * abs(spot - k))
        d = 0.5 + (spot - k) / 40 if opt == "call" else -(0.5 + (k - spot) / 40)
        out.append({"symbol": f"SPY261124{opt[0].upper()}{k}", "strike": float(k),
                    "bid": round(mid - half, 2), "ask": round(mid + half, 2),
                    "delta": max(-0.99, min(0.99, d))})
    return out


# ── pure helpers ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("hhmm, used, want", [
    ("09:59", [], None), ("10:00", [], "10:00"), ("10:19", [], "10:00"), ("10:20", [], None),
    ("10:05", ["10:00"], None), ("12:40", [], "12:30"), ("14:10", ["10:00"], "14:00"),
])
def test_due_slot(hhmm, used, want):
    now = datetime(2026, 11, 24, *map(int, hhmm.split(":")))
    assert sdm.due_slot(now, PRE.same_day_entry_times_et, 20, used) == want


def test_direction():
    assert sdm.direction(721.0, 718.0) == "call"
    assert sdm.direction(715.0, 718.0) == "put"
    assert sdm.direction(718.0, 718.0) is None
    assert sdm.direction(718.0, None) is None


def test_implied_spot_reads_parity_not_the_last_price():
    """2026-11-23 09:32: the quote's last price was Friday's 689.81 while the
    options priced ~721 — parity on the option quotes gives the live level."""
    got = sdm.implied_spot(_chain("call", 721.0), _chain("put", 721.0))
    assert got == pytest.approx(721.0, abs=0.05)
    assert sdm.implied_spot([], _chain("put")) is None


def test_build_plan_call_and_put():
    p = sdm.build_plan(ticker="SPY", expiration="2026-11-24", side="call", spot=720.2,
                       contracts=_chain("call"), width=5.0)
    assert p.valid and p.playbook == sdm.PLAYBOOK and p.strategy_name == "Call Debit Spread"
    assert [l.strike for l in p.legs] == [720.0, 725.0] and [l.action for l in p.legs] == ["buy", "sell"]
    assert p.net_credit < 0 and p.max_debit == pytest.approx(-p.net_credit)
    assert p.max_loss == pytest.approx(-p.net_credit * 100)
    q = sdm.build_plan(ticker="SPY", expiration="2026-11-24", side="put", spot=719.8,
                       contracts=_chain("put"), width=5.0)
    assert q.valid and [l.strike for l in q.legs] == [720.0, 715.0] and q.strategy_name == "Put Debit Spread"


@pytest.mark.parametrize("contracts, why", [
    ([], "no 0DTE chain"),
    (_chain("call", half=0.40), "leg_spread_wide"),
])
def test_build_plan_rejects(contracts, why):
    p = sdm.build_plan(ticker="SPY", expiration="2026-11-24", side="call", spot=720.0,
                       contracts=contracts, width=5.0)
    assert not p.valid and why in p.rejection_reason and p.playbook == sdm.PLAYBOOK


def test_risk_pct_sizes_to_one_contract():
    p = sdm.build_plan(ticker="SPY", expiration="2026-11-24", side="call", spot=720.0,
                       contracts=_chain("call"), width=5.0)
    pct = sdm.trade_risk_pct(p, 28533.0, 0.01, 1)
    assert pct <= 0.01
    assert calculate_position_qty(p, 28533.0, pct) == 1
    cheap = replace(p, net_credit=-0.80, max_loss=80.0, max_debit=0.80)
    assert calculate_position_qty(cheap, 28533.0, sdm.trade_risk_pct(cheap, 28533.0, 0.01, 1)) == 1
    assert sdm.trade_risk_pct(p, 0.0, 0.01, 1) == 0.0


def test_prior_vix_close():
    hist = {date(2026, 11, 20): 35.9, date(2026, 11, 23): 26.6, date(2026, 11, 24): 28.9}
    assert sdm.prior_vix_close(date(2026, 11, 24), lambda: hist) == pytest.approx(26.6)
    assert sdm.prior_vix_close(date(2026, 11, 20), lambda: hist) is None

    def boom():
        raise RuntimeError("no network")
    assert sdm.prior_vix_close(date(2026, 11, 24), boom) is None


def test_state_round_trip_and_day_reset(tmp_path):
    fp = tmp_path / "s.json"
    st = sdm.load_state(date(2026, 11, 24), fp)
    st.open_ref, st.slots_used = 718.0, ["10:00"]
    sdm.save_state(st, fp)
    assert not (tmp_path / "s.json.tmp").exists()
    again = sdm.load_state(date(2026, 11, 24), fp)
    assert again.open_ref == 718.0 and again.slots_used == ["10:00"]
    assert sdm.load_state(date(2026, 11, 25), fp).slots_used == []          # new day, fresh
    fp.write_text("{not json")
    assert sdm.load_state(date(2026, 11, 24), fp).open_ref is None          # unreadable, fresh


# ── exits ──────────────────────────────────────────────────────────────────

def _pos(pl, playbook=sdm.PLAYBOOK, strategy="Call Debit Spread"):
    return SpreadPosition(underlying="SPY", strategy_name=strategy, legs=[], original_credit=-2.50,
                          max_loss=250.0, spread_width=5.0, net_unrealized_pl=pl,
                          expiration="2099-01-01", playbook=playbook)


def _mon(close_et="23:59"):
    return PositionMonitor(api_key="x", secret_key="x", post_fill_grace_seconds=0,
                           debit_exit_price="mid", same_day_close_et=close_et)


@pytest.mark.parametrize("pl, want", [
    (130.0, ExitSignal.PROFIT_TARGET), (-130.0, ExitSignal.STOP_LOSS), (40.0, ExitSignal.HOLD),
])
def test_same_day_target_and_stop(pl, want):
    if datetime.now().strftime("%H:%M") >= "23:59":
        pytest.skip("clock at the forced-close minute")
    assert _mon()._check_exit(_pos(pl), {}, 720.0)[0] == want


def test_forced_close_and_no_regime_exit():
    sig, why = _mon("00:00")._check_exit(_pos(40.0), {}, 720.0)
    assert sig == ExitSignal.DTE_SAFETY and "Same-day close" in why
    # A call debit opened against a bearish regime: no regime-shift exit.
    if datetime.now().strftime("%H:%M") < "23:59":
        assert _mon()._check_exit(_pos(10.0), {"SPY": Regime.BEARISH}, 720.0)[0] == ExitSignal.HOLD
    # An ordinary call debit is unaffected by the same-day rules.
    assert _mon("00:00")._check_exit(_pos(40.0, playbook=""), {}, 720.0)[0] != ExitSignal.DTE_SAFETY


def test_same_day_stop_does_not_start_a_cooldown():
    rows = [ClosedTrade("SPY", "Call Debit Spread", "stop_loss", "", -130.0, "2026-11-24",
                        "2026-11-24T16:00:00Z", playbook=sdm.PLAYBOOK),
            ClosedTrade("SPY", "Call Debit Spread", "stop_loss", "", -130.0, "2026-12-18",
                        "2026-11-24T17:00:00Z")]
    got = TradingAgent._stops_from(SimpleNamespace(closes_since=lambda days: rows))
    assert len(got) == 1


# ── the agent stage ────────────────────────────────────────────────────────

class _Journal:
    def __init__(self):
        self.rows = []

    def log_signal(self, **kw):
        self.rows.append(kw)


def _agent(tmp_path, monkeypatch, *, now=datetime(2026, 11, 24, 10, 5), vix=27.0, spot=722.0,
           ref=718.0, preset=PRE, status="submitted"):
    monkeypatch.setattr(sdm, "STATE_PATH", tmp_path / "same_day.json")
    monkeypatch.setattr(sdm, "prior_vix_close", lambda today, history=None: vix)
    if ref is not None:
        sdm.save_state(sdm.SameDayState(day=now.date().isoformat(), open_ref=ref))
    sent = []
    risk_pcts = []

    class Exec:
        def execute(self, verdict):
            sent.append(verdict)
            return {"status": status if verdict.approved else "rejected", "qty": 1, "run_id": "r1"}

    rm = RiskManager(max_risk_pct=0.02)
    me = SimpleNamespace(
        preset=preset, _now_et=lambda: now, journal_kb=_Journal(), executor=Exec(), risk_manager=rm,
        config=SimpleNamespace(trading=SimpleNamespace(force_market_open=False)),
        data_provider=SimpleNamespace(
            fetch_option_chain=lambda tk, exp, opt: _chain(opt, spot),
            get_underlying_bid_ask=lambda tk: (spot - 0.01, spot + 0.01)),
        _total_risk_state=lambda mr, bal: (0.0, 0.2 * bal),
        _count_entry=lambda tk: None,
    )
    me._set_trade_risk_pct = lambda pct: (risk_pcts.append(pct), setattr(rm, "max_risk_pct", pct))
    me._same_day_skip_reason = TradingAgent._same_day_skip_reason.__get__(me)
    me.sent, me.risk_pcts = sent, risk_pcts
    return me


def _run(me, held=None):
    held = set() if held is None else held
    res = TradingAgent._stage_same_day(me, 28533.0, 28533.0, "paper", True, {}, held)
    return res, held


def test_stage_buys_in_the_direction_of_the_move(tmp_path, monkeypatch):
    me = _agent(tmp_path, monkeypatch)
    res, held = _run(me)
    assert res["execution"]["status"] == "submitted" and res["strategy"] == "Call Debit Spread"
    assert me.sent[0].approved and me.sent[0].plan.playbook == sdm.PLAYBOOK
    assert "SPY" in held and me.risk_pcts[-1] <= 0.01
    assert sdm.load_state(date(2026, 11, 24)).slots_used == ["10:00"]
    row = me.journal_kb.rows[-1]
    assert row["raw_signal"]["playbook"] == sdm.PLAYBOOK and row["raw_signal"]["prior_vix"] == 27.0
    # Same slot, next cycle: no second attempt.
    assert _run(me)[0] is None


def test_stage_put_side(tmp_path, monkeypatch):
    res, _ = _run(_agent(tmp_path, monkeypatch, spot=712.0, ref=718.0))
    assert res["strategy"] == "Put Debit Spread"


@pytest.mark.parametrize("kw, held, reason", [
    (dict(vix=22.0), set(), "prior VIX close 22.0 < 25"),
    (dict(vix=None), set(), "prior VIX close unavailable"),
    (dict(spot=718.0), set(), "no move"),
    ({}, {"SPY"}, "existing position"),
])
def test_stage_skips(tmp_path, monkeypatch, kw, held, reason):
    me = _agent(tmp_path, monkeypatch, **kw)
    res, _ = _run(me, held)
    assert res["status"] == "skipped" and reason in res["reason"] and not me.sent
    assert me.journal_kb.rows[-1]["action"] == "skipped_same_day"


@pytest.mark.parametrize("kw", [
    dict(preset=replace(PRE, same_day_mode="off")),
    dict(now=datetime(2026, 11, 24, 9, 32)),        # before the 09:35 reference
    dict(now=datetime(2026, 11, 24, 10, 40)),       # between slots
    dict(now=datetime(2026, 11, 24, 15, 26)),       # past the close time
])
def test_stage_does_nothing(tmp_path, monkeypatch, kw):
    me = _agent(tmp_path, monkeypatch, **kw)
    assert _run(me)[0] is None and not me.sent and not me.journal_kb.rows


def test_reference_recorded_on_first_cycle(tmp_path, monkeypatch):
    me = _agent(tmp_path, monkeypatch, now=datetime(2026, 11, 24, 9, 36), ref=None, spot=716.0)
    assert _run(me)[0] is None
    assert sdm.load_state(date(2026, 11, 24)).open_ref == pytest.approx(716.0, abs=0.05)


# ── preset ─────────────────────────────────────────────────────────────────

def test_summary_tag_and_validation():
    assert "SameDay SPY VIX≥25" in PRE.to_summary_line()
    assert "SameDay" not in PRESETS["balanced"].to_summary_line()
    ok = _coerce_overrides({"same_day_mode": "live", "same_day_entry_times_et": ["10:00", "13:00"],
                            "same_day_close_et": "15:00", "same_day_stop_pct": 0.4})
    assert ok["same_day_entry_times_et"] == ("10:00", "13:00") and ok["same_day_stop_pct"] == 0.4
    bad = _coerce_overrides({"same_day_mode": "shadow", "same_day_close_et": "3pm",
                             "same_day_target_pct": 1.5, "same_day_entry_times_et": ["25:00"]})
    assert bad == {}


def test_close_rows_carry_the_playbook():
    from trading_agent.journal_reader import JournalReader
    rec = {"ticker": "SPY", "timestamp": "2026-11-24T16:00:00Z",
           "raw_signal": {"strategy": "Call Debit Spread", "exit_signal": "stop_loss",
                          "playbook": sdm.PLAYBOOK}}
    assert JournalReader._closed_from_row(json.loads(json.dumps(rec))).playbook == sdm.PLAYBOOK


def test_same_day_exit_skips_the_debounce(monkeypatch, tmp_path):
    from trading_agent import entry_confirmation
    from trading_agent.daily_state import DailyStateStore
    monkeypatch.setattr(entry_confirmation, "before_entry_window", lambda now, hhmm: False)
    me = SimpleNamespace(daily_state=DailyStateStore(str(tmp_path)), preset=PRE,
                         _exit_vote_key=TradingAgent._exit_vote_key)
    pos = _pos(130.0)
    pos.exit_signal = ExitSignal.PROFIT_TARGET
    assert TradingAgent._should_exit_spread(me, pos) is True          # first cycle
    plain = _pos(130.0, playbook="")
    plain.exit_signal = ExitSignal.PROFIT_TARGET
    assert TradingAgent._should_exit_spread(me, plain) is False       # vote 1 of 3


def test_premarket_brief_line():
    from trading_agent import session_brief as sb
    base = dict(today=date(2026, 11, 24), preset=PRE)
    hot = sb.build_premarket(dict(base, market_state={"state": "DEFENSIVE", "inputs": {"vix": 26.6},
                                                      "gate": {"allowed_strategies": []}}))
    assert "Same-day momentum (SPY 0DTE" in hot and "VIX 26.6 ≥ 25" in hot
    calm = sb.build_premarket(dict(base, market_state={"state": "NORMAL", "inputs": {"vix": 17.0},
                                                       "gate": {"allowed_strategies": []}}))
    assert "VIX 17.0 < 25 — no trades" in calm
