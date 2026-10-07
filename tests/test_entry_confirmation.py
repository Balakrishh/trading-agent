"""2026-10-05 — wait N cycles before opening; entry window; hourly limit."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from trading_agent.entry_confirmation import (
    EntryConfirmations, before_entry_window, entries_in_last_hour, signature,
)

T0 = datetime(2026, 10, 6, 15, 0, tzinfo=timezone.utc)          # 11:00 ET


def plan(strategy="Put Debit Spread", exp="2026-11-06", far=""):
    return SimpleNamespace(strategy_name=strategy, expiration=exp, far_expiration=far)


def cycle(path, required, observations, at):
    ec = EntryConfirmations(required, path=path, now=at).begin()
    out = {t: ec.observe(t, p) for t, p in observations.items()}
    ec.save()
    return out, ec


def test_confirms_after_three_consecutive_cycles(tmp_path):
    fp = tmp_path / "c.json"
    counts = []
    for i in range(3):
        out, _ = cycle(fp, 3, {"IWM": plan()}, T0 + timedelta(seconds=75 * i))
        counts.append((out["IWM"].count, out["IWM"].confirmed))
    assert counts == [(1, False), (2, False), (3, True)]


def test_regime_flicker_resets(tmp_path):
    """IWM 2026-10-05: put debit one cycle, nothing the next → start over."""
    fp = tmp_path / "c.json"
    cycle(fp, 3, {"IWM": plan()}, T0)
    cycle(fp, 3, {}, T0 + timedelta(seconds=75))                # no approved plan
    out, _ = cycle(fp, 3, {"IWM": plan()}, T0 + timedelta(seconds=150))
    assert out["IWM"].count == 1


def test_changed_strategy_or_expiry_resets(tmp_path):
    fp = tmp_path / "c.json"
    cycle(fp, 3, {"QQQ": plan("Call Debit Spread")}, T0)
    out, _ = cycle(fp, 3, {"QQQ": plan("Call Debit Spread", "2026-11-13")}, T0 + timedelta(seconds=75))
    assert out["QQQ"].count == 1
    assert signature(plan("Calendar Spread", "2026-10-30", "2026-11-20")) == \
        "Calendar Spread|2026-10-30|2026-11-20"


def test_gap_longer_than_a_cycle_resets(tmp_path):
    fp = tmp_path / "c.json"
    cycle(fp, 3, {"SPY": plan()}, T0)
    out, _ = cycle(fp, 3, {"SPY": plan()}, T0 + timedelta(minutes=10))
    assert out["SPY"].count == 1


def test_consumed_starts_over(tmp_path):
    fp = tmp_path / "c.json"
    for i in range(2):
        cycle(fp, 2, {"GLD": plan()}, T0 + timedelta(seconds=75 * i))
    ec = EntryConfirmations(2, path=fp, now=T0 + timedelta(seconds=150)).begin()
    assert ec.observe("GLD", plan()).confirmed
    ec.consumed("GLD")
    ec.save()
    out, _ = cycle(fp, 2, {"GLD": plan()}, T0 + timedelta(seconds=225))
    assert out["GLD"].count == 1


def test_required_one_is_legacy_immediate(tmp_path):
    out, _ = cycle(tmp_path / "c.json", 1, {"SPY": plan()}, T0)
    assert out["SPY"].confirmed


@pytest.mark.parametrize("utc,blocked", [
    (datetime(2026, 10, 6, 13, 40, tzinfo=timezone.utc), True),   # 09:40 ET
    (datetime(2026, 10, 6, 13, 45, tzinfo=timezone.utc), False),  # 09:45 ET
])
def test_entry_window(utc, blocked):
    assert before_entry_window(utc, "09:45") is blocked
    assert before_entry_window(utc, "bad") is False


def test_entries_in_last_hour():
    times = [T0 - timedelta(minutes=m) for m in (5, 59, 61, 120)]
    assert entries_in_last_hour(times, T0) == 2


def _agent(preset_kw, tmp_path):
    from trading_agent.agent import TradingAgent
    from trading_agent.strategy_presets import PRESETS
    from dataclasses import replace
    a = TradingAgent.__new__(TradingAgent)
    a.preset = replace(PRESETS["balanced"], **preset_kw)
    rows = []
    a.journal_kb = SimpleNamespace(log_signal=lambda **kw: rows.append(kw), jsonl_path=None)
    a._cached_price = lambda t: 1.0
    return a, rows


def test_agent_gates(tmp_path, monkeypatch):
    import trading_agent.agent as agent_mod
    a, rows = _agent({"entry_confirm_cycles": 2, "no_entry_before_et": "",
                      "max_new_entries_per_hour": 1}, tmp_path)
    a._begin_entry_gates()
    assert a._entry_confirmation_block("IWM", plan()) == "entry_confirming (1/2)"
    assert a._entry_rate_gate("IWM") is None
    a._count_entry("IWM")                                            # Broad Market used
    skip = a._entry_rate_gate("QQQ")                                 # same sector → held
    assert skip["reason"] == "Entry rate limit" and rows[-1]["action"] == "skipped_entry_rate"
    assert rows[-1]["raw_signal"]["scope"] == "Broad Market"
    assert a._entry_rate_gate("XLF") is None                         # other sector → free
    a._entry_confirm.save()
    a._begin_entry_gates()
    assert a._entry_confirmation_block("IWM", plan()) is None          # 2/2 → go


# ── entry timing (2026-10-06) ─────────────────────────────────────────────

from trading_agent.entry_confirmation import (  # noqa: E402
    TimingParams, score_legs, score_plan, timing_decision,
)

TP = TimingParams(max_wait_cycles=6, best_tolerance_pct=0.01, chase_limit_pct=0.03)


def debit_plan(long_ask, short_bid, width=2.0, long_bid=None, short_ask=None):
    legs = [SimpleNamespace(symbol="L", action="buy", bid=long_bid or long_ask - 0.04, ask=long_ask),
            SimpleNamespace(symbol="S", action="sell", bid=short_bid, ask=short_ask or short_bid + 0.04)]
    return SimpleNamespace(strategy_name="Call Debit Spread", expiration="2026-11-06",
                           far_expiration="", legs=legs, spread_width=width, net_credit=-(long_ask - short_bid))


def test_score_legs_by_structure():
    credit = score_legs([{"action": "sell", "bid": 1.20, "ask": 1.30},
                         {"action": "buy", "bid": 0.50, "ask": 0.60}], 5.0)
    assert credit == {"score": pytest.approx(0.12), "gap": pytest.approx(0.10), "net": pytest.approx(0.60)}
    debit = score_plan(debit_plan(2.60, 1.60))                 # debit 1.00 on width 2
    assert debit["score"] == pytest.approx(1.0) and debit["net"] == pytest.approx(-1.0)
    cal = score_legs([{"action": "buy", "bid": 2.95, "ask": 2.97},
                      {"action": "sell", "bid": 1.88, "ask": 1.90}], 0.0)
    assert cal["score"] == pytest.approx(-1.09 / 1.07, abs=1e-4)
    assert score_legs([{"action": "buy", "bid": 0, "ask": 0}], 2.0) is None


def h(*scores, gap=0.05):
    return [{"score": s, "gap": gap, "net": 0} for s in scores]


@pytest.mark.parametrize("hist,decision", [
    (h(1.0, 1.0), "wait"),                               # still confirming (needs 3)
    (h(1.0, 1.0, 1.0), "enter"),                         # confirmed at the best price
    (h(1.0, 1.1, 1.0), "wait"),                          # below the best seen → wait
    (h(1.0, 1.1, 1.0, 1.1), "enter"),                    # back at the best
    (h(1.0, 1.1, 1.0, 0.99, 0.99, 0.98), "enter"),       # max wait, within 3 % of 1.0
    (h(1.0, 1.1, 1.0, 0.9, 0.9, 0.9), "skip"),           # max wait, 10 % worse → don't chase
])
def test_timing_decision(hist, decision):
    assert timing_decision(hist, 3, TP)[0] == decision


def test_wide_gap_waits_even_at_best_score():
    hist = [{"score": 1.0, "gap": 0.04, "net": 0}, {"score": 1.0, "gap": 0.04, "net": 0},
            {"score": 1.0, "gap": 0.20, "net": 0}]
    assert timing_decision(hist, 3, TP)[0] == "wait"


def test_live_mode_waits_then_enters_on_best(tmp_path, monkeypatch):
    a, _ = _agent({"entry_confirm_cycles": 3, "no_entry_before_et": "",
                   "entry_timing_mode": "live", "entry_max_wait_cycles": 6}, tmp_path)
    prices = [(2.60, 1.60), (2.50, 1.60), (2.60, 1.60), (2.50, 1.60)]   # debit 1.0, .9, 1.0, .9
    out = []
    for la, sb in prices:
        a._begin_entry_gates()
        out.append(a._entry_confirmation_block("QQQ", debit_plan(la, sb)))
        a._entry_confirm.save()
    assert out == ["entry_confirming (1/3)", "entry_confirming (2/3)", "entry_timing_wait", None]


def test_shadow_mode_journals_what_timing_would_have_saved(tmp_path):
    a, rows = _agent({"entry_confirm_cycles": 2, "no_entry_before_et": "",
                      "entry_timing_mode": "shadow", "entry_max_wait_cycles": 4}, tmp_path)
    p1, p2 = debit_plan(2.50, 1.60), debit_plan(2.60, 1.60)        # debit .90 then 1.00
    a._begin_entry_gates(); a._entry_confirmation_block("QQQ", p1); a._entry_confirm.save()
    a._begin_entry_gates()
    assert a._entry_confirmation_block("QQQ", p2) is None            # shadow: enter at confirmation
    a._consume_entry("QQQ", {"execution": {"qty": 2}})
    a._entry_confirm.save()
    # Next cycle the filled legs quote back at a 0.90 debit (the best seen) → rule would enter there.
    a.data_provider = SimpleNamespace(fetch_option_quotes=lambda syms: {
        "L": {"bid": 2.46, "ask": 2.50}, "S": {"bid": 1.60, "ask": 1.64}})
    a._begin_entry_gates()
    a._entry_confirm.now = a._entry_confirm.now + timedelta(seconds=75)
    a._advance_entry_timing_shadow()
    (row,) = [r for r in rows if r["action"] == "entry_timing_shadow"]
    o = row["raw_signal"]
    assert o["decision"] == "enter" and o["actual_net"] == pytest.approx(-1.0)
    assert o["timing_net"] == pytest.approx(-0.90)
    assert o["improvement_usd"] == pytest.approx(20.0)              # $0.10 × 100 × 2


def test_shadow_records_agreement_at_entry(tmp_path):
    a, rows = _agent({"entry_confirm_cycles": 2, "no_entry_before_et": "",
                      "entry_timing_mode": "shadow"}, tmp_path)
    for _ in range(2):
        a._begin_entry_gates()
        a._entry_confirmation_block("SPY", debit_plan(2.60, 1.60))
        a._entry_confirm.save()
    a._consume_entry("SPY", {"execution": {"qty": 1}})
    a.data_provider = SimpleNamespace(fetch_option_quotes=lambda syms: {})
    a._advance_entry_timing_shadow()
    (row,) = [r for r in rows if r["action"] == "entry_timing_shadow"]
    assert row["raw_signal"]["improvement_usd"] == 0.0


def test_entry_rate_scope_global_and_journal_seed(tmp_path):
    """Per-sector by default (2026-10-06: one AMZN entry held all 9 tickers
    for an hour under the global count); "global" keeps one bucket."""
    import json
    from datetime import datetime, timezone
    a, _ = _agent({"max_new_entries_per_hour": 1, "entry_rate_scope": "global"}, tmp_path)
    a._begin_entry_gates()
    a._count_entry("AMZN")
    assert a._entry_rate_gate("XLF") is not None                     # global: everything held
    fp = tmp_path / "signals_live.jsonl"
    fp.write_text(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                              "ticker": "AMZN", "action": "submitted", "raw_signal": {}}) + "\n")
    a, _ = _agent({"max_new_entries_per_hour": 1}, tmp_path)
    a.journal_kb.jsonl_path = str(fp)
    a._begin_entry_gates()                                           # seeded from the journal
    assert a._entries_by_scope == {"Consumer Discretionary": 1}
    assert a._entry_rate_gate("META") is None                        # Communications free
    assert a._entry_rate_gate("AMZN") is not None
