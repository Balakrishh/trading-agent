"""2026-10-07 — open positions keep their trade plan; inferred positions
carry their real contract count (GLD 2-contract put debit read as 1)."""
from __future__ import annotations

from datetime import date

from trading_agent.executor import MAX_HISTORY, trim_plan_history
from trading_agent.position_monitor import PositionMonitor, PositionSnapshot

TODAY = date(2026, 10, 7)


def run(i, *, submitted=False, exp="2026-11-06", valid=True):
    e = {"run_id": f"r{i}", "trade_plan": {"expiration": exp, "valid": valid}}
    if submitted:
        e["order_result"] = {"status": "submitted"}
    return e


def test_submitted_open_run_survives_trim():
    hist = [run(0, submitted=True)] + [run(i) for i in range(1, MAX_HISTORY + 50)]
    out = trim_plan_history(hist, TODAY)
    assert out[0]["run_id"] == "r0" and len(out) == MAX_HISTORY + 1


def test_expired_or_invalid_submitted_runs_are_trimmed():
    hist = ([run(0, submitted=True, exp="2026-09-01"), run(1, submitted=True, valid=False)]
            + [run(i) for i in range(2, MAX_HISTORY + 50)])
    out = trim_plan_history(hist, TODAY)
    assert [e["run_id"] for e in out[:1]] != ["r0"] and len(out) == MAX_HISTORY
    assert all(e["run_id"] not in ("r0", "r1") for e in out)


def test_short_history_untouched():
    hist = [run(0), run(1, submitted=True)]
    assert trim_plan_history(hist, TODAY) is hist


def leg(sym, qty, avg):
    return PositionSnapshot(symbol=sym, qty=qty, side="short" if qty < 0 else "long",
                            avg_entry_price=avg, current_price=avg, market_value=0.0,
                            cost_basis=0.0, unrealized_pl=0.0, unrealized_plpc=0.0,
                            asset_class="us_option")


def test_inferred_debit_spread_counts_contracts():
    legs = [leg("GLD261106P00379000", 2, 9.25), leg("GLD261106P00369000", -2, 5.00)]
    (s,) = PositionMonitor("k", "s").group_into_spreads(legs, [])
    assert s.strategy_name == "Put Debit Spread" and s.contracts_open == 2
    assert s.original_credit == -4.25 and s.max_loss == 425.0      # per contract
    kind, basis, target = PositionMonitor("k", "s").profit_basis(s)
    assert basis == (10 - 4.25) * 100 * 2                          # max profit, both contracts


def test_inferred_calendar_and_partial_fill_counts():
    legs = [leg("QQQ261026C00500000", -3, 5.0), leg("QQQ261123C00500000", 3, 8.0),
            leg("SPY261106P00500000", 4, 3.0), leg("SPY261106P00490000", -3, 1.5)]
    by = {s.underlying: s for s in PositionMonitor("k", "s").group_into_spreads(legs, [])}
    assert by["QQQ"].contracts_open == 3
    assert by["SPY"].contracts_open == 3                            # min |qty| of a partial fill
