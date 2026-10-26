"""Quality-at-lows sleeve shadow scanner (skill 67)."""
import json
from dataclasses import replace
from datetime import date

import pytest

from trading_agent import quality_lows as ql
from trading_agent.market_data_schwab import SchwabMarketDataProvider

CFG = ql.DEFAULT_CONFIG


def _series(n_decline=230, low=50.0, base=30, end=56.0):
    """Down from 100 to ``low`` over ``n_decline`` sessions, then a base
    that drifts up to ``end`` over ``base`` sessions (higher lows)."""
    closes = [100 - (100 - low) * i / (n_decline - 1) for i in range(n_decline)]
    closes += [low + (end - low) * (i + 1) / base for i in range(base)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    return highs, lows, closes


# ── price features ────────────────────────────────────────────────────

def test_price_features_needs_a_year():
    h, l, c = _series()
    assert ql.price_features(h[-200:], l[-200:], c[-200:]) is None
    assert ql.price_features(h, l[:-1], c) is None


def test_price_features_base_after_low():
    pf = ql.price_features(*_series())
    assert pf["low_52w"] == 49.5 and pf["sessions_since_low"] == 30
    # The 30-session base starts the session after the low.
    assert pf["base_low"] == pytest.approx(49.7) and pf["base_range_pos"] < 0.01
    assert pf["higher_low"] is True
    assert pf["close"] > pf["sma20"]
    assert 0 < pf["range_pos"] < 0.35 and pf["drawdown_pct"] < 0
    assert pf["rv30"] >= 0


def test_new_low_today_is_no_higher_low():
    h, l, c = _series(n_decline=260, base=1, end=49.0)
    pf = ql.price_features(h, l, c)
    assert pf["sessions_since_low"] == 0 and pf["higher_low"] is False


def test_rsi_bounds():
    assert ql._rsi([1.0] * 5) is None
    assert ql._rsi([float(i) for i in range(30)]) == 100.0
    assert ql._rsi([float(30 - i) for i in range(30)]) < 1.0


# ── quality, growth, cause, stage ─────────────────────────────────────

GOOD_FUND = {"market_cap": 2.7e11, "pe_ratio": 60.0, "eps_ttm": 3.4, "net_profit_margin_ttm": 29.7,
             "roe": 40.0, "beta": 2.0, "vol_avg_10d": 3e7, "rev_change_ttm_pct": 14.7,
             "net_profit_margin_mrq": 28.5}
GOOD_EST = {"eps_cur_year_rev_30d_pct": -0.05, "eps_next_year_rev_30d_pct": -1.16,
            "revisions_up_30d": 3, "revisions_down_30d": 10, "rev_growth_next_year_pct": 10.8}


def test_quality_relaxes_pe_and_beta_but_not_profit():
    assert ql.quality_check(GOOD_FUND)[0] is True
    ok, reasons = ql.quality_check({**GOOD_FUND, "eps_ttm": -1.0})
    assert not ok and any("eps_ttm" in r for r in reasons)
    assert ql.quality_check(None) == (False, ["missing:fundamentals"])


def test_growth_size_of_cut_decides_not_the_count():
    # NFLX 2026-10-25: 3 up / 10 down but next-year EPS only −1.2 %.
    assert ql.growth_check(GOOD_FUND, GOOD_EST)[0] is True
    intact, reasons = ql.growth_check(GOOD_FUND, {**GOOD_EST, "eps_next_year_rev_30d_pct": -5.0})
    assert intact is False and any("eps_next_year_cut" in r for r in reasons)


def test_growth_counts_are_the_fallback_without_sizes():
    est = {"revisions_up_30d": 0, "revisions_down_30d": 30}
    assert ql.growth_check(GOOD_FUND, est)[0] is False
    assert ql.growth_check(GOOD_FUND, {"revisions_up_30d": 4, "revisions_down_30d": 6})[0] is True


def test_growth_fails_on_revenue_margin_or_forward_growth():
    assert ql.growth_check({**GOOD_FUND, "rev_change_ttm_pct": -3.0}, GOOD_EST)[0] is False
    assert ql.growth_check({**GOOD_FUND, "net_profit_margin_mrq": 20.0}, GOOD_EST)[0] is False
    assert ql.growth_check(GOOD_FUND, {**GOOD_EST, "rev_growth_next_year_pct": -1.0})[0] is False


def test_growth_unknown_without_any_input():
    intact, reasons = ql.growth_check({}, None)
    assert intact is None and "missing:rev_change_ttm_pct" in reasons


@pytest.mark.parametrize("intact,stock,sector,want", [
    (False, -40.0, -5.0, "fundamental"),
    (None, -40.0, -5.0, "unclear"),
    (True, None, -5.0, "unclear"),
    (True, -40.0, -30.0, "sector"),
    (True, -40.0, -10.0, "sentiment"),
    (True, -40.0, None, "sentiment"),
])
def test_drop_cause(intact, stock, sector, want):
    assert ql.drop_cause(intact, stock, sector) == want


def _pf(**kw):
    base = {"base_range_pos": 0.0, "range_pos": 0.1, "sessions_since_low": 20, "rsi14": 50.0,
            "close": 55.0, "sma20": 53.0, "sma50": 58.0, "higher_low": True}
    return {**base, **kw}


@pytest.mark.parametrize("pf,q_ok,cause,want", [
    (None, True, "sentiment", "excluded"),
    (_pf(base_range_pos=0.3), True, "sentiment", "not_near_low"),
    (_pf(range_pos=0.5), True, "sentiment", "not_near_low"),
    (_pf(), False, "sentiment", "excluded"),
    (_pf(), True, "fundamental", "excluded"),
    (_pf(sessions_since_low=3), True, "sentiment", "watch"),
    (_pf(rsi14=35.0), True, "sentiment", "watch"),
    (_pf(close=52.0), True, "sentiment", "basing"),
    (_pf(higher_low=False), True, "sentiment", "basing"),
    (_pf(), True, "sentiment", "turn"),
    (_pf(close=60.0, sma20=57.0), True, "unclear", "confirmed"),
])
def test_stage(pf, q_ok, cause, want):
    assert ql.stage(pf, q_ok, cause) == want


# ── structures ────────────────────────────────────────────────────────

def test_choose_structure():
    assert ql.choose_structure(0.5, 0.3, 10) == "wait_earnings"
    assert ql.choose_structure(0.5, 0.3, 30) == "call_debit_spread"
    assert ql.choose_structure(0.35, 0.36, None) == "leaps_call"
    assert ql.choose_structure(None, 0.3, None) == "leaps_call"
    assert ql.choose_structure(0.5, 0.3, -2) == "call_debit_spread"   # report already out


def test_monthly_expirations_are_third_fridays_in_range():
    today = date(2026, 10, 26)
    exps = ql.monthly_expirations(today, (270, 550))
    assert exps and all(d.weekday() == 4 and 15 <= d.day <= 21 for d in exps)
    assert all(270 <= (d - today).days <= 550 for d in exps)
    mid = 410
    assert abs((exps[0] - today).days - mid) <= abs((exps[-1] - today).days - mid)


def _c(strike, bid, ask, delta, iv=0.35, kind="C"):
    return {"symbol": f"X{kind}{strike}", "strike": strike, "bid": bid, "ask": ask,
            "delta": delta, "iv": iv}


CALLS = [_c(56, 15.75, 18.40, 0.746), _c(65, 6.90, 7.30, 0.573), _c(82.5, 1.61, 1.74, 0.20),
         _c(100, 0.50, 0.60, 0.08), _c(115, 0.13, 0.19, 0.026), _c(120, 0, 0.05, 0.01)]


def test_price_leaps_at_natural():
    q = ql.price_leaps(CALLS, 65.85, 418)
    assert q["legs"][0]["strike"] == 56 and q["cost_per_contract"] == 1840.0
    assert q["breakeven"] == 74.4
    assert q["extrinsic_pct_of_spot"] == round((18.40 - 9.85) / 65.85 * 100, 2)


def test_price_debit_spread_to_the_high():
    q = ql.price_debit_spread(CALLS, 116.7)
    assert [l["strike"] for l in q["legs"]] == [65, 115]
    assert q["cost_per_contract"] == round((7.30 - 0.13) * 100, 2)
    assert q["reward_risk"] == round((50 - 7.17) / 7.17, 2)


def test_fitting_spread_is_widest_under_budget_with_rr_1():
    q = ql.price_fitting_spread(CALLS, 578.0)
    assert [l["strike"] for l in q["legs"]] == [65, 82.5]
    assert q["cost_per_contract"] == 569.0 and q["reward_risk"] >= 1.0
    assert ql.price_fitting_spread(CALLS, 100.0) is None
    assert ql.price_fitting_spread(CALLS, None) is None


def test_price_csp_nearest_base_low():
    puts = [_c(60, 1.0, 1.1, -0.2, kind="P"), _c(64, 2.14, 2.25, -0.377, kind="P"),
            _c(70, 6.0, 6.2, -0.6, kind="P")]
    q = ql.price_csp(puts, 64.1)
    assert q["legs"][0]["strike"] == 64 and q["credit_per_contract"] == 214.0
    assert q["cash_secured"] == 6400.0 and q["effective_entry"] == 61.86
    assert ql.price_csp([], 64.1) is None


def test_size_contracts():
    assert ql.size_contracts(569.0, 28918.0) == {"budget": 578.36, "contracts": 1, "fits": True}
    assert ql.size_contracts(1840.0, 28918.0)["fits"] is False
    assert ql.size_contracts(None, 28918.0)["contracts"] == 0
    assert ql.size_contracts(500.0, None)["fits"] is False


def test_affordable_structure_order():
    quotes = {"leaps_call": {"fits": False}, "call_debit_spread_fit": {"fits": True},
              "cash_secured_put": {"fits": True}}
    assert ql.affordable_structure("leaps_call", quotes) == "call_debit_spread_fit"
    assert ql.affordable_structure("leaps_call", {"leaps_call": {"fits": True}}) == "leaps_call"
    assert ql.affordable_structure("wait_earnings", quotes) is None
    assert ql.affordable_structure("leaps_call", {}) is None


def test_structure_quotes_prices_all_and_caps_csp():
    puts = [_c(64, 2.14, 2.25, -0.377, kind="P")]

    def get(path):
        return {"contracts": puts if "option_type=put" in path else CALLS}
    pf = {"close": 65.85, "high_52w": 116.7, "base_low": 64.1}
    out = ql.structure_quotes("NFLX", pf, "leaps_call", date(2026, 10, 26), get, 28918.0, pause_s=0)
    assert set(out) == {"leaps_call", "call_debit_spread", "call_debit_spread_fit", "cash_secured_put"}
    assert out["cash_secured_put"]["fits"] is False          # $6,400 > 15 % of $28,918
    assert out["call_debit_spread_fit"]["fits"] is True


def test_structure_quotes_tolerates_empty_chains():
    out = ql.structure_quotes("X", {"close": 50, "high_52w": 90, "base_low": 48}, "leaps_call",
                              date(2026, 10, 26), lambda p: None, 28918.0, pause_s=0)
    assert out == {}


# ── collection ────────────────────────────────────────────────────────

def _deps(fund=GOOD_FUND, est=GOOD_EST, bars=None, get=None, dte=None):
    calls = {"fund": 0}

    def fundamentals(t):
        calls["fund"] += 1
        if t == "BOOM":
            raise RuntimeError("down")
        return fund
    d = ql.Deps(bars=bars or (lambda t: _series()), fundamentals=fundamentals,
                estimates=lambda t, spot: est, days_to_earnings=lambda t, d: dte,
                get=get, equity=28918.0)
    return d, calls


def test_not_near_low_skips_network_calls():
    up = [float(50 + i * 0.2) for i in range(260)]
    d, calls = _deps(bars=lambda t: ([c + 0.5 for c in up], [c - 0.5 for c in up], up))
    row = ql.evaluate_ticker("AAPL", date(2026, 10, 26), d, {})
    assert row["stage"] == "not_near_low" and calls["fund"] == 0


def test_turn_row_has_structure_and_quotes():
    def get(path):
        return {"contracts": CALLS if "call" in path else [_c(50, 1.0, 1.1, -0.3, kind="P")]}
    d, _ = _deps(get=get)
    row = ql.evaluate_ticker("NFLX", date(2026, 10, 26), d, {"XLC": -12.0})
    assert row["stage"] in ("turn", "confirmed") and row["cause"] == "sentiment"
    # The synthetic series barely moves (rv30 ≈ 0), so 35 % IV is rich → spread.
    assert row["iv_rv"] > CFG.iv_rich_ratio
    assert row["structure"] == "call_debit_spread" and "leaps_call" in row["quotes"]
    assert row["sector_etf"] == "XLC" and row["affordable"] == "call_debit_spread_fit"


def test_turn_inside_earnings_blackout_waits():
    d, _ = _deps(dte=5)
    row = ql.evaluate_ticker("NFLX", date(2026, 10, 26), d, {})
    assert row["structure"] == "wait_earnings" and row["affordable"] is None


def test_collect_appends_dedupes_and_survives_a_failure(tmp_path):
    fp = tmp_path / "ql.jsonl"
    d, _ = _deps()
    rows = ql.collect(["NFLX", "BOOM", "NFLX"], d, date(2026, 10, 26), path=fp)
    assert [r["ticker"] for r in rows] == ["NFLX"]
    assert len(fp.read_text().splitlines()) == 1
    assert ql.collect(["NFLX"], d, date(2026, 10, 26), path=fp) == []
    ql.collect(["ORCL"], d, date(2026, 10, 26), path=fp, dry_run=True)
    assert len(fp.read_text().splitlines()) == 1


def test_sector_cap_marks_the_third_in_a_sector():
    rows = [{"ticker": t, "stage": "turn", "estimates": {"eps_next_year_rev_30d_pct": v}}
            for t, v in (("AAPL", 1.0), ("MSFT", 3.0), ("NVDA", 2.0))]
    rows.append({"ticker": "JNJ", "stage": "turn", "estimates": {}})
    ql.apply_sector_cap(rows, replace(CFG, max_per_sector=2))
    capped = {r["ticker"]: r["sector_capped"] for r in rows}
    assert capped == {"AAPL": True, "MSFT": False, "NVDA": False, "JNJ": False}


def test_universe_drops_broad_market_and_reads_extras(monkeypatch):
    from trading_agent import watchlist_store
    wl = watchlist_store.Watchlist(tickers=[watchlist_store.WatchlistEntry(symbol="SPY"),
                                            watchlist_store.WatchlistEntry(symbol="NFLX")])
    monkeypatch.setattr(watchlist_store, "load_watchlist", lambda: wl)
    monkeypatch.setenv("QUALITY_LOWS_EXTRA_TICKERS", "jnj, NFLX")
    assert ql.universe() == ["NFLX", "JNJ"]
    monkeypatch.delenv("QUALITY_LOWS_EXTRA_TICKERS")
    assert ql.universe()[1:] == list(ql.EXTRA_TICKERS)


def test_report_latest_day(tmp_path):
    fp = tmp_path / "ql.jsonl"
    rows = [{"date": "2026-10-23", "ticker": "OLD", "stage": "watch"},
            {"date": "2026-10-26", "ticker": "AAPL", "stage": "not_near_low"},
            {"date": "2026-10-26", "ticker": "NFLX", "stage": "turn", "cause": "sentiment",
             "structure": "leaps_call", "affordable": "call_debit_spread_fit",
             "quotes": {"leaps_call": {"cost_per_contract": 1840.0, "contracts": 0}}}]
    fp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    lines = ql.report(fp)
    assert lines[0].startswith("quality-at-lows 2026-10-26: 2 tickers")
    assert len(lines) == 2 and "NFLX" in lines[1] and "affordable=call_debit_spread_fit" in lines[1]
    assert ql.report(tmp_path / "none.jsonl") == ["no quality-lows rows yet"]


def test_account_equity_reads_market_state(tmp_path):
    fp = tmp_path / "ms.json"
    fp.write_text(json.dumps({"account_balance": 28918.33}))
    assert ql._account_equity(fp) == 28918.33
    assert ql._account_equity(tmp_path / "missing.json") is None


# ── Schwab growth fields (skill 67 §3.1) ──────────────────────────────

def test_schwab_fundamentals_growth_fields():
    p = SchwabMarketDataProvider.__new__(SchwabMarketDataProvider)
    raw = {"revChangeTTM": 14.6519, "revChangeYear": 15.851, "revChangeIn": 0.0,
           "grossMarginMRQ": 52.78, "netProfitMarginMRQ": 28.46, "netProfitMarginTTM": 29.69,
           "returnOnInvestment": 18.2, "currentRatio": 1.1, "interestCoverage": 25.0,
           "ltDebtToEquity": 0.6, "totalDebtToCapital": 37.0, "shortIntDayToCover": 0.0}
    p._get = lambda path, params=None: {"instruments": [{"symbol": "NFLX", "fundamental": raw}]}
    f = p.fetch_fundamentals("NFLX")
    assert f["rev_change_ttm_pct"] == 14.6519 and f["rev_change_year_pct"] == 15.851
    assert f["gross_margin_mrq"] == 52.78 and f["net_profit_margin_mrq"] == 28.46
    assert f["return_on_investment"] == 18.2 and f["total_debt_to_capital"] == 37.0
    assert f["short_int_days_to_cover"] is None             # 0.0 = not provided


def test_schwab_zero_revenue_growth_is_missing():
    p = SchwabMarketDataProvider.__new__(SchwabMarketDataProvider)
    p._get = lambda path, params=None: {"instruments": [{"symbol": "X", "fundamental":
                                                         {"revChangeTTM": 0.0}}]}
    assert p.fetch_fundamentals("X")["rev_change_ttm_pct"] is None
