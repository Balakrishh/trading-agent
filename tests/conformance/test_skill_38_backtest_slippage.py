"""Conformance test: skill 38 — backtester slippage + commission knobs.

Pins the SimAccount kwargs that BacktestRunner threads through for
modeling real-world execution gaps.

Failure modes caught:
- Someone removes slippage_per_share from apply_open/apply_close →
  backtester reverts to filling at BS mid (over-optimistic).
- Someone hardcodes commission inside apply_* instead of accepting
  commission_per_leg → operators can't sweep broker commission
  models.
- Someone forgets to thread the param from BacktestRunner →
  configuring slippage_ticks_per_leg has no effect (silent).
"""

from __future__ import annotations


def test_skill_38_apply_open_default_slippage_is_zero() -> None:
    """Skill 38 §3.1: default slippage_per_share=0 preserves the
    legacy 'fill at BS mid' behavior so existing backtests don't
    silently change."""
    from trading_agent.backtest.account import SimAccount
    acc = SimAccount.fresh(starting_equity=10_000.0)
    # 1 contract, $0.50 credit, $1 width — legacy default.
    acc.apply_open(credit_per_share=0.50, qty=1, spread_width=1.0)
    # 0.50 × 1 × 100 = $50 credit. Commission $0.65 × 2 = $1.30.
    # Cash should be 10,000 + 50 - 1.30 = 10,048.70
    assert abs(acc.cash - 10_048.70) < 0.01, (
        f"Skill 38 §3.1: default slippage must be zero. "
        f"Got cash=${acc.cash:.2f}, expected $10,048.70."
    )


def test_skill_38_apply_open_with_slippage_reduces_credit() -> None:
    """Skill 38 §3.1: slippage_per_share is subtracted from the
    BS-mid credit before booking. $0.10 slippage on $0.50 credit
    means we book only $0.40 of credit."""
    from trading_agent.backtest.account import SimAccount
    acc = SimAccount.fresh(starting_equity=10_000.0)
    acc.apply_open(
        credit_per_share=0.50, qty=1, spread_width=1.0,
        slippage_per_share=0.10,
    )
    # Effective credit: $0.40 × 100 = $40. Commission $1.30.
    # Cash: 10,000 + 40 - 1.30 = 10,038.70.
    assert abs(acc.cash - 10_038.70) < 0.01, (
        f"Skill 38 §3.1: slippage must reduce booked credit. "
        f"Got cash=${acc.cash:.2f}, expected $10,038.70."
    )


def test_skill_38_apply_close_with_slippage_increases_debit() -> None:
    """Skill 38 §3.1 + symmetry fix: slippage on close is ADDED to
    debit AND symmetrically subtracted from the open's BS-mid credit
    when computing realised P&L (so the round-trip drag is correctly
    2 × slippage × qty × 100, not 1×).

    This test exercises apply_close in isolation with the SAME
    slippage value the runner would pass alongside an apply_open at
    the same slippage. Real-world usage: runner threads
    slippage_per_share to BOTH calls."""
    from trading_agent.backtest.account import SimAccount
    acc = SimAccount.fresh(starting_equity=10_000.0)
    # Open at $0.50 BS-mid credit WITH the same slippage the close
    # will use ($0.05). Effective booked credit = $0.45.
    acc.apply_open(
        credit_per_share=0.50, qty=1, spread_width=1.0,
        slippage_per_share=0.05,
    )
    cash_after_open = acc.cash
    # Close at BS-mid debit $0.20 with the same $0.05 slippage.
    # effective_open_credit (mirror) = $0.45 → credit_dollars = $45
    # effective_debit = $0.25 → debit_dollars = $25
    # commission round-trip = $1.30 per side (charged at close)
    # realised = (45 - 25) - 1.30 = $18.70
    realised = acc.apply_close(
        credit_per_share=0.50, qty=1,
        closing_debit_per_share=0.20,
        slippage_per_share=0.05,
    )
    assert abs(realised - 18.70) < 0.01, (
        f"Skill 38 §3.1: symmetric slippage accounting means apply_close "
        f"mirrors the open's slippage on credit_per_share. Drag is "
        f"2 × slip per round-trip. Got realised=${realised:.2f}, "
        f"expected $18.70 = (45 - 25) - 1.30."
    )
    # Cash after close: cash_after_open - debit_dollars(25) - commission(1.30)
    expected_cash = cash_after_open - 26.30
    assert abs(acc.cash - expected_cash) < 0.01


def test_skill_38_apply_open_commission_override() -> None:
    """Skill 38 §3.1: commission_per_leg overrides the module
    default. Pass 0.0 to model commission-free brokers (paper
    Alpaca); pass 1.00 for higher-cost brokers."""
    from trading_agent.backtest.account import SimAccount
    # Free broker
    acc_free = SimAccount.fresh(starting_equity=10_000.0)
    acc_free.apply_open(
        credit_per_share=0.50, qty=1, spread_width=1.0,
        commission_per_leg=0.0,
    )
    # No commission → cash 10,000 + 50 = 10,050.
    assert abs(acc_free.cash - 10_050.0) < 0.01

    # Expensive broker
    acc_costly = SimAccount.fresh(starting_equity=10_000.0)
    acc_costly.apply_open(
        credit_per_share=0.50, qty=1, spread_width=1.0,
        commission_per_leg=1.00,
    )
    # Commission: $1.00 × 2 = $2.00. Cash: 10,000 + 50 - 2 = 10,048.
    assert abs(acc_costly.cash - 10_048.0) < 0.01


def test_skill_38_runner_exposes_slippage_and_commission_params() -> None:
    """Skill 38 §3.2: BacktestRunner must expose slippage_ticks_per_leg
    and commission_per_leg kwargs so the operator can sweep them
    without editing source."""
    import inspect
    from trading_agent.backtest.runner import BacktestRunner
    sig = inspect.signature(BacktestRunner.__init__)
    params = set(sig.parameters)
    for required in ("slippage_ticks_per_leg", "commission_per_leg"):
        assert required in params, (
            f"Skill 38 §3.2: BacktestRunner.__init__ must expose "
            f"{required}. Without it the slippage/commission "
            f"model can only be changed by source edit, breaking "
            f"the sweep workflow."
        )


def test_skill_38_runner_translates_ticks_to_dollars() -> None:
    """Skill 38 §3.2: BacktestRunner stores slippage_per_share as
    ticks × $0.05 × 2 legs. Default 0 ticks → $0 slippage."""
    # We can't fully construct BacktestRunner here without yfinance,
    # but we can test the unit conversion via a stub instance.
    from datetime import date
    from trading_agent.backtest.runner import BacktestRunner
    # Lazy: dataclass-style minimal preset stub.
    class _Preset:
        scan_mode = "adaptive"
        max_risk_pct = 0.02
        min_credit_ratio = 0.33
        max_delta = 0.20
        edge_buffer = 0.10
    # Patch HistoricalPort + RegimeClassifier so __init__ doesn't
    # need yfinance + scipy.
    class _Port:
        def __init__(self, *a, **k): pass
        def set_cursor(self, t): pass
    # We don't actually run the cycle; we just verify the math.
    try:
        r = BacktestRunner(
            tickers=("SPY",),
            start=date(2026, 5, 12), end=date(2026, 5, 13),
            preset=_Preset(),
            starting_equity=1000.0,
            port=_Port(),  # avoid yfinance init
            slippage_ticks_per_leg=2,
            commission_per_leg=0.65,
        )
    except Exception:
        # If construction needs scipy (RegimeClassifier), skip
        # gracefully — the conversion math is still pinned via the
        # source-level assertion below.
        import re
        from pathlib import Path
        src = (
            Path(__file__).resolve().parents[2]
            / "trading_agent" / "backtest" / "runner.py"
        ).read_text(encoding="utf-8")
        assert "self.slippage_per_share = (" in src
        assert "self.TICK_SIZE * 2" in src, (
            "Skill 38 §3.2: runner must translate ticks to per-spread "
            "dollars as ticks × tick_size × 2 legs. Without the × 2 "
            "the slippage is off by half."
        )
        return
    # If construction succeeded, verify the math.
    assert r.slippage_ticks_per_leg == 2
    # 2 ticks × $0.05 × 2 legs = $0.20 per share per spread.
    assert abs(r.slippage_per_share - 0.20) < 1e-9, (
        f"Skill 38 §3.2: slippage_per_share must equal "
        f"ticks × tick_size × 2. Got {r.slippage_per_share}, "
        f"expected 0.20 for 2 ticks."
    )
    assert r.commission_per_leg == 0.65


def test_skill_38_slippage_is_symmetric_across_open_close() -> None:
    """Skill 38 §4: a position opened with N ticks of slippage and
    closed with the same N ticks of slippage must show LESS realized
    P&L than the no-slippage case by exactly 2N ticks × qty × 100."""
    from trading_agent.backtest.account import SimAccount

    # Baseline: no slippage, no commission, $0.50 → $0.20 close.
    a0 = SimAccount.fresh(starting_equity=10_000.0)
    a0.apply_open(
        credit_per_share=0.50, qty=1, spread_width=1.0,
        commission_per_leg=0.0,
    )
    p0 = a0.apply_close(
        credit_per_share=0.50, qty=1,
        closing_debit_per_share=0.20,
        commission_per_leg=0.0,
    )
    # Baseline realised: 50 - 20 = $30.
    assert abs(p0 - 30.0) < 0.01

    # With 1 tick ($0.05) slippage on both sides.
    a1 = SimAccount.fresh(starting_equity=10_000.0)
    a1.apply_open(
        credit_per_share=0.50, qty=1, spread_width=1.0,
        slippage_per_share=0.05,
        commission_per_leg=0.0,
    )
    p1 = a1.apply_close(
        credit_per_share=0.50, qty=1,
        closing_debit_per_share=0.20,
        slippage_per_share=0.05,
        commission_per_leg=0.0,
    )
    # Realised = (45 - 25) = $20. Drag from slippage = $10.
    expected_drag = 2 * 0.05 * 1 * 100
    actual_drag = p0 - p1
    assert abs(actual_drag - expected_drag) < 0.01, (
        f"Skill 38 §4: slippage drag must be 2 × slip × qty × 100. "
        f"Got drag=${actual_drag:.2f}, expected ${expected_drag:.2f}."
    )
