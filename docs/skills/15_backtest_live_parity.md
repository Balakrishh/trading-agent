# Backtest ↔ live parity (the `trading_agent.backtest` package)

> **One-line summary:** A historical-replay simulator that drives the *exact same* live-agent primitives — `decision_engine.decide()`, `RiskManager.evaluate()`, `executor.calculate_position_qty()`, and `PositionMonitor`-equivalent exit rules — against synthesized option chains and yfinance bars, with a hybrid intraday-vs-daily cadence picked by window length and a VIX-proxy IV scaling for mid-life mark-to-market.
> **Source of truth:** [`trading_agent/backtest/__init__.py`](../../trading_agent/backtest/__init__.py); [`runner.py`](../../trading_agent/backtest/runner.py); [`cycle.py`](../../trading_agent/backtest/cycle.py); [`sim_position.py`](../../trading_agent/backtest/sim_position.py); [`account.py`](../../trading_agent/backtest/account.py); [`historical_port.py`](../../trading_agent/backtest/historical_port.py); [`synthetic_chain.py`](../../trading_agent/backtest/synthetic_chain.py); [`clock.py`](../../trading_agent/backtest/clock.py); [`black_scholes.py`](../../trading_agent/backtest/black_scholes.py); [`trading_agent/streamlit/backtest_ui.py`](../../trading_agent/streamlit/backtest_ui.py).
> **Phase:** 2  •  **Group:** architecture
> **Depends on:** [03 Credit/Width floor](03_credit_to_width_floor.md), [05 EV per dollar risked](05_ev_per_dollar_risked.md), [13 Preset system & hot-reload](13_preset_system_hot_reload.md), [14 Adaptive vs static scan modes](14_adaptive_vs_static_scan_modes.md).
> **Consumed by:** `trading_agent.streamlit.backtest_ui.render_backtest_ui` — the Backtesting tab in the dashboard.

---

## 1. Theory & Objective

The pre-2026-05 backtester was a 4 057-line single-file `Backtester` class that re-implemented its own scoring, sigma-strike heuristic, exit logic, and account ledger. That created two sources of truth — the live agent and the backtest — which drifted silently. The CI invariant scanner and the C/W floor formula triple-write existed precisely because that drift had been costing real money in misleading backtest reports.

The replacement, `trading_agent.backtest`, deliberately owns *zero* trading logic. It owns simulation infrastructure (a calendar-aware clock, a cursor-bound historical data port, a Black-Scholes synthesizer for option chains, a cash + open-market-value ledger, and a position record with VIX-proxy IV scaling for marks). Everything that decides *what* trade to put on or *whether* to exit it is borrowed from the live agent: the `decide()` engine picks strikes, the `RiskManager` gates the plan, `calculate_position_qty()` sizes it, and `SimPosition.evaluate_exit()` mirrors `PositionMonitor._check_exit` rule-for-rule. Order acceptance is assumed: if a plan clears the eight risk guardrails, the simulator books it at the synthetic mid minus a fill haircut and re-marks every event.

The hybrid cadence reflects yfinance's hard 30-day window for 5-minute bars: when the requested backtest window is ≤ 29 days from today, the runner emits intraday `intraday_decision` events every 5 minutes during RTH so entry decisions are scored at the same cadence the live agent uses; for longer windows it collapses to one decision per RTH open. Mid-life mark-to-market always happens daily — the synthetic Black-Scholes mark is sensitive enough that intraday remarks add noise without information.

VIX-proxy IV scaling solves the synthetic-chain mark-stability problem. At entry the position records `(σ_entry, vix_entry)`. On every later mark, the runner queries the historical VIX close and rescales: `σ_t = σ_entry × (vix_t / vix_entry)` clipped to `[0.01, 5.0]`. This means a vol regime shift (VIX 15 → 30) doubles the synthetic mark, propagating into the exit-rule check the same way the live agent's NBBO mid would. Without this scaling, every position would mark at near-flat-to-theta until expiry, hiding all the regime-driven losses the live agent actually takes.

The whole package fits in nine source files (~1 800 lines) plus six tests (~600 lines). The Streamlit tab shrank from 4 057 lines to 475 — an 88 % reduction — because nothing in the UI does any computation; it builds a `BacktestRunner`, calls `.run()`, and renders the resulting `BacktestResult`.

## 2. Mathematical Formula

```text
─────────────────────────────────────────────────────────────────────
Cadence selection (runner.py:_pick_cadence)
─────────────────────────────────────────────────────────────────────
  INTRADAY_LOOKBACK_LIMIT_DAYS = 29

  cadence(today) =
    "intraday"  if (today - run_start).days ≤ 29
    "daily"     otherwise

  intraday cadence emits, per trading day:
    1 × day_open
    78 × intraday_decision  (09:30 → 16:00 every 5 min)
    1 × daily_mark
    1 × day_close

  daily cadence collapses to:
    1 × day_open + 1 × daily_mark + 1 × day_close

─────────────────────────────────────────────────────────────────────
VIX-proxy IV scaling (sim_position.py:remark)
─────────────────────────────────────────────────────────────────────
  σ_t = clip(σ_entry × (vix_t / vix_entry),  0.01,  5.0)

  fallbacks:
    vix_t is None        → σ_t = σ_entry         (no rescale)
    vix_entry == 0       → σ_t = σ_entry         (scaling disabled)
    spot ≤ 0             → skip the mark         (bad data)
    expiration reached   → mark = max(0, K_short - spot) - max(0, K_long - spot)
                           (intrinsic spread debit; bull_put case shown)

─────────────────────────────────────────────────────────────────────
Account ledger identities (account.py:apply_open / apply_close)
─────────────────────────────────────────────────────────────────────
  At open:
    cash             += credit × qty × 100 - 0.65 × 2 × qty
    open_market_value -= credit × qty × 100
    equity            = cash + open_market_value
                      = starting - 0.65 × 2 × qty           (drops only by commission)

  At close:
    realised_pnl      = (credit_open - debit_close) × qty × 100
                       - 0.65 × 2 × qty                     (round-trip commission)
    cash             += realised_pnl
    open_market_value += credit × qty × 100  (retire the slice)

─────────────────────────────────────────────────────────────────────
Synthetic chain construction (synthetic_chain.py:build_chain_slice)
─────────────────────────────────────────────────────────────────────
  STRIKE_STEP            = 1.0       (project-standard $1 grid)
  SYNTHETIC_HALF_SPREAD  = 0.05      (5 ¢ on either side of BS mid)

  For each (target_Δ, width_%) in (preset.delta_grid × preset.width_grid_pct):
    K_short = nearest STRIKE_STEP-aligned strike with |Δ_BS| ≈ target_Δ
    K_long  = K_short ± width_% × spot   (snapped to STRIKE_STEP)
    mid     = bs_price(spot, K, t_years, σ, option_type)
    delta   = bs_delta(spot, K, t_years, σ, option_type)
    bid     = mid - 0.05
    ask     = mid + 0.05

  The 5 ¢ half-spread is chosen so that _quote_credit (the engine's
  pricing helper) recovers the original BS mid back out:
    _quote_credit(short_bid, short_ask, long_bid, long_ask)
      = (short_mid - long_mid) - 0.02   (2 ¢ fill haircut)
```

## 3. Reference Python Implementation

```python
# trading_agent/backtest/runner.py — core orchestrator
class BacktestRunner:
    def __init__(self, *, tickers, start, end, preset,
                 starting_equity=100_000.0,
                 risk_manager=None, port=None, progress_callback=None):
        self.tickers, self.start, self.end = tuple(tickers), start, end
        self.preset = preset
        self.port = port or HistoricalPort()
        # Default RM mirrors the preset so adaptive-floor parity holds.
        self.risk_manager = risk_manager or RiskManager(
            max_risk_pct=getattr(preset, "max_risk_pct", 0.02),
            min_credit_ratio=getattr(preset, "min_credit_ratio", 0.33),
            max_delta=getattr(preset, "max_delta", 0.20),
            delta_aware_floor=(getattr(preset, "scan_mode", "static") == "adaptive"),
            edge_buffer=getattr(preset, "edge_buffer", 0.10),
        )
        self._classifier = RegimeClassifier(data_provider=None)

    def _pick_cadence(self, today):
        return ("intraday"
                if (today - self.start).days <= INTRADAY_LOOKBACK_LIMIT_DAYS
                else "daily")
```

```python
# trading_agent/backtest/sim_position.py — VIX-proxy IV scaling
def remark(self, *, t, spot, vix_t, r=0.0):
    """Re-mark the spread at time t, with VIX-proxy IV scaling."""
    if spot <= 0:
        logger.debug("[%s] SimPosition.remark: non-positive spot, skipping",
                     self.ticker)
        return
    if self.vix_entry > 0 and vix_t is not None and vix_t > 0:
        sigma_t = max(0.01, min(5.0, self.sigma_entry * (vix_t / self.vix_entry)))
    else:
        sigma_t = self.sigma_entry
    self.sigma_current = sigma_t
    self.vix_current   = vix_t

    t_years = max(0.0, (self.expiration - t.date()).days / 365.0)
    if t_years <= 0:
        # At/past expiration: use intrinsic spread debit
        self.current_mark = max(0.0, self._intrinsic_at(spot))
    else:
        short_mid = bs_price(spot, self.short_strike, t_years, sigma_t,
                             option_type=("put" if self.side == "bull_put" else "call"),
                             r=r)
        long_mid  = bs_price(spot, self.long_strike, t_years, sigma_t,
                             option_type=("put" if self.side == "bull_put" else "call"),
                             r=r)
        self.current_mark = max(0.0, short_mid - long_mid)
    self.current_t = t
```

```python
# trading_agent/backtest/cycle.py — the live↔backtest seam
def run_one_cycle(*, ticker, t, port, preset, account,
                  classifier, risk_manager) -> CycleOutcome:
    # PERCEIVE — pull historical bars (cursor-bound, no look-ahead)
    daily = port.fetch_underlying_daily(ticker)
    spot  = float(daily["Close"].iloc[-1])

    # CLASSIFY — use the live regime classifier on historical data
    regime, reasoning = _classify_from_history(classifier, daily)

    # PLAN — synth chain across preset DTE grid, call decide()
    side = _side_for_regime(regime)
    cfg  = build_chain_config_from_preset(side, preset)
    chain_slices = [
        build_chain_slice(ticker=ticker, side=side, spot=spot,
                          sigma_annual=_atm_iv_proxy(daily["Close"]),
                          now=t.date(), expiration=exp, cfg=cfg)
        for exp in _resolve_expirations(t.date(), preset)
    ]
    output = decide(DecisionInput(side=side, chain_slices=chain_slices,
                                  preset=preset),
                    max_candidates=1)
    if not output.candidates:
        return CycleOutcome(..., status="no_candidate", ...)

    plan = _candidate_to_spread_plan(ticker, output.candidates[0],
                                     regime, reasoning)

    # RISK — same 8-guardrail evaluator the live agent uses
    verdict = risk_manager.evaluate(plan, account_balance=account.equity, ...)
    if not verdict.approved:
        return CycleOutcome(..., status="risk_rejected",
                            reason=verdict.failure_reasons[0])

    # EXECUTE — same sizing, same fill-haircut model
    qty = calculate_position_qty(plan, account.equity,
                                 risk_manager.max_risk_pct)
    account.apply_open(credit_per_share=plan.credit_per_share,
                       qty=qty, spread_width=plan.spread_width)
    return CycleOutcome(..., status="opened", ...)
```

```python
# trading_agent/streamlit/backtest_ui.py — UI shim that wires it all
def render_backtest_ui() -> None:
    # Settings panel → run BacktestRunner.run() → render BacktestResult.
    # No business logic in this file.
    runner = BacktestRunner(tickers=tickers, start=start_date, end=end_date,
                            preset=preset, starting_equity=starting_equity,
                            risk_manager=risk_manager, port=HistoricalPort())
    result = runner.run()
    st.plotly_chart(equity_curve_chart(_equity_curve_to_df(result.equity_curve)))
    closed_trades_table(result.closed_trades)

def _preview_decision(ticker, preset_name, side="bull_put", ...):
    # Diagnostic helper — calls decide() once on a synth chain so the
    # operator can preview engine picks without booting a full backtest.
    # ALSO satisfies CI invariant #3: this file MUST contain a decide(
    # call so the unified path stays alive.
    out = decide(DecisionInput(side=side, chain_slices=chain_slices,
                               preset=preset), max_candidates=3)
```

## 4. Edge Cases / Guardrails

- **Cursor enforcement is non-negotiable.** `HistoricalPort` raises `RuntimeError("cursor not set")` on any data fetch before `set_cursor(t)` has been called, and silently filters every returned frame to rows with index ≤ cursor. Any future-data leak would invalidate the entire run; the test suite pins this with `test_cursor_filters_future_rows`.
- **5-minute bars exist only for the last ~30 days.** This is a yfinance limitation, not a project choice. The runner's `_pick_cadence` clamps to `daily` once `(today - start).days > 29`. UIs should warn when a user picks an intraday window outside the 30-day envelope — the dashboard caption does this.
- **Adaptive presets MUST set `delta_aware_floor=True` on the RiskManager.** The default `BacktestRunner` constructor wires this from `preset.scan_mode == "adaptive"` so a scanner-picked candidate cannot be vetoed by a stricter static floor at risk time. If you instantiate your own `RiskManager` for a backtest, you must set this flag yourself or you'll get phantom rejections that don't reproduce in live. See [03 C/W floor](03_credit_to_width_floor.md).
- **VIX-proxy is not a substitute for live IV.** The scaling captures vol regime *level* but ignores skew, term structure, and event premia. A 50-strike 14-DTE put on earnings day would mark closer to its true value if we had vendor IV; the proxy understates it. This is acceptable for portfolio-level economics but should not be cited as an upper-bound on per-trade exit timing.
- **Synthetic chains use a $1 strike grid.** `STRIKE_STEP = 1.0` matches the project-standard grid from `ChainScanner._infer_grid_step`. Tickers whose actual chains snap to $0.50 (some single-name stocks) will see the synth chain be slightly less granular than reality. Effect on outcome: negligible — the engine's strike picker snaps the same way.
- **Force-close at end of window.** Any position still open when `BacktestRunner.run()` exhausts the calendar is force-closed at its final synthetic mark with `exit_signal=ExitSignal.EXPIRED` and `exit_reason="run_end_force_close"`. Realised P&L includes this final mark; the equity curve does not extrapolate.
- **`run_one_cycle` wraps the whole pipeline in try/except.** A bug in any stage produces a `CycleOutcome(status="no_data", reason=str(exc))` row instead of crashing the run. The cycle outcomes table in the UI surfaces these rows so the operator can see *why* a cycle didn't open.
- **`SimAccount.apply_open(qty=0)` is a no-op.** When `calculate_position_qty` returns 0 (account too small for one contract at preset risk), the runner does not attempt to book; the outcome is `status="risk_rejected"` with a `qty_zero` reason. No commissions are charged in this path.
- **Commission of $0.65/leg × 4 legs round-trip is hard-coded.** It matches Alpaca's $0.65/contract option commission and is pinned in `account.py:COMMISSION_PER_LEG`. Brokers with different fee structures need a fork or a constructor arg — currently no preset field for it. Consider promoting to `PresetConfig` if this ever varies in production.
- **CI invariant #3 requires a `decide(` call in `streamlit/backtest_ui.py`.** The `_preview_decision` helper carries this invariant. Removing or renaming it is a CI failure (`scripts/checks/scan_invariant_check.py`). The runner's call inside `cycle.py` does *not* satisfy the invariant — the scanner walks the AST of the UI file specifically so a UI rewrite that bypasses `decide()` is caught.

## 5. Cross-References

- [03 Credit/Width floor](03_credit_to_width_floor.md) — the formula `|Δ| × (1 + edge_buffer)` written identically in three places; the backtester relies on the triple-wire by passing `delta_aware_floor=True` to `RiskManager` for adaptive presets.
- [05 EV per dollar risked](05_ev_per_dollar_risked.md) — the scoring function the backtester inherits via `decide()` → `_score_candidate_with_reason`.
- [13 Preset system & hot-reload](13_preset_system_hot_reload.md) — same `PresetConfig` library; a backtest run names the preset and `load_active_preset()` materializes it.
- [14 Adaptive vs static scan modes](14_adaptive_vs_static_scan_modes.md) — the backtester branches on `preset.scan_mode` exactly the same way `agent.py` does; this is the critical seam where parity is enforced.

---

*Last verified against repo HEAD on 2026-05-04.*
