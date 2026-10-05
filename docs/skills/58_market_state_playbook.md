# Market Risk State & Playbook Table

> **One-line summary:** classifies the whole market once per cycle (NORMAL / CAUTION / DEFENSIVE / CAPITULATION / RECOVERY), gates new entries by it (size multiplier, allowed strategies, Wheel CSP pause), and tags every ticker with the trend × volatility × RSI playbook it calls for.
> **Source of truth:** [`trading_agent/market_state.py`](../../trading_agent/market_state.py)
> **Phase:** 2  •  **Group:** risk / regime
> **Depends on:** `11_six_regime_classifier.md`, `09_vix_zscore_inhibitor.md`, `40_long_term_options_evaluator.md`, `16_market_data_provider_routing.md`
> **Consumed by:** `agent.py` (Stage 2 gate, `_process_ticker` strategy gate, raw_signal fields), `executor_promote.py` (Wheel CSP pause), `mcp/tools/market.py:get_market_state`, `mcp/tools/strategy.py:wheel_screen`

---

## 1. Theory & Objective

The per-ticker regime (skill 11) answers "which direction is this ticker trending?" but not "is the whole market in a state where selling downside premium is dangerous?". Backlog §1 (written after week 1) asked for a market-wide overlay that sees a 10–20 % correction coming early enough to stop opening bull puts and cash-secured puts, shrink size, and — once the selling exhausts — re-enable bullish premium at half size. Sizes follow the backlog: 1.0 / 0.5 / 0.25 / 0 for NORMAL / CAUTION / DEFENSIVE / CAPITULATION, 0.5 in RECOVERY. It uses four inputs any options desk watches: SPY's trend against its 20/50/200-day averages, the VIX level, the VIX/VIX3M term structure (inversion = near-term fear), and breadth (share of equity ETFs on the watchlist above their 50-day). Rules are deterministic and ordered; the first match wins, so every decision is explainable from the reasons list. Backlog §6.2 adds the playbook table: the strategy family each trend × volatility × RSI combination calls for, flagged `implemented=False` when the agent lacks that tool yet (debit spreads, calendars, the bounce bull put — backlog §6.3–6.5), so the journal measures how often the market asked for a missing tool. Exits are never gated.

## 2. Mathematical Formula

```text
inputs:  P = SPY close, S20/S50/S200 = SPY simple moving averages, RSI = SPY RSI-14
         V = VIX, V3 = VIX3M, T = V / V3 (None when either missing)
         B = fraction of breadth tickers (watchlist equities, not SPY) with close > own SMA-50

missing P / S50 / S200            → CAUTION (fail safe)
CAPITULATION  V ≥ 35  or  T ≥ 1.10  or  (RSI ≤ 25 and V ≥ 28)
DEFENSIVE     P < S200  and  (V ≥ 22  or  T ≥ 1.00  or  B < 0.40)
(prior ∈ DEFENSIVE/CAPITULATION/RECOVERY)
  NORMAL      P > S50 and V < 20
  RECOVERY    P > S20
CAUTION       P < S50  or  V ≥ 20  or  T ≥ 0.95  or  B < 0.50
(prior = CAUTION, hysteresis) stay CAUTION unless P ≥ 1.01·S50, V < 19, T ≤ 0.93, B ≥ 0.55
NORMAL        otherwise

gate:  state         size×  allowed new spreads                    new CSPs
       NORMAL        1.0    bull put, bear call, IC, IB            yes
       CAUTION       0.5    bear call, IC, IB                      no
       DEFENSIVE     0.25   bear call                              no
       CAPITULATION  0.0    none                                   no
       RECOVERY      0.5    bull put, IC, IB                       yes
       (Mean Reversion Spread gated as bull put / bear call by its sold side)

effective max_risk_pct = preset.max_risk_pct × size×   (RiskManager and executor sizer)

playbook(regime, vol_rank, RSI):  vol buckets high ≥ 50, mid 30–50, low < 30
  mean_reversion          → mean_reversion
  bearish & RSI < 30      → bounce_bull_put (vol ≥ 30) | wait_for_stabilization
  bullish                 → bull_put (vol ≥ 30) | call_debit
  bearish                 → bear_call (vol ≥ 30) | put_debit
  sideways                → iron_condor (vol ≥ 30) | calendar
```

## 3. Reference Python Implementation

```python
# trading_agent/market_state.py:101-107
GATES: Dict[str, StateGate] = {
    NORMAL:       StateGate(1.0, ALL_SPREADS, True),
    CAUTION:      StateGate(0.5, frozenset({BEAR_CALL, IRON_CONDOR, IRON_BUTTERFLY}), False),
    DEFENSIVE:    StateGate(0.25, frozenset({BEAR_CALL}), False),
    CAPITULATION: StateGate(0.0, frozenset(), False),
    RECOVERY:     StateGate(0.5, frozenset({BULL_PUT, IRON_CONDOR, IRON_BUTTERFLY}), True),
}
```

```python
# trading_agent/market_state.py:173-240
def classify_market_state(inp: MarketInputs, prior_state: Optional[str] = None,
                          cfg: MarketStateConfig = MarketStateConfig()) -> MarketStateResult:
    """Deterministic rules, checked in order; the first that matches wins."""
    def result(state: str, reasons: List[str]) -> MarketStateResult:
        return MarketStateResult(state, tuple(reasons), inp, GATES[state])

    p, s20, s50, s200 = inp.spy_price, inp.spy_sma20, inp.spy_sma50, inp.spy_sma200
    if p is None or s50 is None or s200 is None:
        return result(CAUTION, ["spy_data_unavailable — failing safe to CAUTION"])
    vix, tr, br, rsi = inp.vix, inp.term_ratio, inp.breadth, inp.spy_rsi

    cap = []
    if vix is not None and vix >= cfg.capitulation_vix:
        cap.append(f"VIX {vix:.1f} ≥ {cfg.capitulation_vix:g}")
    if tr is not None and tr >= cfg.capitulation_term_ratio:
        cap.append(f"VIX/VIX3M {tr:.2f} ≥ {cfg.capitulation_term_ratio:g}")
    if (rsi is not None and vix is not None and rsi <= cfg.capitulation_rsi
            and vix >= cfg.capitulation_rsi_vix):
        cap.append(f"SPY RSI {rsi:.1f} ≤ {cfg.capitulation_rsi:g} with VIX {vix:.1f}")
    if cap:
        return result(CAPITULATION, cap)

    defn = []
    if p < s200:
        if vix is not None and vix >= cfg.defensive_vix:
            defn.append(f"SPY < SMA-200 and VIX {vix:.1f} ≥ {cfg.defensive_vix:g}")
        if tr is not None and tr >= cfg.defensive_term_ratio:
            defn.append(f"SPY < SMA-200 and VIX/VIX3M {tr:.2f} ≥ {cfg.defensive_term_ratio:g}")
        if br is not None and br < cfg.defensive_breadth:
            defn.append(f"SPY < SMA-200 and breadth {br:.0%} < {cfg.defensive_breadth:.0%}")
    if defn:
        return result(DEFENSIVE, defn)

    if prior_state in (DEFENSIVE, CAPITULATION, RECOVERY):
        if p > s50 and (vix is None or vix < cfg.recovery_exit_vix):
            return result(NORMAL, [f"recovered: SPY > SMA-50 and VIX "
                                   f"{'n/a' if vix is None else f'{vix:.1f}'} < {cfg.recovery_exit_vix:g}"])
        if s20 is not None and p > s20:
            return result(RECOVERY, [f"after {prior_state}: SPY reclaimed SMA-20 "
                                     f"({p:.2f} > {s20:.2f})"])

    caution = []
    if p < s50:
        caution.append(f"SPY {p:.2f} < SMA-50 {s50:.2f}")
    if vix is not None and vix >= cfg.caution_vix:
        caution.append(f"VIX {vix:.1f} ≥ {cfg.caution_vix:g}")
    if tr is not None and tr >= cfg.caution_term_ratio:
        caution.append(f"VIX/VIX3M {tr:.2f} ≥ {cfg.caution_term_ratio:g}")
    if br is not None and br < cfg.caution_breadth:
        caution.append(f"breadth {br:.0%} < {cfg.caution_breadth:.0%}")
    if caution:
        return result(CAUTION, caution)
    if prior_state == CAUTION:
        hold = []
        if p < s50 * (1 + cfg.caution_exit_sma_margin):
            hold.append(f"SPY within {cfg.caution_exit_sma_margin:.0%} of SMA-50")
        if vix is not None and vix >= cfg.caution_vix - cfg.caution_exit_vix_margin:
            hold.append(f"VIX {vix:.1f} not yet < {cfg.caution_vix - cfg.caution_exit_vix_margin:g}")
        if tr is not None and tr > cfg.caution_exit_term_ratio:
            hold.append(f"VIX/VIX3M {tr:.2f} not yet ≤ {cfg.caution_exit_term_ratio:g}")
        if br is not None and br < cfg.caution_exit_breadth:
            hold.append(f"breadth {br:.0%} not yet ≥ {cfg.caution_exit_breadth:.0%}")
        if hold:
            return result(CAUTION, ["hysteresis: " + "; ".join(hold)])
    return result(NORMAL, ["no risk condition met"])


# ---------------------------------------------------------------------------
```

```python
# trading_agent/market_state.py:256-283
def playbook_for(regime: str, vol_rank: Optional[float], rsi: Optional[float]) -> Playbook:
    """Trend × volatility × RSI extreme → playbook. ``vol_rank`` is the
    regime classifier's ``iv_rank`` (a realized-volatility percentile)."""
    r = (regime or "").lower()
    v = vol_rank if vol_rank is not None and not math.isnan(vol_rank) else None
    rich = v is not None and v >= VOL_LOW         # premium worth selling
    vb = ("unknown" if v is None else "high" if v >= VOL_HIGH
          else "mid" if v >= VOL_LOW else "low")

    def pb(name: str) -> Playbook:
        return Playbook(name, name in _IMPLEMENTED,
                        f"{r or 'unknown'} trend, {vb} volatility"
                        + (f", RSI {rsi:.0f}" if rsi is not None else ""))

    if r == "mean_reversion":
        return pb("mean_reversion")
    if r == "bearish" and rsi is not None and rsi < RSI_OVERSOLD:
        return pb("bounce_bull_put" if rich else "wait_for_stabilization")
    if r == "bullish":
        return pb("bull_put" if rich else "call_debit")
    if r == "bearish":
        return pb("bear_call" if rich else "put_debit")
    if r == "sideways":
        return pb("iron_condor" if rich else "calendar")
    return pb("none")


# ---------------------------------------------------------------------------
```

```python
# trading_agent/market_state.py:382-396
def csp_pause_reason(snapshot: Optional[Dict[str, Any]],
                     max_age_s: float = CSP_PAUSE_MAX_AGE_S) -> Optional[str]:
    """Reason a new cash-secured put is paused by the last market-state
    snapshot (``read_state()``), or ``None``. A missing or stale snapshot
    does not pause (the overlay may be disabled or the agent not running)."""
    if not snapshot:
        return None
    age = snapshot.get("age_seconds")
    if age is not None and age > max_age_s:
        return None
    if (snapshot.get("gate") or {}).get("allow_new_csp", True):
        return None
    return f"market_state_{snapshot.get('state')}_pauses_new_csp"
```

```python
# trading_agent/agent.py:2208-2251
    def _update_market_state(self, tickers, account_balance: float = 0.0) -> None:
        """Skill 58: classify the market once per cycle, scale max_risk_pct
        on the RiskManager and the executor, and persist the snapshot read
        by the MCP ``get_market_state`` tool and ``wheel_screen``. The
        previous state is read back from the snapshot (the process restarts
        every cycle), which is what lets RECOVERY follow DEFENSIVE."""
        if not getattr(self.preset, "market_state_enabled", False):
            self._market_state = None
            self._apply_risk_multiplier(1.0)
            return
        prior = market_state.read_state()
        prior_state = prior.get("state") if prior else None
        try:
            inputs = market_state.compute_inputs(
                lambda t: self.data_provider.fetch_historical_prices(t, period_days=200),
                market_state.breadth_universe(tickers),
                market_state.yfinance_level,
            )
            result = market_state.classify_market_state(inputs, prior_state)
        except Exception as exc:
            logger.warning("Market state unavailable (%s) — failing safe to CAUTION", exc)
            self._exception_monitor.record(
                source="agent._update_market_state", exc=exc,
                message="market state classification failed — CAUTION fallback")
            result = market_state.classify_market_state(
                market_state.MarketInputs(), prior_state)
        self._market_state = result
        self._apply_risk_multiplier(result.gate.size_multiplier)
        logger.info("MARKET STATE: %s (size ×%.1f) — %s", result.state,
                    result.gate.size_multiplier, "; ".join(result.reasons))
        try:
            market_state.write_state(result, extra={"account_balance": account_balance})
        except OSError as exc:
            logger.warning("Market state snapshot write failed: %s", exc)
        if result.state != prior_state:
            try:
                self.journal_kb.log_signal(
                    ticker="__market__", action="market_state",
                    price=result.inputs.spy_price or 0.0,
                    raw_signal={**result.to_dict(), "prior_state": prior_state},
                )
            except Exception as exc:  # noqa: skill-34-exempt — state-change journal row is best-effort; snapshot already written
                logger.warning("Market state journal row failed: %s", exc)
```

```python
# trading_agent/agent.py:2324-2332
            ms_block = market_state.gate_failure(
                self._market_state, plan.strategy_name,
                [l.option_type for l in plan.legs if l.action == "sell"])
            if ms_block:
                logger.info("[%s] %s", ticker, ms_block)
                verdict = dataclasses.replace(
                    verdict, approved=False,
                    checks_failed=list(verdict.checks_failed) + [ms_block],
                    summary=f"{verdict.summary} | {ms_block}")
```

## 4. Edge Cases / Guardrails

- **SPY history unavailable** — `compute_inputs` returns `spy_price=None`; the classifier fails safe to CAUTION (`spy_data_unavailable`). An exception anywhere in the overlay is recorded by the ExceptionMonitor and also yields CAUTION; it never aborts the cycle.
- **VIX / VIX3M / breadth missing** — each input is fetched in its own try/except; a missing input is `None` (sentinel) and its rules are skipped, never read as zero. `term_ratio` is `None` unless both VIX and VIX3M are present.
- **Non-equity tickers in breadth** — TLT, GLD, GDX, etc. often rise in an equity sell-off; `breadth_universe` drops them (and SPY) so breadth is not flattered exactly when it matters.
- **Process restarts every cycle** — the prior state (for RECOVERY and the CAUTION hysteresis) is read back from `trade_journal/market_state.json`, written atomically (temp + rename) each cycle.
- **NORMAL↔CAUTION churn** — without hysteresis the 2019–2026 replay flipped 159 times (median CAUTION run 2 days); the exit margins cut total state changes from 34 to 26 per year.
- **CAPITULATION** — Stage 2 returns early with `skipped_reason=market_state_CAPITULATION`; Stage 1 exits already ran. `get_market_state` adds a note rather than a hedge (puts are dearest then).
- **Strategy blocked** — recorded as a failed risk check (`market_state_<STATE>_blocks_<strategy>`), so the journal reason reads `risk: market_state_…` and the reject histogram counts it.
- **Size multiplier** — applied to both `RiskManager.max_risk_pct` and the executor's `max_risk_pct` every cycle from the preset base, so the validator and the sizer never disagree (CLAUDE.md invariant #4 discipline) and a multiplier never compounds.
- **Wheel CSP pause** — `wheel_screen` drops CSP rows (diagnostic `market_state_<STATE>_pauses_new_csp`) and `executor_promote` refuses a staged CSP. A snapshot older than 4 days (agent not running) does not pause; covered calls are never paused.
- **Overlay disabled** (`market_state_enabled=False`) — no snapshot is read or written by the gates, max_risk_pct stays at the preset value, behaviour is pre-2026-10-05.
- **Journal volume** — a `__market__` / `market_state` row is written only when the state changes, not every cycle.
- **Threshold validation** (`scripts/research/validate_market_state.py`, SPY/VIX/VIX3M daily, 2019-06 → 2026-10, 1,845 days): P(SPY falls ≥ 5 % within 30 days) = 8.5 % in NORMAL, 18.9 % CAUTION, 31.1 % DEFENSIVE, 29.1 % CAPITULATION, 24.7 % RECOVERY. First CAUTION came at −2.2 % to −2.6 % from the high in the 2020, 2022, 2024 and 2025 drawdowns; DEFENSIVE at −7.5 % (2022) and −8.5 % (2025); CAPITULATION on 2024-08-05 and 2025-04-04. Weak spot: 2023 regional banks (CAUTION only once SPY was already 14.5 % below its 2022 high). RECOVERY's 24.7 % tail is why it trades at half size.
- **VIX3M source** — yfinance `^VIX3M` daily bar (the backlog suggested Schwab `$VIX3M.X`; the data server has no index-quote route yet). Unavailable → term-structure rules skipped.
- **Daily vs intraday inputs** — SPY averages use daily closes through the prior session (cached 4 h); VIX/VIX3M use the latest daily bar, which is live during the session. The CAUTION hysteresis absorbs intraday VIX noise around 20.

## 5. Cross-References

- `11_six_regime_classifier.md` — per-ticker regime and `iv_rank` feed `playbook_for`.
- `09_vix_zscore_inhibitor.md` — per-ticker VIX z-score inhibitor; this skill uses VIX as one input of a market-wide state.
- `40_long_term_options_evaluator.md` — Wheel CSP pause in `wheel_screen` / promote.
- `48_claude_code_mcp_surface.md` — `get_market_state` read-only tool.
- `55_pending_orders_promotion.md` — `check_wheel_order` market-state check.

---

*Last verified against repo HEAD on 2026-10-05.*
