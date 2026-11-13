# Vol-Band Bull Put and the Credit-Vertical Proximity Switch

> **One-line summary:** when no other plan is valid, sell a 0.25Δ / ~21 DTE bull put spread on SPY / QQQ / IWM while the vol rank is in [50, 80), at the widest width ≤ 2 % of spot that fits the risk cap, judged by its own C/W floor; and optionally drop the 1 % strike-proximity close for credit verticals.
> **Source of truth:** [`trading_agent/strategy.py:518-591`](../../trading_agent/strategy.py), [`trading_agent/position_monitor.py:1181-1193`](../../trading_agent/position_monitor.py)
> **Phase:** 2  •  **Group:** strategy / risk
> **Depends on:** `03_credit_to_width_floor.md`, `58_market_state_playbook.md`, `30_profit_target_management.md`, `13_preset_system_hot_reload.md`
> **Consumed by:** `StrategyPlanner._plan_body` (routing), `RiskManager.evaluate` / `OrderExecutor._recheck_live_economics` (plan floor), `PositionMonitor._check_exit` (exits), `TradingAgent._trend_filter_block` (filter exemption)

---

## 1. Theory & Objective

On real SPY / QQQ / IWM quotes 2008–2025 (`docs/plans/research_credit_vol.md` §2, §5) the agent's own credit scan almost never sells a put spread: the delta-aware C/W floor (credit / width ≥ |Δ|) rejects nearly every candidate, and the ~2 % that pass are the worst trades. A fixed 0.25Δ, 21 DTE, 2 %-wide bull put held one at a time, entered only at vol rank 50–80 and without the 1 % strike-proximity close, made +0.024 per $ risked (88 % winners, about +1.6 % a year at 2 % risk, max drawdown 8.5 %); the 95 % interval [−0.004, +0.049] still includes zero and the band was chosen in-sample, so it runs on paper as a forward test. It is a fallback: the call debits on SPY / QQQ (the larger measured edge) keep priority. Separately, the 1 % proximity close cut 0.25Δ put spreads from 86 % to 66 % winners and cost ~0.008 per $ in every comparison, so `credit_strike_proximity_exit` can turn it off for credit verticals.

## 2. Mathematical Formula

```text
eligible     ⇔  vol_band_put_enabled ∧ ticker ∈ vol_band_put_tickers
                ∧ min_rank ≤ iv_rank < max_rank           (regime iv_rank, 0–100)
                ∧ the regular plan is invalid and not a deliberate no-trade
expiry       =  weekly nearest vol_band_put_dte (± dte_window_days, earnings-capped)
short        =  put with |Δ| closest to vol_band_put_short_delta
long         =  lowest listed strike K < K_short with
                  K_short − K ≤ vol_band_put_width_pct × spot
                  credit_nat = bid_short − ask_long > 0
                  credit_nat / width ≥ vol_band_put_min_cw
                  (width − credit_nat) × 100 ≤ max_loss_budget
                (strikes ascend → the first that fits is the widest)
C/W floor    =  plan.min_cw (= vol_band_put_min_cw) instead of |Δ| × (1 + edge_buffer)
```

## 3. Reference Python Implementation

```python
# trading_agent/strategy.py:518-591
    def _with_vol_band_put(self, ticker: str, analysis: RegimeAnalysis,
                           plan: SpreadPlan) -> SpreadPlan:
        """Return ``plan`` when valid or a deliberate no-trade (empty
        strategy name); otherwise try the vol-band bull put, appending
        its reason to the original rejection when it finds nothing."""
        if plan.valid or not plan.strategy_name:
            return plan
        vb = self._plan_vol_band_put(ticker, analysis)
        if vb is None:
            return plan
        if vb.valid:
            return vb
        plan.rejection_reason = f"{plan.rejection_reason}; vol-band put {vb.rejection_reason}"
        return plan

    def _plan_vol_band_put(self, ticker: str, analysis: RegimeAnalysis) -> Optional[SpreadPlan]:
        """A |Δ| ``vol_band_put_short_delta`` put spread ~``vol_band_put_dte``
        out, widest listed width ≤ ``vol_band_put_width_pct`` of spot whose
        natural max loss fits the per-trade budget. None when the playbook
        is off, the ticker is not listed, or the vol rank is outside the band."""
        from trading_agent.chain_scanner import _leg_spread_too_wide
        pre = getattr(self, "preset", None)
        if pre is None or not getattr(pre, "vol_band_put_enabled", False):
            return None
        if ticker not in tuple(pre.vol_band_put_tickers):
            return None
        rank = getattr(analysis, "iv_rank", None)
        if rank is None or not (pre.vol_band_put_min_rank <= float(rank) < pre.vol_band_put_max_rank):
            return None
        name = "Bull Put Spread"
        expiration = self._expiration_near(int(pre.vol_band_put_dte))
        spot = float(getattr(analysis, "current_price", 0.0) or 0.0)
        puts = [c for c in (self.data.fetch_option_chain(ticker, expiration, "put") or [])
                if c.get("bid", 0) > 0 and c.get("ask", 0) > 0 and c.get("delta") is not None]
        if not puts or spot <= 0:
            return self._empty_plan(ticker, name, analysis, expiration, "no put chain")

        def wide(c):
            return _leg_spread_too_wide(c["bid"], c["ask"], pre.max_leg_spread_cents,
                                        pre.max_leg_spread_pct_mid)

        sold = min(puts, key=lambda c: abs(abs(c["delta"]) - pre.vol_band_put_short_delta))
        if wide(sold):
            return self._empty_plan(ticker, name, analysis, expiration,
                                    f"short {sold['strike']:g} leg quotes too wide")
        budget = self.max_loss_budget
        max_width = pre.vol_band_put_width_pct * spot
        best = None
        for bought in sorted((c for c in puts if c["strike"] < sold["strike"]),
                             key=lambda c: c["strike"]):
            width = sold["strike"] - bought["strike"]
            if width > max_width + 1e-9 or wide(bought):
                continue
            credit = sold["bid"] - bought["ask"]
            if credit <= 0 or credit / width < pre.vol_band_put_min_cw:
                continue
            if budget and budget > 0 and (width - credit) * 100 > budget:
                continue
            best = bought
            break                     # strikes ascend → the first fit is the widest
        if best is None:
            return self._empty_plan(
                ticker, name, analysis, expiration,
                f"no width ≤ {pre.vol_band_put_width_pct:.0%} of spot with C/W ≥ "
                f"{pre.vol_band_put_min_cw:g} within the ${budget or 0:,.0f} risk cap")
        plan = self._assemble_plan(ticker, name, analysis, expiration, sold, best, "put")
        plan.playbook, plan.min_cw = "vol_band_put", float(pre.vol_band_put_min_cw)
        plan.valid, plan.rejection_reason = True, ""     # min_cw replaces min_credit_ratio
        plan.reasoning = (f"Vol-band bull put (vol rank {float(rank):.0f} in "
                          f"[{pre.vol_band_put_min_rank:g}, {pre.vol_band_put_max_rank:g})). "
                          + plan.reasoning)
        logger.info("[%s] Vol-band bull put: %s/%s %s, credit $%.2f, width $%g",
                    ticker, sold["strike"], best["strike"], expiration, plan.net_credit, plan.spread_width)
        return plan
```

```python
# trading_agent/position_monitor.py:1181-1193
        # --- 4. Strike proximity guard (IMMEDIATE) ---
        prox_on = (self.credit_strike_proximity_exit
                   or spread.strategy_name not in CREDIT_VERTICALS)
        if prox_on and underlying_price > 0 and spread.short_strikes:
            for strike in spread.short_strikes:
                proximity = abs(underlying_price - strike) / strike
                if proximity <= self.strike_proximity_pct:
                    return (
                        ExitSignal.STRIKE_PROXIMITY,
                        f"Underlying ${underlying_price:.2f} is within "
                        f"{proximity*100:.2f}% of short strike ${strike:.0f} "
                        f"— closing to prevent ITM assignment"
                    )
```

```python
# trading_agent/risk_manager.py:139-143  (OrderExecutor._recheck_live_economics mirrors it)
        # A playbook with its own tested floor (vol-band bull put,
        # research_credit_vol.md §5) replaces the delta-aware one.
        if getattr(plan, "min_cw", None) is not None:
            cw_floor = float(plan.min_cw)
            floor_label = f"{cw_floor:g} ({plan.playbook} floor)"
```

Exits otherwise follow the credit rules (`PositionMonitor._check_exit`): profit target 50 % of credit, stop at 50 % of max loss, hard stop 3× credit, DTE safety, the profit trail — but `spread.playbook == "vol_band_put"` skips the regime-shift exit (the playbook enters in any trend). The tag travels plan → `SpreadPlan.to_dict()["playbook"]` → trade plan JSON → `SpreadPosition.playbook`. `TradingAgent._trend_filter_block` journals the RS / pullback verdict for these plans but never blocks them. Other entry gates apply as for any bull put: HighIV (rank > 95), earnings, the macro-event guard, the IV/RV floor (`vrp_min_ratio`), stop cool-down, direction-risk and net-delta caps, one position per ticker.

Preset fields (Strategy-Profile block "Vol-band bull put"): `vol_band_put_enabled` (False), `vol_band_put_tickers` (SPY, QQQ, IWM), `vol_band_put_min_rank` / `_max_rank` (50 / 80), `vol_band_put_dte` (21), `vol_band_put_short_delta` (0.25), `vol_band_put_width_pct` (0.02), `vol_band_put_min_cw` (0.10), `credit_strike_proximity_exit` (True). The paper preset runs both from 2026-11-09 (vol-band on, proximity close off).

## 4. Edge Cases / Guardrails

- **Risk cap narrower than 2 %** — at 2 % of a ~$29k account (~$570) a 2 %-wide SPY / QQQ spread (~$1,400 risk) does not fit; the width narrows to the widest listed strike that does (≈ 1 % on SPY). The backtest's 1 %-wide variant was weaker (+0.023 vs +0.034 per $ at rank 50–80), so expect less than the report's headline. Nothing fits → invalid plan, reason appended to the regular plan's.
- **Deliberate no-trade** (`strategy_name == ""`, e.g. wait-for-stabilisation, earnings with no expiry) — never overridden.
- **Thin credit / bad quotes** — `vol_band_put_min_cw` (0.10, ~1 % of backtest entries below it) at planning and again on the live recheck; legs must pass the leg-spread gate.
- **Proximity off** — bull puts / bear calls rely on the 50 %-of-max-loss and 3× credit stops and DTE safety; iron condors and the bounce bull put keep the proximity close. Defensive rolls (skill 31) trigger on the proximity signal, so they stop for credit verticals too.
- **Trade plan rotated out** — an inferred position has no playbook tag and gets the regime-shift exit again (conservative).
- **Rank band misconfigured** (min ≥ max or outside 0–100) — warned in `_coerce_overrides`, defaults used.
- **CAUTION above SMA-200 (2026-11-12, `vol_band_put_in_caution`, live on paper).** The skill-58 gate blocks "Bull Put Spread" in CAUTION, which is where the playbook backtested best: SPY < SMA-50 with the ticker above its SMA-200 gave +0.037/$, 90 % wins (n=154); below SMA-200 −0.025/$, 77 % (n=114). `TradingAgent._market_gate_block` lets a `playbook == "vol_band_put"` plan through only in CAUTION (not RISK_OFF) and only when `current_price > sma_200`; a missing price or SMA fails closed. The CAUTION size multiplier (×0.5) still sets the budget. Account effect 2008–2025: CAGR +0.9 % → +1.1 %, max DD 8.6 % → 8.2 %; CI still spans 0 (`research_credit_vol.md` §8).
- **Review point** — after ~30 closes compare the journal's realized P&L per $ risked with the backtest (+0.024); turn it off if it is negative by then.

## 5. Cross-References

- `03_credit_to_width_floor.md` — the delta-aware floor this playbook replaces with `min_cw`.
- `58_market_state_playbook.md` — routing; this is the last fallback.
- `30_profit_target_management.md` — the credit exits, proximity switch.
- `31_defensive_roll.md` — rolls ride on the proximity signal.
- `docs/plans/research_credit_vol.md` — the evidence (§2, §5).

---

*Last verified against repo HEAD on 2026-11-12.*
