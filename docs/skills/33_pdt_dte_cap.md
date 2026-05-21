# PDT-aware DTE cap

> **One-line summary:** When the trading account is below FINRA's $25K Pattern Day Trader threshold, cap each strategy's target DTE at `preset.pdt_dte_cap` (default 14 days) so the planner picks shorter expirations. Reduces drift risk for positions that can't be closed same-day due to PDT denials. When the account is at-or-above $25K, the cap is ignored and the preset's original DTE applies.
> **Source of truth:** [`trading_agent/strategy_presets.py:PresetConfig`](../../trading_agent/strategy_presets.py), [`trading_agent/strategy.py:apply_pdt_dte_cap`](../../trading_agent/strategy.py), [`trading_agent/agent.py`](../../trading_agent/agent.py) (one-call-per-cycle wiring).
> **Phase:** 2  •  **Group:** risk
> **Depends on:** `13_preset_system_hot_reload.md` (where the cap field lives), `17_close_failure_and_cooldown.md` (the same-day PDT block this cap mitigates), `30_profit_target_management.md` (companion exit-management knob).
> **Consumed by:** `StrategyPlanner._pick_expiration` — reads the current (possibly-capped) `_dte_*` attributes when scoring expirations.

---

## 1. Theory & Objective

A position's risk of an unrecoverable assignment scales with how long the underlying has to drift to a short strike *before the agent can next close it*. For accounts ≥ $25K, that window is small — same-day closes work fine. For sub-$25K accounts, FINRA's PDT rule blocks same-day opens-then-closes, so a position opened today can't be auto-closed until tomorrow's first cycle. If the underlying drifts past the short strike overnight, the position rides to ITM.

Capping DTE shorter on PDT-restricted accounts trades some theta runway for less drift risk. A 21-DTE Iron Condor has ~3 weeks for an adverse move; a 14-DTE has half that. Theta decay also accelerates in the second half of the position's life, so a shorter-DTE entry collects most of the same theta in a shorter window.

The cap is **reactive** — it's applied per cycle based on the current account balance. If balance crosses $25K mid-day (deposit, large unrealised gain), the next cycle removes the cap automatically; if it crosses back down, the cap re-engages.

**Why a preset field, not a hardcoded constant.** Aggressive presets may want a tighter cap (e.g., 10 days for faster theta cycling); Conservative may want looser (21 days). Per-preset tunability matches the rest of the configuration story.

## 2. Mathematical Formula

```text
For each strategy ∈ {vertical, iron_condor, mean_reversion}:

  dte_current = preset.dte_<strategy>                 if account_balance >= PDT_EQUITY_THRESHOLD
              = min(preset.dte_<strategy>, pdt_dte_cap) if account_balance <  PDT_EQUITY_THRESHOLD

where
  PDT_EQUITY_THRESHOLD = $25,000  (FINRA rule)
  pdt_dte_cap          ∈ ℕ      (default 14)
```

Per-strategy independence: an Iron Condor at 35d and a Vertical at 21d would both cap to 14 days under the default. Mean Reversion at 14d already satisfies the cap and is unchanged.

## 3. Reference Python Implementation

### 3.1 PresetConfig field

```python
# trading_agent/strategy_presets.py:PresetConfig
pdt_dte_cap:                       int   = 14
```

### 3.2 Planner method

```python
# trading_agent/strategy.py
def apply_pdt_dte_cap(self, cap: Optional[int]) -> None:
```

### 3.3 Per-cycle agent wiring

```python
# trading_agent/agent.py — run_cycle, right after pdt_restricted is computed
cap = self.preset.pdt_dte_cap if pdt_restricted else None
try:
    self.strategy_planner.apply_pdt_dte_cap(cap)
```

## 4. Edge Cases / Guardrails

- **Cap higher than original DTE → no-op.** `min(35, 60) == 35` keeps the preset's value. A cap loose enough to never bind is harmless.
- **`cap=None` restores originals.** Idempotent restoration handles intraday balance transitions in both directions (cap engages → cap clears as balance crosses threshold up or down).
- **Originals stored separately.** `_dte_vertical_orig`, `_dte_iron_condor_orig`, `_dte_mean_reversion_orig` are immutable from constructor. The mutable `_dte_*` shadow values are what the planner's `_pick_expiration` reads. Restoring is a copy from `_orig` — no recomputation needed.
- **`apply_pdt_dte_cap` is wrapped in try/except at the agent call site.** A planner-side bug (unlikely) must not crash the cycle.
- **Mean-Reversion DTE typically already short.** Default Balanced has 14d; the cap is a no-op for Mean Reversion under default. The cap matters mostly for Vertical (21d) and Iron Condor (35d).
- **Cap doesn't affect existing positions.** It only changes future entry-DTE. A position opened with DTE=35 stays open with its original expiry; only the next planning cycle sees the cap.

## 5. Cross-References

- `13_preset_system_hot_reload.md` — `pdt_dte_cap` is hot-reloadable like other preset fields; STRATEGY_PRESET.json changes apply next cycle.
- `17_close_failure_and_cooldown.md` §4 — describes the same-day PDT block this cap mitigates. The two are companions: skill 17 stops the noise *after* a block happens; skill 33 reduces the chance of an overnight ITM that would *trigger* the block.
- `30_profit_target_management.md` — exit-management companion; together with this skill, shorter DTE + earlier profit-take both compress the active-window of each trade on PDT-restricted accounts.

---

*Last verified against repo HEAD on 2026-05-21.*
