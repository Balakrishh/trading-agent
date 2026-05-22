# Ticker filters — early-return pipeline

> **One-line summary:** Three early-return checks at the top of `agent._process_ticker` (directional bias, RSI gate, high-IV block) were ~120 lines of inlined branching that decided whether to spin up the expensive sentiment + chain-fetch + planner pipeline for a ticker. Skill 36 extracts them into `TickerFilters.evaluate(ticker, analysis) -> Optional[FilterResult]` so each filter is testable in isolation and the orchestrator branch is one line.
> **Source of truth:** [`trading_agent/ticker_filters.py`](../../trading_agent/ticker_filters.py), [`trading_agent/agent.py`](../../trading_agent/agent.py) construction + delegation.
> **Phase:** 2  •  **Group:** architecture
> **Depends on:** `13_preset_system_hot_reload.md` (PresetConfig is the input), `09_vix_zscore_inhibitor.md` (high-IV semantics).
> **Consumed by:** `agent._process_ticker` (the only caller).

---

## 1. Theory & Objective

`_process_ticker` is the per-ticker entry point of the cycle. Pre-2026-05-22 the first ~120 lines were three early-return checks gated by IF blocks, each with its own journal-write call and per-skip return dict. The pattern was:

```python
def _process_ticker(self, ticker, balance, ...):
    analysis = self.regime_classifier.classify(ticker)
    if not regime_is_allowed(analysis.regime.value, self.preset.directional_bias):
        # ~25 lines: journal + return dict
        return {...}
    if rsi_gate_enabled:
        decision = evaluate_rsi_gate(...)
        if not decision.allow:
            # ~25 lines: journal + return dict
            return {...}
        if decision.override_regime is not None:
            analysis = dataclasses.replace(analysis, regime=...)
    if analysis.high_iv_warning:
        # ~25 lines: journal + return dict
        return {...}
    # Phase III planning continues here...
```

Three problems with the inlined form:

1. **Testing required `MagicMock(spec=TradingAgent)`.** Every filter was implicit in the method body, so exercising one required spinning up the whole agent fixture. The same fixture-bloat pattern that motivated skill 35.
2. **Adding a filter meant editing the method.** A new gate (e.g. "skip if vega exceeds threshold") had to be slotted into the right position in the IF chain, which is a coordination change with the surrounding code.
3. **Filter ordering was implicit.** The fact that directional-bias comes before RSI which comes before high-IV was buried in line order; no docstring made it explicit.

The extraction:

- **`TickerFilters`** is constructor-injected with `journal_kb` + the active `preset`. Each filter has a private method (`_check_directional_bias`, `_check_rsi_gate`, `_check_high_iv`). `evaluate()` runs them in documented order and returns the first triggered `FilterResult`, or `None` if all pass.
- **`FilterResult`** is a frozen dataclass carrying the result dict the caller returns, the filter name (for tests/logging), and an optional `analysis_override` for the RSI-gate regime rewrite case.

After the extraction, the caller looks like:

```python
analysis = self.regime_classifier.classify(ticker)
filter_res = self._ticker_filters.evaluate(ticker, analysis)
if filter_res is not None:
    if filter_res.analysis_override is not None:
        analysis = filter_res.analysis_override
    if filter_res.result.get("status") == "skipped":
        return filter_res.result
# Phase III planning continues here...
```

`_process_ticker` shrinks from 340 → 229 lines.

## 2. Mathematical Formula

Control flow only.

```text
TickerFilters.evaluate(ticker, analysis):
    1. directional bias:
         if not preset.directional_bias allows analysis.regime:
             journal skipped_bias, return FilterResult(skipped)
    2. RSI gate (env-gated by RSI_GATE_ENABLED):
         decision = evaluate_rsi_gate(analysis.regime, analysis.rsi_14)
         if not decision.allow:
             journal skipped_rsi_gate, return FilterResult(skipped)
         if decision.override_regime is not None:
             return FilterResult(analysis_override=replace(analysis, regime=...))
    3. high-IV block:
         if analysis.high_iv_warning:
             journal defense_first row, return FilterResult(skipped)
    return None  # all clear, proceed to planning
```

## 3. Reference Python Implementation

### 3.1 TickerFilters.evaluate

```python
# trading_agent/ticker_filters.py
class TickerFilters:
    def evaluate(self, ticker: str,
                 analysis: "RegimeAnalysis") -> Optional[FilterResult]:
        ...
```

### 3.2 FilterResult

```python
# trading_agent/ticker_filters.py
@dataclass(frozen=True)
class FilterResult:
    result: Dict[str, Any]
    triggered_by: str
    analysis_override: Optional[Any] = None
```

### 3.3 Agent integration

```python
# trading_agent/agent.py — TradingAgent.__init__
self._ticker_filters = TickerFilters(
    journal_kb=self.journal_kb,
    preset=self.preset,
)
```

## 4. Edge Cases / Guardrails

- **RSI gate override returns FilterResult with analysis_override but no skip.** The caller must check `analysis_override` BEFORE checking `result.status == "skipped"`. Failing to apply the override would mean the planner sees the original regime — Iron Condor instead of the gate's downgrade.
- **Mean-reversion is always allowed.** The directional-bias check delegates to `regime_is_allowed` which special-cases MEAN_REVERSION. Adding a new "block mean reversion" preset means changing `regime_is_allowed`, not `TickerFilters`.
- **High-IV uses `log_defense_first`, not `log_signal`.** Different journal action by design — the dashboard's "Defense-first today" panel reads that specific action.
- **Preset is captured at construction.** A hot-reload of the preset file doesn't propagate to existing `TickerFilters` instances. `_run_cycle_impl` reloads the preset at each cycle's start and reconstructs filters as needed (this is currently a TODO — the preset is captured once at agent __init__).
- **The filter pipeline doesn't catch exceptions from `regime_is_allowed` or `evaluate_rsi_gate`.** Those are pure functions that should never throw on valid inputs; if they do, the exception propagates to `_process_ticker` and is caught by its outer except (which is skill-34-instrumented).
- **Filter order is documented in evaluate() docstring.** Directional bias → RSI gate → high-IV. Changing this order requires updating the docstring + a regression test.

## 5. Cross-References

- `13_preset_system_hot_reload.md` — defines PresetConfig.directional_bias.
- `09_vix_zscore_inhibitor.md` — companion macro filter (high-IV is its hard-cap sibling).
- `35_close_event_collaborators.md` — same extraction pattern applied to the close path.

---

*Last verified against repo HEAD on 2026-05-22.*
