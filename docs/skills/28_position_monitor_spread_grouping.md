# Position-monitor spread grouping (plan-match-then-infer)

> **One-line summary:** How the position monitor turns a flat list of broker option legs into aggregated spread rows for the dashboard — first by matching each leg's symbol against the trade plan that created it, then by inferring spread structure for any leftover legs.
> **Source of truth:** [`trading_agent/position_monitor.py:294-373`](../../trading_agent/position_monitor.py)
> **Phase:** 1  •  **Group:** architecture
> **Depends on:** `19_journal_schema.md` (state_history retains rejected plans), `00_sdlc_and_conventions.md` (envelope vs inner shape conventions)
> **Consumed by:** `agent.py:_load_trade_plans` (Stage 1 monitor), `streamlit/live_monitor.py:_load_positions_with_plans` (Open-Positions panel) — both feed it the same flat broker leg list.

---

## 1. Theory & Objective

The broker (Alpaca) reports each option leg as a flat position. A 4-leg Iron Condor shows up as four separate rows. For the user to see "one spread P&L = $-23" instead of four leg-level P&L numbers, something has to aggregate the legs back into spreads.

Two designs were considered. The naïve one — group purely by `(underlying, expiration)` and infer the strategy from leg structure — fails the moment two strategies on the same ticker overlap expirations (e.g. a Bull Put and a Bear Call rolled in the same day). The chosen design instead **matches each leg back to the trade plan that originated it**: every plan's `legs` list is a canonical fingerprint. When the plan-match fails (e.g. the plan was rotated out of `state_history`, or the position was opened manually outside the agent), the system falls through to the `(underlying, expiration)` inference so the user always sees the position.

A subtle requirement: only plans the agent **actually executed against** may claim broker legs. Rejected plans live in `state_history` too, and can share leg symbols with the eventually-submitted plan — without filtering, they greedily claim those shared legs and split one Iron Condor into two display rows. See §4 for the 2026-05-15 XLF incident.

## 2. Mathematical Formula

N/A — control flow only. The interesting properties are which-plan-wins (§3) and the edge-case failure modes (§4).

## 3. Reference Python Implementation

### 3.1 The filter that gates which plans may claim legs

```python
# trading_agent/position_monitor.py:311-335
for plan in trade_plans:
    tp = self._extract_inner_plan(plan)

    # ── Skip plans the agent never actually executed ──────────
    # ``state_history`` retains every plan the chain scanner
    # emitted that cycle, including rejections (``valid=False``)
    # and risk-vetoed plans (``risk_verdict.approved=False``).
    # A rejected plan whose leg symbols partially overlap the
    # actually-submitted plan would otherwise greedily claim
    # the shared legs first — splitting a single Iron Condor
    # into two display rows and double-counting against the
    # per-ticker position cap. See the 2026-05-15 XLF incident:
    # 4 rejected plans shared the call wing with the submitted
    # plan and stole those legs into a phantom spread row.
    #
    # Defaults are permissive — missing ``valid`` or missing
    # ``risk_verdict`` are treated as valid/approved so that
    # inner-shaped plans (which don't carry risk_verdict) and
    # older history files (pre-``valid`` field) aren't dropped.
    if tp.get("valid") is False:
        continue
    if isinstance(plan, dict):
        rv = plan.get("risk_verdict")
        if isinstance(rv, dict) and rv.get("approved") is False:
            continue
```

### 3.2 The match step itself

```python
# trading_agent/position_monitor.py:337-379
plan_legs = tp.get("legs", [])
plan_symbols = {leg["symbol"] for leg in plan_legs}
if not plan_symbols:
    continue

# Skip plans whose every leg has already been claimed by an
# earlier matching plan. ``state_history`` typically retains
# several plan entries with the same leg-symbol set (re-runs,
# duplicate fills, planner re-emissions); without this guard
# the same broker spread shows up as N duplicate rows in the
# dashboard.
if plan_symbols.issubset(matched_symbols):
    continue

matched_legs = [
    p for p in positions
    if p.symbol in plan_symbols and p.symbol not in matched_symbols
]
if not matched_legs:
    continue
# … (build the SpreadPosition row, then:)
matched_symbols.update(p.symbol for p in matched_legs)
```

### 3.3 The inference fallback for unmatched legs

```python
# trading_agent/position_monitor.py:381-390
unmatched = [p for p in positions if p.symbol not in matched_symbols]
inferred = self._infer_spreads_from_legs(unmatched)
spreads.extend(inferred)
```

`_infer_spreads_from_legs` buckets legs by `(underlying, expiration)` then classifies the strategy by leg-count + put/call/short/long mix. 4 legs spanning both sides → "Iron Condor"; 2 puts → "Bull Put Spread"; 2 calls → "Bear Call Spread"; 1 short → "Naked Short"; everything else → "Multi-leg Position".

## 4. Edge Cases / Guardrails

- **Rejected-plan leg-claim (the 2026-05-15 XLF bug).** State history retained 4 rejected XLF Iron Condor plans (`valid=False`, CW ratio too low) that all shared the call wing `C52.5/C54.0` with the eventually-submitted plan but had a different invalid put wing. Without the `valid != False` filter, the rejected plans iterated first, claimed the call legs, and the submitted plan only got the put legs — producing two `Iron Condor` rows with 2 legs each and the rejected plan's mis-attributed economics on the first row. Cap-counting at `agent.py` then over-counted `positions_per_ticker['XLF']` as 2, accidentally blocking further entries (right outcome, wrong reason). Filter shipped 2026-05-15.

- **Risk-vetoed envelope.** Belt-and-braces — even if `tp["valid"]` is `True` (chain scanner approved), `plan["risk_verdict"]["approved"]` may be `False` (risk manager vetoed). Both gates must pass for the plan to claim legs.

- **Legacy history without `valid` field.** Older `trade_plan_*.json` files predate the `valid` field. The filter defaults missing to "valid" — so upgrading the agent against an existing state_history archive doesn't blank the Open-Positions panel.

- **Inner vs envelope shape ambiguity.** Two callers pass differently shaped dicts: agent.py passes the full state_history envelope (`{run_id, trade_plan: {...}, risk_verdict: {...}}`), and as of 2026-05-15 Streamlit also passes envelopes (it previously pre-unwrapped to inner — see the GLD bug below). The `valid` check works in both shapes because `_extract_inner_plan` returns the inner regardless. The `risk_verdict.approved` check only fires when an envelope is provided.

- **Caller must pass envelopes if risk-verdict filtering matters (the 2026-05-15 GLD bug).** The risk-vetoed-plan filter only engages when the caller hands `group_into_spreads` the FULL envelope. Streamlit's `_fetch_spreads_cached` historically appended `entry["trade_plan"]` (inner only), which dropped `risk_verdict` and silently neutered the second half of the filter. A risk-rejected plan (chain scanner emitted `valid=True`, risk manager set `approved=False`) with leg symbols matching the submitted plan then claimed the broker legs first, and the dashboard rendered the rejected plan's `net_credit` / `max_loss` instead of the actual fill economics. Concrete impact: GLD on 2026-05-15 displayed `credit=$1.75 / max_loss=$325 / P&L=-60%` when the broker fill was `credit=$1.95 / max_loss=$305 / P&L=-54%`. The −60% looked alarming and was a fake number generated by attribution error, not a real loss. Fix: `live_monitor.py:_fetch_spreads_cached` now appends the full envelope, matching what `agent.py:_load_trade_plans` already did. Any new caller threading data into `group_into_spreads` MUST pass envelopes — passing inner-only re-opens this bug class.

- **Plan with empty `legs` list.** Caught by `if not plan_symbols: continue`. Happens for rejected plans where the chain scanner gave up before producing candidate legs (e.g. "no positive-EV candidate found").

- **Duplicate plan entries in `state_history`.** Re-runs and planner re-emissions append a fresh history entry every cycle. Once the first matching plan has claimed all of a plan's symbols, the `plan_symbols.issubset(matched_symbols)` check short-circuits the rest. **Order of iteration matters** — the first plan to reach the match step keeps its economics on the row. After the `valid` filter, the first plan to reach the match is necessarily a real submission, so the displayed economics are correct.

- **Manual/orphan positions outside the agent.** No matching plan → falls through to `_infer_spreads_from_legs`, which produces a row with `origin="inferred"` (vs `origin="trade_plan"` for plan-matched rows). The UI uses this to label inferred rows differently.

## 5. Cross-References

- `19_journal_schema.md` — explains why `state_history` retains rejected plans at all (audit trail / replay debugging).
- `13_preset_system_hot_reload.md` — preset-level state lives separately from trade-plan state; the two don't share history.
- `17_close_failure_and_cooldown.md` — the per-ticker position cap relies on `group_into_spreads`'s output count; pre-fix the cap was over-counting and the close-failed cooldown was indirectly affected on tickers with rejected plans.
- **Runbook 03** (zombie position recovery) — when the inference fallback labels an unfamiliar row, the user lands here first.

---

*Last verified against repo HEAD on 2026-05-15 (incl. GLD envelope-unwrap follow-up shipped same day).*
