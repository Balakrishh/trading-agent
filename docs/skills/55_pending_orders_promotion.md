# Pending-Orders Promotion — Write Gate

> **One-line summary:** The CLI (`python -m trading_agent.executor_promote <uuid>`) that consumes proposals written by skill 51 and either waits for the operator's `--yes` or auto-promotes when every one of seven gate conditions holds. This module + `executor.py` are the ONLY two files allowed to import order-submission primitives (CI-verified in Phase 5).
> **Source of truth:** [`trading_agent/executor_promote.py`](../../trading_agent/executor_promote.py), [`trading_agent/pending_orders_writer.py`](../../trading_agent/pending_orders_writer.py), [`trading_agent/strategy_presets.py`](../../trading_agent/strategy_presets.py).
> **Phase:** 3  •  **Group:** ops
> **Depends on:** `51_pre_trade_approval.md` (the proposal producer), `18_order_submission_idempotency.md` (the executor primitive this CLI is the sole caller of), `13_preset_system_hot_reload.md` (`AutoPromoteConfig` fields on `PresetConfig`).
> **Consumed by:** the operator terminal + `.claude/commands/propose.md` after skill 51 stages a proposal.

---

## 1. Theory & Objective

Skill 51 lets Claude Code produce a fully-scored, risk-checked order proposal as a JSON file in `pending_orders/`. This skill owns the second half: consuming that JSON and deciding whether the order actually gets submitted. Two paths:

- **Manual (default).** The CLI prints the proposal diff and waits for `--yes` on stdin. This is the normal path; the operator eyeballs the trade and confirms.
- **Auto-promote (opt-in).** Small trades ("paper cuts") that satisfy every one of seven AND-ed conditions get submitted without a prompt. Every one of the conditions is a fail-closed default, so the delta between "master switch on but not configured" and "auto-promote everything" is real work — the operator must lift each field explicitly.

The read/write boundary from skill 48 lands here: the MCP surface produces proposals, this CLI is the sole write gate to the executor. The invariant scanner in Phase 5 verifies that no third file bypasses this seam.

## 2. Mathematical Formula

N/A — control flow. The seven gate predicates are boolean, evaluated left-to-right with early exit on any failure. Numeric thresholds all live on `PresetConfig`:

```text
auto_promote_max_notional_usd   ∈ ℝ⁺       (0 disables)
auto_promote_max_contracts      ∈ ℕ         (default 1)
auto_promote_allowed_strategies ⊆ strategies (default empty → disabled)
```

Master switch: `TRADING_AGENT_AUTO_PROMOTE_ENABLED` env var, truthy = 1/true/yes/on, everything else false.

## 3. Reference Python Implementation

### 3.1 Config surface

```python
# trading_agent/strategy_presets.py — PresetConfig fields
auto_promote_max_notional_usd:     float = 0.0
auto_promote_max_contracts:        int   = 1
auto_promote_allowed_strategies:   Tuple[str, ...] = ()
```

Defaults are the SAFE end: `max_notional=0.0` makes the config-side gate fail even if the env master switch is on. Operator opts in per preset by editing these fields via `dataclasses.replace` (skill 00 conventions — frozen dataclass mutations).

### 3.2 Gate evaluation

```python
# trading_agent/executor_promote.py
def evaluate_auto_promote_gate(
    *,
    proposal: Dict[str, Any],
    preset: Any,
    current_notional: float,
    current_contracts: int,
    risk_warnings: int,
    now_utc: datetime,
    within_market_hours: bool,
    rescore_drift_pct: float,
    minutes_since_open: Optional[int] = None,
    minutes_until_close: Optional[int] = None,
) -> GateResult:
    """Return a GateResult; caller decides auto-promote vs manual."""
```

The function is pure (no I/O). All seven predicates set to `"pass"` → `GateResult.all_pass == True` → auto-promote. Any failure returns a human string on the field so the CLI can render it.

### 3.3 The seven gate conditions (all AND-ed)

1. `TRADING_AGENT_AUTO_PROMOTE_ENABLED` env var truthy
2. Proposal debit notional ≤ `preset.auto_promote_max_notional_usd`
3. Contract count ≤ `preset.auto_promote_max_contracts`
4. `proposal["strategy"] ∈ preset.auto_promote_allowed_strategies`
5. `risk_snapshot["warning_count"] == 0` (not just errors — a warning fails)
6. Regular market hours + ≥5 min from open + ≥5 min from close
7. Re-score drift from proposal-time < 5%

### 3.4 CLI

```
python -m trading_agent.executor_promote <uuid>              # interactive
python -m trading_agent.executor_promote <uuid> --yes        # skip prompt (still runs gate)
python -m trading_agent.executor_promote <uuid> --dry-run    # print + exit
```

The CLI always prints the proposal diff first, then runs the gate. If the operator supplied `--yes` OR every gate condition passes AND the proposal's `auto_promote_requested` flag is set, the executor call fires. Otherwise the CLI prompts.

## 4. Edge Cases / Guardrails

- **Fail-closed defaults everywhere.** `max_notional=0.0` and `allowed_strategies=()` both make auto-promote unreachable even with the env switch on. Two levers means one wrong flip doesn't drain the account.
- **Re-score is mandatory before submission.** A stale proposal (>5% score drift) falls back to manual approval regardless of the other gates.
- **Missing proposal → exit 2.** No proposal file, no prompt, no executor call.
- **Operator declines interactive prompt → exit 3.** Clean exit, no journal entry needed (the proposal stays in `pending_orders/` for review or deletion).
- **`OrderExecutor` import lives at the call site.** Not module-top. This lets the conformance test verify (a) the module *does* import it — so submission actually happens — and (b) *only* this file and `executor.py` import it.
- **`.gitignore` must include `pending_orders/`.** Proposals carry live account context; they must never enter version control.
- **Auto-promote emits an INFO to ExceptionMonitor.** The operator's Telegram channel gets a "Claude auto-promoted trade X" line (visibility without a manual gate). Auto-promote *failures* (any of the 7 predicates failing) log a WARNING with the field name.
- **Wheel proposals (2026-09-29).** A plan whose `strategy` is `Cash-Secured Put` or `Covered Call` skips `RiskManager` (spread C/W and width rules don't apply to one short option) and goes through `check_wheel_order` + `OrderExecutor.execute_single_leg` (skill 40 §2.9): up to three sell-to-open attempts — mid, halfway to bid, bid (the last only if the live quote passes the preset width gate). Exit codes: 0 filled / dry-run, 6 check rejected, 8 unfilled, 7 other (incl. unresolved cancel). `params.contracts` sets qty (default 1).

## 5. Cross-References

- `51_pre_trade_approval.md` — the proposal producer this CLI consumes.
- `48_claude_code_mcp_surface.md` — the read surface Claude Code uses to build the proposal.
- `18_order_submission_idempotency.md` — the executor invariant this CLI extends (idempotent submission, one client_order_id per proposal).
- `13_preset_system_hot_reload.md` — the `PresetConfig` shape the gate reads.

---

*Last verified against repo HEAD on 2026-09-30.*
