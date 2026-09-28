# Pre-Trade Approval — Playbook

> **One-line summary:** Turns a natural-language "consider a JPM iron condor at 35 DTE" into a fully-scored candidate + risk report + a `pending_orders/<uuid>.json` file the promote CLI (skill 55) can then submit. This playbook is the ONLY way a Claude Code session produces an order proposal; direct writes to the executor are forbidden.
> **Source of truth:** [`trading_agent/mcp/tools/strategy.py`](../../trading_agent/mcp/tools/strategy.py), [`trading_agent/pending_orders_writer.py`](../../trading_agent/pending_orders_writer.py), [`trading_agent/executor_promote.py`](../../trading_agent/executor_promote.py) (skill 55).
> **Phase:** 2  •  **Group:** ops
> **Depends on:** `48_claude_code_mcp_surface.md`, `55_pending_orders_promotion.md`, `18_order_submission_idempotency.md`, `03_credit_to_width_floor.md`.
> **Consumed by:** `.claude/commands/propose.md`, the `risk-reviewer` subagent (which runs after the proposal is scored and before it's staged).

---

## 1. Theory & Objective

The operator wants to say "consider X" and have Claude Code produce a concrete, scored, risk-checked order proposal in the staging directory — without touching the executor. Two goals: (a) the operator can review the proposal at a terminal before running `python -m trading_agent.executor.promote <uuid>`; (b) even if Claude misjudges, the promote CLI re-scores against fresh market data and re-runs risk checks, so a stale or drifted proposal fails at the gate.

The read/write boundary is symmetric: this skill only *writes JSON files* into `pending_orders/`. It never imports the executor. Skill 55 owns the executor call; that's the ONLY module besides `executor.py` allowed to import `submit_order` (CI-verified in Phase 5).

## 2. Mathematical Formula

Reuses:

- **Scoring**: `decision_engine.decide()` via `score_candidate()`. Invariant #2 forbids redefining scoring anywhere else.
- **C/W floor**: `|Δshort| × (1 + edge_buffer)` — invariant #1. The proposal writer includes this in the JSON so the promote CLI can re-verify.
- **Risk cap**: `PresetConfig.max_risk_pct × account_balance` (skill 3).

## 3. Reference tool sequence

```python
# Claude Code executes:

preset    = get_preset()
chain     = get_chain(underlying, expiration, option_type)
verdict   = score_candidate(underlying, strategy, params)
risk      = get_risk_report()

# risk-reviewer subagent (.claude/agents/risk-reviewer.md) reads the
# above four dicts and returns "approve" or "reject with reasons".

# On approve → write pending_orders/<uuid>.json via the writer helper
# (which is a plain file-write, NOT an executor call):
proposal_uuid = pending_orders_writer.write(
    underlying=underlying,
    strategy=strategy,
    params=params,
    verdict=verdict,
    risk_snapshot=risk,
    preset_name=preset["preset"]["name"],
)
```

The playbook then tells the operator: "Proposal written to `pending_orders/<uuid>.json`. Run `python -m trading_agent.executor_promote <uuid>` to review + submit."

## 4. Edge Cases / Guardrails

- **NEVER call the executor.** The playbook writes a JSON file and stops. Even if the operator says "just submit it," Claude Code responds by running the promote CLI as a shell command (which shows the diff + waits for `--yes` unless auto-promote conditions all hold — see skill 55 §3.5). No path in this playbook imports `trading_agent.executor.*`.
- **risk-reviewer subagent runs BEFORE the write.** A "reject" from the subagent means no file is created. The operator sees the reasons but no proposal enters staging.
- **Staleness label.** The JSON includes `proposed_at_utc` so the promote CLI can enforce the <5% re-score-drift cap.
- **`pending_orders/` must be `.gitignore`d.** Proposals contain live account context; they must never enter version control.
- **UUID collision guard.** The writer uses `uuid4()` + a millisecond suffix; the promote CLI refuses to run on a UUID that doesn't match one file exactly.
- **Auto-promote is opt-in per proposal.** The proposal JSON carries an `auto_promote_requested: bool` flag Claude Code sets only when the user explicitly asks for it. The promote CLI still has final say — it re-checks the master switch + all `AutoPromoteConfig` caps.

## 5. Cross-References

- `55_pending_orders_promotion.md` — the CLI that consumes proposals written here.
- `48_claude_code_mcp_surface.md` — the read tools this playbook calls.
- `18_order_submission_idempotency.md` — the executor invariant this playbook deliberately does NOT bypass.
- `03_credit_to_width_floor.md` — the C/W formula the proposal JSON records.

---

*Last verified against repo HEAD on 2026-09-28.*
