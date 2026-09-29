# End-of-Day Trade Journal Reviewer

> **One-line summary:** Once-per-day retrospective LLM pass over the trade journal that produces observations, a preset-tuning proposal, and a watchlist diff — all staged as JSON for operator approval and echoed to Telegram. Does NOT trade; the numeric-threshold layer (skills 30, 31) keeps handling real-time exits.
> **Source of truth:** [`trading_agent/daily_reviewer.py`](../../trading_agent/daily_reviewer.py), [`trading_agent/daily_reviewer_main.py`](../../trading_agent/daily_reviewer_main.py), [`trading_agent/pending_preset_updates_writer.py`](../../trading_agent/pending_preset_updates_writer.py), [`trading_agent/apply_preset_update.py`](../../trading_agent/apply_preset_update.py).
> **Phase:** 3  •  **Group:** ops
> **Depends on:** `19_journal_schema.md` (journal reads), `30_profit_target_management.md` + `31_defensive_roll_evaluator.md` (the numeric layer this reviewer runs alongside), `34_exception_monitor.md` (Telegram channel), `48_claude_code_mcp_surface.md` + `55_pending_orders_promotion.md` (the read/write seam this skill mirrors).
> **Consumed by:** the launchd unit at `ops/launchd/com.trading-agent.daily-reviewer.plist`, plus the `/review` slash command (Phase C).

---

## 1. Theory & Objective

Per-cycle × per-position LLM calls burn tokens on what's mostly "hold" verdicts the numeric layer already handles. A once-a-day retrospective pass over the journal delivers the actual LLM value — pattern recognition across the day's activity — for ~1/100th the cost, and produces actionable tuning proposals rather than real-time judgments.

The reviewer is **retrospective and read-only**. It never trades. It never mutates configuration. Its two outputs are (a) an audit JSON on disk (`daily_reviews/YYYY-MM-DD.json`) and (b) a Telegram digest. When the LLM produces a concrete diff, a third output — a proposal JSON in `pending_preset_updates/<uuid>.json` — is staged for the operator to review and apply with a separate CLI.

## 2. Mathematical Formula

N/A — control-flow + LLM prompt structure. No arithmetic beyond aggregating journal fields.

## 3. Reference Python Implementation

### 3.1 Context assembly

```python
# trading_agent/daily_reviewer.py
def assemble_context(review_date: Optional[str] = None) -> ReviewContext:
    """Gather every field the LLM needs. No LLM call yet."""
```

Reads: today's opens/closes/rejects/PnL (from `JournalReader`), the currently-loaded `PresetConfig`, the watchlist, a best-effort macro snapshot (VIX zone), and today's `ExceptionMonitor` silenced-exception list. All fields are pure data; `ReviewContext` is a frozen dataclass so tests can pin an exact input.

### 3.2 LLM prompt allowlist

```python
# trading_agent/daily_reviewer.py
_ALLOWED_PROPOSAL_FIELDS: tuple[str, ...] = (
    "profit_target_pct",
    "edge_buffer",
    "min_pop",
    "max_leg_spread_cents",
    "defensive_roll_enabled",
)
```

Anything outside this list is filtered out at parse time — a hallucinated `max_risk_pct` proposal from the LLM cannot land in a pending update. Fields excluded on purpose: `max_risk_pct` (account-level judgment), `max_delta` (same), all `dte_*` (structural strategy choices).

### 3.3 Entry point

```python
# trading_agent/daily_reviewer.py
def run(review_date: Optional[str] = None,
        *,
        send_telegram: bool = True) -> Dict[str, Any]:
    """End-to-end. Returns a dict summarizing what was written."""
```

Order: assemble context → LLM → write audit → maybe stage a proposal → render digest → send to Telegram. Every step is best-effort: an LLM failure produces an empty `ReviewOutput` with `raw_llm_text` populated; a Telegram outage doesn't block the audit write.

### 3.4 New `PresetConfig` fields

```python
# trading_agent/strategy_presets.py
auto_apply_preset_updates_enabled: bool  = False
auto_apply_max_delta_change_pct:   float = 0.0
auto_apply_allowed_fields:         Tuple[str, ...] = ()
```

All three default to the SAFE end — auto-apply is unreachable without operator opt-in on both the env master switch AND the allowlist. Symmetric with `AutoPromoteConfig` from skill 55.

### 3.5 Environment variables

- `TRADING_AGENT_DAILY_REVIEWER_ENABLED` — master switch. Anything falsy (or unset) → CLI exits cleanly with a stderr note. Anything truthy → reviewer runs.
- `TRADING_AGENT_REVIEWS_DIR` — audit JSON destination (default `daily_reviews/`).
- `TRADING_AGENT_PENDING_PRESET_UPDATES_DIR` — proposal destination (default `pending_preset_updates/`).

### 3.6 launchd unit

`ops/launchd/com.trading-agent.daily-reviewer.plist` — Mon-Fri at 16:15 local. Requires two hand-edits (venv python path + repo dir) before installing.

### 3.7 Apply CLI (Phase B write path)

```python
# trading_agent/apply_preset_update.py
def evaluate_apply_gate(
    *,
    proposal: Dict[str, Any],
    preset: Any,
) -> ApplyGateResult:
    """Return an ApplyGateResult. Every predicate mirrors skill 56 §3.5."""
```

Three predicates for auto-apply:

1. `TRADING_AGENT_AUTO_APPLY_PRESET_UPDATES_ENABLED` env truthy.
2. Every field in `proposal["preset_diff"]` appears in
   `preset.auto_apply_allowed_fields`.
3. For each numeric field, `|new − old| / max(|old|, 1e-9) ≤
   preset.auto_apply_max_delta_change_pct`. Boolean fields (e.g.
   `defensive_roll_enabled`) skip the cap — a flip is a flip.

Any fail → the CLI prints the diff, lists the gate failures, and
prompts `y/N`. All pass → auto-apply fires. Preset changes go through
`save_active_preset` with the LLM-proposed field passed as an overlay
kwarg; watchlist changes go through `add_ticker` / `remove_ticker`.

### 3.8 Slash command (Phase C)

`.claude/commands/review.md` — `/review [YYYY-MM-DD]` re-renders any
past daily review from disk using the same
`render_telegram_digest()` code path. Read-only; never invokes the
apply CLI.

### 3.9 Prompt evals (Phase D)

`evals/daily_reviewer/scenarios.jsonl` — hand-curated day scenarios.
`tests/eval/test_daily_reviewer_scenarios.py` runs offline in CI
(parser + digest structure claims); opt-in online mode
(`EVAL_DAILY_REVIEWER_ONLINE=true`) sends each scenario to the live
LLM and checks the parsed output.

## 4. Edge Cases / Guardrails

- **Read-only invariant.** `daily_reviewer.py`, `daily_reviewer_main.py`, `apply_preset_update.py`, and `pending_preset_updates_writer.py` must never import `trading_agent.executor`, `submit_order`, `place_order`, or `OrderExecutor`. AST-verified.
- **LLM output validation.** Anything outside the JSON schema — missing keys, wrong types, hallucinated fields — is filtered to `None`/`[]` at parse time. The reviewer never crashes on bad LLM output.
- **Field allowlist enforced at parse time.** `_parse_llm_output` only accepts `preset_proposal.field` from `_ALLOWED_PROPOSAL_FIELDS`. This is belt (system prompt) + suspenders (code check).
- **Cold start.** When `daily_reviews/` is empty, `ctx.cold_start = True` and the digest header prefixes "(cold start — no prior reviews)". The reviewer still runs normally.
- **Weekend skip.** The launchd unit only schedules Mon–Fri. Manual runs on weekends still work; the reviewer processes whatever the journal contains.
- **Telegram outage.** A send failure logs a warning and returns `sent=False`. The audit + proposal writes still complete.
- **Malformed LLM JSON.** Parse fails → returns an empty `ReviewOutput` with `raw_llm_text` echoing the failure. Digest renders with the "no proposals" branch; no proposal file is staged.
- **Apply CLI is opt-in.** `apply_preset_update.py` never runs unattended. Auto-apply requires the operator to (a) flip `TRADING_AGENT_AUTO_APPLY_PRESET_UPDATES_ENABLED=true`, (b) populate `preset.auto_apply_allowed_fields`, AND (c) set `auto_apply_max_delta_change_pct > 0`. Any one of the three at its default disables auto-apply.
- **`save_active_preset` callers are CI-restricted.** Only `apply_preset_update.py`, `strategy_presets.py` itself, and Streamlit surfaces may import it. A new scheduled task or MCP tool cannot silently mutate live config — enforced by `test_skill_56_save_active_preset_callers_restricted`.
- **`pending_preset_updates/` + `daily_reviews/` must be `.gitignore`d.** Both may contain live PnL and account context.

## 5. Cross-References

- `30_profit_target_management.md` + `31_defensive_roll_evaluator.md` — the numeric-threshold layer this reviewer runs alongside. Real-time exits happen there; this skill is retrospective tuning.
- `34_exception_monitor.md` — the Telegram channel the digest lands in.
- `48_claude_code_mcp_surface.md` + `55_pending_orders_promotion.md` — the read/write-seam pattern this skill mirrors (proposals → CLI review → gate → apply).
- `19_journal_schema.md` — the journal schema the reviewer reads.
- `docs/plans/portfolio_llm_evaluator_plan.md` — the plan doc this skill lands from, including the four-phase rollout.

---

*Last verified against repo HEAD on 2026-09-29.*
