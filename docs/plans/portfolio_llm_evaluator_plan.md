# End-of-Day Trade Journal Reviewer — Plan

**Working document — not a skill doc yet.** Becomes
`docs/skills/56_daily_journal_reviewer.md` once Phase A lands.

**One-line intent.** Once per day, after market close, an LLM reads
the day's full trade journal (opens, closes, rejects, PnL, alerts) and
produces a structured "what happened + what to tune" report. Preset
tuning proposals are staged as JSON diffs the operator approves with
one command; the auto-close threshold layer (skills 30, 31) keeps
running as-is at cycle time.

**Why this shape.** Per-cycle × per-position LLM calls burn tokens on
what's mostly "hold" verdicts. A once-a-day pass over the journal
delivers the actual value — pattern recognition across the day's
trades — for two orders of magnitude less cost, and produces
actionable tuning rather than real-time judgments the numeric layer
already handles.

---

## Locked-in decisions (from previous iteration)

- **Trigger** — one scheduled run per day, ~4:15 PM ET (15 min after
  close, so late fills have settled). launchd unit on the Mac.
- **Numeric-threshold closes** — unchanged. `agent.py` + skill 30 +
  skill 31 keep firing profit-target / stop-loss / strike-proximity
  exits every cycle. This reviewer does NOT trade.
- **Telegram** — same channel as ExceptionMonitor. One digest message
  after the review completes.
- **LLM authority** — proposes preset + watchlist diffs; never
  applies them. Operator runs one command to accept.

---

## Architecture

```
launchd (4:15 PM ET, weekdays)
    │
    └── python -m trading_agent.daily_reviewer
            │
            ├── gather:
            │     journal_reader.opens_today / closes_today / rejects
            │     load_active_preset() → current PresetConfig
            │     watchlist_store.load()
            │     recent ExceptionMonitor alerts
            │     macro snapshot from vix_regime_monitor
            │
            ├── llm_client.review(structured_context) → ReviewOutput
            │     (Sonnet — this is a judgment call worth quality)
            │
            ├── write daily_reviews/YYYY-MM-DD.json  (permanent audit)
            ├── if diffs proposed:
            │     pending_preset_updates/<uuid>.json  (stage for approval)
            ├── telegram_notifier.send(digest_markdown)
            └── journal_kb.log_signal(action="daily_review", ...)
```

The operator sees the Telegram digest on their phone, reads the
proposal, and decides whether to run:

```bash
python -m trading_agent.apply_preset_update <uuid>
```

which prints the diff and requires `--yes` (or the same 3-predicate
gate as skill 55 — env master switch + notional cap analog + preset
allowlist — for auto-apply on well-understood tuning).

### New modules

- **`trading_agent/daily_reviewer.py`**
  - `ReviewContext` frozen dataclass — everything gathered before the
    LLM call.
  - `ReviewOutput` frozen dataclass — LLM verdict + reasoning +
    diff proposals.
  - `run(target_date) -> ReviewOutput` — main entry.
  - `_render_telegram_digest(output) -> str`.

- **`trading_agent/pending_preset_updates_writer.py`**
  - Mirror of `pending_orders_writer.py` — plain file-write, zero
    executor imports.

- **`trading_agent/apply_preset_update.py`**
  - CLI (`python -m trading_agent.apply_preset_update <uuid>`).
  - Reads the pending update JSON, prints a Rich-style diff of
    `PresetConfig` field-by-field, prompts for `--yes`.
  - On approve: uses `dataclasses.replace(current, **diff)` to
    produce the new frozen preset, then calls
    `strategy_presets.save_active_preset(...)`.
  - Also updates `watchlist_store.save_watchlist(...)` when the
    proposal touches watchlist entries.

- **`trading_agent/journal_reader.py`** — one extension:
  `closes_in_window(days: int)` and `opens_in_window(days: int)` for
  multi-day rollups. Currently the reader is today-only (limitation
  called out in skill 52 and the previous plan). This unblocks both.

### Modified modules

- **`config.py`** — one field:
  ```python
  daily_reviewer_enabled: bool = False
  ```
  Reads `TRADING_AGENT_DAILY_REVIEWER_ENABLED`.

- **`strategy_presets.py`** — three new `PresetConfig` fields for the
  auto-apply path (same shape as `AutoPromoteConfig`, same fail-closed
  defaults):
  ```python
  auto_apply_preset_updates_enabled:      bool  = False
  auto_apply_max_delta_change_pct:        float = 0.0  # e.g. 0.10 = 10%
  auto_apply_allowed_fields: Tuple[str, ...] = ()
      # explicit allowlist — auto-apply of arbitrary fields is dangerous;
      # operator names exactly which fields (e.g. "profit_target_pct")
      # can flip without a manual review
  ```

### LLM prompt structure

Single call per day. Sonnet, temperature 0.2 (small variance is fine —
the operator sees the proposal before it lands).

```
TODAY'S TRADES  ({date})
  opens:     [list with ticker, strategy, credit, C/W, POP-at-open]
  closes:    [list with ticker, strategy, PnL, reason, dte-held]
  rejects:   [top-5 by count, with reason string + ticker distribution]

CURRENT PRESET
  <full PresetConfig as JSON>

WATCHLIST
  <current tickers>

MACRO CONTEXT
  VIX regime, market direction, sector performers/laggards

RECENT ALERTS
  <ExceptionMonitor rows from today>

TASK
Produce a JSON object with three sections:

{
  "observations": [
    "<one sentence patterns from today>",
    ...
  ],
  "preset_proposal": {
    "field": "profit_target_pct",
    "current": 0.50,
    "proposed": 0.55,
    "reason": "<one sentence>",
    "confidence": 0.0-1.0
  } | null,
  "watchlist_proposal": {
    "drops": [...],
    "adds":  [...],   // adds only when operator has previously
                      // shown interest in the ticker; do not invent
    "reason": "<one sentence per change>"
  } | null,
  "digest_lines": [
    "<3-6 short prose lines for Telegram>"
  ]
}
```

Fields the LLM is allowed to propose changes to are enumerated in
the skill file (a subset of PresetConfig fields — never things like
`max_risk_pct` which requires human judgment on account-level risk).

### Telegram digest format

```
📓 Daily Review — Mon 2026-09-28

Today: 3 opens · 2 closes · 4 rejects · P&L +$127

Observations
  • Bull-puts on tech names hit profit target within 3 days (avg $61 credit)
  • GLD rejects continue — 4 of 5 for per-leg spread cap; consider dropping
  • Defensive-roll fired on JPM; net negative $40 vs. straight close

Proposals
  • Preset: raise profit_target_pct 0.50 → 0.55  (conf 0.72)
    Reason: today's 2 winners both closed at 51% mark; +5% wringing out
    ~$8/contract extra without meaningfully changing hit rate

  • Watchlist: drop GLD (per-leg spread rejects, 4-of-5-cycles trend)

Apply with:
  python -m trading_agent.apply_preset_update <uuid>
```

### Conformance (skill 56)

- `daily_reviewer.py` never imports the executor or order-submission
  primitives (AST-verified — extends the skill 48 walker).
- Every LLM output is validated against a JSON schema; malformed
  output → log warning, skip the pending_update write, still emit
  digest with the "review-failed" line.
- `apply_preset_update.py` is the ONLY module besides
  `strategy_presets.py` that calls `save_active_preset()` (new
  invariant, added to `scan_invariant_check.py`).
- Auto-apply gate (3-predicate) exhaustive branch-tested.

---

## Rollout — one PR per phase

1. **Phase A — Reviewer + digest.**
   `daily_reviewer.py` + `pending_preset_updates_writer.py` +
   `journal_reader` multi-day extension + `PresetConfig` fields +
   config env var + launchd unit template. Runs when enabled, writes
   audit JSON, sends Telegram digest. `apply_preset_update.py` is a
   stub that prints the diff (no save yet).

2. **Phase B — Apply CLI.**
   `apply_preset_update.py` actually calls `save_active_preset` +
   `save_watchlist`. Manual-only (no auto-apply gate yet). Skill 56
   §Apply-flow populated.

3. **Phase C — Auto-apply gate + `/review` slash command.**
   3-predicate gate (env master + `auto_apply_allowed_fields` +
   change-size cap) with exhaustive branch tests. `.claude/commands/
   review.md` renders the last review from disk on demand.

4. **Phase D — Prompt evals + regression fixtures.**
   `evals/daily_reviewer/*.jsonl` — hand-curated day scenarios with
   expected proposal shapes. Snapshot tests over digest rendering.

---

## Open questions to resolve before Phase A

1. **LLM model.** Sonnet (recommended) vs Haiku. Sonnet cost ≈ $0.02
   per review at expected token counts; runs once daily so
   $0.40/month. Worth it for judgment quality.

2. **Ticker-add discovery.** Should the reviewer propose adds
   (e.g. "SMH looks like a good fit given today's regime")? **Recommend
   NO** for v1 — the LLM would need broader market data, and adds
   introduce novel risk. Drops are strictly journal-driven, safe to
   propose. Adds stay manual.

3. **Fields the LLM may propose.** Concrete allowlist for v1:
   `profit_target_pct`, `edge_buffer`, `min_pop`, `max_leg_spread_cents`,
   `defensive_roll_enabled`. NOT: `max_risk_pct`, `max_delta` (both
   need human account-level judgment). Full list finalized in the
   skill file.

4. **Weekend behavior.** Sat/Sun the reviewer has nothing to review.
   **Recommend skip** — launchd unit gated to weekdays.

5. **Cold start (first run, no prior reviews).** Reviewer runs
   normally; the digest just doesn't have a "vs. yesterday's
   proposal" section. **Recommend explicit "first run" flag in the
   digest header** so the operator sees why baseline comparisons are
   missing.

---

## Acceptance for Phase A PR

- [ ] `trading_agent/daily_reviewer.py` + `run()` entry point.
- [ ] `trading_agent/pending_preset_updates_writer.py`.
- [ ] `trading_agent/journal_reader.py` grows
      `closes_in_window(days)` + `opens_in_window(days)` methods
      (currently today-only).
- [ ] `PresetConfig` gets 3 `auto_apply_*` fields, all safe defaults.
- [ ] `config.py` gets `daily_reviewer_enabled` from env.
- [ ] `apply_preset_update.py` stub (prints diff, exits 0, doesn't
      save).
- [ ] launchd `.plist` template committed under `ops/launchd/`.
- [ ] `docs/skills/56_daily_journal_reviewer.md`.
- [ ] Conformance: no executor imports in reviewer; JSON schema
      validation; digest render deterministic.
- [ ] `.gitignore` covers `daily_reviews/` +
      `pending_preset_updates/`.
- [ ] All 8 existing CI gates green.

---

*Last updated: 2026-09-28. Author: Bala + Claude. Supersedes the
per-cycle-LLM plan from the same day — cost/value analysis made the
retrospective shape a better fit.*
