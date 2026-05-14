# LLM Prompt Library

Pre-tuned prompts for working with another AI on this codebase. Use these as-is or adapt — they encode the patterns that have worked well in actual diagnostic sessions.

For the meta-context on how to brief a new LLM, see [`00_using_this_with_an_llm.md`](00_using_this_with_an_llm.md). This file is just the prompts themselves, ready to copy.

## Table of contents

**Diagnostic prompts (read-only):**

- [§1 Standard onboarding](#1-the-standard-onboarding-prompt) — required first prompt for any fresh LLM session
- [§2 Daily close review](#2-daily-close-review-prompt) — end-of-day post-market analysis
- [§3 Specific-symptom diagnosis](#3-specific-symptom--x-is-acting-weird-prompt) — when you have an observation but don't know which runbook applies

**Repo-change prompts (the LLM will write code / docs / tests):**

- [§4.1 Fix this bug](#41--fix-this-bug-prompt) — targeted bug fix with regression test
- [§4.2 Add a new feature](#42--add-a-new-feature-prompt) — skill-first SDLC for new capabilities
- [§4.3 Add a new PresetConfig field](#43--add-a-new-presetconfig-field-prompt) — the 5-touchpoint propagation
- [§4.4 Refactor an area](#44--refactor-an-area-prompt) — structural change with zero behavior change
- [§4.5 Update skill NN to match new code](#45--update-skill-nn-to-match-new-code-prompt) — when docs drift from source
- [§4.6 Review my PR](#46--review-my-pr-prompt) — independent rule-compliance check
- [§4.7 Add a new runbook](#47--add-a-new-runbook-prompt) — capture a new failure mode for future-you

**Higher-level prompts:**

- [§5 Strategy review](#5-strategy-review--should-i-change-my-preset-prompt) — parameter tuning grounded in journal data
- [§6 Zombie cleanup walkthrough](#6-zombie-cleanup--walk-me-through-this-prompt)
- [§7 Ghost-process triage](#7-ghost-process-triage-prompt)
- [§8 Deployment validation](#8-deployment-validation-prompt)
- [§9 New-failure investigation](#9-new-failure-mode-investigation-prompt)
- [§10 Quick-paste mini-prompts](#10-quick-paste-mini-prompts) — for recurring small questions

---

## 1. The standard onboarding prompt

Paste this verbatim to a fresh chat after attaching `PROJECT_CONTEXT.md` + `CLAUDE.md`:

```
I'm working on a credit-spread options trading agent on Raspberry Pi.
Before I describe my actual question, please complete this briefing:

1. Read PROJECT_CONTEXT.md (~900 lines). Focus on:
   - §2 hard invariants (CI-enforced architectural rules)
   - §7 pitfalls (real bugs that have shipped before)
   - §10 entries 17 and 18 (recent production-readiness fixes)
2. Read CLAUDE.md. Internalise the "Definition of done" checklist
   and the soft conventions (frozen dataclasses, journal-derived
   state, atomic temp+rename, etc.).

Confirm you've internalised the briefing by listing:
- The three CI invariants from PROJECT_CONTEXT.md §2.1-2.3
- The four-item "Definition of done" checklist
- The three production-failure-mode fixes from May 13 (PROJECT_CONTEXT §10 entry 18)

After that, I'll describe my question.

Hard rules for our session:
- Cite specific file:line or journal rows for every claim
- Diagnose before proposing code; don't skip to fix
- If you can't ground a hypothesis in attached data, say so explicitly
- Don't paraphrase the codebase — quote it
```

**Why this works:** the verification step (list the invariants, list the checklist) acts as a comprehension gate. If the LLM can't reproduce the invariants, it didn't actually read the brief — restart.

---

## 2. Daily close review prompt

Use after market close. Attach the three diagnostic files from runbook 01 §2.

```
Run the daily close review per docs/runbooks/01_daily_close_review.md
against the attached files.

Attached:
- pi-diagnostics/trade_journal/signals_live.jsonl
- pi-diagnostics/logs/trading_agent.log
- pi-diagnostics/AGENT_LOG

Today's date for filtering: YYYY-MM-DD  ← set this

Report findings in EXACTLY this structure:

## Action histogram
[Counter output, one line per action]

## Daily P&L
- Start equity: $X
- End equity: $Y
- Net delta: $Z
- Realized P&L breakdown by close (ticker, P&L, exit signal)

## Anomaly flags (non-zero only)
- close_failed: N rows
- warning: N rows, sources: [...]
- ghost-process clusters: N buckets
- chain-empty rejections: N rows
- error: N rows

## Per-ticker EOD state
[ticker | open positions | unrealized P&L]

## Recommended next steps
1. [specific runbook or action]
2. [specific runbook or action]

Do not propose code changes. This is observational diagnosis only.
End with: "Should I dig deeper into any of these?" — wait for my reply.
```

---

## 3. Specific symptom — "X is acting weird" prompt

When you have a specific observation but don't know yet which runbook applies. Attach today's journal slice.

```
I'm observing the following symptom on my trading agent:

<one paragraph: what you see on the dashboard, what you expected to see,
when it started, what changed recently if anything>

Attached:
- pi-diagnostics/trade_journal/signals_live.jsonl (last 200 rows)
- pi-diagnostics/logs/trading_agent.log (today only)

Please:

1. Identify the most likely runbook from docs/runbooks/README.md
   that addresses this symptom. State your confidence (low/med/high).
2. Walk through that runbook's §3 (step-by-step diagnostic) against
   the attached data. For each step, quote the actual command output
   you'd expect to produce — don't say "run this command" without
   showing me what it returns from the attached files.
3. Report findings using the runbook's §4 (decision tree).
4. Stop before §5 (remediation). I want to confirm the hypothesis
   before taking any action.
```

---

## 4. Repo-change prompts (code, docs, configuration)

Every prompt in this section assumes the LLM has already completed the §1 onboarding briefing. The shared invariant: **diagnose / understand before you propose code**, and **respect the cross-cutting hooks table in CONTRIBUTING.md §1 step 2**.

When you use any prompt below, attach:

- `PROJECT_CONTEXT.md` and `CLAUDE.md` (always)
- The skill file(s) most relevant to your change
- Recent git history (`git log --oneline -10` output) so the LLM knows the codebase's recent direction
- Any specific files you expect to touch (or want the LLM to consider touching)

---

### 4.1 — "Fix this bug" prompt

Use when you've ALREADY diagnosed the root cause and want code surgery. The diagnostic-first rule still applies — you've done it, so attach the evidence.

```
I've diagnosed a bug. Need a fix shipped following CLAUDE.md's SDLC.

Symptom: <one sentence>

Root cause: <one paragraph; cite the file:line of the bug if you've found it,
and the journal/log evidence that proves it>

Proposed fix (high-level): <one sentence; or "you propose" if open-ended>

Cross-cutting hooks I think this touches (cite the row in CONTRIBUTING.md §1
step 2 — or "I don't believe this touches any cross-cutting hook"):
<your list>

Required deliverables:
1. The code change (cite file:line; under 50 LOC if surgical)
2. A regression test in tests/test_<module>.py that pins the bug fixed
   — the test MUST fail without the fix and pass with it
3. Updated docs/skills/NN_*.md §3 (Reference Python) for any quoted source
4. Re-stamped "Last verified" footer of every touched skill
5. Output of: python scripts/checks/scan_invariant_check.py
6. A 2-line commit message: "<title>" + 1-2 sentence "why"

Hard rules:
- Don't refactor unrelated code in the same PR
- Don't add new top-level files unless explicitly required
- Don't paraphrase or summarise the codebase — quote it verbatim
- If the fix requires touching > 3 source files (excluding tests + docs),
  stop and explain why before proceeding
```

---

### 4.2 — "Add a new feature" prompt

Use when adding a capability that didn't exist before (a new strategy variant, a new dashboard panel, a new market-data adapter, etc.). This prompt forces the LLM to start with the skill file, NOT with the code.

```
I want to add a new feature to the trading agent.

Feature description: <one paragraph; what it does, who uses it, why it matters>

Per CLAUDE.md §"Before you write any code" and CONTRIBUTING.md §1 step 1
("Write or update the skill file FIRST"), please complete these steps
IN THIS ORDER. Do not skip ahead.

Step 1 — Skill identification.
  Which existing docs/skills/NN_*.md documents the concept this feature
  modifies or extends? If none exists, what skill number would the new
  one take? Quote the relevant sections of existing skills you've identified.

Step 2 — Cross-cutting hook walkthrough.
  Run through the table in CONTRIBUTING.md §1 step 2.  For each row,
  state whether this feature touches it AND what specifically needs to
  update in each affected component. Don't skip rows — state "not touched"
  explicitly for rows that don't apply. Pay special attention to:
    - C/W floor formula (3 files must stay identical)
    - Scoring math (only allowed in chain_scanner.py + decision_engine.py)
    - Backtest seam (backtest_ui.py must contain `decide(` call)
    - RegimeAnalysis fields (append-only)
    - PresetConfig fields (5-touchpoint propagation)
    - New journal action values (dedup-bypass + UI surface)

Step 3 — Skill draft.
  Draft §1 (theory) and §4 (edge cases) of the new/updated skill
  BEFORE writing any code. Cite the existing source-of-truth lines
  the new code will need to integrate with.

Step 4 — Stop and wait for my review.
  Don't write code yet. Show me steps 1-3 first. After I approve,
  proceed to §3 (reference Python) of the skill + the actual code
  + the regression tests + the invariant scan.

Hard rules carried from CLAUDE.md:
- New tunables MUST be PresetConfig fields, not class-level constants
- Frozen dataclasses; mutate via dataclasses.replace
- Append-only dataclass fields
- Sentinel pattern (*_signal_available: bool) for fields that can be absent
- Atomic temp+rename for sentinel files

Don't ship code until I've seen the skill draft. Surface trade-offs
before writing.
```

---

### 4.3 — "Add a new PresetConfig field" prompt

The 5-touchpoint propagation for new preset fields trips up most LLMs. This prompt walks them through it.

```
I want to add a new field to PresetConfig: <field_name>: <type> = <default>

Purpose: <one sentence describing what this knob controls>

Per CLAUDE.md §"Definition of done" — when adding a PresetConfig field
the change must touch FIVE specific places. Please walk through them
in this exact order:

1. trading_agent/strategy_presets.py — add the field with its default
   value to PresetConfig. The default MUST keep existing
   STRATEGY_PRESET.json files loadable without modification.

2. trading_agent/strategy_presets.py — update PresetConfig.to_summary_line()
   so the field appears on the dashboard status line.

3. trading_agent/streamlit/live_monitor.py:849+ — surface the field
   in the Strategy Profile expander panel (slider / selectbox / text
   input depending on type).

4. trading_agent/agent.py:169-204 (TradingAgent.__init__) — thread
   the field from self.preset.<field_name> through to whichever
   component(s) consume it (StrategyPlanner, RiskManager,
   chain scanner, etc.).

5. docs/skills/13_preset_system_hot_reload.md — document the field
   in §3 (Reference Python). Re-stamp the footer.

Plus the standard deliverables:
- Regression test in tests/ that verifies the field's behavior end-to-end
- python scripts/checks/scan_invariant_check.py output
- One built-in preset (CONSERVATIVE/BALANCED/AGGRESSIVE) MUST be updated
  to a non-default value of this field, to prove the propagation works
  through to runtime
- A 2-line commit message

For each of the 5 touchpoints, show me your proposed diff BEFORE
applying. I want to verify the propagation is complete.

Anti-pattern to avoid: adding the field as a class-level attribute
on StrategyPlanner or any other consumer. CLAUDE.md soft rule:
"new tunables go in PresetConfig, not class constants."
```

---

### 4.4 — "Refactor an area" prompt

Use when no behavior change is intended but the structure needs work. Refactors are dangerous because they're invisible to users — only structural — but they can silently break invariants.

```
I want to refactor an area of the codebase. No behavior change is intended.

Refactor scope: <one paragraph describing what you want to change structurally>

Hard refactor rules carried from CLAUDE.md:
- The three CI invariants must continue to hold:
  1. Single C/W floor formula in chain_scanner.py + risk_manager.py + executor.py
  2. Scoring primitives only defined in chain_scanner.py + decision_engine.py
  3. backtest_ui.py contains a literal `decide(` call
- No behavior changes — only structure. If you think a behavior change
  is required by the refactor, stop and tell me; we'll split into two PRs.
- Public API surface stays unchanged (callers shouldn't have to update).
- All existing tests must continue to pass with zero edits to them.

Please:

1. Inventory what the refactor will touch. List every file + section
   that needs to change. State which CI invariants each touches.

2. Identify the "before" structure clearly. Cite the existing
   source-of-truth lines.

3. Propose the "after" structure as a skill-file edit FIRST.
   Update the relevant docs/skills/NN_*.md §3 (Reference Python) to
   show what the new shape looks like.

4. Run python scripts/checks/scan_invariant_check.py BEFORE the
   refactor and report the baseline.

5. Apply the refactor.

6. Run the same invariant scan again — output must be identical.

7. Re-run the existing tests — ZERO must require modification.
   If any do, the refactor is changing behavior and we need to
   reassess.

8. Re-stamp the "Last verified" footer of every touched skill.

Stop and ask before proceeding with step 5 (the actual code change).
I want to see steps 1-4 first.
```

---

### 4.5 — "Update skill NN to match new code" prompt

Use when you've changed source code in a PR and need the corresponding skill file to catch up. (CLAUDE.md "Definition of done" requires this BUT you might miss it in the heat of the moment.)

```
I changed code in <file>:<line_range> and need the corresponding skill
file to catch up. The skill is currently stale relative to the source.

Changed source: <paste the new code or git diff>

Per CLAUDE.md "Definition of done" checklist:
  - The skill file(s) under docs/skills/ documenting the affected
    concept have been updated. §3 (Reference Python) reflects the
    new code; §4 (Edge Cases) covers any new failure modes.
  - The "Last verified against repo HEAD on YYYY-MM-DD" footer of
    every touched skill has been re-stamped to today's date.

Please:

1. Identify the relevant skill file(s). Search docs/skills/ for any
   file that quotes (in §3) the source lines I changed.

2. Update §3 of each affected skill to match the new source. Quote
   the new code verbatim, including comments. Update the
   `Source of truth` header at the top if line numbers shifted.

3. Review §4 (Edge Cases). If the source change introduces a new
   failure mode or removes an existing one, update §4 accordingly.

4. Review §1 (Theory). If the rationale for the design changed,
   update §1. If not, leave it alone.

5. Re-stamp the "Last verified" footer to today's date (use
   `date -I` output, format YYYY-MM-DD).

6. If the skill cross-references other skills whose §3 quotes also
   need updating because of cascading impact, list them and ask
   me whether to update those too (don't auto-cascade — sometimes
   the cascading update is its own PR).

Diff format: show me each skill change as a unified diff so I can
review before you write to disk.
```

---

### 4.6 — "Review my PR" prompt

Independent second-pair-of-eyes review before you commit / push.

```
Review the attached git diff against CLAUDE.md and CONTRIBUTING.md rules.
This is an independent check — assume I might have missed something.

Attached:
- git diff of the change
- Any newly-added/modified test files
- Any newly-added/modified skill files

Please check, in order:

## CI invariants (CLAUDE.md hard rules)
1. C/W floor formula still identical in all 3 files (chain_scanner.py,
   risk_manager.py, executor.py)? If the diff touched the floor,
   verify all 3 changed identically.
2. Scoring primitives still defined only in chain_scanner.py +
   decision_engine.py? No new `_score_candidate*` or `_quote_credit`
   definitions elsewhere?
3. backtest_ui.py still contains a literal `decide(` call?
   `grep -c 'decide(' streamlit/backtest_ui.py` should be >= 1.

## Definition of done (CLAUDE.md checklist)
- Skill file(s) updated to match new code?
- Skill footer(s) re-stamped to today?
- PresetConfig field touchpoints all 5 covered (if a new field was added)?
- Unit tests cover every §4 edge case from the relevant skill?
- Tests actually fail without the change and pass with it?
- Commit message follows repo convention?

## Soft conventions (CLAUDE.md)
- Frozen dataclasses with `dataclasses.replace` (no field assignment)?
- Append-only on RegimeAnalysis / SpreadPlan / SpreadCandidate / WatchlistRow?
- Sentinel pattern (*_signal_available) for fields that can be absent?
- Per-overlay try/except in macro overlay code?
- Atomic temp+rename for sentinel files (not in-place writes)?
- New tunables in PresetConfig (not class-level constants)?
- Logging level discipline (info / warning / error / debug correct)?

## Scope
- Any incidental refactors that should be in a separate PR?
- Any files touched outside the obvious scope of the change?
- Commit message accurately reflects what changed?

Output format:
🟢 = pass
🟡 = concern, explain
🔴 = blocker, must fix

For every 🔴 / 🟡, cite the specific file:line and what's wrong.
Don't approve — just report findings.
```

---

### 4.7 — "Add a new runbook" prompt

When you've hit a failure mode that doesn't fit any existing runbook and want a new one drafted.

```
I hit a new failure mode that took non-trivial time to diagnose.
Please draft a new runbook so the next occurrence is faster.

Symptom: <one paragraph>

Diagnostic path I followed: <what you actually did to find the root cause>

Root cause (confirmed): <what was actually wrong>

Remediation: <what fixed it>

Attached:
- The journal slice that showed the symptom
- The agent log slice that surfaced the root cause
- pi-diagnostics/ as it looked when the issue was active

Please:

1. Identify the next free runbook integer from docs/runbooks/ (current
   filled: 00, 01, 02, 03, 04, 05).

2. Draft the runbook following the 8-section structure used in
   01_daily_close_review.md and 03_zombie_position_recovery.md:
     §1 When to use this
     §2 What you need first
     §3 Step-by-step diagnostic (with exact commands + expected output)
     §4 Decision tree
     §5 Remediation
     §6 Verification
     §7 Prevention (link to any code fix that should land + skill update)
     §8 LLM hand-off template
     §9 Related runbooks + skills

3. Update docs/runbooks/README.md:
     - Add a row to the symptom-to-runbook table
     - Update the "Currently filled" range in the "Adding a new runbook" section

4. Stamp the "Last verified" footer at the bottom with today's date.

5. The runbook should be SELF-CONTAINED — a future LLM session reading
   ONLY this runbook + PROJECT_CONTEXT.md should be able to diagnose
   this symptom without our chat context. Don't reference "what we
   discussed" — bake the necessary context into the runbook itself.

Target length: 150-300 lines. If yours grows past that, consider whether
it's trying to cover multiple symptoms — split into two.
```

---

## 5. Strategy review — "should I change my preset?" prompt

Higher-level: not debugging a bug, evaluating performance. Attach the last 30 days of journal.

```
I want a strategy review of the last 30 days of trading.

Attached:
- pi-diagnostics/trade_journal/signals_live.jsonl (full file)
- The active STRATEGY_PRESET.json

Analyse and report:

## Trade-level statistics
- Total submitted: N
- Total closed (complete): N
- Total close_failed: N (zombie incidents — flag if > 2)
- Average winning trade P&L
- Average losing trade P&L
- Win rate
- Largest single loss

## Per-ticker performance
[ticker | trades | win rate | net P&L]

## Why-not-traded analysis
- % of cycles where adaptive scanner sat out for "No positive-EV"
- % where RSI gate blocked
- % where defense_first blocked (high IV)
- % where directional bias filter blocked

## Strategy preset evaluation
Given the data, would you adjust:
- edge_buffer (current value: <X>)? toward what?
- min_pop (current: <X>)?
- max_delta (current: <X>)?
- DTE targets?

Show me the historical evidence for each suggested change.
DO NOT make changes yet — this is an analysis pass.

I am NOT asking for investment advice. I'm asking for parameter-tuning
recommendations grounded in observed agent behavior.
```

The disclaimer at the bottom matters — without it, some LLMs reflexively refuse "should I change my preset" as a financial-advice question. Framing as parameter tuning of an algorithmic system gets you usable output.

---

## 6. Zombie cleanup — "walk me through this" prompt

When you have an active zombie position and need step-by-step guidance.

```
I have a zombie position on ticker <TICKER> that needs manual cleanup.

Please walk me through docs/runbooks/03_zombie_position_recovery.md
against the attached data.

Attached:
- pi-diagnostics/trade_journal/signals_live.jsonl
- pi-diagnostics/logs/trading_agent.log

Specifically:

1. Run §3.1 — confirm the ticker is in the close_failed set.
2. Run §3.2 — list the contract symbols that are still open at the broker.
3. Run §3.4 — decode the Alpaca error codes for each failed leg.
4. Tell me the EXACT order to close the legs in (short legs FIRST per §5).
5. Tell me how many legs I expect to see at Alpaca after each close.

Format: a numbered cleanup checklist I can work through in Alpaca's UI.

After cleanup, I'll re-attach a fresh journal and ask you to run §6.
```

---

## 7. Ghost-process triage prompt

When `pgrep` shows multiple agent processes or the journal has same-second clusters.

```
I think I have multiple agent processes running on the Pi.

Attached:
- pi-diagnostics/trade_journal/signals_live.jsonl
- pi-diagnostics/AGENT_LOG

Please:

1. Confirm whether there are multiple processes from the journal
   evidence — count same-second/same-ticker clusters in the last
   24 hours and report.
2. Estimate how many distinct processes are running by looking at
   the time offsets between cycle starts in AGENT_LOG.
3. Identify the most likely cause from:
   - Two Streamlit processes (each with its own daemon thread)
   - Streamlit + cron both launching cycles
   - systemd service + dashboard-click stacking
   Use the AGENT_LOG cycle-start patterns to discriminate.
4. Give me the cleanup commands per docs/runbooks/05 (or the
   equivalent — kill streamlit, kill agent, clear sentinels, verify clean).

Then I'll run the cleanup commands and re-pull diagnostics. We'll re-verify.
```

---

## 8. Deployment validation prompt

After deploying a code change or migrating to a new host. Attach `pi-diagnostics/` from the new environment + the relevant commit diff.

```
I just deployed a code change to my trading agent (running on Pi).
Please verify it landed cleanly.

Attached:
- pi-diagnostics/ (fresh from the deployed host)
- git diff <commit_sha>...HEAD (the changes I just deployed)

Verify:

1. The change actually loaded — check the agent log for the deployed
   version's distinctive log line / new behavior.
2. No new ERROR or WARNING entries that weren't present before.
3. Cycle cadence is healthy: cycles fire ~every 5 min, none have
   exited with a non-zero code, no after-hours blackouts during
   market hours.
4. The journal action histogram for today is consistent with the
   pre-deploy histogram (no sudden shift in skipped_existing rate,
   rejection rate, etc.).
5. Process count: exactly one streamlit, at most one agent.

Report green/red for each of the 5 checks. If anything is red,
propose the minimum diagnostic command to investigate.
```

---

## 9. New-failure-mode investigation prompt

When you've hit something the runbooks don't cover and want to expand the library.

```
I've hit a failure mode that doesn't match any existing runbook in
docs/runbooks/. Please help diagnose it AND draft a new runbook for
future occurrences.

Symptom: <description>

Attached:
- pi-diagnostics/

Please:

1. Investigate using the same diagnostic patterns as the existing
   runbooks (action histogram, log greps, broker cross-check).
2. Identify the root cause with citations.
3. Draft a new runbook following the 8-section structure used in
   docs/runbooks/01 and 03. Use the next free integer for the
   filename. Include:
   - §1 When to use this
   - §2 What you need first
   - §3 Step-by-step diagnostic
   - §4 Decision tree
   - §5 Remediation
   - §6 Verification
   - §7 Prevention
   - §8 LLM hand-off template
   - §9 Related runbooks + skills
4. Update docs/runbooks/README.md to add a row in the symptom table.

The runbook should be self-contained — a future LLM session reading
ONLY that runbook + PROJECT_CONTEXT.md should be able to diagnose
this symptom without our chat context.
```

---

## 10. Quick-paste mini-prompts

Short prompts for specific recurring questions where the long ones are overkill.

**"What does this error code mean?"**

```
What does Alpaca error code <NNNNNNNN> "<message>" mean?
Cite docs/runbooks/04_broker_api_errors.md if I have a reference for it.
Tell me the typical operator response.
```

**"Why is GLD/SPY/etc. doing X today?"**

```
Trace what happened with <TICKER> today in the attached journal +
log slice. Walk me through the day chronologically:
- Stage 1 (monitor) decisions
- Stage 2 (open) decisions and rejections
- Any close attempts and their outcomes
- Where it ended the day

Don't propose action. Just narrate.
```

**"Is this safe to commit?"**

```
Review the attached git diff. Check:
1. Are CLAUDE.md hard rules respected? (CI invariants, skill updates,
   regression tests, footer re-stamps)
2. Are CLAUDE.md soft conventions followed? (frozen dataclasses,
   journal-derived state where applicable, atomic writes, Optional[]
   over magic defaults)
3. Is the change as small as it could be? Any incidental
   refactors that should be in a separate PR?
4. Do the tests actually pin the behavior the fix targets?

Don't approve; just list red/yellow/green findings.
```

**"Generate a commit message"**

```
Generate a commit message for the attached git diff.

Format:
- 1-line title (under 70 chars)
- Blank line
- 2-4 bullet points describing the why (not the what — the diff
  shows the what)
- If a CLAUDE.md invariant or skill file was touched, note it

Match the tone of existing commit messages in `git log --oneline -20`.
```

---

## How prompts are versioned

This file is a living document. When you discover a pattern that produces noticeably better LLM output, add it here. When a prompt becomes stale (e.g., references a deleted file), remove or update it.

To propose a new prompt:

1. Add it as a new section, numbered after the existing prompts.
2. Include a short note on WHEN to use it and WHY it works.
3. Test it against at least one fresh LLM session before committing.

To deprecate a prompt:

1. Move it to a "Deprecated" section at the bottom (don't delete — old prompts may be referenced in PRs and chat history).
2. Note why and what to use instead.

---

*Last updated: 2026-05-13.*
