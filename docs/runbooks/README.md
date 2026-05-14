# Operational Runbooks

These are **action-oriented playbooks** for when something goes wrong (or might have). They're different from `docs/skills/`:

| | `docs/skills/*.md` | `docs/runbooks/*.md` |
|---|---|---|
| **Audience** | Code maintainers | Operators, on-call, future-you |
| **Question answered** | "Why does the code do this?" | "What do I do when I see this?" |
| **Format** | Theory → math → reference Python → edge cases | Symptom → diagnostic commands → expected output → remediation |
| **Update cadence** | When source-of-truth code changes | When a new failure mode is discovered |
| **Cross-linking** | Skills link to other skills | Runbooks link to skills (for "why") AND other runbooks (for follow-up) |

If you're trying to understand HOW the dedup gate works, read `docs/skills/17_close_failure_and_cooldown.md`. If you're trying to figure out WHY there's a stuck position on the broker right now, read `docs/runbooks/03_zombie_position_recovery.md`.

---

## When to read which runbook

| Symptom you're seeing | Runbook |
|---|---|
| "I'm about to start a fresh LLM session for diagnosis" | [`00_using_this_with_an_llm.md`](00_using_this_with_an_llm.md) |
| "Market just closed, did anything go wrong today?" | [`01_daily_close_review.md`](01_daily_close_review.md) |
| "The agent didn't submit any trades, why?" | [`02_zero_trades_diagnostic.md`](02_zero_trades_diagnostic.md) *(stub — to be written when first needed)* |
| "I have a partial-fill broken spread on the broker" | [`03_zombie_position_recovery.md`](03_zombie_position_recovery.md) |
| "What does this Alpaca/Schwab error code mean?" | [`04_broker_api_errors.md`](04_broker_api_errors.md) *(stub)* |
| "I think multiple agent processes are running" | [`05_ghost_process_diagnostic.md`](05_ghost_process_diagnostic.md) *(stub)* |
| "I want pre-tuned prompts for talking to ChatGPT / Claude / Gemini" | [`prompts.md`](prompts.md) |

---

## The runbook structure

Every runbook follows the same eight sections. This is so you can skim to the relevant part fast:

1. **When to use this** — the symptom or scheduled trigger that brings you here
2. **What you need first** — files, SSH access, dashboard visibility
3. **Step-by-step diagnostic** — exact commands, what to look for, what each output means
4. **Decision tree** — based on what you found, where to go next
5. **Remediation** — the actual fix or cleanup action
6. **Verification** — how to confirm the issue is resolved
7. **Prevention** — what to do differently next time
8. **LLM hand-off template** — paste-ready prompt for asking another LLM to help diagnose

That last section is what makes these usable across LLM vendors — see `00_using_this_with_an_llm.md` for the meta-pattern.

---

## Adding a new runbook

When you hit a new failure mode that took non-trivial time to diagnose:

1. Copy `_template.md` (or any existing runbook) as a starting point.
2. Number it (next free integer after `05`).
3. Add an entry to the symptom table above.
4. Cross-link from the runbook to the relevant `docs/skills/NN_*.md` files for the "why."
5. Cross-link from related runbooks (e.g. runbook 03 might suggest reading runbook 05 if the zombie was caused by ghost processes).

Each runbook is intentionally small — ~150-300 lines. If yours is growing past that, it's probably trying to cover multiple symptoms — split it.

---

## Why runbooks > "ask the LLM fresh every time"

Three real costs of debugging-from-scratch every session:

- **Context-rebuild tax.** Every fresh LLM session needs ~20-30 minutes to understand the journal schema, what `close_failed` means, what the recent incidents were. Runbooks compress that into reading time.
- **Diagnostic drift.** Without a written procedure, you'll subtly diagnose the same problem differently across sessions. The third investigation of "why didn't the agent trade today" should not take as long as the first.
- **Reproducibility.** Runbooks make the diagnostic commands greppable + commitable + reviewable. When a bug is fixed, the corresponding runbook can be deprecated or amended in the same PR.

The first runbook to read in any new context is always `00_using_this_with_an_llm.md` — it tells you what to attach, what to ask, and how to interpret the response.

---

*Created: 2026-05-13.  Living document — add new runbooks as failure modes are discovered.*
