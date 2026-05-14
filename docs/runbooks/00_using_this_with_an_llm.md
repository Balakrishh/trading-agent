# Runbook 00 — Using This Codebase With Another LLM

> **The meta-runbook.** Read this first if you're about to start a fresh AI session (Claude, ChatGPT, Gemini, etc.) to debug or analyse anything in this repo. Following the pattern below cuts the LLM's context-build time from ~20-30 min to ~3 min and makes its diagnoses much more reliable.

---

## 1. The four-file briefing

A fresh LLM session needs four things in its context, in this order. Attach them as files (most chat UIs support drag-drop) or paste their contents inline:

1. **`PROJECT_CONTEXT.md`** — the long-form briefing (~900 lines). Covers architecture, invariants, conventions, and the full change-log of resolved technical debt. The LLM needs this to know what to NOT re-litigate.
2. **`CLAUDE.md`** — the short, immutable rules for AI contributors. Hard rules (CI invariants), soft rules (patterns), and the "definition of done" checklist.
3. **The relevant runbook** for your symptom (see the table in [`README.md`](README.md)).
4. **The data slice** the runbook calls out — usually a journal snippet or log tail (see §2).

Don't attach the whole codebase. Most LLMs degrade past ~50 KB of context. The four files above are ~30 KB total; the runbook+data add another ~10-30 KB.

---

## 2. How to package the data slice

For most diagnostic sessions, the LLM needs **today's journal + relevant log lines**, not the whole 5 MB of history. The standard package:

```bash
# On your Mac, after syncing pi-diagnostics:
cd ~/pi-diagnostics

# Latest 200 journal rows
tail -200 trade_journal/signals_live.jsonl > /tmp/journal_recent.jsonl

# Today's log only (assumes Pi is in America/New_York)
TODAY=$(date -I)
grep "^${TODAY}" logs/trading_agent.log > /tmp/agent_log_today.log

# Just the ERROR + WARNING lines from the full log
grep -E "ERROR|WARNING" logs/trading_agent.log | tail -100 > /tmp/agent_warnings.log

ls -lh /tmp/journal_recent.jsonl /tmp/agent_log_today.log /tmp/agent_warnings.log
```

Attach the three `/tmp/*.log*` files to your chat. Total size ~50-100 KB.

---

## 3. The standard opening prompt

Paste this verbatim into a fresh chat after attaching the four files:

```
I'm working on a credit-spread options trading agent. Before answering my question, please:

1. Read PROJECT_CONTEXT.md in full — pay particular attention to §2 (hard invariants) and §7 (pitfalls).
2. Read CLAUDE.md to understand the hard rules.
3. Read the runbook I've attached for the symptom I'm reporting.

Confirm you've read them by listing:
- The three CI-enforced invariants (PROJECT_CONTEXT §2.1-2.3)
- The runbook's recommended diagnostic steps
- What you'll need from me beyond the attached files

Then I'll describe my actual question.
```

Why this works: it forces the LLM to actually read the briefing before generating output, and gives you a verification gate — if it can't list the invariants correctly, it didn't read them. Restart with the same prompt if so.

---

## 4. The standard follow-up prompt (after the LLM confirms it's read the brief)

```
Symptom: <one-paragraph description of what you're seeing on the dashboard / in the journal / at the broker>

Data attached:
- /tmp/journal_recent.jsonl — last 200 journal rows
- /tmp/agent_log_today.log — today's full agent log
- /tmp/agent_warnings.log — ERROR + WARNING lines from the full log

Please follow runbook §3 (Step-by-step diagnostic). Report your findings as:
1. What the data shows happened (chronological)
2. The hypothesis you'd assign with what % confidence
3. The single command you'd run next to confirm
4. The remediation step if confirmed

Do NOT propose code changes yet. Diagnose first, fix second.
```

The "no code changes yet" rule is important. LLMs love to immediately propose code edits. Diagnose first → confirm with one more piece of data → then propose. Skipping the confirm step is how you get fixes for the wrong bug.

---

## 5. Per-LLM-vendor adjustments

**Claude (this conversation's home).** Already has file-reading + bash tools in most surfaces. Works best with the standard prompt above. If you're in Claude Code or Cowork mode, the LLM can run the diagnostic commands itself against `pi-diagnostics/` — let it.

**ChatGPT (with Code Interpreter / Advanced Data Analysis).** Same prompt works. The Code Interpreter sandbox can run Python directly on the attached files — pre-fix the runbook commands to use Python instead of grep/awk. Example: `pandas.read_json(file, lines=True)` instead of `cat | grep`.

**Gemini.** Larger context window helps — you can attach the WHOLE journal (not just tail-200). Gemini sometimes hallucinates that it ran a command when it didn't; add "show me the raw command output before you interpret it" to the follow-up prompt.

**Local Ollama / LM Studio.** Smaller models (7B-13B) struggle with the full briefing. Pre-summarise: instead of attaching `PROJECT_CONTEXT.md`, paste the cross-LLM handoff prompt from `PROJECT_CONTEXT.md` §12 only. Cuts the context budget by 80%.

---

## 6. What to do with the LLM's output

If you got a hypothesis with high confidence + a specific confirmation command:

1. Run the confirmation command yourself (don't trust the LLM to run it inside the chat unless you verify the output).
2. Paste the actual output back.
3. The LLM will either confirm and propose remediation, OR pivot to a different hypothesis.

If the LLM proposed a code change immediately (skipping diagnosis):

1. Don't apply it. Re-prompt with "Diagnose first. What's the data evidence that this code is the root cause?"
2. If the LLM still can't ground the proposal in journal/log evidence, the proposal is speculative — discard.

---

## 7. After the diagnostic — closing the loop

Once you've fixed the issue:

1. If the fix changed code, the LLM should have updated the relevant skill file under `docs/skills/` (per `CLAUDE.md`'s "definition of done"). Verify that happened.
2. If the symptom matched an existing runbook, no documentation work needed.
3. If the symptom did NOT match an existing runbook, **create one** before closing the session. New runbook = next free integer + entry in `README.md`. Even a stub captures the diagnostic path you just walked — invaluable for the next occurrence.

---

## 8. LLM hand-off template — using this runbook

This is the meta-template. Paste into a fresh chat:

```
I'm starting a diagnostic session on the trading-agent repo. Before I describe my question, follow this 4-step briefing:

1. Read the attached PROJECT_CONTEXT.md fully. Confirm by listing the three CI invariants from §2.1-2.3.
2. Read CLAUDE.md. Confirm by stating what the "Definition of done" checklist requires.
3. Read docs/runbooks/00_using_this_with_an_llm.md and tell me you've internalised §3 (the standard prompt) and §6 (what NOT to do with your output).
4. Read the runbook for my specific symptom (attached separately): docs/runbooks/<NN>_<symptom>.md

After the four-step briefing, ask me what symptom I'm seeing. Then we'll proceed with §4 (the follow-up prompt) from runbook 00.

Hard rules:
- Diagnose before you propose code. No code edits until I confirm a hypothesis.
- Cite specific journal rows or log lines for every claim.
- If you can't ground a hypothesis in attached data, say so explicitly.
```

Adopt this verbatim. The four-step briefing forces the LLM to internalise the conventions; the "no code edits before diagnosis" rule prevents the most common AI failure mode in debugging.

---

## 9. Common anti-patterns to avoid

- **Don't paste the journal as plain text in chat.** It's noisy and the LLM will skim. Attach as a file so it can scan systematically.
- **Don't ask "what's wrong with my agent?"** Too vague. Ask "the dashboard shows X for ticker Y — please trace through runbook NN."
- **Don't accept a fix without a regression test.** If the LLM proposes a code change, ask for the test that pins the bug fixed. CLAUDE.md requires this; the LLM should know.
- **Don't run two LLM sessions in parallel on the same diagnostic.** They'll diverge and you'll waste time reconciling.
- **Don't paste API keys into the chat.** Ever. Strip `.env` and `~/.schwab_tokens.json` from any attached folder before sharing. The `pi-diagnostics/` package per runbook 01 §2 is curated to exclude these by default.

---

*Last verified against repo HEAD on 2026-05-13.*
