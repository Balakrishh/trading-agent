---
description: Render the most recent daily journal review from disk (skill 56).
argument-hint: "[YYYY-MM-DD]"
---

Follow `docs/skills/56_daily_journal_reviewer.md`.

Arguments: `$ARGUMENTS` — optional ISO date. Empty → most recent
review in `daily_reviews/`.

Steps:

1. Read the target audit JSON. If `$ARGUMENTS` is empty, list
   `daily_reviews/*.json` in reverse-chronological order and read the
   first entry. If empty entirely, tell the operator "no reviews yet
   — run `python -m trading_agent.daily_reviewer_main` to produce one".
2. Render the digest exactly as it was sent to Telegram: reuse
   `trading_agent.daily_reviewer.render_telegram_digest(ctx, output,
   proposal_id)`. The audit JSON stores everything needed to rebuild
   both dataclasses.
3. Tell the operator whether there's a staged proposal
   (`pending_preset_updates/<uuid>.json`) they can apply with
   `python -m trading_agent.apply_preset_update <uuid>`.

Do NOT call `apply_preset_update` yourself. Do NOT invoke
`save_active_preset`. This command is read-only — it re-renders past
reviews so the operator can revisit them from Claude Code.
