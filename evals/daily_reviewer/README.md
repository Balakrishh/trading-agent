# Daily Reviewer — Prompt Evals

Curated day-scenario fixtures for skill 56. Each JSONL row is one
scenario:

```json
{
  "id": "<slug>",
  "description": "<what this tests>",
  "context": { ...ReviewContext-shaped payload... },
  "expected": { ...assertions about the LLM's output... }
}
```

## Running

Two modes:

- **Offline (default).** Runs the parser + rendering path over the
  scenarios' `expected` claims that don't require the LLM (parser
  filtering, cold-start labeling, digest structure). No network,
  no cost. This is what CI runs.

  ```bash
  pytest tests/eval/test_daily_reviewer_scenarios.py
  ```

- **Online.** Sends each scenario's `context` to the live LLM and
  asserts the parsed output matches `expected`. Opt-in per scenario
  via the `EVAL_DAILY_REVIEWER_ONLINE=true` env var. Costs one LLM
  call per scenario (~$0.02 each on Sonnet).

  ```bash
  EVAL_DAILY_REVIEWER_ONLINE=true \
    pytest tests/eval/test_daily_reviewer_scenarios.py
  ```

  The LLM is whatever `daily_reviewer.call_llm` uses: the Ollama /
  OpenAI-compatible client by default, or the logged-in `claude` CLI
  when `TRADING_AGENT_REVIEWER_LLM=claude_cli` (skill 56 §3.5.1 — the
  Pi's reviewer service sets this). With `claude_cli`, **point the
  session file and working directory somewhere else**, or the evals
  resume the live reviewer's day-to-day session and land in its
  history:

  ```bash
  EVAL_DAILY_REVIEWER_ONLINE=true TRADING_AGENT_REVIEWER_LLM=claude_cli \
  TRADING_AGENT_REVIEWER_SESSION=/tmp/eval_reviewer_session.json \
  TRADING_AGENT_REVIEWER_WORKDIR=/tmp/eval-reviewer \
    pytest tests/eval/test_daily_reviewer_scenarios.py
  ```

  Each scenario then continues one eval session (the reviewer keeps
  context across calls by design); delete the session file for
  independent runs.

Online runs are excluded from CI — flake risk from LLM sampling isn't
worth the signal. Run them locally when tuning the prompt or the
allowlist.

## Adding a scenario

1. Append a JSONL row.
2. Set `expected` to the invariant claim (a proposal field name, a
   digest substring, a `watchlist_drops_contains` ticker, etc.).
3. Run the offline suite first; if it passes but you want LLM
   verification, run online once and pin any additional expected
   fields.

## Existing scenarios

| id | tests |
|---|---|
| `winners_at_51_pct` | LLM proposes raising `profit_target_pct` when both closes hit 51% |
| `chronic_wide_spread_ticker` | LLM proposes watchlist drop for a repeatedly-rejected ticker |
| `quiet_day_no_changes` | Empty day produces no proposals |
| `hallucinated_max_risk_pct` | Parser drops forbidden fields (offline-only) |
| `cold_start` | Digest labels first-ever run |
