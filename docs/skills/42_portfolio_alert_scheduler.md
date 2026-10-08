# Portfolio alert scheduler

> **One-line summary:** Hourly portfolio digest folded INTO the spread agent's cycle (no separate cron / Cowork task). Runs the long-term evaluator against the operator's persisted holdings + watchlist, fetches option chains from **Schwab by default** (better coverage than Alpaca's `indicative` feed for the income-overlay scorer), posts covered-call candidates + skipped reasons + watchlist entry-candidate universe + portfolio snapshot to the dedicated Telegram `long_term` channel. Same `journal_kb` body-hash dedup the trade alerts use, so an identical digest within a UTC day stays silent.
> **Source of truth:** [`trading_agent/portfolio_alert_scheduler.py`](../../trading_agent/portfolio_alert_scheduler.py), [`trading_agent/telegram_notifier.py:notify_portfolio_review`](../../trading_agent/telegram_notifier.py), [`trading_agent/agent.py:_maybe_run_portfolio_review`](../../trading_agent/agent.py).
> **Phase:** 2  •  **Group:** ops
> **Depends on:** `40_long_term_options_evaluator.md` (the scoring engine), `41_positions_provider.md` (holdings input), `32_telegram_operator_alerts.md` (channel-routing + per-day dedup pattern), `19_journal_schema.md` (the `telegram_alert_sent` action).
> **Consumed by:** the operator's Telegram `long_term` channel · the journal (for dedup state).

---

## 1. Theory & Objective

The Long-Term Evaluator Streamlit tab gives the operator a manual review surface: paste holdings, click Parse, scan recommendations. That works during a sit-down review but leaves the rest of the day blind — if a covered-call setup that fits the operator's gates lights up at 11:14 ET, they won't see it until they next open the dashboard.

This skill closes that loop. The spread agent's existing cycle (≈ every 75 s) is the production invocation path — `agent._maybe_run_portfolio_review` calls `run_scheduler` once per hour on cycles whose wall-clock minute falls in `[30, 34]`. The standalone CLI (`python -m trading_agent.portfolio_alert_scheduler`) is the manual / dev path for `--dry-run` or `--force` testing outside market hours. Both paths share the same `run_scheduler` orchestrator so behavior is bit-for-bit identical. On each invocation the scheduler:

1. Gates on env opt-out (`PORTFOLIO_ALERTS_ENABLED=false` kills the alerts without redeploying code) and market hours (only Mon-Fri 09:30-16:00 ET).
2. Loads the operator's persisted holdings paste from `knowledge_base/holdings.json` (skill 41 §3.4).
3. Loads the watchlist from `knowledge_base/watchlist.json`.
4. Runs `LongTermEvaluator.recommend(...)` — same code path the Streamlit tab uses, no shadow implementation.
5. Composes a Telegram-friendly digest body: portfolio one-liner, sector breakdown, top-N covered-call candidates with bracket sketches, skipped-ticker reasons.
6. Hashes the body. If a `telegram_alert_sent` journal row exists for `ticker="_portfolio_"`, `alert_type=lt_portfolio_review:<UTC-date>:<body-hash>`, skip — the operator already saw this exact digest today.
7. Sends via `TelegramNotifier.notify_portfolio_review(...)`, then writes the dedup row.

The scheduler is **read-only by design** — same as the evaluator. It never submits an order, never modifies holdings or watchlist files, never touches Alpaca's trading API. The Telegram body is informational; the operator places orders by hand (or, when Phase 5 lands, via the Schwab Trader API "Preview → Confirm → Place" flow).

## 2. Mathematical Formula

N/A — control flow only. The dedup key construction has one line of substance:

```text
dedup_key = "lt_portfolio_review:" + YYYY-MM-DD(UTC now) + ":" + first16(sha256(body))
```

The truncated SHA-256 keeps the journal row's `alert_type` string short while staying collision-resistant at the scale the scheduler runs at (≤7 invocations per day × ≤2 distinct bodies per day).

## 3. Reference Python Implementation

### 3.1 Scheduler entry point

```python
# trading_agent/portfolio_alert_scheduler.py
def run_scheduler(
    *,
    deps: SchedulerDeps,
    now: Optional[datetime] = None,
    force: bool = False,
    dry_run: bool = False,
) -> AlertResult:
```

`SchedulerDeps` is a dataclass of seven callables — holdings loader, watchlist loader, evaluator factory, telegram send, market-hours check, dedup check, dedup write. Production wires them via `build_default_deps()` against `holdings_store`, `watchlist_store`, `LongTermEvaluator`, `TelegramNotifier`, `market_hours.is_within_market_hours`, and `JournalReader.telegram_alert_sent_today_utc`. Tests pass fixtures so the scheduler runs hermetically without env vars, network, or filesystem.

### 3.2 Body composition

```python
# trading_agent/portfolio_alert_scheduler.py
def compose_digest_body(
    *,
    positions: List[Position],
    watchlist_symbols: List[str],
    recommendations: List[Recommendation],
    now: datetime,
    cc_top_n: int = 3,
) -> str:
```

Pure function — no I/O, no env reads. Returns a plaintext body the notifier wraps in `<pre>` for Telegram fixed-width rendering. Sections:

- Header line with date + UTC stamp.
- Holdings one-liner with total cost basis + position count.
- Sector breakdown (top 4 by cost basis, with % of total).
- Income overlay — top `cc_top_n` covered-call recs with the rationale string + bracket sketch (STO @ entry → BTC @ TP + stop if underlying < SL).
- Skipped section — held∩watched tickers below the 100-share floor (max 6 lines + a "…N more" marker).
- "Held but not on watchlist" hint (max 8 tickers inline) so the operator notices coverage gaps.

### 3.3 Telegram method

```python
# trading_agent/telegram_notifier.py
def notify_portfolio_review(self, *, body: str, dedup_key: str) -> bool:
```

Wraps the body in `<pre>{html_escape(body)}</pre>` so Telegram renders it monospaced. Truncates at 4000 chars (Telegram's hard cap is 4096) with a `… (truncated)` marker so the operator can tell when their holdings list grew past the digest size.

**Routing — dedicated `long_term` channel.** Reads `TELEGRAM_LONG_TERM_BOT_TOKEN` / `TELEGRAM_LONG_TERM_CHAT_ID` when both are set; falls back to the info channel credentials otherwise. Operators set the LT env vars when they want the hourly digest in its own Telegram channel separate from the spread agent's trade alerts. Single-bot deployments stay unchanged — the LT channel reuses the info bot's creds. `TelegramNotifier.long_term_channel_distinct` reports `True` when both env vars are set and the LT bot is actually in effect.

## 4. Edge Cases / Guardrails

- **Env opt-out is the hard kill switch.** `PORTFOLIO_ALERTS_ENABLED=false` (or `0`, `no`, `off`) returns `AlertResult(skipped_reason="PORTFOLIO_ALERTS_ENABLED=false")` BEFORE any I/O. Operator setting this on the pi silences the alerts without a code change or notifier teardown. Default is on.
- **Market-hours gate is bypassable with `--force`.** Outside `is_within_market_hours(now)` the scheduler returns `skipped_reason="outside market hours"`. The `--force` CLI flag overrides for one-shot testing. Cron jobs MUST NOT set `--force` in production; the gate exists so an off-hours invocation doesn't send a stale digest.
- **Market-data routing defaults to Schwab.** Both the Streamlit Long-Term Evaluator tab and the in-cycle scheduler call `build_market_data_provider(surface="long_term", default_provider="schwab")`. The factory walks `MARKET_DATA_PROVIDER_LONG_TERM` → `MARKET_DATA_PROVIDER` → `default_provider` (skill 16). Operator overrides per-tab by setting `MARKET_DATA_PROVIDER_LONG_TERM=alpaca` (or yahoo) in `.env`; the credit-spread `live` / `watchlist` surfaces are unaffected. Schwab is the default because its options-chain coverage includes ADRs (NOK), small-caps (SOFI, ZS), and tail-strike contracts that Alpaca's `indicative` feed gaps. Conformance: `test_factory_long_term_surface_defaults_to_schwab`.
- **Empty holdings file = silent skip.** If `holdings_store.load_holdings()` returns an empty snapshot (operator hasn't pasted yet, or clicked Reset), the scheduler skips with `skipped_reason="holdings file empty"` — never sends an "I have no holdings" digest, which would be noise.
- **Holdings parse failure = silent skip.** If the persisted paste fails `ManualPositionsProvider.from_json_text` (rare — usually means the file was hand-edited or schema changed), the scheduler logs the failure and skips. The Streamlit tab is where parse errors get surfaced to the operator; the hourly scheduler doesn't bother them.
- **Empty watchlist = silent skip.** No tickers to recommend on. Same fail-quiet contract.
- **Body-hash dedup uses the UTC date, not the local date.** The journal's `_DEDUP_BYPASS_ACTIONS` rules mirror the existing `telegram_alert_sent` action used for trade alerts — same UTC-date semantics so the dedup matches the rest of the agent's day boundary.
- **Telegram-inactive = no journal row written.** When `TelegramNotifier.is_active` is False, the scheduler still composes the body (so the CLI exit message shows what WOULD have been sent), but skips both the send and the dedup write. This means a Telegram outage doesn't burn the dedup slot — the operator gets the next attempt as soon as Telegram recovers.
- **Telegram send failure (network, 429, 5xx) = no dedup write.** Same principle: the dedup row only goes in on confirmed delivery. A failed POST doesn't accidentally suppress tomorrow's first attempt.
- **Dry-run mode.** `--dry-run` runs the full pipeline (load → score → compose → hash → dedup check) but skips the actual `_send` and the dedup write. Returns `skipped_reason="dry-run"` and the body is in `AlertResult.debug["body"]` for stdout printing via `--print-body`. Operator uses this to preview the digest without spamming the channel during development.
- **The scheduler does NOT import `executor.*`.** Conformance: `test_skill_42_scheduler_does_not_submit_orders` greps the module source for `submit_order` / `place_order` / `from trading_agent.executor`. The scheduler is read-only.
- **The scheduler does NOT redefine scoring helpers.** CI invariant 2 prevents any `_score_*` definition outside `chain_scanner.py` / `decision_engine.py`. The scheduler imports and composes, never defines.

## 5. Cross-References

- `40_long_term_options_evaluator.md` — the scoring engine the scheduler invokes; same code path the Streamlit tab uses (no shadow implementation).
- `41_positions_provider.md` — the holdings input contract.
- `32_telegram_operator_alerts.md` — info-channel routing + per-day dedup pattern; this skill extends the per-day dedup with body-hash discrimination.
- `19_journal_schema.md` — registers the `telegram_alert_sent` action vocabulary the dedup row uses.
- `00_sdlc_and_conventions.md` — adding a new recommendation section to the digest is a new branch in `compose_digest_body`, not a new module.

---

*Last verified against repo HEAD on 2026-10-08.*
