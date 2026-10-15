# scripts/checks — drift-prevention smoke tests

These checks back the live ↔ backtest unification. **All must pass
before any change to scoring, pricing, or floor logic ships.** They are
designed to run in CI (see `.github/workflows/ci.yml`) and are also
runnable individually during local iteration.

| Script | What it asserts | Runtime |
|---|---|---|
| `scan_invariant_check.py` | AST: `\|Δ\|×(1+edge_buffer)` floor appears in `chain_scanner` / `risk_manager` / `executor`; no shadow `_score_candidate` / `_quote_credit` outside `chain_scanner` + `decision_engine`; `streamlit/backtest_ui.py` calls `decide(...)` | < 1 s |
| `run_scan_diagnostics_check.py` | `ChainScanner` + `decision_engine.decide()` integration | ~2 s |
| `run_journal_split_check.py` | `JournalKB(run_mode=...)` writes to the right file; rejects unknown modes | < 1 s |

**Spec-driven development (SDD) checks**, also in CI:

| Script | What it asserts |
|---|---|
| `scan_skill_quotes_match.py` | Every skill's §3 quoted code lines still appear in the cited source |
| `scan_skill_freshness.py` | Every skill's "Last verified" footer is no older than the last commit to each source file it cites (commit date in UTC — stamp with `TZ=UTC date -I`) |
| `build_traceability.py` | Regenerates `docs/traceability.md`; CI fails if the committed copy differs |
| `scan_dryrun_isolation.py` | Production code never reads `signals_dryrun.jsonl` |

**Local gates** (not in the CI workflow; run them before pushing):

| Script | What it asserts |
|---|---|
| `scan_exception_monitor_coverage.py` | Every `except Exception` in production code records to the ExceptionMonitor or carries a `skill-34-exempt` note (skill 34) |
| `scan_method_length.py` | No method in `agent.py` and its collaborators grows past the current ceiling |
| `scan_subagent_allowlists.py` | Every tool named in `.claude/agents/*.md` frontmatter exists on the MCP server and stays inside the read-only surface (skills 48, 55) |
| `check_skill_updates_for_diff.py --ref origin/main` | Lists the skills that cite each changed source file; each must be in the same diff (helper, not a gate) |

> **Note (2026-05-04).** The legacy `run_unified_backtest_check.py` and
> `run_live_vs_backtest_parity_check.py` were retired after the backtest
> rewrite (skill 15). They exercised `Backtester._build_alpaca_plan_via_decide`,
> a method on a class that no longer exists. Live ↔ backtest parity is
> now structural: the new `trading_agent.backtest` package calls
> `decision_engine.decide()` directly with the same `ChainSlice` shape
> the live scanner builds, so there is no second path to compare.
> `scan_invariant_check.py` invariant #3 still pins that the call site
> exists in `streamlit/backtest_ui.py`; per-component tests under
> `tests/test_backtest/` cover the runner end-to-end.

## Running locally

From the repo root:

```bash
python3 scripts/checks/scan_invariant_check.py
python3 scripts/checks/run_scan_diagnostics_check.py
python3 scripts/checks/run_journal_split_check.py
python3 scripts/checks/scan_skill_quotes_match.py
python3 scripts/checks/scan_skill_freshness.py
python3 scripts/checks/check_skill_updates_for_diff.py --ref origin/main
```

Or all of them at once:

```bash
for f in scripts/checks/*.py; do python3 "$f" || { echo "FAIL: $f"; exit 1; }; done
```

## When a check fails

| Failing check | What it usually means |
|---|---|
| `scan_invariant_check` | Someone added a scoring helper outside `chain_scanner` / `decision_engine`, or the C/W floor formula moved on one side without the other, or the literal `decide(` call was deleted from `streamlit/backtest_ui.py`. Grep for `cw_floor` and `_score_candidate` to find the drift. |
| `run_scan_diagnostics_check` | `ChainScanner.scan()` and `decision_engine.decide()` disagree on strike picks or credit. Usually means a shadow scorer crept back into the live path. |
| `scan_skill_freshness` | A source file changed after the footer date of a skill that cites it. Re-check that skill's §3/§4 and re-stamp its footer with `TZ=UTC date -I` (the check compares against the UTC commit date, so a local-time stamp late in the evening can fall a day short). |
| `run_journal_split_check` | `JournalKB.run_mode` resolution broke — either `signals_live.jsonl` or `signals_backtest.jsonl` isn't being written, or the legacy `signals.jsonl` is being recreated. |
