# Claude Code Portfolio Integration — Plan

**Working document — not a skill doc yet.** Individual phases become
`docs/skills/48_*.md` … `54_*.md` as they land.

**One-line intent.** Give a Claude Code session opened in this repo a
runtime read/write surface over the live trading agent, so the operator
can drive portfolio-management flows in chat — daily reviews, position
triage, pre-trade proposals, incident response — without every task
turning into a hand-written Python script. Mirrors OpenMontage's
"AI-is-the-orchestrator, tools + skills + YAML manifests are the
knowledge" pattern (see below), adapted to the safety constraints of
a system that moves money.

---

## Why this shape

The trading-agent repo already ships the OpenMontage pattern for
**code-change** work: `CLAUDE.md` + `PROJECT_MANIFEST.md` +
`docs/skills/*.md` + a conformance gate that enforces the contract.
Any Claude Code session that opens the repo already reads them. What
that pattern doesn't do today is give Claude Code access to the
**live** portfolio — open positions, real-time PnL, current quotes,
the scan/decide/execute pipeline as an interactive callable. Every
operator flow that isn't "edit code and land a PR" (morning brief,
triage a position, propose a trade, respond to an alert) still means
opening Streamlit or reading raw JSON.

This plan closes that gap by adding **two new surfaces** on top of
what already exists:

1. **A read MCP layer** so Claude Code discovers what it can ask the
   running agent (positions, journal, scan results, quotes) as
   first-class tools — the equivalent of OpenMontage's
   `registry.discover()`.
2. **A guarded write path** so Claude Code can propose orders that
   flow through the existing executor invariant (skill 18), with a
   capped auto-promote path for small trades and a mandatory human
   gate for anything above the cap.

The autonomous `agent.py` continues to run headless. Claude Code is a
**second seat**, not a replacement.

---

## OpenMontage patterns adopted verbatim

- **Three-layer knowledge split.** Layer 1: what exists (MCP tool
  registry auto-listing). Layer 2: how we want it used (`docs/skills/`
  — already in place). Layer 3: external tech knowledge packs
  (`.agents/skills/` — Schwab OAuth quirks, IRS wash-sale rules, TDA
  API deprecations).
- **Stage-director skills.** Each portfolio flow gets one atomic
  skill file that teaches Claude to execute that flow using the
  Phase-1 MCP tools. Same shape as OpenMontage's per-pipeline stage
  directors.
- **Capability envelope discovery.** New CLI:
  `python -m trading_agent.mcp --list-tools`. First thing Claude Code
  is instructed to run at session start.
- **Reviewer subagent + approval gate on every write path.** A
  `risk-reviewer` subagent runs before any `/propose` finalizes,
  regardless of whether auto-promote is enabled.

## OpenMontage patterns NOT adopted

- No YAML pipeline manifests. The trading agent's "pipelines" are the
  existing `agent.py` cycle + `decide()` — already declarative
  enough via `PresetConfig`.
- No dynamic tool registry Python decorator. The MCP tool list is
  hand-written and CI-verified — this repo values invariant scanners
  over runtime introspection.

---

## Phase 1 — Read MCP layer

**New module.** `trading_agent/mcp/` — stdio MCP server that Claude
Code auto-discovers via `.mcp.json` at the repo root.

**Tools exposed** (all read-only; all import-checked by conformance):

| Tool | Wraps |
|---|---|
| `list_positions()` | `journal_kb` open-position query |
| `get_position(ticker, strategy)` | single-position detail |
| `list_recent_trades(days=7)` | `journal_kb` closed-trade query |
| `get_journal_summary()` | roll-up: win rate, hit rate, avg PnL per preset |
| `get_preset()` | current `PresetConfig` |
| `get_risk_report()` | `risk_manager` snapshot: exposure, buying power, correlation |
| `run_scan(watchlist, preset?)` | `chain_scanner.scan()` — returns candidates |
| `score_candidate(underlying, strategy, params)` | `decision_engine.decide()` — same call the backtester uses |
| `get_chain(underlying, expiration, option_type)` | passthrough to skill 47 |
| `get_quote(symbol)` | passthrough to skill 47 |
| `get_market_status()` | passthrough to skill 47 |
| `get_recent_alerts(hours=24)` | `ExceptionMonitor` recent events |

**Conformance.** The skill 47 AST walker is generalized:
`tests/conformance/test_readonly_module_imports.py` runs against every
module registered in `READONLY_MODULES = ["data_server", "mcp"]` and
fails CI on any import of `executor`, `submit_order`, `place_order`,
`TradingClient`, or `pending_orders`.

**Skill file.** `docs/skills/48_claude_code_mcp_surface.md`.

---

## Phase 2 — Portfolio-management playbooks

Each is one atomic `docs/skills/*.md` teaching Claude Code how to run
that flow using the Phase-1 tools. Structure mirrors existing skills
(theory → endpoint/tool surface → reference code → edge cases →
cross-references).

- **49 `daily_portfolio_review.md`** — Claude reads positions, PnL,
  expiring-soon, macro overlays; produces the morning brief. Shares
  infrastructure with the `morning` skill but is repo-scoped.
- **50 `position_triage.md`** — for one open position: hold / roll /
  close, with scoring rationale sourced from `decide()`.
- **51 `pre_trade_approval.md`** — natural-language "consider a JPM
  iron condor at 35 DTE" → fully-scored candidate + risk report + a
  `pending_orders/<uuid>.json` file. Explicit "does NOT call the
  executor" statement in §4.
- **52 `watchlist_curation.md`** — hit-rate analysis over the
  journal, proposed adds/drops.
- **53 `incident_response.md`** — when ExceptionMonitor pages, Claude
  reads the error + recent journal + proposes a fix. Fix goes through
  the normal SDLC (skill + failing test + code) — never bypassed.
- **54 `tax_lot_review.md`** — quarterly wash-sale and short-vs-long-
  term optimization pass.

Each skill file gets its own conformance test verifying that (a) the
skill's cited code lines still appear in source (existing
`scan_skill_quotes_match` handles this automatically); (b) the
producer code the skill points at doesn't import the executor.

---

## Phase 3 — Write surface (guarded, with capped auto-promote)

**New directory.** `pending_orders/` at repo root. `.gitignore`d.

**New CLI.** `python -m trading_agent.executor.promote <uuid>`:

1. Reads `pending_orders/<uuid>.json`.
2. Re-scores the candidate against **current** market data (staleness
   guard — a proposal from 2h ago at open must be re-verified at 3pm).
3. Runs the same C/W floor formula (invariant #1) and risk-manager
   checks the live agent runs.
4. Prints the parsed order + risk diff.
5. **Approval gate — one of two paths:**
   - **Manual (default).** Waits for `--yes` on the command line.
   - **Auto-promote.** Only when ALL of these hold:
     - `TRADING_AGENT_AUTO_PROMOTE_ENABLED=true` (env master switch,
       same discipline as `SCHWAB_API_CACHE_ENABLED`).
     - Notional debit ≤ `PresetConfig.auto_promote_max_notional_usd`
       (new field, defaulted to `0` which disables auto-promote
       preset-side too — so the master switch alone is insufficient).
     - Contract count ≤ `PresetConfig.auto_promote_max_contracts`
       (defaulted to `1`).
     - Strategy ∈ `PresetConfig.auto_promote_allowed_strategies`
       (defaulted to `[]`).
     - Passes `risk_manager.check()` with zero warnings (not just
       zero errors — a warning is enough to fall back to manual).
     - Within regular market hours (not the first/last 5 minutes).
     - Re-score delta from original proposal < 5% (staleness cap).
6. Only after approval (either path) is the existing `executor.py`
   `submit_order()` called.

**Every auto-promote emits an `ExceptionMonitor` INFO event** so the
operator's Telegram channel gets a "Claude auto-promoted trade X"
line — visibility without a manual gate.

**Every auto-promote failure** (any of the AND conditions failing)
falls back to manual and logs a WARNING event.

**Conformance.** New invariant: `trading_agent/mcp/` never writes to
`pending_orders/` (the write path is a Claude Code text-file edit; the
MCP layer stays read-only end-to-end). A separate conformance test
enforces that `promote.py` is the ONLY module besides `executor.py`
that imports `submit_order`.

**Skill file.** `docs/skills/55_pending_orders_promotion.md` documents
the full gate + auto-promote decision tree.

---

## Phase 4 — Slash commands + subagents

**`.claude/commands/`** (each is one markdown file with frontmatter +
prompt body per Claude Code convention):

- `portfolio.md` → runs skill 49 flow
- `triage.md` → takes `$ARGUMENTS` = position id/ticker, runs skill 50
- `propose.md` → takes `$ARGUMENTS` = "AAPL iron condor 35 DTE",
  runs skill 51 through Phase-3 write path
- `incident.md` → takes recent error id, runs skill 53
- `taxreview.md` → runs skill 54 for the current quarter

**`.claude/agents/`**:

- `risk-reviewer.md` — read-only subagent with tool allowlist
  `[get_risk_report, get_position, list_positions, score_candidate]`.
  Invoked by `/propose` before the write path fires.
- `scanner-runner.md` — read-only subagent with `[run_scan,
  get_journal_summary, get_preset]`. Returns top-N candidates.
- `journal-analyst.md` — read-only subagent with
  `[list_recent_trades, get_journal_summary]` for post-mortems and
  hit-rate questions.

Subagent tool allow-lists are CI-verified: a subagent .md file may
only reference tools that appear in `READONLY_TOOLS` (defined in
`trading_agent/mcp/__init__.py` and imported by the conformance test).

---

## Phase 5 — Conformance & documentation gates

New tests under `tests/conformance/`:

- `test_skill_48_mcp_readonly.py` — AST walker forbidding executor
  imports in `trading_agent/mcp/`.
- `test_skill_55_promote_is_sole_writer.py` — grep-and-AST verifying
  only `promote.py` and `executor.py` import `submit_order`.
- `test_subagent_tool_allowlists.py` — parses every
  `.claude/agents/*.md`, extracts the `tools:` frontmatter, verifies
  every listed tool is in `READONLY_TOOLS`.
- `test_slash_command_shape.py` — sanity-check for slash command
  frontmatter (name, description, `$ARGUMENTS` if present).
- `test_auto_promote_gate_exhaustive.py` — table-driven test walking
  every branch of the Phase-3 AND condition; explicit fail cases for
  each fail-open scenario.

New CI wiring:

- `scripts/checks/scan_readonly_modules.py` — generalized version of
  the skill 47 walker.
- `scripts/checks/scan_subagent_allowlists.py` — new.

Skill freshness footers get updated on every phase land; traceability
regenerated.

---

## Open questions to resolve before Phase 1

1. **MCP transport.** stdio (default, simplest, per-session) vs. a
   local HTTP MCP that survives across Claude Code sessions? Default
   to stdio; revisit if session-startup latency matters.
2. **Journal-KB access.** Skill 47 forbids `journal_kb` imports in
   `data_server/`. For skill 48, `journal_kb` reads are legitimate —
   `list_positions()` needs them. Do we (a) extend the forbidden-
   imports allowlist for `mcp/` to permit `journal_kb.read.*` only,
   or (b) create a `journal_kb/read_only.py` facade the MCP layer
   uses? **Recommend (b)** — cleaner boundary, matches the
   read/write split we're already imposing.
3. **`PresetConfig` field naming.** `auto_promote_max_notional_usd`
   is verbose but explicit. Alternative: nest into a
   `AutoPromoteConfig` sub-dataclass. **Recommend the sub-dataclass**
   — three new fields want a namespace.
4. **Handling of a Claude Code session while the headless agent is
   also trading.** Race condition: Claude proposes a JPM IC while
   the headless agent's next cycle also picks JPM IC. Need a
   position-lock (advisory file lock in `pending_orders/.lock`) that
   both paths respect. Alternative: Claude Code operations block on
   the next cycle boundary. **Recommend the advisory lock** — cycle-
   blocking is a worse UX and doesn't scale to multi-symbol cases.
5. **Backtester wiring.** Skill 48 tools like `score_candidate` will
   want to work in backtest mode too, so a user can ask Claude "how
   would this trade have played in Q1?" without opening Streamlit.
   Do we add a `--backtest` flag to `run_scan` and `score_candidate`?
   **Recommend yes** — invariant #3 already requires the backtester
   to wire through `decide()`, so tooling parity is free.

---

## Decisions locked in

- **Auto-promote path.** Enabled, capped, master-switched off by
  default. Notional cap, contract cap, strategy allowlist all live
  in `PresetConfig` (per invariant "new tunables go in PresetConfig,
  not class constants").
- **Executor remains the sole writer.** `promote.py` calls into
  `executor.submit_order()`; no new order-placement primitive.
- **Skill numbering.** 48 (MCP surface) → 49 (daily review) → 50
  (triage) → 51 (pre-trade) → 52 (watchlist) → 53 (incident) → 54
  (tax) → 55 (promotion gate). 55 sits after the playbooks because
  it's the shared write path they all funnel through.
- **Phasing.** Ship 1 → 2 → 3 → 4 → 5 as separate PRs. Each PR
  ships its own skill, tests, and CI gate updates. No phase merges
  until its `docs/skills/*.md` footer is re-stamped and all 8
  existing invariant checks stay green.

---

## What ships in Phase 1 PR (concrete acceptance)

- [ ] `trading_agent/mcp/__init__.py` — `READONLY_TOOLS` constant,
      `SERVER_READ_ONLY = True` sentinel.
- [ ] `trading_agent/mcp/server.py` — stdio MCP server.
- [ ] `trading_agent/mcp/tools/*.py` — one file per tool group.
- [ ] `trading_agent/mcp/__main__.py` — CLI with `--list-tools`.
- [ ] `journal_kb/read_only.py` — read-only facade (decision from
      open question 2).
- [ ] `.mcp.json` at repo root wiring the server.
- [ ] `docs/skills/48_claude_code_mcp_surface.md`.
- [ ] `tests/conformance/test_skill_48_mcp_readonly.py`.
- [ ] `scripts/checks/scan_readonly_modules.py` — generalized walker.
- [ ] `PROJECT_MANIFEST.md` handoff prompt updated.
- [ ] `docs/traceability.md` regenerated.

---

*Last updated: 2026-09-28. Author: Bala + Claude.*
