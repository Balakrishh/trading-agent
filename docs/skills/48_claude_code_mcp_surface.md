# Claude Code MCP Surface — Read-Only Tools

> **One-line summary:** Stdio MCP server that gives a Claude Code session opened in this repo runtime access to positions, journal, preset, scanner, and market-data reads — without ever importing the executor. This is the "capability envelope" Claude Code discovers at session start (mirrors OpenMontage's `registry.discover()`).
> **Source of truth:** [`trading_agent/mcp/__init__.py`](../../trading_agent/mcp/__init__.py), [`trading_agent/mcp/server.py`](../../trading_agent/mcp/server.py), [`trading_agent/mcp/tools/positions.py`](../../trading_agent/mcp/tools/positions.py), [`trading_agent/mcp/tools/strategy.py`](../../trading_agent/mcp/tools/strategy.py), [`trading_agent/mcp/tools/market.py`](../../trading_agent/mcp/tools/market.py).
> **Phase:** 2  •  **Group:** ops
> **Depends on:** `47_schwab_data_api.md` (market-data pass-through), `19_journal_schema.md` (position + trade reads), `13_preset_system_hot_reload.md` (`get_preset()`), `18_order_submission_idempotency.md` (the write path this surface deliberately does NOT expose).
> **Consumed by:** Claude Code sessions in this repo, plus the `.claude/agents/*.md` subagents (`risk-reviewer`, `scanner-runner`, `journal-analyst`).

---

## 1. Theory & Objective

The trading agent already runs autonomously. Claude Code sessions opened in this repo handle code changes well but have no runtime handle on the live portfolio — every question ("what am I holding?", "score me an AAPL iron condor at 35 DTE") turns into an ad-hoc Python session. This skill closes that gap by exposing a small, closed set of **read-only** tools over the MCP protocol so Claude Code can call them by name (`list_positions()`, `get_quote("SPY")`) instead of writing scripts.

The read/write boundary is enforced at three layers: (a) the `SERVER_READ_ONLY = True` sentinel in `trading_agent/mcp/__init__.py`; (b) a conformance test (`test_skill_48_mcp_readonly.py`) that AST-walks every file under `trading_agent/mcp/` and fails CI on any import of the executor or order-submission primitives; (c) subagent tool allow-lists (see skill 55 §4 for how the write-path skill 51 flows into a *separate* CLI, never through this surface).

## 2. Tool surface

Exactly 12 tools. The registry is a closed set defined in `READONLY_TOOLS`; adding a tool means editing that constant + wiring the handler + re-running conformance.

```text
list_positions()                                — open positions
get_position(ticker, strategy=None)             — single-position detail
list_recent_trades(days=7)                      — closes window
get_journal_summary()                           — daily roll-up
get_preset()                                    — active PresetConfig
get_risk_report()                               — current risk posture
run_scan(watchlist, preset_name=None,           — scanner entry
         backtest=False)
score_candidate(underlying, strategy, params)   — routes to decide()
wheel_screen(watchlist, target_dte=35,          — CSP / Wheel entries (skill 40 §2.2, §2.7)
             max_collateral=None)
get_chain(underlying, expiration=None,          — skill-47 passthrough
          option_type=None)
get_quote(symbol)                               — skill-47 passthrough
get_market_status()                             — skill-47 passthrough
get_market_state()                              — market risk state + gate + hedge idea (skill 58)
get_playbook_scorecard(days=365, min_trades=20) — per-playbook track record, advisory size (skill 60)
get_trading_halt()                              — kill switch / drawdown governor state (skill 62)
get_fundamentals(ticker)                        — P/E, EPS, mcap, dividend, beta, 52w, ROE/ROA (skill-47 passthrough)
get_news(ticker, days=30, limit=60)             — Alpaca + Yahoo RSS headlines (skill-47 /news, 45 s timeout)
get_research(ticker)                            — analyst estimates + latest quality-at-lows row (skill-47 /research)
get_sentiment(ticker)                           — stored Claude drop-cause label, analyst stance, news flow (skill-47 /sentiment)
get_macro_events(days=45)                       — FOMC / CPI / NFP ahead + coverage (skill-47 /macro-events, skill 68)
get_credit_stress()                             — HYG/IEF stress vs the CAUTION threshold (skill-47 /credit-stress)
get_vol(ticker)                                 — 30d ATM IV, 20d realized, IV/RV verdict (skill-47 /vol)
get_gamma(ticker="SPY")                         — latest dealer-gamma row (skill-47 /gamma)
get_market_context()                            — market state + all of the above for SPY (skill-47 /market-context)
get_recent_alerts(hours=24)                     — ExceptionMonitor tail
```

**Explicitly NOT exposed** — CI-verified via `test_skill_48_mcp_readonly.py`:

- Any order-placement path (`executor`, `submit_order`, `place_order`, `TradingClient`)
- Any write to `pending_orders/` (that's skill 55's job, called only from `promote.py`)
- Journal writes (`JournalKB.log_*`)
- Preset mutations (`save_active_preset`)

## 3. Reference Python Implementation

### 3.1 Registry — the closed tool set

```python
# trading_agent/mcp/__init__.py
SERVER_READ_ONLY: bool = True

READONLY_TOOLS: tuple[str, ...] = (
    "list_positions",
    "get_position",
    "get_position_valuations",
    "list_recent_trades",
    "get_journal_summary",
    "get_preset",
    "get_risk_report",
    "run_scan",
    "score_candidate",
    "wheel_screen",
    "get_chain",
    "get_quote",
    "get_market_status",
    "get_market_state",
    "get_trading_halt",
    "get_playbook_scorecard",
    "get_fundamentals",
    "get_news",
    "get_research",
    "get_sentiment",
    "get_macro_events",
    "get_credit_stress",
    "get_vol",
    "get_gamma",
    "get_market_context",
    "get_recent_alerts",
)
```

### 3.2 Server dispatch

```python
# trading_agent/mcp/server.py
_HANDLERS: Dict[str, Callable[..., Any]] = {
    "list_positions":     _positions.list_positions,
    "get_position":       _positions.get_position,
    "list_recent_trades": _positions.list_recent_trades,
    "get_journal_summary": _positions.get_journal_summary,
    "get_preset":         _strategy.get_preset,
    "get_risk_report":    _strategy.get_risk_report,
    "run_scan":           _strategy.run_scan,
    "score_candidate":    _strategy.score_candidate,
    "get_chain":          _market.get_chain,
    "get_quote":          _market.get_quote,
    "get_market_status":  _market.get_market_status,
    "get_fundamentals":   _market.get_fundamentals,
    "get_news":           _market.get_news,
    "get_research":       _market.get_research,
    "get_sentiment":      _market.get_sentiment,
    "get_macro_events":   _market.get_macro_events,
    "get_credit_stress":  _market.get_credit_stress,
    "get_vol":            _market.get_vol,
    "get_gamma":          _market.get_gamma,
    "get_market_context": _market.get_market_context,
    "get_recent_alerts":  _market.get_recent_alerts,
}
```

Every key in `_HANDLERS` MUST appear in `READONLY_TOOLS`, and vice versa; the conformance test enforces both directions so a handler added without registering its name (or a name registered without a handler) fails CI.

### 3.3 MCP wiring

The repo ships `.mcp.json` at the root so Claude Code auto-registers the server when the operator opens the repo:

```json
{
  "mcpServers": {
    "trading-agent": {
      "command": "sh",
      "args": ["-c", "if [ -x myenv/bin/python ]; then exec myenv/bin/python -m trading_agent.mcp; else exec python -m trading_agent.mcp; fi"]
    }
  }
}
```

The `sh -c` wrapper prefers the repo's `myenv` virtualenv. With a bare `python`, a Claude Code session started from a shell without the venv activated (2026-10-13, after a Pi reboot) launched the system Python, the server died on `ModuleNotFoundError: pandas_market_calendars`, and `/mcp` reported only `CONNECTION_CLOSED`. Without `myenv` it falls back to `python` on `PATH`.

`python -m trading_agent.mcp --list-tools` prints the capability envelope as JSON — the operator (or Claude Code at session start) uses this to confirm the closed set.

## 4. Edge Cases / Guardrails

- **Read-only enforced by AST walker.** `test_skill_48_mcp_readonly.py` parses every `.py` under `trading_agent/mcp/` and rejects any `import`/`from` referencing `trading_agent.executor`, `submit_order`, `place_order`, `alpaca.trading`, or `pending_orders`.
- **Handler registry is closed and symmetric.** The conformance test asserts `set(_HANDLERS) == set(READONLY_TOOLS)`. A drift is a CI failure.
- **Data-server fallback is silent-but-explicit.** `_market.get_quote` returns `{"source": "unavailable"}` when `SCHWAB_API_BASE_URL` is unset or unreachable, rather than raising — the LLM sees a structured answer and can decide whether to ask the operator to bring the server up.
- **Option symbols go to `/quotes`, equities to `/price`** (2026-10-13). `get_quote` matches compact OCC symbols (`GLD261106P00379000`) and POSTs `{"symbols": [sym]}` to the data server's `/quotes` — a read, not an order — returning `bid` / `ask` / `mid` with `price` = mid, or `note: "no quote for this contract"`. Before, option symbols hit `/price`, got HTTP 400 `invalid_ticker`, and read as "data server unreachable".
- **Backtest parity flag.** `run_scan(backtest=True)` and `score_candidate(...)` route through `decision_engine.decide()` — the same primitive the live agent uses (invariant #3). Never redefine scoring in `trading_agent/mcp/tools/*.py` (invariant #2 — no shadow scorers).
- **Multi-day journal reads (2026-09-29).** `list_positions()` returns `open_positions` — every submitted spread without a matching close, any open date, with `opened_at`, `width`, `max_loss`, `short_strikes` (from the trade plan by `run_id`; null once aged out of the 200-row history), `order_id`, `status` (`open` | `expired_unrecorded`). `opens_today` is kept for back-compat. `list_recent_trades(days)` honours `days` via `JournalReader.closes_since` and returns `closes` + `realized_pl_window` alongside the legacy `closes_today`. Previously positions were today-only and width/opened_at/pnl were always null (mapped to non-existent attributes).
- **Contracts and total risk (F6, 2026-10-26).** `max_loss` on a `list_positions` row is **per contract**; the 10-16 and 10-19 briefs read it as the position's risk ($1,769 shown vs $2,974 held). Each row now also has `contracts` (the broker's count from the agent's last valuation snapshot when the position is in it, else the journal's), `max_loss_per_contract`, `total_max_loss`, `mid_pl`, `broker_legs`, `underlying_price`, `exit_signal`, `in_broker_snapshot`; positions the broker holds without a journal row come back as `status="broker_only"`. The new `book` key has `totals` (`total_max_loss` + % of equity, `max_loss_by_direction`, `max_loss_by_group_direction` — the same buckets as the E6 cap, skill 37 — `total_mid_pl`), the snapshot's age and the E4 `net_delta` reading. One builder, `trading_agent/book_view.py:open_book`, is shared with the daily reviewer (skill 56).
- **Errors surface as tool errors, not server errors.** Any exception inside a handler becomes a JSON-RPC error with the exception's type + message. The MCP loop keeps serving; the Claude Code session sees a structured failure.
- **stdio transport, one process per session.** The server is spawned per Claude Code session. No shared state between sessions.

- **`.env` loading (2026-09-29).** `python -m trading_agent.mcp` calls `load_dotenv()` at startup, the same source the data server reads via `load_config()`, so `SCHWAB_API_SERVER_KEY` / `SCHWAB_API_BASE_URL` in `.env` reach the MCP. As with `load_config()`, a variable already exported in the launching shell (e.g. `~/.zshrc`) wins over `.env` — keep the key in exactly one place or the two servers can disagree (every data-server call then surfaces as "unreachable", since the client swallows 401s).

- **`get_position_valuations()` (2026-10-05).** Reads `trade_journal/position_valuations.json`, written by the agent after every exit evaluation (`trading_agent/position_snapshot.py`): per-position mid P&L, exit signal, short delta, underlying price, `age_seconds`. `/triage` uses it so it never re-prices legs from a second feed.
- **`get_market_state()` (2026-10-05).** Reads `trade_journal/market_state.json`, written by the agent every cycle (skill 58): state, reasons, inputs, gate, `age_seconds`, `account_balance`. In CAUTION / DEFENSIVE it adds a SPY put-spread `hedge_suggestion` (≈3 % / 10 % OTM, 30–45 DTE, sized to half the account) — a suggestion for `/propose`, never an order. No snapshot → `{"state": null, "source": "unavailable"}`. `wheel_screen` output now carries `market_state` and `csp_paused`.
- **`get_playbook_scorecard()` (2026-10-05, skill 60).** Round trips from the live journal grouped by playbook: win rate, avg win / loss, expectancy, return on risk (trades with a known contract count), entry slippage, advisory verdict / `suggested_risk_pct`. Read-only; the `journal-analyst` subagent has it. `wheel_screen` diagnostics now include `below_200d_sma`, `trend_unavailable` and `sector_cap` (skill 40).
- **Fundamentals `short_int_to_float`:** Schwab's explicit 0.0 is mapped to `None` (not provided).
- **Subagent tool names** must be the full `mcp__trading-agent__<tool>` form; bare names leave the subagent with zero tools (Claude Code refuses to spawn it). `scan_subagent_allowlists.py` enforces the prefix.

## 5. Cross-References

- `47_schwab_data_api.md` — the HTTP data server this MCP layer prefers for quotes/chains/market-status. When it's up, `_data_server_get()` uses it; when it's down, the tools return `{"source": "unavailable"}` structured rows.
- `18_order_submission_idempotency.md` — the write path this surface deliberately doesn't touch. The `SERVER_READ_ONLY = True` invariant is symmetric to "the executor is the ONLY place that submits orders."
- `19_journal_schema.md` — every position/trade read here consumes that schema.
- `docs/skills/55_pending_orders_promotion.md` — the separate CLI that owns the write path Claude Code proposes into.

---

*Last verified against repo HEAD on 2026-10-27.*
