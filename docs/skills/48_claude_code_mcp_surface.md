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
get_chain(underlying, expiration=None,          — skill-47 passthrough
          option_type=None)
get_quote(symbol)                               — skill-47 passthrough
get_market_status()                             — skill-47 passthrough
get_fundamentals(ticker)                        — P/E, EPS, mcap, dividend, beta, 52w, ROE/ROA (skill-47 passthrough)
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
    "list_recent_trades",
    "get_journal_summary",
    "get_preset",
    "get_risk_report",
    "run_scan",
    "score_candidate",
    "get_chain",
    "get_quote",
    "get_market_status",
    "get_fundamentals",
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
      "command": "python",
      "args": ["-m", "trading_agent.mcp"]
    }
  }
}
```

`python -m trading_agent.mcp --list-tools` prints the capability envelope as JSON — the operator (or Claude Code at session start) uses this to confirm the closed set.

## 4. Edge Cases / Guardrails

- **Read-only enforced by AST walker.** `test_skill_48_mcp_readonly.py` parses every `.py` under `trading_agent/mcp/` and rejects any `import`/`from` referencing `trading_agent.executor`, `submit_order`, `place_order`, `alpaca.trading`, or `pending_orders`.
- **Handler registry is closed and symmetric.** The conformance test asserts `set(_HANDLERS) == set(READONLY_TOOLS)`. A drift is a CI failure.
- **Data-server fallback is silent-but-explicit.** `_market.get_quote` returns `{"source": "unavailable"}` when `SCHWAB_API_BASE_URL` is unset or unreachable, rather than raising — the LLM sees a structured answer and can decide whether to ask the operator to bring the server up.
- **Backtest parity flag.** `run_scan(backtest=True)` and `score_candidate(...)` route through `decision_engine.decide()` — the same primitive the live agent uses (invariant #3). Never redefine scoring in `trading_agent/mcp/tools/*.py` (invariant #2 — no shadow scorers).
- **Multi-day journal reads (2026-09-29).** `list_positions()` returns `open_positions` — every submitted spread without a matching close, any open date, with `opened_at`, `width`, `max_loss`, `short_strikes` (from the trade plan by `run_id`; null once aged out of the 200-row history), `order_id`, `status` (`open` | `expired_unrecorded`). `opens_today` is kept for back-compat. `list_recent_trades(days)` honours `days` via `JournalReader.closes_since` and returns `closes` + `realized_pl_window` alongside the legacy `closes_today`. Previously positions were today-only and width/opened_at/pnl were always null (mapped to non-existent attributes).
- **Errors surface as tool errors, not server errors.** Any exception inside a handler becomes a JSON-RPC error with the exception's type + message. The MCP loop keeps serving; the Claude Code session sees a structured failure.
- **stdio transport, one process per session.** The server is spawned per Claude Code session. No shared state between sessions.

## 5. Cross-References

- `47_schwab_data_api.md` — the HTTP data server this MCP layer prefers for quotes/chains/market-status. When it's up, `_data_server_get()` uses it; when it's down, the tools return `{"source": "unavailable"}` structured rows.
- `18_order_submission_idempotency.md` — the write path this surface deliberately doesn't touch. The `SERVER_READ_ONLY = True` invariant is symmetric to "the executor is the ONLY place that submits orders."
- `19_journal_schema.md` — every position/trade read here consumes that schema.
- `docs/skills/55_pending_orders_promotion.md` — the separate CLI that owns the write path Claude Code proposes into.

---

*Last verified against repo HEAD on 2026-09-29.*
