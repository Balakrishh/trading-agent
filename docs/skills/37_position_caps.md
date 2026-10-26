# Position-cap dedup

> **One-line summary:** Per-ticker and per-sector position-count caps that block Stage 2 (OPEN new positions) from stacking on tickers/sectors already at their ceiling, plus the book-level risk limits: total open risk, one-direction risk per correlated group and per book (E6), the stop cool-down, and the book's net delta in shadow (E4). Pure functions so the cap math is testable in isolation.
> **Source of truth:** [`trading_agent/position_caps.py`](../../trading_agent/position_caps.py), [`trading_agent/book_delta.py`](../../trading_agent/book_delta.py), [`trading_agent/agent.py`](../../trading_agent/agent.py) call site.
> **Phase:** 2  •  **Group:** risk
> **Depends on:** `utils.sector_for` (ticker → sector lookup), Stage 1 monitor output schema.
> **Consumed by:** `_run_cycle_impl` Stage 2 dedup.

---

## 1. Theory & Objective

Pre-2026-05-13 the dedup logic in `_run_cycle_impl` had a subtle bug: it only counted positions whose `signal == "HOLD"`. A position that had triggered a `regime_shift` or `STRIKE_PROXIMITY` exit but hadn't yet closed was silently invisible to Stage 2's dedup. The canonical failure was GLD on 2026-05-12 — exit-pending but not yet HOLD, so Stage 2 happily planned a fresh GLD spread on top of the closing one.

The fix counts EVERY reported position regardless of signal. To prevent the same class of regression, the cap derivation moved to a pure function in `trading_agent/position_caps.py`.

The two caps:

- **`MAX_POSITIONS_PER_TICKER`** (default 1) — single position per underlying. Blocking same-day re-opens forces the operator to think about why an exit was triggered before adding more exposure.
- **`MAX_POSITIONS_PER_SECTOR`** (default 2) — concentration limit. Without it, several ETFs in the same sector (XLF + KRE both financials, GLD + SLV both metals) could pile on at once and the account becomes a single macro bet.

## 2. Mathematical Formula

```text
For each open position p reported by Stage 1 monitor:
  positions_per_ticker[p.underlying] += 1
  positions_per_sector[sector_for(p.underlying)] += 1

blocked = { t : positions_per_ticker[t] >= MAX_POSITIONS_PER_TICKER }
sectors_at_cap = { s : positions_per_sector[s] >= MAX_POSITIONS_PER_SECTOR }
blocked |= { t in universe : sector_for(t) in sectors_at_cap
                              AND t not in blocked }
return blocked, positions_per_ticker, positions_per_sector, sectors_at_cap
```

The `AND t not in blocked` clause is intentional — a ticker already caught by the per-ticker cap shouldn't fire the sector-block log line. Keeps operator output focused on the marginal signal.

Book-level limits (2026-10-26, backlog §11):

```text
direction(strategy): bull | bear | (neutral: own name, not counted)
exposure["book:<dir>"]          = Σ max_loss × contracts, open + submitted this cycle, Wheel excluded
exposure["group:<g>:<dir>"]     = same over RISK_GROUPS[g]
  rates = TLT, IEF, TLH, XLU · gold = GLD, IAU, GDX, GDXJ, SLV · us_equity = SPY, QQQ, DIA, IWM, XLK, SMH
room = min( max_group_direction_risk_pct (0.03) × equity − exposure[group],
            max_book_direction_risk_pct  (0.06) × equity − exposure[book] )
blocked  if room < one contract's max loss            → risk: direction_risk_cap (…)
else     per-trade risk % = min(current, room / equity)   (never raised)

stop cool-down: blocked for sessions_between(last stop, today) ≤ N,
  N = stop_cooldown_sessions (5), or stop_cooldown_repeat_sessions (10) when an
  earlier same-direction stop on the ticker is within stop_cooldown_repeat_window (20)

net delta (E4, shadow): share Δ(u) = Σ leg Δ × signed qty × 100
  book % = Σ share Δ(u) × spot(u) × β(u, SPY, 60 sessions) / equity
  would block if |book %| > max_net_beta_delta_pct (0.30) and the candidate's Δ has the same sign
```

## 3. Reference Python Implementation

### 3.1 compute_position_cap_dedup_set

```python
# trading_agent/position_caps.py
def compute_position_cap_dedup_set(
    monitor_results: Dict,
    tickers: Iterable[str],
    *,
    sector_for,
    max_positions_per_ticker: int,
    max_positions_per_sector: int,
) -> Tuple[Set[str], Dict[str, int], Dict[str, int], Set[str]]:
    ...
```

### 3.2 Agent integration

```python
# trading_agent/agent.py — inside _run_cycle_impl
from trading_agent.position_caps import compute_position_cap_dedup_set
(
    tickers_with_positions,
    positions_per_ticker,
    positions_per_sector,
    sectors_at_cap,
) = compute_position_cap_dedup_set(
    monitor_results, tickers,
    sector_for=sector_for,
    max_positions_per_ticker=MAX_POSITIONS_PER_TICKER,
    max_positions_per_sector=MAX_POSITIONS_PER_SECTOR,
)
```

### 3.3 Book-level limits (2026-10-26)

```python
# trading_agent/position_caps.py
RISK_GROUPS = {"rates": {...}, "gold": {...}, "us_equity": {...}}
def risk_group_for(ticker) -> Optional[str]
def direction_exposure(monitor_results) -> Dict[str, float]        # "book:bear", "group:rates:bear", …
def add_exposure(exposure, ticker, strategy, dollars) -> None
def direction_room(ticker, strategy, equity, exposure, group_pct, book_pct) -> (room | None, label)
def direction_risk_failure(...) -> (reason | None, room | None)
def stop_cooldown_failure(ticker, strategy, today, recent_stops, sessions,
                          repeat_sessions=0, repeat_window=20) -> Optional[str]

# trading_agent/book_delta.py (E4, shadow)
def share_deltas(legs, quotes) -> {underlying: share Δ | None}   # from the monitor's re-mark quotes
def beta_from_closes(closes, spy, n=60) -> Optional[float]
def book_reading(deltas, spots, betas, equity) -> {"spy_equiv_dollars", "pct_of_equity", "unknown", …}
def plan_share_delta(plan) -> Optional[float]                   # one contract, buy +, sell −
def shadow_verdict(book_pct, candidate_delta, budget_pct) -> Optional[str]

# trading_agent/agent.py
#   _stage_monitor: self._share_deltas = share_deltas(positions, quotes)
#   Stage 2 setup:  self._direction_exposure = direction_exposure(monitor_results)
#                   self._book_delta_update(account_balance)   → trade_journal/book_delta.json
#   gate chain:     market state → ladder → stop cool-down → _direction_risk_block
#                   → trend filters → entry confirmation; then _net_delta_shadow (never blocks)
#   after a submit: add_exposure(self._direction_exposure, ticker, strategy, max_loss × qty)
#   _entry_scope_key: "group:<g>" for grouped tickers when entry_rate_scope="sector"
```

Preset fields: `max_group_direction_risk_pct` 0.03, `max_book_direction_risk_pct` 0.06, `stop_cooldown_sessions` 5 (was 1), `stop_cooldown_repeat_sessions` 10, `stop_cooldown_repeat_window` 20, `net_delta_mode` "shadow" ("off"; "live" not built — coerced to shadow), `max_net_beta_delta_pct` 0.30. Dashboard: "Direction risk" block below the ladder controls. Summary tokens `Stop cool-down 5 sess (10 after a repeat)`, `Dir risk ≤3%/group ≤6%/book`, `NetΔ shadow ≤30%`.

## 4. Edge Cases / Guardrails

- **All signals count.** A position with `signal=regime_shift` or `signal=STRIKE_PROXIMITY` (exit pending) increments the count just like a HOLD. This is the 2026-05-13 GLD fix hard-pinned by `test_position_count_aggregates_across_signals`.
- **Sector block excludes already-blocked tickers.** When the per-ticker cap already catches a ticker, the sector loop skips it so the operator log doesn't say "blocked by sector AND blocked by ticker" — only the first reason fires.
- **Malformed position entries are tolerated.** A position dict missing `underlying` or with an empty string is skipped silently. Defense-in-depth against an upstream schema change.
- **`sector_for` is injected, not imported.** The function takes `sector_for` as a kwarg so tests can pass a fixture mapping. Production passes `trading_agent.utils.sector_for`.
- **Returns are tuples, not dicts.** A 4-tuple keeps the contract explicit; future additions (e.g. a per-strategy cap) would add a fifth element rather than mutate a dict shape.

- **In-cycle counting (2026-10-05).** The caps were computed once from Stage 1's *filled* positions, so the first live debit cycle opened SPY, QQQ and IWM (all Broad Market, cap 2) in one loop. Now `register_open()` (`position_caps.py`) increments the per-ticker / per-sector counts for (a) every ticker with an order still pending fill in Stage 1.5 and (b) every submission (`submitted` / `dry_run`) inside the Stage 2 loop; when a sector reaches its cap the rest of its tickers join the blocked set immediately (`TradingAgent._register_open`).
- **Total open-risk cap (2026-10-05).** `PresetConfig.max_total_risk_pct` (default 0.10): Σ per-contract `max_loss` × contracts of open defined-risk positions (`open_defined_risk`, from the monitor summary's new `max_loss` / `contracts` fields) plus everything submitted earlier in the cycle (`_submitted_risk`, using the executor's new `qty` result field) may not exceed that share of equity. Before each ticker, `_total_risk_gate` sets the per-trade `max_risk_pct` on BOTH RiskManager and the executor to `min(per-trade × market-state multiplier, remaining / equity)`; with nothing left it journals `skipped_total_risk_cap`. A trade whose single contract no longer fits fails RiskManager Check 4 as before. Wheel legs (`TOTAL_RISK_EXCLUDED`) are not counted — their collateral has its own 40 %-of-equity rule — and orders pending from a *previous* cycle are not counted toward risk (their ticker and sector are blocked, and stale orders are cancelled).

- **Laddering (2026-10-05, backlog §6.7).** The per-ticker cap is now `PresetConfig.max_positions_per_ticker` (default 2; `MAX_POSITIONS_PER_TICKER = 1` remains the fallback for presets without the field). A 2nd+ position on a ticker passes the dedup set but then must clear `ladder_failure()` after planning (`TradingAgent._ladder_block`, recorded as a failed risk check): no entry on a ticker that already opened a position today (`ladder_same_day_entry` — otherwise the adaptive DTE grid could stack two expiries 75 s apart) and an expiration ≥ `ladder_min_gap_days` (7) from every open position on that ticker (`ladder_gap_Nd_lt_7d`). Pending orders still block the ticker entirely; sector and total-risk caps still apply.
- **Add to winners only (2026-10-06).** `ladder_requires_profit` (default True): a 2nd+ position also needs every open position on the ticker at or above breakeven by mid P&L (`open_pls`, from the monitor summary) — otherwise `ladder_existing_losing (open P&L $…)`. The GLD put debit was −$135 to −$175 while the planner kept proposing a second GLD put debit; laddering a loser is doubling down.
- **Gate order (2026-10-15).** After RiskManager: market-state gate → ladder gate → trend filters (skill 65; blocks only in live mode) → entry confirmation. The sector ETFs added the same day (XLK, XLV, XLY, XLI, SMH, XLU) are already mapped in `sector_map.py`; XLK + SMH share Technology and XLY + AMZN share Consumer Discretionary, so each pair counts against one sector's cap.

- **Cool-down after a stop (2026-10-16).** `PresetConfig.stop_cooldown_sessions` (default 1; 0 = off). A stop-type close (`STOP_EXIT_SIGNALS`: `stop_loss`, `hard_stop`, `delta_stop`, `strike_proximity`) blocks new entries on that ticker in the same direction for the rest of the stop's session and the next N sessions (`stop_cooldown_failure`, journaled as `risk: stop_cooldown (…)`). Direction (`trade_direction`): bull = Bull Put, Call Debit, Bounce Bull Put, Cash-Secured Put; bear = Bear Call, Put Debit; any other structure cools down only its own kind. Sessions are weekdays (`sessions_between`; an exchange holiday shortens the cool-down by one session). The stops come from `JournalReader.closes_since(7)` once per cycle in `_begin_entry_gates` (`TradingAgent._stops_from`, close time converted to the ET date). Why: TLT's put debit was stopped at 09:46 ET on 2026-10-15 and a new TLT put debit on the same view was bought at 09:50 — the ladder rules only see open positions, so a full close cleared every block. Gate order is now market-state → ladder → stop cool-down → trend filters → entry confirmation. Dashboard: "Stop cool-down (sessions)" next to the ladder controls; summary token `Stop cool-down N sess`.
- **Longer cool-down and repeat stops (2026-10-26).** Default 1 → 5 sessions; a second same-direction stop within 20 sessions blocks for 10. Evidence: TLT stopped 10-15, re-entered 10-19 (2 sessions later — allowed by the 1-session rule), stopped again 10-22 (−$186, −$148). The stops are read from `closes_since(35)` so the window is covered.
- **Correlated groups (E6, 2026-10-26).** Sector caps saw Treasuries and Utilities as unrelated while TLT ×2 + XLU (≈ $839, 2.9 % of equity on 10-19) were one bet on higher yields, and every entry from 10-14 to 10-22 was bearish. The group / book one-direction caps size a trade into the room left and block it below one contract. On the 2026-10-23 book: bear $1,066 of a $1,735 book budget, rates bear $555 of $868 — one more full-size rates bear does not fit. Grouped tickers also share one hourly entry bucket (`group:rates`), so XLU waits an hour after a TLT entry. Orders pending from a previous cycle are not in the exposure (same gap as the total-risk cap; their tickers are blocked). Neutral structures (condor, calendar, butterfly) and Wheel legs are not counted.
- **Net delta in shadow (E4, 2026-10-26).** Leg deltas come from the monitor's mid re-mark quotes (Schwab `QuoteOption.delta`, passed through `fetch_option_quotes` as `delta`, None when absent), betas from the cached 200-day history — no extra API calls. A leg with no delta makes its underlying `unknown` (listed in the reading; the book % covers the rest). `net_delta_shadow` rows are journalled once per ticker + strategy a day (keys in `book_delta.json`, because the process restarts every cycle). The 30 % budget is a starting guess: dollar delta of a put debit is large next to its max loss (2 IWM 273/270 at Δ ≈ −0.2 ≈ −$10k notional), so read a week of readings before choosing a live budget.

## 5. Cross-References

- `17_close_failure_and_cooldown.md` — the close-side counterpart to this open-side gate.
- `35_close_event_collaborators.md` — same extraction pattern applied to the close path.
- `36_ticker_filters.md` — same pattern applied to the regime/RSI/IV filters at the top of `_process_ticker`.
- `48_claude_code_mcp_surface.md` — `list_positions` reports the same direction / group totals (`book_view`, F6).
- `docs/plans/agent_backlog.md` §11 — evidence and status for E4 / E6.

---

*Last verified against repo HEAD on 2026-10-26.*
