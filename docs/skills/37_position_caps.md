# Position-cap dedup

> **One-line summary:** Per-ticker and per-sector position-count caps that block Stage 2 (OPEN new positions) from stacking on tickers/sectors already at their ceiling. Extracted from `_run_cycle_impl` as a pure function so the cap math is testable in isolation.
> **Source of truth:** [`trading_agent/position_caps.py`](../../trading_agent/position_caps.py), [`trading_agent/agent.py`](../../trading_agent/agent.py) call site.
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

## 4. Edge Cases / Guardrails

- **All signals count.** A position with `signal=regime_shift` or `signal=STRIKE_PROXIMITY` (exit pending) increments the count just like a HOLD. This is the 2026-05-13 GLD fix hard-pinned by `test_position_count_aggregates_across_signals`.
- **Sector block excludes already-blocked tickers.** When the per-ticker cap already catches a ticker, the sector loop skips it so the operator log doesn't say "blocked by sector AND blocked by ticker" — only the first reason fires.
- **Malformed position entries are tolerated.** A position dict missing `underlying` or with an empty string is skipped silently. Defense-in-depth against an upstream schema change.
- **`sector_for` is injected, not imported.** The function takes `sector_for` as a kwarg so tests can pass a fixture mapping. Production passes `trading_agent.utils.sector_for`.
- **Returns are tuples, not dicts.** A 4-tuple keeps the contract explicit; future additions (e.g. a per-strategy cap) would add a fifth element rather than mutate a dict shape.

## 5. Cross-References

- `17_close_failure_and_cooldown.md` — the close-side counterpart to this open-side gate.
- `35_close_event_collaborators.md` — same extraction pattern applied to the close path.
- `36_ticker_filters.md` — same pattern applied to the regime/RSI/IV filters at the top of `_process_ticker`.

---

*Last verified against repo HEAD on 2026-05-22.*
