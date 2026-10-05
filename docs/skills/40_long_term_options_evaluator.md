# Long-term options evaluator

> **One-line summary:** Portfolio-aware evaluator that compares your watchlist against your current holdings and produces ranked **long-dated** options strategies (covered call, cash-secured put, LEAPS call, PMCC, vertical debit spread) instead of outright stock purchases — each suggestion carrying entry legs, exit-anchor TP/SL, and an OCO bracket sketch the order-placement layer can submit atomically.
> **Source of truth:** [`trading_agent/long_term_evaluator.py`](../../trading_agent/long_term_evaluator.py) (orchestrator, next session), [`trading_agent/decision_engine.py`](../../trading_agent/decision_engine.py) (strategy scorers — per CI invariant 2, scoring helpers may only be **defined** here or in `chain_scanner.py`), [`trading_agent/streamlit/long_term_evaluator_ui.py`](../../trading_agent/streamlit/long_term_evaluator_ui.py) (consumer).
> **Phase:** 2  •  **Group:** strategy
> **Depends on:** `41_positions_provider.md` (holdings input), `13_preset_system_hot_reload.md` (risk-tolerance knobs), `01_pop_from_delta.md` (POP approximation reused for short legs), `03_credit_to_width_floor.md` (the invariant C/W formula a debit spread inverts).
> **Consumed by:** `streamlit/long_term_evaluator_ui.py` — renders the five-section evaluator panel.

---

## 1. Theory & Objective

The credit-spread agent earns *theta* on short-DTE volatility-selling. That book is one income lane. This skill adds a **long-term** lane: when the operator already owns stock or has tickers on a watchlist they want long exposure to, an outright share purchase often isn't the right vehicle. Three options-structured alternatives almost always dominate on capital efficiency, downside, or both:

1. **Sell premium against held stock** — covered calls (and collars at higher quantity) convert dormant stock into a yield-producing position with no incremental capital outlay.
2. **Get paid to wait for a better entry** — cash-secured puts on watchlist tickers monetise the wait until the underlying trades at the strike. Worst case: assigned at a price you already wanted.
3. **Replace shares with deep-ITM long-dated calls** — a LEAPS call with Δ ≈ 0.85 behaves like 85 shares but costs a third of the capital. Frees the remaining 67% for income overlays or diversification.

The evaluator's job is to score each candidate under the same risk-tolerance preset the credit-spread agent uses (`balanced`, `conservative`, `aggressive`), rank within and across strategy families, and surface the top-N alongside their **exit anchors** so the operator has a complete entry-to-exit plan before placing the order. Exit anchors feed directly into the Schwab `TRIGGER + OCO` bracket the order layer (next session) submits as one atomic ticket — entry fill triggers a child OCO holding the take-profit limit and the stop, and whichever child fills cancels its sibling. The operator never holds an entry without a paired exit.

The evaluator is **read-only by design**. It produces recommendations + bracket sketches. The operator clicks `Place` (next session). The system rules forbid this agent from executing trades on the operator's behalf.

## 2. Mathematical Formula

Each strategy family has its own scoring function. All scorers live inside `decision_engine.py` (CI invariant 2). All consume a normalised `OptionContract` dict (`strike`, `delta`, `bid`, `ask`, `dte`, `iv`, `symbol`) and the active `PresetConfig`.

### 2.1 Covered call (THIS SESSION)

```text
For each (held_ticker, candidate_short_call):
  net_credit       = mid(bid, ask) × 100                       [dollars per contract]
  capital_at_risk  = cost_basis_per_share × 100 − net_credit   [dollars, ≥ 0]
  static_return    = net_credit / capital_at_risk              [unitless]
  if_assigned_ret  = ((strike − cost_basis_per_share) × 100 + net_credit) / capital_at_risk
  annualised_static = (1 + static_return)^(365 / dte) − 1
  pop_short_call    = 1 − |Δ_short|                            [reuses skill 01]

  score = annualised_static × pop_short_call
          subject to:
            |Δ_short|       ≤ preset.cc_max_short_delta   (default 0.30)
            dte              ∈ preset.cc_dte_band          (default [30, 60])
            iv_rank_today    ≥ preset.cc_min_iv_rank       (default 0.25)
            strike           ≥ cost_basis_per_share × 1.01 (covered, never write below cost basis)
```

`annualised_static × pop_short_call` is the *risk-adjusted yield*: high pop_short_call means low assignment probability (you keep the premium AND the stock), high annualised static return rewards income density. A 60-DTE call at Δ=0.25 with 1.5% credit beats a 30-DTE Δ=0.40 call with 2.0% credit on the same risk-adjusted yield because the latter's assignment probability triples.

### 2.2 Cash-secured put — Wheel entry (implemented 2026-09-29)

```text
For each (watchlist_ticker passing §2.7, candidate_short_put):
  credit          = single-leg quote credit × 100              [dollars per contract]
  collateral      = strike × 100                               [cash held for assignment]
  capital_at_risk = collateral − credit
  static_return   = credit / capital_at_risk
  annualised      = (1 + static_return)^(365 / dte) − 1
  pop_short_put   = 1 − |Δ_short|                              [skill 01]
  effective_entry = strike − credit / 100                      [cost basis if assigned]

  score = annualised × pop_short_put
          subject to:
            strike         ∈ [0.85 × spot, 0.97 × spot]   (csp_strike_band)
            |Δ_short|      ≤ 0.30                         (csp_max_short_delta)
            dte            ∈ [21, 60]                     (csp_dte_band)
            collateral     ≤ max_collateral               (when supplied)
            bid/ask        not wide: > 15¢ AND > 5% of mid fails  (skill 29; preset max_leg_spread_*)
            credit         > 0
            iv_rank        ≥ 0.25 when supplied           (csp_min_iv_rank; fail-open if absent)
```

The Wheel: CSP on a §2.7-screened ticker → if assigned, the shares flow into the §2.1 covered-call leg (the evaluator routes tickers held ≥ 100 shares to covered calls and everything else on the watchlist to CSPs).

### 2.7 Fundamentals quality screen — "willing to own" (implemented 2026-09-29)

`trading_agent/fundamentals_screen.py` gates CSP candidates on the skill-47 `/fundamentals` block **before** any chain fetch. Pass/fail only — ranking stays in the scorer (invariant 2).

| Field | Pass when | Default |
|---|---|---|
| `market_cap` | ≥ min | $10B |
| `pe_ratio` | 0 < P/E ≤ max | 40 |
| `eps_ttm` | > min | 0 |
| `net_profit_margin_ttm` (%) | ≥ min | 8 |
| `roe` (%) | ≥ min | 10 |
| `beta` | ≤ max | 1.6 |
| `vol_avg_10d` | ≥ min (0.0 = unknown) | 1M shares |

Missing / `None` / non-numeric → `missing:<field>` → **fail closed**. Every failing reason is returned, not just the first.

### 2.3 LEAPS call as synthetic stock (NEXT SESSION)

```text
[Next session — section reserved]

For deep-ITM long call with dte ≥ 365:
  effective_share_proxy = Δ_long × 100              [Δ-equivalent shares per contract]
  capital_efficiency    = (spot × 100) / (mid × 100)  = spot / mid
  score ≈ capital_efficiency × Δ_long
      subject to Δ_long ≥ preset.leaps_min_delta (default 0.80)
                 dte    ≥ preset.leaps_min_dte    (default 365)
                 bid-ask spread ≤ preset.leaps_max_spread_pct  (default 0.05)
```

### 2.4 Poor-man's covered call (NEXT SESSION)

```text
[Next session — section reserved]

PMCC = long LEAPS call + short near-dated call
  effective_cost = long_call_debit − short_call_credit
  effective_max  = (short_strike − long_strike) × 100 − effective_cost   (intrinsic-capped)
  score ≈ effective_max / effective_cost × pop_short_call
      gate: short_strike > long_strike  AND  short_strike > spot
```

### 2.5 Vertical debit spread (NEXT SESSION)

```text
[Next session — section reserved]

Bull call debit spread:
  net_debit   = long_call_mid − short_call_mid       [dollars per contract × 100]
  max_profit  = (short_strike − long_strike) × 100 − net_debit
  max_loss    = net_debit
  reward_risk = max_profit / max_loss
  pop         = N((short_strike − spot − net_debit) / (σ × √(dte/365)))   (rough Black-Scholes proxy)
  score       = reward_risk × pop
```

### 2.6 Exit anchors per strategy

| Strategy | Take-profit | Stop-loss | Stop type |
|---|---|---|---|
| Covered call | BTC short call at 50% of credit received | Underlying drops below cost_basis × 0.92 → roll/close | Underlying-price-based |
| Cash-secured put | BTC short put at 50% of credit | \|Δ_short_put\| ≥ 0.45 → close (assignment risk material) | Δ-based, evaluated each cycle |
| LEAPS call | Scale 50% at +50% on debit, full at +100% | Underlying < entry × 0.85 OR LEAPS premium ≤ 50% of debit | Underlying-price-based + premium floor |
| PMCC | Long: per LEAPS rules. Short: per CC rules | Same | Two independent brackets |
| Vertical debit | STC at 80% of max profit | At 50% of debit paid | Spread-mid-based |

The OCO bracket the order layer submits encodes these anchors as: `take_profit` = limit order at the TP price, `stop_loss` = stop-limit order at the SL trigger with a `stop_limit_offset` (default 5% of stop trigger) to avoid getting gapped through. The two children are linked under `orderStrategyType: "OCO"` so filling either cancels the other.

### 2.8 Earnings gate (implemented 2026-09-29)

A scheduled earnings report inside the option's life is a known gap risk: the premium is richer because the market prices the move, and a miss can drive the stock straight through the strike. `wheel_screen` looks up days-to-earnings via `EarningsCalendar` (yfinance, cached 12 h) and applies `earnings_policy`:

| Policy | Behaviour |
|---|---|
| `avoid` (default) | Only expirations with DTE **<** days-to-earnings are tried. If none of the listed expirations in [21, 60] DTE lands before the report → skip with `earnings_in_<N>d (no listed expiration ≥21d before it)`. |
| `allow` | All expirations tried; each recommendation carries `earnings_before_expiry: true/false`. |

Unknown earnings date (lookup failed / none listed) → **not excluded**, flagged `earnings_known: false` so the operator checks manually. Candidate expirations are every weekly Friday plus each monthly (third Friday) in [21, 60] DTE, nearest to `target_dte` first; the first one with a listed chain wins.

### 2.9 Wheel trade lifecycle — stage, submit, manage, expire (implemented 2026-09-29)

`trading_agent/wheel_policy.py` is the single source for the constants every stage uses (`TAKE_PROFIT_PCT_OF_CREDIT=0.50`, `CSP_STOP_ABS_DELTA=0.45`, `MAX_CSP_COLLATERAL_PCT_OF_EQUITY=0.40`, strategy names, exit-signal names).

| Stage | Where | Rule |
|---|---|---|
| Screen | MCP `wheel_screen` | CSPs for watchlist names held < 100 shares; covered calls for names held ≥ 100 (Alpaca holdings). Each row carries a stageable `plan` (`build_single_leg_plan`). CSP rows are dropped (diagnostic `market_state_<STATE>_pauses_new_csp`) while the market risk state pauses new puts (skill 58). |
| Stage | `/propose` → `pending_orders/` | Operator approval, unchanged (skill 51). |
| Submit | `executor_promote._submit_wheel` | `check_wheel_order`: CSP collateral ≤ options buying power (unknown → reject) and ≤ 40 % of equity; no new CSP while the market risk state pauses them (skill 58 — CAUTION / DEFENSIVE / CAPITULATION, snapshot ≤ 4 days old); covered call requires ≥ 100 × qty shares (never naked). Then `OrderExecutor.execute_single_leg`: sell-to-open limit at mid, then halfway mid→bid, then **at the bid** — the bid attempt only while the live quote still passes the skill-29 width gate (preset `max_leg_spread_*`); journal `submitted` **only on a confirmed fill**. |
| Manage | `PositionMonitor._check_wheel_exit` (5-min cycle) | `PROFIT_TARGET` at 50 % of credit; CSP `DELTA_STOP` when \|Δ\| ≥ 0.45 (debounced 3 cycles; Δ from the chain via `attach_wheel_short_deltas`). No hard stop, strike-proximity, DTE-safety or regime-shift exits — assignment is the plan. Closes use a single-leg limit order (`close_spread` → `_close_spread_mleg`). |
| Expire | `wheel_lifecycle.reconcile` (once per day) | Expired CSP + ≥ 100 × contracts shares → `assigned`, else `expired_worthless`; expired CC + shares gone → `called_away`, else `expired_worthless`. Journals a `closed` row (premium kept) so the trade leaves `open_positions`. |

## 3. Reference Python Implementation

### 3.1 Walking-skeleton orchestrator (this session)

```python
# trading_agent/long_term_evaluator.py
@dataclass(frozen=True)
class Recommendation:
    ticker: str
    strategy: str                # "covered_call" | "cash_secured_put" | ...
    legs: List[Dict[str, Any]]   # OCC symbol + side + qty per leg
    entry_limit: float
    take_profit_limit: float
    stop_trigger: float
    stop_limit_offset: float
    score: float
    rationale: str               # one-line operator-facing summary
    metrics: Dict[str, float]    # ROC, annualised yield, POP, etc.

class LongTermEvaluator:
    def __init__(self, *, market_data, positions_provider, preset,
                 sector_for=trading_agent.utils.sector_for):
        ...

    def recommend(self, watchlist: List[str], *,
                  max_per_strategy: int = 3) -> List[Recommendation]:
        positions = self.positions_provider.snapshot()
        held = {p.ticker for p in positions if p.kind == "stock"}
        recs: List[Recommendation] = []

        # § Income overlay (this session)
        for ticker in held & set(watchlist):
            recs.extend(self._income_overlay(ticker, positions))

        # § Entry vehicles (next session)
        # for ticker in set(watchlist) - held:
        #     recs.extend(self._entry_vehicle(ticker))

        # § Manage existing options (next session)
        # for opt_pos in [p for p in positions if p.kind == "option"]:
        #     recs.append(self._surveil_existing(opt_pos))

        recs.sort(key=lambda r: r.score, reverse=True)
        return recs
```

### 3.2 Covered-call scorer — lives in `decision_engine.py` (CI invariant 2)

```python
# trading_agent/decision_engine.py — additions
def _score_covered_call(
    *,
    short_call: Dict[str, Any],   # OptionContract dict
    cost_basis: float,            # per-share
    preset: Any,
) -> Optional[Tuple[float, Dict[str, float]]]:
    """Return (score, metrics) or None if any gate rejects."""
    strike = float(short_call["strike"])
    if strike < cost_basis * 1.01:           # never write below cost basis
        return None
    delta_abs = abs(float(short_call["delta"]))
    if delta_abs > preset.cc_max_short_delta:
        return None
    dte = int(short_call["dte"])
    lo, hi = preset.cc_dte_band
    if not (lo <= dte <= hi):
        return None
    credit = _quote_credit_single(
        bid=float(short_call["bid"]), ask=float(short_call["ask"]),
    ) * 100.0
    if credit <= 0:
        return None
    capital_at_risk = cost_basis * 100.0 - credit
    static_return = credit / capital_at_risk
    annualised = (1.0 + static_return) ** (365.0 / dte) - 1.0
    pop = _pop_from_delta(float(short_call["delta"]))   # reuses skill 01
    score = annualised * pop
    return score, {
        "credit": credit,
        "capital_at_risk": capital_at_risk,
        "static_return": static_return,
        "annualised_return": annualised,
        "pop": pop,
        "dte": float(dte),
        "short_delta_abs": delta_abs,
    }
```

`_quote_credit_single` is a single-leg sibling of `_quote_credit` (skill 03) — bid-ask mid with the same fill-haircut. Defined alongside `_quote_credit` in `chain_scanner.py` so the C/W floor invariant scanner still sees the spread-side `_quote_credit` it expects.

### 3.3 PresetConfig fields (CSP / CC wired 2026-10-05; LEAPS / debit-spread rows still design-only)

```python
# trading_agent/strategy_presets.py — PresetConfig additions
cc_max_short_delta:    float = 0.30
cc_dte_band:           Tuple[int, int] = (30, 60)
cc_min_iv_rank:        float = 0.25
csp_max_short_delta:   float = 0.25
csp_strike_band_pct:   Tuple[float, float] = (0.85, 0.97)   # of spot
leaps_min_delta:       float = 0.80
leaps_min_dte:         int = 365
leaps_max_spread_pct:  float = 0.05
debit_spread_max_dte:  int = 90
debit_spread_min_pop:  float = 0.40
```

Conservative preset increases all `min_*` knobs; aggressive relaxes them. See skill 13 for the hot-reload contract.

### 3.4 Streamlit consumer

```python
# trading_agent/streamlit/long_term_evaluator_ui.py
def render_long_term_evaluator() -> None:
    """Top-level Streamlit renderer for the Long-Term Evaluator tab.

    Wired into ``trading_agent/streamlit/app.py`` alongside the existing
    Live / Backtest / LLM / Watchlist tabs.
    """
    positions = _render_holdings_input()
    if positions is None:
        return
    provider = ManualPositionsProvider(positions=positions)
    _render_portfolio_snapshot(provider)
    evaluator = LongTermEvaluator(
        positions_provider=provider,
        call_chain_fetcher=_make_chain_fetcher(),
        preset=_active_preset_or_none(),
        config=EvaluatorConfig(),
    )
    recs = evaluator.recommend(load_watchlist().symbols())
    _render_manage_existing(provider, recs)
    _render_income_overlay(recs)
    # _render_entry_vehicles(recs)   # next session
    # _render_portfolio_gaps(...)    # next session
```

### 3.5 Cash-secured put scorer + Wheel path (2026-09-29)

```python
    capital_at_risk = max(0.01, collateral - credit)
    static_return = credit / capital_at_risk
    annualised_return = (1.0 + static_return) ** (365.0 / max(1, dte)) - 1.0
    pop = _pop_from_delta(float(short_put["delta"]))   # skill 01
    effective_entry = strike - credit / 100.0          # cost basis if assigned
```

- `decision_engine._score_cash_secured_put[_with_reason]` — reject taxonomy adds `LT_REJECT_STRIKE_OUT_OF_BAND`, `LT_REJECT_COLLATERAL_OVER_BUDGET`.
- `LongTermEvaluator(..., put_chain_fetcher=, fundamentals_fetcher=, spot_fetcher=)` — CSP path runs only when all three are supplied; `EvaluatorConfig.csp_*` holds TP (50 % of credit), stop (|Δ| ≥ 0.45, `stop_kind="delta_threshold"`), `csp_max_collateral`, and `wheel_screen: WheelScreenConfig`. `evaluator.last_diagnostics[ticker]` explains every skip.
- MCP `wheel_screen(watchlist, target_dte=35, max_collateral=None)` (skill 48) — read-only; picks the weekly expiration nearest `target_dte` via `calendar_utils.next_weekly_expiration`.
- Tunables: `PresetConfig.cc_max_short_delta / cc_dte_band / cc_min_iv_rank / csp_max_short_delta / csp_dte_band / csp_min_iv_rank / csp_strike_band` (2026-10-05) with the same defaults as the `_CC_DEFAULT_*` / `_CSP_DEFAULT_*` fallbacks, so wiring changed no behaviour; summary token `Wheel CSP Δ≤… • CC Δ≤…`, Streamlit "Wheel" block.

## 4. Edge Cases / Guardrails

- **Read-only contract.** `LongTermEvaluator.recommend()` never writes, never submits an order. The order layer (Phase 5, dedicated session) is the only thing that talks to Schwab Trader API. Conformance: `test_skill_40_evaluator_does_not_submit_orders` asserts no `_submit_*` calls are reachable from the evaluator's import graph.
- **Cost basis required for covered calls.** A held-stock position without a `cost_basis` field is skipped with a `MISSING_COST_BASIS` reject reason rather than defaulting to spot. Writing CCs below cost basis is the canonical wash-sale + locked-in-loss trap; the evaluator refuses by design.
- **Strikes below cost basis silently filtered.** `_score_covered_call` returns `None` (not raise) when `strike < cost_basis × 1.01`. The 1% buffer absorbs penny-strike rounding near a recent buy.
- **OCC option positions outside the universe.** If `positions_provider.snapshot()` returns an option position whose underlying isn't on the watchlist, the surveillance section still includes it — open positions always get monitored, watchlist only gates *new* recommendations.
- **Quantity below 100 → no CC suggestion.** Covered calls require 100 shares per contract. A 50-share holding gets surfaced in the snapshot but produces no covered-call recommendation. A note in the rationale field explains why.
- **Multiple holdings of the same ticker (lots).** When the operator has two lots of AAPL at different cost bases, the evaluator uses the **average cost** (qty-weighted) for the gate. A future enhancement (skill 40 v2) could surface a per-lot recommendation set; out of scope this session.
- **High-IV-rank gate.** `cc_min_iv_rank` defaults to 0.25. The intent is to avoid writing calls in low-vol environments where premium is too thin to be worth the upside cap. If IV rank data isn't available for a ticker, the gate is **skipped, not failed** (fail-open) and a `MISSING_IV_RANK` note is attached to the rationale.
- **Exit-anchor sanity check.** The take-profit limit must satisfy `tp_limit > 0` and `tp_limit < entry_limit` (you can never BTC for more than you sold for). The stop trigger must satisfy `stop_trigger > entry_limit × 1.5` for a short call (closing at >150% of the credit is the canonical "this trade went wrong" line). Violations are caught by `Recommendation.__post_init__`.
- **OCO bracket constraints, single-leg vs multi-leg.** Single-leg tickets (CC, CSP, LEAPS) submit native Schwab OCO. Multi-leg tickets (PMCC, debit spread) use a two-step pattern: entry submits as a multi-leg order, the agent submits the closing OCO **only after the entry fill is confirmed**, gated by an `entry_fill_observed` flag in the journal. Phase 5 owns this branching; this skill documents the contract.
- **Preset hot-reload survives evaluator instances.** The evaluator captures `preset` at construction (same caveat as skill 36's `TickerFilters`). The Streamlit consumer recreates the evaluator each render so a preset edit is picked up on the next refresh.

- **CSP collateral vs account size** — a CSP ties up strike × 100 in cash. On a $30k account, `max_collateral` keeps a single $300 strike from consuming the whole book; over-budget contracts are rejected with `collateral_over_budget` and surfaced in `last_diagnostics`.
- **Fundamentals missing or zero-volume** — screen fails closed (`missing:<field>`); `vol_avg_10d == 0.0` counts as missing (Schwab returned 0.0 for unmapped volume on 2026-09-29).
- **Tiny credit** — when 50 % of the credit rounds to 0 or to the entry in cents, the contract is skipped (a `Recommendation` with TP ≥ entry would raise).
- **Held ≥ 100 shares** — no CSP; the ticker is on the covered-call leg of the wheel.
- **Fetcher failure** — contained per ticker (`data_unavailable`); other tickers still evaluate.
- **No iv_rank in chain** — the MCP chain carries no `iv_rank`, so the IV gate is fail-open there; low-IV names can pass and should be judged on the yield shown.

- **Earnings inside every candidate expiration** — during earnings season `avoid` can empty the list (2026-09-29: all 9 candidates reported before the 11/20 monthly; PEP in 9 days). That is the gate working; use `allow` only with an explicit decision to hold through the report.
- **Earnings date unknown** — flagged, not excluded (`earnings_known: false`); yfinance outages must not silently hide or silently approve.
- **Unfilled open order** — `execute_single_leg` cancels after each attempt and returns `unfilled` / `unresolved`; nothing is journalled, so the reconciler can never book the credit of a trade that never existed.
- **Delta unavailable** — `short_delta=None`; the CSP delta stop cannot fire that cycle (profit target still works). Logged as a warning.
- **Holdings unavailable at expiry** — reconcile is skipped and the day-sentinel not written, so it retries next cycle instead of booking every put as worthless.
- **Pre-existing shares** — a CSP on a ticker already held ≥ 100 shares would reconcile as `assigned` even if it expired worthless. The screen never proposes CSPs for such tickers.
- **Manually placed Wheel legs** (no trade plan) are still inferred as `Naked Short` and get spread exits — always stage through `/propose`.

- **Wide or pre-market quotes (2026-09-30)** — both Wheel scorers apply the skill-29 per-leg gate with the preset's `max_leg_spread_cents` / `max_leg_spread_pct_mid`. Pre-market BMY $60P at 0.40/1.11 had ranked first on a mid-based 21.8 % yield; it now rejects as `leg_spread_wide` and shows in `last_diagnostics`.

- **Final attempt at the bid (2026-09-30)** — VZ $43P 0.21/0.36 went unfilled at 0.28 and 0.25 on the paper account (Alpaca's paper simulator rarely fills inside the spread). A third attempt sells at the bid, but is skipped when the live quote has widened past the width gate since screening, so the premium is never given away on a wide quote.

- **Fill price vs estimate (2026-09-30)** — the trade plan is saved before submission with the estimated credit, and the monitor's 50 % target reads `net_credit` from it. On a confirmed fill `execute_single_leg` reads `filled_avg_price` (falls back to the limit) and `_record_fill_credit` rewrites that run's `net_credit`, `max_loss` and `credit_to_width_ratio` (keeping `estimated_net_credit`), atomically; promote journals the fill price. VZ $43P: estimate 0.26, fill 0.21 → target corrected from $13 to $10.50.
- **Unfilled runs shadowing a fill (2026-09-30)** — `group_into_spreads` walks trade-plan entries oldest-first and the first entry containing the legs claims them. An earlier run whose attempts all cancelled unfilled still carried the same contract and the estimate, so it claimed the live position. `execute_single_leg` now calls `_mark_run_unfilled` (sets `valid=False`) when every attempt is confirmed cancelled; an `unresolved` cancel is left valid because it may still fill. Both updates go through `_update_run_trade_plan` (atomic temp+rename).

- **Watchlist shapes (2026-10-05).** `wheel_screen` accepts a list, a comma string, or a list sent as JSON text (`'["VZ"]'`, as some MCP clients send). Before the fix that last form became a single ticker and failed `missing:fundamentals`.

- **Wheel credits at natural (2026-10-05, §6.1).** Both Wheel scorers and `build_single_leg_plan` price the short leg at the bid under the default `fill_model`, so screen yields match what the paper account fills (VZ: mid 0.28 vs fill 0.21).

- **200-day filter (2026-10-05, backlog §2).** `LongTermEvaluator(trend_fetcher=…)` returns the 200-day SMA; a CSP is refused when spot < SMA (`below_200d_sma (…)`) and **fails closed** when the SMA is unknown (`trend_unavailable`). `PresetConfig.csp_require_above_sma200` (default True) turns it off. `wheel_screen` supplies yfinance daily closes (cached per process). Legacy callers without a `trend_fetcher` are unaffected.
- **One pick per sector (2026-10-05).** `sector_fetcher` + `PresetConfig.csp_max_per_sector` (default 1, 0 = off): CSPs from at most N distinct tickers per sector survive, best score first (several strikes of a kept ticker stay); dropped tickers get `sector_cap (<sector>: <kept> ranked higher)`. Sector = `sector_map.TICKER_SECTOR_MAP` (now incl. common Wheel names: banks, telecom, staples, pharma, energy, utilities), else yfinance `info["sector"]` translated to SPDR names, else the ticker itself — unknown names are never lumped together.

## 5. Cross-References

- `41_positions_provider.md` — the holdings-input contract (`PositionsProvider` ABC + `ManualPositionsProvider`).
- `13_preset_system_hot_reload.md` — defines the `cc_*`, `csp_*`, `leaps_*`, `debit_spread_*` knobs.
- `01_pop_from_delta.md` — `pop ≈ 1 − |Δ|` reused for short-leg POP across CC / CSP scoring.
- `03_credit_to_width_floor.md` — debit-spread scoring inverts the C/W invariant; the formula `|Δshort| × (1 + edge_buffer)` is unchanged on the credit side.
- `30_profit_target_management.md` — the credit-spread agent's 50%-of-credit take-profit anchor is the same number we reuse for CC and CSP exits.
- `32_telegram_operator_alerts.md` — when a managed long-term position hits its TP/SL anchor, the evaluator fires the same operator-alert dedup path (next session).

---

*Last verified against repo HEAD on 2026-10-05.*
