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

### 2.2 Cash-secured put (NEXT SESSION)

```text
[Next session — section reserved]

score ≈ annualised_premium_yield × pop_short_put
      where pop_short_put = 1 − |Δ_short|
      capital_at_risk = strike × 100 − premium     [collateral held for assignment]
      gate: strike between [0.85 × spot, 0.97 × spot]   (room to fall to your entry)
```

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

### 3.3 PresetConfig fields added (next session, listed here for design completeness)

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

## 5. Cross-References

- `41_positions_provider.md` — the holdings-input contract (`PositionsProvider` ABC + `ManualPositionsProvider`).
- `13_preset_system_hot_reload.md` — defines the `cc_*`, `csp_*`, `leaps_*`, `debit_spread_*` knobs.
- `01_pop_from_delta.md` — `pop ≈ 1 − |Δ|` reused for short-leg POP across CC / CSP scoring.
- `03_credit_to_width_floor.md` — debit-spread scoring inverts the C/W invariant; the formula `|Δshort| × (1 + edge_buffer)` is unchanged on the credit side.
- `30_profit_target_management.md` — the credit-spread agent's 50%-of-credit take-profit anchor is the same number we reuse for CC and CSP exits.
- `32_telegram_operator_alerts.md` — when a managed long-term position hits its TP/SL anchor, the evaluator fires the same operator-alert dedup path (next session).

---

*Last verified against repo HEAD on 2026-06-16.*
