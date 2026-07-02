# Sell-Signals Feature Plan

**Working document — not a skill doc yet.** Becomes `docs/skills/43_sell_signals.md` once Phase 0 lands.

**One-line intent.** Advisory sell-side decision support for the operator's long-term stock holdings — take-profit / trim / cut-loss / rotate signals delivered through the same Long-Term Evaluator surface + Telegram digest that already produces buy-side recommendations. Read-only by design; the operator places the sell orders.

**Why now.** The evaluator today only tells you what to BUY (covered calls on held, entry vehicles on watchlist-only). Half the decision surface is missing — nothing tells you when GOOG's +120% run is stretched, when SOFI's -17% drawdown is signaling to cut, or when NFLX's regime has flipped. Filling the gap makes the hourly digest complete.

---

## Signal taxonomy — five signal types, ranked by staleness risk

The plan covers five distinct signal families. Each phase can ship them individually so you can gate on which ones actually work in your book before building the next.

| Signal | Fires when | Freshness needed |
|---|---|---|
| **Take-profit target** | Position up ≥ N% on cost basis (e.g. GOOG +120%, NOK +191% both flag) | End-of-day OK |
| **Trailing stop** | Drawdown from 20-day local max exceeds M% | Intraday matters |
| **Regime shift** | Regime classifier flips (bullish → bearish, mean-rev → trend-down) | Intraday matters |
| **Momentum exhaustion** | RSI > 70 + MACD divergence, or RSI < 30 + failing to bounce | End-of-day OK |
| **Sector rotation** | Ticker's sector enters relative underperformance vs SPY | Weekly OK |

Each signal produces a 0-1 score + a reasoning string. An aggregate `sell_score` per position combines them (weighted sum, weights configurable per operator preset). Tiered action tag:

- `≥ 0.75` — **SELL** (cut position now)
- `0.50–0.75` — **TRIM** (scale out 30-50%)
- `0.25–0.50` — **WATCH** (no action, on radar)
- `< 0.25` — **HOLD**

---

## Phased build

Each phase is one session, unless noted. Phases are additive — after Phase 1 you have working sell signals; every subsequent phase makes them smarter.

### Phase 0 — Design doc (½ session)

Draft `docs/skills/43_sell_signals.md` covering the taxonomy above, the scoring math per signal type, the aggregation formula, edge cases, and the read-only contract. Includes the cross-reference table to skills 40 (evaluator), 41 (positions), 42 (scheduler) so the composition story is explicit.

Deliverables: skill doc committed, invariant scan passes, freshness gate green.

### Phase 1 — Take-profit + trailing-stop scorers (1 session)

The two lowest-hanging signals ship first because they need nothing except cost basis (which you already paste) + current price + N-day rolling max (cheap to compute from yfinance or Schwab historical).

New pure-function scorers in `decision_engine.py` (CI invariant 2):

- `_score_take_profit(position, current_price, config) → (score, metrics)` — score rises linearly from `config.tp_start_pct` (e.g. 50% gain) to `config.tp_full_pct` (e.g. 150% gain).
- `_score_trailing_stop(position, current_price, high_20d, config) → (score, metrics)` — score rises from 0 at 5% drawdown to 1 at `config.trailing_stop_pct` (default 15%).

New orchestrator method on `LongTermEvaluator`:

- `sell_signals(positions, prices, highs) → List[SellRecommendation]` — walks each held stock, calls the scorers, aggregates.

New dataclass `SellRecommendation` mirrors the shape of the buy-side `Recommendation` (ticker, action_tag, qty_to_sell, entry_limit, rationale, metrics) so the Streamlit + Telegram consumers treat both uniformly.

Conformance tests: `test_skill_43_take_profit_scores_at_tp_start`, `test_skill_43_trailing_stop_hits_at_drawdown_threshold`, `test_skill_43_scorer_lives_in_decision_engine`.

Deliverables: two scorers + orchestrator method + 8-10 tests + skill 43 §2 + §3 filled in.

### Phase 2 — Streamlit "Manage Existing" gets teeth (1 session)

Today the Manage Existing section on the Long-Term Evaluator tab is a stub. Replace it with a table:

| Ticker | Qty | Cost basis | Current | P/L% | Signal | Action | Detail |
|---|---|---|---|---|---|---|---|
| GOOG | 15 | $169.07 | $371.53 | +119.7% | 🎯 Take-profit | **TRIM 5 shares** | Score 0.68 · scale-out band hit |
| NFLX | 20 | $102.06 | $79.50 | -22.1% | ⚠ Trailing stop | **SELL 20 shares** | Score 0.81 · drawdown 22% > 15% floor |
| MSFT | 9 | $414.66 | $393.03 | -5.2% | · Watch | — | Score 0.32 |

Rows are color-coded by action tag. Clicking a row expands the reasoning trail (which signal fired, what score, what threshold was hit). No order placement — copy-paste-ready ticker/qty/price the operator drops into their broker.

Deliverables: `_render_manage_existing()` upgraded from stub to real table, driven by `sell_signals()`.

### Phase 3 — Telegram digest gets a sell section (½ session)

The hourly Telegram digest picks up a new `🚨 Sell signals` block above the covered-call section (sell decisions are more urgent than income-overlay decisions):

```
🚨 Sell signals — 2 actionable
1. NFLX: SELL 20 shares · trailing stop hit (–22% drawdown)
   Target: market at open, expected proceeds ~$1,590
2. GOOG: TRIM 5 shares · take-profit band 0.68
   Target: limit $370 (scale-out at 50% of position)

🎯 Income overlay — 1 covered-call candidate
(existing section)
```

Body-hash dedup already handles "identical digest within a UTC day stays silent" — no new dedup logic needed. Bracket sketch for each sell is a single-leg market-sell for `SELL` or limit at current mid for `TRIM`.

Deliverables: `compose_digest_body()` gets a sell-section, one new test pinning the section rendering + section ordering.

### Phase 4 — Regime shift + momentum exhaustion (1 session)

Adds the two "intraday matters" signals. Both reuse the existing regime classifier + RSI/MACD infrastructure that's already in `multi_tf_regime.py`.

- `_score_regime_shift(position, current_regime, prior_regime, config)` — score = 1 when regime flipped bullish→bearish on the daily; decays over 5 trading days if no confirmation.
- `_score_momentum_exhaustion(position, rsi, macd_hist, config)` — score rises when RSI > 70 AND MACD histogram has topped-and-turned; symmetric on the downside.

These signals require a **prior state** — Phase 4 introduces a small `sell_signal_state` sidecar JSON at `knowledge_base/sell_signal_state.json` (atomic write, same pattern as watchlist_store) that snapshots per-position regime + RSI at every scheduler cycle so the "regime just flipped" and "RSI just topped" detections are cheap.

Deliverables: 2 more scorers + state sidecar + skill 43 §2.3-§2.4 + conformance tests.

### Phase 5 — Sector rotation (½ session)

The final signal. Uses `sector_map.py` (already extended for your book) to group holdings by sector, then compares each sector's 20-day return vs SPY. When a held position's sector enters bottom-quartile relative performance, sector-rotation score fires.

Adds one new scorer `_score_sector_rotation(position, sector_ret_20d, benchmark_ret_20d, config)`. Cheap because sector ETF returns come from the same yfinance/Schwab fetch as the rest.

Deliverables: last scorer + skill 43 §2.5 + conformance test + docs update.

### Phase 6 (optional, dedicated session) — Schwab order placement for sells

Currently gated behind Phase 5 of the buy-side work (Schwab Trader API integration). Sell orders share the same infrastructure — the "Preview → Confirm → Place" flow handles buy or sell equivalently. Once the Schwab executor lands for buys, sells are a small delta.

Explicitly NOT in scope for the current sell-signals feature. Advisory-only is the launch shape.

---

## Cross-cutting decisions to make BEFORE Phase 0

Three questions I'd want your answer to before writing the skill doc. Each has a sensible default; call out where you want to override.

**1. Tax awareness — should short-term vs long-term capital gains factor into signals?** A `SELL` on NOK bought 60 days ago produces short-term gains (taxed as income) vs the same sell held past 366 days (long-term rates). Signals could either (a) ignore tax entirely and score purely on market signals, or (b) score-adjust to prefer holding an extra ~30 days when a position is approaching the long-term boundary. Default: (a) ignore tax, keep signals pure — tax planning is a separate consideration you make when you actually place the order. Override with a note if you want (b).

**2. Interaction with covered calls — if a position is short a call and gets a SELL signal, what happens?** If GOOG has a short call open and hits a take-profit sell signal, closing the shares before the call expires leaves you naked short — the call has to be bought back first. Default: the sell-signal payload includes a `call_position_check` boolean; when true, the bracket sketch adds "BTC short call first" as a prerequisite step. Simpler than blocking the sell entirely.

**3. Per-position thresholds vs preset-wide.** The `tp_start_pct` and `trailing_stop_pct` numbers might not fit every ticker — NOK at +191% deserves a different scale than NFLX at -22%. Default: preset-wide thresholds first (fast to ship), with a `per_ticker_overrides` config field in Phase 4 for ticker-specific tuning. Override if you want per-ticker from Phase 1 — adds ~½ session.

---

## What each phase costs

| Phase | Deliverables | Effort |
|---|---|---|
| 0 | Skill 43 draft | ½ session |
| 1 | TP + trailing-stop scorers + orchestrator + tests | 1 session |
| 2 | Streamlit "Manage Existing" table | 1 session |
| 3 | Telegram digest sell section | ½ session |
| 4 | Regime shift + momentum exhaustion + state sidecar | 1 session |
| 5 | Sector rotation | ½ session |
| 6 | Schwab order placement (blocked on buy-side executor) | Dedicated session |

Total for the advisory-only launch (Phases 0-5): **~4½ sessions**. Phase 6 lands whenever the buy-side executor does.

---

## Fastest useful ship — MVP path

If you want ONE hourly-digest addition that meaningfully helps this book today, the MVP is **Phase 0 + Phase 1 + Phase 3** (skip Streamlit for now). Two signals (take-profit + trailing stop) + Telegram delivery.

That covers the most valuable cases in your current book immediately:

- **NOK +191%** → strong take-profit signal, TRIM alert
- **GOOG +120%** → moderate take-profit, TRIM alert
- **NFLX -22%** → trailing-stop hit, SELL alert
- **SOFI -17%** → approaching trailing-stop threshold, WATCH → SELL

That's a 2-session investment (Phase 0 + Phase 1) with the Telegram surface as the last ~2 hours of work, plus one commit for CI housekeeping. You'd have working sell signals on your phone tomorrow. The Streamlit UI and the more sophisticated signals (regime shift, momentum, sector) layer on top later without breaking anything.

---

## What lands in skill 43 §4 — edge cases worth calling out now

- **Positions with cost basis of $0** — should never happen in the Schwab export, but a defensive check protects against a divide-by-zero in the take-profit score.
- **Fresh positions (< 5 trading days)** — trailing stop needs 20-day history; positions with insufficient history skip the trailing-stop signal but still score other signals.
- **Positions with active options overlay** — sell signals include the `call_position_check` boolean flag (decision 2 above).
- **Multiple lots of the same ticker** — averaged via qty-weighted cost basis (same as covered-call scorer, skill 40 §4).
- **Signals against illiquid tickers** — no liquidity gate this phase; the operator judges execution risk before placing the sell.
- **De-dedup — sell alerts must not spam.** Once a SELL fires for NFLX on Monday, don't re-alert on Tuesday if you haven't sold yet. Journal-derived dedup with a 5-day cooldown per (ticker, signal_type). Explicit override: if the trailing-stop signal hits a new max drawdown, break the dedup and re-alert with the deeper number.

---

## Next step

Say the word and I'll draft `skill_43_sell_signals.md` (Phase 0) plus the two scorers (Phase 1) in one commit. That's the "MVP path" — takes ~1 session and you'll have working take-profit + trailing-stop signals showing in the next hourly Telegram digest after you push.

If you want to pivot on scope (short-selling instead, or automated order placement), tell me before I start Phase 0 — the skill doc changes shape depending on which direction you point.
