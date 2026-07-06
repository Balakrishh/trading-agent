# Additional Options Strategies — Plan

**Working document — not a skill doc yet.** Each phase becomes its own skill under `docs/skills/45+_*.md` as it lands.

**One-line intent.** Expand the credit-spread agent from three strategies (Bull Put, Bear Call, Iron Condor) into a broader menu that works across different volatility and directional regimes. Butterflies capture range-bound expectations, calendars monetise theta + IV expansion, broken-wing structures let you take directional bets with defined risk. Every addition backtests first; nothing goes live until the backtester says the edge holds.

**Why the current book struggles to trade today.** The 2026-07-02 diagnosis showed the agent correctly skipping trades because market premium isn't clearing the invariant C/W floor at Δ=0.18-0.25. That's a *regime-fit* problem — Bull Put / Bear Call / IC all price identically off the vertical-spread breakeven formula, so when the market underprices verticals, ALL three strategies sit out simultaneously. Adding structures with *different* breakeven math (butterfly = ATM peak, calendar = time-decay slope) gives the agent options when verticals are dead.

---

## Strategy shortlist — ranked by "trade this book"

The universe of possible option strategies is large (I count ~15 canonical ones). The right question is which ones fit **your account** (sub-$25K, PDT-restricted, thin premium environment) and **complement what already ships**. Filtered list:

| # | Strategy | Regime it likes | Complexity | Priority for your book |
|---|---|---|---|---|
| 1 | **Iron Butterfly** | Range-bound, IV-mean-revert | Trivial (reuses IC) | **High — Phase 1** |
| 2 | **Broken-Wing Butterfly** | Directional + defined-risk | Medium (new leg logic) | High — Phase 2 |
| 3 | **Butterfly (debit)** | Precise range-bound, low-IV | Medium | Medium |
| 4 | **Calendar spread** | Front-month IV expanding | Hard (multi-expiration) | Medium — Phase 3 |
| 5 | **Diagonal spread** | Directional + IV expansion | Hard (multi-exp + Δ mgmt) | Medium |
| 6 | **Ratio spread (1×2, 1×3)** | Contained directional | Medium risk | **Skip — poor fit** |
| 7 | **Straddle / Strangle (long)** | Volatility explosion | High risk | **Skip — undefined-risk short side; big theta drag long side** |
| 8 | **Jade Lizard** | Modest bullish + short vol | Requires margin | **Skip — sub-$25K cash constraint** |

Priority driver: strategies that reuse existing infrastructure (leg pricing, C/W floor invariant, journal shapes) ship first. Multi-expiration structures (calendar, diagonal) are architecturally expensive and land after the easy wins prove the abstraction generalises.

---

## Phased build

Each phase is 1-2 sessions unless noted. Every phase includes design doc, scorer, backtester wiring, conformance tests, and skill footer stamp.

### Phase 1 — Iron Butterfly (1 session)

The lowest-risk highest-value addition. Iron Butterfly is structurally identical to Iron Condor — 4 legs, both short strikes at ATM, both long strikes N-points-wide. It slots into `chain_scanner._score_iron_butterfly` alongside the existing IC path with minimal new code.

**Why now.** In low-vol environments, IC credit is often too thin to clear the C/W floor (exactly what today's rejects showed). Iron Butterfly collects roughly 2× the credit of an IC at the same DTE because both short strikes are ATM instead of OTM — but the "profit zone" is narrower (only a single point vs. a range). When IV rank is elevated but the market isn't paying enough on OTM verticals, IB is the trade.

Deliverables:
- `_score_iron_butterfly` in `chain_scanner.py`, reusing the existing 4-leg scaffold from IC.
- `PresetConfig.iron_butterfly_enabled: bool = False` (start disabled, backtest first).
- New reject-reason taxonomy strings: `REJECT_IB_WINGS_TOO_NARROW`, etc.
- Skill 45 documenting the theory (peak-vs-range trade-off), scoring math (POP inverted for narrow-profit-zone strategies), and gates.
- 10-12 conformance tests.
- Backtester's `SyntheticChain` handles IB pricing natively (same leg dictionaries, no new fetch path).

**Architectural decision to make here.** The C/W floor invariant `|Δ_short| × (1 + edge_buffer)` was derived for verticals where breakeven C/W = |Δ|. For Iron Butterfly the equivalent invariant is `|Δ_short_call| + |Δ_short_put| ≈ 1.0` at fair value — a different formula entirely. Options: (a) extend the CI invariant scanner to check IB has its own scaling invariant, OR (b) exempt IB from the credit invariant and rely purely on the EV floor. Recommendation: (b) — the EV/$risked calculation naturally captures the trade-off; the invariant existed for verticals specifically.

### Phase 2 — Broken-Wing Butterfly (1-2 sessions)

Broken-wing takes an Iron Butterfly and moves ONE long wing further out, creating an asymmetric structure that's net-credit (no downside risk on one side) instead of the debit-neutral IB. Powerful when you have a directional view but want defined risk on the "wrong" side.

**Why now, after IB.** The 4-leg scaffold from Phase 1 handles broken-wing with a single change: allow unequal wing widths. That's a preset knob (`bwb_wing_ratio: float = 1.5`) not new infrastructure. The scoring math is a small delta.

Deliverables:
- `_score_broken_wing_butterfly` (call and put variants) in `chain_scanner.py`.
- Signed EV computation — one-sided max loss changes the risk denominator.
- Two new regime → strategy mappings in `strategy.py`: bullish + IB-preferring → Bull BWB; bearish + IB-preferring → Bear BWB.
- Skill 46, 8-10 conformance tests, backtester parity.

### Phase 3 — Debit Butterfly (1 session, optional)

Standard debit butterfly (1 long low, 2 short middle, 1 long high). Pays a small debit, wins on pin-risk at the middle strike. Ships fast because it's structurally simpler than IB.

**Why it's optional.** Debit butterflies have narrow profit zones and require directional precision. In practice, IB and BWB cover most of the same use cases with better risk-reward at your account size. Ship only if backtest shows meaningful edge over the credit-side equivalents.

### Phase 4 — Calendar spreads (2-3 sessions, big architectural lift)

Sell near-dated, buy far-dated at the same strike. Profits from front-month theta decay + IV expansion on the back-month. Structurally different from every existing strategy — requires the chain scanner to fetch and reason about **two different expirations simultaneously**.

**Why the effort.** Calendars have unique edge in specific regimes: pre-earnings (IV expansion expected), post-vol-crush (mean reversion trade), or low-IV environments where verticals are dead. That's exactly the setup showing up in your book right now — VIX zscore near zero, credit spreads not clearing. A calendar can print positive EV where a vertical can't.

**Architectural work required.**
- `ChainScanner.get_chain(ticker, expiration)` needs to handle multi-expiration cache. Currently it caches per (ticker, expiration).
- `SpreadPosition` needs to model legs with different expirations (the `expiration: str` field becomes per-leg).
- The C/W floor invariant is **structurally inapplicable** — calendars price off IV differential, not credit-to-width. Requires either a new invariant or an exemption category in the CI scanner.
- The backtester's synthetic chain must extend to price the second expiration (adds ~200 lines of Black-Scholes-with-vol-differential machinery).
- Position monitor's TP/SL logic needs new anchors — calendars don't have a max-loss in the vertical sense; they have "front-month expires worthless AND back-month retains value" as the primary success condition.

Deliverables:
- Skill 47 covering theory, math, multi-expiration architecture.
- New `_score_calendar_spread` in `decision_engine.py`.
- `SpreadPosition.legs` typing update.
- `ChainScanner` multi-expiration extension.
- Backtester's `synthetic_chain.py` extended with IV term-structure modelling.
- 15+ conformance tests including calendar-specific edge cases (front-month expiry, IV crush events, dividends).

### Phase 5 — Diagonal spread (2 sessions)

Diagonal = calendar + vertical. Sell near-dated OTM, buy far-dated further-OTM at a different strike. Adds a directional bias to the calendar's time-decay play. Ships fast after Phase 4 because the multi-expiration infrastructure is already in place.

Deliverables: skill 48, `_score_diagonal_spread`, backtester wiring, conformance tests.

---

## Cross-cutting architectural decisions

Four questions I want your answer on before Phase 0 kickoff. Each has a sensible default; call out where you want to override.

**1. Backtest-first-or-parallel policy.** Every new strategy needs a backtest showing meaningful edge before it ever trades live. Default: no strategy ships to the live `PresetConfig.enabled_strategies` list until it produces a positive backtested Sharpe on the last 12 months of ETF data. Override only if you want to paper-trade before backtesting (higher risk, faster iteration).

**2. Strategy family gating in `PresetConfig`.** Add new per-strategy toggles (`iron_butterfly_enabled`, `bwb_enabled`, `calendar_enabled`, etc.) OR a single `enabled_strategies: List[str] = ["bull_put", "bear_call", "iron_condor"]` field. Default: single list — easier to hot-reload, easier to enable one strategy per preset, easier to reason about. Individual booleans get out of sync.

**3. Invariant scanner treatment of non-vertical strategies.** The C/W floor invariant `|Δ| × (1 + edge_buffer)` is vertical-specific. Iron Butterfly and calendar both have different scaling laws. Options: (a) extend `scan_invariant_check.py` to know per-strategy invariants (more work, tighter guarantees), or (b) exempt non-vertical strategies from that specific invariant and rely on their own scoring functions for edge (less work, thinner safety net). Default: (a) — the whole point of the invariant scanner is preventing structural drift; exempting strategies erodes that. It's a couple of hours of extension work each time.

**4. Position monitor generalisation.** Skill 44 just fixed contract-count scaling in `_check_exit`. The same code path will need to handle calendar TP/SL, BWB asymmetric max-loss, and butterfly pin-risk. Default: keep the exit-signal enum small (`HARD_STOP`, `STOP_LOSS`, `PROFIT_TARGET`, `STRIKE_PROXIMITY`) but let each strategy pass its own threshold formulas through a new `SpreadPosition.exit_policy: ExitPolicy` field. Explicit; testable; doesn't grow the exit-signal type set unnecessarily.

---

## What each phase costs

| Phase | Strategy | New code | Backtest lift | Skill number | Effort |
|---|---|---|---|---|---|
| 1 | Iron Butterfly | Small (reuses IC) | Small | 45 | 1 session |
| 2 | Broken-Wing Butterfly | Small-medium | Small-medium | 46 | 1-2 sessions |
| 3 | Debit Butterfly | Medium | Medium | (optional) | 1 session |
| 4 | Calendar spread | **Large** (multi-exp architecture) | Large | 47 | 2-3 sessions |
| 5 | Diagonal spread | Small (after Phase 4) | Small | 48 | 2 sessions |

Total for the full menu: **7-9 sessions** across 5 skills. First value ships after Phase 1 (~1 session).

---

## Fastest useful ship — the MVP path

Phase 1 alone gives you Iron Butterfly, which is what today's rejected-nothing-cleared journal actually wants. Ship it, backtest it, and if it produces edge in your current low-vol regime, enable it on live. That's one session, one skill, meaningful trading behavior change.

The rest of the phases can wait until you see whether the credit-spread-plus-Iron-Butterfly combination gives the agent enough regime coverage. If it does, the plan pauses at Phase 2 (BWB) as a nice-to-have. If it doesn't, Phase 4 (calendars) becomes urgent because calendars monetise the exact environment credit spreads can't.

---

## What NOT to do at this account size

Three strategies I strongly recommend against for the current book:

**Ratio spreads (1×2, 1×3).** Undefined-risk on one side unless carefully constructed. The math is subtle; the failure modes are large. Not worth the mental overhead when butterflies cover similar directional-plus-defined-risk territory.

**Long straddles / long strangles.** These are volatility-expansion bets. The theta drag is punishing on a small account, and the win rate is low — you need the underlying to move MORE than the market's implied vol pricing predicts. Reliably profitable only if you have an edge in vol forecasting, which the current system doesn't have.

**Jade Lizard.** Requires the short put to be either cash-secured (ties up a lot of capital) or margin-secured (not available on cash accounts). Sub-$25K + PDT + margin restrictions kill it structurally.

---

## Next step

Say the word and I'll draft `skill_45_iron_butterfly.md` (Phase 1 design doc) plus the `_score_iron_butterfly` scorer in `chain_scanner.py` in one commit. That's the "MVP path" — one session, one commit, and you'll have Iron Butterfly candidates surfacing in the guardrail grid on the next agent cycle (initially still gated behind a `PresetConfig.iron_butterfly_enabled=False` flag until backtest signs off).

If you want a different phase to start first (say, jumping straight to calendars because low-vol is expected to persist), tell me before Phase 0 — the architectural work in each phase is meaningful and hard to reorder mid-session.
