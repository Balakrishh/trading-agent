# Journal learnings — observation week

Daily notes from the trading agent's paper run (backlog §0 in `docs/plans/agent_backlog.md`). One dated section per trading day, written after the 16:15 ET daily reviewer. Friday adds "Week 1 results".

---

## 2026-09-30 (Wed)

**Cycles:** 393, 09:25–16:05 ET, 0 errors, 0 silenced alerts, clean after-hours shutdown. Cadence ≈ one cycle per 60–75 s (backlog §4 tracks correcting the "every 5 minutes" docs).

**Trades**
- Opened: **VZ $43 cash-secured put, 2026-10-23**, 1 contract — the Wheel's first live paper trade. Filled at **$0.21** (the bid) on the third attempt at 10:15; attempts at 0.28 and 0.24/0.25 went unfilled and were cancelled (twice: 10:00 and 10:14 runs).
- Closed: none. Realized P&L: $0.00.
- Open at close: SPY iron condor (−$72 at 16:04, agent mid), VZ put (−$4). XLE July condor still `expired_unrecorded`.

**Top reject reasons — verdict**

| Count | Reason | Verdict |
|---|---|---|
| 1,869 | No positive-EV spread across DTE × Δ × width | Correct: oversold, low-premium tape; nothing clears the EV / 55 % POP bar. |
| 312 | Existing open position or pending order | Correct: SPY already holds the condor. |
| 312 | Sideways, RSI 31.1 outside [35, 65) → no condor | Correct: momentum still strong. |
| 312 + 198 | Bearish, RSI 26.6 / 27.7 ≤ 30 → no bear call | Correct: too late in the down-move. |

No reject reason looked like a bug today. The agent stood aside in an extended sell-off, which is the designed behaviour and the case the planned market risk-state overlay (backlog §1) would formalise.

**Exit signals:** none fired all day (0 EXIT SIGNAL lines). Nothing to judge as noise.

**Slippage / fills**
- VZ: screen showed 0.21/0.35 (mid 0.28) → filled at 0.21, i.e. the full half-spread (7¢ = 25 % of the mid credit) conceded. Alpaca paper only filled at the bid.
- VZ proposal estimate was 0.26 vs fill 0.21; the monitor initially managed it on 0.26 (fixed same day, see below).

**SPY iron condor (exp 2026-10-23, 16 × $1 wings, 752P / 780C, credit 0.49)**
- Agent mid P&L ranged about −$136 to −$64 during the day; −$72 at the 16:04 check. HOLD throughout.
- Stop loss −$408, profit target +$392, SPY roughly 1.5–2 % from each short strike. Structurally negative EV (Δ-sum ≈ 0.59); close on a bounce toward break-even.

**Errors / alerts:** none. The 16:15 daily reviewer ran but its LLM step returned empty (Ollama not running) — stats only.

**Fixed and pushed today** (all tested, CI green)
1. Wheel bid/ask liquidity gate (pre-market BMY/CSCO quotes had ranked as 20 %+ yields).
2. Final sell-to-open attempt at the bid, width-gated.
3. Trade plan rewritten with the actual fill credit.
4. Unfilled order runs invalidated so they can't shadow the filled run.
Also: MCP server had been running yesterday's code until an explicit `/mcp` reconnect — a `claude --continue` restart does not reload it.

**Process issues found (backlog §4)**
- `/triage` priced legs from Alpaca's indicative feed (−$392, "close now") while the agent's Schwab mid showed −$112 and HOLD.
- `wheel_screen` treats a JSON-array string watchlist as one ticker.

**Question for Friday:** On paper, Wheel entries only filled at the bid (25 % of the credit given up on VZ). Will the 50 % buyback (≈ $0.10) only fill at the ask too — and if so, should Wheel targets and the screen's yield be computed from the bid (sell) / ask (buy) instead of the mid, so the expected income matches what paper actually fills?

---

## 2026-10-01 (Thu)

**Cycles:** 387, 09:25–16:05 ET, 0 errors, 0 silenced alerts, clean shutdown (16:05).

**Trades**
- Opened: none.
- Closed: **SPY iron condor (exp 2026-10-23) — realized −$176.** Exit `strike_proximity` at 11:03:52: SPY $759.48, 0.99 % above the $752 short put (immediate signal, no debounce). Defensive roll found no valid candidate → normal close. Single 4-leg order: improved attempt $0.55 debit unfilled/cancelled, natural **$0.59 filled, all 16 contracts**. Entry credit was $0.48 → (0.48 − 0.59) × 100 × 16 = −$176.
- The journal first recorded **−$64** (mid mark at the exit signal). Fixed the same day (close P&L now from the actual fill, `pl_source`), and this row was corrected at 16:09 before the 16:15 review; the daily review shows −$176.
- Still open: **VZ $43 CSP** (fill 0.21), agent mid P&L −$2 to −$17.50 during the day, −$13.50 at 16:04 — HOLD. XLE July condor still `expired_unrecorded`.

**Top reject reasons — verdict**

| Count | Reason | Verdict |
|---|---|---|
| 1,815 | No positive-EV spread across DTE × Δ × width | Correct by the model, but this is the "only sells premium / zero-edge POP" limit — see backlog §6. |
| 303 + 303 | RSI 24.5 / 24.8 ≤ 30 → no bear call (XLF, TLT) | Correct; the missing counterpart is the §6.4 bounce bull put. |
| 298 | Sideways, RSI 29.7 → no condor (IWM) | Correct. |
| 76 | Existing position | Correct (SPY until 11:04, then VZ). |

No bugs in the reject reasons.

**Exit signals:** one (SPY `strike_proximity`). Not noise: SPY fell from ~$763.7 (10 AM) to $759.5 in an hour and kept the oversold tape; the rule fired at its designed level on a structurally negative-EV position (Δ-sum ≈ 0.59). Outcome −$176 vs $832 max loss.

**Slippage / fills**
- SPY close: mid mark ≈ $0.52 debit at the signal → filled $0.59 (natural). $0.07 × 1,600 = **$112 of slippage**, more than half of the loss.
- Paper fill pattern confirmed for the second day: improved/mid prices never fill; only natural (bid for sells, ask for buys) does.

**Errors / alerts:** none. Daily reviewer still stats-only (Ollama down).

**Also today**
- GitHub CI green again (first time since June 3): skill freshness in UTC, traceability, missing fastapi/uvicorn/httpx2, stale cache tests. Clean-env suite: 1,411 passed, 0 failed — the "23 pre-existing failures" were local-environment only.
- Backlog §6 "Full playbook" added (debit spreads, bounce bull puts, calendars, regime → playbook table, per-playbook scorecard, fill-realistic pricing).

**SPY iron condor status:** closed 2026-10-01 11:04 ET, −$176 realized. The last position from the $1-wing bug is gone.

**Question for Friday:** Two of two paper exits/entries filled only at natural, and on SPY the slippage ($112) exceeded the move-driven loss. Should backlog 6.1 (price EV, credit floors and exit targets at natural) go first — and should the 50 % profit target be re-expressed as "50 % of credit *after* the expected natural-price buyback" so winners don't give it back in slippage?

---

## 2026-10-02 (Fri)

**Cycles:** 389, 09:25–16:05 ET, 0 errors, 0 silenced alerts, clean shutdown.

**Trades:** none opened, none closed; realized $0. Open: **VZ $43 CSP** (fill $0.21) — agent mid P&L −$1 at 16:04, HOLD; expires 10/23. XLE July condor still `expired_unrecorded`.

**Top reject reasons — verdict**

| Count | Reason | Verdict |
|---|---|---|
| 1,808 | No positive-EV spread across DTE × Δ × width | Correct by the model; structural (backlog §6). |
| 303 | Sideways, RSI 32.9 → no condor (IWM, IV Rank 0) | Correct. |
| 303 + 300 | RSI 24.5 / 25.4 ≤ 30 → no bear call (TLT, XLF) | Correct; §6.4 bounce bull put would be the counter-trade. |
| 10 | "(no reason recorded)" | Minor logging gap: not `rejected` rows with empty notes nor empty journal rows — likely a skip type whose reason is in a field `reject_reasons_today` doesn't read. Investigate with §4 hygiene. |

**Exit signals:** none.

**Slippage / fills:** no fills today.

**Errors / alerts:** none. Daily reviewer still stats-only (Ollama down).

**SPY iron condor:** closed Thu 2026-10-01 11:04 ET, −$176 (see Thu and Week 1).

**Regimes at the close:** SPY sideways (RSI 49, IVR 31), QQQ bullish (RSI 62, IVR 17), IWM sideways (RSI 33, IVR 0), XLF bearish (RSI 25), TLT bearish (RSI 25, IVR 69). Same low-IV, partly oversold tape as all week.

**Lesson for next week:** a whole week in one regime (low IV, oversold pockets) produced zero spread entries — the agent's activity is set by its playbook coverage, not by tuning thresholds. Measure the effect of §6.1–6.3 by trades opened per regime, not just by P&L.

---

## Week 1 results (Mon 2026-09-28 – Fri 2026-10-02)

*Written Friday 16:47 ET from the journal, the four daily reviews and the agent logs. The `journal-analyst` subagent could not start (its `tools:` list uses bare names; see backlog §4), so the same read-only MCP tools were used directly.*

**Headline:** 2 entries, 1 exit, realized **−$176** on the $30k paper account (−0.59 %). The loss came from the $1-wing condor opened before the width fix; the new exit and close machinery worked as designed. One week and two entries prove nothing about edge — this week measured *behaviour*, not *profitability*.

### §0 questions

| Question | Answer |
|---|---|
| Trades opened / closed, win rate, realized P&L | Opened 2: SPY iron condor (Tue 09:31, $0.48 fill, 16 × $1 wings — pre-fix bug), VZ $43 cash-secured put (Wed 10:15, $0.21 fill). Closed 1: SPY, −$176 (`strike_proximity`). Win rate 0/1 — not meaningful. VZ open at −$1 (Fri 16:04), expires 10/23. |
| Top 5 reject reasons — correct or bug? | 8,433 "no positive-EV candidate" — correct by the model, but it is the structural limit (sell-premium only, delta-as-POP = zero edge; backlog §6). 3,573 RSI-gate skips — correct (oversold XLF/TLT, momentum in IWM). 676 existing position — correct. 16 condors blocked by the new Δ-sum floor (C/W 0.295 vs Δ-sum 0.581) — correct, the fix working. 10 rows "(no reason recorded)" on Friday — minor logging gap. |
| Slippage vs mid | VZ entry: mid ≈ $0.28 → fill $0.21 (25 % of the credit). SPY exit: mark ≈ $0.52 → fill $0.59 = **$112** on 16 contracts, more than half the loss. **Paper filled only at natural every time** (2 of 2 entries, 1 of 1 exits; improved/mid attempts never filled). |
| Exits that fired on noise | None. The one exit (SPY, 0.99 % from the short put) was at its designed level on a negative-EV position. |
| Wheel screen | The earnings gate worked (moved BMY/VZ/KO/CSCO to pre-earnings expirations; skipped PEP/WFC/BAC/USB/T). In-hours, the new bid/ask gate left only VZ $43 and KO as tradeable — high screen yields were wide-quote artefacts. |
| SPY iron condor | Closed Thu 11:04 ET, −$176 (entry $0.48 credit, exit $0.59 debit, 16 contracts) vs $832 max loss. Journal corrected from −$64 (mark) to −$176 (fill). |
| Cycles per day / missed days | Mon 0 (supervisor installed Tue 09:23), Tue 293, Wed 312, Thu 303, Fri 303. No missed day after install; cadence ≈ 75 s, not 5 min. 0 errors all week. |

### Bugs found and fixed this week (all pushed, CI green)
$1-wing condors (truncated Schwab chain + no width guard); mid-price valuation and single-order closes; condor Δ-sum floor; multi-day reporting; data-server / MCP key source; Wheel liquidity gate; bid fallback attempt; actual fill credit on entries; unfilled runs shadowing fills; realized P&L from close fills; CI red since June (freshness in UTC, traceability, missing deps, stale tests).

### $50k annual return — re-estimated from this week
Previous estimate (assumptions): +$3k–$5k a year (6–10 %), from ~1–3 spreads a week plus the Wheel.

What the week actually showed:
- **Spreads: 0 opened under current rules** in a low-IV (IV Rank 0–36 except TLT), oversold market. At this pace their contribution is ≈ $0, not the assumed 1–3 trades a week.
- **Wheel:** ≈ 1 position at a time; VZ pays $21 on $4,300 collateral for 23 days ≈ **8 % annualized at the bid** (the screen's mid yield was 10 %).
- **Fills:** natural only → every mid-based yield or EV is ~20–25 % optimistic on paper.

Re-estimate for $50k if the market stays like this week and rules stay as they are:
- Wheel: ~$20k deployed × 8 % × ~70 % utilisation ≈ **+$1,100 / yr**
- Spreads: ≈ **$0** (no trades in this regime)
- Cash (~$25k): **+$1,000 / yr** in T-bills at a real broker; $0 on paper
- **Total ≈ +$1k to +$2k a year (2–4 %)**, well below the earlier 6–10 % — the earlier figure assumed spread activity that this regime does not produce. A correction scenario is unchanged (−$5k to −$8k without the risk overlay).

**How little this proves:** 2 entries and 1 exit (a bug trade). A credible estimate needs ~100 closed trades across regimes; this week only established trade frequency in one regime and the paper fill reality.

### Recommendation for next build
1. **Quick hygiene first (≈ 1 hour):** fix the three subagents' `tools:` names (the weekly review's own analyst could not run) and the after-hours `test_after_hours_shutdown.py` SIGKILL (CI risk on evening pushes).
2. **Then backlog §6.1 — fill-realistic pricing.** The strongest evidence of the week: paper filled only at natural (3/3), and slippage was the largest part of the only loss. Every EV, credit floor, yield and profit target is currently computed at mid and is therefore optimistic. Small change; makes every later decision honest.
3. **Then §6.2 regime → playbook table**, which subsumes §1's market risk-state overlay. The week's reject mix (8,433 no-EV in a low-IV tape) shows the agent has no tool for that regime; 6.2 is the foundation that 6.3 (debit spreads) builds on.

---

## 2026-10-05 (Mon) — first live day of the full playbook

**Cycles:** 313 agent runs (366 cycle-minutes), first at 09:44 ET — the supervisor overslept the 09:25 start because macOS sleep stretched its 12 h wait (fixed: wall-clock sleep). 0 errors, 0 silenced alerts, clean 16:05 shutdown. Daily reviewer ran at 16:15 but its LLM step returned empty (Ollama down) — stats only, no proposal staged.

**Account:** $29,765 → $29,632 (−$133, all unrealized). Realized $0.

**Trades:** 4 opened 13:29–13:37 ET (first trades from the new debit / calendar playbooks), 0 closed.

| Ticker | Structure | Size | Fill (debit) | Mid at entry | Paid over mid | Mid P&L at close |
|---|---|---|---|---|---|---|
| SPY | Calendar 774 call, Oct 30 / Nov 20 | 1 | 5.79 | 5.72 | $7 | −$2.50 |
| QQQ | Call debit 756/772, Nov 6 | 1 | 7.91 | 7.81 | $10.50 | +$8 |
| IWM | Put debit 283/278, Nov 6 | 4 | 1.92 | 1.85 | $28 | −$44 |
| GLD | Put debit 379/369, Nov 6 | 2 | 4.25 | 4.05 | $40 (4.9 %) | −$70 |

Still open from before: VZ $43 CSP (Oct 23), +$8.

**Top reject reasons — verdict**

| Count | Reason | Verdict |
|---|---|---|
| 1,219 | No positive-EV credit spread | Correct; the debit / calendar fallback now acts on these. |
| 933 | Total-risk cap (10 %) | Correct, but the budget was spent in 8 minutes and the agent was idle the last 2.5 h. |
| 593 | Existing position / pending order | Correct. |
| 186 | XLF RSI 23.5 → no bear call | Correct (oversold rule). |
| 112 | IWM sideways, RSI 25.9 → no condor | Correct. |

**Exit signals:** IWM put debit voted `regime_shift` 2 of 3 (13:31, 13:32) when IWM flickered bearish → sideways one cycle after filling — **noise**. Fixed before the third vote: debit spreads now regime-exit only on a trend reversal. Nothing closed.

**Slippage:** all 4 filled at the planned natural price (fill reconciler 4/4); vs mid they cost $86 in total (1.2–4.9 % of the debit), which is most of today's −$100 mid P&L.

**Errors / alerts:** none from the agent. Fixed during the day (all merged): sector cap ignored in-cycle submissions (SPY + QQQ + IWM opened together), no total-risk cap, the IWM regime-flicker exit, a CI crash from a mock journal path. CI on the last pushes could not run — GitHub Actions incident (runners not acquired). XLE July condor reconciled as P&L unknown (the MCP server still lists it until `/mcp` reconnect).

**SPY iron condor (exp 2026-10-23):** closed Thu 2026-10-01 at −$176 (strike proximity, $752 short strike). Nothing left to track.

**Question for Friday:** debit entries go straight to the natural price and paid $86 over mid today (GLD 4.9 %). Would a mid → halfway → natural ladder, like the Wheel's, still fill on paper and recover most of that? Compare this week's debit fills against their mids before deciding.

---

## 2026-10-06 (Tue) — entry confirmation and the 20 % cap go live

**Cycles:** 644 supervisor iterations since Monday's start (387 cycle-minutes today), 0 errors, 0 silenced alerts, clean 16:05 shutdown. Entry confirmation, entry timing (shadow) and trailing profit (shadow) went live at 10:52 ET; the total-risk cap went from 10 % to 20 % at 10:56. Daily reviewer ran at 16:15 — LLM step empty again (Ollama not running), no proposal staged.

**Account:** $29,632 → $29,651 (+$19). Realized $0.

**Trades:** 1 opened, 0 closed.

| Ticker | Structure | Size | Fill (debit) | Mid at entry | Paid over mid | Mid P&L at close |
|---|---|---|---|---|---|---|
| AMZN | Call debit 255/265, Nov 6 | 1 | 4.60 | 4.40 | $20 (4.5 %) | −$12.50 |

Entry path: `entry_confirming (1/3)` at 13:35, reset to 1/3 at 13:42 (the streak broke), 2/3 at 13:44, submitted 13:45 — confirmation behaved as designed. Shadow entry timing resolved at 13:51 with $0 improvement (the price never beat the confirmation price within 8 cycles).

Open at the close (mid P&L): IWM +$78, QQQ +$64, SPY calendar +$4.50, VZ CSP +$2, AMZN −$12.50, GLD −$175 (stop at −$425; hold per the morning analysis). Open defined risk $3,448 of a $5,930 budget.

**Top reject reasons — verdict**

| Count | Reason | Verdict |
|---|---|---|
| 1,320 | Existing position / pending order (incl. Broad Market sector full, GLD ladder same-expiry) | Correct. |
| 664 | Total-risk cap | Correct — all before the 10:56 cap change. |
| 432 | Entry rate limit (1 per hour) | Correct by design — every ticker was held for the hour after AMZN filled (≈ 48 cycles × 9 tickers). |
| 390 + 250 | No positive-EV credit spread; debit fallback also not acceptable | Correct. |

**Exit signals:** SPY calendar voted `regime_shift` once at 10:00 (SPY classified mean_reversion, a 3-σ band touch) — reset next cycle; noise, absorbed by the debounce. Nothing closed. No trailing position armed (none reached its target).

**Slippage:** AMZN paid $0.20/sh over mid (4.5 %), the same size as Monday's entries (1.2–4.9 %). The fill equalled the natural-price estimate (fill reconciler).

**Errors / alerts:** none. Shadow POP log: 152 rows today (e.g. GLD put debit: delta POP 0.43 vs realized-vol POP 0.46). MCP still lists the XLE July condor until a `/mcp` reconnect loads the reconciliation code.

**SPY iron condor (exp 2026-10-23):** closed Thu 2026-10-01 at −$176 (strike proximity). Nothing to track.

**Question for Friday:** with the cap at 20 % there is room for ~4 more trades, but the 1-entry-per-hour limit held all 9 open tickers for an hour after the single AMZN entry (432 skips) and only one trade opened all day. Should the limit count per sector or per direction instead of globally, or rise to 2 per hour — and does the week's entry flow justify it?

