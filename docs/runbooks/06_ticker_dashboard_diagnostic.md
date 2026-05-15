# Runbook 06 — Per-Ticker Dashboard Data Diagnostic

> **Trigger:** Open Positions panel shows numbers for a specific ticker that don't match what you'd expect — credit doesn't look right, day-1 P&L is alarming, strike proximity feels off, or two rows for the same ticker that should be one.
> **Time required:** 10–20 minutes if the pi-diagnostics tarball is fresh, +5 min for a broker-UI cross-check.
> **What you'll know after:** Whether the issue is (a) a rendering / attribution bug in `group_into_spreads`, (b) a real broker-side problem, (c) routine bid-ask drag misread as a loss, (d) a positioning risk that warrants manual close, or (e) a stale display.

---

## 1. When to use this

Use this when **one ticker** on the Open Positions panel is the focus of concern — not when ALL tickers look wrong (that's runbook 02 or a Streamlit-level failure). Common entry symptoms:

- The Credit / Max Loss / % Profit column for the ticker disagrees with what the journal recorded at submission.
- Day-1 P&L is showing a 40-60% loss on a freshly-opened spread.
- Two dashboard rows for the same ticker that should be one (Iron Condor split into Bull-Put + Bear-Call halves).
- The "Why" column shows a regime that no longer matches today's market.
- The Expiry column is wrong, or `Entered (ET)` is missing/yesterday.
- The Entry Justification expander shows thesis/checks but the numbers in the strip caption don't match the table row.

If your symptom is "no positions show up at all" → see runbook 02 (zero trades).
If your symptom is "I have a partial-fill broken spread" → see runbook 03 (zombie recovery).

---

## 2. What you need first

- Pi accessible via SSH (`balakrishh@myrasberrypi.local`).
- A fresh pi-diagnostics pull (less than 30 minutes old). To refresh:

```bash
ssh balakrishh@myrasberrypi.local \
    "tar czf - -C /home/balakrishh/Documents/trading-agent \
        trade_journal logs AGENT_LOG trade_plans" \
    | tar xzf - -C ~/pi-diagnostics/
```

- The Alpaca paper UI (https://app.alpaca.markets/) open in a tab — you'll cross-check the broker's view against the journal's view.
- The ticker symbol you're investigating, and roughly when the position was opened (from the dashboard's `Entered (ET)` column).

---

## 3. Step-by-step diagnostic

Run these in order. Each step's expected output is documented inline — deviation is the diagnostic signal.

### Step 3.1 — Pull the ticker's lifecycle from the journal

```bash
python3 << 'PY'
import json
TICKER = "GLD"        # ← set to your ticker
DATE   = "2026-05-15" # ← set to today (or whenever the position opened)
rows = [json.loads(l) for l in open(
    '/Users/<you>/pi-diagnostics/trade_journal/signals_live.jsonl')
    if l.strip()]
t = [r for r in rows
     if r.get('ticker') == TICKER and r.get('timestamp','') >= DATE
     and r.get('action') not in ('skipped_existing','skipped_rsi_gate',
                                  'skipped_defense_first','skipped_bias')]
print(f"=== {TICKER} non-skip events on/after {DATE}: {len(t)} ===")
for r in t:
    ts = r.get('timestamp','')[:19]
    print(f"  {ts}  {r.get('action','?'):14s}  "
          f"| {(r.get('notes') or '')[:120]}")
PY
```

**Expected output for a healthy ticker (one IC opened today):**

```
=== GLD non-skip events on/after 2026-05-15: 5 ===
  2026-05-15T16:54:24  rejected       | rejected: Iron Condor, cr=1.85, ratio=0.37
  2026-05-15T16:54:58  rejected       | rejected: Iron Condor, cr=1.90, ratio=0.38
  2026-05-15T16:55:29  submitted      | submitted: Iron Condor, cr=1.95, ratio=0.39
  2026-05-15T16:56:30  skipped_existing
  …
```

**What it tells you:** One `submitted` row is the canonical entry. Multiple `submitted` rows means the dedup gate let two trades through (see runbook 03 — zombie recovery). Zero `submitted` rows means the position you're looking at was inherited from an earlier day; pull the journal further back. `rejected` rows preceding the submission are normal — the chain scanner sweeps strikes until one passes the CW floor.

### Step 3.2 — Pull the actual leg set from `trade_plan_<TICKER>.json`

```bash
python3 << 'PY'
import json
TICKER = "GLD"
DATE   = "2026-05-15"
plan = json.load(open(
    f'/Users/<you>/pi-diagnostics/trade_plans/trade_plan_{TICKER}.json'))
# Find the SUBMITTED entry (valid=True AND approved=True)
hits = [e for e in plan['state_history']
        if e['timestamp'].startswith(DATE)
        and e['trade_plan'].get('valid')
        and (e.get('risk_verdict') or {}).get('approved')]
print(f"=== {TICKER} SUBMITTED plan entries on {DATE}: {len(hits)} ===")
for e in hits:
    tp = e['trade_plan']
    print(f"\nTimestamp: {e['timestamp']}")
    print(f"  strategy:    {tp.get('strategy')}")
    print(f"  net_credit:  {tp.get('net_credit')}")
    print(f"  max_loss:    {tp.get('max_loss')}")
    print(f"  spread_width:{tp.get('spread_width')}")
    print(f"  expiration:  {tp.get('expiration')}")
    print(f"  legs:")
    for L in tp.get('legs', []):
        print(f"    {L['symbol']}  {L['action']:4s}  K={L['strike']}  Δ={L.get('delta')}  bid={L.get('bid')}  ask={L.get('ask')}")
PY
```

**What it tells you:** The CANONICAL credit / max_loss / leg-set the agent intended to submit. This is the ground truth. **Compare against the dashboard's Open Positions row.** If they disagree, you've found the bug.

### Step 3.3 — Look for "shared-legs" rejected plans (the leg-claim bug)

This is the pattern that produced the 2026-05-15 GLD `cr=$1.75 / -60%` display when the actual fill was `cr=$1.95 / -54%`. Skill 28 documents the mechanism.

```bash
python3 << 'PY'
import json
TICKER = "GLD"; DATE = "2026-05-15"
plan = json.load(open(
    f'/Users/<you>/pi-diagnostics/trade_plans/trade_plan_{TICKER}.json'))

# Find the SUBMITTED plan's legs
sub = [e for e in plan['state_history']
       if e['timestamp'].startswith(DATE)
       and e['trade_plan'].get('valid')
       and (e.get('risk_verdict') or {}).get('approved')]
if not sub:
    print("No submitted plan found for the date — skip this step.")
else:
    SUBMITTED_LEGS = {L['symbol'] for L in sub[0]['trade_plan']['legs']}
    SUBMITTED_TS = sub[0]['timestamp']
    print(f"Submitted at {SUBMITTED_TS[:19]} with legs:")
    for s in sorted(SUBMITTED_LEGS): print(f"  {s}")
    print()
    # Find EARLIER plans with overlapping legs
    shared = []
    for e in plan['state_history']:
        if e['timestamp'] >= SUBMITTED_TS: continue
        legs = {L['symbol'] for L in e['trade_plan'].get('legs',[])}
        overlap = legs & SUBMITTED_LEGS
        if overlap:
            rv = e.get('risk_verdict') or {}
            shared.append({
                'ts':       e['timestamp'][:19],
                'valid':    e['trade_plan'].get('valid'),
                'approved': rv.get('approved'),
                'cr':       e['trade_plan'].get('net_credit'),
                'ml':       e['trade_plan'].get('max_loss'),
                'overlap':  f"{len(overlap)}/4",
            })
    print(f"=== Earlier plans with overlapping legs: {len(shared)} ===")
    bad = [s for s in shared if s['valid'] and not s['approved']]
    print(f"    of which valid=True approved=False: {len(bad)}  ← the ones that triggered the GLD bug")
    for s in bad[:5]:
        print(f"  {s['ts']}  cr={s['cr']:>5.2f}  ml={s['ml']:>6.1f}  overlap={s['overlap']}")
PY
```

**What it tells you:**

- `valid=True approved=False overlap=4/4` rows with cr/ml DIFFERENT from the submitted plan → if your Pi runs code older than the 2026-05-15 evening fix, the dashboard is showing one of THESE plans' economics, not the submitted plan's. **Verify by deploying the latest commit (`git pull` + restart Streamlit).**
- If the rows DO match the submitted plan's cr/ml → no rendering bug, dashboard is showing the correct numbers; investigate other causes (bid-ask drag, strike proximity, broker-side issue).
- Zero overlapping rows → not a leg-claim bug. Move on.

### Step 3.4 — Cross-check legs against the broker's actual position

Open https://app.alpaca.markets/, click **Positions**, find the ticker. Confirm:

- The four legs (or two, for a vertical) match the symbols from step 3.2 verbatim.
- Quantities match the journal's `qty`.
- The broker's `Cost Basis` ≈ `net_credit × qty × 100`. If the broker shows a materially different cost basis, the broker filled at a worse price than the journal recorded — investigate the executor's price-haircut logic.

### Step 3.5 — Compute expected bid-ask drag (day-1 P&L sanity check)

If the position opened today and shows a steep day-1 loss, this distinguishes "expected bid-ask drag" from "actual adverse move."

```bash
python3 << 'PY'
import json
TICKER = "GLD"; DATE = "2026-05-15"
plan = json.load(open(
    f'/Users/<you>/pi-diagnostics/trade_plans/trade_plan_{TICKER}.json'))
hits = [e for e in plan['state_history']
        if e['timestamp'].startswith(DATE)
        and e['trade_plan'].get('valid')
        and (e.get('risk_verdict') or {}).get('approved')]
for e in hits:
    legs = e['trade_plan']['legs']
    credit = e['trade_plan']['net_credit']
    # Mark-to-market right at fill: shorts marked at ASK, longs at BID
    cost_to_close = 0.0
    for L in legs:
        if L['action'] == 'sell':  # short → marked at ASK
            cost_to_close += L.get('ask', 0)
        else:  # long → marked at BID
            cost_to_close -= L.get('bid', 0)
    drag = (credit - cost_to_close) * 100
    print(f"{e['timestamp'][:19]}  credit=${credit*100:.0f}  "
          f"day-1 cost_to_close=${cost_to_close*100:.0f}  "
          f"expected day-1 P&L = ${drag:.0f}")
PY
```

**What it tells you:** Compare the `expected day-1 P&L` with what the dashboard shows.

- Within ±15% → It's bid-ask drag. Theta will erase most of it within 1-3 days if the underlying stays in range. **Do not panic-close.**
- Worse than expected by >50% → The underlying has moved against you since fill. Go to step 3.6.
- Better than expected → Fill executed inside the spread (lucky); position is in better shape than the legs implied.

### Step 3.6 — Check strike proximity (real risk)

```bash
grep "\[$TICKER\] Regime" ~/pi-diagnostics/logs/trading_agent.log | tail -3
```

**Example output:**

```
2026-05-15 13:32:01 | INFO | [GLD] Regime → sideways | Price=418.73 SMA50=435.36 …
```

Then compare the current price against the **short** strikes from step 3.2 (`action: sell` legs):

- Distance to short put `(spot − short_put_strike) / spot` → buffer below.
- Distance to short call `(short_call_strike − spot) / spot` → buffer above.

**Healthy buffers (21-DTE IC at Δ-0.30):** 2–3% on each side.
**Warning (need to watch closely):** 1–2%.
**Action (consider closing):** <1% — the short strike is at risk of breach.

### Step 3.7 — Compare entry timestamp UTC ↔ ET

The journal stores UTC, the dashboard now (post-2026-05-15) shows ET. If they're off by anything other than 4–5 hours, the timezone helper is broken — see Skill 28's "Format" notes.

```bash
python3 -c "
import json
TICKER = 'GLD'
rows = [json.loads(l) for l in open('/Users/<you>/pi-diagnostics/trade_journal/signals_live.jsonl') if l.strip()]
sub = [r for r in rows if r.get('ticker')==TICKER and r.get('action')=='submitted']
for r in sub[-3:]:
    print(f\"  journal UTC: {r['timestamp'][:19]}  → dashboard ET should show (UTC − 4h):\")"
```

---

## 4. Decision tree

Based on what step 3 surfaced:

```
Are step 3.1's submitted row's cr/ml = step 3.2's trade_plan cr/ml?
├── No → journal-vs-plan mismatch. RARE. Investigate journal_kb.log_signal write path.
└── Yes → continue:
    │
    Does step 3.2's cr/ml = the dashboard's row?
    ├── No → Step 3.3 shows shared-legs rejected plans (valid=True approved=False)?
    │   ├── Yes → LEG-CLAIM BUG. Deploy latest commit on Pi
    │   │        (the 2026-05-15 GLD fix). Restart Streamlit.
    │   │        Re-verify dashboard rendering.
    │   └── No  → Streamlit cache may be stale. Trigger a Streamlit
    │             rerun (hamburger → "Rerun") or restart the Pi's
    │             Streamlit process. If still wrong, this is a NEW
    │             rendering bug — escalate to draft a new runbook
    │             via prompts.md §9.
    └── Yes → dashboard is correct. Continue:
        │
        Is the alarming-looking number a day-1 P&L loss?
        ├── Yes → step 3.5 shows expected drag matches?
        │   ├── Yes → BID-ASK DRAG. Don't close. Wait 24-72h for
        │   │        theta. Re-evaluate.
        │   └── No  → underlying moved against you. Go to 3.6.
        └── No → Is the alarming number strike proximity?
            ├── Yes → step 3.6 buffer < 1% → CONSIDER CLOSING. The
            │        agent will fire strike-proximity exit on its
            │        own, but if you have PDT restrictions
            │        (sub-$25K account) you may need to override.
            └── No  → underlying issue beyond this runbook's scope.
                     Escalate via prompts.md §9.
```

---

## 5. Remediation

### Case A — Dashboard rendering bug (leg-claim)

1. SSH to Pi, navigate to `/home/balakrishh/Documents/trading-agent`.
2. `git pull` to fetch the latest fix.
3. Verify the fix is present:
   ```bash
   grep -c "risk_verdict" trading_agent/position_monitor.py
   ```
   Should output ≥1.
4. Verify Streamlit also has the envelope-passing fix:
   ```bash
   grep -A2 "trade_plans.append" trading_agent/streamlit/live_monitor.py
   ```
   Should show `trade_plans.append(entry)` (the envelope), NOT `entry["trade_plan"]` (the inner-only form).
5. Restart Streamlit:
   ```bash
   pkill -9 -f 'streamlit run' && sleep 2
   cd /home/balakrishh/Documents/trading-agent
   nohup streamlit run trading_agent/streamlit/app.py \
       > /tmp/streamlit.log 2>&1 &
   ```
6. Reload the dashboard in browser. Verify the ticker's row now shows the correct cr/ml.

### Case B — Bid-ask drag misread as a loss

No action needed. Document in your daily review (runbook 01) that the day-1 mark looked alarming but was within expected drag bounds. If the same ticker shows this pattern repeatedly, consider tightening the chain scanner's liquidity floor (`liquidity_max_spread` in `.env`) to avoid trading thinner-spread strikes on this name.

### Case C — Underlying moved against you

If buffer is still ≥1% on the threatened side: hold. The agent's strike-proximity exit will fire when the buffer hits 0.5%.
If buffer is <1%: manually close at the broker UI. Steps in runbook 03 §5 (Close failures + manual cleanup).
**PDT-restricted accounts:** if you're under $25K and have already used your 3 day-trades this rolling-5-day window, you can't close intraday. The position must ride to expiry; budget for full max loss.

### Case D — Stale Streamlit display (no real bug)

```bash
# Force a Streamlit cache flush via menu OR via shell:
pkill -f 'streamlit run' && sleep 2 && \
nohup streamlit run trading_agent/streamlit/app.py > /tmp/streamlit.log 2>&1 &
```

---

## 6. Verification

After remediation, re-run step 3.2 and confirm dashboard numbers match. Take a screenshot for the daily review.

```bash
# Quick one-liner that re-runs steps 3.1 and 3.2 for the ticker:
python3 - << 'PY'
import json
TICKER, DATE = "GLD", "2026-05-15"
# (same as steps 3.1 + 3.2 — copy from above)
PY
```

If the dashboard NUMBERS match the journal NUMBERS, you're done. Move on.

---

## 7. Prevention

- **Run the daily close review (runbook 01) every trading day.** Step 3.4 of that runbook now includes a "per-ticker reconciliation" pass that catches leg-claim mismatches early.
- **Keep `pi-diagnostics` synced to the Pi at least daily.** A stale tarball makes ticker-specific diagnosis 5x slower.
- **When changing chain scanner output (`net_credit`, `legs`, `max_loss`):** the SDLC step "footer re-stamp" should also bump the date in skill 28 if `group_into_spreads` semantics change.
- **Don't introduce a new caller of `group_into_spreads` that pre-unwraps envelopes.** Skill 28 §4 documents this contract — read it before threading data into `position_monitor`.

---

## 8. LLM hand-off template

If you want to delegate the diagnosis to a fresh LLM session, paste this. The placeholders in `<…>` are the only fields you need to fill.

```
I'm seeing wrong/suspicious data for ticker <TICKER> on the Trading Agent
dashboard. Walk through docs/runbooks/06_ticker_dashboard_diagnostic.md
§3 against my attached pi-diagnostics.

Specifically:
- Step 3.1 (journal lifecycle) — paste me the output
- Step 3.2 (canonical trade plan) — paste me the legs + cr/ml
- Step 3.3 (shared-legs rejected plans) — was the leg-claim bug triggered?
- Step 3.5 (expected bid-ask drag) — does it match the dashboard's P&L?
- Step 3.6 (strike proximity) — what's the buffer on each short leg?

Then use the §4 decision tree to classify the issue. Stop before §5
(remediation) — I want to confirm the classification before any action.

Attached:
- pi-diagnostics/trade_journal/signals_live.jsonl
- pi-diagnostics/trade_plans/trade_plan_<TICKER>.json
- pi-diagnostics/logs/trading_agent.log
- Screenshot of the dashboard's Open Positions row for <TICKER>
```

---

## Worked example — GLD on 2026-05-15

Dashboard showed: `GLD Iron Condor • credit $175 • P&L −$105 • % Profit −60%`.

| Step | Finding |
|---|---|
| 3.1 | One `submitted` row at 16:55:29 UTC (12:55:29 ET) with `cr=1.95 ratio=0.39`. |
| 3.2 | Plan at 16:55:29 with legs `P408/P403/C428/C432`, `net_credit=1.95`, `max_loss=305`, `spread_width=5`. |
| 3.3 | THREE earlier `valid=True approved=False` plans (13:32, 13:33, 14:13) with **identical legs** as the submitted one, different cr/ml (1.75 / 1.80 / 1.95). |
| 3.4 | Alpaca confirmed the four legs match step 3.2. Cost basis = $195. |
| 3.5 | Expected drag = (1.95 − 3.10) × 100 = **−$115**. Dashboard showed −$105 — within expected range. |
| 3.6 | GLD at $418.68; short put $408 (2.55% buffer), short call $428 (2.23% buffer). Healthy. |

**Classification (per §4 decision tree):** Case A — leg-claim rendering bug. The dashboard was attributing the 13:32 rejected plan's `cr=1.75 / ml=325` to today's broker fill because Streamlit's `_fetch_spreads_cached` pre-unwrapped envelopes and silently neutered the `risk_verdict.approved` filter.

**Remediation:** `git pull` on Pi → restart Streamlit → dashboard now correctly shows `cr=$1.95 / ml=$305 / P&L −54%`. Position itself was never wrong — only the display.

**Lesson:** When the dashboard's % Profit is dramatically worse than the journal-derived expected drag, suspect attribution before suspecting the broker.

---

## Cross-references

- **Skill 28** — `group_into_spreads` algorithm + the two attribution bugs (XLF + GLD).
- **Runbook 01** — daily close review (the routine sweep that should catch this).
- **Runbook 03** — zombie position recovery (if the discrepancy is partial-fill not attribution).
- **Runbook 04** — broker API errors (if the broker UI itself disagrees with the journal).
- **`docs/runbooks/prompts.md` §3** — generic "X is acting weird" LLM prompt (this runbook is a more specific version for the ticker-data subset).

---

*Created 2026-05-15 in response to the GLD `cr=$1.75 / −60%` incident.*
