# Runbook 03 — Zombie Position Recovery

> **Trigger:** You see `close_failed` rows in the journal, or the Close Failures Today panel shows a 🚨 cooldown banner, or the broker has a position whose legs don't add up to a defined-risk spread.
> **Time required:** 5-15 minutes per ticker.
> **What you'll know after:** Which legs are stuck, whether your cooldown is engaged, and exactly what to click in Alpaca's UI to flat the position.

---

## 1. When to use this

A "zombie" is a partial-fill broken spread. The position was originally an Iron Condor or Vertical, but during a close attempt some legs were accepted by Alpaca while others were rejected. The result is a position the agent can't reason about — it's no longer the defined-risk shape that was originally opened.

You arrive here from:

- **Runbook 01** flagged `close_failed` rows in the daily review.
- The **Close Failures Today** dashboard panel auto-expanded with a 🚨 cooldown banner.
- You see "X has open spread or pending order" repeating in the log but the dashboard says no positions for X.
- The position's max-loss line in the position monitor doesn't match the original spread's max-loss.

---

## 2. What you need first

- Pi accessible via SSH.
- Alpaca paper UI logged in.
- `pi-diagnostics/` synced to your Mac (per runbook 01 §2).
- ~15 minutes of focused attention. Don't multitask through this — closing the wrong leg can leave you with naked options.

---

## 3. Step-by-step diagnostic

### Step 3.1 — Identify the affected ticker(s)

```bash
python3 << 'PY'
import json
from collections import Counter, defaultdict
path = '/Users/<you>/pi-diagnostics/trade_journal/signals_live.jsonl'
rows = [json.loads(l) for l in open(path) if l.strip()]
# Last 24 hours of close_failed
recent = [r for r in rows[-2000:] if r.get('action') == 'close_failed']
by_ticker = defaultdict(list)
for r in recent:
    by_ticker[r.get('ticker')].append(r)
for tk, group in sorted(by_ticker.items()):
    last = group[-1]
    rs = last.get('raw_signal') or {}
    print(f"{tk}: {len(group)} close_failed rows")
    print(f"   latest:    {last['timestamp'][:19]}")
    print(f"   streak:    {rs.get('partial_close_streak', '?')}/{rs.get('partial_close_threshold', '?')}")
    print(f"   cooldown:  {rs.get('close_cooldown_until', 'not engaged')}")
    print(f"   exit sig:  {rs.get('exit_signal','?')}")
    print(f"   reason:    {(rs.get('exit_reason') or '')[:80]}")
PY
```

Each ticker listed needs cleanup. The ones with `cooldown: not engaged` are early-streak situations that haven't yet hit the 3-strike lockout — those have a chance to self-resolve. The ones with a cooldown ISO timestamp are confirmed zombies.

### Step 3.2 — Identify the broken legs

For each affected ticker, find which legs are still open at the broker vs which have closed:

```bash
python3 << 'PY'
import json
path = '/Users/<you>/pi-diagnostics/trade_journal/signals_live.jsonl'
rows = [json.loads(l) for l in open(path) if l.strip()]

TICKER = 'GLD'  # set to your zombie

# Walk back through close_failed events for this ticker, gather all legs ever seen
last_close_failed = None
for r in reversed(rows):
    if r.get('ticker') == TICKER and r.get('action') == 'close_failed':
        last_close_failed = r
        break

if last_close_failed:
    rs = last_close_failed.get('raw_signal') or {}
    legs = rs.get('leg_close_results', [])
    print(f"Latest close attempt: {last_close_failed['timestamp'][:19]}")
    print(f"Per-leg results:")
    for leg in legs:
        sym = leg.get('symbol', '?')
        status = leg.get('status', '?')
        marker = "✓ closed" if status == "closed" else f"✗ {status}"
        print(f"  {sym:30s}  {marker}")
PY
```

The legs marked `closed` are GONE from the broker. The legs marked anything else (`error`, `rejected`, etc.) are **still open at Alpaca**. Those are what you need to manually close.

### Step 3.3 — Cross-check with the actual broker state

Don't trust the journal blindly. Verify what's actually at the broker:

- Open Alpaca paper UI → Activity → Positions.
- Search for the ticker symbol.
- Compare the listed positions to step 3.2's "still open" legs.

If the broker shows MORE legs than step 3.2 suggested → there are positions the journal doesn't know about (probably from a much earlier session whose history was rotated out). Treat the broker as authoritative.

If the broker shows FEWER legs → some closed asynchronously after the journal was last written. Re-sync `pi-diagnostics/` and re-run step 3.2.

### Step 3.4 — Check what failed and WHY

The most recent close_failed row's notes tell you what Alpaca rejected:

```bash
python3 << 'PY'
import json
path = '/Users/<you>/pi-diagnostics/trade_journal/signals_live.jsonl'
rows = [json.loads(l) for l in open(path) if l.strip()]
TICKER = 'GLD'  # set
last = None
for r in reversed(rows):
    if r.get('ticker') == TICKER and r.get('action') == 'close_failed':
        last = r
        break
if last:
    print(f"Notes: {last.get('notes')}")
    print(f"Failed legs: {last.get('raw_signal', {}).get('leg_close_results')}")
PY
```

Also grep the agent log for the specific Alpaca error codes:

```bash
grep -E "Failed to close position.*$TICKER" pi-diagnostics/logs/trading_agent.log | tail -10
```

The error codes you'll see, decoded:

| Code | Meaning | Operator action |
|---|---|---|
| `40310000` "account not eligible to trade uncovered option contracts" | You tried to close the long leg first → remaining short leg would be naked. Close shorts FIRST. | When manually closing, close ALL short legs before any long legs. |
| `40310000` "insufficient options buying power" | A cash-secured-put close requires more BP than you have. Sub-$25K accounts hit this on big-strike puts (e.g. GLD strikes near 415-420). | Close OTHER positions first to free BP, then return to this one. |
| `40310100` "trade denied due to pattern day trading protection" | Sub-$25K account, you've already used your 3 day-trade quota in the last 5 days. | Either wait for the 5-day window to roll, or accept that this position can't auto-close until expiry. |
| `40310010` "stock too volatile" | Rare; usually means a halt or near-halt. | Wait until the halt clears, retry manually. |

---

## 4. Decision tree

```
Step 3.3 result?
├── Broker shows zero legs → not a zombie, journal was stale.  No action.
├── Broker shows broken-shape position (some legs but not all) → continue to §5 (manual cleanup).
└── Broker shows the full original spread → the close never happened (probably permanent reject).
    Manually issue a close at Alpaca for the spread. Verify next cycle picks up the close
    by looking for action="closed" in the journal.

Step 3.4 showed "uncovered options" error?
└── In §5, close SHORT legs FIRST, then LONGS.  Closing in the other order will keep reproducing this error.

Step 3.4 showed "insufficient buying power" error?
└── In §5, you may need to close another (smaller) position first to free BP, then come back here.

Step 3.4 showed "PDT protection" error?
├── Account < $25K and this ticker was opened today → can't be auto-closed by agent;
│    your only option is to wait for tomorrow OR for expiry.
└── Account < $25K and this ticker was opened earlier → same restriction may or may not apply
     depending on day-trade count history.  Try a manual close at Alpaca; if it errors with
     the same code, you're stuck waiting.
```

---

## 5. Remediation — manual close at Alpaca

For each leg listed as still open in step 3.2:

1. Open Alpaca paper UI → Activity → Positions.
2. Find the specific contract symbol (e.g. `GLD260605C00441000`).
3. Click the contract row → "Close Position" button.
4. **Order type:** Market (faster; the position is already a zombie — slippage doesn't matter).
5. **CRITICAL — close ORDER:** close ALL **short** legs (qty < 0) before ANY **long** legs (qty > 0). The qty column in Alpaca's UI tells you which is which.

   - Why: closing a long-protective-leg first leaves the matching short uncovered → Alpaca rejects subsequent closes with `40310000`. You can get into a stuck state where you can't close anything.
   - The agent's executor sorts legs by qty ascending (shorts first) for this same reason — see `executor.py:close_spread`.

6. After each close, refresh the Positions page. Confirm the leg disappears.
7. When all legs are flat, the zombie is resolved.

If a leg can't be closed manually (PDT or BP error from the broker UI too), you have two options:

- **Let it ride to expiration.** If the position is more than ~5 DTE OUT-OF-MONEY, low risk; theta decays it toward zero. If ITM or near, you'll get assigned at expiration. Calculate the worst-case loss and decide.
- **Free buying power.** Close other (smaller) positions first to make room. Come back and retry.

---

## 6. Verification

After manual cleanup:

### 6.1 — Check the broker

Alpaca → Activity → Positions. Search for the ticker. Expect zero rows.

### 6.2 — Check the cooldown clears on next agent cycle

Wait 5-10 minutes (one or two cycles), then re-pull `pi-diagnostics/` and:

```bash
python3 << 'PY'
import json
path = '/Users/<you>/pi-diagnostics/trade_journal/signals_live.jsonl'
rows = [json.loads(l) for l in open(path) if l.strip()]
TICKER = 'GLD'
# Look for any close_failed events AFTER the cleanup time
# (replace the timestamp string with whatever you finished cleanup at)
CUTOFF = '2026-05-13T17:30:00'
recent = [r for r in rows
          if r.get('ticker') == TICKER
          and r.get('timestamp','') > CUTOFF]
actions = [r.get('action') for r in recent]
print(f"Actions for {TICKER} since cleanup at {CUTOFF}:")
print(actions[:20])
# Should NOT see any further 'close_failed' rows — the position is gone
PY
```

You should see ZERO further `close_failed` rows. If new ones appear, the cleanup didn't actually clear everything — go back to step 3.3.

### 6.3 — Check the dashboard

The Close Failures Today panel should still show the historical `close_failed` rows (those don't disappear from the journal), but the count of "tickers in cooldown" should drop to zero on the next dashboard refresh.

---

## 7. Prevention

The structural fixes for zombie creation went in over 2026-05-06 and 2026-05-13:

- **Leg-close ordering** (executor.py): shorts before longs, prevents the "uncovered options" cascade.
- **`client_order_id` UUID** on order open: prevents accidental duplicate spreads from being opened in the first place (a duplicate is the most common precursor to a zombie because it doubles BP needs at close time).
- **Per-ticker position cap `MAX_POSITIONS_PER_TICKER = 1`**: blocks stacking. With one spread per ticker, the close-time BP requirement matches the open-time BP allocation.
- **Journal-derived cooldown**: ensures 3 failed close attempts park the ticker for 60 minutes instead of looping forever.

The remaining failure mode you can't fully prevent in code is the **PDT lockout on sub-$25K accounts** — that's a broker-side rule. Mitigations:

- Avoid opening + closing the same ticker the same day (the agent's `_tickers_opened_today` suppresses REGIME_SHIFT exits for this reason, but the suppression doesn't fire for STRIKE_PROXIMITY or HARD_STOP because those are real-risk exits worth the PDT hit).
- Keep equity above $25K if possible. Below the threshold, you're rationed to 3 day-trades per 5-day rolling window.

---

## 8. LLM hand-off template

To get a fresh LLM to walk you through this:

```
I have a zombie position on ticker XLF that needs manual cleanup. Please help me work through docs/runbooks/03_zombie_position_recovery.md against the attached data.

Attached:
- pi-diagnostics/trade_journal/signals_live.jsonl
- pi-diagnostics/logs/trading_agent.log

Specifically:
1. Run §3.1 to identify all affected tickers.
2. For XLF specifically, run §3.2 to identify the still-open legs.
3. Run §3.4 to decode the Alpaca error codes.
4. Tell me the exact order I should close the legs in (short legs FIRST per the §5 critical rule).

Do NOT propose code changes — this is broker UI cleanup work, not code work.
End with the verification queries from §6.2 I should run after cleanup.
```

The LLM should produce a numbered list of specific contract symbols, in close order, with the broker error context. You then drive Alpaca's UI manually.

---

## 9. Related runbooks + skills

- **Runbook 01** — daily close review. Catches zombies on the day they're created.
- **Runbook 04** — full reference for Alpaca/Schwab error codes (stub for now; expand as new codes appear).
- **Skill 17** — close-failure action + cooldown design (the "why" behind this runbook's diagnostics).
- **Skill 18** — order-submission idempotency (preventing the duplicate spread that becomes tomorrow's zombie).
- **Skill 19** — journal schema, including `close_failed` field semantics.

---

*Last verified against repo HEAD on 2026-05-13.*
