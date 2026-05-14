# Runbook 01 — Daily Close Review

> **Trigger:** End of trading day (after 4:30 PM ET) or first thing the next morning.
> **Time required:** 5-10 minutes manual, ~30 seconds if you script it.
> **What you'll know after:** Daily P&L, which positions are still open, any anomalies that need attention before tomorrow's open.

---

## 1. When to use this

Every trading day. The review catches three classes of issue early:

- **Slow-bleeding configuration problems** (e.g. May 12: Schwab tokens missing → 0 trades for a whole day).
- **Process-state issues** (e.g. May 13: two Streamlit instances running in parallel).
- **Stuck positions** that need manual broker intervention before they decay further.

Skipping the review means these compound silently. With it, the failure window stays at one trading day.

---

## 2. What you need first

- Pi accessible via SSH (`balakrishh@myrasberrypi.local`).
- `~/pi-diagnostics/` folder on your Mac with TODAY's pull. If you haven't synced, run:

```bash
ssh balakrishh@myrasberrypi.local \
    "tar czf - -C /home/balakrishh/Documents/trading-agent trade_journal logs AGENT_LOG trade_plans" \
    | tar xzf - -C ~/pi-diagnostics/
```

- The Alpaca paper UI open in a tab (https://alpaca.markets/) — to cross-check positions against the journal.

---

## 3. Step-by-step diagnostic

Run these in order. Each one's expected output is documented; deviation is the diagnostic signal.

### Step 3.1 — Action histogram

```bash
python3 << 'PY'
import json
from collections import Counter
rows = [json.loads(l) for l in open('/Users/<you>/pi-diagnostics/trade_journal/signals_live.jsonl') if l.strip()]
today = [r for r in rows if r.get('timestamp', '').startswith('2026-05-13')]   # set today's date
print(f"Total rows today: {len(today)}")
print(Counter(r.get('action') for r in today))
PY
```

**Expected ranges:**

| Action | Healthy daily count | Anomaly if... |
|---|---|---|
| `rejected` | 50-500 (adaptive scanner sitting out — premium too thin) | 0 (chain probably broken, see step 3.3) |
| `skipped_existing` | 100-300 per held position | 0 when you have open positions (dedup gate broken) |
| `skipped_rsi_gate` | 0-200 (depends on regime + tickers) | n/a |
| `skipped_defense_first` | 0-200 (depends on volatility) | n/a |
| `submitted` | 0-5 (varies wildly) | >10 (over-trading?) |
| `closed` | 0-3 | n/a |
| `close_failed` | **0** | **any** — go to runbook 03 immediately |
| `warning` | **0** | **any** — broker/data issue, see step 3.4 |
| `error` | **0** | **any** — cycle aborted; check log |
| `cycle_error` | event-only rows (after_hours_shutdown is benign) | ANY error_message other than after_hours |

### Step 3.2 — Daily P&L

```bash
python3 << 'PY'
import json
path = '/Users/<you>/pi-diagnostics/trade_journal/signals_live.jsonl'
rows = [json.loads(l) for l in open(path) if l.strip()]
today = [r for r in rows if r.get('timestamp', '').startswith('2026-05-13')]

# Account balance trajectory
bals = [(r['timestamp'][:19], (r.get('raw_signal') or {}).get('account_balance'))
        for r in today]
bals = [(t, b) for t, b in bals if b and b > 0]
if bals:
    print(f"Equity start of day:  ${bals[0][1]:,.2f}  ({bals[0][0]})")
    print(f"Equity end of day:    ${bals[-1][1]:,.2f}  ({bals[-1][0]})")
    print(f"Net delta:            ${bals[-1][1] - bals[0][1]:+,.2f}")

# Realized P&L (closes)
realized = 0
print("\nRealized P&L from closes:")
for r in today:
    if r.get('action') == 'closed':
        rs = r.get('raw_signal') or {}
        pl = float(rs.get('net_unrealized_pl') or 0)
        realized += pl
        print(f"  {r['timestamp'][:19]}  {r.get('ticker')}  ${pl:+.2f}  ({rs.get('exit_signal')})")
print(f"  Total realized: ${realized:+.2f}")
PY
```

Realized P&L should reconcile with the broker's daily P&L from Alpaca's Activity → Today tab. If they disagree by more than $5 (commission/slippage), something happened off-journal — investigate.

### Step 3.3 — Anomaly: were ALL rejections "No positive-EV"?

If step 3.1 showed 0 submitted + 0 closed, check whether the agent was BLIND vs CORRECTLY SITTING OUT:

```bash
python3 << 'PY'
import json
from collections import Counter
path = '/Users/<you>/pi-diagnostics/trade_journal/signals_live.jsonl'
rows = [json.loads(l) for l in open(path) if l.strip()]
today = [r for r in rows if r.get('timestamp','').startswith('2026-05-13')]
reasons = Counter()
for r in today:
    if r.get('action') != 'rejected': continue
    for c in (r.get('raw_signal') or {}).get('checks_failed') or []:
        reasons[c[:50]] += 1
for reason, n in reasons.most_common(10):
    print(f"  {n:4d}  {reason}")
PY
```

- **"No put contracts available" / "No call contracts available"** → chain fetch broken. Verify `.env` has the right `MARKET_DATA_PROVIDER` and tokens exist. **This is the May 12 failure mode.**
- **"No positive-EV candidate found"** → chain is fine, adaptive scanner correctly decided premium was too thin. Healthy decision.
- **"Credit/Width ratio X < Y"** with X > 0 → scanner found candidates but they didn't clear the floor. Normal in low-vol environments.

### Step 3.4 — Warnings

```bash
python3 << 'PY'
import json
path = '/Users/<you>/pi-diagnostics/trade_journal/signals_live.jsonl'
rows = [json.loads(l) for l in open(path) if l.strip()]
warns = [r for r in rows
         if r.get('timestamp','').startswith('2026-05-13')
         and r.get('action') == 'warning']
for r in warns:
    rs = r.get('raw_signal') or {}
    print(f"{r['timestamp'][:19]}  [{rs.get('source','?')}]  {rs.get('message','')[:120]}")
PY
```

Each warning is a vendor-API or subsystem failure that survived all retries. Common sources:

| Source | What it means | What to do |
|---|---|---|
| `executor` | Order POST failed after 2 retries | Check `client_order_id` field in the row → search Alpaca by that ID to see if ANY attempt landed |
| `schwab_oauth` | Token refresh failed | Run `python -m trading_agent.schwab_oauth login` |
| `position_monitor` | Position fetch failed → dedup gate fail-closed | Usually transient; see if it recurred more than once |

### Step 3.5 — Ghost-process check

```bash
python3 << 'PY'
import json
from collections import Counter
path = '/Users/<you>/pi-diagnostics/trade_journal/signals_live.jsonl'
rows = [json.loads(l) for l in open(path) if l.strip()]
today = [r for r in rows if r.get('timestamp','').startswith('2026-05-13')]
clusters = Counter()
for r in today:
    t = r.get('timestamp','')[:19]
    if t and r.get('ticker'):
        clusters[(t, r['ticker'])] += 1
multi = [(k, n) for k, n in clusters.items() if n > 1]
print(f"Same-second same-ticker journal rows: {len(multi)} buckets")
if multi:
    print("⚠️  GHOST PROCESSES — go to runbook 05")
    for (ts, tk), n in sorted(multi, key=lambda x: -x[1])[:10]:
        print(f"  {ts}  {tk}: {n} rows")
else:
    print("✅ No same-second clusters — single agent process all day")
PY
```

### Step 3.6 — Open positions at end of day

```bash
# From the broker's view (live state, not journal-derived):
ssh balakrishh@myrasberrypi.local \
    "tail -100 /home/balakrishh/Documents/trading-agent/logs/trading_agent.log" \
    | grep -E "HOLD —" | tail -10
```

Cross-reference with the Alpaca paper UI (Activity → Positions). The numbers should match exactly. Discrepancy means the journal or the broker is wrong — investigate.

---

## 4. Decision tree

Based on what steps 3.1-3.6 showed:

```
Did step 3.1 show close_failed > 0?
├── YES → Go to runbook 03 (zombie position recovery)
└── NO  → continue

Did step 3.1 show warning > 0?
├── YES → Go to runbook 04 (broker API errors) for the specific source
└── NO  → continue

Did step 3.3 show "No put/call contracts available"?
├── YES → Chain fetch is broken. Check MARKET_DATA_PROVIDER + tokens. See runbook 02.
└── NO  → continue

Did step 3.5 show same-second clusters?
├── YES → Go to runbook 05 (ghost processes) — kill stragglers before tomorrow's open
└── NO  → continue

Did step 3.6 show positions the journal doesn't?
├── YES → Manual broker reconciliation needed. Check Alpaca's Activity feed.
└── NO  → Day was clean. Close the laptop.
```

---

## 5. Remediation actions

For each issue type, the specific fix:

- **`close_failed` rows** → manually close the broken legs at Alpaca's UI, then runbook 03 §5 to verify the cooldown has the position parked correctly.
- **`warning` rows for `schwab_oauth`** → re-run the OAuth login on the Pi.
- **`warning` rows for `executor`** → search Alpaca by the `client_order_id` from the row; if the order is on the book, cancel it; if it's filled, accept it; either way, journal a manual note.
- **Chain-empty rejections** → fix `.env` (likely `MARKET_DATA_PROVIDER` mismatch), restart agent.
- **Ghost processes** → kill all `streamlit run` and `python.*trading_agent.agent` processes on the Pi, restart cleanly.
- **Position-journal mismatch** → reconcile manually; if frequent, file an issue.

---

## 6. Verification

After remediation, re-run step 3.1 + step 3.5 the next trading day. Expect:

- All `close_failed` rows from the bad day should be followed by EITHER `closed` rows (you closed them) OR no further close attempts (cooldown engaged correctly).
- `warning` row sources from the bad day should not reappear.
- Same-second clusters should be 0.

If anomalies recur after remediation, escalate — likely a code-level bug, not an operational hygiene issue.

---

## 7. Prevention

Three habits prevent ~80% of the issues this runbook catches:

1. **Always run the pre-start hygiene** (kill all stragglers, clear sentinels, verify `pgrep` is clean) before clicking Start each session.
2. **Read AGENT_LOG once per day** — it captures the Streamlit-side Start/Stop transitions and shows ghost-process accumulation early.
3. **Run this runbook every market-close.** ~5 minutes scripted; bills itself back fast the first time it catches something.

You can automate steps 3.1, 3.2, 3.4, 3.5 into a single shell script and cron it to run at 4:30 PM ET. Email yourself the output. Then daily review is "did I get the email; is anything red."

---

## 8. LLM hand-off template

To get a fresh LLM to do this analysis for you, paste:

```
I just synced today's diagnostics from my trading agent. Please run the daily close review per docs/runbooks/01_daily_close_review.md against the attached files.

Attached:
- pi-diagnostics/trade_journal/signals_live.jsonl
- pi-diagnostics/logs/trading_agent.log
- pi-diagnostics/AGENT_LOG

Today's date (for filtering): 2026-05-13

Report your findings in this exact structure:

## Action histogram
[counter output]

## Daily P&L
- Start equity: $X
- End equity: $Y
- Net delta: $Z
- Realized P&L breakdown by close

## Anomaly flags (any that are non-zero)
- close_failed: [N or "none"]
- warning: [N or "none"]
- ghost-process clusters: [N or "none"]
- chain-empty rejections: [N or "none"]

## Recommended next steps
[list 0-3 specific runbooks to read based on the anomalies]

Do not propose code changes. Just diagnose.
```

The LLM should produce a 1-page report. If it strays into proposing fixes, redirect: "Diagnose only. I'll ask about fixes after I read the runbooks you suggested."

---

## 9. Related runbooks + skills

- Runbook 02 (zero trades diagnostic) — if today produced 0 submitted/closed and you want to dig into why.
- Runbook 03 (zombie recovery) — if step 3.1 showed any `close_failed`.
- Runbook 04 (broker errors) — to decode specific Alpaca/Schwab error codes from warnings.
- Runbook 05 (ghost processes) — if step 3.5 showed same-second clusters.
- Skill 19 (journal schema) — for the full meaning of every `action` value the runbook references.

---

*Last verified against repo HEAD on 2026-05-13.*
