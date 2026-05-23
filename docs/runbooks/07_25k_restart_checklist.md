# 25K Paper Restart — Pre-Flight Checklist + Decision Criteria

**Date drafted:** 2026-05-23. Use this for the upcoming paper-trade restart at $25K.

This runbook replaces the backtester as the primary validator. The backtester's synthetic-chain pricing undershoots real Schwab chains by ~3×, so it can't validate strategy outcomes for a real-money decision. Direct paper-trading evidence is the only honest path.

---

## Pre-flight (do BEFORE turning the agent back on)

### 1. Update the preset for $25K capital

Edit `STRATEGY_PRESET.json` at the repo root. The current preset is calibrated for a $5K account where 5% per-trade risk is tolerable. At $25K with PDT lifted, the math should look different:

```diff
 {
   "profile": "custom",
   "directional_bias": "auto",
   "scan_mode": "adaptive",
   "edge_buffer": 0.0,
   "custom": {
-    "max_delta": 0.35,
+    "max_delta": 0.20,
     "dte_vertical": 7,
     "dte_iron_condor": 21,
     "dte_mean_reversion": 7,
     "dte_window_days": 5,
     "width_mode": "pct_of_spot",
     "width_value": 0.025,
-    "min_credit_ratio": 0.20,
+    "min_credit_ratio": 0.25,
-    "max_risk_pct": 0.05,
+    "max_risk_pct": 0.02,
     "min_pop": 0.55,
     "dte_grid": [7, 14, 21, 30],
-    "delta_grid": [0.20, 0.25, 0.30, 0.35],
+    "delta_grid": [0.15, 0.18, 0.20, 0.22],
     "width_grid_pct": [0.005, 0.01, 0.015, 0.02, 0.025]
   }
 }
```

The rationale:

- **`max_delta: 0.20`** (was 0.35) — 80%-POP shorts instead of 65%. Three of your six live losses were `STRIKE_PROXIMITY` exits; tighter shorts dramatically reduce that exit rate.
- **`max_risk_pct: 0.02`** (was 0.05) — $500 max risk per trade on $25K. One bad trade caps at 2% of equity. You can take 50 losers in a row before margin call.
- **`min_credit_ratio: 0.25`** (was 0.20) — Slightly higher quality bar on the trades that do open. With lower delta, this matters less, but the floor is structurally tighter.
- **`delta_grid`** tightened — the scanner sweep range matches the new max_delta.

### 2. Reconcile the broker account

Before flipping the switch, log into Alpaca paper and:

- [ ] Confirm all open positions from the $5K run are closed
- [ ] Confirm cash balance equals what you intend to seed ($25K target)
- [ ] If there are stuck XLF positions (139 close_failed events suggested at least one), close them manually via the Alpaca UI

### 3. Reset the journal cleanly

Don't carry old journal state into the new run. The pre-fix phantom DIA closes are still in `signals_live.jsonl` and will pollute your EOD recap on day 1.

```bash
# Archive the old journal
cd ~/trading-agent/trade_journal
mv signals_live.jsonl signals_live_5k_run_2026-05-12_to_2026-05-23.jsonl
mv signals_dryrun.jsonl signals_dryrun_5k_archived.jsonl
# Agent will recreate empty journals on next start
```

Same for `trade_plans/`:
```bash
mkdir -p trade_plans_archive_5k
mv trade_plans/*.json trade_plans_archive_5k/
```

### 4. Health-check the integration surfaces

- [ ] **Schwab token fresh**: `python -m trading_agent.schwab_oauth login` if access token within 6 hours of expiry. Last issue was multi-hour silent failure when token expired; skill-34 alerts catch this now, but starting fresh avoids the noise.
- [ ] **Both Telegram bots respond**: Send a `/status` to each. Error channel + info channel should be distinct chats.
- [ ] **Pi disk has headroom**: `df -h ~/trading-agent` — journals grow ~50KB/day. Need >1GB free.
- [ ] **Agent service restarts cleanly**: `systemctl --user restart trading-agent && sleep 30 && journalctl --user -u trading-agent --since '1 min ago' | grep -iE 'ERROR|CRITICAL'` — expect zero hits.

### 5. Document the start state

Write down (or screenshot for the dashboard):

- [ ] Start date + time
- [ ] Starting equity ($25K target)
- [ ] Preset config snapshot (`cp STRATEGY_PRESET.json STRATEGY_PRESET_25k_start.json`)
- [ ] Universe (tickers from `.env`)
- [ ] Git SHA: `git rev-parse HEAD`

This is the baseline you'll compare against in 30 days.

---

## During the run (weekly)

### Weekly checkpoint (every Friday 5 PM ET)

Run this command on the pi:

```bash
cd ~/trading-agent
python -c "
from trading_agent.journal_reader import JournalReader
from datetime import datetime, timezone, timedelta
r = JournalReader('trade_journal/signals_live.jsonl')
# Today
print('Closes today:', len(r.closes_today()))
print('Realized P&L today: \${:.2f}'.format(r.realized_pl_today()))
print('Cycle count today:', r.cycle_minute_count_today())
print('Errors today:', r.error_count_today())
print('Stuck positions:', len(r.stuck_positions()))
print('Silenced exceptions:', len(r.silenced_exceptions_today()))
"
```

### Capture each week:

| Metric | Target / Threshold |
|---|---|
| Closed trades this week | 5-15 (varies by regime) |
| Win rate cumulative | track only — decision at day 30 |
| Realized P&L cumulative | track only |
| Stuck partial-fill events | should be ≤ 1 per week |
| Silenced exceptions paged | should be 0 (red flag if ≥1) |
| Schwab auth failures | should be 0 |
| Manual interventions required | should be 0 |

Save the weekly snapshot to `pi-diagnostics/weekly_YYYY-MM-DD.txt` and commit to git. That's your audit trail.

---

## Decision criteria (at day 30)

Don't extend the timeline. Don't make exceptions. Either it meets all of these, or it doesn't:

### MUST-meet (any failure → no real money)

- [ ] **Win rate ≥ 65%.** Credit-spread theta-decay strategies fundamentally require this. Below 65% the math doesn't work even with perfect execution.
- [ ] **Max drawdown < 15%.** $25K → never below $21,250 during the trial. If you saw 15% you'd be uncomfortable; in real money, you'd panic.
- [ ] **Win/loss ratio ≥ 1.0.** Average winner must be at least as large as average loser. If wins are smaller, you need a >65% win rate to compensate; if you have a 65% win rate AND wins are smaller than losses, the EV is negative.
- [ ] **Zero stuck partial-fill incidents lasting > 1 trading day.** The 139-event XLF stuck position is operationally unacceptable on real money — naked-leg risk.
- [ ] **Zero unauthorized state divergence.** No "the journal said one thing and the broker said another" incidents.

### NICE-to-have (failure → reduce starting real-money capital but proceed)

- [ ] At least 30 closed trades total (sample size for win-rate confidence)
- [ ] At least 3 different regime types observed (BULL, BEAR, SIDEWAYS days)
- [ ] At least one volatility spike day handled without panic-closing

### Honest off-ramps

If the strategy doesn't meet criteria after 30 days, the response is NOT to extend or tweak. The right reactions:

1. **30 days, missed by < 5%** — extend to 60 days. Sample size matters.
2. **30 days, missed by > 5%** — strategy needs structural change. Consider switching to fewer tickers, different DTE band, or a different strategy class (e.g., calendar spreads). Don't deploy real money on a hopeful tweak.
3. **30 days, met criteria but operationally fragile** — fix operations first. Real money on a broken system is dangerous regardless of strategy quality.

---

## What NOT to do

- **Don't run the backtester for validation.** It's miscalibrated. Use it only for code-regression checks.
- **Don't compare to last 11-day live performance.** Different account size, different preset, partial code paths. Treat the new run as a clean slate.
- **Don't change the preset mid-trial.** If you must change it, the 30-day clock resets.
- **Don't deploy real money before day 30 even if results look great by day 14.** Small-sample wins are noise.
- **Don't deploy real money based on "feeling good about the system."** The criteria are checkboxes for a reason. Either yes or no.

---

## Real-money capital sequencing (when criteria are met)

Even after a clean 30-day paper trial, the right rollout is gradual:

1. **Week 1 real:** $1000 — half normal position size. Watch for execution differences (slippage, partial fills) that paper doesn't reveal.
2. **Week 2-4 real:** $5000 — normal position size. Look for any difference vs paper.
3. **Week 5+ real:** Scale to target. Continue paper trading $25K alongside as a control.

Treat the first real-money month as paying for education even if profitable. The information about live-vs-paper drift is worth more than the small profits.

---

*Last updated: 2026-05-23.*
