# Runbook 05 — Ghost Process Diagnostic

> **Trigger:** Runbook 01 §3.5 flagged same-second clusters in the journal, OR `pgrep` shows >1 agent/streamlit process, OR you see paired cycle-start timestamps in the agent log offset by a small number of seconds.
> **Status:** Stub — will be expanded the next time multi-process diagnosis takes >30 min.

---

## What "ghost process" means

The trading agent's intended state is exactly one Streamlit process running the dashboard, with that Streamlit's daemon thread spawning exactly one cycle subprocess every 5 minutes. A "ghost" is any deviation from this:

- Two `streamlit run` processes alive (each with its own daemon thread → two cycle schedulers).
- A cycle subprocess from a previous Streamlit session that the orphan sweep didn't catch.
- A cron job AND a dashboard daemon both launching cycles independently.
- A systemd service AND a dashboard daemon stacking.

The journal evidence is **same-second-same-ticker rows** — two processes hitting the same ticker in their respective cycles at almost the same instant. The agent log evidence is **paired cycle starts offset by ~10-40 seconds**, repeating every 5 minutes.

---

## Quick-start diagnostic

### Step 1 — Verify from process state

```bash
ssh balakrishh@myrasberrypi.local "
    echo '=== streamlit processes ==='
    pgrep -af 'streamlit run'
    echo
    echo '=== agent cycle processes ==='
    pgrep -af 'python.*trading_agent.agent'
    echo
    echo '=== cron schedule ==='
    crontab -l 2>/dev/null
    echo
    echo '=== systemd trading services ==='
    systemctl list-units --all 2>/dev/null | grep -i trading
"
```

Expected healthy state: one streamlit line, zero-or-one agent line, no cron job for the agent, no systemd unit (unless you set one up intentionally).

### Step 2 — Verify from the journal

```bash
python3 << 'PY'
import json
from collections import Counter
rows = [json.loads(l) for l in open('pi-diagnostics/trade_journal/signals_live.jsonl') if l.strip()]
today = [r for r in rows if r.get('timestamp','').startswith('2026-05-XX')]  # set date
clusters = Counter()
for r in today:
    t = r.get('timestamp','')[:19]
    if t and r.get('ticker'):
        clusters[(t, r['ticker'])] += 1
multi = [(k, n) for k, n in clusters.items() if n > 1]
print(f"Same-second same-ticker buckets: {len(multi)}")
if multi:
    print("⚠️  GHOST PROCESSES CONFIRMED")
    # Bucket size 2 = two processes; 3 = three; etc.
    sizes = Counter(n for _, n in multi)
    print(f"  Cluster sizes: {dict(sizes)}")
PY
```

### Step 3 — Cleanup

The nuclear option — kill everything that might be spawning cycles, then start fresh:

```bash
ssh balakrishh@myrasberrypi.local "
    pkill -9 -f 'streamlit run' 2>/dev/null
    pkill -9 -f 'python.*trading_agent.agent' 2>/dev/null
    sleep 2
    cd /home/balakrishh/Documents/trading-agent
    rm -f AGENT_RUNNING AGENT_PID DRY_RUN_FLAG PAUSE_FLAG
    pgrep -af 'streamlit|trading_agent' | grep -v 'pet server\|vscode' \
        || echo 'Clean'
"
```

Then start ONE Streamlit instance from a single terminal, click Start in the dashboard ONCE, verify with step 1 that exactly one of each is alive.

---

## Going deeper

When this stub stops being enough, expand into the full 8-section runbook covering:

1. Forensic decoding of cycle-start timestamp patterns from AGENT_LOG (e.g., the 31-second offset signature of two daemon threads).
2. Determining which process is "yours" vs orphan when you have legitimate ancestor processes.
3. Step-by-step manual identification of WHICH launcher (Streamlit / cron / systemd / tmux session) is responsible for each ghost.
4. The structural fixes shipped on 2026-05-13: orphan-sweep extension to match `streamlit run`, `_ancestor_pids()` exclusion logic.

---

## Cross-references

- **Runbook 01** §3.5 — daily ghost-process check.
- **Skill 17** §4 edge cases — process-restart resilience (why the cooldown counter had to migrate to journal-derived state).
- **`streamlit/live_monitor.py:_sweep_orphan_agents`** — the structural fix that catches both agent and Streamlit orphans.
- **`PROJECT_CONTEXT.md` §10 entry 18c** — the post-mortem from the 2026-05-13 incident.

---

*Stub created 2026-05-13. Expand to full runbook on first significant recurrence after the orphan-sweep extension was shipped.*
