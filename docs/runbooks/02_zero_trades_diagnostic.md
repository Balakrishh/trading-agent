# Runbook 02 — Zero Trades Diagnostic

> **Trigger:** End of day shows 0 `submitted` rows AND your strategy preset says trades should be possible (i.e. you're not in a directional-bias-blocks-everything state).
> **Status:** Stub — will be expanded the next time this scenario takes >30 min to diagnose.

---

## Quick-start decision tree (full runbook coming)

Walk through these checks in order; the first one that's "yes" is the most likely cause.

### Was the market-data provider broken?

```bash
grep "Schwab API call.*aborted" pi-diagnostics/logs/trading_agent.log | tail -5
grep "No put contracts available\|No call contracts available" pi-diagnostics/logs/trading_agent.log | tail -5
```

If you see either → chain fetch is broken. Check `.env`:

```bash
ssh balakrishh@myrasberrypi.local "grep '^MARKET_DATA_PROVIDER' /home/balakrishh/Documents/trading-agent/.env"
```

Schwab routes with no tokens → empty chain → "No put contracts available" → all rejections. **This is the 2026-05-12 failure mode.**

### Did adaptive scanner sit out (legitimately)?

```bash
python3 -c "
import json
rows = [json.loads(l) for l in open('pi-diagnostics/trade_journal/signals_live.jsonl') if l.strip()]
today = [r for r in rows if r.get('timestamp','').startswith('2026-05-XX')]  # set date
nope = sum(1 for r in today
           if r.get('action') == 'rejected'
           and 'No positive-EV' in str((r.get('raw_signal') or {}).get('checks_failed', [])))
total = sum(1 for r in today if r.get('action') == 'rejected')
print(f'{nope}/{total} rejections were No-positive-EV (legitimate sit-out)')
"
```

If >80% of rejections were "No positive-EV," the adaptive scanner correctly decided premium was too thin. Healthy decision — no fix needed.

### Were RSI-gate / defense-first / bias filters blocking everything?

```bash
python3 -c "
import json
from collections import Counter
rows = [json.loads(l) for l in open('pi-diagnostics/trade_journal/signals_live.jsonl') if l.strip()]
today = [r for r in rows if r.get('timestamp','').startswith('2026-05-XX')]  # set date
print(Counter(r.get('action') for r in today))
"
```

If `skipped_rsi_gate` + `skipped_defense_first` + `skipped_bias` together account for ~80% of cycles → filters are right (extreme RSI, high IV, regime mismatch). Consider whether the filters are too aggressive for current market conditions; revisit preset config.

### Were no tickers even being processed?

```bash
grep "Tickers:" pi-diagnostics/logs/trading_agent.log | tail -3
```

If the tickers list is empty or your `.env` `TICKERS=` line is missing, the agent literally has nothing to evaluate.

---

## Going deeper

When this stub stops being enough, expand into the full 8-section runbook format. The full structure should cover:

1. When to use this (more specific symptoms)
2. What you need first
3. Step-by-step diagnostic (chain-fetch probe, adaptive scanner diagnostics, preset evaluation)
4. Decision tree (chain broken / scanner sitting out / filters blocking / config error)
5. Remediation (fix .env / tune preset / fix filter thresholds)
6. Verification (next-cycle journal check)
7. Prevention
8. LLM hand-off template

---

## Cross-references

- **Runbook 01** — daily close review. Surfaces "0 trades" as an anomaly.
- **Skill 09** — VIX z-score inhibitor (one filter that can block trades).
- **Skill 14** — adaptive vs static scan modes (chain-scanner sit-out logic).
- **Skill 16** — market-data provider routing (where Schwab token issues live).

---

*Stub created 2026-05-13. Expand to full runbook on first significant occurrence.*
