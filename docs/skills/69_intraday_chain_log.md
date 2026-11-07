# Intraday Option-Chain Log

> **One-line summary:** while `intraday_chain_log_mode == "on"`, each cycle appends a snapshot of the next 0–1 DTE option chains (strikes near spot, bid / ask / delta / IV) to a gzipped JSON-lines file per ticker and day — research data for same-day trade ideas; places no orders.
> **Source of truth:** [`trading_agent/intraday_chain_log.py:42-104`](../../trading_agent/intraday_chain_log.py)
> **Phase:** 2  •  **Group:** data_quality
> **Depends on:** `16_market_data_provider_routing.md` (`fetch_option_chain`), `13_preset_system_hot_reload.md` (preset fields)
> **Consumed by:** research scripts only — nothing in the live decision path reads these files.

---

## 1. Theory & Objective

The free historical chains (`~/data/options_chains`, SPY / QQQ / IWM 2008–2025) are end-of-day only. A same-day trade (open and close within the session, typically 0DTE) cannot be tested on them: the 1-day study in `docs/plans/research_credit_vol.md` §3 had to hold overnight. Paid intraday option history is expensive, so the agent records its own: Schwab already serves the chain to the agent every cycle, and four extra requests per cycle (put + call × two expirations) are cheap. After 4–6 weeks the files support an honest backtest of same-day ideas (opening-range credit spreads, straddle sales after a gap, etc.) with real intraday bid / ask. Pure logging — no decision reads it, so it cannot change trading.

## 2. Mathematical Formula

N/A — control flow only.

```text
snapshot taken  ⇔  mode == "on"
                 ∧  09:30 ≤ now_ET < 16:00
                 ∧  now − last_snapshot ≥ intraday_chain_log_interval_s
expirations     =  the next (intraday_chain_log_max_dte + 1) NYSE sessions, today inclusive
strikes kept    =  |K − spot| / spot ≤ intraday_chain_log_band_pct
row             =  {ts, ticker, spot, chains: {expiry: [[P|C, K, bid, ask, Δ, iv], …]}}
file            =  $INTRADAY_CHAIN_DIR/<TICKER>_<YYYY-MM-DD>.jsonl.gz   (default ~/data/intraday_chains)
```

Size: SPY at ±4 % on a $1 grid ≈ 60 strikes × 2 sides × 2 expiries ≈ 240 contracts a snapshot, ~250 snapshots a day at the ~95 s cycle — a few MB a day gzipped.

## 3. Reference Python Implementation

```python
# trading_agent/intraday_chain_log.py:42-104
def compact(contracts: Iterable[dict], spot: float, band: float) -> List[list]:
    """[type, strike, bid, ask, delta, iv] for strikes within ``band`` of spot."""
    out = []
    for c in contracts or []:
        k = c.get("strike")
        if not k or abs(k - spot) / spot > band:
            continue
        out.append([c.get("type", "")[:1].upper(), k, c.get("bid"), c.get("ask"),
                    c.get("delta"), c.get("iv")])
    return sorted(out, key=lambda r: (r[0], r[1]))


def due(sentinel: Path, interval_s: float, now: float) -> bool:
    try:
        return now - float(sentinel.read_text().strip()) >= interval_s
    except (OSError, ValueError):
        return True


def maybe_log(preset, data_provider, now_et: datetime, out_dir: Optional[Path] = None) -> int:
    """Append one snapshot per configured ticker when due. Returns the
    number of snapshots written."""
    if getattr(preset, "intraday_chain_log_mode", "off") != "on":
        return 0
    if not (SESSION_OPEN <= now_et.time() < SESSION_CLOSE):
        return 0
    out_dir = out_dir or log_dir()
    sentinel = out_dir / ".last_snapshot"
    now = time.time()
    if not due(sentinel, preset.intraday_chain_log_interval_s, now):
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    today = now_et.date()
    exps = next_sessions(today, preset.intraday_chain_log_max_dte + 1)
    written = 0
    for tk in preset.intraday_chain_log_tickers:
        try:
            spot = data_provider.get_cached_price(tk) or data_provider.get_current_price(tk)
            if not spot:
                continue
            chains = {}
            for exp in exps:
                rows = []
                for side in ("put", "call"):
                    rows += compact(data_provider.fetch_option_chain(tk, exp.isoformat(), side),
                                    spot, preset.intraday_chain_log_band_pct)
                if rows:
                    chains[exp.isoformat()] = rows
            if not chains:
                continue
            row = {"ts": now_et.isoformat(timespec="seconds"), "ticker": tk,
                   "spot": spot, "chains": chains}
            with gzip.open(out_dir / f"{tk}_{today.isoformat()}.jsonl.gz", "at") as fh:
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            written += 1
        except Exception as exc:                        # noqa: BLE001, skill-34-exempt — research logging must never break a cycle
            logger.warning("[%s] intraday chain snapshot failed: %s", tk, exc)
    tmp = sentinel.with_suffix(".tmp")
    tmp.write_text(str(now))
    tmp.replace(sentinel)
    if written:
        logger.info("Intraday chain log: %d snapshot(s) → %s", written, out_dir)
    return written
```

```python
# trading_agent/calendar_utils.py:57-60
def next_sessions(today: date, n: int) -> List[date]:
    """The next *n* NYSE trading days from *today* (inclusive) — the daily
    expirations SPY / QQQ list."""
    return _valid_days(today, today + timedelta(days=3 * n + 7))[:n]
```

The agent calls it once per cycle, after the hourly portfolio review and before Stage 1, through `TradingAgent._maybe_log_intraday_chains` (`trading_agent/agent.py`), which swallows any exception.

Preset fields (`trading_agent/strategy_presets.py`, Strategy-Profile panel "Intraday option-chain log"): `intraday_chain_log_mode` (`"off"` | `"on"`, default off), `intraday_chain_log_tickers` (`("SPY",)`), `intraday_chain_log_max_dte` (1), `intraday_chain_log_band_pct` (0.04), `intraday_chain_log_interval_s` (60 — every ~95 s cycle).

## 4. Edge Cases / Guardrails

- **Each cycle is a fresh process** — the throttle cannot live in memory; the last write time is a `.last_snapshot` sentinel in the output directory (atomic temp + rename). A missing or garbled sentinel means "due".
- **Pre-open and post-close cycles** (the session buffer starts 09:25) — no snapshot outside 09:30–16:00 ET; pre-open quotes are not tradable prices.
- **A ticker fails** (Schwab error, no price) — logged at warning, the other tickers still write; the sentinel is still advanced so a broken feed is not hammered every cycle.
- **An expiry with no contracts** (holiday-shortened week, ticker without daily expiries) — skipped; a ticker with no chains at all writes no row.
- **Holidays** — expirations come from the NYSE calendar (`next_sessions`), so the day after Thanksgiving follows the Wednesday, not the holiday.
- **Disk** — a few MB a day for SPY; the directory is outside the repo (`INTRADAY_CHAIN_DIR` overrides). Prune by date when no longer needed.
- **Schwab token** — uses the agent's own provider, so it adds no second token holder.
- **Invalid mode in the preset file** (e.g. `"live"`) — warned and reset to the default `"off"`.

## 5. Cross-References

- `16_market_data_provider_routing.md` — the chain fetch this reuses.
- `13_preset_system_hot_reload.md` — the preset fields.
- `docs/plans/research_credit_vol.md` — why the data is needed (EOD-only history).

---

*Last verified against repo HEAD on 2026-11-06.*
