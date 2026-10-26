# Quality-at-Lows Sleeve — Growth Data, Analyst Revisions, Long-Option Shadow Scan

> **One-line summary:** Once a day, find strong companies trading near their 52-week low for a sentiment reason (revenue growing, analyst estimates not cut, sector not the cause), follow them through a base to a confirmed turn, and journal which long-option structure the sleeve *would* buy — LEAPS call, call debit spread, or cash-secured put — priced from real chains and sized to the account. Shadow only: no orders.
> **Source of truth:** [`trading_agent/quality_lows.py`](../../trading_agent/quality_lows.py), [`trading_agent/analyst_estimates.py`](../../trading_agent/analyst_estimates.py), [`trading_agent/drop_cause_llm.py`](../../trading_agent/drop_cause_llm.py), [`trading_agent/news_aggregator.py:ticker_news`](../../trading_agent/news_aggregator.py), [`trading_agent/market_data_schwab.py:fetch_fundamentals`](../../trading_agent/market_data_schwab.py), [`scripts/research/quality_lows_backtest.py`](../../scripts/research/quality_lows_backtest.py), [`deploy/systemd/trading-agent-quality-lows.*`](../../deploy/systemd/).
> **Phase:** 2  •  **Group:** strategy / data_quality
> **Depends on:** `47_schwab_data_api.md` (fundamentals + chains, one Schwab token holder), `16_market_data_provider_routing.md`, `52_watchlist_curation.md` (the universe), `65_earnings_expiry_and_trend_filters.md` (earnings dates), `66_calibrated_probability_model.md` (same daily-timer pattern as the vol surface), `56_daily_journal_reviewer.md` (the `claude` CLI pattern the drop-cause label reuses).
> **Consumed by:** `trade_journal/quality_lows.jsonl` (shadow evidence for backlog §12.5–12.6) · `python -m trading_agent.quality_lows --report`.

---

## 1. Theory & Objective

Operator goal (2026-10-25): put idle cash to work by buying good companies the market sold on mood, entering after a 2–3 week base and exiting toward the 52-week high, preferably with long options and sized for diversification. The known headwind is 52-week-high momentum — stocks near their lows tend to keep lagging — so being near the low only admits a name; it must also pass a quality screen, show its **growth intact** (Schwab revenue growth and margin direction, analyst EPS estimates not cut), and the **drop cause** must not be the business or the whole sector. Only then do the price stages decide timing.

Options make timing and price paid matter far more than with stock, so for every turn the scanner prices three structures from live chains and records which one fits the account: a deep ITM LEAPS call (stock replacement with little time decay), a call debit spread up to the 52-week high (cheaper; the short leg sells back some of the post-sell-off implied volatility), and a cash-secured put at the base (paid to wait). Nothing is traded: backlog §12.5 asks for 6–8 weeks of these rows before §12.6 paper trades.

Data sources: Schwab fundamentals already held the trailing growth fields (verified on the raw payload 2026-10-25) but no forecasts. Yahoo Finance (via `yfinance`, already a dependency for daily bars) publishes consensus EPS now and 7–90 days ago, analyst up/down revision counts, revenue and EPS growth estimates, price targets and the recommendation trend, with no API key. Both are optional inputs: a missing table weakens the verdict to `unclear`, never crashes the scan.

Numbers cannot tell a guidance cut from a panic over a customer, a layoff from a one-off. So every name near a low that passes quality also gets a **news-reading label** (backlog 12.3, added 2026-10-26): 30 days of headlines and summaries plus the row's numbers go to Claude through the operator's logged-in `claude` CLI, which answers `sentiment` / `sector` / `fundamental` / `unclear` and must cite the items it used. The label is journalled beside the rule-based cause and does not change the stage; the shadow weeks show which of the two calls the rebounds better.

## 2. Mathematical Formula

```text
Price (last 252 sessions of daily high/low/close, Yahoo, unadjusted):
  range_pos       = (close − low52) / (high52 − low52)
  base_low        = min(low, last base_window=30 sessions)
  base_range_pos  = (base_low − low52) / (high52 − low52)
  sessions_since_low = sessions since the most recent 52-week low
  higher_low      = min(low, last 10) > low52  and  sessions_since_low ≥ 10
  atr_narrowing   = ATR14(now) < ATR14(20 sessions ago)       (journalled, not gating)
  rv30            = stdev(daily log returns, 30) × √252

Quality (fundamentals_screen with long-term limits):
  market cap ≥ $10B, P/E ∈ (0, 120], EPS_ttm > 0, net margin ≥ 8 %, ROE ≥ 10 %,
  beta ≤ 3.0, 10-day volume ≥ 1M            (the Wheel screen uses P/E ≤ 40, beta ≤ 1.6)

Growth intact (False if any known check fails; None if nothing is known):
  rev_change_ttm_pct ≥ 0
  net_margin_mrq − net_margin_ttm ≥ −3 pp
  consensus EPS change over 30 days ≥ −3 %   for current and next year
  forward revenue growth (next year) > 0
  fallback only when no EPS change is known:  down − up revisions (30 d) ≤ 5

Drop cause:
  fundamental  if growth intact is False
  unclear      if growth intact is None or no drawdown
  sector       if sector-ETF drawdown ≤ 0.6 × stock drawdown   (both negative %)
  sentiment    otherwise

News-reading label (llm_cause; journal only, stage unchanged):
  eligible     stage ≠ not_near_low and quality passes (any rule-based cause)
  order        most advanced stage first, at most llm_max_names = 10 per run
  input        DATA = price, sector-ETF drawdown, fundamentals, estimates, rule label
               N1…Nk = up to 60 news items, 30 days, round-robin across days
  check        cited ids ∉ {N1…Nk, DATA} → unsupported_citations (dropped)
               label ∉ {sentiment, sector, fundamental, unclear} → unclear
               label ≠ unclear with no valid citation → unclear

Stage:
  not_near_low  if base_range_pos > 0.10 or range_pos > 0.35
  excluded      if quality fails or cause = fundamental
  watch         until sessions_since_low ≥ 10 and RSI14 > 40      (basing)
  basing        until close > SMA20 and higher_low                (turn)
  turn → confirmed when close > SMA50

Structure (turn / confirmed):
  wait_earnings       if 0 ≤ days to earnings ≤ 14
  call_debit_spread   if iv / rv30 > 1.3   (iv = the 4–9-month long call's IV)
  leaps_call          otherwise
  LEAPS:   third-Friday expiry 270–550 DTE, call nearest Δ 0.75, cost at the ask
  spread:  120–270 DTE, long call Δ ≈ 0.60, short at the strike nearest high52;
           debit = long ask − short bid; reward/risk = (width − debit) / debit
  fitting spread: same long leg, widest short whose debit ≤ budget with reward/risk ≥ 1
  fitting spread @alt: the same at the 5 % budget          (call_debit_spread_fit_alt)
  fixed-width spread: 120–270 DTE, long call Δ ≈ 0.55, short at the strike nearest
           long strike + 10 % × spot; move_to_max_pct = short / spot − 1   (for blue chips)
  CSP:     30–60 DTE put at the strike nearest base_low; credit at the bid
  shares:  stop = base_low × (1 − 0.08); target = high52 × (1 − 0.03)       (= the backtest exits)
           qty = min(7.5 % × equity / spot, 1.5 % × equity / (spot − stop)), fractional (4 dp)
Sizing:
  budget = 2 % of equity (premium = max loss);  contracts = ⌊budget / cost⌋
  budget_alt = 5 % of equity → contracts_alt, fits_alt on every option structure
  CSP fits if strike × 100 ≤ 15 % of equity (no alt)
  affordable     = first that fits of [chosen, spread to high, fitting spread, fitting spread @alt,
                   fixed-width spread, CSP]
  affordable_alt = the same with fits_alt;  shares are always priced but never "affordable"
  ≤ 2 turn rows per sector (sector_capped, strongest next-year revisions first)
```

## 3. Reference Python Implementation

### 3.1 Schwab growth fields

```python
# trading_agent/market_data_schwab.py — fetch_fundamentals (added 2026-10-26)
            "rev_change_ttm_pct":   _safe_float(_pick("revChangeTTM")) or None,
            "rev_change_year_pct":  _safe_float(_pick("revChangeYear")) or None,
            "gross_margin_mrq":     _safe_float(_pick("grossMarginMRQ")),
            "net_profit_margin_mrq": _safe_float(_pick("netProfitMarginMRQ")),
            "return_on_investment": _safe_float(_pick("returnOnInvestment")),
            "current_ratio":        _safe_float(_pick("currentRatio")),
            "interest_coverage":    _safe_float(_pick("interestCoverage")),
            "lt_debt_to_equity":    _safe_float(_pick("ltDebtToEquity")),
            "total_debt_to_capital": _safe_float(_pick("totalDebtToCapital")),
            "short_int_days_to_cover": _safe_float(_pick("shortIntDayToCover")) or None,
```

### 3.2 Analyst estimates

```python
# trading_agent/analyst_estimates.py
def summarize(tables: Dict[str, Any], spot: Optional[float] = None) -> Optional[Dict[str, Any]]:
    # eps_{cur,next}_year, eps_*_rev_30d_pct / _90d_pct, eps_growth_*_pct, rev_growth_*_pct,
    # revisions_up_30d / revisions_down_30d, analysts_cur_year, target_mean,
    # target_upside_pct, rec_score_now / rec_score_3m_ago, revision_trend

def get_estimates(ticker, *, spot=None, path=None, max_age_hours=CACHE_MAX_AGE_H,
                  fetch=None, now=None) -> Optional[Dict[str, Any]]:
    # cached in trade_journal/analyst_estimates.json for 20 h; never raises
```

### 3.3 Growth check — size of the cut decides

```python
# trading_agent/quality_lows.py — growth_check
    cut_known = False
    for tag in ("cur_year", "next_year"):
        v = est.get(f"eps_{tag}_rev_30d_pct")
        if v is not None:
            known += 1
            cut_known = True
            if v < -cfg.max_eps_cut_30d_pct:
                reasons.append(f"eps_{tag}_cut {v:+.1f}%/30d")
    up, down = est.get("revisions_up_30d"), est.get("revisions_down_30d")
    if not cut_known and up is not None and down is not None:
        known += 1
        if down - up > cfg.max_net_down_revisions:
            reasons.append(f"revisions {up}up/{down}down")
```

### 3.4 Stage

```python
# trading_agent/quality_lows.py
def stage(pf, quality_ok, cause, cfg=DEFAULT_CONFIG) -> str:
    if pf is None:
        return "excluded"
    near = pf["base_range_pos"] <= cfg.base_max_range_pos and pf["range_pos"] <= cfg.max_range_pos
    if not near:
        return "not_near_low"
    if not quality_ok or cause == "fundamental":
        return "excluded"
    rsi = pf.get("rsi14")
    basing = (pf["sessions_since_low"] >= cfg.min_sessions_since_low
              and rsi is not None and rsi > cfg.min_rsi)
    if not basing:
        return "watch"
    if not (pf["close"] > pf["sma20"] and pf["higher_low"]):
        return "basing"
    return "confirmed" if pf["close"] > pf["sma50"] else "turn"
```

### 3.4a Blue-chip structures and the 5 % cap (added 2026-10-26)

```python
# trading_agent/quality_lows.py
def price_fixed_width_spread(calls, spot, cfg):   # long Δ 0.55, short ≈ long + 10 % of spot
def price_shares(pf, equity, cfg):                # 7.5 % notional, cut to 1.5 % loss at the stop
def _sized(cost, equity, cfg):                    # size_contracts at 2 % + *_alt at 5 %
def affordable_structure(choice, quotes, key="fits"):   # key="fits_alt" → affordable_alt
```

`QualityLowsConfig`: `fixed_long_delta`, `fixed_width_pct`, `stop_below_base`, `target_below_high` (the backtest now reads the last two from here), `alt_premium_pct`, `shares_pct`, `shares_max_risk_pct`. Shares are priced for every `turn` / `confirmed` row, also under `--no-chains`.

Why: on a ~$29k account a deep ITM LEAPS fits neither cap on any blue chip (KO $1,695 is the cheapest; LLY $35,400), and their CSPs secure $8.5k–113k. Live chains 2026-10-26 (pre-market, Friday's quotes), 2 % = $578, 5 % = $1,446:

| | spot | fixed-width spread | cost | rr | 2 % / 5 % contracts | shares (sized by) | loss at stop |
|---|---|---|---|---|---|---|---|
| KO | 93 | 92.5/100 | $400 | 0.88 | 1 / 3 | 23.4 ($2,169, notional) | 1.16 % |
| WMT | 120 | 120/130 | $490 | 1.04 | 1 / 2 | 17.5 ($2,099, stop) | 1.50 % |
| PG | 161 | 160/175 | $720 | 1.08 | 0 / 2 | 13.5 ($2,169, notional) | 1.36 % |
| JNJ | 264 | 260/290 | $1,385 | 1.17 | 0 / 1 | 8.2 ($2,169, notional) | 1.09 % |
| HD | 320 | 315/345 | $1,545 | 0.94 | 0 / 0 | 6.7 ($2,139, stop) | 1.50 % |
| MSFT | 525 | 520/570 | $2,460 | 1.03 | 0 / 0 | 4.1 ($2,169, notional) | 1.09 % |
| COST | 1,003 | 995/1100 | $4,920 | 1.13 | 0 / 0 | 2.2 ($2,169, notional) | 1.43 % |
| LLY | 1,274 | 1270/1400 | $6,555 | 0.98 | 0 / 0 | 1.7 ($2,169, notional) | 1.37 % |

**Read:** at 5 %, option spreads reach blue chips up to about $270 a share; above ~$300 nothing with reward/risk ≥ 1 fits even at 5 % (the widest-fitting spread at $1,446 found none for HD, GOOG, JPM, UNH, MSFT, CAT, COST, LLY — an at-the-money spread costs about half its width). There, fractional shares are the only structure the account can hold.

### 3.5 Daily run

`python -m trading_agent.quality_lows` (timer 15:40 ET weekdays; holidays exit): universe = watchlist + `QUALITY_LOWS_EXTRA_TICKERS` (default JNJ, LLY, PG, KO, WMT, COST, XOM, CAT, HD, UNH), broad-market ETFs dropped. Sector-ETF drawdowns first, then per ticker `evaluate_ticker` → row appended to `trade_journal/quality_lows.jsonl` (one per ticker per day). Names not near a low skip the fundamentals / estimates / chain calls. `--dry-run` prints and appends nothing; `--no-chains` skips pricing; `--report` prints the latest day.

`QualityLowsConfig` is a frozen dataclass in `quality_lows.py`, deliberately not in `PresetConfig`: the sleeve is outside the spread agent's decision path (the `WheelScreenConfig` / `EvaluatorConfig` precedent). Move it into the preset when §12.6 makes it trade.

### 3.6a News-reading drop-cause label (backlog 12.3)

```python
# trading_agent/news_aggregator.py — added 2026-10-26
def ticker_news(ticker, days=30, max_items=60):
    # fetch_alpaca_news: Alpaca /v1beta1/news (Benzinga, fiscal.ai …; the paper keys),
    #   up to 6 pages of 50, round-ups tagging > 3 symbols dropped
    # fetch_yahoo_rss: feeds.finance.yahoo.com headline RSS (~20 newest)
    # de-duplicated by title slug, then spread_by_day(): round-robin across
    # calendar days so a busy last week does not crowd out the event that started the drop

# trading_agent/drop_cause_llm.py
def classify(row, news_items=None, news_fn=None, runner=subprocess.run, model=MODEL):
    news = _news_payload(news_items)                       # N1…Nk, summaries ≤ 300 chars
    proc = runner(build_cmd(build_prompt(row, news), model), cwd=WORKDIR, ...)
    # claude -p … --tools "" --strict-mcp-config --setting-sources "" --no-session-persistence
    #   --system-prompt SYSTEM_PROMPT --model sonnet
    return {**verify(parsed, ids={N…} | {"DATA"}), "news_count": …, "news_from": …,
            "model": …, "output_tokens": …, "cost_usd_equivalent": …}
    # any failure → {"label": None, "error": "..."} (never raises)

# trading_agent/quality_lows.py
def add_llm_labels(rows, classify, cfg):                  # called in collect() after apply_sector_cap
    todo = sorted(r for r in rows if wants_llm_label(r))  # most advanced stage first
    for r in todo[:cfg.llm_max_names]:
        r["llm_cause"] = classify(r)                       # exceptions → {"label": None, "error": …}
```

`main()` sets `deps.classify` unless `--no-llm` or `QUALITY_LOWS_LLM_CAUSE=false`; `--report` prints `llm=<label>(<confidence>)` and a `why:` line. One-off: `python -m trading_agent.drop_cause_llm NFLX` labels the latest recorded row and prints it (writes nothing). Model: `TRADING_AGENT_DROP_CAUSE_MODEL` (default `sonnet`; empty = the CLI default); the CLI runs in `~/.trading-agent-drop-cause`.

First live dry run, 2026-10-26 (pre-market, 60 items each): **NFLX** rule `sentiment` → Claude `unclear` 0.55 ("results strong … but guidance was cut, engagement and share to YouTube look weak, and next-year EPS estimates are drifting down, so it is not a clean sentiment sell-off"; 13 citations, none invented). **ORCL** rule `sentiment` → Claude `sentiment` 0.60 (−56 % vs XLK −10 % on AI-debt / financing / OpenAI-dependence fears while revisions run 58 up / 11 down; risks: $76B financing need, gross margin 60 % MRQ vs 64 % TTM). About 1–2 minutes per name.

### 3.6 Backtest (backlog §12.4)

`scripts/research/quality_lows_backtest.py` replays `price_features` + `stage` (quality and cause treated as passing) on adjusted daily bars, buys at the next open on the first `turn` / `confirmed`, exits on a close 8 % below the base low, within 3 % of the entry-time 52-week high, below SMA-50 after a +10 % gain, or after 126 sessions; compares with SPY over the same dates.

Result 2026-10-26, 6 years, 38 tickers (36 with enough history), 142 signals:

| | n | mean | median | win |
|---|---|---|---|---|
| Trade return, rule exits | 134 | +7.6 % | +10.7 % | 58 % |
| Excess vs SPY, same dates | 134 | +4.8 % | +1.7 % | 51 % |
| Fixed 126 sessions, excess | 129 | +19.6 % | −3.1 % | 47 % |
| Fixed 252 sessions, excess | 115 | +35.3 % | +3.5 % | 57 % |

Exits: 44 stops, 24 near the high, 34 trailing, 32 time, 8 open; median hold 68 sessions. **Read:** the median trade roughly matches SPY; the mean is lifted by a few large rebounds in names that are on today's watchlist *because* they recovered (IREN, IONQ, SOFI — survivorship). The price signal alone is not an edge. The bet is that the quality + growth + revision filters (which have no free point-in-time history) remove the INTC / NOK / CRM-style value traps — the shadow log tests that going forward.

Quality-only subset (the 26 names that pass today's screen — itself look-ahead): 89 signals; rule exits mean +6.4 %, median +12.1 %, win 60 %; excess vs SPY mean +3.1 %, **median +2.0 %, win 53 %**; fixed 126 sessions median excess −4.1 %, 252 sessions +5.1 % (win 56 %). Slightly better median than the full universe, still small: a 2–3 week base plus a turn does not by itself beat holding SPY by much. Options amplify both the timing error and the cost, which is why §12.5 shadows them on real prices first.

## 4. Edge Cases / Guardrails

- **Shadow only** — no order path, no import of the executor; the rows are evidence for §12.5. Paper trading is §12.6 and an operator decision.
- **Yahoo endpoints are unofficial** — each table fetched separately; a failing one is None; a ticker with nothing usable gives `estimates: null` and, without Schwab growth either, `growth_intact: null` → cause `unclear` (not excluded, never `sentiment`).
- **Schwab zeros** — `revChangeIn`, `epsChangeYear`, `epsChange` read 0.0 on every ticker checked; exact 0.0 for revenue growth and days-to-cover means "not provided" → None. `operatingMargin*` equals net margin on all four tickers checked — not used.
- **Cold fundamentals call** — the data-server client times out at 5 s and a cold Schwab instruments call can take longer (NFLX 2026-10-25); one retry after 1 s.
- **Count of analysts revising vs size of the change** — the size decides; counts are only the fallback when no EPS trend exists (NFLX: 3 up / 10 down, next-year EPS −1.2 % after a 44 % fall → intact).
- **New listings** — fewer than 252 sessions of bars → `excluded`, `missing:price_history` (CBRS, SPCX).
- **Not near a low** — skips the fundamentals, estimates and chain calls (the bulk of the universe).
- **Earnings inside 14 days** — `structure: wait_earnings`, `affordable: null`; a report already out (negative days) does not block.
- **Account size** — on ~$29k the 2 % premium cap ($578) rules out most LEAPS (NFLX Δ 0.75 Dec 2027: $1,840; ORCL: $4,575) and the spread up to the 52-week high (NFLX $717, ORCL $1,993); the budget-fitting spread finds NFLX 65/82.5 Mar 2027 at $569 (reward/risk 2.1); ORCL / AVGO have nothing that fits. The CSP at the base secures $6,400–12,500 — over the 15 % cash cap. `affordable` records this per row.
- **Blue chips above ~$300** — no option structure fits 5 % with reward/risk ≥ 1 (§3.4a); `affordable_alt: null` and only `quotes.shares` is usable. Shares are not in `affordable` on purpose: they always "fit", which would hide whether an option does.
- **Shares sizing** — when the base is far below the price, the 1.5 %-loss-at-stop rule, not the 7.5 % notional, sets the size (`sized_by: stop_risk`); when the price is already below the stop (`stop ≥ spot`) no shares row is written.
- **Fixed-width spread has no reward/risk floor** — near-ATM 10 %-wide spreads price at 0.9–1.2; it ranks after both fitting spreads in `affordable`.
- **Wide LEAPS quotes** — `spread_pct` is journalled (NFLX LEAPS 15.5 % wide on 2026-10-25); judge fills on it before §12.6.
- **Wheel conflict (for §12.6)** — `wheel_lifecycle` reads shares on a ticker with a CSP as "assigned"; the sleeve must not hold or put-sell a ticker the Wheel trades.
- **Duplicate tickers / reruns** — tickers de-duplicated; a ticker already recorded today is skipped unless `--dry-run`.
- **One ticker failing** — logged and skipped; the rest of the run completes.
- **Claude invents a source** — citations are checked against the ids actually sent; invented ones are kept only under `unsupported_citations`, and a label with no valid citation becomes `unclear` (`flags: no_valid_evidence_for:<label>`). This is a code check, not a second model (the `SentimentVerifier` route needs an LLM endpoint the Pi lacks).
- **Claude CLI missing, logged out, rate-limited or slow** — `llm_cause: {"label": null, "error": …}` (timeout 180 s); the row is still written and the stage is unaffected. A Claude login expiry shows up here and in the reviewer at the same time.
- **No news** — Alpaca keys missing or both sources down → `news_count: 0`; Claude labels from DATA alone and usually answers `unclear`.
- **yfinance news is empty** — `yf.Ticker(t).news` returned [] for NFLX on 2026-10-26 (yfinance 1.7.0), which also starves the disabled FinGPT path's Yahoo source; the label uses Alpaca news + Yahoo RSS instead.
- **Round-ups** — Alpaca items tagging more than 3 symbols ("5 stocks investors couldn't stop buzzing about") are dropped.
- **Too many names near lows** — at most `llm_max_names` (10) calls per run, most advanced stage first; the rest get `error: skipped: llm_max_names`.
- **Label vs rule disagree** — expected (NFLX on day one). Neither overrides the other until §12.6; compare them against the rows' later returns.

## 5. Cross-References

- `47_schwab_data_api.md` — `/fundamentals`, `/chain`; the scanner never opens its own Schwab session.
- `40_long_term_options_evaluator.md` — covered calls / CSPs on held stock; this skill is the entry side for names not held.
- `42_portfolio_alert_scheduler.md` — the paused Telegram `long_term` digest; candidate channel for sleeve alerts once its dedup bug is fixed.
- `59_*` / debit spreads — the agent's own call debit spread scoring (`debit_above_fair`) applies when §12.6 trades.
- `56_daily_journal_reviewer.md` — the `claude` CLI call pattern (`claude_bin`, `parse_result_json`) reused here.
- `docs/plans/agent_backlog.md` §12 — plan and status.

---

*Last verified against repo HEAD on 2026-10-26.*
