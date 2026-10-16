# Calibrated probability model — walk-forward results (skill 66, phase 1)

*Generated 2026-10-15 by `scripts/research/calibrate_prob_model.py` on the Pi. Yahoo daily OHLC (adjusted) 2009-01-01 → 2026-10-15, 18 tickers, 362,137 samples (ticker × day × horizon (7, 14, 21, 30, 45)), test months 2013-01 → 2026-10 (166 folds). Samples overlap: n overstates independent evidence, so intervals come from resampling whole months.*

## Variants (out-of-sample log loss, all events pooled; lower is better)

Chosen on the first half of the test period, confirmed on the second.

| Variant | dims | K | refit | log loss 2013–19 | log loss 2020→ | Brier 2020→ |
|---|---|---|---|---|---|---|
| horizon only | horizon | 30 | 1m | 0.5714 | 0.5774 | 0.1957 |
| core | horizon, asset, direction, vol, market | 30 | 1m | 0.5684 | 0.5740 | 0.1942 |
| core+term | horizon, asset, direction, vol, market, term | 30 | 1m | 0.5697 | 0.5751 | 0.1946 |
| core+term+skew | horizon, asset, direction, vol, market, term, skew | 30 | 1m | 0.5710 | 0.5765 | 0.1951 |
| core K=10 | horizon, asset, direction, vol, market | 10 | 1m | 0.5703 | 0.5754 | 0.1947 |
| core K=100 **(chosen)** | horizon, asset, direction, vol, market | 100 | 1m | 0.5677 | 0.5735 | 0.1941 |
| core+term K=100 | horizon, asset, direction, vol, market, term | 100 | 1m | 0.5679 | 0.5737 | 0.1941 |
| pop_rv baseline | — | — | — | 0.5836 | 0.5860 | 0.1993 |

## Chosen variant `core K=100` vs pop_rv, by event (all test months)

ΔBrier = model − pop_rv (negative = model better), 90 % interval from a month-block bootstrap.

| Event | what it means | n | realized | model | +isotonic Brier | pop_rv | Brier model | Brier rv | ΔBrier [90 %] |
|---|---|---|---|---|---|---|---|---|---|
| put_survive_0.50 | bull put / CSP short strike finishes OTM | 301,677 | 76.6% | 76.8% | 0.1798 | 68.3% | 0.1792 | 0.1865 | -0.0073 [-0.0109, -0.0032] |
| put_survive_0.75 | bull put / CSP short strike finishes OTM | 301,677 | 82.7% | 82.7% | 0.1439 | 76.6% | 0.1433 | 0.1473 | -0.0040 [-0.0063, -0.0012] |
| put_survive_1.00 | bull put / CSP short strike finishes OTM | 301,677 | 87.4% | 87.4% | 0.1108 | 83.5% | 0.1104 | 0.1121 | -0.0016 [-0.0029, -0.0001] |
| call_survive_0.50 | bear call short strike finishes OTM | 301,677 | 61.5% | 61.4% | 0.2362 | 70.0% | 0.2353 | 0.2437 | -0.0084 [-0.0123, -0.0039] |
| call_survive_0.75 | bear call short strike finishes OTM | 301,677 | 71.9% | 71.9% | 0.2009 | 78.1% | 0.2002 | 0.2055 | -0.0053 [-0.0079, -0.0025] |
| call_survive_1.00 | bear call short strike finishes OTM | 301,677 | 80.5% | 80.9% | 0.1555 | 84.7% | 0.1549 | 0.1583 | -0.0035 [-0.0051, -0.0017] |
| call_debit_win | call debit: finish ≥ 0.25σ up | 301,677 | 49.4% | 49.7% | 0.2497 | 39.2% | 0.2489 | 0.2605 | -0.0116 [-0.0166, -0.0058] |
| put_debit_win | put debit: finish ≥ 0.25σ down | 301,677 | 31.0% | 30.7% | 0.2146 | 41.1% | 0.2137 | 0.2244 | -0.0107 [-0.0154, -0.0056] |
| condor_1.00 | condor ±1σ: finish inside | 301,677 | 67.9% | 68.2% | 0.2145 | 68.2% | 0.2138 | 0.2181 | -0.0043 [-0.0053, -0.0031] |
| put_touch_1.00 | price touches −1σ before expiry (stop risk, puts) | 301,677 | 29.1% | 29.2% | 0.2058 | 32.9% | 0.2046 | 0.2089 | -0.0043 [-0.0068, -0.0015] |
| call_touch_1.00 | price touches +1σ before expiry (stop risk, calls) | 301,677 | 33.2% | 32.7% | 0.2178 | 30.6% | 0.2169 | 0.2215 | -0.0046 [-0.0065, -0.0026] |

## Against implied odds (SPY ^VIX, QQQ ^VXN, GLD ^GVZ)

| Event | n | realized | implied | pop_rv | model | Brier implied | Brier rv | Brier model |
|---|---|---|---|---|---|---|---|---|
| put_survive_0.50 | 51,768 | 77.9% | 65.1% | 68.5% | 75.3% | 0.1876 | 0.1812 | 0.1732 |
| put_survive_0.75 | 51,768 | 83.6% | 72.2% | 76.8% | 81.5% | 0.1491 | 0.1420 | 0.1378 |
| put_survive_1.00 | 51,768 | 87.8% | 78.4% | 83.7% | 86.6% | 0.1154 | 0.1088 | 0.1074 |
| call_survive_0.50 | 51,768 | 58.5% | 66.8% | 69.8% | 62.4% | 0.2465 | 0.2554 | 0.2410 |
| call_survive_0.75 | 51,768 | 69.3% | 73.7% | 77.9% | 72.8% | 0.2084 | 0.2196 | 0.2101 |
| call_survive_1.00 | 51,768 | 78.4% | 79.6% | 84.6% | 81.6% | 0.1626 | 0.1727 | 0.1667 |
| call_debit_win | 51,768 | 52.3% | 40.9% | 39.4% | 48.4% | 0.2620 | 0.2662 | 0.2485 |
| put_debit_win | 51,768 | 29.1% | 42.7% | 40.9% | 32.3% | 0.2246 | 0.2204 | 0.2074 |
| condor_1.00 | 51,768 | 66.3% | 58.0% | 68.3% | 68.2% | 0.2157 | 0.2240 | 0.2172 |
| put_touch_1.00 | 51,768 | 28.3% | 43.3% | 32.7% | 30.0% | 0.2147 | 0.2054 | 0.2019 |
| call_touch_1.00 | 51,768 | 34.7% | 40.8% | 30.8% | 31.7% | 0.2124 | 0.2274 | 0.2200 |

## `put_survive_0.75` by situation (all test months)

**Direction** — bucket: n · realized · model · pop_rv · Brier model / rv

- `bear`: 55,909 · 82.6% · 84.3% · 76.3% · 0.1442 / 0.1487
- `bull`: 146,794 · 83.5% · 82.7% · 76.6% · 0.1378 / 0.1425
- `mr`: 1,333 · 87.5% · 83.3% · 76.5% · 0.1109 / 0.1217
- `na`: 154 · 82.5% · 83.9% · 75.9% · 0.1449 / 0.1486
- `side`: 97,487 · 81.4% · 81.6% · 76.7% · 0.1516 / 0.1541

**Market regime** — bucket: n · realized · model · pop_rv · Brier model / rv

- `caution`: 87,086 · 83.3% · 83.2% · 76.6% · 0.1392 / 0.1439
- `normal`: 164,606 · 82.0% · 81.7% · 76.7% · 0.1477 / 0.1506
- `recovery`: 10,142 · 84.7% · 83.2% · 76.4% · 0.1311 / 0.1366
- `stress`: 39,843 · 83.6% · 85.0% · 76.1% · 0.1375 / 0.1439

**Vol regime** — bucket: n · realized · model · pop_rv · Brier model / rv

- `high`: 103,141 · 85.7% · 84.7% · 76.3% · 0.1235 / 0.1318
- `low`: 108,114 · 79.0% · 80.2% · 76.8% · 0.1660 / 0.1662
- `mid`: 90,422 · 83.5% · 83.3% · 76.7% · 0.1388 / 0.1424

**Term structure** — bucket: n · realized · model · pop_rv · Brier model / rv

- `contango`: 180,676 · 81.6% · 82.0% · 76.7% · 0.1503 / 0.1527
- `flat`: 99,333 · 83.8% · 83.4% · 76.5% · 0.1363 / 0.1415
- `inverted`: 21,668 · 86.4% · 84.8% · 76.2% · 0.1176 / 0.1292

**DTE** — bucket: n · realized · model · pop_rv · Brier model / rv

- `14`: 61,168 · 82.1% · 82.3% · 76.7% · 0.1471 / 0.1501
- `21`: 60,557 · 82.7% · 82.9% · 76.6% · 0.1429 / 0.1468
- `30`: 59,715 · 83.0% · 83.1% · 76.5% · 0.1412 / 0.1457
- `45`: 58,449 · 84.5% · 84.1% · 76.3% · 0.1315 / 0.1380
- `7`: 61,788 · 81.1% · 81.0% · 76.9% · 0.1533 / 0.1553

**Asset group** — bucket: n · realized · model · pop_rv · Brier model / rv

- `alt`: 51,768 · 79.0% · 79.3% · 76.5% · 0.1662 / 0.1669
- `broad`: 86,280 · 83.3% · 82.7% · 76.7% · 0.1400 / 0.1439
- `sector`: 138,048 · 83.2% · 83.4% · 76.6% · 0.1400 / 0.1444
- `single`: 25,581 · 85.2% · 85.3% · 76.2% · 0.1264 / 0.1348


## `call_survive_0.75` by situation (all test months)

**Direction** — bucket: n · realized · model · pop_rv · Brier model / rv

- `bear`: 55,909 · 72.9% · 73.0% · 78.4% · 0.1977 / 0.2004
- `bull`: 146,794 · 72.9% · 72.8% · 78.0% · 0.1960 / 0.2001
- `mr`: 1,333 · 63.3% · 71.1% · 78.1% · 0.2361 / 0.2539
- `na`: 154 · 91.6% · 72.3% · 78.7% · 0.1193 / 0.0933
- `side`: 97,487 · 70.0% · 69.9% · 77.9% · 0.2075 / 0.2161

**Market regime** — bucket: n · realized · model · pop_rv · Brier model / rv

- `caution`: 87,086 · 67.3% · 70.0% · 78.0% · 0.2188 / 0.2310
- `normal`: 164,606 · 74.1% · 72.5% · 77.9% · 0.1910 / 0.1930
- `recovery`: 10,142 · 74.7% · 72.1% · 78.3% · 0.1857 / 0.1899
- `stress`: 39,843 · 72.0% · 73.4% · 78.5% · 0.2012 / 0.2052

**Vol regime** — bucket: n · realized · model · pop_rv · Brier model / rv

- `high`: 103,141 · 77.0% · 74.3% · 78.3% · 0.1773 / 0.1770
- `low`: 108,114 · 67.5% · 69.1% · 77.8% · 0.2189 / 0.2298
- `mid`: 90,422 · 71.3% · 72.5% · 78.0% · 0.2039 / 0.2091

**Term structure** — bucket: n · realized · model · pop_rv · Brier model / rv

- `contango`: 180,676 · 74.1% · 72.0% · 78.0% · 0.1911 / 0.1931
- `flat`: 99,333 · 69.1% · 71.6% · 78.1% · 0.2116 / 0.2212
- `inverted`: 21,668 · 66.3% · 72.3% · 78.5% · 0.2235 / 0.2369

**DTE** — bucket: n · realized · model · pop_rv · Brier model / rv

- `14`: 61,168 · 73.4% · 73.3% · 77.9% · 0.1941 / 0.1971
- `21`: 60,557 · 72.0% · 72.1% · 78.1% · 0.2002 / 0.2050
- `30`: 59,715 · 70.4% · 70.1% · 78.2% · 0.2072 / 0.2142
- `45`: 58,449 · 69.0% · 69.3% · 78.4% · 0.2116 / 0.2219
- `7`: 61,788 · 74.6% · 74.5% · 77.8% · 0.1885 / 0.1904

**Asset group** — bucket: n · realized · model · pop_rv · Brier model / rv

- `alt`: 51,768 · 76.2% · 76.1% · 78.1% · 0.1819 / 0.1818
- `broad`: 86,280 · 70.4% · 70.9% · 77.9% · 0.2060 / 0.2138
- `sector`: 138,048 · 70.6% · 70.6% · 78.0% · 0.2065 / 0.2130
- `single`: 25,581 · 75.5% · 74.1% · 78.5% · 0.1834 / 0.1851


## `put_debit_win` by situation (all test months)

**Direction** — bucket: n · realized · model · pop_rv · Brier model / rv

- `bear`: 55,909 · 30.9% · 29.2% · 41.5% · 0.2143 / 0.2259
- `bull`: 146,794 · 30.4% · 31.0% · 41.0% · 0.2117 / 0.2231
- `mr`: 1,333 · 25.1% · 30.1% · 41.2% · 0.1895 / 0.2140
- `na`: 154 · 53.2% · 32.2% · 41.9% · 0.3066 / 0.2600
- `side`: 97,487 · 32.0% · 31.2% · 40.9% · 0.2165 / 0.2257

**Market regime** — bucket: n · realized · model · pop_rv · Brier model / rv

- `caution`: 87,086 · 28.8% · 29.4% · 41.0% · 0.2042 / 0.2201
- `normal`: 164,606 · 32.7% · 32.1% · 40.9% · 0.2202 / 0.2269
- `recovery`: 10,142 · 30.4% · 30.1% · 41.3% · 0.2134 / 0.2234
- `stress`: 39,843 · 29.1% · 28.4% · 41.7% · 0.2075 / 0.2238

**Vol regime** — bucket: n · realized · model · pop_rv · Brier model / rv

- `high`: 103,141 · 28.4% · 28.8% · 41.5% · 0.2041 / 0.2210
- `low`: 108,114 · 34.1% · 33.0% · 40.8% · 0.2245 / 0.2293
- `mid`: 90,422 · 30.2% · 30.2% · 41.0% · 0.2117 / 0.2225

**Term structure** — bucket: n · realized · model · pop_rv · Brier model / rv

- `contango`: 180,676 · 33.0% · 31.6% · 41.0% · 0.2218 / 0.2276
- `flat`: 99,333 · 28.8% · 29.8% · 41.2% · 0.2052 / 0.2205
- `inverted`: 21,668 · 24.4% · 28.4% · 41.6% · 0.1847 / 0.2159

**DTE** — bucket: n · realized · model · pop_rv · Brier model / rv

- `14`: 61,168 · 32.0% · 31.7% · 40.9% · 0.2179 / 0.2258
- `21`: 60,557 · 30.8% · 30.4% · 41.1% · 0.2130 / 0.2239
- `30`: 59,715 · 30.1% · 29.8% · 41.2% · 0.2105 / 0.2232
- `45`: 58,449 · 28.3% · 28.1% · 41.5% · 0.2031 / 0.2207
- `7`: 61,788 · 33.6% · 33.5% · 40.7% · 0.2235 / 0.2283

**Asset group** — bucket: n · realized · model · pop_rv · Brier model / rv

- `alt`: 51,768 · 38.0% · 37.2% · 41.2% · 0.2375 / 0.2369
- `broad`: 86,280 · 29.1% · 29.7% · 40.9% · 0.2075 / 0.2207
- `sector`: 138,048 · 29.9% · 29.2% · 41.0% · 0.2096 / 0.2224
- `single`: 25,581 · 29.4% · 29.1% · 41.6% · 0.2088 / 0.2231


## `call_debit_win` by situation (all test months)

**Direction** — bucket: n · realized · model · pop_rv · Brier model / rv

- `bear`: 55,909 · 49.7% · 50.0% · 38.8% · 0.2504 / 0.2626
- `bull`: 146,794 · 49.1% · 49.1% · 39.3% · 0.2491 / 0.2596
- `mr`: 1,333 · 56.4% · 50.5% · 39.1% · 0.2461 / 0.2760
- `na`: 154 · 22.1% · 48.2% · 38.3% · 0.2608 / 0.1966
- `side`: 97,487 · 49.7% · 50.5% · 39.3% · 0.2477 / 0.2605

**Market regime** — bucket: n · realized · model · pop_rv · Brier model / rv

- `caution`: 87,086 · 53.8% · 51.8% · 39.2% · 0.2472 / 0.2696
- `normal`: 164,606 · 46.7% · 48.4% · 39.4% · 0.2494 / 0.2542
- `recovery`: 10,142 · 48.3% · 50.3% · 38.9% · 0.2480 / 0.2577
- `stress`: 39,843 · 51.4% · 50.3% · 38.6% · 0.2505 / 0.2672

**Vol regime** — bucket: n · realized · model · pop_rv · Brier model / rv

- `high`: 103,141 · 48.5% · 49.6% · 38.8% · 0.2485 / 0.2595
- `low`: 108,114 · 49.7% · 49.6% · 39.5% · 0.2486 / 0.2604
- `mid`: 90,422 · 50.2% · 50.0% · 39.3% · 0.2496 / 0.2618

**Term structure** — bucket: n · realized · model · pop_rv · Brier model / rv

- `contango`: 180,676 · 46.5% · 49.1% · 39.3% · 0.2496 / 0.2539
- `flat`: 99,333 · 53.0% · 50.6% · 39.1% · 0.2482 / 0.2684
- `inverted`: 21,668 · 57.1% · 50.9% · 38.7% · 0.2461 / 0.2799

**DTE** — bucket: n · realized · model · pop_rv · Brier model / rv

- `14`: 61,168 · 47.7% · 47.9% · 39.4% · 0.2490 / 0.2564
- `21`: 60,557 · 49.3% · 49.5% · 39.2% · 0.2497 / 0.2602
- `30`: 59,715 · 51.2% · 51.6% · 39.0% · 0.2492 / 0.2646
- `45`: 58,449 · 53.2% · 53.6% · 38.8% · 0.2479 / 0.2696
- `7`: 61,788 · 46.0% · 46.1% · 39.6% · 0.2484 / 0.2524

**Asset group** — bucket: n · realized · model · pop_rv · Brier model / rv

- `alt`: 51,768 · 41.3% · 42.5% · 39.1% · 0.2451 / 0.2429
- `broad`: 86,280 · 52.3% · 51.5% · 39.3% · 0.2495 / 0.2665
- `sector`: 138,048 · 50.9% · 51.4% · 39.2% · 0.2498 / 0.2638
- `single`: 25,581 · 48.0% · 49.2% · 38.7% · 0.2494 / 0.2581

## Caveats

- Outcomes are expiry / touch events on the underlying, not P&L with the agent's exits; the live shadow (phase 3) is the check on that.
- 2010–2026 is mostly a rising market; drift shows up as put-side edge. Walk-forward shows how stable it was, not that it continues.
- Per-ticker skew has no free history; ^SKEW (S&P 500) is the only skew input here. The daily `vol_surface.jsonl` snapshot (phase 2) collects per-ticker skew from 2026-10-16.
- Earnings windows are excluded for META / AMZN only where Yahoo lists the dates (2012 →).

<!-- findings: hand-written, kept when the report is regenerated -->

## Findings (2026-10-16)

1. **The model is better calibrated than both baselines, out of sample.** Pooled log loss 0.5735 vs 0.5860 for `pop_rv` on 2020 → months, after choosing the variant on 2013–2019. It beats `pop_rv` on all 11 events with month-block 90 % intervals below zero, and implied odds (SPY / QQQ / GLD) on puts, debits and touches.
2. **Most of the improvement is not situational.** Horizon-only (drift and fat tails, no situation) already takes log loss from 0.5836 to 0.5714; the five situation dimensions add 0.5714 → 0.5677. Term structure and the SKEW index added nothing and were dropped.
3. **Implied odds still win on the call side** (call survival 0.75σ / 1σ and condors on SPY / QQQ / GLD). Implied vol prices call skew; realized vol does not. Per-ticker skew from `vol_surface.jsonl` is the candidate fix once months of data exist; an IV/RV ratio feature is another.
4. **Volatility regime is the situation that matters most.** Short put at 0.75σ survives 85.7 % in high-vol regimes vs 79.0 % in low; short call 77.0 % vs 67.5 %. Selling calls in quiet markets was the worst case in the sample.
5. **Trend direction barely moved the odds.** Short put 0.75σ: 82.6 % in bearish regimes vs 83.5 % bullish; short call 72.9 % in both. Sideways was slightly worse for both sides.
6. **Put debits rarely paid, in every regime.** A finish ≥ 0.25σ lower (about a 30-DTE put debit's breakeven) happened 31 % of the time overall, 30.9 % in bearish regimes, 28.8 % in CAUTION, 29.1 % in stress — against `pop_rv`'s 41 % and implied 43 %. Call debits: 49 % vs `pop_rv` 39 %. Since bear calls were switched off (2026-10-14) the agent's only bearish trade is a put debit, so this deserves a look before more capital goes there (backlog B3 / E-series).
7. **Caveats:** expiry and touch events on the underlying, not trade P&L with exits; one mostly-rising era; overlapping samples (intervals resample whole months).
