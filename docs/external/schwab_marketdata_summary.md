# Schwab Trader API — Market Data v1 — Field Reference

Captured from the official OpenAPI 3.0.3 spec (Market Data Production)
on 2026-05-05.  This summary exists so future contributors don't have
to re-fetch the swagger to understand the field-mapping decisions in
`trading_agent/market_data_schwab.py`.

## Base URL & Auth

- Base: `https://api.schwabapi.com/marketdata/v1`
- OAuth 2.0 authorization-code flow:
  - Authorize: `https://api.schwabapi.com/v1/oauth/authorize`
  - Token: `https://api.schwabapi.com/v1/oauth/token`
- Access tokens last **30 minutes**; refresh tokens last **7 days**
  (absolute, NOT rolling — user re-auths weekly).
- Header on every request: `Authorization: Bearer <access_token>`.
- Schwab also returns a `Schwab-Client-CorrelId` UUID in every response;
  log it for support tickets.

## Endpoints used by the agent

| Path | Used for |
|------|----------|
| `GET /chains` | Option-chain snapshot for spread scanner |
| `GET /quotes?symbols=...` | Pre-submit live-quote refresh of leg symbols + spot snapshots |
| `GET /pricehistory` | Intraday bars for the regime layer + watchlist |
| `GET /expirationchain` | (Optional) DTE list for adaptive scan |
| `GET /markets` | Market hours / is-open check |

## Symbol format quirk

Schwab option symbols are **space-padded** to 21 characters:

```
AMZN  220617C03170000      ← Schwab (root padded to 6 chars)
AMZN220617C03170000        ← Alpaca / standard OCC compact
```

Both refer to AMZN 2022-06-17 $3170 Call.  The adapter exposes
`_to_schwab_symbol()` / `_from_schwab_symbol()` translators so the
agent always speaks Alpaca-compact OCC.

## OptionChain response shape

```
{
  "symbol": "AAPL",
  "underlying": { ... },
  "callExpDateMap": {
    "2026-05-29:24": {                    ← key = "YYYY-MM-DD:DTE"
      "150.0": [                          ← key = strike string
        {                                  ← OptionContract object
          "putCall": "CALL",
          "symbol": "AAPL  260529C00150000",
          "bidPrice": 1.20, "askPrice": 1.25, "markPrice": 1.225,
          "bidSize": 100, "askSize": 100, "lastSize": 0,
          "delta": 0.31, "gamma": 0.04,
          "theta": -0.05, "vega": 0.12, "rho": 0.02,
          "volatility": 22.3,             ← IV in %, NOT decimal
          "openInterest": 5421,
          "totalVolume": 234,
          "expirationDate": "2026-05-29",
          "daysToExpiration": 24,
          "strikePrice": 150.0,
          "isInTheMoney": false,
          "isPennyPilot": true,
          ...
        }
      ]
    }
  },
  "putExpDateMap": { ... same shape ... }
}
```

Watch-outs:
- `volatility` is given as a percent (22.3 means 22.3%, not 0.223).
- `delta` for puts is positive in Schwab's response on some endpoints
  but signed on others; treat `abs(delta)` as the working value and
  keep the sign internally where it matters.
- A strike key can map to a list of >1 contracts when there are
  weeklies + standards on the same date (rare for liquid ETFs but
  possible for indices); take the one matching `expirationType=S`
  (standard) when filtering.

## Quote response shape (multi-symbol)

```
GET /quotes?symbols=XLF,AAPL  260529C00150000

{
  "AAPL  260529C00150000": {
    "assetMainType": "OPTION",
    "symbol": "AAPL  260529C00150000",
    "quote": {
      "bidPrice": 1.20, "askPrice": 1.25, "mark": 1.225,
      "bidSize": 100, "askSize": 100,
      "delta": 0.31, "gamma": 0.04, "theta": -0.05,
      "vega": 0.12, "rho": 0.02,
      "volatility": 22.3,
      "openInterest": 5421,
      "totalVolume": 234,
      "underlyingPrice": 168.4
    },
    "reference": {
      "contractType": "C",
      "strikePrice": 150.0,
      "expirationDay": 29, "expirationMonth": 5, "expirationYear": 2026,
      "isPennyPilot": true,
      "underlying": "AAPL"
    }
  },
  "XLF": {
    "assetMainType": "EQUITY",
    "symbol": "XLF",
    "quote": {
      "bidPrice": 51.49, "askPrice": 51.50, "lastPrice": 51.495,
      "mark": 51.495, "bidSize": 800, "askSize": 600,
      ...
    }
  }
}
```

The keys on the OUTER dict are exactly the symbols you requested,
including the spaces in option symbols — match-back has to honor
that exactly.

## PriceHistory response shape

```
GET /pricehistory?symbol=XLF&periodType=day&period=10
                  &frequencyType=minute&frequency=5

{
  "symbol": "XLF",
  "candles": [
    { "datetime": 1639137600000,         ← epoch milliseconds, UTC
      "open": 51.01, "high": 51.15,
      "low": 51.00,  "close": 51.04,
      "volume": 10719 },
    ...
  ],
  "previousClose": 50.87,
  "previousCloseDate": 1639029600000,
  "empty": false
}
```

Period / frequency rules (from QueryParamperiod docstring):
- `periodType=day` → period ∈ {1,2,3,4,5,10}, frequencyType=minute, frequency ∈ {1,5,10,15,30}
- `periodType=month` → period ∈ {1,2,3,6}, frequencyType ∈ {daily, weekly}
- `periodType=year` → period ∈ {1,2,3,5,10,15,20}, frequencyType ∈ {daily, weekly, monthly}
- `periodType=ytd` → period=1, frequencyType ∈ {daily, weekly}

Common agent recipes:
- 5-min bars over 5 days: `periodType=day&period=5&frequencyType=minute&frequency=5`
- Daily bars over 200 days: `periodType=year&period=1&frequencyType=daily&frequency=1`
  (then truncate to last 200 closes — Schwab returns the whole year)

## Error responses

All errors return the same shape:

```
{
  "errors": [
    { "id": "uuid", "status": "401",
      "title": "Unauthorized",
      "detail": "...",
      "source": { "header": "Authorization" } }
  ]
}
```

401 → access token expired, refresh and retry once.
429 not in the swagger but Schwab docs say 120 req/min — back off.

## Rate limits

- Officially documented: **120 requests / minute** across the market
  data API.  No per-endpoint sub-budget.
- The agent's 5-min cycle with ~6 tickers × (1 chain + 1 spot + 1 bars
  + 1 quote refresh) = ~24 calls per cycle — 12% utilisation.
- Streamlit watchlist tab can spike if user opens it while a cycle is
  running.  Adapter should rate-limit at the client level just to be
  safe.

## OCC symbol regex

Compact (Alpaca-style):
```
^([A-Z]{1,6})(\d{6})([CP])(\d{8})$
```

Schwab (space-padded to 6 root chars):
```
^([A-Z]{1,6}) *(\d{6})([CP])(\d{8})$
```

Translator pattern: parse with the compact regex, re-emit padded to
6+6+1+8=21 chars when calling Schwab.

---
*Last verified against repo HEAD on 2026-05-05.*
