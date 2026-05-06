"""
market_data_schwab.py — MarketDataPort adapter for Schwab Trader API
=====================================================================

Why this exists
---------------
Alpaca's free/basic feed (``ALPACA_OPTIONS_FEED=indicative``) is
**15 minutes delayed**.  When the executor refreshes a limit price
against that delayed feed and shaves one tick off mid, the resulting
limit lands below where real-time market makers will fill — orders
sit on the book until the 15-min stale-cancel timer wipes them.

Schwab provides real-time options + equity quotes free of charge to
holders of a Schwab brokerage account through the Trader API.  Plugging
Schwab in for *market data only* (Alpaca still does paper execution)
gives the agent live quotes without a $99/mo Alpaca subscription.

What this adapter does
----------------------
``SchwabMarketDataProvider`` is a drop-in replacement for the
network-bound parts of ``MarketDataProvider``.  It subclasses the
Alpaca provider so we inherit:

  * ``fetch_historical_prices`` / ``prefetch_historical_parallel``
    (yfinance — vendor-neutral, no need to re-implement against
    Schwab's pricehistory endpoint for daily history)
  * All ``compute_*`` indicator helpers (pure math)
  * The cache shapes and constructor signature, so callers can swap
    providers without touching the agent core.

It overrides the methods that hit Alpaca's data plane:

  * ``fetch_batch_snapshots`` / ``get_current_price`` / ``get_cached_price``
  * ``get_underlying_bid_ask``
  * ``get_5min_return``
  * ``fetch_intraday_bars`` (used by Watchlist tab + regime layer)
  * ``fetch_option_chain``  (the meat — drives Phase III planning)
  * ``fetch_option_quotes`` (the pre-submission refresh)

Symbol-format translator
------------------------
Schwab option symbols are space-padded to 21 chars:
``"AMZN  220617C03170000"`` (root padded to 6).  The rest of the agent
(executor, journal, position monitor) speaks compact OCC:
``"AMZN220617C03170000"``.  Translation happens at the adapter
boundary so the agent never sees a Schwab-format symbol.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from trading_agent.market_data import (
    ALPACA_TIMEOUT,
    ALPACA_TIMEOUT_LONG,
    INTRADAY_BARS_TTL,
    INTRADAY_RETURN_TTL,
    OPTION_CHAIN_TTL,
    SNAPSHOT_TTL,
    MarketDataProvider,
    _truncate_json,
)
from trading_agent.schwab_oauth import SchwabOAuth

logger = logging.getLogger(__name__)

# Schwab Market Data v1 base URL.
SCHWAB_BASE_URL = "https://api.schwabapi.com/marketdata/v1"

# OCC compact form regex — what the rest of the agent uses.
#   AAPL 250101P00150000  → root + YYMMDD + C/P + strike*1000 (8 digits)
_OCC_COMPACT_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")

# Schwab form regex — root padded to 6 with spaces.
_OCC_SCHWAB_RE = re.compile(r"^([A-Z]{1,6}) *(\d{6})([CP])(\d{8})$")


# ---------------------------------------------------------------------------
# Symbol translators
# ---------------------------------------------------------------------------
def to_schwab_symbol(occ: str) -> str:
    """
    Compact OCC → Schwab-padded OCC.  ``"AMZN220617C03170000"`` →
    ``"AMZN  220617C03170000"`` (root padded to 6 chars).

    Pass-through for non-option symbols (equities, indices like ``$SPX``,
    futures with leading ``/``).
    """
    m = _OCC_COMPACT_RE.match(occ or "")
    if not m:
        return occ
    root, ymd, cp, strike = m.groups()
    return f"{root:<6}{ymd}{cp}{strike}"   # ljust to 6


def from_schwab_symbol(schwab: str) -> str:
    """
    Schwab-padded OCC → compact OCC.  Inverse of :func:`to_schwab_symbol`.
    Pass-through if the input doesn't look like an option symbol.
    """
    m = _OCC_SCHWAB_RE.match(schwab or "")
    if not m:
        return (schwab or "").strip()
    root, ymd, cp, strike = m.groups()
    return f"{root}{ymd}{cp}{strike}"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class SchwabMarketDataProvider(MarketDataProvider):
    """
    MarketDataPort implementation backed by Schwab's Trader API.

    Construction
    ------------
    Needs (a) Schwab OAuth credentials for market-data calls and
    (b) Alpaca credentials for the *non*-market-data primitives the
    parent class still owns (account-port methods, backwards-compat
    methods that are never reached when this adapter is wired but kept
    valid for type sanity).  The simplest path is to leave the existing
    Alpaca creds in ``.env`` alongside the new Schwab creds and let the
    adapter ignore them.
    """

    def __init__(self, *,
                 schwab_oauth: Optional[SchwabOAuth] = None,
                 alpaca_api_key: str = "",
                 alpaca_secret_key: str = "",
                 alpaca_data_url: str = "https://data.alpaca.markets/v2",
                 alpaca_base_url: str = "https://paper-api.alpaca.markets/v2",
                 base_url: str = SCHWAB_BASE_URL):
        super().__init__(
            alpaca_api_key=alpaca_api_key,
            alpaca_secret_key=alpaca_secret_key,
            alpaca_data_url=alpaca_data_url,
            alpaca_base_url=alpaca_base_url,
        )
        self.oauth = schwab_oauth or SchwabOAuth.from_env()
        self.base_url = base_url
        # Reuse a Session so connection pooling cuts a few ms off cycle hot-paths.
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # HTTP helper — handles auth header injection + 401 single-retry
    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.oauth.get_access_token()}",
            "Accept": "application/json",
        }

    def _get(self, path: str, params: Optional[Dict] = None,
             timeout=ALPACA_TIMEOUT_LONG) -> Optional[dict]:
        """
        Authenticated GET against the Schwab marketdata API.  Returns the
        decoded JSON body on success, ``None`` on transient failure.

        On 401, refreshes the token once and retries.  Any other HTTP error
        is logged at WARNING and returns ``None`` — the caller decides how
        to degrade.  We deliberately do NOT raise: the agent's cycle should
        not crash on a momentary Schwab outage.

        Missing-token RuntimeError is caught here too: an unauthenticated
        agent should silently no-op (returning ``None``) rather than crash
        every cycle.  The first occurrence is logged at WARNING with the
        exact CLI command to fix it; subsequent ones drop to DEBUG so the
        log doesn't get hammered until the operator runs the login flow.
        """
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.get(
                url, headers=self._headers(), params=params, timeout=timeout,
            )
        except requests.RequestException as exc:
            logger.warning("Schwab GET %s failed (%s) — params=%s",
                           path, exc, params)
            return None
        except RuntimeError as exc:
            # Raised by SchwabOAuth.get_access_token() when no token cache
            # exists or the refresh token has expired.  This is operator
            # action territory, not a transient outage — surface it loudly
            # ONCE, then quiet so the log doesn't spam every 5-min cycle.
            if not getattr(self, "_auth_warned", False):
                logger.warning(
                    "Schwab API call to %s aborted — %s "
                    "Run `python -m trading_agent.schwab_oauth login` "
                    "in the project root to perform the one-time "
                    "authorization-code exchange. Until then, every "
                    "Schwab call will return None and surfaces using "
                    "MARKET_DATA_PROVIDER=schwab will see no data.",
                    path, exc,
                )
                self._auth_warned = True
            else:
                logger.debug("Schwab GET %s skipped — no auth (%s)", path, exc)
            return None

        # Token expired mid-flight: refresh and retry once.
        if resp.status_code == 401:
            logger.info("Schwab GET %s returned 401 — forcing token refresh "
                        "and retrying once.", path)
            # Clear cached tokens so get_access_token() forces a refresh.
            self.oauth._tokens = None     # noqa: SLF001 — intentional
            try:
                resp = self._session.get(
                    url, headers=self._headers(), params=params, timeout=timeout,
                )
            except requests.RequestException as exc:
                logger.warning("Schwab GET %s retry failed (%s)", path, exc)
                return None

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                logger.warning("Schwab GET %s returned 200 but non-JSON body: %s",
                               path, _truncate_json(resp.text))
                return None

        # 4xx/5xx — log the correlation id Schwab sends back so support
        # tickets are productive.
        correl = resp.headers.get("Schwab-Client-CorrelId", "")
        logger.warning(
            "Schwab GET %s → HTTP %d (correl_id=%s, params=%s, body=%s)",
            path, resp.status_code, correl, params,
            _truncate_json(resp.text),
        )
        return None

    # ------------------------------------------------------------------
    # Snapshots — equity / ETF / index spot prices
    # ------------------------------------------------------------------
    def fetch_batch_snapshots(self, tickers: List[str]) -> Dict[str, float]:
        """
        Multi-symbol spot price fetch via /quotes.  Returns a dict
        ticker → last (or mark) price.  Same TTL as the parent class
        (60 s).
        """
        if not tickers:
            return {}
        # Honor the parent's snapshot cache so cycles still benefit from
        # de-duplication across the SPY/QQQ benchmark calls.
        now = time.monotonic()
        out: Dict[str, float] = {}
        misses: List[str] = []
        for t in tickers:
            cached = self._snapshot_cache.get(t)
            if cached and (now - cached[1]) < SNAPSHOT_TTL:
                out[t] = cached[0]
            else:
                misses.append(t)
        if not misses:
            return out

        body = self._get(
            "/quotes",
            params={"symbols": ",".join(misses), "fields": "quote,reference"},
        )
        if not isinstance(body, dict):
            return out

        for sym, payload in body.items():
            if not isinstance(payload, dict):
                continue
            q = payload.get("quote") or {}
            # Last trade is the most-stable single-number proxy for spot;
            # fall back to mark, then to mid of bid/ask.
            price = (q.get("lastPrice")
                     or q.get("mark")
                     or self._mid_or_zero(q.get("bidPrice"), q.get("askPrice")))
            if price:
                out[sym] = float(price)
                self._snapshot_cache[sym] = (float(price), now)
        # Some symbols (rare names with the indicative=true ETF flag) come
        # back keyed differently; we only return what Schwab actually sent.
        return out

    def get_current_price(self, ticker: str) -> float:
        """Single-ticker spot price.  Delegates to batch with one symbol."""
        prices = self.fetch_batch_snapshots([ticker])
        return float(prices.get(ticker, 0.0))

    def get_underlying_bid_ask(self, ticker: str) -> Optional[Tuple[float, float]]:
        """Bid/ask for the underlying — used by the liquidity guardrail."""
        body = self._get(
            "/quotes",
            params={"symbols": ticker, "fields": "quote"},
            timeout=ALPACA_TIMEOUT,
        )
        if not isinstance(body, dict):
            return None
        payload = body.get(ticker)
        if not isinstance(payload, dict):
            return None
        q = payload.get("quote") or {}
        bid = q.get("bidPrice")
        ask = q.get("askPrice")
        if bid is None or ask is None:
            return None
        return float(bid), float(ask)

    @staticmethod
    def _mid_or_zero(bid, ask) -> float:
        try:
            b, a = float(bid or 0), float(ask or 0)
            return (b + a) / 2 if (b > 0 and a > 0) else 0.0
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------
    # Intraday bars — 5-min / 15-min / 1h / 4h
    # ------------------------------------------------------------------
    # Mapping from agent-level interval string → Schwab pricehistory params.
    # ``periodType=day`` only allows ``frequencyType=minute``, so 60m/1h/4h
    # need to live under ``periodType=year`` with frequencyType=daily — but
    # that loses intraday granularity.  Schwab's documented sweet spots are
    # 1/5/10/15/30 minute resolution under day, so we synthesize 1h from 30m
    # bars and 4h from 30m bars.  See the trade-offs in the module README.
    _INTRADAY_PARAMS: Dict[str, Dict[str, object]] = {
        "5m":  {"periodType": "day", "period": 5,
                "frequencyType": "minute", "frequency": 5},
        "15m": {"periodType": "day", "period": 10,
                "frequencyType": "minute", "frequency": 15},
        "30m": {"periodType": "day", "period": 10,
                "frequencyType": "minute", "frequency": 30},
        "60m": {"periodType": "day", "period": 10,
                "frequencyType": "minute", "frequency": 30},
        "1h":  {"periodType": "day", "period": 10,
                "frequencyType": "minute", "frequency": 30},
        "4h":  {"periodType": "day", "period": 10,
                "frequencyType": "minute", "frequency": 30},
    }

    def fetch_intraday_bars(self, ticker: str, interval: str = "5m",
                            lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        Intraday OHLCV for *ticker* at *interval*.  Schwab returns data
        at 1/5/10/15/30-minute resolution under ``periodType=day``;
        coarser intervals (60m / 1h / 4h) are synthesised by resampling
        the 30-min stream so the regime classifier and watchlist tab
        get a consistent shape across all timeframes.

        Returns a DataFrame indexed by UTC timestamp with columns
        ``["Open", "High", "Low", "Close", "Volume"]`` — same shape the
        Alpaca / yfinance providers return.
        """
        cache_key = (ticker, interval)
        now = time.monotonic()
        cached = self._intraday_bars_cache.get(cache_key)
        if cached and (now - cached[1]) < INTRADAY_BARS_TTL:
            return cached[0]

        params = dict(self._INTRADAY_PARAMS.get(interval) or self._INTRADAY_PARAMS["5m"])
        params["symbol"] = ticker
        # Schwab's query-string parser is strict about boolean shape: it
        # accepts only the lowercase string "true"/"false", not Python's
        # repr "False" (which is what `requests` emits when you pass a
        # bare bool).  Always pass as a lowercase string.
        params["needExtendedHoursData"] = "false"
        if lookback_days and lookback_days <= 10:
            params["period"] = lookback_days

        body = self._get("/pricehistory", params=params)
        if not isinstance(body, dict):
            return pd.DataFrame()

        candles = body.get("candles") or []
        if not candles:
            # No bars returned — could be a delisted symbol, a bad
            # period/frequency combo, or pre-market on a holiday.  Log
            # at WARNING with the params so the user can see what was
            # requested; without this they get a silent empty DataFrame
            # and the watchlist's "0/5 ok" with no clue why.
            logger.warning(
                "[%s] Schwab /pricehistory returned 0 bars for %s "
                "(params=%s). Empty `candles` array means either the "
                "symbol is unavailable on Schwab, the period/frequency "
                "combo is invalid, or extended-hours filtering "
                "stripped everything.",
                ticker, interval, params,
            )
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        df["Datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
        df = df.set_index("Datetime").rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })[["Open", "High", "Low", "Close", "Volume"]]

        # Synthesize coarser intervals via OHLCV resample so callers
        # asking for "1h" / "4h" get the same shape they'd get from
        # yfinance.
        if interval in ("60m", "1h"):
            df = self._resample_ohlcv(df, "60min")
        elif interval == "4h":
            df = self._resample_ohlcv(df, "240min")

        self._intraday_bars_cache[cache_key] = (df, now)
        return df

    @staticmethod
    def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        if df.empty:
            return df
        agg = df.resample(rule, label="right", closed="right").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna(how="any")
        return agg

    def get_5min_return(self, ticker: str) -> Optional[float]:
        """
        Last completed 5-min bar's return (close/prev_close − 1).  Used
        by the regime layer's leadership-z and 5-min momentum logic.
        Cached for 60 s to dedupe SPY/QQQ benchmark fetches.
        """
        cached = self._intraday_return_cache.get(ticker)
        now = time.monotonic()
        if cached and (now - cached[1]) < INTRADAY_RETURN_TTL:
            return cached[0]
        df = self.fetch_intraday_bars(ticker, "5m", lookback_days=1)
        if df.empty or len(df) < 2:
            return None
        last_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        if prev_close <= 0:
            return None
        ret = (last_close / prev_close) - 1.0
        self._intraday_return_cache[ticker] = (ret, now)
        return ret

    # ------------------------------------------------------------------
    # Option chain — the meat
    # ------------------------------------------------------------------
    def fetch_option_chain(self, underlying: str,
                           expiration_date: str,
                           option_type: str = "put") -> Optional[list]:
        """
        Fetch the option chain for ``underlying`` at ``expiration_date``,
        filtered to ``option_type`` ∈ {"put", "call"}.  Returns the same
        list-of-dicts shape the Alpaca adapter returns so the chain
        scanner doesn't need to know which feed it's reading::

            [
              {"symbol": "XLF260529C00053000",
               "bid": 0.36, "ask": 0.37, "mid": 0.365,
               "delta": 0.275, "theta": -0.012, "vega": 0.04,
               "gamma": 0.08, "iv": 0.22,
               "strike": 53.0,
               "expiration": "2026-05-29",
               "type": "call"}
            ]

        Cached for ``OPTION_CHAIN_TTL`` (3 min) — Greeks move but not
        millisecond-fast and the adaptive scan can ask for the same
        chain across multiple DTE candidates within a cycle.
        """
        cache_key = f"{underlying}_{expiration_date}_{option_type}"
        now = time.monotonic()
        if cache_key in self._option_cache:
            contracts, cached_at = self._option_cache[cache_key]
            if (now - cached_at) < OPTION_CHAIN_TTL:
                return contracts

        contract_type_param = "PUT" if option_type.lower() == "put" else "CALL"
        # ``strikeCount`` is the silent-failure footgun: without it Schwab
        # returns the chain "envelope" (status, expiration keys) but with
        # **empty per-strike maps** — the chain scanner then sees 0/N
        # priced grid points and rejects every candidate as
        # "no_short_contract".
        # ``entitlement=NP`` (Non-Pro retail) opts the request into the
        # full Greeks payload that retail brokerage holders are entitled
        # to.  Without it some non-pro accounts get strike maps populated
        # but with delta/gamma/theta/vega all zero — the scanner then
        # can't find any contract within Δ ≤ max_delta and rejects with
        # "no_short_contract×80".
        params = {
            "symbol": underlying,
            "contractType": contract_type_param,
            "strikeCount": 30,
            "fromDate": expiration_date,
            "toDate": expiration_date,
            "strategy": "SINGLE",
            "includeUnderlyingQuote": "false",
            "entitlement": "NP",
        }
        body = self._get("/chains", params=params)
        if not isinstance(body, dict):
            return None

        # Pick the right side map.
        side_map_key = "putExpDateMap" if option_type.lower() == "put" \
                                       else "callExpDateMap"
        exp_map = body.get(side_map_key) or {}
        if not exp_map:
            logger.warning(
                "[%s] Schwab chain returned no %s contracts for %s. "
                "status=%s isDelayed=%s expirations_in_other_side=%s",
                underlying, option_type, expiration_date,
                body.get("status"), body.get("isDelayed"),
                len(body.get(
                    "callExpDateMap" if side_map_key == "putExpDateMap"
                    else "putExpDateMap") or {}),
            )
            return []

        # One-time-per-process raw-response dump.  When the chain scanner
        # reports 0/N priced grid points across all tickers, the next
        # diagnostic step is "what did Schwab actually send?"  Dumping a
        # representative sample to logs (delta/gamma/openInterest/strike)
        # lets the operator see whether Greeks are zero, strikes are
        # missing, or both — without grepping JSON.  Once the chain
        # produces real contracts, this dump is silent for the rest of
        # the session.
        if not getattr(self, "_chain_dump_done", False):
            try:
                first_exp = next(iter(exp_map))
                first_map = exp_map[first_exp] or {}
                first_strike = next(iter(first_map), None)
                if first_strike:
                    sample = (first_map[first_strike] or [None])[0]
                    if isinstance(sample, dict):
                        keys_present = sorted(sample.keys())
                        logger.warning(
                            "[%s] Schwab chain DIAGNOSTIC DUMP — first contract: "
                            "exp=%s strike=%s delta=%s gamma=%s theta=%s "
                            "vega=%s rho=%s vol=%s bid=%s ask=%s mark=%s "
                            "openInterest=%s symbol=%s. Field keys: %s",
                            underlying, first_exp, first_strike,
                            sample.get("delta"), sample.get("gamma"),
                            sample.get("theta"), sample.get("vega"),
                            sample.get("rho"), sample.get("volatility"),
                            # Try both Schwab-prod (bid/ask/mark) and
                            # swagger-spec (bidPrice/askPrice/markPrice).
                            sample.get("bid", sample.get("bidPrice")),
                            sample.get("ask", sample.get("askPrice")),
                            sample.get("mark", sample.get("markPrice")),
                            sample.get("openInterest"),
                            sample.get("symbol"), keys_present,
                        )
            except Exception as exc:                       # noqa: BLE001
                logger.debug("Chain dump skipped: %s", exc)
            self._chain_dump_done = True

        contracts: List[Dict] = []
        empty_strike_maps = 0
        for exp_key, strike_map in exp_map.items():
            # exp_key looks like "2026-05-29:24" — the part before the colon
            # is the expiration; the part after is days-to-expiration.
            exp = (exp_key or "").split(":", 1)[0] or expiration_date
            if not strike_map:
                empty_strike_maps += 1
                continue
            for _strike_str, contract_list in (strike_map or {}).items():
                for c in (contract_list or []):
                    if not isinstance(c, dict):
                        continue
                    contracts.append(self._normalize_contract(c, exp, option_type))

        if not contracts:
            # Distinct from the "no expirations" path above — Schwab DID
            # return expiration keys but every strike-map under them was
            # empty.  Most common cause is the strikeCount param missing
            # or the symbol not having an active chain on this account's
            # entitlement tier.  Log enough detail that the operator can
            # tell which.
            logger.warning(
                "[%s] Schwab chain has %d %s expirations but %d/%d "
                "have empty strike maps. params=%s status=%s. "
                "Check that strikeCount is in the request and the "
                "account is entitled to OPRA options data.",
                underlying, len(exp_map), option_type,
                empty_strike_maps, len(exp_map),
                {k: v for k, v in params.items() if k != "symbol"},
                body.get("status"),
            )

        if contracts:
            self._option_cache[cache_key] = (contracts, time.monotonic())
        return contracts

    @staticmethod
    def _normalize_contract(c: Dict, expiration: str,
                            option_type: str) -> Dict:
        """
        Convert a Schwab OptionContract dict into the agent's canonical
        contract shape.

        Field-name reality check (drift from swagger)
        ---------------------------------------------
        Schwab's swagger spec at developer.schwab.com declares the price
        fields as ``bidPrice``/``askPrice``/``markPrice``, but the
        production ``/marketdata/v1/chains`` response actually emits
        them as the 3-letter ``bid``/``ask``/``mark``.  Discovered
        2026-05-06 via the DIAGNOSTIC DUMP path — every contract had
        Greeks populated correctly but `bidPrice`/`askPrice`/`markPrice`
        were ``None``, causing the chain scanner to reject every
        candidate as ``no_short_contract`` (it filters bid=0 entries).
        We try both names; the production short forms win in practice
        but the long forms keep us forward-compatible if Schwab ever
        ships the swagger-shape.

        Other gotchas
        -------------
        * Schwab returns IV as a percent (``22.3`` = 22.3%); divide by
          100 so the rest of the agent sees decimals.
        * Symbols are space-padded; emit compact OCC at the boundary.
        """
        # Try short Schwab-prod names first; fall back to swagger-spec longs.
        def _f(*names) -> float:
            for n in names:
                v = c.get(n)
                if v is not None:
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass
            return 0.0

        bid = _f("bid", "bidPrice")
        ask = _f("ask", "askPrice")
        mid_raw = c.get("mark")
        if mid_raw is None:
            mid_raw = c.get("markPrice")
        if mid_raw is None:
            mid = round((bid + ask) / 2, 4) if (bid > 0 or ask > 0) else 0.0
        else:
            mid = float(mid_raw)
        return {
            "symbol":     from_schwab_symbol(c.get("symbol", "")),
            "bid":        bid,
            "ask":        ask,
            "mid":        round(float(mid), 4),
            "delta":      _f("delta"),
            "theta":      _f("theta"),
            "vega":       _f("vega"),
            "gamma":      _f("gamma"),
            "iv":         _f("volatility") / 100.0,
            "strike":     _f("strikePrice"),
            "expiration": (c.get("expirationDate") or expiration)[:10],
            "type":       option_type.lower(),
        }

    # ------------------------------------------------------------------
    # Pre-submission live quote refresh
    # ------------------------------------------------------------------
    def fetch_option_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fresh bid/ask for the given OCC option symbols.  The agent calls
        this in ``executor._refresh_limit_price`` right before sending an
        order so the limit price reflects what the market is offering
        right now, not what the planning snapshot said 5 minutes ago.

        Returns a dict keyed by **compact** OCC symbol (the same string
        the executor passes in, not Schwab's space-padded form)::

            { "XLF260529C00053000": {"bid": 0.36, "ask": 0.37, "mid": 0.365} }
        """
        if not symbols:
            return {}
        # Build {compact: schwab_padded} translation map; query Schwab
        # with the padded form, return results keyed by compact.
        compact_to_schwab = {s: to_schwab_symbol(s) for s in symbols}
        body = self._get(
            "/quotes",
            params={"symbols": ",".join(compact_to_schwab.values()),
                    "fields": "quote"},
        )
        if not isinstance(body, dict):
            return {}

        out: Dict[str, Dict] = {}
        for compact, padded in compact_to_schwab.items():
            payload = body.get(padded) or {}
            q = payload.get("quote") if isinstance(payload, dict) else None
            if not isinstance(q, dict):
                continue
            # Schwab's /quotes endpoint payload also drifts from the
            # swagger spec — the production response uses bidPrice/
            # askPrice on the QuoteOption schema (per swagger), but we
            # tolerate both for forward compatibility.  See
            # ``_normalize_contract`` for the same drift on /chains.
            def _q(*names):
                for n in names:
                    v = q.get(n)
                    if v is not None:
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            pass
                return 0.0
            bid = _q("bidPrice", "bid")
            ask = _q("askPrice", "ask")
            mid = q.get("mark")
            if mid is None:
                mid = q.get("markPrice")
            if mid is None:
                mid = round((bid + ask) / 2, 4) if (bid > 0 or ask > 0) else 0.0
            else:
                mid = float(mid)
            out[compact] = {
                "bid": bid,
                "ask": ask,
                "mid": round(float(mid), 4),
            }
        return out

    # ------------------------------------------------------------------
    # Market-hours helper — used by AccountPort.is_market_open
    # ------------------------------------------------------------------
    def is_market_open(self) -> bool:   # type: ignore[override]
        """
        Equity market open/closed flag via /markets?markets=equity.
        Falls back to the parent (Alpaca clock) when Schwab is
        unreachable so a transient outage doesn't blank out the gate.
        """
        body = self._get("/markets", params={
            "markets": "equity",
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }, timeout=ALPACA_TIMEOUT)
        if isinstance(body, dict):
            try:
                # Shape: {"equity": {"EQ": {"isOpen": bool, ...}}}
                equity = body.get("equity") or {}
                for _product_key, hours in equity.items():
                    if isinstance(hours, dict) and "isOpen" in hours:
                        return bool(hours["isOpen"])
            except Exception as exc:        # noqa: BLE001
                logger.debug("Schwab /markets parse failed: %s", exc)
        # Fallback to Alpaca clock if Schwab path didn't resolve.
        try:
            return super().is_market_open()
        except AttributeError:
            return False


# ---------------------------------------------------------------------------
# Backwards-compat shim — re-exports the factory from its new home.
# ---------------------------------------------------------------------------
# The factory used to live here when Schwab was the only alternative
# adapter.  It's now in ``market_data_factory.py`` so the per-surface
# routing is owned by a module that doesn't depend on Schwab specifics.
# Keeping this re-export means any external import path that landed
# during the previous integration still works.
from trading_agent.market_data_factory import build_market_data_provider  # noqa: E402,F401
