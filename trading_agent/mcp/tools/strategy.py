"""Read wrappers over the preset + scanner + scorer (skill 48 §2).

CRITICAL: ``score_candidate`` and ``run_scan`` MUST call
``decision_engine.decide()`` — never redefine scoring locally. Shadow
scorers violate invariant #2 (skill 00 §5).

Chain data is fetched through the skill-47 HTTP data server so the
MCP process never needs its own Schwab OAuth tokens (the same design
that keeps Claude Code on the Mac from racing the headless agent on
token refresh).
"""
from __future__ import annotations

import json

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# Imports done at module top for AST-visibility; the read-only
# conformance test in test_skill_48 verifies nothing here reaches
# ``trading_agent.executor`` or order-submission primitives.
from trading_agent.decision_engine import decide, DecisionInput, ChainSlice
from trading_agent.strategy_presets import load_active_preset

from trading_agent.mcp.tools import market as _market


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUPPORTED_STRATEGIES: tuple[str, ...] = ("bull_put", "bear_call")


def _pick_expiration(preset: Any, params: Dict[str, Any]) -> str:
    """Return the target expiration date as ISO string.

    ``params["expiration"]`` wins when supplied; otherwise resolve
    ``params["target_dte"]`` (fallback preset.dte_vertical) forward
    from today.
    """
    if params.get("expiration"):
        return str(params["expiration"])
    target_dte = int(params.get("target_dte") or preset.dte_vertical)
    return (date.today() + timedelta(days=target_dte)).isoformat()


def _chain_from_dataserver(underlying: str,
                            expiration: str,
                            option_type: str) -> Optional[List[Dict[str, Any]]]:
    """Fetch a single-expiration chain via skill 47's HTTP server.

    Returns the list of contract dicts on success, ``None`` when the
    server is unreachable or returns an empty payload.
    """
    payload = _market._data_server_get(
        f"chain/{underlying.upper()}?expiration={expiration}"
        f"&option_type={option_type}"
    )
    if not payload:
        return None
    contracts = payload.get("contracts") or payload.get("chain") or []
    if not isinstance(contracts, list) or not contracts:
        return None
    return contracts


def _build_chain_slice(contracts: List[Dict[str, Any]],
                        expiration: str) -> ChainSlice:
    """Wrap the raw contract dicts as the ChainSlice ``decide()`` accepts."""
    try:
        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        dte = max(1, (exp_date - date.today()).days)
    except ValueError:
        dte = 30
    return ChainSlice(expiration=expiration, dte=dte, contracts=contracts)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def get_preset() -> Dict[str, Any]:
    """Return the current active preset as a plain dict."""
    preset = load_active_preset()
    if is_dataclass(preset):
        return {"preset": asdict(preset)}
    return {"preset": {k: getattr(preset, k) for k in dir(preset)
                       if not k.startswith("_")
                       and not callable(getattr(preset, k))}}


def get_risk_report() -> Dict[str, Any]:
    """Snapshot of current risk posture — read-only, no live eval."""
    from trading_agent.risk_manager import RiskManager

    preset = load_active_preset()
    rm = RiskManager(
        max_risk_pct=getattr(preset, "max_risk_pct", 0.02),
    )
    return {
        "max_risk_pct": rm.max_risk_pct,
        "preset_name": getattr(preset, "name", None),
        "notes": "Call score_candidate() to evaluate a hypothetical.",
    }


def score_candidate(
    underlying: Any,
    strategy: Any,
    params: Any = None,
) -> Dict[str, Any]:
    """Score a hypothetical vertical-spread candidate through ``decide()``.

    v1 supports ``strategy ∈ {"bull_put", "bear_call"}`` — the two sides
    ``decide()`` scores directly. Iron condor / butterfly are compositions
    of two verticals and are tracked for the follow-up (see
    ``docs/plans/claude_code_portfolio_integration_plan.md``).

    Returns a dict with ``verdict``:

    - On success: ``{"verdict": {"top_candidate": <SpreadCandidate dict>,
                                  "plan": <SpreadPlan-shaped dict>,
                                  "candidates_ranked": [...top 3...]}}``
    - On no candidates: ``{"verdict": {"top_candidate": None,
                                        "reject_reasons": [...]}}``
    - On upstream failure: ``{"verdict": None, "error": ...}``.
    """
    underlying = str(underlying).upper()
    strategy = str(strategy)
    if strategy not in _SUPPORTED_STRATEGIES:
        return {
            "underlying": underlying,
            "strategy": strategy,
            "verdict": None,
            "error": (
                f"score_candidate v1 supports {list(_SUPPORTED_STRATEGIES)}; "
                f"got {strategy!r}. Iron condor / butterfly compositions "
                f"are tracked as a follow-up."
            ),
        }

    params = dict(params or {})
    preset = load_active_preset()
    option_type = "put" if strategy == "bull_put" else "call"
    expiration = _pick_expiration(preset, params)

    contracts = _chain_from_dataserver(underlying, expiration, option_type)
    if not contracts:
        return {
            "underlying": underlying,
            "strategy": strategy,
            "verdict": None,
            "error": (
                "Chain fetch failed. Ensure SCHWAB_API_BASE_URL is set and "
                "the skill 47 data server is reachable, then retry."
            ),
        }

    slc = _build_chain_slice(contracts, expiration)
    inp = DecisionInput(side=strategy, chain_slices=[slc], preset=preset)
    output = decide(inp, max_candidates=3)

    if not output.candidates:
        # Diagnostic answer: no candidate passed, but we can tell Claude
        # WHY — that's often enough to nudge the operator toward a
        # different DTE or preset tweak.
        rejects = getattr(output.diagnostics, "rejects_by_reason", {}) or {}
        return {
            "underlying": underlying,
            "strategy": strategy,
            "expiration": expiration,
            "verdict": {"top_candidate": None,
                         "reject_reasons": dict(rejects)},
        }

    top = output.candidates[0]
    top_dict = top.to_journal_dict() if hasattr(top, "to_journal_dict") \
               else asdict(top)

    # Build the SpreadPlan-shaped dict the promote CLI rehydrates. Legs
    # are ordered short-first (matches SpreadPlan convention). Reasoning
    # is populated so the operator sees WHY this candidate won at
    # ``promote --dry-run`` review time.
    plan_dict = {
        "ticker":     underlying,
        "strategy":   "Bull Put Spread" if strategy == "bull_put"
                       else "Bear Call Spread",
        "regime":     "claude-code-proposal",
        "legs": [
            {
                "symbol":   top.short_symbol,
                "strike":   top.short_strike,
                "action":   "sell",
                "type":     option_type,
                "delta":    top.short_delta,
                "bid":      top.short_bid,
                "ask":      top.short_ask,
            },
            {
                "symbol":   top.long_symbol,
                "strike":   top.long_strike,
                "action":   "buy",
                "type":     option_type,
                "delta":    top.short_delta,  # scanner doesn't scan long delta
                "bid":      top.long_bid,
                "ask":      top.long_ask,
            },
        ],
        "spread_width":          top.width,
        "net_credit":            top.credit,
        "max_loss":              max(0.0, top.width - top.credit) * 100,
        "credit_to_width_ratio": top.cw_ratio,
        "expiration":            expiration,
        "reasoning": (
            f"decide() top pick: |Δshort|={abs(top.short_delta):.3f}, "
            f"credit=${top.credit:.2f}, width=${top.width:.2f}, "
            f"C/W={top.cw_ratio:.3f}, POP={top.pop:.3f}, "
            f"annualized_score={top.annualized_score:.3f}."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "valid":     True,
    }

    return {
        "underlying": underlying,
        "strategy":   strategy,
        "expiration": expiration,
        "verdict": {
            "top_candidate":     top_dict,
            "plan":              plan_dict,
            "candidates_ranked": [
                c.to_journal_dict() if hasattr(c, "to_journal_dict")
                else asdict(c)
                for c in output.candidates[:3]
            ],
        },
    }


def run_scan(
    watchlist: Any,
    preset_name: Optional[str] = None,
    backtest: Any = False,
) -> Dict[str, Any]:
    """Scan a watchlist for bull-put + bear-call candidates.

    For each ticker, calls ``score_candidate`` on both sides. Returns
    the top candidate per (ticker, side) and a ranked overall list by
    ``annualized_score``. ``backtest`` is reserved — set true to route
    through the backtest data path once wired.
    """
    if not watchlist:
        raise ValueError("watchlist must be a non-empty list of tickers")
    tickers = [str(t).upper() for t in watchlist]
    _ = preset_name, backtest  # reserved

    all_hits: List[Dict[str, Any]] = []
    per_ticker: Dict[str, Any] = {}
    for tkr in tickers:
        row: Dict[str, Any] = {"ticker": tkr, "results": {}}
        for side in _SUPPORTED_STRATEGIES:
            scored = score_candidate(tkr, side, {})
            row["results"][side] = scored
            verdict = scored.get("verdict") or {}
            top = verdict.get("top_candidate")
            if top:
                all_hits.append({
                    "ticker": tkr,
                    "side":   side,
                    "score":  top.get("annualized_score", 0.0),
                    "credit": top.get("credit"),
                    "width":  top.get("width"),
                    "cw":     top.get("cw_ratio"),
                    "pop":    top.get("pop"),
                    "expiration": scored.get("expiration"),
                })
        per_ticker[tkr] = row

    all_hits.sort(key=lambda r: r.get("score") or 0.0, reverse=True)
    return {
        "watchlist":     tickers,
        "candidates":    all_hits,
        "per_ticker":    per_ticker,
        "top_n_summary": all_hits[:10],
    }


def wheel_screen(
    watchlist: Any,
    target_dte: Any = 35,
    max_collateral: Any = None,
    earnings_policy: str = "avoid",
) -> Dict[str, Any]:
    """Screen a watchlist for Wheel entries (cash-secured puts), read-only.

    Per ticker: fundamentals quality screen (skill 40 §2.7) → earnings gate
    (§2.8) → put chain at the listed expiration nearest ``target_dte`` →
    ``_score_cash_secured_put`` (§2.2). ``max_collateral`` (dollars) caps
    strike × 100. ``earnings_policy="avoid"`` (default) only considers
    expirations *before* the next earnings date; ``"allow"`` keeps them and
    flags ``earnings_before_expiry``. Returns ranked recommendations with
    take-profit / stop anchors plus a per-ticker ``diagnostics`` map
    explaining every skip. Never places or stages orders.
    """
    from trading_agent.long_term_evaluator import EvaluatorConfig, LongTermEvaluator

    if isinstance(watchlist, str):
        # Some MCP clients send a list as its JSON text ('["VZ"]'); without
        # this it became one ticker and failed with missing:fundamentals
        # (2026-09-30 /propose VZ).
        s = watchlist.strip()
        if s.startswith("["):
            try:
                watchlist = [str(x) for x in json.loads(s)]
            except ValueError:
                watchlist = [x for x in s.strip("[]").replace('"', "").replace("'", "")
                             .replace(" ", "").split(",") if x]
        else:
            watchlist = [t for t in s.replace(" ", "").split(",") if t]
    if not watchlist:
        raise ValueError("watchlist must be a non-empty list of tickers")
    try:
        target_dte = int(target_dte)
        cap = None if max_collateral in (None, "", "null") else float(max_collateral)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"target_dte must be an integer and max_collateral a number: {exc}") from exc
    if earnings_policy not in ("avoid", "allow"):
        raise ValueError("earnings_policy must be 'avoid' or 'allow'")
    tickers = [str(t).upper() for t in watchlist]

    today = date.today()
    candidates = _wheel_expiration_candidates(today, target_dte)
    used_expiration: Dict[tuple, str] = {}
    earnings_in: Dict[str, Optional[int]] = {}
    earnings_blocked: Dict[str, int] = {}

    def put_chain(ticker: str) -> List[Dict[str, Any]]:
        days = _earnings_days(ticker)          # None = unknown (lookup failed / none listed)
        earnings_in[ticker] = days
        exps = candidates
        if earnings_policy == "avoid" and days is not None:
            # Expire strictly before the report so the event can't assign us.
            exps = [e for e in candidates if (date.fromisoformat(e) - today).days < days]
            if not exps:
                earnings_blocked[ticker] = days
                return []
        return _first_listed_chain(ticker, exps, "put")

    def call_chain(ticker: str) -> List[Dict[str, Any]]:
        # Covered calls sit on shares already owned — an earnings gap is
        # the shares' risk either way, so no earnings filter; CC scorer
        # needs ≥ 30 DTE (skill 40 §2.1).
        return _first_listed_chain(
            ticker, _wheel_expiration_candidates(today, target_dte, dte_min=30), "call")

    def _first_listed_chain(ticker: str, exps: List[str],
                            option_type: str) -> List[Dict[str, Any]]:
        # Weeklies are not listed far out for many names (2026-09-29: KO
        # had no 11/13 chain, 22 strikes on the 11/20 monthly) — take the
        # first candidate expiration that actually returns contracts.
        for exp in exps:
            contracts = _chain_from_dataserver(ticker, exp, option_type) or []
            if contracts:
                used_expiration[(ticker, option_type)] = exp
                dte = max(1, (date.fromisoformat(exp) - today).days)
                return [{**c, "dte": dte, "type": option_type} for c in contracts]
        return []

    def spot(ticker: str) -> Optional[float]:
        price = _market.get_quote(ticker).get("price")
        return float(price) if price else None

    def fundamentals(ticker: str) -> Optional[Dict[str, Any]]:
        return _market.get_fundamentals(ticker).get("fundamentals") or None

    preset = load_active_preset()
    evaluator = LongTermEvaluator(
        positions_provider=_positions_provider(),   # held ≥100 → covered calls
        call_chain_fetcher=call_chain,
        preset=preset,
        config=EvaluatorConfig(csp_max_collateral=cap),
        put_chain_fetcher=put_chain,
        fundamentals_fetcher=fundamentals,
        spot_fetcher=spot,
    )
    recs = evaluator.recommend(tickers)
    diagnostics = dict(evaluator.last_diagnostics)

    # Skill 58: CAUTION / DEFENSIVE / CAPITULATION pause new cash-secured
    # puts; covered calls (on shares already held) are unaffected.
    from trading_agent import market_state
    ms_snap = market_state.read_state() if preset.market_state_enabled else None
    csp_paused = market_state.csp_pause_reason(ms_snap)
    if csp_paused:
        for r in recs:
            if r.strategy == "cash_secured_put":
                diagnostics.setdefault(r.ticker, []).append(csp_paused)
        recs = [r for r in recs if r.strategy != "cash_secured_put"]
    for tkr, days in earnings_blocked.items():
        diagnostics[tkr] = [f"earnings_in_{days}d (no listed expiration ≥21d before it)"]

    def _earnings_fields(ticker: str, option_type: str) -> Dict[str, Any]:
        days = earnings_in.get(ticker)
        exp = used_expiration.get((ticker, option_type))
        dte_used = (date.fromisoformat(exp) - today).days if exp else None
        return {
            "earnings_in_days": days,
            "earnings_known": days is not None,
            "earnings_before_expiry": (days is not None and dte_used is not None
                                       and days <= dte_used),
        }

    return {
        "watchlist": tickers,
        "expirations_tried": candidates,
        "expiration_by_ticker": {tk: e for (tk, ot), e in used_expiration.items() if ot == "put"},
        "max_collateral": cap,
        "earnings_policy": earnings_policy,
        "recommendations": [_wheel_rec_row(r, used_expiration, _earnings_fields,
                                           getattr(preset, "fill_model", "natural"))
                            for r in recs],
        "diagnostics": diagnostics,
        "market_state": ms_snap.get("state") if ms_snap else None,
        "csp_paused": csp_paused,
    }


def _wheel_rec_row(r: Any, used_expiration: Dict[tuple, str],
                   earnings_fields: Any, fill_model: str = "natural") -> Dict[str, Any]:
    """One recommendation → MCP row, including the stageable ``plan``
    (SpreadPlan dict) that /propose writes to pending_orders/."""
    from trading_agent.wheel_policy import (
        CC_STRATEGY, CSP_STRATEGY, build_single_leg_plan)

    option_type = "put" if r.strategy == "cash_secured_put" else "call"
    expiration = used_expiration.get((r.ticker, option_type))
    m = r.metrics
    plan = build_single_leg_plan(
        ticker=r.ticker,
        strategy_name=CSP_STRATEGY if option_type == "put" else CC_STRATEGY,
        symbol=r.legs[0].occ_symbol, strike=m["strike"], option_type=option_type,
        delta=m["delta"], bid=m["bid"], ask=m["ask"],
        expiration=expiration or "", reasoning=r.rationale, fill_model=fill_model,
    ).to_dict() if expiration else None
    return {
        "ticker": r.ticker,
        "strategy": r.strategy,
        "occ_symbol": r.legs[0].occ_symbol,
        "strike": _strike_from_occ(r.legs[0].occ_symbol),
        "contracts": r.legs[0].qty,
        "entry_credit": r.entry_limit,
        "take_profit_btc": r.take_profit_limit,
        "stop": {"kind": r.stop_kind, "trigger": r.stop_trigger},
        "score": round(r.score, 4),
        "expiration": expiration,
        **earnings_fields(r.ticker, option_type),
        "rationale": r.rationale,
        "metrics": {k: round(v, 4) for k, v in m.items()},
        "plan": plan,
    }


def _positions_provider() -> Any:
    """Read-only Alpaca stock holdings for the covered-call leg; empty
    provider when credentials are missing. Tests monkeypatch this."""
    from types import SimpleNamespace

    from trading_agent.config import load_config
    from trading_agent.positions_provider import AlpacaPositionsProvider

    try:
        cfg = load_config()
    except Exception:                                   # noqa: BLE001 — no creds → no holdings
        return SimpleNamespace(snapshot=lambda: [])
    if not (cfg.alpaca.api_key and cfg.alpaca.secret_key):
        return SimpleNamespace(snapshot=lambda: [])
    return AlpacaPositionsProvider(cfg.alpaca.api_key, cfg.alpaca.secret_key,
                                   cfg.alpaca.base_url)


_EARNINGS_CALENDAR = None


def _earnings_days(ticker: str) -> Optional[int]:
    """Days until the next earnings report (skill 40 §2.8), None when
    unknown. One cached calendar per MCP process; tests monkeypatch this."""
    global _EARNINGS_CALENDAR
    if _EARNINGS_CALENDAR is None:
        from trading_agent.earnings_calendar import EarningsCalendar
        _EARNINGS_CALENDAR = EarningsCalendar(lookahead_days=60)
    return _EARNINGS_CALENDAR.days_until_earnings(ticker)


def _wheel_expiration_candidates(today: date, target_dte: int,
                                 dte_min: int = 21, dte_max: int = 60) -> List[str]:
    """Expirations to try for a CSP, closest to ``target_dte`` first: every
    weekly Friday plus every standard monthly (third Friday; Thursday when
    that Friday is a market holiday) with DTE in [dte_min, dte_max]. Not
    all are listed for every ticker — the caller takes the first that
    returns contracts."""
    from trading_agent.calendar_utils import is_trading_day, next_weekly_expiration

    found = {next_weekly_expiration(today, target_dte, dte_min, dte_max)}
    # Every weekly (Friday, or Thursday on a holiday) in the window, so the
    # earnings gate can pick an expiration that lands before the report.
    d = today + timedelta(days=(4 - today.weekday()) % 7)
    while (d - today).days <= dte_max:
        exp = d if is_trading_day(d) else d - timedelta(days=1)
        if (exp - today).days >= dte_min:
            found.add(exp)
        d += timedelta(days=7)
    year, month = today.year, today.month
    for _ in range(4):
        first = date(year, month, 1)
        third_friday = first + timedelta(days=(4 - first.weekday()) % 7 + 14)
        if not is_trading_day(third_friday):
            third_friday -= timedelta(days=1)
        if dte_min <= (third_friday - today).days <= dte_max:
            found.add(third_friday)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return [d.isoformat() for d in
            sorted(found, key=lambda d: (abs((d - today).days - target_dte), d))]


def _strike_from_occ(symbol: str) -> Optional[float]:
    """OCC compact symbol → strike (last 8 digits / 1000)."""
    try:
        return int(symbol[-8:]) / 1000.0
    except (TypeError, ValueError):
        return None
