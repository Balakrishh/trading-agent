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
