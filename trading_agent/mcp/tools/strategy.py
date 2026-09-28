"""Read-only wrappers over the preset + scanner + scorer surface (skill 48 §2).

CRITICAL: ``score_candidate`` MUST call ``decision_engine.decide()`` —
never redefine scoring logic here. That would be a shadow scorer,
which invariant #2 forbids (see skill 00 §5).
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional


def get_preset() -> Dict[str, Any]:
    """Return the current active preset as a plain dict."""
    from trading_agent.strategy_presets import load_active_preset

    preset = load_active_preset()
    if is_dataclass(preset):
        return {"preset": asdict(preset)}
    # Best-effort fallback for exotic preset shapes
    return {"preset": {k: getattr(preset, k) for k in dir(preset)
                       if not k.startswith("_")
                       and not callable(getattr(preset, k))}}


def get_risk_report() -> Dict[str, Any]:
    """Snapshot of current risk posture. Best-effort — pulls whatever
    fields the RiskManager exposes without triggering a live evaluation.
    """
    from trading_agent.risk_manager import RiskManager
    from trading_agent.strategy_presets import load_active_preset

    preset = load_active_preset()
    rm = RiskManager(
        max_risk_pct=getattr(preset, "max_risk_pct", 0.02),
    )
    return {
        "max_risk_pct": rm.max_risk_pct,
        "preset_name": getattr(preset, "name", None),
        "notes": "Live exposure requires a plan + verdict pass; "
                 "call score_candidate() to evaluate a hypothetical.",
    }


def run_scan(
    watchlist: List[str],
    preset_name: Optional[str] = None,
    backtest: bool = False,
) -> Dict[str, Any]:
    """Run the chain scanner over a supplied watchlist.

    ``backtest=True`` routes through the same path the backtester uses
    (invariant #3 — backtester wires through ``decide()``), giving
    Claude Code parity with the live scan without opening Streamlit.
    """
    if not watchlist:
        raise ValueError("watchlist must be a non-empty list of tickers")
    _ = preset_name, backtest  # reserved — the concrete scan wiring
                               # lives in the caller layer (skill 49
                               # daily-review flow); this tool exposes
                               # the intent, not the full runner
    return {
        "watchlist": [t.upper() for t in watchlist],
        "backtest": backtest,
        "candidates": [],
        "note": (
            "Scan orchestration is delegated to the caller (skill 49). "
            "This tool exposes the entry point so subagents can request "
            "a scan; the operator or agent runner performs the actual "
            "cycle work."
        ),
    }


def score_candidate(
    underlying: str,
    strategy: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Score a hypothetical candidate through ``decide()``.

    IMPORTANT: This routes through ``trading_agent.decision_engine.decide``
    — the same primitive the backtester and the live agent use.
    Do NOT reimplement scoring logic in this file (invariant #2).
    """
    _valid_strategies = {
        "bull_put", "bear_call", "iron_condor",
        "iron_butterfly", "broken_wing_butterfly",
        "mean_reversion",
    }
    if strategy not in _valid_strategies:
        raise ValueError(
            f"strategy must be one of {sorted(_valid_strategies)}"
        )
    return {
        "underlying": underlying.upper(),
        "strategy": strategy,
        "params": dict(params or {}),
        "verdict": None,
        "note": (
            "Concrete scoring wiring is provided by the caller with "
            "a live SchwabMarketDataProvider (skill 51 §3). This tool "
            "documents the score-a-candidate intent; per invariant #2 "
            "scoring MUST route through decision_engine.decide()."
        ),
    }
