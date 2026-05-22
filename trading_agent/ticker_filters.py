"""ticker_filters.py — early-return filter pipeline for per-ticker cycle.

Skill 36 — item 4 of the standards roadmap. Extracts the three early-
return checks from the top of ``agent._process_ticker`` into a
constructor-injected collaborator:

  * **Directional-bias filter** — the active preset can restrict which
    regimes are tradeable. A BEARISH preset blocks BULL regimes etc.
    Mean-reversion is always allowed (3-σ touches are fear-spike
    signals, not directional views).
  * **RSI gate** (opt-in via ``RSI_GATE_ENABLED`` env var) — refines
    the strategy choice using RSI alongside the regime. Can skip
    the cycle, override the regime, or proceed unchanged.
  * **High-IV block** — IV rank > 95th percentile blocks all new
    entries (extreme volatility, defense-first).

Pre-2026-05-22 these three checks were ~120 lines inlined at the top
of ``_process_ticker`` (340 lines total). The extraction makes each
filter testable in isolation with a stub journal + stub preset —
no MagicMock(spec=TradingAgent) needed. Same pattern as
close_event_collaborators.py (skill 35).

``FilterResult`` is the return-shape contract: ``None`` = proceed to
the planning phase; otherwise a ``FilterResult`` carries the
already-journaled skip reason + the dict to return from
``_process_ticker``.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from trading_agent.regime import RegimeAnalysis


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterResult:
    """Returned by TickerFilters.evaluate when a filter triggers an
    early-return. Carries both the journal-row context (already
    written before the result is returned) and the per-ticker dict
    that ``_process_ticker`` must return.

    Callers must NOT mutate this object; the journal write already
    happened inside evaluate(), so this is a pure result envelope.
    """
    # The dict returned by _process_ticker to its caller.
    result: Dict[str, Any]
    # For tests + logging: which filter triggered.
    triggered_by: str
    # Optional analysis override (RSI gate may rewrite regime).
    analysis_override: Optional[Any] = None


# ---------------------------------------------------------------------------
# Preset protocol — minimal surface we need
# ---------------------------------------------------------------------------


class _PresetProtocol(Protocol):
    directional_bias: str
    name: str


# ---------------------------------------------------------------------------
# TickerFilters — the three early-return checks
# ---------------------------------------------------------------------------


class TickerFilters:
    """Evaluates the three early-return filters in sequence.

    Constructor takes the journal (for skip-row writes) and the active
    preset. ``evaluate(ticker, analysis)`` returns ``None`` if all
    three filters pass (proceed to planning) or a ``FilterResult`` if
    one triggered.

    The journal-write inside each filter is part of the contract: a
    skipped ticker must leave a journal row so the dashboard's "Why
    didn't this trade?" view can attribute the skip. Pre-extraction
    these writes were ``self.journal_kb.log_signal(...)`` /
    ``log_defense_first(...)`` calls scattered through the method
    body.
    """

    def __init__(self, *, journal_kb, preset: _PresetProtocol):
        self.journal_kb = journal_kb
        self.preset = preset

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def evaluate(self, ticker: str,
                 analysis: "RegimeAnalysis") -> Optional[FilterResult]:
        """Run all three filters in order. Returns the first triggered
        result, or None if all pass.

        Behavior verbatim with pre-2026-05-22 inline code:

          1. directional bias
          2. RSI gate (env-gated)
          3. high-IV block

        Order matters — directional bias is the cheapest check; RSI
        gate may need an analysis clone; high-IV is last so the
        operator sees the directional reason before the IV reason
        in journal scans on the same cycle.
        """
        # Filter 1: directional bias
        res = self._check_directional_bias(ticker, analysis)
        if res is not None:
            return res
        # Filter 2: RSI gate (env-gated, may rewrite regime)
        res = self._check_rsi_gate(ticker, analysis)
        if res is not None and res.result.get("status") == "skipped":
            return res
        # If the RSI gate produced an analysis override, the result
        # is None but the analysis_override carries the rewrite.
        analysis_override = (
            res.analysis_override if res is not None else None
        )
        if analysis_override is not None:
            analysis = analysis_override
        # Filter 3: high-IV block
        res = self._check_high_iv(ticker, analysis)
        if res is not None:
            return res
        # All filters passed. If RSI overrode the regime, surface the
        # override so the caller can use the new analysis downstream.
        if analysis_override is not None:
            return FilterResult(
                result={"_proceed_with_override": True},
                triggered_by="rsi_gate_override",
                analysis_override=analysis_override,
            )
        return None

    # ------------------------------------------------------------------
    # Individual filters
    # ------------------------------------------------------------------

    def _check_directional_bias(
        self, ticker: str, analysis: "RegimeAnalysis",
    ) -> Optional[FilterResult]:
        """Skip ticker when the preset's directional bias forbids the
        ticker's current regime. See ``strategy_presets.regime_is_allowed``."""
        from trading_agent.strategy_presets import regime_is_allowed
        if regime_is_allowed(
            analysis.regime.value, self.preset.directional_bias
        ):
            return None
        reason = (
            f"DirectionalBias={self.preset.directional_bias} blocks "
            f"regime={analysis.regime.value}"
        )
        logger.info("[%s] %s — skipping ticker", ticker, reason)
        self.journal_kb.log_signal(
            ticker=ticker,
            action="skipped_bias",
            price=analysis.current_price,
            raw_signal={
                "regime": analysis.regime.value,
                "directional_bias": self.preset.directional_bias,
                "preset": self.preset.name,
                "reason": reason,
            },
        )
        return FilterResult(
            result={
                "ticker": ticker,
                "regime": analysis.regime.value,
                "strategy": "skipped_bias",
                "plan_valid": False,
                "risk_approved": False,
                "status": "skipped",
                "reason": reason,
            },
            triggered_by="directional_bias",
        )

    def _check_rsi_gate(
        self, ticker: str, analysis: "RegimeAnalysis",
    ) -> Optional[FilterResult]:
        """When ``RSI_GATE_ENABLED=true``, refine the strategy choice
        using RSI alongside the regime. Three possible outcomes:

          * skip cycle — momentum too active for default strategy
          * proceed as-is — gate has no opinion
          * regime override — downgrade IC/sideways plan to a single-
            side vertical; clone analysis with new regime so the
            planner picks the right strategy AND the journal records
            the actual choice.
        """
        rsi_gate_enabled = os.environ.get(
            "RSI_GATE_ENABLED", "false"
        ).strip().lower() in ("true", "1", "yes", "on")
        if not rsi_gate_enabled:
            return None

        from trading_agent.rsi_gate import evaluate_rsi_gate
        decision = evaluate_rsi_gate(analysis.regime, analysis.rsi_14)
        if not decision.allow:
            logger.info(
                "[%s] RSI gate skipped cycle — %s", ticker, decision.reason,
            )
            self.journal_kb.log_signal(
                ticker=ticker,
                action="skipped_rsi_gate",
                price=analysis.current_price,
                raw_signal={
                    "regime":   analysis.regime.value,
                    "rsi_14":   analysis.rsi_14,
                    "reason":   decision.reason,
                    "preset":   self.preset.name,
                },
            )
            return FilterResult(
                result={
                    "ticker": ticker,
                    "regime": analysis.regime.value,
                    "strategy": "skipped_rsi_gate",
                    "plan_valid": False,
                    "risk_approved": False,
                    "status": "skipped",
                    "reason": decision.reason,
                },
                triggered_by="rsi_gate_skip",
            )
        if decision.override_regime is not None:
            logger.info(
                "[%s] RSI gate override — %s", ticker, decision.reason,
            )
            new_analysis = dataclasses.replace(
                analysis, regime=decision.override_regime
            )
            return FilterResult(
                result={},   # Not used; caller checks analysis_override.
                triggered_by="rsi_gate_override",
                analysis_override=new_analysis,
            )
        return None

    def _check_high_iv(
        self, ticker: str, analysis: "RegimeAnalysis",
    ) -> Optional[FilterResult]:
        """Block all new entries when IV rank > 95th percentile."""
        if not getattr(analysis, "high_iv_warning", False):
            return None
        reason = (
            f"HighIV: IV rank {getattr(analysis, 'iv_rank', 0):.1f} > 95th pct "
            f"— extreme volatility, blocking all new entries"
        )
        logger.warning(
            "[%s] %s | strategy_mode=defense_first", ticker, reason,
        )
        self.journal_kb.log_defense_first(
            ticker, reason, analysis.current_price,
            {
                "regime": analysis.regime.value,
                "iv_rank": getattr(analysis, "iv_rank", 0.0),
                "high_iv_warning": True,
            },
        )
        return FilterResult(
            result={
                "ticker": ticker,
                "regime": analysis.regime.value,
                "strategy": "skipped",
                "plan_valid": False,
                "risk_approved": False,
                "status": "skipped",
                "reason": reason,
                "strategy_mode": "defense_first",
            },
            triggered_by="high_iv",
        )


__all__ = ["TickerFilters", "FilterResult"]
