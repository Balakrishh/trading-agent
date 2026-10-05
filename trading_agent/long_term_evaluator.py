"""long_term_evaluator.py — portfolio-aware long-term options evaluator.

Skill: ``docs/skills/40_long_term_options_evaluator.md``.

Takes a watchlist + current holdings + market-data provider and produces a
ranked list of ``Recommendation`` objects suggesting long-dated options
strategies instead of outright stock purchases.

This module is a **pure orchestrator**: all strategy scoring lives in
``decision_engine.py`` (CI invariant 2). The evaluator's job is to:

1. Read holdings from a ``PositionsProvider`` (skill 41).
2. For each ticker, fetch the relevant chain via the supplied
   market-data provider.
3. Walk the chain and call the appropriate ``_score_*`` helper from
   ``decision_engine.py``.
4. Assemble entry / take-profit / stop-loss anchors per skill 40 §2.6
   into a ``Recommendation``.
5. Sort by score, truncate, and return.

This session's walking-skeleton scope: covered-call (income-overlay)
recommendations only. CSP / LEAPS / PMCC / debit-spread evaluators
land next session in the same shape.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from trading_agent.decision_engine import (
    _score_cash_secured_put_with_reason,
    _score_covered_call_with_reason,
)
from trading_agent.fundamentals_screen import (
    WheelScreenConfig,
    screen_fundamentals,
)
from trading_agent.positions_provider import Position, PositionsProvider
from trading_agent.wheel_policy import CSP_STOP_ABS_DELTA, TAKE_PROFIT_PCT_OF_CREDIT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recommendation — operator-facing output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecommendationLeg:
    """One leg of a recommended options ticket."""
    action: str          # "STO" (sell-to-open), "BTO" (buy-to-open), etc.
    occ_symbol: str
    qty: int
    side: str            # "long" | "short"
    limit_price: float


@dataclass(frozen=True)
class Recommendation:
    """One actionable suggestion ready for operator review.

    The Streamlit panel renders this directly; the Phase-5 order layer
    consumes the same shape to build a Schwab TRIGGER + OCO bracket
    (skill 40 §2.6 + §4).
    """
    ticker: str
    strategy: str                # "covered_call" | "cash_secured_put" | "leaps_call" | "pmcc" | "debit_spread"
    legs: List[RecommendationLeg]
    entry_limit: float           # net credit (positive) for credit strategies; net debit (positive) for debit strategies
    entry_kind: str              # "credit" | "debit"
    take_profit_limit: float     # BTC price (credit) or STC price (debit)
    stop_trigger: float          # underlying-price level OR option-mid level (per strategy.entry_kind)
    stop_kind: str               # "underlying_price" | "spread_mid" | "delta_threshold"
    stop_limit_offset_pct: float # 0.05 = "limit = stop × (1 ± 0.05)"
    score: float
    rationale: str
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Skill 40 §4 — exit-anchor sanity check.
        if self.entry_kind == "credit":
            # take-profit on a credit strategy is BTC at a lower premium.
            if not (0.0 < self.take_profit_limit < self.entry_limit):
                raise ValueError(
                    f"Recommendation TP-limit ({self.take_profit_limit}) must "
                    f"satisfy 0 < tp < entry_limit ({self.entry_limit}) for "
                    f"credit strategies."
                )
            # Stop on a credit strategy should be > 1.5× the credit (a "this went wrong" line).
            if self.stop_kind == "spread_mid" and self.stop_trigger <= self.entry_limit * 1.5:
                raise ValueError(
                    f"Recommendation stop_trigger ({self.stop_trigger}) must "
                    f"be > entry_limit × 1.5 ({self.entry_limit * 1.5}) for "
                    f"spread-mid credit strategies."
                )
        elif self.entry_kind == "debit":
            # take-profit on debit is STC at a higher price than entry.
            if self.take_profit_limit <= self.entry_limit:
                raise ValueError(
                    f"Recommendation TP-limit ({self.take_profit_limit}) must "
                    f"exceed entry_limit ({self.entry_limit}) for debit strategies."
                )


# ---------------------------------------------------------------------------
# Chain providers — duck-typed so tests can inject fixtures.
# ---------------------------------------------------------------------------

ChainFetcher = Callable[[str], List[Dict[str, Any]]]
"""Returns a list of normalised OPTION-side contract dicts for ``ticker``.

Each contract dict must carry: ``strike`` (float), ``delta`` (signed float),
``bid`` (float), ``ask`` (float), ``dte`` (int), ``symbol`` (str), and the
optional ``iv_rank`` (float ∈ [0, 1]). The dict shape mirrors what
``ChainScanner`` already builds for credit-spread scoring so the evaluator
can reuse the same upstream fetch path.
"""


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@dataclass
class EvaluatorConfig:
    """Knobs the walking-skeleton scope reads. Next session expands."""
    cc_max_recommendations_per_ticker: int = 3
    cc_take_profit_pct_of_credit: float = TAKE_PROFIT_PCT_OF_CREDIT  # BTC at 50% of credit (skill 30)
    cc_stop_underlying_pct: float = 0.92          # close if underlying < cost_basis × 0.92
    cc_stop_limit_offset_pct: float = 0.05        # stop-limit offset from trigger
    # Cash-secured put / Wheel entry (skill 40 §2.2, §2.6, §2.7)
    csp_max_recommendations_per_ticker: int = 2
    csp_take_profit_pct_of_credit: float = TAKE_PROFIT_PCT_OF_CREDIT  # BTC at 50% of credit
    csp_stop_delta: float = CSP_STOP_ABS_DELTA    # close when |Δ_short_put| ≥ this
    csp_stop_limit_offset_pct: float = 0.05
    csp_max_collateral: Optional[float] = None    # dollars; None = no cap
    wheel_screen: WheelScreenConfig = field(default_factory=WheelScreenConfig)


class LongTermEvaluator:
    """Skill 40's orchestrator. Walking-skeleton scope = covered calls only."""

    def __init__(
        self,
        *,
        positions_provider: PositionsProvider,
        call_chain_fetcher: ChainFetcher,
        preset: Any = None,
        config: Optional[EvaluatorConfig] = None,
        put_chain_fetcher: Optional[ChainFetcher] = None,
        fundamentals_fetcher: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
        spot_fetcher: Optional[Callable[[str], Optional[float]]] = None,
        trend_fetcher: Optional[Callable[[str], Optional[float]]] = None,
        sector_fetcher: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.positions_provider = positions_provider
        self.call_chain_fetcher = call_chain_fetcher
        self.preset = preset
        self.config = config or EvaluatorConfig()
        # Wheel entry (CSP) runs only when all three are supplied.
        self.put_chain_fetcher = put_chain_fetcher
        self.fundamentals_fetcher = fundamentals_fetcher
        self.spot_fetcher = spot_fetcher
        # Backlog §2 (2026-10-05). trend_fetcher(ticker) → 200-day SMA (or
        # None when unknown); sector_fetcher(ticker) → sector label. Both
        # optional: absent → that filter is not applied (legacy callers).
        self.trend_fetcher = trend_fetcher
        self.sector_fetcher = sector_fetcher
        # ticker → why no CSP was recommended (screen reasons, no chain,
        # no accepted contract). Reset on every recommend() call.
        self.last_diagnostics: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def recommend(self, watchlist: List[str]) -> List[Recommendation]:
        """Produce ranked recommendations for the (watchlist × holdings) join.

        Walking-skeleton scope: income overlay only. Tickers held with
        100+ shares get covered-call recommendations.

        Next session: entry vehicles (CSP / LEAPS), manage-existing,
        portfolio-gap suggestions.
        """
        positions = self.positions_provider.snapshot()
        held_qty = self._stock_qty_by_ticker(positions)
        held_cost_basis = self._stock_cost_basis_by_ticker(positions)

        wl_set = {t.strip().upper() for t in watchlist if t and t.strip()}
        held_set = set(held_qty.keys())

        recs: List[Recommendation] = []

        # § Income overlay — for tickers both held (≥ 100 shares) AND on watchlist.
        for ticker in sorted(wl_set & held_set):
            qty = held_qty[ticker]
            if qty < 100:
                logger.debug(
                    "Skipping CC for %s — qty=%d < 100 (no covered-call eligibility).",
                    ticker, qty,
                )
                continue
            cost_basis = held_cost_basis[ticker]
            recs.extend(self._recommend_covered_calls(ticker, qty, cost_basis))

        # § Wheel entry — CSPs on watchlist tickers not already held ≥ 100
        # shares (those are on the covered-call leg of the wheel above).
        self.last_diagnostics = {}
        if self.put_chain_fetcher and self.fundamentals_fetcher and self.spot_fetcher:
            for ticker in sorted(wl_set):
                if held_qty.get(ticker, 0) >= 100:
                    continue
                recs.extend(self._recommend_cash_secured_puts(ticker))

        recs.sort(key=lambda r: r.score, reverse=True)
        return self._cap_csp_per_sector(recs)

    def _cap_csp_per_sector(self, recs: List[Recommendation]) -> List[Recommendation]:
        """Keep CSPs from at most ``csp_max_per_sector`` distinct tickers
        per sector (best score first); several strikes of one kept ticker
        stay. Dropped tickers get a ``sector_cap`` diagnostic."""
        if self.sector_fetcher is None:
            return recs
        cap = int(getattr(self.preset, "csp_max_per_sector", 1) or 0)
        if cap <= 0:
            return recs
        kept: Dict[str, List[str]] = {}
        out: List[Recommendation] = []
        for r in recs:
            if r.strategy != "cash_secured_put":
                out.append(r)
                continue
            sector = self.sector_fetcher(r.ticker)
            tickers = kept.setdefault(sector, [])
            if r.ticker in tickers or len(tickers) < cap:
                if r.ticker not in tickers:
                    tickers.append(r.ticker)
                out.append(r)
            else:
                self.last_diagnostics.setdefault(r.ticker, [])
                note = f"sector_cap ({sector}: {', '.join(tickers)} ranked higher)"
                if note not in self.last_diagnostics[r.ticker]:
                    self.last_diagnostics[r.ticker].append(note)
        return out

    def _recommend_cash_secured_puts(self, ticker: str) -> List[Recommendation]:
        """Fundamentals screen → put chain → ``_score_cash_secured_put``.
        Exit anchors per skill 40 §2.6: BTC at 50 % of credit; close when
        |Δ_short_put| ≥ ``csp_stop_delta``."""
        cfg = self.config
        try:
            block = self.fundamentals_fetcher(ticker)
        except Exception as exc:           # noqa: BLE001 — fail closed for this ticker
            logger.warning("fundamentals_fetcher(%s) raised %s — skipping.", ticker, exc)
            block = None
        screen = screen_fundamentals(ticker, block, cfg.wheel_screen)
        if not screen.passed:
            self.last_diagnostics[ticker] = screen.reasons
            return []

        try:
            spot = self.spot_fetcher(ticker)
            chain = self.put_chain_fetcher(ticker) or []
        except Exception as exc:           # noqa: BLE001 — fail closed for this ticker
            logger.warning("CSP data fetch for %s raised %s — skipping.", ticker, exc)
            self.last_diagnostics[ticker] = ["data_unavailable"]
            return []
        if not spot or spot <= 0 or not chain:
            self.last_diagnostics[ticker] = ["no_spot" if not spot or spot <= 0 else "no_chain"]
            return []
        trend_reason = self._trend_block(ticker, float(spot))
        if trend_reason:
            self.last_diagnostics[ticker] = [trend_reason]
            return []

        scored: List[Tuple[float, Dict[str, float], Dict[str, Any]]] = []
        rejects: Dict[str, int] = {}
        for contract in chain:
            try:
                result = _score_cash_secured_put_with_reason(
                    short_put=contract, spot=spot, preset=self.preset,
                    max_collateral=cfg.csp_max_collateral,
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("CSP scorer rejected %s: %s", contract, exc)
                continue
            if result["status"] != "accepted":
                rejects[result["reason"]] = rejects.get(result["reason"], 0) + 1
                continue
            scored.append((float(result["score"]), result["metrics"], contract))

        if not scored:
            top = sorted(rejects.items(), key=lambda kv: -kv[1])[:3]
            self.last_diagnostics[ticker] = (
                [f"no_accepted_put ({', '.join(f'{r}×{n}' for r, n in top)})"]
                if top else ["no_accepted_put"])
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        recs: List[Recommendation] = []
        for score, metrics, contract in scored:
            if len(recs) >= cfg.csp_max_recommendations_per_ticker:
                break
            entry_credit = round(metrics["credit"] / 100.0, 2)
            take_profit = round(entry_credit * (1.0 - cfg.csp_take_profit_pct_of_credit), 2)
            if not (0.0 < take_profit < entry_credit):
                continue    # credit too small to express a 50% BTC in cents
            recs.append(Recommendation(
                ticker=ticker,
                strategy="cash_secured_put",
                legs=[RecommendationLeg(
                    action="STO",
                    occ_symbol=str(contract.get("symbol", "")),
                    qty=1,
                    side="short",
                    limit_price=entry_credit,
                )],
                entry_limit=entry_credit,
                entry_kind="credit",
                take_profit_limit=take_profit,
                stop_trigger=cfg.csp_stop_delta,
                stop_kind="delta_threshold",
                stop_limit_offset_pct=cfg.csp_stop_limit_offset_pct,
                score=score,
                rationale=self._cash_secured_put_rationale(metrics, contract, spot),
                metrics={**metrics, **self._contract_fields(contract), "spot": float(spot)},
            ))
        return recs

    def _trend_block(self, ticker: str, spot: float) -> Optional[str]:
        """Backlog §2: no CSP below the 200-day SMA. Fail closed when the
        trend is unknown — selling puts blind into a downtrend is the
        failure this filter exists to stop."""
        if self.trend_fetcher is None or not getattr(self.preset, "csp_require_above_sma200", True):
            return None
        try:
            sma200 = self.trend_fetcher(ticker)
        except Exception as exc:           # noqa: BLE001 — fail closed for this ticker
            logger.warning("trend_fetcher(%s) raised %s — skipping CSP.", ticker, exc)
            sma200 = None
        if sma200 is None or sma200 <= 0:
            return "trend_unavailable (200-day SMA unknown)"
        if spot < sma200:
            return f"below_200d_sma ({spot:.2f} < {sma200:.2f})"
        return None

    # ------------------------------------------------------------------
    # Strategy-specific assemblers
    # ------------------------------------------------------------------
    def _recommend_covered_calls(
        self,
        ticker: str,
        qty: int,
        cost_basis: float,
    ) -> List[Recommendation]:
        """Walk the call chain for ``ticker`` and emit CC recommendations.

        Each contract is scored via ``_score_covered_call_with_reason``;
        accepted contracts become a ``Recommendation`` with TP/SL anchors
        per skill 40 §2.6. The top-N (config-controlled) are returned.
        """
        try:
            chain = self.call_chain_fetcher(ticker) or []
        except Exception as exc:           # noqa: BLE001 — fail open
            logger.warning("call_chain_fetcher(%s) raised %s — skipping.", ticker, exc)
            return []

        contracts_per_lot = qty // 100   # number of CC contracts the operator could write

        scored: List[Tuple[float, Dict[str, float], Dict[str, Any]]] = []
        for contract in chain:
            try:
                result = _score_covered_call_with_reason(
                    short_call=contract,
                    cost_basis=cost_basis,
                    preset=self.preset,
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("CC scorer rejected %s: %s", contract, exc)
                continue
            if result["status"] != "accepted":
                continue
            scored.append((float(result["score"]), result["metrics"], contract))

        scored.sort(key=lambda x: x[0], reverse=True)

        recs: List[Recommendation] = []
        for score, metrics, contract in scored[: self.config.cc_max_recommendations_per_ticker]:
            entry_credit = metrics["credit"] / 100.0  # per-share back to per-contract limit
            recs.append(Recommendation(
                ticker=ticker,
                strategy="covered_call",
                legs=[RecommendationLeg(
                    action="STO",
                    occ_symbol=str(contract.get("symbol", "")),
                    qty=contracts_per_lot,
                    side="short",
                    limit_price=entry_credit,
                )],
                entry_limit=entry_credit,
                entry_kind="credit",
                take_profit_limit=round(
                    entry_credit * (1.0 - self.config.cc_take_profit_pct_of_credit), 2,
                ),
                stop_trigger=round(cost_basis * self.config.cc_stop_underlying_pct, 2),
                stop_kind="underlying_price",
                stop_limit_offset_pct=self.config.cc_stop_limit_offset_pct,
                score=score,
                rationale=self._covered_call_rationale(metrics, contract, contracts_per_lot),
                metrics={**metrics, **self._contract_fields(contract), "cost_basis": cost_basis},
            ))
        return recs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _stock_qty_by_ticker(positions: List[Position]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for p in positions:
            if p.kind == "stock":
                out[p.ticker] = out.get(p.ticker, 0) + p.qty
        return out

    @staticmethod
    def _stock_cost_basis_by_ticker(positions: List[Position]) -> Dict[str, float]:
        # Qty-weighted average cost across lots (skill 40 §4).
        totals: Dict[str, Tuple[int, float]] = {}
        for p in positions:
            if p.kind != "stock":
                continue
            qty, weighted = totals.get(p.ticker, (0, 0.0))
            qty_new = qty + p.qty
            weighted_new = weighted + p.qty * p.avg_cost
            totals[p.ticker] = (qty_new, weighted_new)
        return {
            t: (weighted / qty) if qty > 0 else 0.0
            for t, (qty, weighted) in totals.items()
        }

    @staticmethod
    def _contract_fields(contract: Dict[str, Any]) -> Dict[str, float]:
        """Quote fields a staged plan needs (wheel_policy.build_single_leg_plan)."""
        return {
            "strike": float(contract.get("strike", 0.0)),
            "delta": float(contract.get("delta", 0.0)),
            "bid": float(contract.get("bid", 0.0) or 0.0),
            "ask": float(contract.get("ask", 0.0) or 0.0),
        }

    @staticmethod
    def _cash_secured_put_rationale(
        metrics: Dict[str, float],
        contract: Dict[str, Any],
        spot: float,
    ) -> str:
        strike = float(contract["strike"])
        return (
            f"Sell 1× CSP @ ${strike:g} ({int(metrics['dte'])}d) on ${spot:.2f} spot, "
            f"|Δ|={metrics['short_delta_abs']:.2f}, POP {metrics['pop'] * 100:.0f}%, "
            f"ann. yield {metrics['annualised_return'] * 100:.1f}%. "
            f"Credit ${metrics['credit']:.2f}; collateral ${metrics['collateral']:,.0f}; "
            f"if assigned, cost basis ${metrics['effective_entry']:.2f} "
            f"({metrics['discount_to_spot'] * 100:.1f}% below spot)."
        )

    @staticmethod
    def _covered_call_rationale(
        metrics: Dict[str, float],
        contract: Dict[str, Any],
        contracts_per_lot: int,
    ) -> str:
        ann_pct = metrics["annualised_return"] * 100.0
        delta = metrics["short_delta_abs"]
        dte = int(metrics["dte"])
        pop_pct = metrics["pop"] * 100.0
        strike = float(contract["strike"])
        return (
            f"Sell {contracts_per_lot}× CC @ ${strike:g} ({dte}d), "
            f"|Δ|={delta:.2f}, ann. yield {ann_pct:.1f}%, "
            f"POP {pop_pct:.0f}%. Credit ${metrics['credit']:.2f}/contract."
        )
