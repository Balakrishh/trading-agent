"""
strategy_presets.py — Risk-profile presets for the live trading agent.

Bundles the four knobs that meaningfully change credit-spread economics
(``max_delta``, DTE per strategy, spread-width policy, and the C/W floor)
into three ready-made profiles plus a Custom slot for manual tuning.

The active preset is persisted in ``STRATEGY_PRESET.json`` at the repo
root (next to ``AGENT_RUNNING``). The Streamlit dashboard writes that
file from the Strategy-Profile panel; the agent subprocess reads it at
the start of every cycle, so changes apply on the next 5-minute tick
without restarting anything.

If the file is missing, malformed, or the named profile is unknown the
loader falls back to ``BALANCED`` — the documented out-of-the-box
default. That keeps the agent operational when the dashboard hasn't
been touched yet (fresh installs, CI, smoke tests).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Repo-root sentinel file (matches AGENT_RUNNING / DRY_RUN_MODE pattern).
PRESET_FILE = Path("STRATEGY_PRESET.json")


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ProfileName = Literal["conservative", "balanced", "aggressive", "custom"]
DirectionalBias = Literal["auto", "bullish_only", "bearish_only", "neutral_only"]

# Width policy: either a fraction of spot ("pct") or a fixed dollar amount.
WidthMode = Literal["pct_of_spot", "fixed_dollar"]

# Strategy planner mode:
#   "static"   — single (Δ, DTE, width) point from the preset's scalar fields,
#                C/W gated by ``min_credit_ratio``. Original behaviour.
#   "adaptive" — scan a grid of (DTE × Δ × width) tuples, score each by
#                ``EV_per_$risked = (POP×C/W − (1−POP)×(1−C/W)) / (1−C/W)``,
#                pick the highest-scoring candidate that clears the
#                breakeven-plus-edge_buffer floor, OR sit out if none do.
ScanMode = Literal["static", "adaptive"]


FILL_MODELS = ("natural", "mid")


@dataclass(frozen=True)
class PresetConfig:
    """Concrete trading parameters that drive Strategy + RiskManager."""

    name:                  str
    max_delta:             float            # short-leg |Δ| ceiling
    dte_vertical:          int              # Bull Put / Bear Call DTE
    dte_iron_condor:       int              # Iron Condor DTE
    dte_mean_reversion:    int              # Mean-reversion DTE
    dte_window_days:       int              # ± window around target DTE
    width_mode:            WidthMode        # pct_of_spot | fixed_dollar
    width_value:           float            # 0.015 = 1.5% spot; or 5.0 = $5
    min_credit_ratio:      float            # C/W floor (static mode)
    max_risk_pct:          float            # account-fraction risk cap
    directional_bias:      DirectionalBias = "auto"
    description:           str = ""

    # ------------------------------------------------------------------
    # Adaptive-scan knobs (only consulted when scan_mode == "adaptive").
    # In static mode these are ignored — the scalar dte_*/max_delta/
    # width_* fields above are authoritative.
    # ------------------------------------------------------------------
    scan_mode:             ScanMode = "static"
    # Required EV-over-breakeven margin. The scanner demands
    #   C/W ≥ |Δshort| × (1 + edge_buffer)
    # because POP ≈ 1 − |Δ|, so |Δ| is the breakeven C/W. 0.10 = "demand
    # 10 % over breakeven before firing", which keeps the strategy
    # self-consistent across delta/DTE/width choices.
    edge_buffer:           float = 0.10
    # POP floor — refuse any candidate with implied POP below this regardless
    # of credit. 0.55 admits Δ-0.45 in extreme regimes; 0.65 is "no aggressive
    # naked-short approximations". Default chosen to match Aggressive preset
    # tail behaviour without being too restrictive.
    min_pop:               float = 0.55
    # Grids the scanner sweeps. Tuples (immutable so dataclass(frozen=True)
    # accepts them). DTE in calendar days, delta as |Δ|, width as fraction
    # of spot.
    dte_grid:              Tuple[int, ...]   = (7, 14, 21, 30)
    delta_grid:            Tuple[float, ...] = (0.20, 0.25, 0.30, 0.35)
    width_grid_pct:        Tuple[float, ...] = (0.010, 0.015, 0.020, 0.025)

    # ------------------------------------------------------------------
    # Per-leg liquidity gate (added 2026-05-15 — see skill 29).
    # Rejects candidates where ANY single leg has a bid-ask spread wider
    # than the looser of two thresholds. Catches wide-spread option
    # chains (typically GLD, high-spot index ETFs) whose day-1 mark
    # would otherwise show 50%+ of credit eaten by bid-ask drag.
    #
    # Defaults are permissive — SPY/QQQ/IWM/XLF chains pass easily
    # (their per-leg spreads are 2–8 ¢). The 2026-05-15 GLD trade
    # (each leg 25–35 ¢, 5–8 % of mid) would have been rejected.
    # ------------------------------------------------------------------
    # Absolute spread cap (dollars). Reject if (ask − bid) > X for any leg.
    max_leg_spread_cents:    float = 0.15
    # Relative spread cap (fraction of mid). Reject if (ask − bid)/mid > X.
    # Applied alongside the absolute cap — both must pass.
    max_leg_spread_pct_mid:  float = 0.05

    # ------------------------------------------------------------------
    # Fill model (added 2026-10-05 — backlog §6.1, skill 03 §4).
    # How credits / debits are estimated everywhere a candidate is scored:
    #   "natural" — sell at the bid, buy at the ask (what the Alpaca paper
    #               account actually filled at: 3 of 3 fills in week 1).
    #   "mid"     — legacy NBBO mid minus DEFAULT_FILL_HAIRCUT.
    # Also selects the price the profit target is judged at (natural =
    # cost to buy back at the ask). Stops stay on the mid valuation.
    # ------------------------------------------------------------------
    fill_model:              str   = "natural"

    # ------------------------------------------------------------------
    # Market risk-state overlay (added 2026-10-05 — backlog §1/§6.2,
    # skill 58). When True the agent classifies the whole market once per
    # cycle (NORMAL / CAUTION / DEFENSIVE / CAPITULATION / RECOVERY) and
    # gates new entries by it: size multiplier on max_risk_pct, allowed
    # strategies, and the Wheel cash-secured-put pause. Exits are never
    # gated. False = legacy behaviour (per-ticker regime only).
    # ------------------------------------------------------------------
    market_state_enabled:    bool  = True

    # ------------------------------------------------------------------
    # Profit-target management (added 2026-05-19 — see skill 30).
    # Close any position whose unrealized P&L ≥ profit_target_pct × initial
    # credit. The 50% default is the industry-standard early-exit rule
    # (see Tastytrade studies, "manage winners at 50%"). Aggressive presets
    # take profit sooner to recycle capital faster; conservative presets
    # ride winners further because they fire less often and want more $/trade.
    #
    # Wired into PositionMonitor's PROFIT_TARGET exit signal (live) and
    # backtest/runner.py:_handle_intraday_decision (backtest parity).
    # ------------------------------------------------------------------
    profit_target_pct:       float = 0.50

    # ------------------------------------------------------------------
    # Defensive-roll management (added 2026-05-19 — see skill 31).
    # When a position's short strike approaches spot, instead of just
    # closing on STRIKE_PROXIMITY, ROLL the spread: close threatened
    # spread + open new further-OTM spread atomically. Six predicates
    # (see defensive_roll_evaluator.evaluate_defensive_roll) gate the
    # decision so the roll only fires when it's economically positive.
    #
    # OFF by default — opt-in per preset because defensive rolls are
    # higher-stakes than profit-takes (mistakes lock in bigger losses).
    # ------------------------------------------------------------------
    defensive_roll_enabled:            bool  = False
    # Proximity band: short strike must be between roll_trigger_min_pct
    # and roll_trigger_max_pct of spot. Below min → new strike also in
    # danger; above max → not actually threatened enough to roll.
    roll_trigger_min_pct:              float = 0.005    # 0.5% of spot
    roll_trigger_max_pct:              float = 0.015    # 1.5% of spot
    # Minimum DTE remaining for a roll to be viable. Inside this window
    # gamma is too explosive to roll usefully — just close.
    min_dte_for_roll:                  int   = 5
    # Hard cap on consecutive defensive rolls per position. Without a
    # cap, a trending market rolls the same position 3-4 times, each
    # at a worse strike, compounding the original loss.
    max_defensive_rolls_per_position:  int   = 1

    # ------------------------------------------------------------------
    # Iron Butterfly (added 2026-07-02 — see skill 45).
    # Structurally distinct from Iron Condor: both shorts sit ATM
    # (Δ≈0.5), profit zone = [K−C, K+C], POP ≈ 2C/W. Ships DISABLED so
    # the scorer is available for backtest sign-off before it can
    # affect any live cycle. Enable per-preset once backtest shows
    # meaningful edge over the credit-spread baseline in the current
    # regime.
    # ------------------------------------------------------------------
    iron_butterfly_enabled:            bool  = False
    # POP floor for IB — looser than IC's default (0.55) because IB's
    # profit zone is inherently narrower and the "credit-scaled POP"
    # approximation biases conservatively. 0.40 admits candidates
    # collecting ≥ 40% of wing width as credit.
    iron_butterfly_min_pop:            float = 0.40
    # DTE grid for IB — favors LONGER DTE than verticals because the
    # profit zone widens with credit collected, and credit scales with
    # implied vol × sqrt(DTE). Below 21 DTE the pin-precision required
    # to be profitable is too tight; above 45 DTE gamma is manageable.
    iron_butterfly_dte_grid:           Tuple[int, ...] = (21, 30, 45)
    # Wing widths as fraction of spot. Wider than verticals — a 2-3%
    # wing on SPY at $735 = $15-22 wing width, enough for the profit
    # zone [K−C, K+C] to be meaningfully wide even at modest credit.
    iron_butterfly_wing_width_pct:     Tuple[float, ...] = (0.020, 0.030, 0.040)

    # ------------------------------------------------------------------
    # Broken-Wing Butterfly (added 2026-07-05 — see skill 46).
    # Asymmetric-wing IB. Direction inferred from which wing is wider
    # (put wing wider → bullish; call wing wider → bearish). Same
    # safety posture as IB: ships DISABLED, backtest first.
    # ------------------------------------------------------------------
    broken_wing_butterfly_enabled:     bool  = False
    # POP floor for BWB — matches IB default; the average-wing POP
    # normalization keeps the numerics on the same scale as IB.
    broken_wing_butterfly_min_pop:     float = 0.40
    # DTE grid — same as IB.
    broken_wing_butterfly_dte_grid:    Tuple[int, ...] = (21, 30, 45)
    # Wing-width grids per side. Symmetric grid: caller sweeps put_wing_pct
    # × call_wing_pct to build asymmetric combinations. Grid entries in
    # ascending order for readable log output.
    broken_wing_butterfly_wing_width_pct: Tuple[float, ...] = (0.020, 0.030, 0.040, 0.060)

    # ------------------------------------------------------------------
    # PDT-aware DTE cap (added 2026-05-20 — see skill 33).
    # When the trading account is sub-$25K (FINRA Pattern Day Trader
    # threshold), same-day closes get blocked by Alpaca code 40310100.
    # The longer the DTE, the more time the underlying has to drift to
    # a short strike before the next trading day's PDT-eligible close
    # window. Capping DTE shorter on PDT-restricted accounts trades
    # some theta runway for less drift risk over a single overnight.
    #
    # When ``account_balance >= PDT_EQUITY_THRESHOLD`` (>= $25K) the
    # cap is IGNORED — full preset DTE applies. The cap only engages
    # for sub-$25K accounts where same-day close failures actually hurt.
    # ------------------------------------------------------------------
    pdt_dte_cap:                       int   = 14

    # ------------------------------------------------------------------
    # Auto-promote caps for Claude-Code-proposed orders (skill 55).
    #
    # When the operator invokes ``python -m trading_agent.executor.promote``
    # on a proposal in ``pending_orders/``, the promote CLI can either
    # wait for ``--yes`` (manual) or bypass the prompt (auto-promote).
    # Auto-promote requires ALL of the following to hold:
    #
    #   1. ``TRADING_AGENT_AUTO_PROMOTE_ENABLED`` env master switch = true
    #   2. debit_notional ≤ auto_promote_max_notional_usd
    #   3. contract_count ≤ auto_promote_max_contracts
    #   4. strategy ∈ auto_promote_allowed_strategies
    #   5. risk_manager.check() returns zero WARNINGS (not just errors)
    #   6. within regular market hours (not first/last 5 min)
    #   7. re-score delta from proposal-time < 5%
    #
    # Defaults are the SAFE end of every choice — max_notional=0 disables
    # the config-side gate even if the env master switch is on. Operator
    # opts in per-preset by editing these fields (frozen dataclass →
    # dataclasses.replace, per skill 00 conventions).
    # ------------------------------------------------------------------
    auto_promote_max_notional_usd:     float = 0.0
    auto_promote_max_contracts:        int   = 1
    auto_promote_allowed_strategies:   Tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Auto-apply caps for LLM-proposed preset updates (skill 56).
    #
    # After each daily_reviewer run, an LLM may propose PresetConfig
    # diffs (e.g. raise profit_target_pct from 0.50 → 0.55). By default
    # the operator applies these manually via
    # ``python -m trading_agent.apply_preset_update <uuid>``.
    #
    # When ``auto_apply_preset_updates_enabled`` is set AND the proposed
    # change stays within the allowlisted fields AND the delta is under
    # the per-field size cap, the apply CLI can auto-persist without
    # prompting. All three fields default to the SAFE end.
    # ------------------------------------------------------------------
    auto_apply_preset_updates_enabled: bool  = False
    auto_apply_max_delta_change_pct:   float = 0.0    # e.g. 0.10 = up to 10 %
    auto_apply_allowed_fields:         Tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Debit structures + bounce bull put (added 2026-10-05 — backlog
    # §6.3–6.5, skill 59). The playbook table (skill 58) routes trend +
    # low volatility to call / put debit spreads, sideways + low
    # volatility to calendars, and bearish + RSI < 30 + elevated
    # volatility to a bull put once price stabilises. Scoring caps the
    # debit at the market mid × (1 + debit_max_overpay) and requires
    # reward/risk ≥ debit_min_reward_risk.
    # ------------------------------------------------------------------
    debit_spreads_enabled:             bool  = True
    calendar_enabled:                  bool  = True
    bounce_bull_put_enabled:           bool  = True
    # Bear call spreads (backlog B6, 2026-10-14). The 2019–2026 calibration
    # study (docs/plans/research_pop_calibration.md) found short calls
    # survived 2–6 points LESS often than implied odds, even in down-trends,
    # while short puts beat them by 9–11. Off: bearish tickers trade put
    # debit spreads (any volatility), and the VIX inhibit no longer turns
    # bullish setups into bear calls. True restores the old routing.
    bear_call_enabled:                 bool  = False
    dte_debit:                         int   = 30
    debit_long_delta:                  float = 0.50     # |Δ| of the bought leg; sold leg one width_grid_pct step out
    debit_max_overpay:                 float = 0.05     # debit ≤ market mid × 1.05 (liquidity cost)
    debit_min_reward_risk:             float = 1.0      # max profit ÷ debit
    debit_profit_target_pct:           float = 0.50     # verticals; measured against debit_profit_target_basis
    debit_stop_loss_pct:               float = 0.50     # of the debit (verticals + calendars)
    dte_calendar_near:                 int   = 21       # sold leg
    calendar_gap_days:                 int   = 28       # bought leg ≈ near + gap
    calendar_option_type:              str   = "call"
    calendar_profit_target_pct:        float = 0.25     # of the debit
    calendar_max_strike_drift_pct:     float = 0.03     # close when the underlying is ≥ 3 % from the strike
    bounce_lookback_days:              int   = 5        # stabilisation = reclaim the N-day high close
    # What debit_profit_target_pct is a percentage of (2026-10-08):
    # "debit" — of what was paid (0.50 → close at +50 % on cost);
    # "max_profit" — of width − debit (the pre-2026-10-08 rule; ≈ +80 % on
    # cost for a 1.6:1 spread, and IWM's +$444 peak never reached it).
    debit_profit_target_basis:         str   = "debit"
    # Mid-first entry ladder for debit verticals (backlog C1, 2026-10-16,
    # skill 59 §3). Orders still go out at natural; "shadow" measures
    # what offering mid → cap in debit_ladder_steps prices, each held
    # debit_ladder_step_cycles cycles, would have saved or unlocked
    # (``debit_ladder_shadow`` journal rows). "live" is not built yet.
    debit_ladder_mode:                 str   = "shadow"
    debit_ladder_steps:                int   = 4
    debit_ladder_step_cycles:          int   = 1
    # Put-debit fallback skipped for these tickers (E7, backlog 11.4,
    # 2026-10-26): their legs quoted 30–90¢ wide on 2026-10-21 (limit 15¢ /
    # 5 % of mid), so every bearish cycle ended in leg_spread_wide — about
    # 940 rejections that day. Re-measure quarterly.
    put_debit_excluded_tickers:        Tuple[str, ...] = ("XLY", "XLK", "XLI", "GDX")

    # ------------------------------------------------------------------
    # Total open-risk cap (added 2026-10-05). Σ max loss of all open
    # defined-risk positions (plus anything submitted earlier in the same
    # cycle) may not exceed this fraction of equity; each new trade is
    # sized into the remaining budget. Wheel legs are excluded (their
    # collateral has its own 40 %-of-equity rule). First live debit cycle
    # opened 3 positions = 7.2 % of equity at once with no such limit.
    # ------------------------------------------------------------------
    max_total_risk_pct:                float = 0.10

    # ------------------------------------------------------------------
    # Wheel (skill 40 §3.3, wired 2026-10-05 — backlog §2). Defaults equal
    # the decision_engine fallbacks the scorers used before, so the
    # wiring changes nothing until edited. csp_require_above_sma200: no
    # cash-secured put on a stock below its 200-day average (fail closed
    # when the trend is unknown). csp_max_per_sector: distinct tickers per
    # sector in one wheel_screen (banks and telecom clustered).
    # ------------------------------------------------------------------
    cc_max_short_delta:                float = 0.30
    cc_dte_band:                       Tuple[int, int] = (30, 60)
    cc_min_iv_rank:                    float = 0.25
    csp_max_short_delta:               float = 0.30
    csp_dte_band:                      Tuple[int, int] = (21, 60)
    csp_min_iv_rank:                   float = 0.25
    csp_strike_band:                   Tuple[float, float] = (0.85, 0.97)
    csp_require_above_sma200:          bool  = True
    csp_max_per_sector:                int   = 1

    # ------------------------------------------------------------------
    # Laddering (added 2026-10-05 — backlog §6.7). Up to this many open
    # positions per underlying; a 2nd+ entry must be on a later day than
    # the ticker's last entry and expire ≥ ladder_min_gap_days from every
    # open position on it. 1 = legacy single position per ticker.
    # ------------------------------------------------------------------
    max_positions_per_ticker:          int   = 2
    ladder_min_gap_days:               int   = 7
    # Add to winners only (2026-10-06): a 2nd+ position on a ticker also
    # needs every open position on it at or above breakeven.
    ladder_requires_profit:            bool  = True
    # Cool-down after a stop (2026-10-16): a stop-type close (stop_loss,
    # hard_stop, delta_stop, strike_proximity) blocks new entries on that
    # ticker in the same direction for the rest of the stop's session and
    # the next N sessions. 0 = off. TLT on 2026-10-15 was stopped at 09:46
    # ET and re-bought on the same bearish view at 09:50.
    # 2026-10-26: 1 → 5 sessions. TLT was stopped 10-15, re-entered 10-19
    # (2 sessions later, allowed by the 1-session rule) and stopped again
    # 10-22. A second same-direction stop within
    # stop_cooldown_repeat_window sessions blocks for
    # stop_cooldown_repeat_sessions instead.
    stop_cooldown_sessions:            int   = 5
    stop_cooldown_repeat_sessions:     int   = 10
    stop_cooldown_repeat_window:       int   = 20
    # Correlated-risk groups (E6, backlog 11.2, 2026-10-26; groups in
    # position_caps.RISK_GROUPS: rates, gold, us_equity). Max loss of open
    # + new positions in one direction may not exceed these shares of
    # equity per group and across the book; a trade is sized into the
    # room left and blocked below one contract. 0 = off.
    max_group_direction_risk_pct:      float = 0.03
    max_book_direction_risk_pct:       float = 0.06
    # Book net delta (E4, backlog 11.1): beta-weighted SPY-equivalent
    # dollar delta of the book as a share of equity. "shadow" journals
    # what a cap at max_net_beta_delta_pct would block (net_delta_shadow
    # rows); "off" skips it. "live" is not built.
    net_delta_mode:                    str   = "shadow"
    max_net_beta_delta_pct:            float = 0.30

    # ------------------------------------------------------------------
    # Entry confirmation (added 2026-10-05). An approved plan is submitted
    # only after the same ticker produced the same strategy + expiry on
    # entry_confirm_cycles consecutive cycles (≈ 75 s apart); no entries
    # before no_entry_before_et; at most max_new_entries_per_hour
    # submissions in a rolling hour (0 = no limit).
    # ------------------------------------------------------------------
    # Drawdown governor (trading_halt.py, 2026-10-07): pause NEW entries
    # (exits keep running) when equity is this far below the day's / the
    # week's first reading; only a human resumes. 0 = off.
    halt_daily_loss_pct:               float = 0.02
    halt_weekly_loss_pct:              float = 0.05
    entry_confirm_cycles:              int   = 3
    no_entry_before_et:                str   = "09:45"
    # Exits that are debounced (profit target, regime shift, delta stop,
    # debit stop, strike drift) wait until this ET time; the open's wide
    # quotes made the 2026-10-09 VZ delta stop fill at $1.18 vs a $1.045
    # mark at 09:36. Immediate exits (hard stop, strike proximity, DTE
    # safety) are never delayed. Empty = no window (skill 30 §4).
    no_exit_before_et:                 str   = "09:45"
    max_new_entries_per_hour:          int   = 1
    # "sector" (default, 2026-10-06): the hourly limit counts per sector —
    # an AMZN entry holds back other Consumer Discretionary tickers for the
    # hour, not the whole watchlist. "global": one count for everything.
    entry_rate_scope:                  str   = "sector"
    # Entry timing (entry_confirmation.py): once confirmed, enter at the
    # best-scored price of the window (score within entry_best_tolerance_pct
    # of the best and natural-to-mid gap ≤ the window median); after
    # entry_max_wait_cycles enter only if no more than entry_chase_limit_pct
    # worse than at confirmation, else skip. "shadow" enters at
    # confirmation and records what the rule would have done.
    entry_timing_mode:                 str   = "shadow"
    entry_max_wait_cycles:             int   = 8
    entry_best_tolerance_pct:          float = 0.01
    entry_chase_limit_pct:             float = 0.03

    # ------------------------------------------------------------------
    # Trailing profit-taking (added 2026-10-05, profit_trail.py). At the
    # profit target the trail arms instead of closing; it closes at the
    # ceiling, after giving back trail_giveback_pct of the peak (never
    # below the credit floor), or at ≤ trail_max_hold_dte days to expiry.
    # "shadow" keeps the legacy close and records what the trail would
    # have done; "live" lets the trail decide; "off" disables it.
    # ------------------------------------------------------------------
    profit_trail_mode:                 str   = "shadow"
    trail_giveback_pct:                float = 0.25
    trail_credit_floor_pct:            float = 0.40     # of the credit
    trail_credit_ceiling_pct:          float = 0.75     # of the credit
    trail_debit_ceiling_pct:           float = 0.90     # of max profit
    trail_calendar_ceiling_pct:        float = 0.40     # of the debit
    trail_max_hold_dte:                int   = 7

    # ------------------------------------------------------------------
    # Earnings-aware expiries (skill 65, added 2026-10-15). A single-stock
    # trade must expire at least ``earnings_buffer_days`` before the next
    # earnings date: the planner picks the latest weekly that fits, and
    # only when none is ≥ ``earnings_min_dte`` days out does the ticker
    # wait until after the report. ETFs are never affected.
    # ------------------------------------------------------------------
    earnings_guard_enabled:            bool  = True
    earnings_buffer_days:              int   = 1
    earnings_min_dte:                  int   = 7

    # ------------------------------------------------------------------
    # Trend entry filters (skill 65, added 2026-10-15) for directional
    # trades. Relative strength: the ticker's 20/60-day return minus
    # SPY's, ranked across the watchlist — bullish trades need a
    # percentile ≥ rs_min_percentile, bearish ≤ 1 − it. Pullback: a
    # bullish trade needs price above the 50-day SMA and no more than
    # pullback_max_atr × ATR(14) above the 20-day SMA (mirror for
    # bearish). "shadow" records the verdict without blocking; "live"
    # blocks; "off" skips the check.
    # ------------------------------------------------------------------
    rs_filter_mode:                    str   = "shadow"
    rs_min_percentile:                 float = 0.50
    pullback_filter_mode:              str   = "shadow"
    pullback_max_atr:                  float = 1.0

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def _wheel_tag(self) -> str:
        """Summary-line token for the Wheel tunables (skill 40 §3.3)."""
        return (f"Wheel CSP Δ≤{self.csp_max_short_delta:.2f} "
                f"{self.csp_dte_band[0]}–{self.csp_dte_band[1]}d "
                f"K {self.csp_strike_band[0]:.0%}–{self.csp_strike_band[1]:.0%} "
                f"{'>200d ' if self.csp_require_above_sma200 else ''}"
                f"{self.csp_max_per_sector}/sector • CC Δ≤{self.cc_max_short_delta:.2f} "
                f"{self.cc_dte_band[0]}–{self.cc_dte_band[1]}d")

    def _debit_tag(self) -> str:
        """Summary-line token for the skill-59 debit playbooks."""
        parts = []
        if self.debit_spreads_enabled:
            parts.append(f"Debit@{self.dte_debit}d Δ{self.debit_long_delta:.2f} "
                         f"RR≥{self.debit_min_reward_risk:g} "
                         f"TP{self.debit_profit_target_pct:.0%} of "
                         f"{'cost' if self.debit_profit_target_basis == 'debit' else 'max'}"
                         f"/SL{self.debit_stop_loss_pct:.0%} "
                         f"Ladder {self.debit_ladder_mode}"
                         f"{'' if self.debit_ladder_mode == 'off' else f' {self.debit_ladder_steps}×{self.debit_ladder_step_cycles}c'}")
        if self.calendar_enabled:
            parts.append(f"Cal@{self.dte_calendar_near}+{self.calendar_gap_days}d "
                         f"TP{self.calendar_profit_target_pct:.0%} drift≤{self.calendar_max_strike_drift_pct:.0%}")
        if self.bounce_bull_put_enabled:
            parts.append(f"Bounce@{self.bounce_lookback_days}d")
        parts.append("BearCall on" if self.bear_call_enabled else "BearCall off→put debit")
        return " • ".join(parts) if parts else "Debit off"

    def _trend_tag(self) -> str:
        """Summary-line token for the skill-65 earnings guard + filters."""
        earn = (f"Earnings exp ≤ report−{self.earnings_buffer_days}d (≥{self.earnings_min_dte}d)"
                if self.earnings_guard_enabled else "Earnings guard OFF")
        return (f"{earn} • RS {self.rs_filter_mode} ≥p{self.rs_min_percentile:.0%} • "
                f"Pullback {self.pullback_filter_mode} ≤{self.pullback_max_atr:g}ATR")

    @property
    def dte_range_vertical(self) -> Tuple[int, int]:
        return (max(1, self.dte_vertical - self.dte_window_days),
                self.dte_vertical + self.dte_window_days)

    @property
    def dte_range_iron_condor(self) -> Tuple[int, int]:
        return (max(1, self.dte_iron_condor - self.dte_window_days),
                self.dte_iron_condor + self.dte_window_days)

    @property
    def dte_range_mean_reversion(self) -> Tuple[int, int]:
        return (max(1, self.dte_mean_reversion - self.dte_window_days),
                self.dte_mean_reversion + self.dte_window_days)

    def _stop_cooldown_tag(self) -> str:
        if not self.stop_cooldown_sessions:
            return "Stop cool-down off"
        return (f"Stop cool-down {self.stop_cooldown_sessions} sess"
                f" ({self.stop_cooldown_repeat_sessions} after a repeat)")

    def _direction_risk_tag(self) -> str:
        excl = ",".join(self.put_debit_excluded_tickers) or "none"
        return (f"Dir risk ≤{self.max_group_direction_risk_pct:.0%}/group "
                f"≤{self.max_book_direction_risk_pct:.0%}/book • "
                f"NetΔ {self.net_delta_mode} ≤{self.max_net_beta_delta_pct:.0%} • "
                f"No put debit: {excl}")

    def to_summary_line(self) -> str:
        """Detailed one-liner for the agent log + dashboard expander body."""
        wstr = (f"{self.width_value*100:.1f}%spot"
                if self.width_mode == "pct_of_spot"
                else f"${self.width_value:.0f}")
        roll_tag = (
            f"DefRoll@{self.roll_trigger_min_pct*100:.1f}–"
            f"{self.roll_trigger_max_pct*100:.1f}%"
            if self.defensive_roll_enabled else "DefRoll OFF"
        )
        if self.scan_mode == "adaptive":
            return (
                f"{self.name.title()} • {self.directional_bias.replace('_', ' ')} • "
                f"ADAPTIVE scan • DTE∈{list(self.dte_grid)} Δ∈{list(self.delta_grid)} "
                f"w∈{[f'{w*100:.1f}%' for w in self.width_grid_pct]} • "
                f"Edge ≥ {self.edge_buffer:.0%} • POP ≥ {self.min_pop:.0%} • "
                f"LegSpread ≤ ${self.max_leg_spread_cents:.2f}/"
                f"{self.max_leg_spread_pct_mid:.0%}mid • "
                f"Fills @ {self.fill_model} • "
                f"MarketState {'on' if self.market_state_enabled else 'off'} • "
                f"{self._debit_tag()} • "
                f"{self._wheel_tag()} • "
                f"Ladder {self.max_positions_per_ticker}/ticker ≥{self.ladder_min_gap_days}d{' winners' if self.ladder_requires_profit else ''} • "
                f"{self._stop_cooldown_tag()} • "
                f"{self._direction_risk_tag()} • "
                f"Halt −{self.halt_daily_loss_pct:.0%}/d −{self.halt_weekly_loss_pct:.0%}/wk • "
                f"Confirm {self.entry_confirm_cycles}× from {self.no_entry_before_et} ≤{self.max_new_entries_per_hour}/h{'/sector' if self.entry_rate_scope == 'sector' else ''} • "
                f"Exits from {self.no_exit_before_et or 'open'} • "
                f"Timing {self.entry_timing_mode} ≤{self.entry_max_wait_cycles}c • "
                f"{self._trend_tag()} • "
                f"Trail {self.profit_trail_mode} −{self.trail_giveback_pct:.0%} • "
                f"Profit-take @ {self.profit_target_pct:.0%} • "
                f"{roll_tag} • "
                f"Max risk {self.max_risk_pct*100:.0f}% "
                f"(total {self.max_total_risk_pct*100:.0f}%)"
            )
        return (
            f"{self.name.title()} • {self.directional_bias.replace('_', ' ')} • "
            f"Vert@{self.dte_vertical}d Δ-{self.max_delta:.2f} w={wstr} • "
            f"IC@{self.dte_iron_condor}d • MR@{self.dte_mean_reversion}d • "
            f"C/W ≥ {self.min_credit_ratio} • "
            f"Fills @ {self.fill_model} • "
            f"MarketState {'on' if self.market_state_enabled else 'off'} • "
            f"{self._debit_tag()} • "
            f"{self._wheel_tag()} • "
            f"Ladder {self.max_positions_per_ticker}/ticker ≥{self.ladder_min_gap_days}d{' winners' if self.ladder_requires_profit else ''} • "
                f"{self._stop_cooldown_tag()} • "
                f"{self._direction_risk_tag()} • "
            f"Halt −{self.halt_daily_loss_pct:.0%}/d −{self.halt_weekly_loss_pct:.0%}/wk • "
            f"Confirm {self.entry_confirm_cycles}× from {self.no_entry_before_et} ≤{self.max_new_entries_per_hour}/h{'/sector' if self.entry_rate_scope == 'sector' else ''} • "
                f"Exits from {self.no_exit_before_et or 'open'} • "
            f"Timing {self.entry_timing_mode} ≤{self.entry_max_wait_cycles}c • "
            f"{self._trend_tag()} • "
            f"Trail {self.profit_trail_mode} −{self.trail_giveback_pct:.0%} • "
            f"Profit-take @ {self.profit_target_pct:.0%} • "
            f"{roll_tag} • "
            f"Max risk {self.max_risk_pct*100:.0f}% "
            f"(total {self.max_total_risk_pct*100:.0f}%)"
        )

    def to_short_summary(self) -> str:
        """Concise one-liner for the COLLAPSED Strategy Profile expander.

        The full ``to_summary_line()`` runs ~180 chars and is hard to
        scan at a glance.  This version trims to the four numbers an
        operator actually looks at in the expander label:

            <Profile> · IC <D>d · Δ <max_delta> · <risk%>

        Examples::

            Custom · IC 21d · Δ 0.25 · 5% risk
            Balanced · IC 35d · Δ 0.25 · 2% risk · ADAPT

        ADAPT suffix makes the scan-mode visible without expanding the
        details panel — useful since adaptive vs static is a meaningful
        operator choice.
        """
        scan = " · ADAPT" if self.scan_mode == "adaptive" else ""
        return (
            f"{self.name.title()} · IC {self.dte_iron_condor}d · "
            f"Δ {self.max_delta:.2f} · {self.max_risk_pct*100:g}% risk{scan}"
        )


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------
# Numbers reflect the design discussion in the README parity matrix.
# Conservative aims for ~85% POP on the short leg and accepts thinner credits;
# Aggressive targets ~65% POP and demands a fat credit floor to compensate.

# ── Adaptive grid defaults (2026-05-15 tuning) ─────────────────────────
# Updated per the "include SPY without raising budget" discussion. The
# narrower 0.5% width and the 0.40 delta entry expand the candidate
# space for high-spot tickers (SPY/QQQ/DIA) without lowering the C/W
# floor — those tickers only fill on IV-rich days, which is the
# correct self-regulating behavior.
_CONS_DELTA_GRID  = (0.15, 0.20, 0.25, 0.30)
_BAL_DELTA_GRID   = (0.20, 0.25, 0.30, 0.35, 0.40)
_AGG_DELTA_GRID   = (0.25, 0.30, 0.35, 0.40, 0.45)

_CONS_DTE_GRID    = (14, 21, 30, 45)
_BAL_DTE_GRID     = (7, 14, 21, 30, 45)
_AGG_DTE_GRID     = (7, 14, 21, 30)

_CONS_WIDTH_GRID  = (0.010, 0.015, 0.020, 0.025)
_BAL_WIDTH_GRID   = (0.005, 0.010, 0.015, 0.020, 0.025)
_AGG_WIDTH_GRID   = (0.005, 0.010, 0.015, 0.020, 0.025, 0.030)


CONSERVATIVE = PresetConfig(
    name="conservative",
    max_delta=0.15,
    dte_vertical=35,
    dte_iron_condor=45,
    dte_mean_reversion=21,
    dte_window_days=7,
    width_mode="pct_of_spot",
    width_value=0.025,           # 2.5% × spot
    min_credit_ratio=0.20,
    max_risk_pct=0.01,           # 1% account
    # Conservative keeps tighter delta (no 0.35+ exposure) and skips
    # the very-narrow 0.5% width — it shouldn't be trading SPY-style
    # spreads in the first place.
    delta_grid=_CONS_DELTA_GRID,
    dte_grid=_CONS_DTE_GRID,
    width_grid_pct=_CONS_WIDTH_GRID,
    edge_buffer=0.15,                    # demand more margin
    min_pop=0.65,                        # require higher POP
    # Conservative fires less often — ride winners further to collect more
    # $/trade. 60% target means longer hold but bigger banked profit.
    profit_target_pct=0.60,
    description=(
        "Low-risk: far-OTM credit spreads (~85% POP), longer DTE, 1% risk "
        "per trade. Fewer trades; smaller premiums; high win rate."
    ),
)

BALANCED = PresetConfig(
    name="balanced",
    max_delta=0.25,
    dte_vertical=21,
    dte_iron_condor=35,
    dte_mean_reversion=14,
    dte_window_days=7,
    width_mode="pct_of_spot",
    width_value=0.015,           # 1.5% × spot
    min_credit_ratio=0.30,
    max_risk_pct=0.02,           # 2% account
    # Recommended 2026-05-15 grid: delta-0.40 added so the scanner can
    # find premium-paying candidates on high-spot tickers during IV-rich
    # days; 45-DTE added for more total theta; 0.5% width opens SPY/QQQ
    # budget-eligibility (still C/W-gated). edge_buffer slightly relaxed.
    delta_grid=_BAL_DELTA_GRID,
    dte_grid=_BAL_DTE_GRID,
    width_grid_pct=_BAL_WIDTH_GRID,
    edge_buffer=0.08,
    min_pop=0.55,
    # Industry-standard 50% early-exit — captures most of the theta with
    # less exposure to late-cycle gamma. Recycles capital ~2× per trade vs.
    # holding to expiration on a 21-DTE position.
    profit_target_pct=0.50,
    # Defensive roll ON with standard 0.5–1.5% trigger band — Balanced
    # is the canonical "real positions, real defense" preset.
    defensive_roll_enabled=True,
    description=(
        "Recommended baseline: credit spreads at ~75% POP when premium pays, "
        "debit spreads / calendars when it doesn't; 21-DTE verticals, 2% risk "
        "per trade."
    ),
)

AGGRESSIVE = PresetConfig(
    name="aggressive",
    max_delta=0.35,
    dte_vertical=10,
    dte_iron_condor=21,
    dte_mean_reversion=7,
    dte_window_days=4,
    width_mode="fixed_dollar",
    width_value=5.0,             # $5 fixed
    min_credit_ratio=0.40,
    max_risk_pct=0.03,           # 3% account
    # Aggressive extends the delta grid up to 0.45 (admits near-ATM
    # shorts when min_pop allows), keeps short DTE for theta velocity,
    # widens to 3.0% for tickers with explosive premium.
    delta_grid=_AGG_DELTA_GRID,
    dte_grid=_AGG_DTE_GRID,
    width_grid_pct=_AGG_WIDTH_GRID,
    edge_buffer=0.05,
    min_pop=0.50,                        # admits up to delta-0.50
    # Aggressive cycles fire often; recycle capital faster by taking profit
    # at 40% to enable more shots per month. Trades off bigger per-trade
    # profit for higher annualised capital turnover.
    profit_target_pct=0.40,
    # Aggressive opens with near-ATM shorts — short strikes are routinely
    # near spot, so defensive rolls fire more often. Tighten the upper
    # band to 1.2% so the agent rolls EARLIER (more reaction time)
    # before gamma blows up.
    defensive_roll_enabled=True,
    roll_trigger_max_pct=0.012,
    description=(
        "High-variance: near-ATM credit spreads (~65% POP), short DTE, 3% "
        "risk per trade. Most trades; largest premiums; gamma-sensitive."
    ),
)

PRESETS: Dict[str, PresetConfig] = {
    "conservative": CONSERVATIVE,
    "balanced":     BALANCED,
    "aggressive":   AGGRESSIVE,
}

DEFAULT_PROFILE: ProfileName = "balanced"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_TUPLE_FIELDS = {"dte_grid", "delta_grid", "width_grid_pct", "put_debit_excluded_tickers",
                 "iron_butterfly_dte_grid", "iron_butterfly_wing_width_pct",
                 "cc_dte_band", "csp_dte_band", "csp_strike_band"}


def _coerce_overrides(overrides: Dict) -> Dict:
    """JSON gives us lists; the dataclass wants tuples (frozen=True)."""
    out = dict(overrides)
    if out.get("entry_rate_scope") not in (None, "sector", "global"):
        logger.warning("Invalid entry_rate_scope %r — using the default",
                       out.pop("entry_rate_scope"))
    if out.get("entry_timing_mode") not in (None, "off", "shadow", "live"):
        logger.warning("Invalid entry_timing_mode %r — using the default",
                       out.pop("entry_timing_mode"))
    if out.get("debit_profit_target_basis") not in (None, "debit", "max_profit"):
        logger.warning("Invalid debit_profit_target_basis %r — using the default",
                       out.pop("debit_profit_target_basis"))
    if out.get("profit_trail_mode") not in (None, "off", "shadow", "live"):
        logger.warning("Invalid profit_trail_mode %r — using the default",
                       out.pop("profit_trail_mode"))
    if out.get("net_delta_mode") not in (None, "off", "shadow"):
        # "live" is not built yet (backlog E4) — fall back to shadow.
        logger.warning("Invalid net_delta_mode %r — using the default",
                       out.pop("net_delta_mode"))
    if out.get("debit_ladder_mode") not in (None, "off", "shadow"):
        # "live" is not built yet (backlog C1) — fall back to shadow.
        logger.warning("Invalid debit_ladder_mode %r — using the default",
                       out.pop("debit_ladder_mode"))
    for mode in ("rs_filter_mode", "pullback_filter_mode"):
        if out.get(mode) not in (None, "off", "shadow", "live"):
            logger.warning("Invalid %s %r — using the default", mode, out.pop(mode))
    for k in _TUPLE_FIELDS:
        if k in out and isinstance(out[k], list):
            out[k] = tuple(out[k])
    return out


def _make_custom(overrides: Dict) -> PresetConfig:
    """
    Build a Custom preset by starting from BALANCED and applying overrides.

    Unknown keys are ignored. This keeps the dashboard forward-compatible
    if older preset files lack a newer field.
    """
    base = BALANCED
    overrides = _coerce_overrides(overrides)
    safe_overrides = {
        k: v for k, v in overrides.items()
        if k in {f.name for f in base.__dataclass_fields__.values()}
    }
    safe_overrides["name"] = "custom"
    return replace(base, **safe_overrides)


def load_active_preset(path: Optional[Path] = None) -> PresetConfig:
    """
    Read the active preset from ``STRATEGY_PRESET.json``. Falls back to
    BALANCED if the file is missing, malformed, or names an unknown
    profile. Always returns a usable PresetConfig — never raises.

    Expected JSON shape::

        {
          "profile": "balanced" | "conservative" | "aggressive" | "custom",
          "directional_bias": "auto" | "bullish_only" | "bearish_only" | "neutral_only",
          "custom": { ...PresetConfig fields when profile == "custom"... }
        }
    """
    fp = path or PRESET_FILE
    if not fp.exists():
        logger.info("No %s — using default profile %s", fp, DEFAULT_PROFILE)
        return PRESETS[DEFAULT_PROFILE]

    try:
        data = json.loads(fp.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s (%s) — falling back to %s",
                       fp, exc, DEFAULT_PROFILE)
        return PRESETS[DEFAULT_PROFILE]

    profile = (data.get("profile") or DEFAULT_PROFILE).lower()
    bias = data.get("directional_bias", "auto")

    if profile == "custom":
        preset = _make_custom(data.get("custom", {}))
    elif profile in PRESETS:
        preset = PRESETS[profile]
    else:
        logger.warning("Unknown profile %r — falling back to %s",
                       profile, DEFAULT_PROFILE)
        preset = PRESETS[DEFAULT_PROFILE]

    if bias not in ("auto", "bullish_only", "bearish_only", "neutral_only"):
        logger.warning("Unknown directional_bias %r — coercing to 'auto'", bias)
        bias = "auto"

    # Scan-mode + edge_buffer round-trip as overlays so a user can pick
    # "Balanced + Adaptive" or "Aggressive + Static" without falling into
    # the Custom profile.  Missing keys preserve whatever the chosen
    # profile already specified — same pattern used for directional_bias.
    overlay: Dict = {"directional_bias": bias}
    scan_mode = data.get("scan_mode")
    if scan_mode in ("static", "adaptive"):
        overlay["scan_mode"] = scan_mode
    elif scan_mode is not None:
        logger.warning("Unknown scan_mode %r — keeping profile default %r",
                       scan_mode, preset.scan_mode)
    edge_buffer = data.get("edge_buffer")
    if isinstance(edge_buffer, (int, float)) and 0.0 <= edge_buffer <= 1.0:
        overlay["edge_buffer"] = float(edge_buffer)
    elif edge_buffer is not None:
        logger.warning("Invalid edge_buffer %r — keeping profile default %r",
                       edge_buffer, preset.edge_buffer)

    # Profit-target overlay — applied on top of whichever profile is chosen,
    # same way `directional_bias` / `scan_mode` / `edge_buffer` overlay. Lets
    # an operator A/B test 30/40/50/60/70% in the backtester against the
    # built-in 50% (Balanced) without falling into the Custom profile.
    profit_target_pct = data.get("profit_target_pct")
    if isinstance(profit_target_pct, (int, float)) and 0.10 <= profit_target_pct <= 0.95:
        overlay["profit_target_pct"] = float(profit_target_pct)
    elif profit_target_pct is not None:
        logger.warning(
            "Invalid profit_target_pct %r — keeping profile default %r",
            profit_target_pct, preset.profit_target_pct,
        )

    # Skill 56 overlays — LLM-tunable fields the daily reviewer may
    # propose. Same validation shape as profit_target_pct: range-checked
    # here; invalid values log a warning and preserve the profile default.
    min_pop = data.get("min_pop")
    if isinstance(min_pop, (int, float)) and 0.30 <= min_pop <= 0.90:
        overlay["min_pop"] = float(min_pop)
    elif min_pop is not None:
        logger.warning("Invalid min_pop %r — keeping profile default %r",
                       min_pop, preset.min_pop)

    max_leg_spread_cents = data.get("max_leg_spread_cents")
    if (isinstance(max_leg_spread_cents, (int, float))
            and 0.01 <= max_leg_spread_cents <= 1.00):
        overlay["max_leg_spread_cents"] = float(max_leg_spread_cents)
    elif max_leg_spread_cents is not None:
        logger.warning(
            "Invalid max_leg_spread_cents %r — keeping profile default %r",
            max_leg_spread_cents, preset.max_leg_spread_cents)

    fill_model = data.get("fill_model")
    if fill_model in FILL_MODELS:
        overlay["fill_model"] = fill_model
    elif fill_model is not None:
        logger.warning("Invalid fill_model %r — keeping profile default %r",
                       fill_model, preset.fill_model)

    market_state_enabled = data.get("market_state_enabled")
    if isinstance(market_state_enabled, bool):
        overlay["market_state_enabled"] = market_state_enabled
    elif market_state_enabled is not None:
        logger.warning("Invalid market_state_enabled %r — keeping profile default %r",
                       market_state_enabled, preset.market_state_enabled)

    defensive_roll_enabled = data.get("defensive_roll_enabled")
    if isinstance(defensive_roll_enabled, bool):
        overlay["defensive_roll_enabled"] = defensive_roll_enabled
    elif defensive_roll_enabled is not None:
        logger.warning(
            "Invalid defensive_roll_enabled %r — keeping profile default %r",
            defensive_roll_enabled, preset.defensive_roll_enabled)

    return replace(preset, **overlay)


def save_active_preset(profile: ProfileName,
                       directional_bias: DirectionalBias = "auto",
                       custom: Optional[Dict] = None,
                       *,
                       scan_mode: Optional[ScanMode] = None,
                       edge_buffer: Optional[float] = None,
                       profit_target_pct: Optional[float] = None,
                       min_pop: Optional[float] = None,
                       max_leg_spread_cents: Optional[float] = None,
                       defensive_roll_enabled: Optional[bool] = None,
                       fill_model: Optional[str] = None,
                       market_state_enabled: Optional[bool] = None,
                       path: Optional[Path] = None) -> Path:
    """
    Persist the active preset selection to ``STRATEGY_PRESET.json``.

    Called from the Streamlit dashboard's Apply button. The file is
    written atomically (temp + rename) so a half-written JSON can never
    be observed by a concurrently-launching agent subprocess.

    ``scan_mode``, ``edge_buffer``, and ``profit_target_pct`` are persisted
    as top-level overlays — they apply on top of whichever profile is
    chosen (mirrors the ``directional_bias`` model). Pass ``None`` to
    omit; the loader will then use the chosen profile's built-in default.
    """
    fp = path or PRESET_FILE
    payload: Dict = {
        "profile": profile,
        "directional_bias": directional_bias,
    }
    if scan_mode is not None:
        payload["scan_mode"] = scan_mode
    if edge_buffer is not None:
        payload["edge_buffer"] = float(edge_buffer)
    if profit_target_pct is not None:
        payload["profit_target_pct"] = float(profit_target_pct)
    # Skill 56 additions — LLM-tunable overlays. Same overlay pattern
    # as edge_buffer / profit_target_pct: applied on top of whichever
    # profile is chosen; the loader falls back to profile defaults
    # when the field is absent.
    if min_pop is not None:
        payload["min_pop"] = float(min_pop)
    if max_leg_spread_cents is not None:
        payload["max_leg_spread_cents"] = float(max_leg_spread_cents)
    if defensive_roll_enabled is not None:
        payload["defensive_roll_enabled"] = bool(defensive_roll_enabled)
    if fill_model is not None:
        if fill_model not in FILL_MODELS:
            raise ValueError(f"fill_model must be one of {FILL_MODELS}")
        payload["fill_model"] = fill_model
    if market_state_enabled is not None:
        payload["market_state_enabled"] = bool(market_state_enabled)
    if profile == "custom" and custom:
        # Only persist the dataclass-known keys.
        valid = {f.name for f in PresetConfig.__dataclass_fields__.values()}
        payload["custom"] = {k: v for k, v in custom.items() if k in valid}

    tmp = fp.with_suffix(fp.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(fp)
    logger.info(
        "Saved %s → profile=%s bias=%s scan_mode=%s edge_buffer=%s profit_target=%s",
        fp, profile, directional_bias, scan_mode, edge_buffer, profit_target_pct,
    )
    return fp


# ---------------------------------------------------------------------------
# Bias helpers
# ---------------------------------------------------------------------------

# Map regime → which biases consider that regime tradeable.
# (Mean-reversion always trades regardless of bias because the 3-σ
# touch override is a fear-spike signal, not a directional view.)
_BIAS_REGIME_ALLOWED: Dict[str, set] = {
    "auto":         {"bullish", "bearish", "sideways", "mean_reversion"},
    "bullish_only": {"bullish", "sideways", "mean_reversion"},
    "bearish_only": {"bearish", "sideways", "mean_reversion"},
    "neutral_only": {"sideways", "mean_reversion"},
}


def regime_is_allowed(regime: str, bias: DirectionalBias) -> bool:
    """
    True if the agent should plan a trade for *regime* under the given
    directional bias. The agent calls this after Phase II classify;
    a False return short-circuits the ticker before Phase III.
    """
    return regime.lower() in _BIAS_REGIME_ALLOWED.get(bias, set())


__all__ = [
    "PresetConfig",
    "ProfileName",
    "DirectionalBias",
    "WidthMode",
    "CONSERVATIVE",
    "BALANCED",
    "AGGRESSIVE",
    "PRESETS",
    "DEFAULT_PROFILE",
    "PRESET_FILE",
    "load_active_preset",
    "save_active_preset",
    "regime_is_allowed",
]
