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
    # Convenience
    # ------------------------------------------------------------------

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

    def to_summary_line(self) -> str:
        """Detailed one-liner for the agent log + dashboard expander body."""
        wstr = (f"{self.width_value*100:.1f}%spot"
                if self.width_mode == "pct_of_spot"
                else f"${self.width_value:.0f}")
        if self.scan_mode == "adaptive":
            return (
                f"{self.name.title()} • {self.directional_bias.replace('_', ' ')} • "
                f"ADAPTIVE scan • DTE∈{list(self.dte_grid)} Δ∈{list(self.delta_grid)} "
                f"w∈{[f'{w*100:.1f}%' for w in self.width_grid_pct]} • "
                f"Edge ≥ {self.edge_buffer:.0%} • POP ≥ {self.min_pop:.0%} • "
                f"Max risk {self.max_risk_pct*100:.0f}%"
            )
        return (
            f"{self.name.title()} • {self.directional_bias.replace('_', ' ')} • "
            f"Vert@{self.dte_vertical}d Δ-{self.max_delta:.2f} w={wstr} • "
            f"IC@{self.dte_iron_condor}d • MR@{self.dte_mean_reversion}d • "
            f"C/W ≥ {self.min_credit_ratio} • Max risk {self.max_risk_pct*100:.0f}%"
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
    description=(
        "Low-risk: ~85% POP, far-OTM shorts, longer DTE. Trades fire less "
        "often; credits are smaller; win rate is high."
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
    description=(
        "Recommended baseline: ~75% POP, 21-DTE verticals, 1.5% width. "
        "Trades fire most days; healthy credits; reasonable win rate."
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
    description=(
        "High-credit / high-variance: ~65% POP, near-ATM shorts, short DTE. "
        "Fires almost every cycle; large credits; gamma-sensitive."
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

_TUPLE_FIELDS = {"dte_grid", "delta_grid", "width_grid_pct"}


def _coerce_overrides(overrides: Dict) -> Dict:
    """JSON gives us lists; the dataclass wants tuples (frozen=True)."""
    out = dict(overrides)
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

    return replace(preset, **overlay)


def save_active_preset(profile: ProfileName,
                       directional_bias: DirectionalBias = "auto",
                       custom: Optional[Dict] = None,
                       *,
                       scan_mode: Optional[ScanMode] = None,
                       edge_buffer: Optional[float] = None,
                       path: Optional[Path] = None) -> Path:
    """
    Persist the active preset selection to ``STRATEGY_PRESET.json``.

    Called from the Streamlit dashboard's Apply button. The file is
    written atomically (temp + rename) so a half-written JSON can never
    be observed by a concurrently-launching agent subprocess.

    ``scan_mode`` and ``edge_buffer`` are persisted as top-level overlays —
    they apply on top of whichever profile is chosen (mirrors the
    ``directional_bias`` model). Pass ``None`` to omit them entirely;
    the loader will then use the chosen profile's built-in default.
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
    if profile == "custom" and custom:
        # Only persist the dataclass-known keys.
        valid = {f.name for f in PresetConfig.__dataclass_fields__.values()}
        payload["custom"] = {k: v for k, v in custom.items() if k in valid}

    tmp = fp.with_suffix(fp.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(fp)
    logger.info("Saved %s → profile=%s bias=%s scan_mode=%s edge_buffer=%s",
                fp, profile, directional_bias, scan_mode, edge_buffer)
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
