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
from typing import Dict, Literal, Optional, Tuple

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
    min_credit_ratio:      float            # C/W floor
    max_risk_pct:          float            # account-fraction risk cap
    directional_bias:      DirectionalBias = "auto"
    description:           str = ""

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
        """One-liner for the dashboard status line."""
        wstr = (f"{self.width_value*100:.1f}%spot"
                if self.width_mode == "pct_of_spot"
                else f"${self.width_value:.0f}")
        return (
            f"{self.name.title()} • {self.directional_bias.replace('_', ' ')} • "
            f"Vert@{self.dte_vertical}d Δ-{self.max_delta:.2f} w={wstr} • "
            f"IC@{self.dte_iron_condor}d • MR@{self.dte_mean_reversion}d • "
            f"C/W ≥ {self.min_credit_ratio} • Max risk {self.max_risk_pct*100:.0f}%"
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

def _make_custom(overrides: Dict) -> PresetConfig:
    """
    Build a Custom preset by starting from BALANCED and applying overrides.

    Unknown keys are ignored. This keeps the dashboard forward-compatible
    if older preset files lack a newer field.
    """
    base = BALANCED
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

    return replace(preset, directional_bias=bias)


def save_active_preset(profile: ProfileName,
                       directional_bias: DirectionalBias = "auto",
                       custom: Optional[Dict] = None,
                       path: Optional[Path] = None) -> Path:
    """
    Persist the active preset selection to ``STRATEGY_PRESET.json``.

    Called from the Streamlit dashboard's Apply button. The file is
    written atomically (temp + rename) so a half-written JSON can never
    be observed by a concurrently-launching agent subprocess.
    """
    fp = path or PRESET_FILE
    payload = {
        "profile": profile,
        "directional_bias": directional_bias,
    }
    if profile == "custom" and custom:
        # Only persist the dataclass-known keys.
        valid = {f.name for f in PresetConfig.__dataclass_fields__.values()}
        payload["custom"] = {k: v for k, v in custom.items() if k in valid}

    tmp = fp.with_suffix(fp.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(fp)
    logger.info("Saved %s → profile=%s bias=%s", fp, profile, directional_bias)
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
