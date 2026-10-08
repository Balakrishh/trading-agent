# Preset system & hot-reload

> **One-line summary:** Three frozen `PresetConfig` profiles (Conservative / Balanced / Aggressive) plus a Custom slot bundle the four knobs that meaningfully change spread economics — `max_delta`, per-strategy DTE, width policy, and the C/W floor — into atomic risk-profile choices. Persisted in `STRATEGY_PRESET.json` at the repo root, re-read by every agent subprocess at the start of each cycle, and applied on the next 5-minute tick **without restarting anything**.
> **Source of truth:** [`trading_agent/strategy_presets.py`](../../trading_agent/strategy_presets.py) (full file); consumer wiring in [`trading_agent/agent.py:74, 169-204`](../../trading_agent/agent.py).
> **Phase:** 1  •  **Group:** config
> **Depends on:** nothing — atomic config primitive.
> **Consumed by:** `TradingAgent.__init__` (agent.py:169), `StrategyPlanner.__init__` (strategy.py:140-196), `RiskManager.__init__` (agent.py:191-204), `OrderExecutor.__init__` (agent.py:209-224), Streamlit dashboard's Strategy-Profile panel.

---

## 1. Theory & Objective

Live trading parameters drift constantly. A risk-conservative week (high VIX, FOMC, post-CPI) wants narrower deltas, longer DTE, smaller account fraction. A calm week wants the opposite. Hard-coding these in `AppConfig` means a code edit + restart for every change — unsafe in live ops.

The preset system solves three problems at once:

1. **Bundle the knobs that move together.** `max_delta`, `dte_vertical`, `width_value`, `min_credit_ratio`, and `max_risk_pct` are not independent — moving one without the others creates incoherent risk profiles (e.g. Δ-0.35 with C/W 0.20 is essentially free money for the counterparty). Three named profiles enforce internally consistent combinations.
2. **Decouple "which profile" from code deploys.** `STRATEGY_PRESET.json` is a sentinel file at the repo root, same pattern as `AGENT_RUNNING` / `DRY_RUN_MODE`. The Streamlit dashboard writes it; the agent reads it; nothing else needs to know.
3. **Hot-reload without restart.** Each subprocess calls `load_active_preset()` at the **start of every cycle**, so a dashboard change becomes effective on the next 5-min tick. No supervisor restart, no in-flight order corruption, no journal gap.

The dataclass is `frozen=True` for two reasons: (a) accidental mutation in one module would silently change behavior in another, and (b) frozen dataclasses are hashable, which lets us use `replace()` to derive a Custom preset from BALANCED without subclassing.

The atomic write pattern (`temp + rename`) prevents a partially-written JSON from being observed by a concurrently-launching agent subprocess. POSIX `rename(2)` is atomic on the same filesystem; the dashboard always writes the temp adjacent to the target, never to `/tmp`.

The fallback chain is **always operational**: missing file → BALANCED; malformed JSON → BALANCED + warning; unknown profile name → BALANCED + warning; invalid `directional_bias` → coerced to `"auto"` + warning. The function `load_active_preset()` is contracted to **never raise** — fresh installs, CI, and smoke tests get a usable agent without touching the dashboard first.

## 2. Mathematical Formula

N/A — this is a config-loading primitive. The "math" is the schema definition:

```text
PresetConfig fields                       Type             Role
─────────────────────────────────────────────────────────────────────────
name                                      str              UI label
max_delta                                 float            short-leg |Δ| ceiling
dte_vertical                              int              Bull Put / Bear Call DTE target
dte_iron_condor                           int              IC DTE target
dte_mean_reversion                        int              MR DTE target
dte_window_days                           int              ± window around target
width_mode                                "pct_of_spot"    width policy selector
                                          | "fixed_dollar"
width_value                               float            0.015 = 1.5% spot, or 5.0 = $5
min_credit_ratio                          float            C/W floor (static mode)
max_risk_pct                              float            account-fraction risk cap
directional_bias                          "auto"           regime filter overlay
                                          | "bullish_only"
                                          | "bearish_only"
                                          | "neutral_only"
scan_mode                                 "static"         single-point vs grid sweep
                                          | "adaptive"
edge_buffer                               float            adaptive C/W floor margin
min_pop                                   float            adaptive POP floor
dte_grid                                  Tuple[int,...]   adaptive DTE sweep
delta_grid                                Tuple[float,...] adaptive Δ sweep
width_grid_pct                            Tuple[float,...] adaptive width sweep

dte_range_vertical → (max(1, dte_vertical - dte_window_days),
                      dte_vertical + dte_window_days)

# Built-in profile values
CONSERVATIVE: Δ≤0.15, DTE-vert=35, width=2.5%, C/W≥0.20, risk≤1%
BALANCED:     Δ≤0.25, DTE-vert=21, width=1.5%, C/W≥0.30, risk≤2%
AGGRESSIVE:   Δ≤0.35, DTE-vert=10, width=$5,   C/W≥0.40, risk≤3%
```

The cycle-time hot-reload "formula":

```text
for each cycle t in {0, 5min, 10min, ...}:
    preset_t = load_active_preset()        # re-read every cycle
    if preset_t != preset_{t-1}:
        log "Strategy preset → ..."
    use preset_t for all planning + risk + execution decisions in cycle t
```

## 3. Reference Python Implementation

```python
# trading_agent/strategy_presets.py:54-94
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

    scan_mode:             ScanMode = "static"
    edge_buffer:           float = 0.10
    min_pop:               float = 0.55
    dte_grid:              Tuple[int, ...]   = (7, 14, 21, 30)
    delta_grid:            Tuple[float, ...] = (0.20, 0.25, 0.30, 0.35)
    width_grid_pct:        Tuple[float, ...] = (0.010, 0.015, 0.020, 0.025)
```

```python
# trading_agent/strategy_presets.py:142-191
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
    description="Low-risk: ~85% POP, far-OTM shorts, longer DTE. ...",
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
    description="Recommended baseline: ~75% POP, 21-DTE verticals, ...",
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
    description="High-credit / high-variance: ~65% POP, near-ATM shorts, ...",
)
```

```python
# trading_agent/strategy_presets.py:235-295
def load_active_preset(path: Optional[Path] = None) -> PresetConfig:
    """
    Read the active preset from STRATEGY_PRESET.json. Falls back to BALANCED
    if the file is missing, malformed, or names an unknown profile. Always
    returns a usable PresetConfig — never raises.
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
    # the Custom profile.
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
```

```python
# trading_agent/strategy_presets.py:298-336
def save_active_preset(profile: ProfileName,
                       directional_bias: DirectionalBias = "auto",
                       custom: Optional[Dict] = None,
                       *,
                       scan_mode: Optional[ScanMode] = None,
                       edge_buffer: Optional[float] = None,
                       path: Optional[Path] = None) -> Path:
    """
    Persist the active preset selection to STRATEGY_PRESET.json.

    The file is written atomically (temp + rename) so a half-written JSON
    can never be observed by a concurrently-launching agent subprocess.
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
        valid = {f.name for f in PresetConfig.__dataclass_fields__.values()}
        payload["custom"] = {k: v for k, v in custom.items() if k in valid}

    tmp = fp.with_suffix(fp.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(fp)              # atomic on POSIX, same-filesystem
    logger.info("Saved %s → profile=%s bias=%s scan_mode=%s edge_buffer=%s",
                fp, profile, directional_bias, scan_mode, edge_buffer)
    return fp
```

```python
# trading_agent/agent.py:162-204 — consumer wiring
# The preset is read at the START of every cycle; the constructor just
# captures the snapshot for this cycle's planning + risk + execution.
self.preset: PresetConfig = load_active_preset()
logger.info("Strategy preset → %s", self.preset.to_summary_line())

max_delta        = self.preset.max_delta
min_credit_ratio = self.preset.min_credit_ratio
max_risk_pct     = self.preset.max_risk_pct

self.strategy_planner = StrategyPlanner(
    data_provider=self.data_provider,
    max_delta=max_delta,
    min_credit_ratio=min_credit_ratio,
    dte_vertical=self.preset.dte_vertical,
    dte_iron_condor=self.preset.dte_iron_condor,
    dte_mean_reversion=self.preset.dte_mean_reversion,
    dte_window_days=self.preset.dte_window_days,
    width_mode=self.preset.width_mode,
    width_value=self.preset.width_value,
    preset=self.preset,
)
self.risk_manager = RiskManager(
    max_risk_pct=max_risk_pct,
    min_credit_ratio=min_credit_ratio,
    max_delta=max_delta,
    ...,
    delta_aware_floor=(self.preset.scan_mode == "adaptive"),
    edge_buffer=self.preset.edge_buffer,
)
```

## 4. Edge Cases / Guardrails

- **`load_active_preset()` MUST NOT raise** — the contract is "always return a usable preset." Every error path falls through to BALANCED with a warning. Adding a new error path? Make sure it has a fallback.
- **Atomic write requires same-filesystem** — `tmp.replace(fp)` is atomic only when `tmp` and `fp` are on the same filesystem. The temp is built via `fp.with_suffix(fp.suffix + ".tmp")` — same directory — so this is guaranteed. Don't "optimize" the temp to `/tmp/` (often a different fs / tmpfs).
- **JSON gives lists; dataclass wants tuples** — `dte_grid`, `delta_grid`, `width_grid_pct` are `Tuple[...]` because `frozen=True` requires hashable fields. `_coerce_overrides()` round-trips lists → tuples; without it, custom-profile loads would crash on the dataclass instantiation.
- **`_make_custom()` ignores unknown keys** — forward-compat: an older preset file that lacks a newer field still loads; a newer file with an extra field on an older agent silently ignores it. This lets the dashboard ship new fields ahead of the agent.
- **Frozen dataclass = use `replace()`, never mutate** — `preset.max_delta = 0.4` raises `FrozenInstanceError`. To override at load time, use `replace(preset, max_delta=0.4)`. The overlay pattern in `load_active_preset()` (for `directional_bias`, `scan_mode`, `edge_buffer`) shows the canonical approach.
- **Hot-reload is per-cycle, not real-time** — a dashboard change made at 14:32:10 takes effect at the next cycle (≈75 s by default) boundary (14:35:00), not immediately. In-flight orders complete under the previous preset. This is intentional; mid-cycle preset changes would create inconsistent risk decisions.
- **Subprocess isolation** — the agent reads the preset in its own process; the dashboard reads it for display in another. They don't share memory. The file is the only sync point. This is why atomic writes matter.
- **`directional_bias` coerces invalid → `"auto"`** — silently. Logged, but not an error. Same forward-compat philosophy: a typo or future bias label still produces a working agent.
- **`edge_buffer` validated range [0.0, 1.0]** — anything outside falls back to the profile default + warning. Negative values would invert the C/W floor; values >1 would be uneconomical (demanding C/W > 100% of width).
- **Scan-mode/edge-buffer as overlays, not Custom** — the loader treats `scan_mode` and `edge_buffer` as top-level overlays (mirroring `directional_bias`) so the user can pick "Balanced + Adaptive" without the Custom profile. The Custom path is reserved for full re-specification of the dataclass.
- **Class constants in `StrategyPlanner` are LEGACY fallbacks** — `TARGET_DTE`, `DTE_RANGE`, `SPREAD_WIDTH` only fire when the planner is instantiated without a preset (older tests, scripts predating the preset system). New code should always pass `preset=…`.

- **`fill_model` (2026-10-05).** `"natural"` (default) | `"mid"`. Top-level overlay in `STRATEGY_PRESET.json` (validated against `FILL_MODELS`; invalid → profile default with a warning) or a key inside `custom`. Shown in `to_summary_line()` as `Fills @ …`, exposed in the Strategy-Profile panel, and threaded to `decide()` (credit), the Wheel scorers and `PositionMonitor(profit_target_basis=…)`.
- **`market_state_enabled` (2026-10-05).** `bool`, default `True`. Top-level overlay (non-bool → profile default with a warning) or a key inside `custom`; `save_active_preset(market_state_enabled=…)`. Shown in `to_summary_line()` as `MarketState on|off`, a checkbox in the Strategy-Profile panel, and read by `TradingAgent._update_market_state`, `wheel_screen` and `executor_promote` (skill 58). `False` = no market-wide gates, `max_risk_pct` unscaled.
- **Debit structures (2026-10-05, skill 59).** `debit_spreads_enabled`, `calendar_enabled`, `bounce_bull_put_enabled` (default `True`), `dte_debit` 30, `debit_long_delta` 0.50 (sold leg one `width_grid_pct` step out), `debit_max_overpay` 0.05 (over the market mid), `debit_min_reward_risk` 1.0, `debit_profit_target_pct` 0.50 with `debit_profit_target_basis` "debit" (+50 % on cost; "max_profit" = the pre-2026-10-08 share of width − debit), `debit_stop_loss_pct` 0.50 (of debit), `dte_calendar_near` 21, `calendar_gap_days` 28, `calendar_option_type` "call", `calendar_profit_target_pct` 0.25, `bounce_lookback_days` 5. Set through `custom`; `to_summary_line()` shows them via `_debit_tag()` (`Debit@30d Δ0.50 RR≥1 TP50% of cost/SL50% • Cal@21+28d TP25% • Bounce@5d`); the Strategy-Profile panel has a "Debit spreads, calendars, bounce bull put" block; the exit thresholds are threaded to `PositionMonitor` in `agent.py`, the rest read by `StrategyPlanner` / `decision_engine`.
- **`max_total_risk_pct` (2026-10-05, skill 37).** Default 0.10 of equity across all open defined-risk positions; set through `custom`; summary shows `Max risk N% (total M%)`; Strategy-Profile slider "Max TOTAL open risk"; read per cycle by the agent's Stage 2 loop.
- **Wheel fields (2026-10-05, skill 40 §3.3).** `cc_max_short_delta` 0.30, `cc_dte_band` (30, 60), `cc_min_iv_rank` 0.25, `csp_max_short_delta` 0.30, `csp_dte_band` (21, 60), `csp_min_iv_rank` 0.25, `csp_strike_band` (0.85, 0.97), `csp_require_above_sma200` True, `csp_max_per_sector` 1. Tuple fields are coerced from JSON lists (`_TUPLE_FIELDS`, which now also covers the iron-butterfly grids — they used to load as lists).
- **Laddering (2026-10-05, skill 37).** `max_positions_per_ticker` 2, `ladder_min_gap_days` 7; summary `Ladder 2/ticker ≥7d`; Strategy-Profile inputs; read by the agent's cap and ladder gate.
- **Entry confirmation + trailing profit (2026-10-05, skills 61 / 30).** `entry_confirm_cycles` 3, `no_entry_before_et` "09:45", `max_new_entries_per_hour` 1, `entry_rate_scope` "sector" (`sector` / `global`; per-sector allowance since 2026-10-06); `entry_timing_mode` "shadow" (`off` / `shadow` / `live`), `entry_max_wait_cycles` 8, `entry_best_tolerance_pct` 0.01, `entry_chase_limit_pct` 0.03; `profit_trail_mode` "shadow" (`off` / `shadow` / `live`; invalid → default), `trail_giveback_pct` 0.25, `trail_credit_floor_pct` 0.40, `trail_credit_ceiling_pct` 0.75, `trail_debit_ceiling_pct` 0.90, `trail_calendar_ceiling_pct` 0.40, `trail_max_hold_dte` 7. Summary `Confirm 3× from 09:45 ≤1/h • Trail shadow −25%`; Strategy-Profile block "Entry confirmation and trailing profit".
- **2026-10-06.** `ladder_requires_profit` True (add to winners only, skill 37); `calendar_max_strike_drift_pct` 0.03 (calendar closes when the underlying is ≥ 3 % from its strike, skill 59). Summary: `Ladder 2/ticker ≥7d winners`, `Cal@21+28d TP25% drift≤3%`.
- **Drawdown governor (2026-10-07, skill 62).** `halt_daily_loss_pct` 0.02, `halt_weekly_loss_pct` 0.05 (0 = off); summary `Halt −2%/d −5%/wk`; Strategy-Profile "Drawdown governor" sliders.

## 5. Cross-References

- [03 Credit/Width floor](03_credit_to_width_floor.md) — the `min_credit_ratio` knob in static mode; `edge_buffer` in adaptive mode.
- [04 Adaptive spread width](04_adaptive_spread_width.md) — `width_mode` / `width_value` consumers.
- [14 Adaptive vs static scan modes](14_adaptive_vs_static_scan_modes.md) — `scan_mode` selects the planning algorithm.
- [05 EV per dollar risked](05_ev_per_dollar_risked.md) — used by the adaptive scanner that the preset enables.

---

*Last verified against repo HEAD on 2026-10-08.*
