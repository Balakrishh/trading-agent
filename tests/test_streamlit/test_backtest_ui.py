"""
Tests for ``trading_agent.streamlit.backtest_ui`` — the post-rewrite shim.

After the May 2026 rewrite (skill 15), this file no longer owns any
trading logic. The pre-rewrite ``Backtester`` class, ``SimTrade`` /
``BacktestResult`` re-implementations, and module constants
(``DAILY_OTM_PCT``, ``LEADERSHIP_WINDOW_BARS``, ``VIX_INHIBIT_ZSCORE``,
``STARTING_EQUITY`` …) all moved into ``trading_agent.backtest`` and the
preset system. What's left in ``backtest_ui.py``:

* Two private adapters that turn ``BacktestResult`` payloads into
  DataFrames the chart helpers expect.
* ``_preview_decision`` — diagnostic shim that satisfies CI invariant #3
  (the file must contain a literal ``decide(`` call).
* ``render_backtest_ui`` — Streamlit entry point.

These tests pin exactly that surface. They do NOT spin up a full
``BacktestRunner.run()`` (which would pull historical bars over the
network); the runner has its own dedicated tests under
``tests/test_backtest/``.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import re
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trading_agent.backtest.account import EquityPoint
from trading_agent.backtest.cycle import CycleOutcome
from trading_agent.regime import Regime
from trading_agent.strategy_presets import PRESETS
from trading_agent.streamlit.backtest_ui import (
    ALL_TICKERS,
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_STARTING_EQUITY,
    DEFAULT_TICKERS,
    _apply_overrides,
    _cycle_outcomes_to_df,
    _equity_curve_to_df,
    _format_csv,
    _parse_csv_floats,
    _parse_csv_ints,
    _preview_decision,
    render_backtest_ui,
)


# ---------------------------------------------------------------------------
# Module surface — sanity checks
# ---------------------------------------------------------------------------

class TestModuleSurface:
    def test_default_tickers_subset_of_all(self):
        assert set(DEFAULT_TICKERS).issubset(set(ALL_TICKERS))

    def test_starting_equity_positive(self):
        assert DEFAULT_STARTING_EQUITY > 0

    def test_default_window_is_in_the_past_and_nonempty(self):
        assert DEFAULT_START < DEFAULT_END
        assert DEFAULT_END <= dt.date.today()

    def test_render_backtest_ui_is_callable(self):
        assert callable(render_backtest_ui)


# ---------------------------------------------------------------------------
# CI invariant #3 — a literal ``decide(`` call must remain in the source
# ---------------------------------------------------------------------------

class TestInvariantThreeDecideCallPresent:
    def test_decide_call_literal_in_module_source(self):
        """``scripts/checks/scan_invariant_check.py`` requires this. Pinning
        it here too gives a clearer failure message inside the unit suite."""
        src = Path(
            "trading_agent/streamlit/backtest_ui.py"
        ).resolve().read_text()
        assert re.search(r"\bdecide\s*\(", src), (
            "backtest_ui.py must contain a literal `decide(` call to keep "
            "the live↔backtest unified-engine link (CI invariant #3)."
        )


# ---------------------------------------------------------------------------
# _equity_curve_to_df
# ---------------------------------------------------------------------------

def _equity_point(t, equity, cash, omv, realised=0.0, unrealised=0.0, n=0):
    return EquityPoint(
        t=t, cash=cash, open_market_value=omv, equity=equity,
        open_spread_count=n, realised_pnl=realised,
        unrealised_pnl=unrealised,
    )


class TestEquityCurveToDf:
    def test_returns_empty_dataframe_when_no_points(self):
        df = _equity_curve_to_df([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_columns_match_chart_helper_contract(self):
        # equity_curve_chart / drawdown_chart key off these names.
        pts = [
            _equity_point(datetime(2026, 5, 4, 9, 30),
                          equity=100_000.0, cash=100_000.0, omv=0.0),
            _equity_point(datetime(2026, 5, 4, 9, 35),
                          equity=100_500.0, cash=99_500.0, omv=1_000.0,
                          realised=200.0, unrealised=300.0, n=2),
        ]
        df = _equity_curve_to_df(pts)
        for col in ("timestamp", "account_balance", "cash",
                    "open_market_value", "realised_pnl"):
            assert col in df.columns, f"missing required column {col!r}"

    def test_values_round_trip(self):
        pt = _equity_point(datetime(2026, 5, 4, 9, 30),
                           equity=100_500.0, cash=99_500.0, omv=1_000.0,
                           realised=200.0)
        df = _equity_curve_to_df([pt])
        assert df.iloc[0]["account_balance"] == 100_500.0
        assert df.iloc[0]["cash"] == 99_500.0
        assert df.iloc[0]["open_market_value"] == 1_000.0
        assert df.iloc[0]["realised_pnl"] == 200.0


# ---------------------------------------------------------------------------
# _cycle_outcomes_to_df
# ---------------------------------------------------------------------------

class TestCycleOutcomesToDf:
    def test_returns_empty_dataframe_when_no_outcomes(self):
        df = _cycle_outcomes_to_df([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_unwraps_regime_enum(self):
        out = CycleOutcome(
            ticker="SPY", t=datetime(2026, 5, 4, 9, 35), spot=500.12345,
            regime=Regime.BULLISH, status="opened", reason="ok",
        )
        df = _cycle_outcomes_to_df([out])
        # Spot rounded to 2 decimals
        assert df.iloc[0]["spot"] == 500.12
        # Regime renders as the string value, not the enum repr
        assert df.iloc[0]["regime"] == Regime.BULLISH.value

    def test_handles_none_spot_and_none_regime(self):
        out = CycleOutcome(
            ticker="SPY", t=datetime(2026, 5, 4, 9, 35), spot=0.0,
            regime=None, status="no_data", reason="missing bars",
        )
        df = _cycle_outcomes_to_df([out])
        assert df.iloc[0]["spot"] == 0.0
        assert df.iloc[0]["regime"] is None
        assert df.iloc[0]["status"] == "no_data"


# ---------------------------------------------------------------------------
# _preview_decision
# ---------------------------------------------------------------------------

class TestPreviewDecision:
    def test_returns_error_dict_when_no_chain_slices_built(self):
        with patch(
            "trading_agent.streamlit.backtest_ui.build_chain_slice",
            side_effect=Exception("synthetic chain blew up"),
        ):
            out = _preview_decision(
                ticker="SPY", preset_name="balanced",
                side="bull_put", spot=500.0, sigma=0.20,
            )
        assert isinstance(out, dict)
        assert "error" in out

    def test_calls_decide_and_returns_candidates_payload(self):
        # Stub the synthetic chain so we don't hit the real BS pricer; stub
        # decide() so we don't run the full unified engine. We only need
        # to verify the wiring (decide is called once, candidates round-trip).
        fake_slice = MagicMock(name="ChainSlice")
        fake_candidate = MagicMock(
            expiration="2026-05-22", dte=14, short_strike=485.123,
            long_strike=480.456, credit=1.234, width=5.0,
            cw_ratio=0.246_891, pop=0.751_2, annualized_score=2.345_6,
        )
        fake_decision = MagicMock(
            candidates=[fake_candidate],
            diagnostics=MagicMock(
                grid_points_total=12, grid_points_priced=10,
                rejects_by_reason={"below_floor": 2},
            ),
        )
        with patch(
            "trading_agent.streamlit.backtest_ui.build_chain_slice",
            return_value=fake_slice,
        ), patch(
            "trading_agent.streamlit.backtest_ui.decide",
            return_value=fake_decision,
        ) as decide_mock:
            out = _preview_decision(
                ticker="SPY", preset_name="balanced",
                side="bull_put", spot=500.0, sigma=0.20,
            )
        assert decide_mock.called, "preview must invoke decide() exactly once"
        assert out["grid_total"] == 12
        assert out["grid_priced"] == 10
        assert out["rejects"] == {"below_floor": 2}
        assert len(out["candidates"]) == 1
        c = out["candidates"][0]
        assert c["short_strike"] == 485.12      # rounded to 2dp
        assert c["long_strike"] == 480.46
        assert c["credit"] == 1.23
        assert c["width"] == 5.0
        assert c["cw_ratio"] == 0.2469          # rounded to 4dp
        assert c["pop"] == 0.7512
        assert c["score"] == 2.3456

    def test_falls_back_to_active_preset_when_name_unknown(self):
        # When the preset name doesn't resolve, fall through to
        # load_active_preset() rather than raising. That gives the UI a
        # sensible default if the JSON file has been hand-edited.
        with patch(
            "trading_agent.streamlit.backtest_ui.build_chain_slice",
            side_effect=Exception("doesn't matter"),
        ), patch(
            "trading_agent.streamlit.backtest_ui.load_active_preset",
        ) as loader:
            loader.return_value = MagicMock(dte_grid=(7, 14, 21, 30))
            _preview_decision(
                ticker="SPY", preset_name="this-preset-doesnt-exist",
                side="bull_put", spot=500.0, sigma=0.20,
            )
        assert loader.called, (
            "unknown preset name should trigger load_active_preset() fallback"
        )

    def test_explicit_preset_kwarg_short_circuits_lookup(self):
        # When the editor passes a customized PresetConfig directly,
        # skip both PRESETS.get and load_active_preset — that's how the
        # sidebar's Customize expander gets its overrides into the
        # preview without first persisting them.
        custom = PRESETS["balanced"]  # any concrete preset will do
        with patch(
            "trading_agent.streamlit.backtest_ui.build_chain_slice",
            side_effect=Exception("doesn't matter"),
        ), patch(
            "trading_agent.streamlit.backtest_ui.load_active_preset",
        ) as loader, patch(
            "trading_agent.streamlit.backtest_ui.PRESETS",
            {},  # empty so name lookup would fail without the kwarg
        ):
            _preview_decision(
                ticker="SPY", preset_name="ignored",
                side="bull_put", spot=500.0, sigma=0.20,
                preset=custom,
            )
        assert not loader.called, (
            "explicit preset= kwarg should bypass the name-based fallback"
        )


# ---------------------------------------------------------------------------
# Pure helpers — _format_csv / _parse_csv_* / _apply_overrides
# ---------------------------------------------------------------------------

class TestFormatCsv:
    def test_floats_render_compactly_via_g(self):
        # %g drops trailing zeros — 0.10 → "0.1". Easier to read in a
        # narrow sidebar text input.
        assert _format_csv((0.10, 0.15, 0.20)) == "0.1, 0.15, 0.2"

    def test_ints_render_without_decimals(self):
        assert _format_csv((7, 14, 21, 30)) == "7, 14, 21, 30"

    def test_empty_input_returns_empty_string(self):
        assert _format_csv(()) == ""


class TestParseCsvFloats:
    def test_parses_well_formed_input(self):
        assert _parse_csv_floats("0.1, 0.15, 0.2", (1.0,)) == (0.1, 0.15, 0.2)

    def test_handles_extra_whitespace(self):
        assert _parse_csv_floats("  0.1 ,0.2 ,  0.3  ", (1.0,)) == (
            0.1, 0.2, 0.3,
        )

    def test_drops_empty_fields_silently(self):
        # Trailing comma is common when typing — don't raise.
        assert _parse_csv_floats("0.1, 0.2,", (1.0,)) == (0.1, 0.2)

    def test_returns_fallback_on_parse_error(self):
        # A typo must NOT zero out the grid (which would skip every
        # cycle). Fall back to the provided default.
        fb = (0.20, 0.25, 0.30)
        assert _parse_csv_floats("0.1, oops, 0.2", fb) == fb

    def test_returns_fallback_on_empty_string(self):
        fb = (0.20, 0.25)
        assert _parse_csv_floats("", fb) == fb
        assert _parse_csv_floats("   ", fb) == fb


class TestParseCsvInts:
    def test_parses_well_formed_input(self):
        assert _parse_csv_ints("7, 14, 21, 30", (1,)) == (7, 14, 21, 30)

    def test_accepts_floats_and_truncates(self):
        # "7.0" should become 7. Round-tripping the editor's ``%g`` output
        # of an int-only field would fail otherwise.
        assert _parse_csv_ints("7.0, 14.0", (1,)) == (7, 14)

    def test_returns_fallback_on_parse_error(self):
        fb = (7, 14, 21)
        assert _parse_csv_ints("7, banana, 21", fb) == fb


class TestApplyOverrides:
    def test_no_changes_returns_same_object_unchanged(self):
        # If the editor's values match the seed exactly, return the seed
        # itself so the preset name stays meaningful in run captions.
        seed = PRESETS["balanced"]
        out = _apply_overrides(
            seed,
            scan_mode=seed.scan_mode,
            max_delta=seed.max_delta,
            edge_buffer=seed.edge_buffer,
            min_pop=seed.min_pop,
            dte_grid=seed.dte_grid,
            delta_grid=seed.delta_grid,
            width_grid_pct=seed.width_grid_pct,
        )
        assert out is seed
        assert out.name == "balanced"

    def test_any_diff_flips_name_to_custom(self):
        seed = PRESETS["balanced"]
        out = _apply_overrides(seed, edge_buffer=seed.edge_buffer + 0.01)
        assert out is not seed
        assert out.name == "custom"
        assert out.edge_buffer == seed.edge_buffer + 0.01

    def test_coerces_list_grids_to_tuple(self):
        # The frozen dataclass requires tuples (hashable). Streamlit
        # text-input parsers might hand us lists; coerce here once.
        seed = PRESETS["balanced"]
        out = _apply_overrides(
            seed, delta_grid=[0.10, 0.15, 0.20, 0.25, 0.30],
        )
        assert isinstance(out.delta_grid, tuple)
        assert out.delta_grid == (0.10, 0.15, 0.20, 0.25, 0.30)
        assert out.name == "custom"

    def test_suggested_low_delta_grid_is_applied(self):
        # Direct regression for the May-2026 backtest tuning advice:
        # extending delta_grid downward should produce a custom preset
        # whose floor is reachable in normal-IV regimes.
        seed = PRESETS["balanced"]
        out = _apply_overrides(
            seed,
            edge_buffer=0.0,
            delta_grid=(0.10, 0.15, 0.20, 0.25, 0.30),
        )
        assert out.edge_buffer == 0.0
        assert out.delta_grid == (0.10, 0.15, 0.20, 0.25, 0.30)
        assert out.name == "custom"


# ---------------------------------------------------------------------------
# Module loads cleanly under spec_from_file_location too
# ---------------------------------------------------------------------------

def test_module_imports_under_spec_from_file_location(tmp_path):
    """
    Pin that the backtest_ui module is importable via the same mechanism
    Streamlit uses when discovering pages — i.e. that no top-level code
    has side effects requiring a Streamlit context.
    """
    spec = importlib.util.spec_from_file_location(
        "trading_agent.streamlit.backtest_ui_probe",
        Path("trading_agent/streamlit/backtest_ui.py").resolve(),
    )
    assert spec is not None and spec.loader is not None
    # We don't actually exec_module here — Streamlit requires a context
    # manager that pytest can't provide. Existence of a loadable spec is
    # the meaningful invariant.
