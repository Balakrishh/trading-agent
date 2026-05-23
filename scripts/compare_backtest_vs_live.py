"""compare_backtest_vs_live.py — apples-to-apples diff harness.

Goal: for a given date window + the currently active preset, run the
backtester and compare its results to what the live agent actually
journalled. Outputs a side-by-side table the operator can scan in
under 30 seconds.

Why this matters
----------------
Skill 15 (backtest↔live parity) says the backtester wires through the
same ``decide()`` primitive as live, so identical inputs SHOULD
produce identical outputs. In practice the parity is approximate
because:

  * The backtester uses Black-Scholes synthetic chains, not real
    Schwab/Alpaca quotes. BS assumes zero spread, no skew, no
    flash-crash dislocations. Live spreads typically transact at
    20-40% worse than BS mid.
  * The backtester re-marks daily; live re-marks every 5 min.
  * The backtester has no PDT lockout, no partial-fill loop, no
    Telegram-alert dedup, no broker rejection codes.
  * Sentiment / earnings gating is absent in backtest by design.

So if the two diverge meaningfully, the gap tells you WHERE the live
execution is bleeding alpha — slippage, stuck partials, regime
classifier drift, or the strategy itself being wrong.

Usage
-----
    python scripts/compare_backtest_vs_live.py \
        --start 2026-05-12 --end 2026-05-23 \
        --tickers SPY,DIA,XLF,GLD,XLE \
        --journal trade_journal/signals_live.jsonl \
        --starting-equity 5000

Output is plain text (no graphics deps). Read top to bottom:
  1. Summary line: live $X vs backtest $Y  (gap $Δ)
  2. Per-ticker breakdown
  3. Trade-by-trade alignment (when same ticker fired on same day)
  4. Diagnostic list of "live but not backtest" + "backtest but not live"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# When invoked from scripts/, the repo root isn't on sys.path. Add it
# so `from trading_agent...` imports resolve regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Suppress noisy library logs so the report is readable.
logging.basicConfig(level=logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Live-side reader
# ---------------------------------------------------------------------------


@dataclass
class LiveTrade:
    """One realized trade pulled from signals_live.jsonl."""
    ticker: str
    opened_at: Optional[str]
    closed_at: Optional[str]
    strategy: str
    credit: float
    max_loss: float
    realized_pl: float
    exit_signal: str
    expiration: str


def load_live_trades(
    journal_path: str, start: date, end: date,
) -> Tuple[List[LiveTrade], float, float]:
    """Walk the journal and pair `submitted` opens to `closed` closes
    by ticker + expiration. Returns:
      * list of completed LiveTrades
      * starting account balance seen on or after `start`
      * ending account balance seen on or before `end`

    Skips `dry_run_close` rows (post commit 9cc5636) and the
    pre-fix phantom `closed`+fill_status="dry_run" pollution.
    """
    opens: Dict[Tuple[str, str], LiveTrade] = {}
    closed: List[LiveTrade] = []
    first_balance: Optional[float] = None
    last_balance: Optional[float] = None

    for line in open(journal_path, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts_str = rec.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts_date = ts.astimezone(timezone.utc).date()
        if not (start <= ts_date <= end):
            continue
        rs = rec.get("raw_signal") or {}
        if not isinstance(rs, dict):
            rs = {}
        # Track balance endpoints.
        bal = rs.get("account_balance")
        if isinstance(bal, (int, float)) and bal > 0:
            if first_balance is None:
                first_balance = float(bal)
            last_balance = float(bal)

        action = rec.get("action")
        ticker = rec.get("ticker", "")
        if action == "submitted":
            key = (ticker, str(rs.get("expiration", "")))
            opens[key] = LiveTrade(
                ticker=ticker,
                opened_at=ts.isoformat(),
                closed_at=None,
                strategy=rs.get("strategy", "?"),
                credit=float(rs.get("original_credit") or
                             rs.get("net_credit", 0) or 0),
                max_loss=float(rs.get("max_loss", 0) or 0),
                realized_pl=0.0,
                exit_signal="",
                expiration=str(rs.get("expiration", "")),
            )
        elif action == "closed":
            # Defense-in-depth: skip the pre-fix phantom rows.
            if rs.get("fill_status") == "dry_run":
                continue
            key = (ticker, str(rs.get("expiration", "")))
            tr = opens.pop(key, None) or LiveTrade(
                ticker=ticker, opened_at=None, closed_at=None,
                strategy=rs.get("strategy", "?"),
                credit=0.0, max_loss=0.0, realized_pl=0.0,
                exit_signal="", expiration=str(rs.get("expiration", "")),
            )
            tr.closed_at = ts.isoformat()
            tr.realized_pl = float(rs.get("net_unrealized_pl", 0) or 0)
            tr.exit_signal = str(rs.get("exit_signal", ""))
            closed.append(tr)

    return closed, (first_balance or 0.0), (last_balance or 0.0)


# ---------------------------------------------------------------------------
# Backtest-side runner
# ---------------------------------------------------------------------------


def run_backtest(
    start: date, end: date, tickers: Sequence[str],
    starting_equity: float,
) -> Tuple[List, float, float]:
    """Run the BacktestRunner against the active preset.

    Returns (closed_trades, starting_equity, ending_equity).
    """
    # Imports here so the script can show its --help even if scipy is
    # missing on the host.
    from trading_agent.backtest.runner import BacktestRunner
    from trading_agent.strategy_presets import load_active_preset
    preset = load_active_preset()
    # Force adaptive scan_mode so the backtester routes through decide()
    # (CI invariant #3 requirement). load_active_preset already does
    # this in practice, but be explicit.
    if getattr(preset, "scan_mode", "adaptive") != "adaptive":
        from dataclasses import replace
        preset = replace(preset, scan_mode="adaptive")
    runner = BacktestRunner(
        tickers=tuple(tickers), start=start, end=end,
        preset=preset, starting_equity=starting_equity,
    )
    result = runner.run()
    return (
        list(getattr(result, "closed_trades", []) or []),
        float(result.starting_equity),
        float(result.ending_equity),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _format_trade_table(rows: List[Tuple[str, ...]]) -> str:
    if not rows:
        return "  (no trades)"
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for r in rows:
        out.append("  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def report(start: date, end: date, tickers: Sequence[str],
           live: List[LiveTrade], live_start: float, live_end: float,
           bt_trades: List, bt_start: float, bt_end: float) -> None:
    print()
    print("=" * 76)
    print(f"BACKTEST vs LIVE — window {start} → {end}  ({(end - start).days + 1}d)")
    print(f"Universe: {', '.join(tickers)}")
    print("=" * 76)

    # ── Headline ──────────────────────────────────────────────────
    live_pl = sum(t.realized_pl for t in live)
    live_ret_pct = (
        (live_end - live_start) / live_start * 100 if live_start > 0 else 0
    )
    bt_pl = sum(
        getattr(c, "realized_pl", 0) for c in bt_trades
    )
    bt_ret_pct = (
        (bt_end - bt_start) / bt_start * 100 if bt_start > 0 else 0
    )

    print("\nACCOUNT-LEVEL")
    print(f"  Live:     ${live_start:,.2f} → ${live_end:,.2f}   "
          f"({live_ret_pct:+.2f}% / ${live_end - live_start:+,.2f})")
    print(f"  Backtest: ${bt_start:,.2f} → ${bt_end:,.2f}   "
          f"({bt_ret_pct:+.2f}% / ${bt_end - bt_start:+,.2f})")
    gap_pct = bt_ret_pct - live_ret_pct
    gap_word = "outperformed" if gap_pct > 0 else "underperformed"
    print(f"  → Live {gap_word} backtest by {abs(gap_pct):.2f}pp")

    print("\nREALIZED-P&L SUM (from closed trades only)")
    print(f"  Live:     ${live_pl:+,.2f}  ({len(live)} closed trades)")
    print(f"  Backtest: ${bt_pl:+,.2f}  ({len(bt_trades)} closed trades)")

    # ── Win rate ──────────────────────────────────────────────────
    def _winrate(pls):
        if not pls: return 0.0, 0.0, 0.0, 0
        wins = [p for p in pls if p > 0]
        losses = [p for p in pls if p <= 0]
        n = len(pls)
        wr = 100 * len(wins) / n
        avg_w = sum(wins) / len(wins) if wins else 0
        avg_l = sum(losses) / len(losses) if losses else 0
        return wr, avg_w, avg_l, n

    live_wr, live_avg_w, live_avg_l, _ = _winrate([t.realized_pl for t in live])
    bt_wr, bt_avg_w, bt_avg_l, _ = _winrate([
        getattr(c, "realized_pl", 0) for c in bt_trades
    ])
    print("\nWIN RATE / AVG TRADE")
    print(f"  Live:     {live_wr:.0f}%  (avg win ${live_avg_w:+.2f} / "
          f"avg loss ${live_avg_l:+.2f})")
    print(f"  Backtest: {bt_wr:.0f}%  (avg win ${bt_avg_w:+.2f} / "
          f"avg loss ${bt_avg_l:+.2f})")
    print("  ↪ credit spreads typically need ≥65% win rate AND "
          "win/loss ratio ≥ 1.0 to be profitable")

    # ── Per-ticker rollup ─────────────────────────────────────────
    print("\nPER-TICKER P&L")
    by_t: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"live_pl": 0.0, "live_n": 0, "bt_pl": 0.0, "bt_n": 0}
    )
    for t in live:
        by_t[t.ticker]["live_pl"] += t.realized_pl
        by_t[t.ticker]["live_n"] += 1
    for c in bt_trades:
        tk = getattr(c, "ticker", "?")
        by_t[tk]["bt_pl"] += getattr(c, "realized_pl", 0)
        by_t[tk]["bt_n"] += 1
    rows = [("Ticker", "Live n", "Live P&L", "BT n", "BT P&L", "Δ (BT − Live)")]
    for tk in sorted(by_t):
        d = by_t[tk]
        delta = d["bt_pl"] - d["live_pl"]
        rows.append((
            tk, str(d["live_n"]), f"${d['live_pl']:+,.2f}",
            str(d["bt_n"]), f"${d['bt_pl']:+,.2f}",
            f"${delta:+,.2f}",
        ))
    print(_format_trade_table(rows))

    # ── Trade-by-trade alignment ─────────────────────────────────
    print("\nTRADE-LEVEL ALIGNMENT (same ticker fired in both?)")
    live_days = defaultdict(list)
    for t in live:
        d = (t.closed_at or t.opened_at or "")[:10]
        live_days[(t.ticker, d)].append(t)
    bt_days = defaultdict(list)
    for c in bt_trades:
        d = (
            getattr(c, "closed_at", None)
            or getattr(c, "opened_at", None)
            or ""
        )
        # closed_at may be a datetime — coerce to date string
        d_str = (
            d.date().isoformat() if hasattr(d, "date")
            else str(d)[:10]
        )
        bt_days[(getattr(c, "ticker", "?"), d_str)].append(c)
    all_keys = set(live_days) | set(bt_days)
    rows = [("Date", "Ticker", "Live (n, P&L)", "Backtest (n, P&L)", "Match?")]
    for k in sorted(all_keys):
        tk, d = k
        lv = live_days.get(k, [])
        bt = bt_days.get(k, [])
        lv_str = (
            f"{len(lv)}, ${sum(t.realized_pl for t in lv):+,.2f}"
            if lv else "—"
        )
        bt_str = (
            f"{len(bt)}, "
            f"${sum(getattr(c, 'realized_pl', 0) for c in bt):+,.2f}"
            if bt else "—"
        )
        match = (
            "✓" if (lv and bt) else
            "live-only" if lv else "bt-only"
        )
        rows.append((d, tk, lv_str, bt_str, match))
    print(_format_trade_table(rows))

    # ── Diagnostics ──────────────────────────────────────────────
    only_live = [k for k in live_days if k not in bt_days]
    only_bt = [k for k in bt_days if k not in live_days]
    print("\nDIAGNOSTIC SIGNALS")
    if only_live:
        print(f"  Live-only trades (backtest missed these): {len(only_live)}")
        for tk, d in only_live[:10]:
            print(f"    {d}  {tk}")
        print("    → If backtest had MORE permissive gates / wrong "
              "regime classification, it would skip trades live took.")
        print("    → If live opened on a signal backtest doesn't compute "
              "(e.g. RSI gate, sentiment), that's the divergence.")
    if only_bt:
        print(f"  Backtest-only trades (live missed these): {len(only_bt)}")
        for tk, d in only_bt[:10]:
            print(f"    {d}  {tk}")
        print("    → Live may have been blocked by a PDT lockout, "
              "dedup gate, or chain-fetch failure that backtest doesn't model.")

    print()
    print("INTERPRETATION GUIDE")
    print("  • Gap < 2pp:  parity holds. Strategy validation valid.")
    print("  • Gap 2-5pp:  meaningful slippage / partial-fill drag in live.")
    print("  • Gap > 5pp:  systemic problem — chain fetch, broker rejection, "
          "or strategy regime classifier diverging.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--tickers", required=True,
                   help="Comma-separated, e.g. SPY,DIA,XLF,GLD,XLE")
    p.add_argument("--journal",
                   default="trade_journal/signals_live.jsonl",
                   help="Path to live journal (default: trade_journal/signals_live.jsonl)")
    p.add_argument("--starting-equity", type=float, default=5000.0,
                   help="Seed for the backtest SimAccount (default: 5000)")
    args = p.parse_args(argv)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if not Path(args.journal).is_file():
        print(f"ERROR: journal not found at {args.journal}", file=sys.stderr)
        return 1

    print(f"Loading live trades from {args.journal}…")
    live, live_start_bal, live_end_bal = load_live_trades(
        args.journal, start, end,
    )
    print(f"  {len(live)} completed trade(s) in window")

    print(f"\nRunning backtest {start} → {end} for {tickers}…")
    print("(this may take 30-90s on yfinance cold cache)")
    bt_trades, bt_start, bt_end = run_backtest(
        start, end, tickers, args.starting_equity,
    )
    print(f"  {len(bt_trades)} backtest closed trade(s)")

    # If live balance endpoints weren't found in the window, fall back
    # to the same starting equity so the report still computes.
    if live_start_bal <= 0:
        live_start_bal = args.starting_equity
    if live_end_bal <= 0:
        live_end_bal = live_start_bal + sum(t.realized_pl for t in live)

    report(
        start, end, tickers,
        live, live_start_bal, live_end_bal,
        bt_trades, bt_start, bt_end,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
