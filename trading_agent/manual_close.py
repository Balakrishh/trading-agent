"""manual_close.py — the operator closes one open spread now (skill 63).

The agent closes positions only on its own exit rules. This is the
operator's lever: the same atomic close order the agent sends (one mleg
limit order, every leg fills together or none does), journalled through
the agent's own close writer so the row is an ordinary ``closed`` row
with ``exit_signal="manual"``.

    python -m trading_agent.manual_close --ticker IWM              # preview — no order
    python -m trading_agent.manual_close --ticker IWM --submit     # close it
        [--strategy "Put Debit Spread"] [--reason "taking profit early"]

Preview is the default. Quotes come from Alpaca, never Schwab: the live
agent holds the Schwab token, whose refresh token rotates on every use.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, List, Optional, Tuple

from trading_agent.executor import close_order_prices, realized_pl_from_close
from trading_agent.position_monitor import (ExitSignal, SpreadPosition,
                                            load_trade_plans, remark_positions_at_mid)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_NOT_FOUND = 1
EXIT_REFUSED = 2
EXIT_NOT_FILLED = 3

DEFAULT_REASON = "operator manual close"


def select_spread(spreads: List[SpreadPosition], ticker: str,
                  strategy: Optional[str]) -> Tuple[Optional[SpreadPosition], Optional[str]]:
    """The one open spread on ``ticker`` (and ``strategy``, when given),
    or an error message. Never picks silently between two positions."""
    ticker = ticker.upper()
    found = [s for s in spreads if s.underlying.upper() == ticker
             and (strategy is None or s.strategy_name.lower() == strategy.lower())]
    if not found:
        held = ", ".join(sorted(f"{s.underlying} {s.strategy_name}" for s in spreads)) or "none"
        return None, f"No open {ticker}{' ' + strategy if strategy else ''} position. Open: {held}."
    if len(found) > 1:
        names = "; ".join(f"{s.strategy_name} exp {s.expiration}" for s in found)
        return None, f"{len(found)} open {ticker} positions ({names}) — pass --strategy."
    return found[0], None


def refusal(spread: SpreadPosition, *, market_open: bool, dry_run: bool) -> Optional[str]:
    """Why ``spread`` cannot be closed with --submit right now, or None."""
    if len(spread.legs) < 2:
        return (f"{spread.underlying} {spread.strategy_name} is a single leg — "
                "close it in the Alpaca UI (the atomic close needs two or more legs).")
    qtys = {abs(int(leg.qty)) for leg in spread.legs}
    if len(qtys) != 1:
        return (f"Unequal leg quantities {sorted(qtys)} (a partial fill) — "
                "close it in the Alpaca UI.")
    if dry_run:
        return "DRY_RUN is on — the agent would not trade, so neither does this."
    if not market_open:
        return "The market is closed — options orders need the regular session."
    return None


def preview(spread: SpreadPosition, quotes: Dict) -> Dict:
    """Close prices per share and the P&L each would realize. ``priced``
    is False when a leg lacks a two-sided quote."""
    priced = close_order_prices(spread.legs, quotes)
    out: Dict = {"ticker": spread.underlying, "strategy": spread.strategy_name,
                 "expiration": spread.expiration, "contracts": spread.contracts_open,
                 "legs": [{"symbol": l.symbol, "qty": int(l.qty),
                           "bid": (quotes.get(l.symbol) or {}).get("bid"),
                           "ask": (quotes.get(l.symbol) or {}).get("ask")} for l in spread.legs],
                 "priced": priced is not None}
    if priced is None:
        return out
    mid, natural, _ = priced
    improved = round((mid + natural) / 2, 2)
    out.update({"mid": round(mid, 4), "natural": round(natural, 4), "first_limit": improved,
                "pl_at_first_limit": realized_pl_from_close(spread.legs, improved),
                "pl_at_natural": realized_pl_from_close(spread.legs, round(natural, 2))})
    return out


def close_context(spread: SpreadPosition, reason: str) -> Dict:
    """The close row's payload — the same keys the agent's own close path
    writes (agent._monitor_positions), so every journal reader works."""
    return {
        "strategy": spread.strategy_name,
        "exit_signal": ExitSignal.MANUAL.value,
        "exit_reason": reason,
        "exit_immediate": True,
        "net_unrealized_pl": float(spread.net_unrealized_pl or 0),
        "original_credit": float(spread.original_credit or 0),
        "max_loss": float(spread.max_loss or 0),
        "spread_width": float(spread.spread_width or 0),
        "expiration": spread.expiration or "",
        "short_strikes": list(spread.short_strikes or []),
        "regime_at_close": "unknown",
        "origin": spread.origin,
    }


def apply_fill(ctx: Dict, result: Dict) -> Tuple[Dict, str]:
    """Fold the executor's close result into ``ctx``; return the
    fill_status the close writer expects ("complete" | "partial")."""
    ctx = dict(ctx, signal_mark_pl=ctx["net_unrealized_pl"])
    realized = result.get("realized_pl")
    if realized is not None:
        ctx.update(net_unrealized_pl=float(realized),
                   close_fill_debit=result.get("fill_debit"), pl_source="fill")
    else:
        ctx["pl_source"] = "signal_mark"
    return ctx, ("complete" if result.get("all_closed") else "partial")


def _print_preview(p: Dict) -> None:
    print(f"{p['ticker']} {p['strategy']} exp {p['expiration']} × {p['contracts']}")
    for leg in p["legs"]:
        print(f"  {leg['symbol']:<22} qty {leg['qty']:>4}  bid {leg['bid']}  ask {leg['ask']}")
    if not p["priced"]:
        print("  No two-sided quote on every leg — cannot price the close.")
        return
    def side(v):
        return f"{abs(v):.2f} {'credit' if v < 0 else 'debit'}"
    print(f"  close at mid {side(p['mid'])}, natural {side(p['natural'])}")
    print(f"  order tries {side(p['first_limit'])} first (P&L {p['pl_at_first_limit']}), "
          f"then natural (P&L {p['pl_at_natural']})")


def _build(cfg):
    """Broker adapters and the agent's close writer, wired as the agent wires them."""
    from trading_agent.agent import CLOSE_COOLDOWN_MINUTES, PARTIAL_CLOSE_COOLDOWN_THRESHOLD
    from trading_agent.close_event_collaborators import (CloseAlertNotifier, CloseJournalWriter,
                                                         PartialFillCooldown, PdtBlockDetector)
    from trading_agent.executor import OrderExecutor
    from trading_agent.journal_kb import JournalKB
    from trading_agent.market_data import MarketDataProvider
    from trading_agent.position_monitor import PositionMonitor
    from trading_agent.telegram_notifier import TelegramNotifier

    a = cfg.alpaca
    data = MarketDataProvider(alpaca_api_key=a.api_key, alpaca_secret_key=a.secret_key,
                              alpaca_data_url=a.data_url, alpaca_base_url=a.base_url)
    monitor = PositionMonitor(api_key=a.api_key, secret_key=a.secret_key, base_url=a.base_url)
    executor = OrderExecutor(api_key=a.api_key, secret_key=a.secret_key, base_url=a.base_url,
                             trade_plan_dir=cfg.logging.trade_plan_dir,
                             dry_run=cfg.trading.dry_run, data_provider=data)
    journal_dir = (cfg.intelligence.journal_dir
                   if cfg.intelligence and cfg.intelligence.journal_dir else "trade_journal")
    journal = JournalKB(journal_dir, run_mode="live")
    telegram = TelegramNotifier()

    def send_alert(*, ticker, alert_type, send_fn, **payload):
        try:
            send_fn(ticker=ticker, **payload)
        except Exception as exc:                                  # noqa: BLE001, skill-34-exempt — alert is best-effort; the close row is already written
            logger.warning("[%s] %s alert failed: %s", ticker, alert_type, exc)

    writer = CloseJournalWriter(
        journal_kb=journal,
        cooldown=PartialFillCooldown(journal_kb=journal, threshold=PARTIAL_CLOSE_COOLDOWN_THRESHOLD,
                                     window_min=CLOSE_COOLDOWN_MINUTES),
        pdt_detector=PdtBlockDetector(journal_kb=journal),
        alerts=CloseAlertNotifier(send_alert=send_alert, telegram=telegram),
        price_lookup=lambda _t: 0.0,
    )
    return data, monitor, executor, writer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--strategy", default=None, help="needed when the ticker has two positions")
    ap.add_argument("--reason", default=DEFAULT_REASON)
    ap.add_argument("--submit", action="store_true", help="send the close order (default: preview)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from trading_agent.config import load_config
    from trading_agent.market_hours import is_within_market_hours
    cfg = load_config()
    data, monitor, executor, writer = _build(cfg)

    positions = monitor.fetch_open_positions()
    if positions is None:
        print("Broker position fetch failed — nothing done.")
        return EXIT_REFUSED
    spreads = monitor.group_into_spreads(positions, load_trade_plans(cfg.logging.trade_plan_dir))
    spread, err = select_spread(spreads, a.ticker, a.strategy)
    if spread is None:
        print(err)
        return EXIT_NOT_FOUND

    quotes = data.fetch_option_quotes([leg.symbol for leg in spread.legs]) or {}
    spread.legs = remark_positions_at_mid(spread.legs, quotes)
    _print_preview(preview(spread, quotes))
    print(f"  account: {cfg.alpaca.base_url}")
    if not a.submit:
        print("Preview only — re-run with --submit to send the close order.")
        return EXIT_OK

    why_not = refusal(spread, market_open=is_within_market_hours(), dry_run=cfg.trading.dry_run)
    if why_not:
        print(f"Refused: {why_not}")
        return EXIT_REFUSED

    spread.exit_signal, spread.exit_reason = ExitSignal.MANUAL, a.reason
    result = executor.close_spread_atomic(spread)
    if result is None:
        print("Not filled at the improved or the natural price — order cancelled, "
              "position unchanged, nothing journalled.")
        return EXIT_NOT_FILLED
    ctx, fill_status = apply_fill(close_context(spread, a.reason), result)
    writer.write(spread, ctx, leg_results=result.get("leg_results", []),
                 fill_status=fill_status, dry_run=False)
    if fill_status != "complete":
        print(f"Close order {result.get('close_method')} is unresolved — check the Alpaca UI. "
              "Journalled as close_failed.")
        return EXIT_NOT_FILLED
    print(f"Closed {spread.underlying} {spread.strategy_name}: {result.get('close_method')} "
          f"fill {result.get('fill_debit')} → realized P&L {ctx['net_unrealized_pl']:+.2f} "
          f"({ctx['pl_source']}). Journalled.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
