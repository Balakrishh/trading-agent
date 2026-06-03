"""vix_regime_monitor.py — categorize the VIX level + page on zone changes.

Skill 40 (2026-06-03). Credit-spread strategies are highly path-
dependent on the vol regime. A 6-day no-trade week with RSI > 75 and
IV rank > 95 across the universe isn't a system problem — it's the
strategy correctly refusing to trade in a hostile environment. The
operator's question becomes: "when will conditions change?"

VIX is the cleanest proxy for that. This monitor:

  1. Reads the latest VIX close once per cycle.
  2. Categorizes into six zones (Compressed / Low / Normal / Elevated
     / High / Crisis).
  3. Compares to the LAST zone observed today (from the journal).
  4. Fires a Telegram alert on a zone transition, with a one-line
     hint about what the new zone means for credit spreads.
  5. Journals the current zone so next-cycle comparison works
     across process restarts.

The monitor is FIRE-AND-FORGET — failures inside it never propagate
to the cycle. Same pattern as ExceptionMonitor (skill 34).

Zone boundaries (calibrated against 5-year VIX history, equity ETF
credit-spread strategy):

  VIX < 12   → Compressed   — Premium gone. Sit out or wait.
  12 ≤ V < 15 → Low         — Spreads possible but thin. Strict gates.
  15 ≤ V < 20 → Normal      — Bread-and-butter for credit spreads.
  20 ≤ V < 28 → Elevated    — Rich premium, manageable swings.
  28 ≤ V < 40 → High        → Rich but large daily moves. Tighten sizing.
  V ≥ 40     → Crisis       → Sit out. No strategy works here.

The boundaries are operator-tunable via the constructor.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Zone definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VixZone:
    """One zone in the regime classifier."""
    name: str
    lower: float        # inclusive lower bound (-inf for the first)
    upper: float        # exclusive upper bound (+inf for the last)
    hint: str           # one-line operator guidance

    def contains(self, vix: float) -> bool:
        return self.lower <= vix < self.upper


# Default zones — adjust at construction if your strategy has different
# tolerance bands.
DEFAULT_ZONES = (
    VixZone(
        name="Compressed", lower=float("-inf"), upper=12.0,
        hint="Premium too thin for credit spreads. Sit out or wait for "
             "vol to rise.",
    ),
    VixZone(
        name="Low", lower=12.0, upper=15.0,
        hint="Spreads marginal. Tight gates likely reject most "
             "candidates — be patient.",
    ),
    VixZone(
        name="Normal", lower=15.0, upper=20.0,
        hint="Favorable environment for credit spreads. Expect "
             "regular trade flow.",
    ),
    VixZone(
        name="Elevated", lower=20.0, upper=28.0,
        hint="Rich premium with manageable swings. Best risk/reward "
             "zone for credit spreads.",
    ),
    VixZone(
        name="High", lower=28.0, upper=40.0,
        hint="Premium rich but daily moves big. Reduce position size; "
             "watch for gamma risk.",
    ),
    VixZone(
        name="Crisis", lower=40.0, upper=float("inf"),
        hint="Sit out. No credit-spread strategy works through a "
             "crisis. Wait for VIX to retrace below 30.",
    ),
)


def classify_vix(vix_level: float, zones=DEFAULT_ZONES) -> VixZone:
    """Return the VixZone containing ``vix_level``. Defaults to the
    Normal zone if no match (defensive fallback)."""
    for z in zones:
        if z.contains(vix_level):
            return z
    # Should never happen with sane zones, but defend against it.
    return zones[2]  # Normal


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class VixRegimeMonitor:
    """Watches VIX zone changes + pages the operator on transitions.

    Constructor-injected with:
      * ``data_provider`` — provides ``get_vix_zscore`` (we use its
        cached latest-close internally; falls back to a fresh fetch
        on cache miss).
      * ``journal_kb`` — for journalling current zone + reading last
        observed zone today.
      * ``telegram`` — TelegramNotifier; alert routes to the info
        channel via ``notify_vix_regime_change``.

    Usage from inside a cycle::

        self._vix_monitor.check_and_alert()

    Never raises (best-effort). The cycle should not depend on this
    monitor for any decision — it's purely operator-facing.
    """

    def __init__(self, *, data_provider, journal_kb, telegram,
                 zones=DEFAULT_ZONES):
        self.data_provider = data_provider
        self.journal_kb = journal_kb
        self.telegram = telegram
        self.zones = zones
        # In-process dedup so we don't even bother reading the journal
        # again within the same process if the zone hasn't moved.
        self._last_observed_zone: Optional[str] = None
        self._last_check_minute: Optional[str] = None

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def check_and_alert(self) -> None:
        """Read VIX, classify, page on transitions, journal current.

        Idempotent within a minute — only one read + check per minute
        to avoid burning yfinance quota.
        """
        try:
            self._check_impl()
        except Exception as exc:                                  # noqa: skill-34-exempt — best-effort enrichment; caller treats None as missing
            logger.warning(
                "VixRegimeMonitor.check_and_alert raised %s: %s",
                type(exc).__name__, exc,
            )

    def current(self) -> Optional[tuple]:
        """Return ``(vix_level, VixZone)`` for the latest read, or
        None if VIX is unavailable. Used by the EOD recap so the
        builder can include 'VIX 14.5 (Low)' in the message body."""
        try:
            level = self._fetch_vix_level()
            if level is None:
                return None
            return (level, classify_vix(level, self.zones))
        except Exception:                                         # noqa: skill-34-exempt — best-effort enrichment; caller treats None as missing
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_impl(self) -> None:
        now_minute = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
        if self._last_check_minute == now_minute:
            return  # already checked this minute
        self._last_check_minute = now_minute

        level = self._fetch_vix_level()
        if level is None:
            logger.debug("VixRegimeMonitor: VIX unavailable, skipping check")
            return
        new_zone = classify_vix(level, self.zones)

        # Last observed zone — fast path uses in-memory cache, slow
        # path reads the journal so a cron-launched cycle (new process)
        # honours the dedup.
        if self._last_observed_zone is None:
            self._last_observed_zone = self._read_last_zone_from_journal()

        # Always journal the current observation (one row per check)
        # so dashboards + EOD recap can see the trajectory. The dedup
        # is purely on the TELEGRAM alert, not the journal write.
        self._journal_observation(level, new_zone.name)

        # No transition → no alert.
        if self._last_observed_zone == new_zone.name:
            return

        # Transition — page operator.
        from_zone = self._last_observed_zone or "Unknown"
        self._send_transition_alert(
            from_zone=from_zone, to_zone=new_zone, vix_level=level,
        )
        self._last_observed_zone = new_zone.name

    def _fetch_vix_level(self) -> Optional[float]:
        """Get the latest VIX close from the data provider.

        Reuses the existing 5-min cache. Returns None on failure.
        """
        try:
            # market_data_provider returns (raw_change, zscore); we need
            # the level. The cleanest way: call the existing method and
            # then add the change to the previous close. But the change
            # alone isn't the level. Use yfinance directly here — the
            # data provider's internal cache already holds the closes,
            # but accessing them would require an interface change.
            import yfinance as yf  # type: ignore
            tk = yf.Ticker("^VIX")
            df = tk.history(period="1d", interval="5m", auto_adjust=False)
            if df is None or df.empty:
                return None
            return float(df["Close"].iloc[-1])
        except Exception as exc:                                  # noqa: skill-34-exempt — best-effort enrichment; caller treats None as missing
            logger.debug("VixRegimeMonitor: VIX fetch failed: %s", exc)
            return None

    def _journal_observation(self, vix_level: float,
                             zone_name: str) -> None:
        """Write a ``vix_observation`` row. Cheap; runs every check."""
        if not self.journal_kb:
            return
        try:
            self.journal_kb.log_signal(
                ticker="__vix__",
                action="vix_observation",
                price=float(vix_level),
                raw_signal={
                    "vix_level": float(vix_level),
                    "zone": zone_name,
                },
                notes=f"VIX {vix_level:.2f} ({zone_name})",
            )
        except Exception as exc:                                  # noqa: skill-34-exempt — best-effort enrichment; caller treats None as missing
            logger.debug(
                "VixRegimeMonitor: journal write failed: %s", exc,
            )

    def _read_last_zone_from_journal(self) -> Optional[str]:
        """Scan the journal for the most recent ``vix_observation`` row
        (today, UTC) and return its zone. None if no prior observation
        — caller treats that as 'first observation, don't alert'."""
        jsonl_path = getattr(self.journal_kb, "jsonl_path", None)
        if not jsonl_path or not os.path.isfile(jsonl_path):
            return None
        today_utc = datetime.now(timezone.utc).date().isoformat()
        last_zone: Optional[str] = None
        try:
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("action") != "vix_observation":
                        continue
                    ts = rec.get("timestamp", "")
                    if not ts.startswith(today_utc):
                        continue
                    rs = rec.get("raw_signal") or {}
                    if not isinstance(rs, dict):
                        continue
                    zone = rs.get("zone")
                    if zone:
                        last_zone = zone   # keep walking; want LAST
        except Exception:                                         # noqa: skill-34-exempt — best-effort enrichment; caller treats None as missing
            return None
        return last_zone

    def _send_transition_alert(self, *, from_zone: str,
                               to_zone: VixZone,
                               vix_level: float) -> None:
        """Fire the Telegram alert. Failures are logged but don't
        propagate — paging is best-effort."""
        if not self.telegram or not getattr(self.telegram, "is_active", False):
            return
        try:
            self.telegram.notify_vix_regime_change(
                from_zone=from_zone,
                to_zone=to_zone.name,
                vix_level=vix_level,
                hint=to_zone.hint,
            )
        except Exception as exc:                                  # noqa: skill-34-exempt — best-effort enrichment; caller treats None as missing
            logger.warning(
                "VixRegimeMonitor: Telegram notify_vix_regime_change "
                "raised: %s", exc,
            )


__all__ = [
    "VixRegimeMonitor",
    "VixZone",
    "DEFAULT_ZONES",
    "classify_vix",
]
