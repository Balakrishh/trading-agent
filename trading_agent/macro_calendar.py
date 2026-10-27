"""macro_calendar.py — scheduled US macro releases that move every ETF (skill 68).

Added 2026-10-27. The earnings guard (skill 65) keeps single-stock trades
away from their reports; ETFs have no earnings, but they gap on three
scheduled releases:

* **FOMC** rate decision — 14:00 ET on the last day of each meeting
  (Federal Reserve calendars, 2015 → 2027; unscheduled and notation-vote
  meetings excluded — nobody could have planned around them);
* **CPI** — 08:30 ET (BLS, via FRED release 10);
* **NFP** (Employment Situation) — 08:30 ET (BLS, via FRED release 50).

The built-in tables below are the single source for the agent, the
backtest (``scripts/research/macro_vrp_credit_study.py``) and the MCP
surface. CPI / NFP dates are published about a year ahead, so the table
runs out: ``refresh()`` (``python -m trading_agent.macro_calendar
--refresh``, run weekly by the gamma-exposure timer) re-reads the Fed and
FRED calendars into ``trade_journal/macro_events.json``, which is merged
over the built-ins. ``coverage_warnings()`` names any kind whose last
known date is under ``MIN_LOOKAHEAD_DAYS`` away.

FRED's public release calendar only lists 2025 onward, so CPI / NFP
history here starts in 2025; FOMC goes back to 2015.

    python -m trading_agent.macro_calendar                # next events + coverage
    python -m trading_agent.macro_calendar --refresh      # update the json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

STORE_PATH = Path(os.environ.get("TRADING_AGENT_MACRO_EVENTS",
                                 "trade_journal/macro_events.json"))
KINDS = ("FOMC", "CPI", "NFP")
# Release time, US/Eastern.
RELEASE_ET = {"FOMC": time(14, 0), "CPI": time(8, 30), "NFP": time(8, 30)}
MIN_LOOKAHEAD_DAYS = 45

_FOMC = (
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29", "2015-09-17",
    "2015-10-28", "2015-12-16", "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14", "2017-02-01", "2017-03-15",
    "2017-05-03", "2017-06-14", "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26",
    "2018-11-08", "2018-12-19", "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11", "2020-01-29", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16", "2021-01-27",
    "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03",
    "2021-12-15", "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14", "2023-02-01", "2023-03-22", "2023-05-03",
    "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13", "2024-01-31",
    "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07",
    "2024-12-18", "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10", "2026-01-28", "2026-03-18", "2026-04-29",
    "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09", "2027-01-27",
    "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28", "2027-09-15", "2027-10-27",
    "2027-12-08",
)
_CPI = (
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13", "2025-06-11",
    "2025-07-15", "2025-08-12", "2025-09-11", "2025-10-24", "2025-12-18", "2026-01-13",
    "2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12", "2026-06-10", "2026-07-14",
    "2026-08-12", "2026-09-11", "2026-10-14", "2026-11-10", "2026-12-10",
)
_NFP = (
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04", "2025-05-02", "2025-06-06",
    "2025-07-03", "2025-08-01", "2025-09-05", "2025-11-20", "2025-12-16", "2026-01-09",
    "2026-02-11", "2026-03-06", "2026-04-03", "2026-05-08", "2026-06-05", "2026-07-02",
    "2026-08-07", "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
)
_BUILTIN = {"FOMC": _FOMC, "CPI": _CPI, "NFP": _NFP}


@dataclass(frozen=True)
class MacroEvent:
    kind: str
    day: date

    @property
    def release_et(self) -> time:
        return RELEASE_ET[self.kind]

    def label(self) -> str:
        return f"{self.kind} {self.day.isoformat()} {self.release_et.strftime('%H:%M')} ET"


def _load_extra(path: Optional[Path]) -> Dict[str, List[str]]:
    try:
        data = json.loads(Path(path or STORE_PATH).read_text())
        return {k: list(v) for k, v in (data.get("events") or {}).items() if k in KINDS}
    except (OSError, ValueError, AttributeError):
        return {}


def all_events(path: Optional[Path] = None) -> List[MacroEvent]:
    """Built-in tables merged with the refreshed json, sorted by day."""
    extra = _load_extra(path)
    out = set()
    for kind in KINDS:
        for d in tuple(_BUILTIN[kind]) + tuple(extra.get(kind, ())):
            try:
                out.add(MacroEvent(kind, date.fromisoformat(d)))
            except (TypeError, ValueError):
                continue
    return sorted(out, key=lambda e: (e.day, e.kind))


def events_between(start: date, end: date, events: Optional[Iterable[MacroEvent]] = None,
                   kinds: Iterable[str] = KINDS) -> List[MacroEvent]:
    """Events with ``start <= day <= end``."""
    ks = set(kinds)
    return sorted((e for e in (events if events is not None else all_events())
                   if start <= e.day <= end and e.kind in ks), key=lambda e: (e.day, e.release_et))


def next_event(on: date, events: Optional[Iterable[MacroEvent]] = None) -> Optional[MacroEvent]:
    later = [e for e in (events if events is not None else all_events()) if e.day >= on]
    return min(later, key=lambda e: (e.day, e.release_et)) if later else None


def coverage_warnings(today: date, events: Optional[Iterable[MacroEvent]] = None,
                      lookahead: int = MIN_LOOKAHEAD_DAYS) -> List[str]:
    evs = list(events if events is not None else all_events())
    out = []
    for kind in KINDS:
        days = [e.day for e in evs if e.kind == kind]
        last = max(days) if days else None
        if last is None or (last - today).days < lookahead:
            out.append(f"{kind} dates known only to {last} — run "
                       f"`python -m trading_agent.macro_calendar --refresh`")
    return out


# ── Refresh from the Fed and FRED ─────────────────────────────────────

FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FRED_URL = ("https://fred.stlouisfed.org/releases/calendar?od=asc&rid={rid}"
            "&view=year&vs={y}-01-01&ve={y}-12-31")
FRED_RELEASE_IDS = {"CPI": 10, "NFP": 50}
_MONTHS = {m[:3]: i for i, m in enumerate(
    ("January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"), 1)}
_FRED_DAY = re.compile(r"(?:Monday|Tuesday|Wednesday|Thursday|Friday) "
                       r"((?:January|February|March|April|May|June|July|August|September|"
                       r"October|November|December) \d{1,2}, 20\d\d)")


def parse_fomc_page(html: str) -> List[date]:
    """Decision days (last day of each scheduled meeting) on the Fed's
    current calendars page; unscheduled / notation-vote rows skipped."""
    marks = sorted((int(m.group(1)), m.start())
                   for m in re.finditer(r"(\d{4}) FOMC Meetings", html))
    marks.sort(key=lambda x: x[1])
    out = []
    for i, (year, start) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(html)
        for month, days in re.findall(
                r"fomc-meeting__month[^>]*>\s*<strong>([^<]*)</strong>.*?"
                r"fomc-meeting__date[^>]*>([^<]*)<", html[start:end], re.S):
            if re.search("unscheduled|notation|cancel", days, re.I):
                continue
            nums = re.findall(r"\d+", days)
            key = month.split("/")[-1].strip()[:3]
            if nums and key in _MONTHS:
                out.append(date(year, _MONTHS[key], int(nums[-1])))
    return sorted(set(out))


def parse_fred_calendar(html: str) -> List[date]:
    return sorted({datetime.strptime(s, "%B %d, %Y").date() for s in _FRED_DAY.findall(html)})


def refresh(today: Optional[date] = None, path: Optional[Path] = None, get=None) -> Dict[str, int]:
    """Re-read the Fed and FRED calendars (this year and next) and store
    the dates; returns the count per kind. A failed source keeps its
    previously stored dates."""
    import requests
    get = get or (lambda url: requests.get(url, timeout=30).text)
    today = today or date.today()
    fp = Path(path or STORE_PATH)
    stored = _load_extra(fp)
    events: Dict[str, List[str]] = {k: list(stored.get(k, [])) for k in KINDS}
    try:
        events["FOMC"] = sorted({*events["FOMC"], *(d.isoformat() for d in parse_fomc_page(get(FOMC_URL)))})
    except Exception as exc:                          # noqa: BLE001 — keep the stored dates
        logger.warning("FOMC calendar refresh failed: %s", exc)
    for kind, rid in FRED_RELEASE_IDS.items():
        for y in (today.year, today.year + 1):
            try:
                days = parse_fred_calendar(get(FRED_URL.format(rid=rid, y=y)))
                events[kind] = sorted({*events[kind], *(d.isoformat() for d in days)})
            except Exception as exc:                  # noqa: BLE001 — keep the stored dates
                logger.warning("%s calendar refresh (%d) failed: %s", kind, y, exc)
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_suffix(fp.suffix + ".tmp")
    tmp.write_text(json.dumps({"refreshed_at": datetime.now().isoformat(timespec="seconds"),
                               "events": events}, indent=1))
    tmp.replace(fp)
    return {k: len(v) for k, v in events.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    today = date.today()
    if args.refresh:
        print("stored:", refresh(today))
    for e in events_between(today, today + timedelta(days=45)):
        print(e.label())
    for w in coverage_warnings(today):
        print("WARNING:", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
