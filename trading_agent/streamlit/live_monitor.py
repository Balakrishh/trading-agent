"""
live_monitor.py — Live Monitoring tab.

This tab is the single entry point for the trading agent.
It can Start / Stop the agent loop directly, shows real-time
account metrics, open positions, equity curve, guardrail status,
and the recent signal journal.

Agent loop design
-----------------
The agent runs in a subprocess (not a thread) so that the
270-second os._exit(1) timeout guard in run_cycle() cannot kill
the Streamlit dashboard process. Each cycle is launched as:

    python -m trading_agent.agent --env .env

State is persisted in three files:
    AGENT_RUNNING  — sentinel file; presence = agent should keep looping
    AGENT_PID      — PID of the current cycle subprocess
    AGENT_LOG      — last 200 lines of agent stdout/stderr
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from trading_agent.streamlit.components import (
    GUARDRAIL_NAMES,
    equity_curve_chart,
    guardrail_grid,
    metric_row,
    positions_table,
    ungrouped_legs_table,
)
from trading_agent.strategy_presets import (
    AGGRESSIVE,
    BALANCED,
    CONSERVATIVE,
    PRESET_FILE,
    PRESETS,
    PresetConfig,
    load_active_preset,
    save_active_preset,
)


_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


def _parse_occ(symbol: str) -> Dict[str, str]:
    """
    Parse an OCC option symbol → underlying, expiration, type, strike.

    OCC format: ROOT(1-6) + YYMMDD(6) + C/P(1) + STRIKE*1000(8)
    Example:  SPY260516P00450000 → SPY, 2026-05-16, Put, 450.00

    Returns the raw symbol as fallback `underlying` if parsing fails.
    """
    m = _OCC_RE.match(symbol or "")
    if not m:
        return {
            "underlying":  symbol or "",
            "expiration": "",
            "type":       "",
            "strike":     "",
        }
    root, ymd, cp, strike = m.groups()
    return {
        "underlying":  root,
        "expiration": f"20{ymd[0:2]}-{ymd[2:4]}-{ymd[4:6]}",
        "type":        "Call" if cp == "C" else "Put",
        "strike":      f"{int(strike) / 1000:.2f}",
    }

# ---------------------------------------------------------------------------
# File-based state paths (relative to repo root)
# ---------------------------------------------------------------------------
# Live agent writes ``signals_live.jsonl`` (Phase 3 of the live↔backtest
# unification — backtests write ``signals_backtest.jsonl`` instead so
# the two streams never mingle in this dashboard or in the LLM corpus).
# Legacy ``signals.jsonl`` is read as a fallback so existing on-disk
# journal history is still surfaced after the rename.
JOURNAL_PATH        = Path("trade_journal/signals_live.jsonl")
LEGACY_JOURNAL_PATH = Path("trade_journal/signals.jsonl")
PAUSE_FLAG     = Path("PAUSED")
AGENT_RUNNING  = Path("AGENT_RUNNING")   # sentinel: agent loop is active
AGENT_PID      = Path("AGENT_PID")       # PID of the running cycle process
AGENT_LOG      = Path("AGENT_LOG")       # rolling log tail
DRY_RUN_FLAG   = Path("DRY_RUN_MODE")   # sentinel: inject DRY_RUN + FORCE_MARKET_OPEN

# Auto-refresh cadence for the live-monitor fragment. Was 30s before the
# watchdog-backed cache landed; now each rerun is O(1) on quiet ticks
# (cache hit → zero disk I/O), so we can poll aggressively for
# near-real-time UX. Override with LIVE_MONITOR_REFRESH_SECS env var if
# you're on a slow filesystem and want to back off.
REFRESH_INTERVAL   = int(os.environ.get("LIVE_MONITOR_REFRESH_SECS", "3"))
CYCLE_INTERVAL_SEC = 300  # 5-minute trading cycle

# Keyword fragments for mapping journal check strings → guardrail slots.
# These must match the EXACT strings written by ``risk_manager.evaluate``
# in lower-case.  Each fragment matched marks the row as belonging to
# that guardrail slot.  When extending: drop the fragment in
# lower-case and grep risk_manager.py to confirm the string actually
# appears.  Examples that motivated each fragment:
#   slot 0 (Plan Validity):       "Plan is structurally valid" / "Plan invalid: …"
#   slot 1 (Credit/Width):        "Credit/Width ratio 0.34 ≥ …"
#   slot 2 (Delta):               "Sold 480.0 |Δ|=0.180 ≤ 0.25"  (UNICODE Δ)
#   slot 3 (Max Loss):            "Max loss $200 ≤ 2% of …"
#   slot 4 (Paper Account):       "Account type is PAPER" / "Account type is 'live' — must be 'paper'"
#   slot 5 (Market Open):         "Market is currently OPEN" / "Market is currently CLOSED"
#   slot 6 (Bid/Ask Spread):      "Underlying bid/ask spread $0.02 < $0.05"
#   slot 7 (Buying Power):        "Buying power 40.0% used ≤ 80% limit"
_GUARDRAIL_KEYWORDS: List[List[str]] = [
    ["plan invalid", "plan is structurally valid"],
    ["credit/width", "credit ratio"],
    ["|δ|", "delta"],
    ["max loss"],
    ["account type"],
    ["market is currently"],
    ["bid/ask spread", "underlying spread"],
    ["buying power"],
]


# Per-guardrail regexes for extracting the headline numbers from the
# verbose check strings written by ``risk_manager.evaluate``.  Each
# pattern targets the SAME message format the risk manager writes today
# (see risk_manager.py:95-225).  If the format changes you must update
# the regex AND keep the contract — the dashboard relies on this to
# show "$200 ≤ $2000" instead of the raw "Max loss $200 ≤ 2% of …".
# Credit/Width is special — risk_manager writes two flavours of the
# floor label depending on whether delta-aware floors are enabled:
#   static:       "Credit/Width ratio 0.3400 ≥ 0.2"
#   delta-aware:  "Credit/Width ratio 0.3400 ≥ |Δ|×(1+edge)=0.180×1.10=0.1980"
# A single greedy regex either eats the floor (in static mode) or grabs
# the wrong number (the |Δ| value, not the final floor) in delta-aware
# mode.  Split into a head regex that locks in ratio+operator, then
# extract the LAST float from the remaining tail — works for both cases.
_RX_CW_HEAD  = re.compile(r"ratio\s+([0-9.]+)\s*([≥<])")
_RX_FLOAT    = re.compile(r"[0-9]+\.[0-9]+")
_RX_DELTA    = re.compile(r"\|Δ\|=([0-9.]+)\s*([≤>])\s*([0-9.]+)")
_RX_MAXLOSS  = re.compile(r"Max loss\s+\$([0-9.,]+)\s*([≤>])[^=]*=\$([0-9.,]+)")
_RX_BP       = re.compile(r"Buying power\s+([0-9.]+)%\s+used\s+([≤>])\s+([0-9.]+)%")
_RX_SPREAD   = re.compile(r"spread\s+\$([0-9.]+)\s*(<|>=)\s*\$([0-9.]+)")
_RX_STALE    = re.compile(r"=\s*([0-9.]+)%\s*>\s*stale threshold")
_RX_PAPER_F  = re.compile(r"is\s+'([^']+)'")


def _compact_for(name: str, state: str, detail: str) -> str:
    """Distill a verbose risk-manager check string into a value-bearing
    chip ≤ ~16 chars for the grid cell.

    Returns ``""`` when there's nothing useful to show (e.g. the cell
    has no data at all — the renderer falls back to just the emoji).
    The verbatim ``detail`` is still surfaced via the cell hover, so
    nothing is lost; the chip is for at-a-glance scanning.

    Examples (input → output)::

        Plan Validity / ok   "Plan is structurally valid"           → "valid"
        Credit/Width / ok    "Credit/Width ratio 0.3400 ≥ 0.20"     → "0.34 ≥ 0.20"
        Delta / ok           "Sold 480.0 |Δ|=0.180 ≤ 0.25"          → "Δ=0.18 ≤ 0.25"
        Max Loss / fail      "Max loss $5000 > 2% of $100,000 (=$2000)"
                                                                   → "$5000 > $2000"
        Paper / fail         "Account type is 'live' — must be …"   → "live"
        Market Open / warn   "Market is currently OPEN (FORCED — …)" → "FORCED"
        Bid/Ask / ok         "Underlying bid/ask spread $0.0200 < $0.0500 …"
                                                                   → "$0.02 < $0.05"
        Buying Power / ok    "Buying power 40.0% used ≤ 80% limit"  → "40% ≤ 80%"
    """
    if not detail or detail == "—":
        return ""

    if name == "Plan Validity":
        if state == "ok":
            return "valid"
        # detail looks like "Plan invalid: <reason>" — strip prefix
        return detail.removeprefix("Plan invalid: ")[:14] or "invalid"

    if name == "Credit/Width Ratio":
        m = _RX_CW_HEAD.search(detail)
        if m:
            ratio, op = m.group(1), m.group(2)
            try:
                ratio_short = f"{float(ratio):.2f}"
            except ValueError:
                ratio_short = ratio
            # Floor is the LAST float after the operator — handles both
            # the static floor ("0.2") and the delta-aware floor whose
            # final term is the computed product ("|Δ|×(1+edge)=…=0.1980").
            tail_text = detail[m.end():]
            floats = _RX_FLOAT.findall(tail_text)
            floor_short = ""
            if floats:
                try:
                    floor_short = f"{float(floats[-1]):.2f}"
                except ValueError:
                    floor_short = floats[-1]
            tail = f" {op} {floor_short}" if floor_short else f" {op}"
            return f"{ratio_short}{tail}"
        return ""

    if name == "Delta ≤ Max Delta":
        m = _RX_DELTA.search(detail)
        if m:
            d, op, lim = m.group(1), m.group(2), m.group(3)
            return f"Δ={float(d):.2f} {op} {lim}"
        return ""

    if name == "Max Loss ≤ 2% Equity":
        m = _RX_MAXLOSS.search(detail)
        if m:
            loss, op, allowed = m.group(1), m.group(2), m.group(3)
            return f"${loss} {op} ${allowed}"
        return ""

    if name == "Paper Account":
        if state == "ok":
            return "PAPER"
        m = _RX_PAPER_F.search(detail)
        return m.group(1) if m else "non-paper"

    if name == "Market Open":
        if state == "warn":
            return "FORCED"
        if state == "fail":
            return "CLOSED"
        return "OPEN"

    if name == "Bid/Ask Spread":
        if "STALE" in detail:
            m = _RX_STALE.search(detail)
            return f"STALE {m.group(1)}%" if m else "STALE"
        m = _RX_SPREAD.search(detail)
        if m:
            spread, op, floor = m.group(1), m.group(2), m.group(3)
            try:
                spread_short = f"${float(spread):.2f}"
                floor_short  = f"${float(floor):.2f}"
            except ValueError:
                spread_short, floor_short = f"${spread}", f"${floor}"
            disp_op = "≥" if op == ">=" else op
            return f"{spread_short} {disp_op} {floor_short}"
        return ""

    if name == "Buying Power ≤ 80%":
        m = _RX_BP.search(detail)
        if m:
            used, op, lim = m.group(1), m.group(2), m.group(3)
            return f"{float(used):.0f}% {op} {lim}%"
        return ""

    return ""


# ---------------------------------------------------------------------------
# Agent loop — runs in a background daemon thread inside the dashboard process
# ---------------------------------------------------------------------------

def _safe_read_text(fp: Path) -> str:
    """``Path.read_text`` that tolerates non-UTF-8 bytes.

    The agent-managed log files (``AGENT_LOG``, ``PAUSE_FLAG``,
    ``AGENT_PID``) are written by a chain of producers — Python's
    logging module, subprocess stdout pipes, sentinel writes — and very
    occasionally a stray non-UTF-8 byte (a Windows-1252 character from
    a vendor SDK trace, an emoji that got width-clipped, etc.) sneaks
    through. The default ``Path.read_text()`` raises ``UnicodeDecodeError``
    on such bytes and crashes the entire dashboard render. We replace
    bad bytes with U+FFFD instead so the log stays readable and the
    dashboard never goes down because of a single corrupt character.
    """
    try:
        return fp.read_text(encoding="utf-8", errors="replace")
    except (OSError, FileNotFoundError):
        return ""


def _append_log(line: str, max_lines: int = 200) -> None:
    """Append one line to AGENT_LOG, keeping only the last max_lines lines.

    Read uses ``_safe_read_text`` so a corrupt byte in the existing log
    can't lose the whole rolling buffer the way the previous strict-UTF-8
    decode + bare ``except: pass`` did. The write is always strict UTF-8
    (``write_text`` defaults), which means a malformed byte gets replaced
    by U+FFFD on the next read-then-rewrite cycle and self-heals.
    """
    try:
        existing = _safe_read_text(AGENT_LOG).splitlines() if AGENT_LOG.exists() else []
        existing.append(line.rstrip())
        AGENT_LOG.write_text(
            "\n".join(existing[-max_lines:]) + "\n",
            encoding="utf-8", errors="replace",
        )
    except Exception:
        pass


def _run_one_cycle(dry_run: bool = False) -> int:
    """
    Spawn a single agent cycle as a child process.

    Parameters
    ----------
    dry_run : bool
        When True, injects DRY_RUN=true and FORCE_MARKET_OPEN=true into the
        subprocess environment so the full agent pipeline runs — regime
        classification, option chain fetching, risk checks — but no orders
        are submitted to Alpaca. Useful for after-hours simulation.

    Returns the process exit code (0 = success, non-zero = error/timeout).
    """
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "-m", "trading_agent.agent"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    if dry_run:
        env["DRY_RUN"] = "true"
        env["FORCE_MARKET_OPEN"] = "true"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # Force UTF-8 with replace on the subprocess stdout pipe so a
        # stray non-UTF-8 byte from a vendor SDK trace, an emoji
        # mid-line-buffer, or a Windows-1252 character can't propagate
        # to AGENT_LOG and crash the dashboard render. Without this
        # the default locale-encoding choice on macOS sometimes lets
        # bytes like 0x80 (€ in CP1252) slip into the log and trigger
        # UnicodeDecodeError when the live-monitor tab tries to render.
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(repo_root),
        env=env,
    )
    AGENT_PID.write_text(str(proc.pid))

    for line in proc.stdout:
        _append_log(line)

    proc.wait()
    if AGENT_PID.exists():
        AGENT_PID.unlink(missing_ok=True)
    return proc.returncode


def _agent_loop() -> None:
    """
    Background thread: run trading cycles every CYCLE_INTERVAL_SEC seconds
    while AGENT_RUNNING sentinel file exists.

    Sleeps 1 second between ticks so the loop reacts quickly to a Stop request.
    """
    _append_log(f"[{_now()}] Agent loop started (PID {os.getpid()})")

    while AGENT_RUNNING.exists():
        if PAUSE_FLAG.exists():
            _append_log(f"[{_now()}] PAUSED — skipping cycle")
            time.sleep(5)
            continue

        is_dry = DRY_RUN_FLAG.exists()
        mode_label = "DRY-RUN" if is_dry else "LIVE"
        _append_log(f"[{_now()}] --- Cycle start [{mode_label}] ---")
        rc = _run_one_cycle(dry_run=is_dry)
        _append_log(f"[{_now()}] --- Cycle end [{mode_label}] (exit={rc}) ---")

        # Wait CYCLE_INTERVAL_SEC, checking every second for a stop/pause signal
        for _ in range(CYCLE_INTERVAL_SEC):
            if not AGENT_RUNNING.exists():
                break
            time.sleep(1)

    _append_log(f"[{_now()}] Agent loop stopped")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Start / Stop helpers
# ---------------------------------------------------------------------------

def _is_loop_running() -> bool:
    """True if the background loop thread is alive (checks sentinel file)."""
    return AGENT_RUNNING.exists()


def _is_dry_run_mode() -> bool:
    return DRY_RUN_FLAG.exists()


# ---------------------------------------------------------------------------
# Orphan-agent sweep — process singleton enforcement
# ---------------------------------------------------------------------------
# Pre-2026-05-06 the dashboard's Start/Stop relied entirely on the
# AGENT_RUNNING sentinel + a daemon-thread inside Streamlit. Three
# silent failure modes:
#   1. Stop only removed the sentinel — the in-flight 18-second cycle
#      subprocess kept running, so a quick Stop→Start spawned a 2nd
#      subprocess overlapping the first.
#   2. Streamlit auto-reload (code change) killed the daemon thread but
#      reparented the cycle subprocess to init — orphan keeps running,
#      next Streamlit boot's auto-revive spawns ANOTHER subprocess on
#      top.
#   3. Opening a NEW browser session never checked for orphans from a
#      previous session.
#
# Result on 2026-05-06: 3 concurrent agent processes each running their
# own 5-min cycle loop, each submitting on the same ticker because their
# threading.Lock instances couldn't see each other.
#
# Fix: ``_sweep_orphan_agents`` runs at every Start and at session
# boot.  It (a) terminates AGENT_PID's subprocess if alive, (b)
# pkill-sweeps any stragglers matching the agent.py command line, and
# (c) clears stale sentinels so the new agent starts from a clean slate.

def _pid_is_alive(pid: int) -> bool:
    """Return True iff ``pid`` is a live process owned by us.

    Uses ``os.kill(pid, 0)`` — sends no signal, just checks deliverability.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — still treat as alive
        # so we don't accidentally stack a duplicate on top.
        return True
    except Exception:                                       # noqa: BLE001
        return False


def _terminate_pid(pid: int, timeout: float = 5.0) -> bool:
    """SIGTERM, wait up to ``timeout`` seconds, SIGKILL if still alive.

    Returns True if the process is gone after the call.
    """
    import signal
    if not _pid_is_alive(pid):
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:                                       # noqa: BLE001
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_is_alive(pid):
            return True
        time.sleep(0.1)
    # Last resort.
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:                                       # noqa: BLE001
        pass
    time.sleep(0.2)
    return not _pid_is_alive(pid)


def _sweep_orphan_agents() -> Dict[str, Any]:
    """
    Hard-kill any orphan agent subprocesses and clear stale sentinels.

    Called at:
      * Start Agent button (so a stacked previous session can't survive)
      * Streamlit session boot (so a fresh browser tab starts clean)
      * Auto-revive (so we don't pile a new subprocess on an orphan)

    Sweep order
    -----------
    1. AGENT_PID file → SIGTERM (5s grace) → SIGKILL.  This is the most
       recent cycle subprocess if any.
    2. ``pkill -f agent.py`` for any straggler.  Matches command lines
       like ``python -m trading_agent.agent`` or ``python agent.py``.
       Filtered to only the user's own processes (-u $USER).
    3. Remove AGENT_RUNNING / AGENT_PID / PAUSE_FLAG sentinels so the
       next Start writes them fresh.

    Returns a dict suitable for surfacing in the dashboard banner:
      ``{"killed_pids": [int, ...], "swept": int, "errors": [str, ...]}``.
    """
    killed: List[int] = []
    errors: List[str] = []

    # Step 1: terminate the tracked AGENT_PID subprocess if alive.
    if AGENT_PID.exists():
        try:
            tracked_pid = int(AGENT_PID.read_text().strip())
        except (OSError, ValueError) as exc:
            errors.append(f"AGENT_PID unreadable: {exc}")
            tracked_pid = -1
        if tracked_pid > 0 and _pid_is_alive(tracked_pid):
            if _terminate_pid(tracked_pid):
                killed.append(tracked_pid)
            else:
                errors.append(f"failed to terminate PID {tracked_pid}")

    # Step 2: pkill-sweep any straggler agent.py processes.
    # We TWO-PHASE the kill:
    #   (a) SIGTERM via pkill — gives the cycle a chance to clean up
    #   (b) Verify via pgrep that no matches remain
    #   (c) If matches survive → SIGKILL via pkill -9
    # Without the verify+escalate step, a single pkill TERM that races
    # the cycle's signal handler can leave straggler processes alive.
    swept = 0

    # ``-u`` filters to the current user — but only if we can reliably
    # determine who that is.  USER env var is sometimes unset under
    # systemd/launchd/Streamlit's process tree; fall back to numeric
    # uid in that case.  If neither resolves, drop the -u filter
    # entirely (acceptable on a single-tenant dev box).
    user_arg: List[str] = []
    user = os.environ.get("USER", "").strip()
    if user:
        user_arg = ["-u", user]
    else:
        try:
            uid = os.getuid()                               # noqa: SLF001
            user_arg = ["-u", str(uid)]
        except AttributeError:                              # Windows
            user_arg = []

    pkill_match = "trading_agent.agent"
    pgrep_match = pkill_match

    def _pkill(signal_arg: Optional[str] = None) -> Optional[int]:
        """Returns pkill's exit code or None on FileNotFoundError."""
        cmd = ["pkill"] + ([signal_arg] if signal_arg else []) + user_arg + [
            "-f", pkill_match,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            return r.returncode
        except FileNotFoundError:
            return None

    def _pgrep_count() -> int:
        try:
            r = subprocess.run(
                ["pgrep"] + user_arg + ["-f", pgrep_match],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode in (0, 1):
                lines = [ln for ln in r.stdout.split("\n") if ln.strip().isdigit()]
                # Filter out our own Streamlit process so we don't
                # commit suicide if the Streamlit command line happens
                # to contain trading_agent.agent (it shouldn't, but
                # belt-and-braces).
                own_pid = os.getpid()
                return sum(1 for ln in lines if int(ln) != own_pid)
        except FileNotFoundError:
            return -1
        except Exception:                                   # noqa: BLE001
            return -1
        return 0

    # Phase (a) — SIGTERM first.
    rc = _pkill(None)
    if rc is None:
        errors.append("pkill not available — relying on AGENT_PID-tracked kill only")
    elif rc not in (0, 1):
        errors.append(f"pkill TERM exited {rc}")
    else:
        swept = 1 if rc == 0 else 0

    # Phase (b)+(c) — verify and escalate.
    if rc == 0:
        # We killed at least one — give them ~2s to exit cleanly,
        # then verify and SIGKILL if any survived.
        time.sleep(0.5)
        survivors = _pgrep_count()
        deadline = time.monotonic() + 2.0
        while survivors > 0 and time.monotonic() < deadline:
            time.sleep(0.2)
            survivors = _pgrep_count()
        if survivors > 0:
            kill_rc = _pkill("-9")
            if kill_rc is not None and kill_rc not in (0, 1):
                errors.append(f"pkill -9 exited {kill_rc}, {survivors} stragglers")
            time.sleep(0.3)
            final = _pgrep_count()
            if final > 0:
                errors.append(f"{final} agent.py processes survived even SIGKILL")

    # Step 3: clear stale sentinels so the new agent boots clean.
    for sentinel in (AGENT_RUNNING, AGENT_PID, PAUSE_FLAG):
        sentinel.unlink(missing_ok=True)

    return {"killed_pids": killed, "swept": swept, "errors": errors}


def _start_agent(dry_run: bool = False) -> None:
    """Write sentinels and launch the loop in a daemon thread.

    Always runs an orphan sweep first to enforce the process singleton
    invariant (see ``_sweep_orphan_agents``).  Without this, clicking
    Start while a previous Streamlit session's subprocess was still
    alive produced concurrent cycles that submitted duplicate orders.
    """
    sweep = _sweep_orphan_agents()
    if sweep["killed_pids"]:
        _append_log(
            f"[{_now()}] Killed orphan agent PID(s) "
            f"{sweep['killed_pids']} before Start"
        )
    if sweep["swept"]:
        _append_log(f"[{_now()}] pkill sweep removed straggler agent.py processes")
    if sweep["errors"]:
        _append_log(f"[{_now()}] Sweep errors: {sweep['errors']}")

    AGENT_RUNNING.write_text(_now())
    if dry_run:
        DRY_RUN_FLAG.write_text(_now())
    else:
        DRY_RUN_FLAG.unlink(missing_ok=True)
    t = threading.Thread(target=_agent_loop, daemon=True, name="agent-loop")
    t.start()
    st.session_state["_agent_thread"] = t


def _ensure_loop_alive_if_intended() -> bool:
    """Self-heal a zombie sentinel after a Streamlit auto-reload.

    The agent loop runs in a ``daemon=True`` thread inside the Streamlit
    process. When Streamlit's file watcher detects a code change it
    restarts the Python process — daemon threads die with the old
    process, but the on-disk ``AGENT_RUNNING`` sentinel survives. The
    new process then boots into a "looks running but isn't" state:
    the dashboard shows the Stop-Agent button and "ACTIVE [LIVE]"
    status because the sentinel exists, but no cycle subprocess ever
    spawns (the thread that would call ``_run_one_cycle`` is dead).

    This helper runs at the top of ``render_live_monitor`` and:
      1. If the sentinel says we should be running, AND
      2. No live thread is recorded in ``st.session_state``, OR the
         recorded thread is no longer alive,
      → restarts the loop in a fresh daemon thread, preserving the
        existing dry-run flag.

    Returns True iff the loop was just revived (lets the caller surface
    a one-shot info banner so the user knows what happened).
    """
    if not AGENT_RUNNING.exists():
        return False  # User has stopped — nothing to revive.

    existing = st.session_state.get("_agent_thread")
    if existing is not None and existing.is_alive():
        return False  # Already healthy — leave it alone.

    # Zombie state: sentinel says run, no live thread → revive.
    # CRITICAL: kill any orphan cycle subprocess from the prior process
    # incarnation BEFORE spawning a new daemon thread. Without this
    # sweep the Streamlit auto-reload path produced duplicate cycles
    # (orphan subprocess + new revived loop both running concurrently).
    sweep = _sweep_orphan_agents()
    if sweep["killed_pids"] or sweep["swept"]:
        _append_log(
            f"[{_now()}] Auto-revive: swept orphans before reviving "
            f"(killed={sweep['killed_pids']}, pkill_match={sweep['swept']})"
        )
    # The sweep cleared AGENT_RUNNING — re-write it so the loop knows
    # to keep looping.  DRY_RUN_FLAG isn't touched by the sweep.
    AGENT_RUNNING.write_text(_now())
    dry_run = DRY_RUN_FLAG.exists()
    t = threading.Thread(target=_agent_loop, daemon=True, name="agent-loop")
    t.start()
    st.session_state["_agent_thread"] = t
    _append_log(
        f"[{_now()}] Auto-revived agent loop after process restart "
        f"(dry_run={dry_run}). Sentinel was set but the thread had died."
    )
    return True


def _render_singleton_pill() -> str:
    """
    Return a Markdown suffix indicating how many agent processes are alive.

    Returns
    -------
    str
        ``""`` when pgrep is unavailable (e.g. Windows) — silent.
        ``" · ✓ 1 process"`` when exactly one cycle subprocess is alive.
        ``" · ⚠ N processes — orphans detected"`` when >1 (visible warning
        the operator should ▶ Stop and ▶ Start to trigger the sweep).
        ``""`` when zero processes (normal STOPPED state — no point
        showing "0 processes" in the pill since the banner already says
        STOPPED).

    Why a count, not just a boolean
    --------------------------------
    Pre-2026-05-06 the dashboard's "Stop" left orphan subprocesses
    alive and a casual operator had no signal that anything was wrong.
    The pill is the always-on visible verification — if the singleton
    invariant ever breaks again, the warning is right there in the
    status banner instead of buried in the agent log.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", "trading_agent.agent"],
            capture_output=True, text=True, timeout=2,
        )
    except (FileNotFoundError, Exception):                  # noqa: BLE001
        return ""
    if result.returncode not in (0, 1):
        return ""
    own_pid = os.getpid()
    pids = [int(ln) for ln in result.stdout.split("\n")
            if ln.strip().isdigit() and int(ln) != own_pid]
    n = len(pids)
    if n == 0:
        return ""
    if n == 1:
        return " · ✓ singleton (1 process)"
    return (
        f" · ⚠ **{n} processes detected** — click Stop then Start to "
        "re-enforce the singleton (orphan sweep)"
    )


def _sweep_orphans_at_session_boot() -> Optional[Dict[str, Any]]:
    """One-time sweep on Streamlit session boot — fresh-browser entrypoint.

    Called at the top of ``render_live_monitor`` exactly once per
    Streamlit session (gated by ``st.session_state``).  When a user
    opens a new browser tab to the dashboard, we want them to land on
    a clean, single-process state regardless of what was running
    before — even if a previous session left orphan subprocesses
    running because of a Streamlit auto-reload, an OS-level crash, or
    a manual ``pkill streamlit``.

    Behaviour
    ---------
    * If AGENT_RUNNING is set AND the recorded AGENT_PID is alive →
      respect operator intent, leave the running agent alone, return
      ``None``.  This is the common "user just reopened the dashboard"
      case.
    * If AGENT_RUNNING is set BUT the recorded AGENT_PID is dead (or
      missing) → treat as orphan state, sweep, return the sweep result
      so the caller can flash a banner.
    * If AGENT_RUNNING is not set → sweep stragglers anyway (defensive)
      and return the sweep result if anything was killed.

    Returns the sweep dict on action, or ``None`` if no action was taken.
    """
    if st.session_state.get("_session_boot_sweep_done"):
        return None
    st.session_state["_session_boot_sweep_done"] = True

    # CRITICAL: leave the agent alone if ANY of these signals indicate
    # it's healthy.  Pre-2026-05-06 this only checked AGENT_RUNNING +
    # AGENT_PID, which produced a destructive bug: between cycles,
    # AGENT_PID is *absent* (the cycle subprocess finished, the next
    # one is 5 minutes away), so the boot sweep saw "RUNNING but no
    # PID alive" → treated it as orphan → deleted AGENT_RUNNING →
    # the daemon-thread loop's ``while AGENT_RUNNING.exists():``
    # silently exited and the agent stopped without any user action.
    #
    # Health checks (any one is sufficient to skip the sweep):
    #   1. AGENT_PID is alive → cycle subprocess in flight, definitely
    #      healthy.
    #   2. The daemon-thread reference in session_state is alive →
    #      we're in a 5-min between-cycle gap, the loop will tick
    #      again on schedule.
    #   3. The agent log was modified in the last LOOP_HEALTH_WINDOW
    #      seconds — across-process signal so even after a Streamlit
    #      auto-reload (which kills the daemon thread + clears
    #      session_state), a recently-active log proves the loop was
    #      ticking.  This catches the auto-reload-during-cycle case.
    #
    # If any signal says "healthy", return None and DO NOT sweep.
    if AGENT_RUNNING.exists():
        # Signal 1: live cycle subprocess.
        if AGENT_PID.exists():
            try:
                pid = int(AGENT_PID.read_text().strip())
                if pid > 0 and _pid_is_alive(pid):
                    return None
            except (OSError, ValueError):
                pass
        # Signal 2: live daemon thread.
        existing_thread = st.session_state.get("_agent_thread")
        if existing_thread is not None and existing_thread.is_alive():
            return None
        # Signal 3: recently-active log file.  10 minutes covers two
        # 5-min cycle intervals so a single missed cycle (e.g. a long
        # data fetch) doesn't trip the sweep.
        try:
            log_path = Path("logs") / "trading_agent.log"
            if log_path.exists():
                age_sec = time.time() - log_path.stat().st_mtime
                if age_sec < 10 * 60:
                    return None
        except Exception:                                   # noqa: BLE001
            pass

    # No health signal → genuine orphan state.  Sweep, clear sentinels,
    # surface a banner so the operator knows what was cleaned up.
    sweep = _sweep_orphan_agents()
    if sweep["killed_pids"] or sweep["swept"]:
        _append_log(
            f"[{_now()}] Session boot: swept orphans "
            f"(killed={sweep['killed_pids']}, pkill_match={sweep['swept']})"
        )
        return sweep
    return None


def _stop_agent() -> None:
    """Remove sentinel AND terminate the in-flight cycle subprocess.

    Pre-2026-05-06 this only removed the sentinel and trusted the
    daemon-thread loop to exit "soon".  In practice a Stop was followed
    by an in-flight 18-second cycle continuing to run; if the user
    clicked Start in the meantime, two cycles overlapped.  Now we also
    SIGTERM the AGENT_PID-tracked subprocess so Stop is genuinely
    synchronous from the operator's POV — when the function returns,
    no more order submissions can happen for the previous run.
    """
    AGENT_RUNNING.unlink(missing_ok=True)
    _append_log(f"[{_now()}] Stop requested from dashboard")

    # Terminate the in-flight cycle subprocess (if any) so a quick
    # Stop→Start can't overlap.
    if AGENT_PID.exists():
        try:
            pid = int(AGENT_PID.read_text().strip())
        except (OSError, ValueError):
            pid = -1
        if pid > 0 and _pid_is_alive(pid):
            if _terminate_pid(pid, timeout=5):
                _append_log(f"[{_now()}] Terminated in-flight cycle PID {pid}")
            else:
                _append_log(f"[{_now()}] FAILED to terminate cycle PID {pid}")
        AGENT_PID.unlink(missing_ok=True)


def _kill_current_cycle() -> None:
    """SIGKILL the running cycle subprocess immediately (emergency only)."""
    if AGENT_PID.exists():
        try:
            pid = int(AGENT_PID.read_text().strip())
            os.kill(pid, 9)
            _append_log(f"[{_now()}] SIGKILL sent to PID {pid}")
        except Exception as exc:
            _append_log(f"[{_now()}] Kill failed: {exc}")
        AGENT_PID.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_config():
    try:
        from trading_agent.config import load_config
        return load_config()
    except Exception:
        return None


def _empty_journal_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["timestamp", "account_balance", "ticker", "action",
                 "regime", "mode", "checks_passed", "checks_failed",
                 "notes", "reason",
                 "rsi_14", "sma_50", "sma_200", "scan_results",
                 "raw_signal"]
    )


@st.cache_data(show_spinner=False)
def _parse_journal_df(path: str, version: int, mtime: float, size: int) -> pd.DataFrame:
    """
    Parse *path* into the dashboard's canonical DataFrame shape.

    The cache key is ``(path, version, mtime, size)``:
      • ``version`` is bumped by the watchdog observer on each modify/create,
        so unrelated Streamlit reruns hit the cache (zero I/O).
      • ``mtime`` + ``size`` are belt-and-suspenders for environments where
        watchdog isn't running (WATCHDOG_DISABLE=1, NFS without polling, etc.)
        — they still detect changes, just lazily on next call.
    The ``version`` arg is unused inside the function but is part of the
    cache signature; do not remove.
    """
    del version  # marker only — see docstring
    rows = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rs = rec.get("raw_signal", {}) or {}
                    action = rec.get("action", "") or ""
                    notes_raw = rec.get("notes", "") or ""
                    reason = (rs.get("reason") or "").strip()
                    # ``notes`` is supposed to be a human-readable
                    # description of what happened. The agent's
                    # ``log_signal`` for some action types (notably
                    # ``skipped_existing``) writes notes equal to the
                    # action label itself ("skipped_existing"), which is
                    # a useless duplicate — the meaningful explanation
                    # lives in ``raw_signal.reason`` (e.g. "Existing
                    # open position or pending order"). Promote the
                    # reason whenever notes is missing or just echoes
                    # the action label, so the journal table actually
                    # tells the operator WHY a cycle was skipped.
                    if reason and (
                        not notes_raw or notes_raw.strip() == action
                    ):
                        notes = reason
                    else:
                        notes = notes_raw
                    rows.append({
                        "timestamp":       pd.to_datetime(rec.get("timestamp")),
                        "account_balance": rs.get("account_balance", 0) or 0,
                        "ticker":          rec.get("ticker", ""),
                        "action":          action,
                        "regime":          rs.get("regime", "unknown"),
                        # Empty string on legacy rows that pre-date the mode
                        # field — _guardrail_status_from_journal treats those
                        # as LIVE so historical data isn't silently filtered out.
                        "mode":            rs.get("mode", "") or "",
                        "checks_passed":   rs.get("checks_passed") or [],
                        "checks_failed":   rs.get("checks_failed") or [],
                        "notes":           notes,
                        # Keep the raw reason as a separate column so the
                        # guardrail grid's SKIPPED branch can use it without
                        # re-parsing the JSON or guessing whether ``notes``
                        # was promoted. Empty string on rows that don't have
                        # a reason field (most rejected / submitted rows).
                        "reason":          reason,
                        "rsi_14":          rs.get("rsi_14", 0) or 0,
                        "sma_50":          rs.get("sma_50", 0) or 0,
                        "sma_200":         rs.get("sma_200", 0) or 0,
                        # Adaptive-scanner block. Empty dict on legacy
                        # records so downstream rendering can do truthiness
                        # checks without KeyError; populated dict on new
                        # records.
                        "scan_results":    rs.get("scan_results") or {},
                        # Full raw_signal payload preserved as a column
                        # (added 2026-05-06) so consumers like the Open
                        # Positions "Why" lookup and the "Closed Today"
                        # expander can read fields the parser doesn't
                        # explicitly project (expiration, thesis,
                        # credit_to_width_ratio, exit_signal, P&L, etc).
                        # Pre-2026-05-06 the parser only kept a handful
                        # of explicit columns so any new consumer was
                        # silently reading None and producing blank
                        # cells / all-zero panels.
                        "raw_signal":      rs,
                    })
                except (json.JSONDecodeError, KeyError):
                    continue
    except FileNotFoundError:
        return _empty_journal_df()
    del mtime, size  # silence linters — already captured in cache key

    if not rows:
        return _empty_journal_df()
    df = pd.DataFrame(rows)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _load_journal_df() -> pd.DataFrame:
    """
    Resolve the active journal path (live → legacy fallback), register
    it with the watchdog observer, and return the parsed DataFrame from
    the version-keyed cache.
    """
    # Prefer the new live-only file, fall back to legacy ``signals.jsonl``
    # so the dashboard keeps surfacing pre-rename history. Once the agent
    # has written one cycle to the new path, this naturally converges.
    if JOURNAL_PATH.exists():
        path = JOURNAL_PATH
    elif LEGACY_JOURNAL_PATH.exists():
        path = LEGACY_JOURNAL_PATH
    else:
        return _empty_journal_df()

    # Lazy watchdog import — module-level import would force watchdog
    # to be installed even for non-Streamlit users of this file.
    try:
        from trading_agent.streamlit import file_watcher
        version = file_watcher.watch(path)
    except Exception:
        version = 0   # graceful degrade: cache key falls back to mtime+size

    try:
        stat = os.stat(path)
        mtime, size = stat.st_mtime, stat.st_size
    except FileNotFoundError:
        return _empty_journal_df()

    return _parse_journal_df(str(path), version, mtime, size)


def _scanner_diagnostics_from_journal(
    df: pd.DataFrame, lookback_rows: int = 200,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    """
    Project scanner output from ``signals.jsonl`` into three render-ready
    pieces: per-ticker latest verdict, aggregate reject-reason histogram,
    and per-side candidate counts.

    The latest-per-ticker frame answers *"what did the scanner find on the
    most recent cycle for SPY/QQQ/...?"* — one row per ticker with the
    near-miss credit/EV gap. The histogram answers *"across the last N
    cycles, which filter is rejecting most candidates?"* — telling you
    whether to lower edge_buffer (cw_below_floor dominates) or widen the
    Δ grid (pop_below_min dominates).
    """
    cols = ["ticker", "side", "candidates", "selected", "edge_buffer",
            "min_pop", "grid_total", "grid_priced", "top_reject",
            "top_reject_count", "near_miss_credit", "near_miss_cw",
            "near_miss_floor", "near_miss_ev", "timestamp"]
    empty = pd.DataFrame(columns=cols)
    if df.empty or "scan_results" not in df.columns:
        return empty, {}, {}

    # Restrict to records that actually carry scan_results — older
    # legacy records will silently fall out.
    has_scan = df[df["scan_results"].apply(lambda x: isinstance(x, dict) and bool(x))]
    if has_scan.empty:
        return empty, {}, {}
    has_scan = has_scan.tail(lookback_rows)

    latest_rows: Dict[str, Dict] = {}        # ticker → latest row dict
    reject_hist: Dict[str, int] = {}         # reason → count across window
    side_counts: Dict[str, int] = {}         # side  → candidates_total sum

    for _, rec in has_scan.iterrows():
        sr = rec["scan_results"] or {}
        diag = sr.get("diagnostics") or {}
        side = sr.get("side") or "—"
        side_counts[side] = side_counts.get(side, 0) + int(sr.get("candidates_total") or 0)

        for reason, count in (diag.get("rejects_by_reason") or {}).items():
            reject_hist[reason] = reject_hist.get(reason, 0) + int(count)

        nm = diag.get("best_near_miss") or {}
        rejects = diag.get("rejects_by_reason") or {}
        if rejects:
            top_r, top_c = max(rejects.items(), key=lambda kv: kv[1])
        else:
            top_r, top_c = "—", 0
        latest_rows[rec["ticker"]] = {
            "ticker":           rec["ticker"],
            "side":             side,
            "candidates":       int(sr.get("candidates_total") or 0),
            "selected":         int(sr.get("selected_index") or -1),
            "edge_buffer":      float(sr.get("edge_buffer") or 0.0),
            "min_pop":          float(sr.get("min_pop") or 0.0),
            "grid_total":       int(diag.get("grid_points_total") or 0),
            "grid_priced":      int(diag.get("grid_points_priced") or 0),
            "top_reject":       top_r,
            "top_reject_count": top_c,
            "near_miss_credit": float(nm.get("credit") or 0.0) if nm else 0.0,
            "near_miss_cw":     float(nm.get("cw_ratio") or 0.0) if nm else 0.0,
            "near_miss_floor":  float(nm.get("cw_floor") or 0.0) if nm else 0.0,
            "near_miss_ev":     float(nm.get("ev") or 0.0) if nm else 0.0,
            "timestamp":        rec["timestamp"],
        }

    latest_df = pd.DataFrame(list(latest_rows.values()), columns=cols)
    if not latest_df.empty:
        latest_df.sort_values("timestamp", ascending=False, inplace=True)
        latest_df.reset_index(drop=True, inplace=True)
    return latest_df, reject_hist, side_counts


def _render_closed_today(journal_df: pd.DataFrame) -> None:
    """
    Render the "Closed Today" expander above Open Positions.

    Source of truth: journal rows where ``action == "closed"`` and the
    timestamp is today (UTC).  These are written by
    ``agent.py:_journal_close_event`` after every COMPLETE close (all
    legs cancelled by Alpaca) — strike_proximity / profit_target /
    hard_stop / regime_shift / dte_safety / expired.

    Pre-2026-05-06 the same row tag was used for partial fills too,
    which made this tile lie when SPY hit a partial-fill loop and
    produced 11 rows for ONE position.  Partial fills now go to
    ``_render_close_failures_today`` under a separate
    ``action="close_failed"`` tag.

    Behavior
    --------
    * If no complete-close events today → hidden (no empty section).
    * If ≥1 → collapsible expander with a per-row table:
        Time · Ticker · Strategy · Exit Signal · P&L · Reason · Fill Status
      Plus a footer with the running net P&L summed across closes.
    """
    if journal_df is None or journal_df.empty:
        return
    if "action" not in journal_df.columns:
        return

    closed = journal_df[journal_df["action"] == "closed"].copy()
    if closed.empty:
        return

    # Filter to today (UTC) so a long-running agent doesn't stack
    # weeks of closes into one panel.
    today_utc = pd.Timestamp.utcnow().normalize()
    closed = closed[pd.to_datetime(closed["timestamp"], utc=True)
                    >= today_utc]
    if closed.empty:
        return

    closed = closed.sort_values("timestamp", ascending=False)

    rows = []
    net_pl = 0.0
    for _, r in closed.iterrows():
        rs = r.get("raw_signal") or {}
        if not isinstance(rs, dict):
            rs = {}
        pl = float(rs.get("net_unrealized_pl") or 0.0)
        net_pl += pl
        rows.append({
            "Time":         pd.Timestamp(r["timestamp"]).strftime("%H:%M:%S"),
            "Ticker":       r.get("ticker", ""),
            "Strategy":     rs.get("strategy", ""),
            "Exit Signal":  rs.get("exit_signal", ""),
            "P&L ($)":      f"{'+' if pl >= 0 else ''}{pl:.2f}",
            "Reason":       (rs.get("exit_reason") or "")[:80],
            "Fill":         rs.get("fill_status", ""),
        })

    label = (f"🚪 Closed Today ({len(rows)} exit"
             f"{'s' if len(rows) != 1 else ''}, net P&L "
             f"{'+' if net_pl >= 0 else ''}${net_pl:,.2f})")
    # Collapsed by default — Open Positions is the primary anchor; the
    # closes panel is a click-to-audit secondary view.
    with st.expander(label, expanded=False):
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        if net_pl != 0:
            st.caption(
                f"Running total: {len(rows)} close{'s' if len(rows) != 1 else ''} "
                f"today · net P&L "
                f"{'+' if net_pl >= 0 else ''}${net_pl:,.2f}"
            )


def _render_close_failures_today(journal_df: pd.DataFrame) -> None:
    """
    Render the "Close Failures Today" expander above Open Positions.

    Source of truth: journal rows where ``action == "close_failed"``
    and the timestamp is today (UTC).  These rows are written by
    ``agent.py:_journal_close_event`` when ``executor.close_spread``
    returns ``fill_status="partial"`` — typically because Alpaca
    rejected one or more legs with codes like:

      * ``40310000`` "account not eligible to trade uncovered option contracts"
      * ``40310100`` "trade denied due to pattern day trading protection"
      * Insufficient buying power on a small (sub-$25K) account.

    These rows mean **the position is still open on the broker**.
    Repeated close_failed entries for the same ticker indicate a
    "zombie" partial-fill state that the agent will park into a
    cooldown after PARTIAL_CLOSE_COOLDOWN_THRESHOLD attempts.

    Behavior
    --------
    * If no close failures today → hidden (no empty section).
    * If ≥1 → collapsible expander with the most recent attempt per
      ticker, plus a "retry count" so the operator knows when the
      cooldown will engage.  The full retry stream is in the journal
      tab below.
    """
    if journal_df is None or journal_df.empty:
        return
    if "action" not in journal_df.columns:
        return

    failures = journal_df[journal_df["action"] == "close_failed"].copy()
    if failures.empty:
        return

    today_utc = pd.Timestamp.utcnow().normalize()
    failures = failures[pd.to_datetime(failures["timestamp"], utc=True)
                        >= today_utc]
    if failures.empty:
        return

    failures = failures.sort_values("timestamp", ascending=False)

    # Group by ticker for compactness — show count + the most recent
    # attempt's reason.  The full per-attempt stream is still
    # available in the Recent Journal Entries panel.
    #
    # Cooldown surfacing
    # ------------------
    # When the most-recent ``close_failed`` row carries
    # ``close_cooldown_until``, the agent has parked the ticker into
    # the manual-intervention cooldown window (3 partial fills +
    # 60-min lockout).  Surface the deadline in the table column AND
    # render a warning banner above so the operator can't miss it.
    # Tickers in cooldown are flagged with 🚨; pre-cooldown rows
    # show the streak (e.g. "2/3") so the operator can see they're
    # one partial fill away from the lockout.
    rows = []
    cooldown_warnings: List[Tuple[str, str]] = []  # (ticker, until-ISO)
    now_utc = pd.Timestamp.utcnow()
    for ticker, group in failures.groupby("ticker", sort=False):
        latest = group.iloc[0]
        rs = latest.get("raw_signal") or {}
        if not isinstance(rs, dict):
            rs = {}
        leg_results = rs.get("leg_close_results") or []
        failed_legs = [
            lr.get("symbol", "?")
            for lr in leg_results
            if isinstance(lr, dict) and lr.get("status") != "closed"
        ]
        # Streak / threshold from the agent's bookkeeping.
        streak = rs.get("partial_close_streak")
        threshold = rs.get("partial_close_threshold")
        cooldown_until_str = rs.get("close_cooldown_until")
        cooldown_until = None
        if cooldown_until_str:
            try:
                cooldown_until = pd.to_datetime(cooldown_until_str, utc=True)
            except Exception:                                    # noqa: BLE001
                cooldown_until = None
        cooldown_active = (
            cooldown_until is not None and cooldown_until > now_utc
        )
        if cooldown_active:
            cooldown_warnings.append(
                (ticker, cooldown_until.strftime("%H:%M:%S UTC"))
            )

        # Build the streak cell — "🚨 cooldown until …" if locked,
        # "2/3" if pre-lockout, "—" if the producer doesn't carry the
        # field (legacy row before this surface was added).
        if cooldown_active:
            streak_cell = (
                f"🚨 cooldown until "
                f"{cooldown_until.strftime('%H:%M:%S UTC')}"
            )
        elif isinstance(streak, (int, float)) and isinstance(threshold, (int, float)):
            streak_cell = f"{int(streak)}/{int(threshold)}"
        else:
            streak_cell = "—"

        rows.append({
            "Last Attempt": pd.Timestamp(latest["timestamp"]).strftime("%H:%M:%S"),
            "Ticker":       ticker,
            "Strategy":     rs.get("strategy", ""),
            "Exit Signal":  rs.get("exit_signal", ""),
            "Retries":      int(len(group)),
            "Streak":       streak_cell,
            "Failed Legs":  ", ".join(failed_legs)[:80] if failed_legs else "—",
            "Reason":       (rs.get("exit_reason") or "")[:80],
        })

    label = (f"⚠️ Close Failures Today ({len(rows)} ticker"
             f"{'s' if len(rows) != 1 else ''}, "
             f"{int(len(failures))} attempt"
             f"{'s' if len(failures) != 1 else ''}"
             f"{', 🚨 ' + str(len(cooldown_warnings)) + ' in cooldown' if cooldown_warnings else ''})")
    # Auto-expand if any ticker is in active cooldown — that's a
    # manual-intervention signal that shouldn't be hidden behind a
    # click.  Otherwise stay collapsed by default (symmetric with
    # "Closed Today").
    with st.expander(label, expanded=bool(cooldown_warnings)):
        # Bright warning banner per cooldown'd ticker — operator should
        # see this even if they don't read the table.
        for ticker, until_str in cooldown_warnings:
            st.error(
                f"🚨 **{ticker} is in 60-min auto-close cooldown until "
                f"{until_str}.** The position is still open on the "
                "broker. The executor will NOT retry until the "
                "deadline passes. Manually flat the position on the "
                "Alpaca UI to clear the zombie state — naive retry is "
                "hopeless because Alpaca's reason (PDT / uncovered / "
                "insufficient buying power) doesn't lift on its own."
            )
        st.caption(
            "These positions are **still open** on the broker — the "
            "executor's DELETE was rejected (typically PDT, uncovered, "
            "or insufficient buying power).  The agent parks each "
            "ticker into a 60-min cooldown after 3 consecutive partial "
            "fills; the **Streak** column shows progress toward the "
            "lockout.  Until then, manually flat the position on "
            "Alpaca's UI to clear the zombie state."
        )
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


def _render_cycle_staleness_beacon(
    journal_df: pd.DataFrame, loop_running: bool,
) -> None:
    """
    Render an at-a-glance health check based on the journal's most
    recent entry age.

    Why this exists
    ---------------
    A silently-dead agent process looks identical to a healthy one
    between 5-min cycles.  The orphan sweep catches a process that
    crashed AND failed to clean up its sentinel — but a process that
    simply hung in the middle of an LLM call or a Schwab fetch keeps
    the sentinel alive while making no progress.  The journal is the
    single source of truth for "the cycle actually finished" because
    every cycle writes at LEAST one row (skipped, submitted, rejected,
    error — they all journal).  Therefore:

      * journal mtime newer than 5 min  → healthy cycle cadence
      * 5-10 min old                    → late, surface a warning
      * 10+ min old                     → almost certainly stalled,
                                           surface an error

    Suppression
    -----------
    * ``loop_running=False``: operator stopped the agent on purpose;
      a stale journal is expected and rendering a warning would
      generate alarm fatigue.
    * Empty journal: brand-new install before any cycle has fired.

    UI shape
    --------
    A single ``st.warning`` / ``st.error`` line (no expander) sitting
    below the metric row so the operator can't miss it.  Carries the
    age in minutes + the wall-clock timestamp of the last entry +
    a "check logs" hint.
    """
    if not loop_running:
        return
    if journal_df is None or journal_df.empty:
        return
    if "timestamp" not in journal_df.columns:
        return

    try:
        latest_ts = pd.to_datetime(journal_df["timestamp"], utc=True).max()
        if pd.isna(latest_ts):
            return
        now = pd.Timestamp.utcnow()
        age_sec = (now - latest_ts).total_seconds()
    except Exception:                                            # noqa: BLE001
        return

    age_min = age_sec / 60.0
    when = latest_ts.strftime("%H:%M:%S UTC")

    # Thresholds match the comment above.  Tighter than the 270s
    # cycle hard-guard so a single hung cycle still surfaces — we
    # don't want to wait until the second hang before warning.
    if age_min >= 10.0:
        st.error(
            f"🛑 **Agent appears stalled — last journal entry was "
            f"{age_min:.1f} min ago** (at {when}). The 5-min cycle "
            "should have fired by now. Check `logs/trading_agent.log` "
            "for the last cycle's output, then **Stop** and **Start** "
            "the agent to recover. The orphan sweep on Start will "
            "kill any hung subprocess."
        )
    elif age_min >= 5.0:
        st.warning(
            f"⏰ Last journal entry was {age_min:.1f} min ago "
            f"(at {when}). One cycle may have skipped — keep an eye "
            "on the next 5-min mark. If it doesn't recover, restart."
        )


def _render_scanner_diagnostics_panel(journal_df: pd.DataFrame) -> None:
    """
    Render the "Adaptive Scanner Diagnostics" expander. No-op-friendly:
    when no records carry ``scan_results`` the panel renders an info
    message instead of disappearing — so users running an older agent
    binary aren't left wondering why the panel is empty.
    """
    with st.expander("🔬 Adaptive Scanner Diagnostics", expanded=False):
        if journal_df.empty:
            st.info("No journal entries yet.")
            return

        latest_df, reject_hist, side_counts = _scanner_diagnostics_from_journal(
            journal_df, lookback_rows=200,
        )
        if latest_df.empty:
            st.info(
                "No `scan_results` in the recent journal. The agent may be "
                "running in static-mode (set `SCAN_MODE=adaptive` in the "
                "Strategy Profile to enable the scanner) or you haven't "
                "completed a cycle since upgrading to the diagnostics build."
            )
            return

        # ── Top: per-ticker latest verdict ─────────────────────────────
        st.caption(
            "Latest scanner verdict per ticker (most recent cycle). "
            "**Near-miss** = the highest-EV candidate that *only* failed "
            "the C/W floor — closing the gap between **near_miss_cw** and "
            "**near_miss_floor** is the lever (lower `EDGE_BUFFER`, or wait)."
        )
        # Format floats compactly without copying the full df.
        display = latest_df.copy()
        for col, fmt in (
            ("edge_buffer",      "{:.2%}"),
            ("min_pop",          "{:.2%}"),
            ("near_miss_cw",     "{:.4f}"),
            ("near_miss_floor",  "{:.4f}"),
            ("near_miss_ev",     "{:+.4f}"),
            ("near_miss_credit", "${:.2f}"),
        ):
            display[col] = display[col].apply(
                lambda v, f=fmt: (f.format(v) if v else "—")
            )
        # Drop the timestamp column from the visible table (keeps the row
        # narrow); we display recency implicitly via sort order.
        visible_cols = [c for c in latest_df.columns if c != "timestamp"]
        st.dataframe(
            display[visible_cols],
            width='stretch',
            hide_index=True,
        )

        # ── Bottom: aggregate reject histogram + side counts ──────────
        col_h, col_s = st.columns([3, 1])
        with col_h:
            st.caption("Reject reasons across the last 200 cycles "
                       "(higher = more candidates rejected by that filter).")
            if reject_hist:
                hist_df = pd.DataFrame(
                    sorted(reject_hist.items(), key=lambda kv: -kv[1]),
                    columns=["reason", "count"],
                )
                st.bar_chart(hist_df.set_index("reason"))
            else:
                st.caption("(No rejects recorded — every cycle passed.)")
        with col_s:
            st.caption("Total candidates by side (last 200 cycles).")
            if side_counts:
                side_df = pd.DataFrame(
                    sorted(side_counts.items(), key=lambda kv: -kv[1]),
                    columns=["side", "candidates"],
                )
                st.dataframe(side_df, width='stretch', hide_index=True)
            else:
                st.caption("—")


def _guardrail_grid_from_journal(
    df: pd.DataFrame,
    *,
    current_mode: Optional[str] = None,
    window_minutes: int = 10,
    held_tickers: Optional[set] = None,
) -> List[Dict]:
    """One row per ticker for the most recent cycle.

    The "most recent cycle" is defined as every journal row within
    ``window_minutes`` of the latest timestamp (after filtering by
    ``current_mode``).  We then keep the last row per ticker so a
    ticker that was journaled twice in the same cycle (e.g. once as
    ``rejected`` then once on a retry) shows the most recent verdict.

    Each result dict has the shape::

        {
          "ticker":    str,
          "timestamp": pd.Timestamp,
          "regime":    str,
          "approved":  bool,
          "cells":     List[{"name": str, "state": str, "detail": str}],
        }

    where ``state`` is one of ``"ok" | "warn" | "fail"`` — the same
    contract as :func:`guardrail_cards` so the renderer can colour
    cells uniformly.

    The grid is ordered alphabetically by ticker for stable presentation
    across cycles (sorting by approved/state would make the table jump
    around as guardrail outcomes flip from cycle to cycle).
    """
    if df.empty:
        return []
    if current_mode is not None and "mode" in df.columns:
        normalised = df["mode"].fillna("").replace("", "LIVE")
        df = df[normalised == current_mode]
        if df.empty:
            return []

    # Drop orphan event-only rows (cycle_error / after_hours_shutdown
    # entries) before computing the latest cycle timestamp. Those rows
    # have no ticker/action/checks_passed and otherwise drag max_ts
    # forward — producing a phantom "blank-ticker ✅APPROVED" row in
    # the grid because drop_duplicates("ticker") collapses all the
    # null-ticker rows into one and the cells default to ok when both
    # checks_passed and checks_failed are empty.
    if "ticker" in df.columns:
        df = df[df["ticker"].notna() & (df["ticker"].astype(str).str.strip() != "")]
        if df.empty:
            return []

    df = df.sort_values("timestamp")
    max_ts = df["timestamp"].max()
    cutoff = max_ts - pd.Timedelta(minutes=window_minutes)
    latest = df[df["timestamp"] >= cutoff]
    latest = latest.drop_duplicates("ticker", keep="last")

    # Build a per-ticker lookup of "the most recent submitted entry
    # trade" from the FULL journal (not just the latest cycle window).
    # Used to substitute meaningful entry-trade details into rows where
    # the latest action was ``skipped_existing`` — the user wants to
    # see what was true *when the open position was opened*, not a
    # blank "we skipped this cycle because the position exists".
    last_entry_by_ticker: Dict[str, pd.Series] = {}
    if "action" in df.columns:
        # An "entry trade" is a row whose action contains "submitted"
        # (the agent emits ``action="submitted"`` after a successful
        # Alpaca POST /v2/orders). Fall back to "approved" rows if no
        # submitted exists, which catches dry-run cycles where the
        # risk manager signed off but no real order was placed.
        candidate_actions = ["submitted", "approved"]
        entries = df[df["action"].isin(candidate_actions)]
        if not entries.empty:
            for ticker, grp in entries.groupby("ticker"):
                last_entry_by_ticker[ticker] = grp.sort_values(
                    "timestamp"
                ).iloc[-1]

    rows: List[Dict] = []
    for _, r in latest.iterrows():
        action = (r.get("action") or "").lower()
        ticker = r["ticker"]

        # Status taxonomy — derived from the journal's ``action`` field,
        # which is the single source of truth for the END outcome of
        # the cycle. We deliberately do NOT infer status from
        # ``checks_failed`` alone, because a trade can be:
        #
        #   * approved by the risk manager (checks_failed = []) AND
        #     subsequently rejected by the executor downstream
        #     (live-credit recheck failure, qty=0 sizing, fill error,
        #     Alpaca POST error). In that case the journal logs
        #     ``action="rejected"`` even though every individual
        #     guardrail passed. Treating those as APPROVED would
        #     blatantly contradict the journal.
        #
        # Status values:
        #   "approved" — action="submitted" (risk approved AND order placed)
        #   "rejected" — action="rejected"  (regardless of checks_failed
        #                content — could be pre- OR post-approval)
        #   "holding"  — action starts with "skipped" AND the ticker is
        #                actually in held_tickers AND there's an entry
        #                trade in the journal to substitute checks from
        #   "skipped"  — action starts with "skipped" but one of the
        #                holding preconditions is missing
        #   (legacy)   — falls back to checks-based heuristic for any
        #                action label we don't recognise
        if action.startswith("skipped"):
            entry = last_entry_by_ticker.get(ticker)
            # Three sub-states for skipped rows, distinguished by
            # whether an entry trade exists in the journal AND whether
            # the broker is currently holding the position:
            #
            #   1. holding  — entry trade exists AND ticker is in
            #                 held_tickers → position is filled and live
            #   2. pending  — entry trade exists but ticker is NOT held
            #                 → limit order is on the book waiting to fill
            #                 (or stale-order canceller hasn't run yet)
            #   3. skipped  — no entry trade on record → truly nothing
            #                 to display, fall back to em-dash cells
            #
            # ``held_tickers=None`` (caller didn't pass a set) is treated
            # as "trust the entry trade" — preserves backward compat for
            # callers that haven't been updated. Production callers
            # always pass the set, so the holding/pending distinction
            # always fires correctly in the dashboard.
            ticker_is_actually_held = (
                held_tickers is None or ticker in held_tickers
            )
            if entry is not None:
                # Substitute cells from the entry trade in BOTH holding
                # and pending — operator wants the same detail either
                # way; only the verdict label and palette change.
                src = entry
                passed_list: List[str] = src.get("checks_passed") or []
                failed_list: List[str] = src.get("checks_failed") or []
                display_timestamp = src["timestamp"]
                display_regime = src.get("regime") or ""
                if ticker_is_actually_held:
                    status = "holding"
                else:
                    status = "pending"
            else:
                # No entry trade — truly nothing to substitute. Fall
                # back to em-dash cells with the journal's reason
                # surfaced as a tooltip.
                status = "skipped"
                passed_list = []
                failed_list = []
                display_timestamp = r["timestamp"]
                display_regime = r.get("regime") or ""
        else:
            passed_list = r.get("checks_passed") or []
            failed_list = r.get("checks_failed") or []
            display_timestamp = r["timestamp"]
            display_regime = r.get("regime") or ""
            # Trust the action field — this is what the agent ACTUALLY
            # decided, after all gates including any post-approval
            # executor-side rejections. ``submitted`` means risk manager
            # approved AND the order was successfully placed at Alpaca.
            # ``rejected`` means SOMETHING blocked the trade — we don't
            # claim approved just because the risk-manager checks
            # happen to all pass. The cells below still render based
            # on the per-check passed/failed strings (so individual
            # cells can show ✅ where checks did pass), but the overall
            # verdict cell honours the journal's action.
            if action == "submitted":
                status = "approved"
            elif action == "rejected":
                status = "rejected"
            elif action == "closed":
                # Position-monitor exit fired (strike_proximity / profit_target
                # / hard_stop / regime_shift / dte_safety / expired). The grid
                # row carries the exit-signal label as the verdict and the
                # P&L + reason as the per-cell tooltip.  See
                # ``agent.py:_journal_close_event`` for the schema.
                status = "closed"
            elif action == "close_failed":
                # Executor's DELETE was rejected by Alpaca (PDT,
                # uncovered, insufficient buying power) — the position
                # is still open on the broker.  Surface this as a
                # distinct verdict so it doesn't get conflated with a
                # successful close OR a regular approval.  See the
                # close-failed action in agent.py:_journal_close_event.
                status = "close_failed"
            else:
                # Legacy / unknown action label — fall back to the
                # checks-based heuristic for backward compatibility.
                status = "rejected" if failed_list else "approved"

        # When the row is in SKIPPED status, every cell gets the same
        # neutral em-dash detail (the risk manager didn't run, so we
        # have nothing factual to claim per-guardrail). The detail
        # string surfaces the ACTUAL reason from the journal row's
        # ``reason`` column whenever it's populated — typically
        # "Existing open position or pending order" for skipped_existing
        # rows, or whatever the agent's gate logged for skipped_bias /
        # skipped_defense_first / skipped_rsi_gate. Fallback to a
        # generic message preserves behaviour on legacy rows that
        # pre-date the parser's notes/reason promotion.
        skip_detail: str = "Skipped — agent short-circuited before risk check"
        if status == "skipped":
            j_reason = (r.get("reason") or "").strip() if hasattr(r, "get") else ""
            if not j_reason:
                # Fallback to notes (which may have been promoted by
                # _parse_journal_df when reason was empty in the source).
                j_reason = (r.get("notes") or "").strip() if hasattr(r, "get") else ""
            if j_reason:
                skip_detail = f"Skipped — {j_reason}"

        cells: List[Dict] = []
        for idx, keywords in enumerate(_GUARDRAIL_KEYWORDS):
            state = "ok"
            detail = "—"
            for fcheck in failed_list:
                if any(kw in fcheck.lower() for kw in keywords):
                    state = "fail"
                    detail = fcheck[:80]
                    break
            if state == "ok":
                for pcheck in passed_list:
                    if any(kw in pcheck.lower() for kw in keywords):
                        detail = pcheck[:80]
                        break
            # Same FORCED contract as the legacy 8-card panel — see
            # risk_manager.py:155 and _guardrail_status_from_journal.
            if state == "ok" and "FORCED" in detail:
                state = "warn"
            # SKIPPED rows: replace state + detail with the row-level
            # skip context. Every cell gets the same neutral em-dash on
            # grey + the same hover tooltip with the actual reason.
            # We only enter this branch when there's no entry trade to
            # substitute from — the risk manager truly didn't run, so
            # per-guardrail claims would be misleading.
            #
            # HOLDING and PENDING rows already had their cells
            # populated from the entry trade's checks_passed list above
            # (see the ``passed_list = src.get(...)`` substitution in
            # the skipped-action branch); the cells render identically
            # to APPROVED. Only the verdict cell + palette change to
            # communicate filled vs unfilled vs skipped.
            if status == "skipped":
                state = "skipped"
                detail = skip_detail
            name = GUARDRAIL_NAMES[idx]
            cells.append({
                "name":    name,
                "state":   state,
                "detail":  detail,
                # ``summary`` is the value-bearing chip rendered inside
                # the cell next to the emoji.  ``detail`` is still the
                # full string, surfaced via the cell hover.
                "summary": _compact_for(name, state, detail),
            })
        rows.append({
            "ticker":    ticker,
            "timestamp": display_timestamp,
            "regime":    display_regime,
            # Backward-compat: legacy callers expect a boolean
            # ``approved`` field. APPROVED, HOLDING, and PENDING all
            # represent positions where the risk manager signed off —
            # the only difference is whether the order has filled
            # (HOLDING) or is still on the book (PENDING). SKIPPED
            # (no entry-trade history) and REJECTED count as not-approved.
            "approved":  status in ("approved", "holding", "pending"),
            "status":    status,
            #            ^^^ "approved" | "rejected" | "holding"
            #                | "pending" | "skipped"
            "cells":     cells,
        })
    rows.sort(key=lambda r: r["ticker"])
    return rows


def _guardrail_status_from_journal(
    df: pd.DataFrame,
    *,
    current_mode: Optional[str] = None,
) -> List[Dict]:
    """Project the latest journal row's checks into 8 guardrail cards.

    ``current_mode`` (``"LIVE"`` or ``"DRY-RUN"``) filters the journal so
    cross-mode verdicts don't bleed through after the operator switches
    modes.  When None we fall back to the last row regardless of mode
    (preserves behaviour for callers that don't yet know the active
    mode).  Rows from before this column was introduced have
    ``mode == ""`` and are treated as LIVE for filtering.

    Each result dict carries ``state`` ∈ {"ok","warn","fail"}.  The
    dry-run forced-market-open pass is mapped to ``"warn"`` via the
    "FORCED" substring contract written by ``risk_manager.py:155``.
    """
    defaults = [
        {"name": n, "passed": True, "state": "ok", "detail": "No data"}
        for n in GUARDRAIL_NAMES
    ]
    if df.empty:
        return defaults

    if current_mode is not None and "mode" in df.columns:
        normalised = df["mode"].fillna("").replace("", "LIVE")
        scoped = df[normalised == current_mode]
        if scoped.empty:
            return defaults
        last = scoped.iloc[-1]
    else:
        last = df.iloc[-1]

    passed_list: List[str] = last.get("checks_passed") or []
    failed_list: List[str] = last.get("checks_failed") or []

    results = []
    for idx, keywords in enumerate(_GUARDRAIL_KEYWORDS):
        passed = True
        detail = "OK"
        for fcheck in failed_list:
            if any(kw in fcheck.lower() for kw in keywords):
                passed = False
                detail = fcheck[:70]
                break
        if passed:
            for pcheck in passed_list:
                if any(kw in pcheck.lower() for kw in keywords):
                    detail = pcheck[:70]
                    break
        # "Passed but synthetic" — keyed off the FORCED substring written
        # by risk_manager when force_market_open=True overrode a closed
        # market.  Card flips amber so the operator sees that the OK is
        # courtesy of the dry-run override, not a true open market.
        if passed and "FORCED" in detail:
            state = "warn"
        else:
            state = "ok" if passed else "fail"
        results.append({
            "name": GUARDRAIL_NAMES[idx],
            "passed": passed,
            "state": state,
            "detail": detail,
        })
    return results


# TTL for broker-state caching. Account + position data only changes when
# orders fill, which is bounded by the agent's 5-min cycle — so 30 s is
# fresh enough for the dashboard while cutting Alpaca API calls and the
# associated "Fetched N positions" log spam by ~10× (vs the 3 s fragment
# refresh interval). Override with BROKER_STATE_TTL_SECS env var.
BROKER_STATE_TTL_SECS = int(os.environ.get("BROKER_STATE_TTL_SECS", "30"))


@st.cache_resource
def _broker_fetch_marker() -> Dict:
    """Process-wide flag that records "the dashboard has fetched broker
    state at least once since this Streamlit process started".

    Lives in ``@st.cache_resource`` (NOT ``@st.cache_data`` and NOT
    ``st.session_state``) on purpose — those two scopes either die on
    Streamlit's ``cache_data.clear()`` or on browser reload (new
    session). ``cache_resource`` is process-scoped and survives both.

    Once set, ``render_live_monitor`` routes every render through the
    cached broker-fetch path. The ``ttl=30s`` on those cache_data
    decorators absorbs fragment ticks (cache hit = microseconds, no
    Alpaca call), so we still poll Alpaca at the same rate as before —
    but a page reload no longer blanks the Open-Positions table.
    """
    return {"activated": False}


@st.cache_data(ttl=BROKER_STATE_TTL_SECS, show_spinner=False)
def _fetch_account_cached(api_key: str, secret_key: str,
                          data_url: str, base_url: str) -> Dict:
    """Cached Alpaca account fetch. Cache key is the credential tuple so
    multiple environments (paper / live) cache independently.

    NOTE: this call is intentionally NOT routed through the
    market-data factory.  ``get_account_info`` is an *AccountPort*
    operation — it returns Alpaca-specific account state (equity,
    buying power, paper-vs-live flag) that has no equivalent shape on
    Schwab or Yahoo, and the executor talks to Alpaca regardless of
    which market-data provider is selected.  Hard-wiring Alpaca here
    keeps the broker-state cache reliable.
    """
    try:
        from trading_agent.market_data import MarketDataProvider
        provider = MarketDataProvider(
            alpaca_api_key=api_key,
            alpaca_secret_key=secret_key,
            alpaca_data_url=data_url,
            alpaca_base_url=base_url,
        )
        return provider.get_account_info() or {}
    except Exception as exc:
        # Re-raise so the wrapper can surface it via st.warning — caching
        # an exception would silently hide the warning for TTL seconds.
        raise RuntimeError(str(exc))


def _fetch_account(config) -> Dict:
    try:
        return _fetch_account_cached(
            config.alpaca.api_key,
            config.alpaca.secret_key,
            config.alpaca.data_url,
            config.alpaca.base_url,
        )
    except Exception as exc:
        st.warning(f"Account fetch failed: {exc}")
        return {}


def _fetch_spreads(config) -> Tuple[List[Dict], List[Dict]]:
    """
    Return (spreads, ungrouped_legs).

    `spreads` are option positions that matched a local `trade_plan_*.json`
    and were aggregated by `PositionMonitor.group_into_spreads`.

    `ungrouped_legs` are option legs in the broker account that did NOT
    match any local trade plan — typically positions opened outside the
    agent or runs whose plan files were rotated/deleted. They are still
    real money in the account, so we surface them rather than silently
    dropping them.
    """
    try:
        return _fetch_spreads_cached(
            config.alpaca.api_key,
            config.alpaca.secret_key,
            config.alpaca.base_url,
            str(config.logging.trade_plan_dir),
        )
    except Exception as exc:
        st.warning(f"Position fetch failed: {exc}")
        return [], []


@st.cache_data(ttl=BROKER_STATE_TTL_SECS, show_spinner=False)
def _fetch_spreads_cached(
    api_key: str, secret_key: str, base_url: str, trade_plan_dir: str,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Cached Alpaca position fetch + spread grouping. Cache key is the
    credential tuple + trade_plan_dir, so different paper/live envs and
    different plan directories cache independently. TTL bounds Alpaca
    API calls to ``BROKER_STATE_TTL_SECS`` (default 30 s) regardless of
    how often the dashboard fragment reruns — kills the 'Fetched N
    positions' log spam without sacrificing freshness.
    """
    from trading_agent.position_monitor import PositionMonitor
    monitor = PositionMonitor(
        api_key=api_key,
        secret_key=secret_key,
        base_url=base_url,
    )
    positions = monitor.fetch_open_positions()

    trade_plans: List[Dict] = []
    plan_dir = Path(trade_plan_dir)
    if plan_dir.exists():
        for fp in plan_dir.glob("trade_plan_*.json"):
            try:
                data = json.loads(fp.read_text())
                for entry in data.get("state_history", []):
                    tp = entry.get("trade_plan")
                    if tp:
                        trade_plans.append(tp)
            except Exception:
                pass

    spread_objs = monitor.group_into_spreads(positions, trade_plans)

    # ``group_into_spreads`` now folds in inferred spreads for legs whose
    # trade_plan_*.json was rotated out of state_history (or never
    # existed). Each SpreadPosition carries an ``origin`` field —
    # "trade_plan" or "inferred" — that we surface in the table so the
    # operator can tell at a glance which positions have full plan
    # context vs which were reconstructed from leg structure.
    spreads = [
        {
            "underlying":        s.underlying,
            "strategy_name":     s.strategy_name,
            "original_credit":   s.original_credit,
            "net_unrealized_pl": s.net_unrealized_pl,
            "expiration":        s.expiration,
            "exit_signal":       s.exit_signal.value,
            "origin":            getattr(s, "origin", "trade_plan"),
        }
        for s in spread_objs
    ]

    # Pure leftovers — broker has these symbols but inference couldn't
    # parse them into a spread (e.g. malformed OCC symbol, unsupported
    # leg combination). These are rare; we still surface them so the
    # user knows something exists at the broker that the dashboard
    # can't classify.
    matched_symbols = {leg.symbol for s in spread_objs for leg in s.legs}
    ungrouped_legs = []
    for p in positions:
        if p.symbol in matched_symbols:
            continue
        occ = _parse_occ(p.symbol)
        ungrouped_legs.append(
            {
                "symbol":          p.symbol,
                "underlying":      occ["underlying"],
                "expiration":      occ["expiration"],
                "type":            occ["type"],
                "strike":          occ["strike"],
                "qty":             p.qty,
                "side":            p.side,
                "avg_entry_price": p.avg_entry_price,
                "current_price":   p.current_price,
                "unrealized_pl":   p.unrealized_pl,
            }
        )

    return spreads, ungrouped_legs


@st.cache_data(ttl=BROKER_STATE_TTL_SECS, show_spinner=False)
def _is_market_open_cached(api_key: str, secret_key: str,
                           data_url: str, base_url: str) -> Optional[bool]:
    """
    Cached market-open check for the dashboard's badge.

    Routed through the factory so the answer comes from whichever
    provider the operator selected for the LIVE surface.  Alpaca uses
    its `/clock` endpoint; Schwab uses `/markets?markets=equity`;
    Yahoo falls back to a Mon-Fri 9:30-16:00 ET heuristic.  Keeps the
    badge honest when MARKET_DATA_PROVIDER_LIVE != alpaca.
    """
    from trading_agent.market_data_factory import build_market_data_provider
    provider = build_market_data_provider(
        alpaca_api_key=api_key,
        alpaca_secret_key=secret_key,
        alpaca_data_url=data_url,
        alpaca_base_url=base_url,
        surface="live",
    )
    return provider.is_market_open()


def _is_market_open(config) -> Optional[bool]:
    try:
        return _is_market_open_cached(
            config.alpaca.api_key,
            config.alpaca.secret_key,
            config.alpaca.data_url,
            config.alpaca.base_url,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Auto-refresh (non-blocking)
# ---------------------------------------------------------------------------

def _auto_refresh(interval_secs: int = REFRESH_INTERVAL) -> None:
    now = time.time()
    last = st.session_state.get("_live_last_refresh", 0)
    elapsed = now - last
    remaining = max(0, int(interval_secs - elapsed))
    if elapsed >= interval_secs:
        st.session_state["_live_last_refresh"] = now
    else:
        st.caption(f"Auto-refreshing in {remaining}s…")
        time.sleep(1)
        st.rerun()


# ---------------------------------------------------------------------------
# Strategy-Profile panel
# ---------------------------------------------------------------------------
#
# The panel writes ``STRATEGY_PRESET.json`` (next to AGENT_RUNNING) when
# the user clicks Apply.  The agent subprocess re-reads that file at the
# start of every cycle (see ``agent.TradingAgent.__init__`` →
# ``load_active_preset``), so changes take effect on the next 5-min tick
# without restarting the loop.
#
# Layout: two top-level selectboxes (risk profile + directional bias)
# plus an expander that's only meaningful when profile == "custom".

_PROFILE_OPTIONS:    List[str] = ["conservative", "balanced", "aggressive", "custom"]
_PROFILE_LABELS: Dict[str, str] = {
    "conservative": "Conservative — ~85% POP, low risk, fewer trades",
    "balanced":     "Balanced — ~75% POP, recommended baseline",
    "aggressive":   "Aggressive — ~65% POP, fat credits, gamma-sensitive",
    "custom":       "Custom — tune every knob yourself",
}

_BIAS_OPTIONS:    List[str] = ["auto", "bullish_only", "bearish_only", "neutral_only"]
_BIAS_LABELS: Dict[str, str] = {
    "auto":         "Auto — trade whatever regime classifier reports",
    "bullish_only": "Bullish only — Bull Puts + Iron Condors + MR",
    "bearish_only": "Bearish only — Bear Calls + Iron Condors + MR",
    "neutral_only": "Neutral only — Iron Condors + MR (no directional)",
}


def _parse_grid(text: str, kind: str) -> Optional[tuple]:
    """
    Parse a comma-separated grid string into a sorted unique tuple.

    ``kind`` is one of ``"int"`` (DTE values) or ``"float"`` (Δ / width %).
    Returns ``None`` on malformed input — caller should fall back to the
    seed value rather than persisting garbage.
    """
    out = []
    for tok in (text or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok) if kind == "int" else float(tok)
        except ValueError:
            return None
        out.append(v)
    if not out:
        return None
    return tuple(sorted(set(out)))


def _custom_inputs(seed: PresetConfig) -> Dict:
    """Render the Custom-mode override widgets and return a dict of values."""
    st.caption(
        "Custom overrides start from the **Balanced** baseline. Only the "
        "fields you change are persisted; everything else stays on the default."
    )

    c1, c2 = st.columns(2)
    with c1:
        max_delta = st.slider(
            "Max short-leg |Δ|", 0.05, 0.45, float(seed.max_delta), 0.01,
            help="0.15 ≈ 85% POP · 0.25 ≈ 75% POP · 0.35 ≈ 65% POP",
            key="cust_max_delta",
        )
        min_credit_ratio = st.slider(
            "Credit/Width floor", 0.10, 0.50, float(seed.min_credit_ratio), 0.05,
            help="Reject spreads whose credit / width is below this floor.",
            key="cust_min_cw",
        )
        max_risk_pct = st.slider(
            "Max account risk per trade (%)",
            0.5, 5.0, float(seed.max_risk_pct) * 100, 0.5,
            help="Hard cap on max-loss as a fraction of account equity.",
            key="cust_max_risk",
        ) / 100.0
        dte_window_days = st.slider(
            "DTE window ± (days)", 1, 14, int(seed.dte_window_days), 1,
            key="cust_dte_window",
        )
    with c2:
        dte_vertical = st.slider(
            "Vertical (Bull Put / Bear Call) DTE",
            5, 60, int(seed.dte_vertical), 1,
            key="cust_dte_v",
        )
        dte_iron_condor = st.slider(
            "Iron Condor DTE", 7, 60, int(seed.dte_iron_condor), 1,
            key="cust_dte_ic",
        )
        dte_mean_reversion = st.slider(
            "Mean-Reversion DTE", 3, 30, int(seed.dte_mean_reversion), 1,
            key="cust_dte_mr",
        )

    st.markdown("**Spread-width policy**")
    wc1, wc2 = st.columns(2)
    with wc1:
        width_mode = st.radio(
            "Width mode",
            options=["pct_of_spot", "fixed_dollar"],
            index=0 if seed.width_mode == "pct_of_spot" else 1,
            format_func=lambda v: "% of spot" if v == "pct_of_spot" else "Fixed $",
            horizontal=True,
            key="cust_width_mode",
        )
    with wc2:
        if width_mode == "pct_of_spot":
            width_value = st.slider(
                "Width (% of spot)", 0.5, 5.0,
                float(seed.width_value) * 100 if seed.width_mode == "pct_of_spot"
                else 1.5,
                0.1, key="cust_width_pct",
            ) / 100.0
        else:
            width_value = st.slider(
                "Width ($)", 1.0, 25.0,
                float(seed.width_value) if seed.width_mode == "fixed_dollar"
                else 5.0,
                0.5, key="cust_width_usd",
            )

    # ── Adaptive scan grids — only meaningful when scan_mode == "adaptive",
    #    but always shown so a user can pre-stage a Custom payload.
    with st.expander(
        "Adaptive scan grids (only used when Scan Mode = Adaptive)",
        expanded=False,
    ):
        st.caption(
            "The chain scanner sweeps the cross-product of these three grids "
            "and picks the highest-EV candidate. Empty an entry to fall back "
            "to the preset default. Comma-separated."
        )
        min_pop = st.slider(
            "Min POP (annualised score floor)", 0.30, 0.85,
            float(seed.min_pop), 0.05,
            help="Drop candidates whose POP (≈ 1 − |Δshort|) is below this.",
            key="cust_min_pop",
        )
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            dte_grid_text = st.text_input(
                "DTE grid (days)",
                value=", ".join(str(d) for d in seed.dte_grid),
                key="cust_dte_grid",
                help="e.g. 7, 14, 21, 30",
            )
        with gc2:
            delta_grid_text = st.text_input(
                "Δ grid (|short delta|)",
                value=", ".join(f"{d:g}" for d in seed.delta_grid),
                key="cust_delta_grid",
                help="e.g. 0.20, 0.25, 0.30, 0.35",
            )
        with gc3:
            width_grid_text = st.text_input(
                "Width grid (% of spot)",
                value=", ".join(f"{w:g}" for w in seed.width_grid_pct),
                key="cust_width_grid",
                help="e.g. 0.010, 0.015, 0.020, 0.025",
            )

    payload = {
        "max_delta":          max_delta,
        "dte_vertical":       dte_vertical,
        "dte_iron_condor":    dte_iron_condor,
        "dte_mean_reversion": dte_mean_reversion,
        "dte_window_days":    dte_window_days,
        "width_mode":         width_mode,
        "width_value":        width_value,
        "min_credit_ratio":   min_credit_ratio,
        "max_risk_pct":       max_risk_pct,
        "min_pop":            min_pop,
    }
    # Only persist grids when they parse cleanly — silently fall back to
    # seed value otherwise so a malformed text box doesn't poison the file.
    parsed_dte    = _parse_grid(dte_grid_text,    "int")
    parsed_delta  = _parse_grid(delta_grid_text,  "float")
    parsed_width  = _parse_grid(width_grid_text,  "float")
    if parsed_dte:    payload["dte_grid"]      = parsed_dte
    if parsed_delta:  payload["delta_grid"]    = parsed_delta
    if parsed_width:  payload["width_grid_pct"] = parsed_width
    return payload


def render_strategy_profile_panel() -> None:
    """
    Two-row Strategy-Profile selector + Apply button.

    Reads the current preset from ``STRATEGY_PRESET.json`` (or the
    BALANCED default when the file is missing) and writes the next
    selection back atomically. Hot-applied on the next cycle.
    """
    current = load_active_preset()
    is_loop_running = AGENT_RUNNING.exists()

    # Defensive: ``to_short_summary()`` was added 2026-05-06.  When
    # Streamlit hot-reloads after a code edit it can keep old class
    # bindings alive in @st.cache_resource — a PresetConfig instance
    # constructed from the OLD class won't have the new method.  Fall
    # back to to_summary_line() so the dashboard renders cleanly even
    # in that transient state; a hard refresh (Cmd+Shift+R) clears it.
    try:
        _expander_label = f"Strategy Profile — {current.to_short_summary()}"
    except AttributeError:
        _expander_label = f"Strategy Profile — {current.to_summary_line()}"

    with st.expander(
        # Concise label for the collapsed state — the verbose
        # ``to_summary_line()`` runs ~180 chars and was hard to scan.
        # Full detail still surfaces inside the expander body.
        _expander_label,
        # User preference: keep configuration grids collapsed by default.
        # The summary line in the header is enough to know what's active
        # at a glance; expand only when tweaking.
        expanded=False,
    ):
        st.markdown(
            "Pick a risk profile + directional bias. Changes are written to "
            "`STRATEGY_PRESET.json` and applied **on the next cycle** — no "
            "restart needed. The active preset drives Δ-short, DTE per "
            "strategy, spread width, C/W floor, and the % of equity at risk."
        )

        col_p, col_b = st.columns(2)

        # Decide the index to show as currently-selected.  If the file says
        # "custom", that's preserved; otherwise lookup the current preset's
        # name in the canonical option list.
        try:
            saved_payload = (
                json.loads(PRESET_FILE.read_text())
                if PRESET_FILE.exists() else {}
            )
        except (json.JSONDecodeError, OSError):
            saved_payload = {}

        profile_default = (
            saved_payload.get("profile", current.name)
            if saved_payload.get("profile") in _PROFILE_OPTIONS
            else current.name
        )
        bias_default = current.directional_bias

        with col_p:
            profile = st.selectbox(
                "Risk profile",
                options=_PROFILE_OPTIONS,
                index=_PROFILE_OPTIONS.index(profile_default),
                format_func=lambda v: _PROFILE_LABELS[v],
                key="strat_profile",
            )
        with col_b:
            bias = st.selectbox(
                "Directional bias",
                options=_BIAS_OPTIONS,
                index=_BIAS_OPTIONS.index(bias_default),
                format_func=lambda v: _BIAS_LABELS[v],
                key="strat_bias",
            )

        # ── Scan-mode overlay row ────────────────────────────────────────
        # Static  → planner uses fixed (Δ, DTE, width) preset values.
        # Adaptive → ChainScanner sweeps (DTE × Δ × width) grid and picks
        # the highest-EV candidate that clears |Δshort|×(1+edge_buffer).
        # Both RiskManager and the executor's live-credit recheck switch
        # to the same Δ-aware floor when adaptive is selected, so the
        # planner / risk / exec floors never drift.
        col_s, col_e = st.columns(2)
        scan_default = saved_payload.get("scan_mode") or current.scan_mode
        if scan_default not in ("static", "adaptive"):
            scan_default = "static"
        edge_default = saved_payload.get("edge_buffer", current.edge_buffer)
        with col_s:
            scan_mode = st.radio(
                "Scan mode",
                options=["static", "adaptive"],
                index=0 if scan_default == "static" else 1,
                format_func=lambda v: (
                    "Static — fixed Δ/DTE/width" if v == "static"
                    else "Adaptive — chain scanner picks best EV"
                ),
                horizontal=True,
                key="strat_scan_mode",
                help="Adaptive replaces the static preset triple with a "
                     "(DTE × Δ × width) grid sweep, scoring each candidate "
                     "by per-dollar-risked EV. Floor becomes "
                     "|Δshort|×(1+edge_buffer) so it stays above breakeven "
                     "for whatever Δ the scanner picks.",
            )
        with col_e:
            edge_buffer = st.slider(
                "Edge buffer (over breakeven C/W)",
                0.0, 0.50, float(edge_default), 0.01,
                disabled=(scan_mode != "adaptive"),
                help="Required C/W = |Δshort| × (1 + edge_buffer). "
                     "10% is a reasonable default — drops to 5% in tight "
                     "tape, raise to 20%+ if you want to demand more cushion "
                     "before taking the trade.",
                key="strat_edge_buffer",
            )

        # Preview line for the chosen built-in.
        if profile in PRESETS:
            preview = PRESETS[profile]
            mode_tag = ("ADAPTIVE scan" if scan_mode == "adaptive"
                        else "STATIC scan")
            st.caption(
                f"Selected preset → {preview.description}  · {mode_tag} "
                f"(edge buffer {edge_buffer:.2f})"
            )

        # Custom overrides — only meaningful when profile == "custom".
        custom_payload: Optional[Dict] = None
        if profile == "custom":
            seed = current if current.name == "custom" else BALANCED
            custom_payload = _custom_inputs(seed)

        # Apply / status row
        ac1, ac2 = st.columns([1, 3])
        with ac1:
            apply_clicked = st.button(
                "💾 Apply", type="primary", width='stretch',
                key="strat_apply",
            )
        with ac2:
            if is_loop_running:
                st.caption(
                    "Agent is running — changes apply on the **next cycle** "
                    "(no restart, no current-cycle interruption)."
                )
            else:
                st.caption(
                    "Agent is stopped — the new preset will load on the "
                    "first cycle when you click ▶ Start Agent."
                )

        if apply_clicked:
            try:
                save_active_preset(
                    profile=profile,
                    directional_bias=bias,
                    custom=custom_payload,
                    scan_mode=scan_mode,
                    edge_buffer=edge_buffer,
                )
                st.toast(
                    f"Saved profile=**{profile}**, bias=**{bias}**, "
                    f"scan=**{scan_mode}** (edge {edge_buffer:.2f}) — "
                    "active on next cycle.",
                    icon="✅",
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save preset: {exc}")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

@st.fragment(run_every=REFRESH_INTERVAL)
def render_live_monitor() -> None:
    """Render the Live Monitoring tab — single entry point for the trading agent."""

    # ── Session-boot orphan sweep ─────────────────────────────────────────
    # Runs exactly once per Streamlit session.  If a previous session's
    # cycle subprocess survived a crash or auto-reload (orphan reparented
    # to init), or AGENT_RUNNING is set but AGENT_PID points to a dead
    # process, this sweep cleans it up so the dashboard always lands on
    # a single-process state.  See ``_sweep_orphans_at_session_boot``.
    boot_sweep = _sweep_orphans_at_session_boot()

    # ── Self-heal a zombie agent loop ─────────────────────────────────────
    # Streamlit's file watcher restarts the Python process on code edits,
    # killing the daemon thread that runs the agent loop while leaving
    # the AGENT_RUNNING sentinel on disk. Without this check the UI shows
    # "ACTIVE [LIVE]" / "Stop Agent" but no cycle subprocess ever spawns
    # because there's no live thread to call ``_run_one_cycle``. See
    # ``_ensure_loop_alive_if_intended`` for the full rationale.
    revived = _ensure_loop_alive_if_intended()

    loop_running  = _is_loop_running()
    cycle_running = AGENT_PID.exists()
    dry_mode      = _is_dry_run_mode()

    # ── Header row ────────────────────────────────────────────────────────
    st.subheader("Live Portfolio Monitor")

    if boot_sweep is not None:
        # A previous session left orphan agent processes alive and we
        # cleaned them up before this dashboard rendered.  Surface what
        # happened so the operator knows why the previous "Stop" didn't
        # fully take and what state we're in now.
        killed = boot_sweep.get("killed_pids", [])
        swept_match = boot_sweep.get("swept", 0)
        bits = []
        if killed:
            bits.append(f"terminated PID(s) {killed}")
        if swept_match:
            bits.append("pkill cleared straggler agent.py processes")
        detail = "; ".join(bits) if bits else "cleared stale sentinels"
        st.warning(
            f"🧹 Cleaned up orphan agent state on session boot — {detail}. "
            "You're on a clean single-process baseline. Click **Start Agent** "
            "to begin a fresh cycle."
        )

    if revived:
        # Surface what just happened so the user understands why their UI
        # says "Stop" but the most-recent log line is timestamped before
        # this rerun — the loop was sleeping behind a dead thread until
        # this render call woke it back up.
        st.warning(
            "🔄 Agent loop was auto-revived after a Streamlit code reload. "
            "Cycle activity will resume on the next tick. If you didn't "
            "expect this — click **Stop Agent** to halt cleanly, then "
            "Start again."
        )

    # ── Strategy profile panel (first — pick risk + bias BEFORE starting) ─
    render_strategy_profile_panel()

    # ── Dry Run toggle (most prominent choice — pick BEFORE starting) ─────
    with st.expander(
        "Dry Run Mode — simulate after hours without real orders",
        # User preference: collapsed by default. The toggle inside is a
        # niche option that only matters once per session, so it
        # shouldn't take vertical space when not in use.
        expanded=False,
    ):
        dry_col, info_col = st.columns([1, 2])
        with dry_col:
            new_dry = st.toggle(
                "Enable Dry Run",
                value=dry_mode,
                disabled=loop_running,
                help="Can only be changed while the agent is stopped.",
                key="dry_run_toggle",
            )
            if not loop_running and new_dry != dry_mode:
                if new_dry:
                    DRY_RUN_FLAG.write_text(_now())
                else:
                    DRY_RUN_FLAG.unlink(missing_ok=True)
                st.rerun()

        with info_col:
            if new_dry or dry_mode:
                st.info(
                    "**Dry Run is ON.** The agent will:\n"
                    "- Run the full pipeline: real market data, option chains, "
                    "regime classification, all 8 risk guardrails\n"
                    "- Bypass the market-hours check (`FORCE_MARKET_OPEN=true`) "
                    "so you can simulate after hours\n"
                    "- Log every decision to `signals.jsonl` as `action: dry_run` — "
                    "no orders submitted to Alpaca\n\n"
                    "**Perfect for after-hours review:** see exactly what the agent "
                    "would trade tonight using today's closing prices and live Greeks."
                )
            else:
                st.info(
                    "**Dry Run is OFF — LIVE mode.** Orders are submitted to Alpaca "
                    "when all risk guardrails pass. The market-hours check is enforced "
                    "(`9:25 AM – 4:05 PM ET` only).\n\n"
                    "Enable Dry Run to simulate the full pipeline after hours "
                    "without risking capital."
                )

    # ── Agent control buttons ──────────────────────────────────────────────
    btn_cols = st.columns([2, 2, 2, 1])

    with btn_cols[0]:
        if loop_running:
            if st.button("⏹ Stop Agent", type="secondary", width='stretch'):
                _stop_agent()
                st.toast("Stop requested — current cycle will finish then halt.", icon="⏹")
                st.rerun()
        else:
            label = "▶ Start (Dry Run)" if dry_mode else "▶ Start Agent"
            btn_type = "secondary" if dry_mode else "primary"
            if st.button(label, type=btn_type, width='stretch'):
                _start_agent(dry_run=dry_mode)
                mode = "DRY-RUN" if dry_mode else "LIVE"
                st.toast(f"Agent loop started [{mode}]!", icon="▶")
                st.rerun()

    with btn_cols[1]:
        if PAUSE_FLAG.exists():
            if st.button("▶ Resume", width='stretch'):
                PAUSE_FLAG.unlink(missing_ok=True)
                st.toast("Agent resumed.", icon="▶")
                st.rerun()
        else:
            if st.button("⏸ Pause", width='stretch', disabled=not loop_running):
                PAUSE_FLAG.write_text(_now())
                st.toast("Agent paused — no new orders until resumed.", icon="⏸")
                st.rerun()

    with btn_cols[2]:
        run_once_label = "⚡ Run Once (Dry)" if dry_mode else "⚡ Run Once"
        if st.button(run_once_label, width='stretch', disabled=cycle_running):
            if not loop_running:
                _is_dry = dry_mode

                def _one_shot(is_dry=_is_dry):
                    mode = "DRY-RUN" if is_dry else "LIVE"
                    _append_log(f"[{_now()}] --- One-shot cycle start [{mode}] ---")
                    rc = _run_one_cycle(dry_run=is_dry)
                    _append_log(f"[{_now()}] --- One-shot cycle end [{mode}] (exit={rc}) ---")

                t = threading.Thread(target=_one_shot, daemon=True)
                t.start()
                mode = "dry-run" if dry_mode else "live"
                st.toast(f"Running one {mode} cycle now…", icon="⚡")
                st.rerun()

    with btn_cols[3]:
        if st.button("🔴", width='stretch',
                     help="SIGKILL the running cycle immediately (emergency)",
                     disabled=not cycle_running):
            _kill_current_cycle()
            st.toast("Cycle killed.", icon="🔴")
            st.rerun()

    # ── Status banner ──────────────────────────────────────────────────────
    mode_badge = " · DRY RUN" if dry_mode else " · LIVE"
    # Process-singleton verification — counts live agent.py processes
    # via the same pgrep pattern the orphan sweep uses.  Renders inline
    # with the status banner so the operator can confirm at a glance
    # that exactly one agent is running (or see the warning if multiple).
    singleton_pill = _render_singleton_pill()
    if loop_running and cycle_running:
        pid = AGENT_PID.read_text().strip() if AGENT_PID.exists() else "?"
        if dry_mode:
            st.warning(f"Agent is **RUNNING [DRY RUN]** — cycle in progress (PID {pid}) — no orders will be placed{singleton_pill}")
        else:
            st.success(f"Agent is **RUNNING [LIVE]** — cycle in progress (PID {pid}){singleton_pill}")
    elif loop_running:
        if dry_mode:
            st.warning(f"Agent is **ACTIVE [DRY RUN]{mode_badge}** — waiting for next cycle — no orders will be placed{singleton_pill}")
        else:
            st.success(f"Agent is **ACTIVE [LIVE]** — waiting for next cycle{singleton_pill}")
    elif PAUSE_FLAG.exists():
        st.warning(
            f"Agent is **PAUSED** since {_safe_read_text(PAUSE_FLAG)[:19]} UTC. "
            f"Click ▶ Resume to continue.{singleton_pill}"
        )
    else:
        if dry_mode:
            st.info(f"Agent is **STOPPED** · Dry Run mode is armed — click ▶ Start (Dry Run) to simulate.{singleton_pill}")
        else:
            st.error(f"Agent is **STOPPED** — click ▶ Start Agent to begin live trading.{singleton_pill}")

    st.divider()

    # ── Load live data ─────────────────────────────────────────────────────
    config = _get_config()
    journal_df = _load_journal_df()

    equity = 0.0
    total_pnl = 0.0
    spreads: List[Dict] = []
    ungrouped_legs: List[Dict] = []

    # ── Broker-state fetch gating ─────────────────────────────────────────
    # Goal: don't hammer Alpaca during idle, but ALSO don't blank the
    # Open-Positions table the moment the user reloads the page.
    #
    # Original design: only call ``_fetch_spreads`` when the agent loop
    # is running OR a one-shot ``_bm_force_refresh`` flag is set. That
    # was great for log-spam reduction but broke two ways:
    #   1. Auto-refresh fragment ticks ~3 s later wiped the
    #      just-displayed spreads (the flag is a one-shot).
    #   2. A browser reload creates a fresh ``st.session_state``, so
    #      any session-scoped snapshot dies on F5 → "No open positions"
    #      reappears even when the broker has filled positions.
    #
    # Fix: track "did the user ask for broker data at least once since
    # this Streamlit process started" via ``_broker_fetch_marker``,
    # which is backed by ``@st.cache_resource`` (process-scoped — it
    # survives session resets). Once that marker is set, EVERY render
    # routes through ``_fetch_spreads_cached``. The 30 s
    # ``@st.cache_data`` TTL on that function absorbs fragment ticks
    # (cache hit = microseconds, no Alpaca call), so we hit Alpaca at
    # most ~once per 30 s — same rate as before — while the displayed
    # data stays continuous across reloads, fragment ticks, and
    # tab-switches within the TTL window.
    fetch_broker_now = st.session_state.pop("_bm_force_refresh", False)
    marker = _broker_fetch_marker()
    if fetch_broker_now or loop_running:
        marker["activated"] = True

    should_fetch_broker = bool(config) and (
        loop_running or fetch_broker_now or marker["activated"]
    )
    if should_fetch_broker:
        account = _fetch_account(config)
        equity = float(account.get("equity") or 0)
        spreads, ungrouped_legs = _fetch_spreads(config)
        # Compute total Unrealized P&L by SUMMING what's actually shown
        # in the Open Positions table. The previous version read
        # ``account.unrealized_pl`` directly from Alpaca's /v2/account
        # endpoint, but that field is unreliable for option positions
        # on paper accounts — Alpaca consistently reports 0 even when
        # individual leg snapshots have correct unrealized_pl values.
        # Summing per-spread (+ per-leg for any unclassified) gives a
        # number that always matches the table the user is looking at.
        # Falls back to the broker's account-level field only if our
        # local sum is exactly zero — covers the edge case of an
        # account holding only long stock with no option spreads.
        total_pnl = (
            sum(float(s.get("net_unrealized_pl") or 0) for s in spreads)
            + sum(float(L.get("unrealized_pl") or 0) for L in ungrouped_legs)
        )
        if total_pnl == 0.0:
            total_pnl = float(account.get("unrealized_pl") or 0)

    if equity == 0.0 and not journal_df.empty:
        nonzero = journal_df[journal_df["account_balance"] > 0]["account_balance"]
        if not nonzero.empty:
            equity = nonzero.iloc[-1]

    # ── Dominant Regime ───────────────────────────────────────────────
    # Take the mode regime across the last 50 *classified* journal rows.
    # We exclude rows whose regime field is empty / "unknown" because
    # those are predominantly ``skipped_existing`` entries — the agent
    # skips a ticker before classification fires when it already has an
    # open position or pending order, so no regime label is recorded.
    # Once you hold positions on 3+ tickers (typical), those skip rows
    # outnumber the genuinely-classified rejected/submitted rows in any
    # 20-row window and would push the mode to "unknown" — exactly the
    # UI bug we're fixing here.
    #
    # Widening the window from 20 → 50 also helps so a single cycle's
    # batch of skip rows doesn't dominate the count when classification
    # rows from earlier cycles are still relevant context.
    regime = "unknown"
    if not journal_df.empty:
        recent = journal_df.tail(50)
        valid = recent[
            recent["regime"].notna()
            & (recent["regime"] != "")
            & (recent["regime"] != "unknown")
        ]
        if not valid.empty:
            regime = valid["regime"].value_counts().idxmax()

    # The countdown is only meaningful when the background loop is alive
    # — otherwise nothing is going to fire when the wall clock hits zero.
    # Pass ``None`` so ``metric_row`` renders "—" + a "Stopped" chip instead
    # of a misleading "290s" timer.
    if loop_running:
        now = datetime.now()
        cycle_secs: Optional[int] = max(
            0,
            CYCLE_INTERVAL_SEC - (now.minute * 60 + now.second) % CYCLE_INTERVAL_SEC,
        )
    else:
        cycle_secs = None

    # ── Realized P&L from today's closes ───────────────────────────────────
    # Sum action="closed" rows with timestamps in today UTC.  This is the
    # operator's "money I locked in today" number; combined with
    # total_pnl (currently-open) it produces the headline "Daily P&L"
    # tile.  Pre-2026-05-06 the tile only showed unrealized — after the
    # first close of the day, the headline number stopped reflecting
    # the operator's true daily P&L.
    realized_today = 0.0
    if not journal_df.empty and "action" in journal_df.columns:
        try:
            today_utc_ts = pd.Timestamp.utcnow().normalize()
            closed_today = journal_df[
                (journal_df["action"] == "closed")
                & (pd.to_datetime(journal_df["timestamp"], utc=True)
                   >= today_utc_ts)
            ]
            for _, row in closed_today.iterrows():
                rs = row.get("raw_signal") or {}
                if not isinstance(rs, dict):
                    continue
                # ``net_unrealized_pl`` at close time IS the realized
                # P&L of that position — Alpaca freezes the value once
                # the legs settle.
                realized_today += float(rs.get("net_unrealized_pl") or 0)
        except Exception as exc:                                 # noqa: BLE001
            # Defensive: if the journal schema drifts, prefer to render
            # the unrealized-only tile rather than crash the dashboard.
            logger.warning("Realized-P&L compute failed: %s", exc)
            realized_today = 0.0

    # ── Metrics row ────────────────────────────────────────────────────────
    metric_row(equity, total_pnl, regime, cycle_secs,
               realized_pnl=realized_today)

    # ── Cycle staleness beacon ────────────────────────────────────────────
    # If the latest journal entry is older than 5 minutes, the agent
    # may be in trouble — a cycle takes ~30-60s and fires every 5 min,
    # so a fresh journal row should appear at most ~6 min after the
    # previous one.  Render a clear orange/red banner when stale.  The
    # beacon is suppressed when the loop is stopped (the operator
    # already knows nothing's running) and on a brand-new install
    # (empty journal).  See _sweep_orphan_agents for the complementary
    # process-side health check.
    _render_cycle_staleness_beacon(journal_df, loop_running)

    st.divider()

    # ── Open positions ─────────────────────────────────────────────────────
    op_hdr_col, op_btn_col = st.columns([4, 1])
    with op_hdr_col:
        st.subheader("Open Positions")
    with op_btn_col:
        # Manual one-off broker pull — useful when the agent is stopped
        # but the operator wants to peek at live broker state. Sets a
        # session-state flag that the next fragment-tick consumes.
        if not loop_running and st.button(
            "↻ Refresh broker state",
            key="bm_force_refresh_btn",
            help=(
                "Hit Alpaca once for the latest positions + account "
                "equity. While the agent is stopped, broker state is "
                "not auto-polled (no signal to monitor)."
            ),
        ):
            st.session_state["_bm_force_refresh"] = True
            # Clear the cached fetches so the manual click definitely
            # round-trips to Alpaca rather than serving the TTL cache.
            try:
                _fetch_account_cached.clear()
                _fetch_spreads_cached.clear()
            except Exception:
                pass
            st.rerun()
    if not should_fetch_broker:
        st.caption(
            "ℹ️ Agent stopped — broker positions not auto-polled. "
            "Click **↻ Refresh broker state** above for a one-off "
            "snapshot, or **Start Agent** to resume polling."
        )
    # Pass journal_df so the table can add the "Why" column + entry-
    # justification expander pulling thesis + risk-manager checks from
    # the matching submitted entry trade for each open spread.
    positions_table(spreads, journal_df=journal_df)

    # Origin breakdown — concise inline summary so the operator can tell
    # at a glance how many spreads have full trade-plan context vs were
    # reconstructed from leg structure (still tradeable; just less rich
    # exit-rule context).
    n_matched = sum(1 for s in spreads if s.get("origin") == "trade_plan")
    n_inferred = sum(1 for s in spreads if s.get("origin") == "inferred")
    if n_inferred > 0:
        st.caption(
            f"📋 {n_matched} spread(s) matched to a `trade_plan_*.json`, "
            f"{n_inferred} **inferred** from broker leg structure (no "
            "matching plan — likely an older fill whose plan history "
            "was rotated out, or opened manually). Inferred rows still "
            "compute live P&L from leg snapshots; their `original_credit` "
            "is recovered from each leg's `avg_entry_price`."
        )

    # Truly malformed legs (failed OCC parse, etc.) still surface as a
    # last-resort fallback so nothing on the broker is silently hidden.
    if ungrouped_legs:
        st.caption(
            f"⚠️ Unclassifiable legs ({len(ungrouped_legs)}) — broker "
            "positions whose OCC symbol couldn't be parsed or whose leg "
            "combination doesn't fit a known spread shape."
        )
        ungrouped_legs_table(ungrouped_legs)

    # ── Closed Today (collapsed by default) ───────────────────────────────
    # Lists every spread the position-monitor closed today via an exit
    # signal (strike_proximity / profit_target / hard_stop / regime_shift
    # / dte_safety / expired) along with the realised P&L and the exit
    # justification.  Sourced from journal rows where action="closed" —
    # written by ``agent.py:_journal_close_event`` (added 2026-05-06).
    # Sits below Open Positions so the live broker state stays the
    # operator's primary anchor; closes are a secondary audit-trail
    # view.  If no closes happened today the expander is hidden
    # entirely so the operator doesn't see an empty section.
    _render_closed_today(journal_df)

    # ── Close Failures Today (collapsed by default) ───────────────────────
    # Same source — but filters to action="close_failed" rows that are
    # written when the executor's DELETE was rejected by Alpaca (PDT,
    # uncovered, insufficient buying power).  These positions are still
    # open on the broker.  Hidden when there are zero failures, so the
    # absence of the panel is itself a positive signal.
    _render_close_failures_today(journal_df)
    st.divider()

    # ── Equity curve (collapsed by default per user preference) ───────────
    equity_df = journal_df[journal_df["account_balance"] > 0].copy()
    if not equity_df.empty:
        with st.expander("📈 Equity Curve", expanded=False):
            st.plotly_chart(equity_curve_chart(equity_df), width='stretch')

    # ── Guardrail status (always expanded per user preference) ────────────
    # Per-ticker × per-guardrail grid for the latest cycle (mode-scoped).
    # Each cell carries the value-bearing chip (e.g. "0.34 ≥ 0.20",
    # "$200 ≤ $2000", "FORCED") computed by ``_compact_for`` so the
    # operator can see WHY a check passed/failed without hovering.
    # Hover the cell for the verbatim risk-manager string.  The legacy
    # single-row 8-card panel was removed because the grid now carries
    # strictly more information for every ticker in the cycle.
    active_mode = "DRY-RUN" if dry_mode else "LIVE"
    # Build the held-tickers set from the same broker fetch that drives
    # the Open Positions table — single source of truth. Passed to the
    # guardrail grid so a ticker can only show as "🔒 HOLDING" when its
    # position genuinely exists at the broker right now. Without this
    # cross-check, a ticker whose limit order was submitted earlier and
    # then cancelled (stale-order maintenance / DAY-tif expiry / manual
    # cancel) would keep showing HOLDING for the rest of the day even
    # though nothing is actually open.
    held_tickers = {s.get("underlying", "") for s in spreads}
    held_tickers.discard("")  # defensive — drop any blank symbols
    with st.expander(
        f"🛡️ Risk Guardrail Status — Latest Cycle [{active_mode}]",
        expanded=True,
    ):
        # Read the active preset's max_risk_pct so the Max-Loss column
        # header reflects the actual budget (e.g. "≤ 5% Equity" instead
        # of the hardcoded legacy "≤ 2% Equity").  Defensive try/except —
        # a missing or malformed STRATEGY_PRESET.json must not crash the
        # dashboard render; the helper falls back to the legacy label.
        try:
            _grid_max_risk_pct = float(load_active_preset().max_risk_pct)
        except Exception:                                    # noqa: BLE001
            _grid_max_risk_pct = None
        guardrail_grid(
            _guardrail_grid_from_journal(
                journal_df,
                current_mode=active_mode,
                held_tickers=held_tickers,
            ),
            max_risk_pct=_grid_max_risk_pct,
        )

    # ── SMA drift caption (Market OPEN/CLOSED moved to page header) ───────
    # The OPEN/CLOSED status now lives as a coloured badge in the
    # top-right of the page header (rendered in app.py), so it's always
    # visible from any tab without taking vertical space here. We keep
    # only the SMA50/SMA200 drift + RSI line as a compact one-line
    # caption, which is genuinely scrollable context that isn't
    # redundant with the header badge.
    if not journal_df.empty:
        last = journal_df.iloc[-1]
        sma50, sma200, rsi = last.get("sma_50", 0), last.get("sma_200", 0), last.get("rsi_14", 0)
        if sma50 and sma200:
            drift = (sma50 - sma200) / sma200 * 100
            st.caption(
                f"Last signal — SMA50/SMA200 drift: {drift:+.2f}%  |  "
                f"RSI-14: {rsi:.1f}"
            )
    st.divider()

    # ── Agent log expander ─────────────────────────────────────────────────
    # User preference: collapsed by default. The log is a verbose debug
    # tool; users who need it can click open. Default-collapse keeps the
    # important panels above the fold.
    with st.expander("Agent Log (last 50 lines)", expanded=False):
        if AGENT_LOG.exists():
            # _safe_read_text handles the rare case where a non-UTF-8
            # byte slips into the log (e.g. a Windows-1252 char from a
            # vendor SDK trace). The previous strict ``read_text()``
            # crashed the entire dashboard render with
            # ``UnicodeDecodeError`` when this happened.
            lines = _safe_read_text(AGENT_LOG).splitlines()
            log_text = "\n".join(lines[-50:])
            st.code(log_text, language="text")
        else:
            st.caption("No log yet — start the agent to see output here.")

    # ── Adaptive scanner diagnostics ───────────────────────────────────────
    # Surfaces the per-ticker scan_results block written by the agent so
    # users can see why candidates are being rejected even when zero
    # trades fire (e.g. cw_below_floor with edge_buffer too tight).
    _render_scanner_diagnostics_panel(journal_df)

    # ── Journal expander ───────────────────────────────────────────────────
    # User preference: ALWAYS expanded. Journal entries are the operator's
    # primary signal stream — what the agent decided this cycle, why
    # candidates were rejected, what got submitted. They want this open
    # at all times so they can scroll through cycle history without
    # clicking.
    with st.expander("Recent Journal Entries", expanded=True):
        if journal_df.empty:
            st.info(
                "No journal entries found at "
                "trade_journal/signals_live.jsonl "
                "(or legacy trade_journal/signals.jsonl)."
            )
        else:
            # Filter out orphan event-only rows (cycle_error /
            # after_hours_shutdown entries written by the stale-order
            # canceller and the after-hours guard). They have no
            # ticker/action and otherwise pollute the table with
            # rows full of blanks + regime="unknown". A separate caption
            # below counts them so the operator still sees the activity.
            display_df = journal_df
            n_orphans = 0
            if "ticker" in display_df.columns:
                has_ticker = display_df["ticker"].notna() & (
                    display_df["ticker"].astype(str).str.strip() != ""
                )
                n_orphans = int((~has_ticker).sum())
                display_df = display_df[has_ticker]

            cols = ["timestamp", "ticker", "action", "regime", "notes"]
            cols = [c for c in cols if c in display_df.columns]
            st.dataframe(
                display_df[cols].tail(100).iloc[::-1],
                width='stretch',
                hide_index=True,
            )
            if n_orphans:
                st.caption(
                    f"ℹ️ Hiding {n_orphans} agent-event row(s) "
                    "(after-hours shutdown / stale-order cancellation "
                    "logs) that have no ticker/action context."
                )

    # ── Manual refresh ─────────────────────────────────────────────────────
    # Note: auto-refresh is handled by @st.fragment(run_every=REFRESH_INTERVAL)
    # above. Do NOT call _auto_refresh() here — it would add an extra
    # time.sleep+rerun loop that triggers full-page reruns every second,
    # breaking other tabs (especially the backtesting sidebar).
    st.divider()
    col_r1, col_r2 = st.columns([5, 1])
    with col_r1:
        st.caption(f"Auto-refreshes every {REFRESH_INTERVAL}s via fragment")
    with col_r2:
        if st.button("Refresh Now"):
            st.rerun()
