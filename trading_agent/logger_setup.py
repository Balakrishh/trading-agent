"""
Logging configuration — sets up console + rotating file handlers
with a consistent format across all modules.

Week 3-4 upgrade
----------------
The prior implementation used a plain FileHandler keyed by UTC date.
That had two failure modes for a long-running agent:

  1. Within a single day the log file grew unbounded.  A noisy cycle
     (e.g. LLM analyst dumping full prompts) could fill the disk.
  2. Old files were never purged.  Over weeks of running, thousands
     of ``trading_agent_YYYYMMDD.log`` files would accumulate.

The new implementation uses ``RotatingFileHandler``:
  • 10 MB per file (tunable via env ``LOG_MAX_BYTES``)
  • 7 rollover files kept (tunable via env ``LOG_BACKUP_COUNT``)
  • Old files are automatically renamed ``...log.1``, ``...log.2``, …
    and anything past BACKUP_COUNT is deleted on rollover.

The file path no longer embeds the date — rotation is size-driven.
Callers that rely on per-date files should grep by timestamp inside
the file contents instead.
"""

import logging
import logging.handlers
import os
from typing import Optional

# ---------------------------------------------------------------------------
# Rotation defaults
# ---------------------------------------------------------------------------
# Plain integers so they're trivially overridable from the environment
# without import-time side effects on tests that monkey-patch os.environ.
DEFAULT_MAX_BYTES      = 10 * 1024 * 1024   # 10 MB
DEFAULT_BACKUP_COUNT   = 7
LOG_FILENAME           = "trading_agent.log"

# Sentinel attribute we attach to handlers we install so a re-entrant
# ``setup_logging`` call (Streamlit hot reload, test fixtures, ``from_env``
# called from inside an app that already initialised logging) can detect
# that initialisation already happened and short-circuit. Without this the
# user sees "Logging initialised — ..." printed twice on every Streamlit
# render.
_HANDLER_TAG = "_trading_agent_log_handler"


def _int_from_env(name: str, default: int) -> int:
    """Read an int env var with graceful fallback on parse errors."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return default


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
) -> logging.Logger:
    """
    Configure the root logger with console + size-rotated file output.

    Parameters
    ----------
    log_level    : INFO / DEBUG / WARNING / …
    log_dir      : directory for log files (created if missing)
    max_bytes    : rotation threshold per file (None → env/DEFAULT)
    backup_count : number of rollover files kept (None → env/DEFAULT)

    Returns
    -------
    The root logger.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    log_file_target = os.path.join(log_dir, LOG_FILENAME)

    # Idempotence guard — if our handlers are already attached to root and
    # pointing at the same file at the same level, this is a redundant call
    # (typical sources: Streamlit hot reload, ``TradingAgent.from_env``
    # called from a Streamlit app that already initialised logging). Return
    # the existing root logger without printing a second
    # "Logging initialised — ..." line.
    root = logging.getLogger()
    existing = [h for h in root.handlers if getattr(h, _HANDLER_TAG, False)]
    if existing:
        same_level = root.level == level
        same_file = any(
            isinstance(h, logging.handlers.RotatingFileHandler)
            and os.path.abspath(getattr(h, "baseFilename", "")) ==
                os.path.abspath(log_file_target)
            for h in existing
        )
        if same_level and same_file:
            return root

    os.makedirs(log_dir, exist_ok=True)
    log_file = log_file_target

    if max_bytes is None:
        max_bytes = _int_from_env("LOG_MAX_BYTES", DEFAULT_MAX_BYTES)
    if backup_count is None:
        backup_count = _int_from_env("LOG_BACKUP_COUNT", DEFAULT_BACKUP_COUNT)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — stderr by default
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    setattr(console, _HANDLER_TAG, True)

    # Rotating file handler — size-driven, bounded retention
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
        delay=False,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    setattr(file_handler, _HANDLER_TAG, True)

    root.setLevel(level)
    # Avoid duplicate handlers on re-init (Streamlit hot reload / test reuse).
    # Drop only handlers we own — leave anything pytest's caplog or a host
    # app installed in place so we don't break their log capture.
    root.handlers = [h for h in root.handlers
                     if not getattr(h, _HANDLER_TAG, False)]
    root.addHandler(console)
    root.addHandler(file_handler)

    # Silence chatty third-party libraries at INFO / DEBUG
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)

    root.info(
        "Logging initialised — level=%s, file=%s, rotate=%dMB × %d backups",
        log_level, log_file, max_bytes // (1024 * 1024), backup_count,
    )
    return root
