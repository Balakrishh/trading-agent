"""SDD gate: every `except Exception` in production code must page.

Skill 34 §3.3 documents the contract: a bare `except Exception` block
in production code is a place where a bug can silently hide. The cure
is one extra line — `self._exception_monitor.record(...)` (or
`get_global_monitor().record(...)` for pre-agent modules) — that
paged the operator on Schwab token expiry, Telegram TypeError, and
the four other multi-day-undetected bugs that motivated skill 34.

Pre-2026-05-22 we relied on reviewer eyes to enforce this. That works
for the first commit; not the hundredth. This gate AST-walks every
production module and fails if an `except Exception` (or `except
BaseException`) handler doesn't either:

  * call `.record(` somewhere in its body, OR
  * be tagged with a `# noqa: skill-34-exempt — <reason>` comment on
    the except line.

The exemption mechanism covers the ~25 legitimately-silent paths:
regime classify falling back to neutral, sentiment timeout, journal-
read failures inside dedup helpers, etc. Each exemption is a deliberate
operator choice that survives code review; flipping a real failure
back to silent requires an explicit annotation, not just deletion.

Exit codes:
  0 — every except Exception block either records or is exempt
  1 — at least one bare except Exception block is unhandled
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Production modules subject to the gate. Tests + scripts are excluded
# — they may legitimately catch broad exceptions for assertion paths.
PRODUCTION_GLOBS: Tuple[str, ...] = (
    "trading_agent/agent.py",
    "trading_agent/executor.py",
    "trading_agent/strategy.py",
    "trading_agent/chain_scanner.py",
    "trading_agent/decision_engine.py",
    "trading_agent/risk_manager.py",
    "trading_agent/market_data.py",
    "trading_agent/market_data_schwab.py",
    "trading_agent/market_data_yahoo.py",
    "trading_agent/market_data_factory.py",
    "trading_agent/position_monitor.py",
    "trading_agent/journal_kb.py",
    "trading_agent/journal_reader.py",
    "trading_agent/close_event_collaborators.py",
    "trading_agent/telegram_notifier.py",
    "trading_agent/regime.py",
    "trading_agent/multi_tf_regime.py",
    "trading_agent/exception_monitor.py",   # self-instrumentation included
    "trading_agent/schwab_oauth.py",
    "trading_agent/order_tracker.py",
)

EXEMPT_TAG = "skill-34-exempt"


# ---------------------------------------------------------------------------
# AST walker
# ---------------------------------------------------------------------------


def _records_in_body(body: List[ast.stmt]) -> bool:
    """Returns True if any node in `body` calls `.record(...)`.

    Matches both `self._exception_monitor.record(...)` and
    `mon.record(...)` (after the `mon = get_global_monitor()`
    indirection used by pre-agent modules)."""
    for node in body:
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "record"):
                return True
    return False


def _exempt_at_line(source_lines: List[str], lineno: int) -> bool:
    """Returns True if the except clause at `lineno` carries the
    exemption tag in a trailing comment (same line or previous line)."""
    # Same-line comment:  except Exception:  # noqa: skill-34-exempt — <reason>
    if lineno <= len(source_lines):
        line = source_lines[lineno - 1]
        if EXEMPT_TAG in line:
            return True
    # Previous-line comment (operator put the rationale above).
    if lineno >= 2 and lineno - 2 < len(source_lines):
        prev = source_lines[lineno - 2].lstrip()
        if prev.startswith("#") and EXEMPT_TAG in prev:
            return True
    return False


def scan_file(path: Path) -> List[Tuple[int, str]]:
    """Returns list of (lineno, message) for offending except blocks."""
    text = path.read_text(encoding="utf-8")
    source_lines = text.splitlines()
    tree = ast.parse(text)
    offenders: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        # Bare `except:` or `except Exception:` or `except BaseException:`
        exc_type = node.type
        if exc_type is None:
            name = "<bare except>"
        elif isinstance(exc_type, ast.Name):
            name = exc_type.id
        elif isinstance(exc_type, ast.Tuple):
            # `except (A, B):` — only flag if it contains Exception/BaseException
            ids = [e.id for e in exc_type.elts if isinstance(e, ast.Name)]
            if not any(i in ("Exception", "BaseException") for i in ids):
                continue
            name = " | ".join(ids)
        else:
            continue
        if name not in ("Exception", "BaseException", "<bare except>"):
            continue
        # Exempt?
        if _exempt_at_line(source_lines, node.lineno):
            continue
        # Has a .record(...) call somewhere in the body?
        if _records_in_body(node.body):
            continue
        offenders.append((node.lineno, name))
    return offenders


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    total_offenders = 0
    total_files_scanned = 0
    files_with_offenders = 0
    for rel in PRODUCTION_GLOBS:
        path = repo_root / rel
        if not path.is_file():
            continue
        total_files_scanned += 1
        offenders = scan_file(path)
        if not offenders:
            continue
        files_with_offenders += 1
        total_offenders += len(offenders)
        print(f"FAIL {rel}: {len(offenders)} unhandled except Exception block(s)")
        for lineno, name in offenders:
            print(f"     line {lineno}: except {name}  "
                  f"— must call .record(...) OR carry "
                  f"`# noqa: {EXEMPT_TAG} — <reason>` annotation")

    print(
        f"\nSDD ExceptionMonitor coverage — scanned "
        f"{total_files_scanned} production module(s), "
        f"{total_offenders} unhandled block(s) across "
        f"{files_with_offenders} file(s)."
    )
    if total_offenders == 0:
        print("\nOK   every except Exception block either records or is exempt.")
        return 0
    print(
        f"\nFAIL — fix by either calling "
        f"self._exception_monitor.record(...) "
        f"(or get_global_monitor().record(...)) inside the handler, "
        f"OR annotating the except line:\n"
        f"    except Exception as exc:  # noqa: {EXEMPT_TAG} — "
        f"<one-line rationale>"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
