"""claude_reviewer.py — the daily review through the logged-in ``claude`` CLI.

Skill 56 §3 (backlog A6, 2026-10-14). Ollama was not running on the Pi, so
every review since 2026-10-05 was empty. This runs the same prompt through
``claude -p`` (the operator's Claude Code login — no API key) and keeps
**one persistent session** across days, so each review can compare today
with what it saw and said before.

* The CLI runs from its own directory (``~/.trading-agent-reviewer``) with
  ``--system-prompt`` replacing Claude Code's, no tools (``--tools ""``), no
  MCP servers and no settings files: it reads the JSON we send and answers
  in JSON, nothing else. Overhead ≈ 600 tokens.
* The session id lives in ``trade_journal/reviewer_session.json``. Day 1
  starts it with ``--session-id``; later days ``--resume`` it.
* A resumed session re-reads its whole history, so input grows ~2.5k tokens
  a day. The session rotates after ``MAX_REVIEWS`` reviews or once a call's
  context passes ``MAX_CONTEXT_TOKENS``; the new session opens with the last
  ``CARRY_OVER_DAYS`` days' observations so nothing important is lost.
* A failed resume (session missing) starts a fresh session once.

Never trades: like ``daily_reviewer`` it imports nothing that can place an
order (skill 56 AST check).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("trading_agent.claude_reviewer")

STATE_PATH = Path(os.environ.get("TRADING_AGENT_REVIEWER_SESSION",
                                 "trade_journal/reviewer_session.json"))
WORKDIR = Path(os.environ.get("TRADING_AGENT_REVIEWER_WORKDIR",
                              str(Path.home() / ".trading-agent-reviewer")))
MAX_REVIEWS = int(os.environ.get("TRADING_AGENT_REVIEWER_MAX_REVIEWS", "20"))
MAX_CONTEXT_TOKENS = int(os.environ.get("TRADING_AGENT_REVIEWER_MAX_CONTEXT", "60000"))
CARRY_OVER_DAYS = 5
TIMEOUT_S = 300

CONTINUITY_NOTE = (
    " You keep one conversation across trading days: compare today with "
    "earlier days you reviewed, say when a pattern repeats or changes, and "
    "do not repeat a proposal the operator has already seen unless the "
    "evidence got stronger."
)


def claude_bin() -> str:
    return (os.environ.get("CLAUDE_BIN")
            or shutil.which("claude")
            or str(Path.home() / ".local" / "bin" / "claude"))


def load_state(path: Optional[Path] = None) -> Dict[str, Any]:
    try:
        return json.loads((path or STATE_PATH).read_text())
    except (OSError, ValueError):
        return {}


def save_state(state: Dict[str, Any], path: Optional[Path] = None) -> None:
    p = path or STATE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(p)


def context_tokens(usage: Dict[str, Any]) -> int:
    """Everything the model read on a call (fresh + cached input)."""
    return int(sum(int(usage.get(k) or 0) for k in
                   ("input_tokens", "cache_read_input_tokens",
                    "cache_creation_input_tokens")))


def should_rotate(state: Dict[str, Any]) -> bool:
    if not state.get("session_id"):
        return True
    return (int(state.get("reviews", 0)) >= MAX_REVIEWS
            or int(state.get("last_context_tokens", 0)) >= MAX_CONTEXT_TOKENS)


def build_cmd(prompt: str, system: str, session_id: str, resume: bool) -> List[str]:
    return [claude_bin(), "-p", prompt, "--output-format", "json",
            "--tools", "", "--strict-mcp-config", "--setting-sources", "",
            "--system-prompt", system,
            "--resume" if resume else "--session-id", session_id]


def parse_result_json(text: str) -> Optional[Dict[str, Any]]:
    """The model's answer as a dict: plain JSON, fenced JSON, or the first
    {...} block. None when nothing parses."""
    t = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
    for cand in ([fence.group(1)] if fence else []) + [t]:
        try:
            out = json.loads(cand)
            return out if isinstance(out, dict) else None
        except ValueError:
            pass
    i, j = t.find("{"), t.rfind("}")
    if 0 <= i < j:
        try:
            out = json.loads(t[i:j + 1])
            return out if isinstance(out, dict) else None
        except ValueError:
            return None
    return None


def carry_over(reviews_dir: Path, before: str, days: int = CARRY_OVER_DAYS) -> str:
    """Observations of the last ``days`` reviews before ``before``, for the
    first prompt of a new session."""
    lines = []
    for fp in sorted(reviews_dir.glob("*.json"))[::-1]:
        if fp.stem >= before:
            continue
        try:
            obs = (json.loads(fp.read_text()).get("output") or {}).get("observations") or []
        except (OSError, ValueError):
            continue
        if obs:
            lines.append(f"{fp.stem}: " + " | ".join(str(o) for o in obs[:5]))
        if len(lines) >= days:
            break
    if not lines:
        return ""
    return ("Earlier reviews (oldest last), carried over from the previous "
            "session:\n" + "\n".join(lines) + "\n\n")


def _call(prompt: str, system: str, session_id: str, resume: bool,
          runner: Callable[..., Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    try:
        proc = runner(build_cmd(prompt, system, session_id, resume), cwd=str(WORKDIR),
                      capture_output=True, text=True, timeout=TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"claude CLI failed to run: {exc}"
    try:
        out = json.loads(proc.stdout or "")
    except ValueError:
        return None, f"claude CLI exit {proc.returncode}: {(proc.stderr or proc.stdout or '')[:300]}"
    if out.get("is_error"):
        return None, f"claude CLI error: {str(out.get('result'))[:300]}"
    return out, ""


def review(system: str, user_prompt: str, review_date: str, reviews_dir: Path,
           runner: Callable[..., Any] = subprocess.run
           ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], str]:
    """Run one review. Returns (parsed JSON answer or None, usage record, error)."""
    state = load_state()
    system = system + CONTINUITY_NOTE
    resume = not should_rotate(state)
    for attempt in (1, 2):
        if resume:
            sid, prompt = state["session_id"], user_prompt
        else:
            sid = str(uuid.uuid4())
            prompt = carry_over(reviews_dir, review_date) + user_prompt
        out, err = _call(prompt, system, sid, resume, runner)
        if out is not None or not resume or attempt == 2:
            break
        log.warning("Resuming reviewer session %s failed (%s) — starting a new one", sid, err)
        resume = False
    if out is None:
        return None, {}, err
    usage = out.get("usage") or {}
    record = {"session_id": out.get("session_id") or sid, "resumed": resume,
              "context_tokens": context_tokens(usage),
              "output_tokens": int(usage.get("output_tokens") or 0),
              "cost_usd_equivalent": out.get("total_cost_usd")}
    now = datetime.now(timezone.utc).isoformat()
    if resume:
        state["reviews"] = int(state.get("reviews", 0)) + 1
    else:
        state = {"session_id": record["session_id"], "started_at": now, "reviews": 1,
                 "previous_session_id": state.get("session_id")}
    state.update(last_review_date=review_date, last_run_at=now,
                 last_context_tokens=record["context_tokens"],
                 last_output_tokens=record["output_tokens"])
    save_state(state)
    parsed = parse_result_json(str(out.get("result") or ""))
    if parsed is None:
        return None, record, f"unparseable answer: {str(out.get('result'))[:300]}"
    return parsed, record, ""
