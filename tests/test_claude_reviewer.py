"""Skill 56 §3 — the daily review through one persistent `claude -p` session."""
import json
from types import SimpleNamespace

import pytest

from trading_agent import claude_reviewer as cr


@pytest.fixture(autouse=True)
def _paths(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "STATE_PATH", tmp_path / "reviewer_session.json")
    monkeypatch.setattr(cr, "WORKDIR", tmp_path / "wd")
    monkeypatch.setenv("CLAUDE_BIN", "/bin/claude")
    return tmp_path


ANSWER = {"observations": ["quiet day"], "preset_proposal": None, "watchlist_proposal": None,
          "digest_lines": ["no trades"], "confidence": 0.4}


def runner_returning(*answers):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        a = answers[min(len(calls), len(answers)) - 1]
        if isinstance(a, Exception):
            raise a
        return SimpleNamespace(returncode=0, stdout=json.dumps(a), stderr="")
    return run, calls


def ok(sid, result=ANSWER, context=3000):
    return {"type": "result", "is_error": False, "session_id": sid,
            "result": result if isinstance(result, str) else json.dumps(result),
            "usage": {"input_tokens": 2, "cache_creation_input_tokens": context - 2,
                      "cache_read_input_tokens": 0, "output_tokens": 400},
            "total_cost_usd": 0.02}


def flag(cmd, name):
    return cmd[cmd.index(name) + 1] if name in cmd else None


def test_first_review_starts_a_session_then_later_days_resume(_paths):
    run, calls = runner_returning(ok("S1"), ok("S1", context=6000))
    parsed, usage, err = cr.review("SYS", "day1", "2026-10-14", _paths, runner=run)
    assert parsed == ANSWER and err == "" and usage["context_tokens"] == 3000
    cmd = calls[0][0]
    assert flag(cmd, "--session-id") and "--resume" not in cmd
    assert flag(cmd, "--tools") == "" and "--strict-mcp-config" in cmd
    assert flag(cmd, "--system-prompt").startswith("SYS") and calls[0][1]["cwd"] == str(cr.WORKDIR)
    state = cr.load_state()
    sid = state["session_id"]
    assert state["reviews"] == 1
    cr.review("SYS", "day2", "2026-10-15", _paths, runner=run)
    assert flag(calls[1][0], "--resume") == sid
    state = cr.load_state()
    assert state["reviews"] == 2 and state["last_context_tokens"] == 6000


@pytest.mark.parametrize("state", [{"session_id": "S", "reviews": 20},
                                   {"session_id": "S", "reviews": 3, "last_context_tokens": 60000}])
def test_rotates_after_max_reviews_or_context(_paths, state):
    cr.save_state(state)
    (_paths / "2026-10-13.json").write_text(json.dumps({"output": {"observations": ["GLD drifting"]}}))
    run, calls = runner_returning(ok("NEW"))
    cr.review("SYS", "today", "2026-10-14", _paths, runner=run)
    cmd = calls[0][0]
    assert "--resume" not in cmd
    prompt = cmd[cmd.index("-p") + 1]
    assert prompt.startswith("Earlier reviews") and "GLD drifting" in prompt
    assert cr.load_state()["previous_session_id"] == "S"


def test_failed_resume_falls_back_to_a_new_session(_paths):
    cr.save_state({"session_id": "GONE", "reviews": 2})
    run, calls = runner_returning({"type": "result", "is_error": True, "result": "No conversation found"},
                                  ok("FRESH"))
    parsed, usage, err = cr.review("SYS", "today", "2026-10-14", _paths, runner=run)
    assert parsed == ANSWER and len(calls) == 2
    assert flag(calls[0][0], "--resume") == "GONE" and flag(calls[1][0], "--session-id")
    assert cr.load_state()["reviews"] == 1


def test_cli_missing_returns_error_without_state(_paths):
    run, _ = runner_returning(FileNotFoundError("claude"))
    parsed, usage, err = cr.review("SYS", "x", "2026-10-14", _paths, runner=run)
    assert parsed is None and "failed to run" in err and cr.load_state() == {}


def test_unparseable_answer_reports_text(_paths):
    run, _ = runner_returning(ok("S1", result="I cannot do that"))
    parsed, usage, err = cr.review("SYS", "x", "2026-10-14", _paths, runner=run)
    assert parsed is None and "unparseable" in err and usage["context_tokens"] == 3000


@pytest.mark.parametrize("text", ['{"a": 1}', '```json\n{"a": 1}\n```', 'Here:\n{"a": 1}\nDone.'])
def test_parse_result_json(text):
    assert cr.parse_result_json(text) == {"a": 1}


def test_daily_reviewer_uses_claude_cli_when_configured(monkeypatch, _paths):
    from trading_agent import daily_reviewer as dr
    monkeypatch.setenv("TRADING_AGENT_REVIEWER_LLM", "claude_cli")
    monkeypatch.setattr(cr, "review", lambda system, user, date, d: (
        {**ANSWER, "preset_proposal": {"field": "max_risk_pct", "proposed": 0.1}},
        {"session_id": "S", "context_tokens": 2500}, ""))
    ctx = dr.ReviewContext(review_date="2026-10-14", opens=[], closes=[], reject_reasons=[],
                           realized_pl=0.0, preset={}, watchlist=[], macro={}, recent_alerts=[])
    out = dr.call_llm(ctx)
    assert out.observations == ["quiet day"] and out.llm_usage["context_tokens"] == 2500
    assert out.preset_proposal is None          # disallowed field still filtered
