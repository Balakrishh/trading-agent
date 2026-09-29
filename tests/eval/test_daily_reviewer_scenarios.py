"""Prompt evals for the daily reviewer (skill 56 Phase D).

Two modes:

- **Offline (default, always runs in CI).** Exercises the parser +
  rendering path over each scenario's `expected` claims that don't
  require the LLM: forbidden-field filtering, cold-start labeling,
  digest structure.

- **Online (opt-in).** Set ``EVAL_DAILY_REVIEWER_ONLINE=true`` to
  actually send each scenario's context to ``LLMClient.chat_json``
  and assert the parsed output matches ``expected``. Costs one LLM
  call per scenario.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest


SCENARIOS_PATH = (Path(__file__).resolve().parents[2]
                   / "evals" / "daily_reviewer" / "scenarios.jsonl")


def _iter_scenarios() -> Iterator[Dict[str, Any]]:
    with open(SCENARIOS_PATH) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _online() -> bool:
    return os.environ.get(
        "EVAL_DAILY_REVIEWER_ONLINE", "").strip().lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Offline path — no LLM
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", list(_iter_scenarios()),
                         ids=lambda s: s["id"])
def test_offline_scenario(scenario):
    """Every scenario must at minimum: load, have a `context` +
    `expected` block, and pass its offline-verifiable claims.
    """
    assert "context" in scenario
    assert "expected" in scenario
    exp = scenario["expected"]

    # forbidden_field_filter — parser must drop hallucinations.
    if exp.get("test_type") == "forbidden_field_filter":
        from trading_agent.daily_reviewer import _parse_llm_output
        # Simulate a bad LLM response that names a forbidden field.
        raw = {
            "observations": ["test"],
            "preset_proposal": {
                "field": "max_risk_pct",
                "current": 0.02,
                "proposed": 0.05,
                "reason": "hallucinated",
                "confidence": 0.9,
            },
            "digest_lines": [],
            "confidence": 0.5,
        }
        out = _parse_llm_output(raw)
        assert out.preset_proposal is None, (
            f"{scenario['id']}: forbidden field survived parsing")
        return

    # digest_contains — render the digest against a synthetic output.
    if "digest_contains" in exp:
        from trading_agent.daily_reviewer import (
            render_telegram_digest, ReviewContext, ReviewOutput,
        )
        ctx = _ctx_from_scenario(scenario)
        digest = render_telegram_digest(
            ctx, ReviewOutput([], None, None, [], 0.0, ""))
        assert exp["digest_contains"] in digest, (
            f"{scenario['id']}: digest missing {exp['digest_contains']!r}")


def _ctx_from_scenario(scenario):
    from trading_agent.daily_reviewer import ReviewContext
    c = scenario["context"]
    return ReviewContext(
        review_date="2026-09-29",
        opens=c.get("opens", []),
        closes=c.get("closes", []),
        reject_reasons=c.get("reject_reasons_top5", []),
        realized_pl=float(c.get("realized_pl_today", 0.0)),
        preset=c.get("current_preset", {}),
        watchlist=c.get("watchlist", []),
        macro=c.get("macro", {}),
        recent_alerts=c.get("recent_alerts", []),
        cold_start=bool(c.get("cold_start", False)),
    )


# ---------------------------------------------------------------------------
# Online path — real LLM (opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _online(),
                    reason="EVAL_DAILY_REVIEWER_ONLINE not set")
@pytest.mark.parametrize("scenario", list(_iter_scenarios()),
                         ids=lambda s: s["id"])
def test_online_scenario(scenario):
    """Send the scenario to the live LLM and check ``expected`` claims
    against the parsed output. Skipped by default; opt-in via env var.
    """
    from trading_agent.daily_reviewer import (
        _SYSTEM_PROMPT, _build_user_prompt, _parse_llm_output,
    )
    from trading_agent.llm_client import LLMClient

    ctx = _ctx_from_scenario(scenario)
    client = LLMClient()
    result = client.chat_json(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(ctx)},
        ],
        temperature=0.2,
    )
    assert result, f"{scenario['id']}: LLM returned nothing"
    out = _parse_llm_output(result)

    exp = scenario["expected"]
    if exp.get("preset_proposal_field") is None:
        assert out.preset_proposal is None, (
            f"{scenario['id']}: expected no proposal, got {out.preset_proposal}")
    elif exp.get("preset_proposal_field"):
        assert out.preset_proposal is not None
        assert out.preset_proposal["field"] == exp["preset_proposal_field"], (
            f"{scenario['id']}: wrong field")
        if exp.get("direction") == "increase":
            assert out.preset_proposal["proposed"] > out.preset_proposal["current"]
        elif exp.get("direction") == "decrease":
            assert out.preset_proposal["proposed"] < out.preset_proposal["current"]
        if exp.get("min_confidence"):
            assert out.preset_proposal["confidence"] >= exp["min_confidence"]

    if exp.get("watchlist_drops_contains"):
        assert out.watchlist_proposal is not None
        assert exp["watchlist_drops_contains"] in (
            out.watchlist_proposal.get("drops") or [])

    if exp.get("watchlist_drops_empty"):
        drops = (out.watchlist_proposal or {}).get("drops", [])
        assert not drops, f"{scenario['id']}: expected empty drops, got {drops}"
