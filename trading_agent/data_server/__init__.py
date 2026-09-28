"""
data_server — local HTTP server that exposes SchwabMarketDataProvider
as read-only JSON endpoints. Skill 47.

READ-ONLY BY CONSTRUCTION.

This module and its siblings deliberately do NOT import:
  * trading_agent.executor      — order placement
  * trading_agent.journal_kb    — journal writes/reads
  * anything named submit_order, place_order, TradingClient

A conformance test (test_skill_47_readonly_no_forbidden_imports)
greps every file in this package for those tokens and fails CI on
any hit. The invariant is not just documentation — it's mechanical.
"""

from __future__ import annotations

# Sentinel constant the invariant scanner + conformance test both key on.
# Do not remove without updating skill 47 §4.
SERVER_READ_ONLY: bool = True

__all__ = ["SERVER_READ_ONLY"]
