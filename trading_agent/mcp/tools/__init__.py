"""Tool implementations for the trading-agent MCP surface (skill 48).

Every callable in this package must be a **read-only** wrapper over an
existing repo API. The conformance test walks each module's AST and
fails CI on any import of the executor, order-submission primitives,
or ``pending_orders``.
"""
