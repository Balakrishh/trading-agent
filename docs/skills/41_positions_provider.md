# Positions provider — uniform holdings input

> **One-line summary:** Abstract base + three implementations that hand the long-term evaluator a normalised list of `Position` objects (stock OR option, both sources). Decouples the evaluator from any specific brokerage so we can drive it with a manual paste this session, plug Alpaca live next session, and plug Schwab live the session after that — all without touching the evaluator.
> **Source of truth:** [`trading_agent/positions_provider.py`](../../trading_agent/positions_provider.py).
> **Phase:** 2  •  **Group:** data_quality
> **Depends on:** none (intentionally — this is a pure data port).
> **Consumed by:** `trading_agent/long_term_evaluator.py` (skill 40), `trading_agent/streamlit/long_term_evaluator_ui.py` (renders the portfolio snapshot panel).

---

## 1. Theory & Objective

The evaluator (skill 40) needs to know what the operator already holds before it can recommend anything sensible. "Already holds" can come from three places:

1. **Manual paste** — operator drops a JSON blob into a Streamlit textarea. No API auth required. Useful when prototyping, when offline, or when the operator wants to evaluate a hypothetical portfolio.
2. **Alpaca paper** — the same Alpaca account the credit-spread agent uses. Reuses existing `alpaca-py` plumbing. Useful for paper-testing the evaluator end-to-end with the holdings the operator replicates into the paper book.
3. **Schwab brokerage** — live real-money positions via Schwab Trader API's `/trader/v1/accounts/{accountId}/positions`. Requires the trading scope of the Schwab OAuth flow (the market-data scope the repo uses today is insufficient). This is the eventual production source.

Rather than scatter brokerage-specific calls through the evaluator, this skill defines a single `PositionsProvider` abstract base with one method (`snapshot() -> List[Position]`) and a tightly normalised `Position` dataclass. The evaluator sees only `Position`. The three implementations live behind the same interface; the Streamlit panel picks one via a sidebar radio.

Separating the holdings source from the evaluator also makes paper-testing trivial: replicate your Schwab positions into Alpaca, click `Alpaca paper`, and you can validate the recommendations on the same book without putting real money in motion.

## 2. Mathematical Formula

N/A — pure data structure.

## 3. Reference Python Implementation

### 3.1 `Position` dataclass

```python
# trading_agent/positions_provider.py
@dataclass(frozen=True)
class Position:
    """One holding, brokerage-agnostic.

    Stock positions: kind == "stock", occ_symbol == "", side == "long".
    Option positions: kind == "option", occ_symbol set (21-char OCC), side ∈ {"long","short"}.
    """
    ticker: str             # underlying, uppercased
    qty: int                # shares (stock) or contracts (option); always positive
    avg_cost: float         # per-share for stock, per-contract debit/credit for option
    kind: str               # "stock" | "option"
    occ_symbol: str = ""    # blank for stock
    side: str = "long"      # "long" | "short"
    account: str = ""       # free-text tag: "schwab_live" | "alpaca_paper" | "manual"
    notes: str = ""
```

### 3.2 `PositionsProvider` abstract base

The ABC declares one abstract method:

```python
# trading_agent/positions_provider.py
@abc.abstractmethod
def snapshot(self) -> List[Position]:
    """Returns the current list of holdings. Always returns a fresh list."""
```

Plus a default `source_label` property for the Streamlit header. Implementations MUST be idempotent and side-effect-free — calling `snapshot()` twice in quick succession must not mutate the underlying brokerage. The evaluator calls `snapshot()` once per UI refresh; the Streamlit panel may call it more aggressively when the operator clicks Refresh.

Implementations are read-only. There is no `add_position` or `close_position` on this ABC — order placement is a separate layer (Phase 5). Positions are eventually re-read on the next `snapshot()` call after the broker fills.

### 3.3 `ManualPositionsProvider` (this session)

Parses a JSON blob into a list of `Position` objects:

```python
# trading_agent/positions_provider.py
@classmethod
def from_json_text(cls, text: str) -> "ManualPositionsProvider":
    """Parse a JSON array of position dicts into a provider.

    Raises ``ValueError`` with the offending row included when a
    dict fails validation. The Streamlit panel catches this and
    surfaces the error inline without clearing the textarea so the
    operator can fix it in place.
    """
```

The constructor accepts an already-built list of `Position` objects; `snapshot()` returns a defensive copy so the evaluator can sort/filter without mutating the operator's pasted state. `source_label` returns `"Manual"` for the Streamlit header.

`_position_from_dict` (module-level helper) validates the shape (`ticker`, `qty`, `avg_cost`, `kind` required) and raises `ValueError` with a useful message on malformed rows. The Streamlit panel catches the exception and shows the operator the offending row.

### 3.4 `AlpacaPositionsProvider` (next session — stub here for design completeness)

```python
# trading_agent/positions_provider.py — next session
class AlpacaPositionsProvider(PositionsProvider):
    """Pulls positions from the paper Alpaca account.

    Reuses the same TradingClient the credit-spread executor uses; we
    construct a thin wrapper rather than depending on the executor so
    the evaluator's import graph stays narrow.
    """
    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self._client = alpaca_py.trading.client.TradingClient(api_key, secret_key, paper=True)

    def snapshot(self) -> List[Position]:
        raw = self._client.get_all_positions()
        return [_position_from_alpaca(p) for p in raw]
```

### 3.5 `SchwabPositionsProvider` (next session — stub here for design completeness)

```python
# trading_agent/positions_provider.py — next session
class SchwabPositionsProvider(PositionsProvider):
    """Pulls positions from a Schwab brokerage account via Trader API.

    Requires the `trading` OAuth scope (not the `marketdata` scope the
    repo uses today). See `schwab_oauth login --scope trading` to
    obtain tokens with the correct scope.
    """
    def __init__(self, account_id: str, oauth_session):
        self._account_id = account_id
        self._oauth = oauth_session

    def snapshot(self) -> List[Position]:
        resp = self._oauth.get(
            f"/trader/v1/accounts/{self._account_id}/positions"
        ).json()
        return [_position_from_schwab(p) for p in resp.get("positions", [])]
```

## 4. Edge Cases / Guardrails

- **Defensive copy on snapshot.** `ManualPositionsProvider.snapshot()` returns a fresh list so the evaluator can sort/filter without mutating the operator's pasted state. Same contract applies to the Alpaca/Schwab impls.
- **Empty list ≠ error.** A brand-new operator with no holdings yet still gets `snapshot() → []`. The evaluator's recommendations degrade gracefully: portfolio snapshot is empty, manage-existing is empty, income-overlay produces zero rows (no held stock), entry-vehicle suggestions fire on every watchlist ticker.
- **Mixed qty signs are forbidden.** `Position.qty` is always positive; `side` carries the long/short signal. This avoids the perennial broker-API confusion where `qty < 0` sometimes means "short" and sometimes means "transfer-out adjustment". Conformance: `test_skill_41_qty_is_always_positive`.
- **Malformed JSON paste.** `ManualPositionsProvider.from_json_text` re-raises with the offending row included. The Streamlit panel surfaces the error inline without clearing the textarea so the operator can fix it in place.
- **OCC symbol validation for options.** When `kind == "option"`, the OCC symbol must be 21 characters and pass the YYMMDD strike-format check used elsewhere in the repo. Malformed OCC strings raise `ValueError`.
- **Account-tag preserved end-to-end.** The optional `account` tag flows into `Recommendation.metrics` so the Streamlit panel can show which account a recommendation is for when the operator has both Schwab + Alpaca selected (skill 40 §3.4 "Both" mode, next session).
- **No write methods.** This provider is read-only by design. There is no `add_position()` or `close_position()` on the ABC. The order-placement layer (Phase 5) talks to brokerage APIs directly; positions are eventually re-read on the next `snapshot()`.
- **Idempotence.** Repeated calls to `snapshot()` within a few seconds must return equal lists (modulo broker-side fills). Conformance: `test_skill_41_snapshot_is_idempotent`.

## 5. Cross-References

- `40_long_term_options_evaluator.md` — the sole consumer; the evaluator's `recommend()` method takes a `PositionsProvider`.
- `19_journal_schema.md` — when an option position is surfaced in the manage-existing section, its current state is journalled as `lt_position_surveyed` action (next session adds this action string).
- `16_market_data_provider_routing.md` — same abstraction pattern (one ABC, multiple impls behind it). The evaluator's positions story mirrors the market-data story.

---

*Last verified against repo HEAD on 2026-06-16.*
