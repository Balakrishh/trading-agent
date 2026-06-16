"""positions_provider.py — uniform holdings input for the long-term evaluator.

Skill: ``docs/skills/41_positions_provider.md``.

The long-term evaluator (skill 40) needs to know what the operator already
holds before it can recommend anything sensible. Holdings can come from
three places:

  1. **Manual paste** — operator drops a JSON blob into a Streamlit
     textarea. No API auth required. Lives in this file as
     ``ManualPositionsProvider``.
  2. **Alpaca paper** — reuses the executor's TradingClient. Stub here
     for the design contract; the impl lands next session.
  3. **Schwab brokerage** — Trader API ``/accounts/{id}/positions``.
     Requires the ``trading`` OAuth scope (not the ``marketdata`` scope
     the repo uses today). Stub here for the design contract.

Rather than scatter brokerage-specific calls through the evaluator, this
module defines a single ``PositionsProvider`` abstract base with one
method (``snapshot() -> List[Position]``) plus a tightly normalised
``Position`` dataclass. The evaluator sees only ``Position``. The three
implementations live behind the same interface; the Streamlit panel
picks one via a sidebar radio.

Separating the holdings source from the evaluator also makes
paper-testing trivial: replicate Schwab positions into Alpaca, click
"Alpaca paper" in the sidebar, and validate the recommendations on the
same book without putting real money in motion.
"""

from __future__ import annotations

import abc
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position — brokerage-agnostic holding record
# ---------------------------------------------------------------------------

VALID_KINDS = frozenset({"stock", "option"})
VALID_SIDES = frozenset({"long", "short"})


@dataclass(frozen=True)
class Position:
    """One holding, brokerage-agnostic.

    Stock positions: ``kind == "stock"``, ``occ_symbol == ""``,
    ``side == "long"``.

    Option positions: ``kind == "option"``, ``occ_symbol`` set to the
    21-char OCC symbol, ``side ∈ {"long","short"}``.

    Invariants enforced in ``__post_init__``:

    * ``ticker`` is non-empty and uppercased
    * ``qty`` is strictly positive (the ``side`` field carries the sign)
    * ``avg_cost`` is non-negative
    * ``kind`` is in :data:`VALID_KINDS`
    * ``side`` is in :data:`VALID_SIDES`
    * When ``kind == "option"`` the OCC symbol must be exactly 21
      characters (the standard OCC format ``RRRRRRYYMMDDCPSSSSSSSSS``).
    * When ``kind == "stock"`` the OCC symbol must be blank.
    """

    ticker: str
    qty: int
    avg_cost: float
    kind: str
    occ_symbol: str = ""
    side: str = "long"
    account: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        # frozen=True forbids attribute assignment; use object.__setattr__
        ticker = (self.ticker or "").strip().upper()
        if not ticker:
            raise ValueError("Position.ticker must be a non-empty string.")
        object.__setattr__(self, "ticker", ticker)

        if not isinstance(self.qty, int) or self.qty <= 0:
            raise ValueError(
                f"Position.qty must be a positive int "
                f"(got {self.qty!r}); use the `side` field for long/short."
            )

        if self.avg_cost < 0:
            raise ValueError(
                f"Position.avg_cost must be ≥ 0 (got {self.avg_cost!r})."
            )

        if self.kind not in VALID_KINDS:
            raise ValueError(
                f"Position.kind must be one of {sorted(VALID_KINDS)} "
                f"(got {self.kind!r})."
            )
        if self.side not in VALID_SIDES:
            raise ValueError(
                f"Position.side must be one of {sorted(VALID_SIDES)} "
                f"(got {self.side!r})."
            )

        if self.kind == "option":
            if len(self.occ_symbol) != 21:
                raise ValueError(
                    f"Option position must carry a 21-char OCC symbol "
                    f"(got {self.occ_symbol!r}, len={len(self.occ_symbol)})."
                )
        else:  # stock
            if self.occ_symbol:
                raise ValueError(
                    "Stock positions must have empty occ_symbol "
                    f"(got {self.occ_symbol!r})."
                )
            if self.side != "long":
                # Short-stock positions are out of scope for this evaluator.
                # We don't recommend covered-call overlays on short stock.
                raise ValueError(
                    "Short-stock positions are not supported by this "
                    "evaluator (kind=stock requires side=long)."
                )


# ---------------------------------------------------------------------------
# PositionsProvider — abstract base
# ---------------------------------------------------------------------------

class PositionsProvider(abc.ABC):
    """Returns a normalised list of ``Position`` objects.

    Implementations MUST be **idempotent** and **side-effect-free**:
    calling ``snapshot()`` twice in quick succession must not mutate the
    underlying brokerage. The evaluator calls ``snapshot()`` once per UI
    refresh; the Streamlit panel may call it more aggressively when the
    operator clicks the Refresh button.

    Implementations are **read-only**. There is no ``add_position`` or
    ``close_position`` on this ABC — order placement is a separate
    layer (Phase 5, dedicated session). Positions are eventually
    re-read on the next ``snapshot()`` call after the broker fills.
    """

    @abc.abstractmethod
    def snapshot(self) -> List[Position]:
        """Return the current list of holdings. Always returns a fresh list."""

    @property
    def source_label(self) -> str:
        """Human-readable name for the Streamlit header. Override in impls."""
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# ManualPositionsProvider — JSON paste, this session's impl
# ---------------------------------------------------------------------------

class ManualPositionsProvider(PositionsProvider):
    """Reads positions from a JSON blob (string or list-of-dicts).

    Expected JSON shape::

        [
          {"ticker": "AAPL", "qty": 100, "avg_cost": 215.40, "kind": "stock"},
          {"ticker": "NVDA", "qty": 200, "avg_cost": 132.10, "kind": "stock"},
          {"ticker": "MSFT", "qty": 1,   "avg_cost": 18.50,  "kind": "option",
           "occ_symbol": "MSFT  270115C00400000", "side": "long"}
        ]

    The OCC-21 symbol for options uses the standard format
    ``RRRRRRYYMMDDCPSSSSSSSSS`` (6-char ticker, padded with spaces;
    YYMMDD expiration; C/P; 8-digit strike × 1000). Both Alpaca and
    Schwab use this format internally so a position pasted in this
    shape is directly usable downstream.
    """

    def __init__(self, positions: Optional[List[Position]] = None) -> None:
        self._positions: List[Position] = list(positions or [])

    @property
    def source_label(self) -> str:
        return "Manual"

    def snapshot(self) -> List[Position]:
        # Defensive copy so the evaluator can sort/filter without
        # mutating the operator's pasted state.
        return list(self._positions)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_json_text(cls, text: str) -> "ManualPositionsProvider":
        """Parse a JSON array of position dicts into a provider.

        Raises ``ValueError`` with the offending row included when a
        dict fails validation. The Streamlit panel catches this and
        surfaces the error inline without clearing the textarea so the
        operator can fix it in place.
        """
        text = (text or "").strip()
        if not text:
            return cls(positions=[])

        # Tolerate pastes that are a bare comma-separated list of objects
        # (Schwab's portfolio export sometimes ships without the outer
        # ``[ ... ]`` brackets when the operator copies only the rows).
        # We only auto-wrap when the text clearly contains MULTIPLE
        # objects — a single ``{...}`` paste keeps the original
        # "expected array at root" error so the operator notices.
        if not text.startswith("[") and _looks_like_object_list(text):
            text = "[" + text + "]"

        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Holdings JSON is not parseable: {exc.msg} "
                f"(line {exc.lineno}, col {exc.colno})."
            ) from exc

        if not isinstance(raw, list):
            raise ValueError(
                f"Expected JSON array at root, got {type(raw).__name__}."
            )

        positions: List[Position] = []
        for idx, row in enumerate(raw):
            if not isinstance(row, dict):
                raise ValueError(
                    f"Row {idx}: expected an object, got {type(row).__name__}."
                )
            # Auto-detect Schwab portfolio-export shape and translate.
            if _is_schwab_export_row(row):
                translated = _schwab_row_to_canonical(row)
                if translated is None:
                    # Cash / summary row — skip silently, not an error.
                    continue
                row = translated
            try:
                positions.append(_position_from_dict(row))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Row {idx} ({row!r}): {exc}") from exc
        return cls(positions=positions)

    @classmethod
    def from_dicts(cls, rows: List[Dict[str, Any]]) -> "ManualPositionsProvider":
        """Construct from already-parsed dicts (skips the JSON step)."""
        return cls(positions=[_position_from_dict(r) for r in rows])


def _position_from_dict(row: Dict[str, Any]) -> Position:
    """Validate one row dict and return a ``Position``.

    Required keys: ``ticker``, ``qty``, ``avg_cost``, ``kind``.
    Optional keys: ``occ_symbol``, ``side``, ``account``, ``notes``.

    Raises ``KeyError`` for missing required keys; ``ValueError`` /
    ``TypeError`` for malformed values (delegates to ``Position``'s own
    ``__post_init__`` invariants for shape checks).
    """
    required = ("ticker", "qty", "avg_cost", "kind")
    for key in required:
        if key not in row:
            raise KeyError(f"missing required key {key!r}")

    qty = row["qty"]
    if isinstance(qty, float):
        if qty != int(qty):
            raise ValueError(
                f"qty must be a whole number "
                f"(got {qty!r} — fractional shares unsupported)."
            )
        qty = int(qty)
    elif not isinstance(qty, int):
        raise TypeError(f"qty must be int (got {type(qty).__name__})")

    try:
        avg_cost = float(row["avg_cost"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"avg_cost must be numeric: {exc}") from exc

    return Position(
        ticker=str(row["ticker"]),
        qty=qty,
        avg_cost=avg_cost,
        kind=str(row["kind"]),
        occ_symbol=str(row.get("occ_symbol", "")),
        side=str(row.get("side", "long")),
        account=str(row.get("account", "manual")),
        notes=str(row.get("notes", "")),
    )


# ---------------------------------------------------------------------------
# Schwab portfolio-export auto-detect (skill 41 §4)
# ---------------------------------------------------------------------------
# Schwab's "Export Positions" download in the brokerage UI emits rows shaped:
#
#   {
#     "Symbol": "AMZN",
#     "Description": "AMAZON.COM INC",
#     "Qty (Quantity)": "9",
#     "Price": "248.07",
#     "Cost Basis": "$1,968.47",
#     "Asset Type": "Equity",
#     ...
#   }
#
# Plus two synthetic summary rows at the bottom — "Cash & Cash Investments"
# and "Positions Total" — which carry "--" placeholders for most fields and
# are NOT real positions. We filter both out silently rather than make the
# operator hand-edit before paste.
#
# Cost basis arrives as a dollar string with `$` and thousands separators
# (e.g., "$1,968.47") so we strip both before computing per-share avg_cost
# as cost_basis_dollars / qty.

_OBJECT_LIST_SEPARATOR_RE = __import__("re").compile(r"\}\s*,\s*\{")


def _looks_like_object_list(text: str) -> bool:
    """Heuristic: text starts with ``{``, ends with ``}``, and contains at
    least one ``},{`` separator — i.e., it's a list-of-objects paste with
    the outer brackets stripped. Avoids auto-wrapping a single-object
    paste (which the operator should fix explicitly).
    """
    t = text.strip()
    if not t.startswith("{") or not t.endswith("}"):
        return False
    return bool(_OBJECT_LIST_SEPARATOR_RE.search(t))


_SCHWAB_FINGERPRINT_KEYS = frozenset({"Symbol", "Asset Type", "Cost Basis"})
_SCHWAB_SKIPPABLE_SYMBOLS = frozenset({
    "Cash & Cash Investments",
    "Positions Total",
})
_SCHWAB_EQUITY_TYPES = frozenset({
    "Equity",
    "ETFs & Closed End Funds",
})


def _is_schwab_export_row(row: Dict[str, Any]) -> bool:
    """Cheap fingerprint: the row carries Schwab's portfolio-export key set."""
    return _SCHWAB_FINGERPRINT_KEYS.issubset(set(row.keys()))


def _schwab_row_to_canonical(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Translate one Schwab-export row into the canonical position dict.

    Returns ``None`` for rows that should be silently skipped (the two
    summary rows + any row whose Asset Type isn't a supported equity
    instrument). Raises ``ValueError`` only when a row LOOKS LIKE a
    real position but its fields can't be parsed — in that case the
    operator wants to know about the malformed input.
    """
    symbol = str(row.get("Symbol", "")).strip()
    if not symbol or symbol in _SCHWAB_SKIPPABLE_SYMBOLS:
        return None

    asset_type = str(row.get("Asset Type", "")).strip()
    if asset_type not in _SCHWAB_EQUITY_TYPES:
        # Cash, money-market, options-not-yet-supported, etc. → skip.
        # When options support lands next session this branch grows a case.
        return None

    qty_raw = str(row.get("Qty (Quantity)", "")).strip()
    qty = _schwab_parse_int(qty_raw)
    if qty is None or qty <= 0:
        raise ValueError(
            f"Schwab row {symbol!r}: unparseable Qty (Quantity) {qty_raw!r}."
        )

    cost_raw = str(row.get("Cost Basis", "")).strip()
    total_cost = _schwab_parse_dollars(cost_raw)
    if total_cost is None or total_cost < 0:
        raise ValueError(
            f"Schwab row {symbol!r}: unparseable Cost Basis {cost_raw!r}."
        )

    avg_cost = total_cost / qty
    description = str(row.get("Description", "")).strip()

    return {
        "ticker": symbol,
        "qty": qty,
        "avg_cost": round(avg_cost, 4),
        "kind": "stock",
        "side": "long",
        "account": "schwab_export",
        "notes": description,
    }


def _schwab_parse_int(value: str) -> Optional[int]:
    """Parse Schwab's quoted integer string. Returns None for ``"--"``."""
    if not value or value == "--":
        return None
    try:
        cleaned = value.replace(",", "").strip()
        if "." in cleaned:
            # Schwab sometimes emits "9.0" for whole-share lots.
            fval = float(cleaned)
            if fval != int(fval):
                return None
            return int(fval)
        return int(cleaned)
    except (TypeError, ValueError):
        return None


def _schwab_parse_dollars(value: str) -> Optional[float]:
    """Parse Schwab's quoted dollar string (e.g., ``"$1,968.47"``)."""
    if not value or value == "--":
        return None
    try:
        cleaned = (
            value.replace("$", "")
                 .replace(",", "")
                 .replace("(", "-")  # negatives sometimes appear as (123.45)
                 .replace(")", "")
                 .strip()
        )
        return float(cleaned)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Portfolio aggregation helpers
# ---------------------------------------------------------------------------

@dataclass
class PortfolioSnapshot:
    """Aggregated view of a positions list for the Streamlit snapshot panel."""

    total_stock_market_value: float = 0.0   # qty × avg_cost (cost-basis proxy)
    total_option_premium_paid: float = 0.0
    held_stock_tickers: List[str] = field(default_factory=list)
    held_option_tickers: List[str] = field(default_factory=list)
    by_sector: Dict[str, float] = field(default_factory=dict)
    position_count_by_kind: Dict[str, int] = field(default_factory=dict)


def aggregate_snapshot(
    positions: List[Position],
    *,
    sector_for,
) -> PortfolioSnapshot:
    """Roll ``positions`` up into a ``PortfolioSnapshot``.

    ``sector_for`` is the ``trading_agent.sector_map.sector_for`` callable
    (passed as an arg so this helper is testable with a fixture mapping).
    """
    snap = PortfolioSnapshot()
    snap.position_count_by_kind = {"stock": 0, "option": 0}
    seen_stock: set[str] = set()
    seen_opt: set[str] = set()

    for p in positions:
        snap.position_count_by_kind[p.kind] += 1
        sector = sector_for(p.ticker) or "Other"
        if p.kind == "stock":
            mv = p.qty * p.avg_cost
            snap.total_stock_market_value += mv
            snap.by_sector[sector] = snap.by_sector.get(sector, 0.0) + mv
            if p.ticker not in seen_stock:
                snap.held_stock_tickers.append(p.ticker)
                seen_stock.add(p.ticker)
        else:  # option
            premium = p.qty * p.avg_cost * 100.0
            snap.total_option_premium_paid += premium
            if p.ticker not in seen_opt:
                snap.held_option_tickers.append(p.ticker)
                seen_opt.add(p.ticker)

    return snap


# ---------------------------------------------------------------------------
# Stubs for next-session implementations
# ---------------------------------------------------------------------------

class AlpacaPositionsProvider(PositionsProvider):  # pragma: no cover — next session
    """Pulls positions from the paper Alpaca account.

    NOT IMPLEMENTED THIS SESSION. The class exists so importers can
    reference it; ``snapshot()`` raises ``NotImplementedError``.
    """

    def __init__(self, api_key: str, secret_key: str, base_url: str) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._base_url = base_url

    @property
    def source_label(self) -> str:
        return "Alpaca (paper)"

    def snapshot(self) -> List[Position]:
        raise NotImplementedError(
            "AlpacaPositionsProvider lands next session. Use "
            "ManualPositionsProvider for now."
        )


class SchwabPositionsProvider(PositionsProvider):  # pragma: no cover — next session
    """Pulls positions from a Schwab brokerage account.

    NOT IMPLEMENTED THIS SESSION. Requires the ``trading`` OAuth scope
    which the repo doesn't yet have a flow for; that lands in a
    dedicated session alongside the Schwab order placement.
    """

    def __init__(self, account_id: str, oauth_session: Any) -> None:
        self._account_id = account_id
        self._oauth = oauth_session

    @property
    def source_label(self) -> str:
        return "Schwab (live)"

    def snapshot(self) -> List[Position]:
        raise NotImplementedError(
            "SchwabPositionsProvider lands next session. Use "
            "ManualPositionsProvider for now."
        )
