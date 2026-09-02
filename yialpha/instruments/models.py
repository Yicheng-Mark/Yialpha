"""Value objects and the exchangeInfo classification rule for perps.

V2.1 Measurability (record stage): Binance USDT-M exchangeInfo types every
contract with an ``underlyingType`` (COIN / EQUITY / HK_EQUITY / KR_EQUITY /
CN_EQUITY / COMMODITY / PREMARKET / INDEX). That field is the authoritative
answer to "is this perp a tokenized US stock?", so the registry stores one
:class:`InstrumentRecord` per (symbol, snapshot time) and
:func:`classify_exchangeinfo_row` turns a raw exchangeInfo row into the
routing vocabulary — including the explicit ``unknown_perp`` bucket for
contract types this framework does not support (non-US equity, commodities,
indexes, pre-IPO), which were previously mislabeled as pure crypto.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

#: The perp classification vocabulary (routing's perp answers superset).
PerpInstrumentClass = Literal["stock_perp", "pure_crypto_perp", "unknown_perp"]

#: ``underlyingType`` values this framework cannot analyze: non-US equity has
#: no rule-based vendor mapping, and commodities / pre-IPO / index contracts
#: have no company fundamentals at all.
UNSUPPORTED_UNDERLYING_TYPES = frozenset(
    {"HK_EQUITY", "KR_EQUITY", "CN_EQUITY", "COMMODITY", "INDEX", "PREMARKET"}
)


@dataclass(frozen=True)
class InstrumentRecord:
    """One point-in-time registry row for a Binance USDT-M perpetual.

    Field notes:

    * ``classification_source`` — where the class came from:
      ``"binance_exchangeinfo"`` (persisted exchangeInfo evidence, confidence
      1.0), ``"registry"`` / ``"warm_cache"`` / ``"static_seed"`` (reserved
      for future source tiers), ``"registry_empty"`` (no snapshot at or
      before the requested as-of; confidence 0.0), or ``"unknown"``.
    * ``classification_confidence`` — 0..1; 1.0 for direct exchangeInfo
      evidence, 0.0 when nothing is known.
    * ``unsupported_reason`` — machine-readable reason when
      ``instrument_class == "unknown_perp"``
      (``unsupported-contract-type:<underlyingType>`` or
      ``underlying_type_missing``), else ``None``.
    * ``underlying_type`` — the raw Binance ``underlyingType`` string.
    * ``underlying_symbol`` — the US-equity base for stock perps (base asset
      or symbol minus quote suffix); ``None`` otherwise.
    * ``onboard_date`` — ISO date (``YYYY-MM-DD``) from the exchangeInfo
      ``onboardDate`` ms epoch; ``None`` when the row carried none.
    * ``session_calendar`` — one of the constants from
      :mod:`yialpha.instruments.sessions` (``None`` for unknown perps).
    * ``leverage_bracket_version`` — reserved for a future snapshot tier;
      always ``None`` at the record stage.
    * ``snapshot_available_at`` — when this evidence became knowable (ISO
      UTC); the PIT queries in :mod:`yialpha.instruments.registry` bound on it.
    * ``listed_asof`` — ``True``/``False`` only when BOTH an as-of and an
      onboard date are known (as-of date < onboard date → ``False``, else
      ``True``); ``None`` means "no evidence to answer".
    """

    symbol: str
    instrument_class: PerpInstrumentClass
    classification_source: str
    classification_confidence: float
    unsupported_reason: str | None = None
    underlying_type: str | None = None
    underlying_symbol: str | None = None
    quote_asset: str | None = None
    margin_asset: str | None = None
    onboard_date: str | None = None
    status: str | None = None
    session_calendar: str | None = None
    tick_size: float | None = None
    step_size: float | None = None
    min_qty: float | None = None
    min_notional: float | None = None
    leverage_bracket_version: str | None = None
    snapshot_available_at: str | None = None
    listed_asof: bool | None = None

    def as_disclosure(self) -> str:
        """One-line human summary of this classification (disclosure lines)."""
        parts = [
            f"{self.symbol}: {self.instrument_class}",
            f"source={self.classification_source}",
            f"confidence={self.classification_confidence:.2f}",
        ]
        if self.unsupported_reason:
            parts.append(f"reason={self.unsupported_reason}")
        if self.underlying_type:
            parts.append(f"underlying_type={self.underlying_type}")
        if self.underlying_symbol:
            parts.append(f"underlying={self.underlying_symbol}")
        if self.onboard_date:
            parts.append(f"onboard={self.onboard_date}")
        if self.status:
            parts.append(f"status={self.status}")
        if self.snapshot_available_at:
            parts.append(f"snapshot={self.snapshot_available_at}")
        return ", ".join(parts)


def classify_exchangeinfo_row(row: dict[str, Any]) -> tuple[PerpInstrumentClass, str | None]:
    """Classify one exchangeInfo symbol row into the routing vocabulary.

    Truth table on ``underlyingType``:

    * ``"EQUITY"`` → ``("stock_perp", None)`` — a tokenized US equity.
    * one of :data:`UNSUPPORTED_UNDERLYING_TYPES` (HK/KR/CN equity, commodity,
      index, premarket) → ``("unknown_perp",
      "unsupported-contract-type:<underlyingType>")``.
    * ``"COIN"`` → ``("pure_crypto_perp", None)``.
    * missing or any other value → ``("unknown_perp", "underlying_type_missing")``.

    Pure function, no I/O — safe on hot paths and in tests.
    """
    underlying_type = row.get("underlyingType") if isinstance(row, dict) else None
    if underlying_type == "EQUITY":
        return "stock_perp", None
    if underlying_type in UNSUPPORTED_UNDERLYING_TYPES:
        return "unknown_perp", f"unsupported-contract-type:{underlying_type}"
    if underlying_type == "COIN":
        return "pure_crypto_perp", None
    return "unknown_perp", "underlying_type_missing"
