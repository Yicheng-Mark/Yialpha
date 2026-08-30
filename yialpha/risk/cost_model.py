"""Round-trip cost model — the single source of truth for trading frictions.

V2.0 P0.1: the decision-time tradeability gate needs to price a round trip
BEFORE a trade is allowed, with the same numbers the backtester charges on
fills. The Binance fee schedule previously lived (only) in
``backtest/engine.py``; it moves here so the backtester, the ExecutionTicket's
``estimated_cost`` field, and any future portfolio preview price identical
frictions (one definition, no drift).

``CostEstimate`` decomposes the round trip into named bps legs and a signed
carry leg; ``estimate_round_trip_cost`` composes one per asset type. All
functions are pure — no I/O, no config reads — so tests and the deterministic
overlay can call them freely. Costs are expressed in basis points per leg;
``total_bps`` is the full round trip the edge must clear.
"""

from __future__ import annotations

from dataclasses import dataclass

from yialpha.versions import COST_MODEL_VERSION

# --- Binance USDT-M regular-tier fee schedule (VIP-0, no promo) -------------
# Moved verbatim from backtest/engine.py (which re-imports them for backward
# compatibility). Market-order fills pay the TAKER rate; the maker constant is
# carried for limit-style variants. BNB fee settlement takes 10% off.
BINANCE_USDT_M_TAKER_BPS = 5.0    # 0.05%
BINANCE_USDT_M_MAKER_BPS = 2.0    # 0.02%
BNB_FEE_DISCOUNT = 0.10

# --- Binance spot regular tier (VIP-0, 0.1% per side) -----------------------
BINANCE_SPOT_FEE_BPS = 10.0   # 0.10%

# --- Conservative defaults for venues the framework does not trade on -------
# Zero-commission US retail equity brokers charge nothing explicit per fill,
# but executions still cross spread/impact; 5 bps per side approximates liquid
# US names and UNDERSTATES small caps or HK/A-share legs — price those
# explicitly at the call site when they matter. Stock shorts additionally pay
# margin borrow, defaulted to a conservative 1%/yr placeholder.
STOCK_FEE_BPS = 0.0
STOCK_SLIPPAGE_BPS = 5.0
STOCK_BORROW_APR_BPS = 100.0   # 1.0%/yr, accrues over the holding horizon

CRYPTO_SLIPPAGE_BPS = 2.0      # per side, liquid USDT pairs

_PERP_ASSET_TYPES = frozenset({"crypto_perp"})
_SPOT_LIKE_ASSET_TYPES = frozenset({"crypto", "crypto_spot"})


@dataclass(frozen=True)
class CostEstimate:
    """One round trip decomposed into legs, all in basis points.

    ``funding_bps`` and ``borrow_bps`` are SIGNED over the holding horizon:
    positive = paid (a cost), negative = received (a credit that legitimately
    reduces ``total_bps``). ``total_bps`` sums every leg; a negative total
    means the carry credit exceeds the explicit frictions and is reported as
    such rather than clamped.
    """

    entry_fee_bps: float = 0.0
    exit_fee_bps: float = 0.0
    entry_slippage_bps: float = 0.0
    exit_slippage_bps: float = 0.0
    funding_bps: float = 0.0
    borrow_bps: float = 0.0

    @property
    def total_bps(self) -> float:
        return (
            self.entry_fee_bps
            + self.exit_fee_bps
            + self.entry_slippage_bps
            + self.exit_slippage_bps
            + self.funding_bps
            + self.borrow_bps
        )

    def as_dict(self) -> dict[str, float]:
        """Flat dict for the ExecutionTicket / ledger (total included)."""
        return {
            "entry_fee_bps": self.entry_fee_bps,
            "exit_fee_bps": self.exit_fee_bps,
            "entry_slippage_bps": self.entry_slippage_bps,
            "exit_slippage_bps": self.exit_slippage_bps,
            "funding_bps": self.funding_bps,
            "borrow_bps": self.borrow_bps,
            "total_bps": self.total_bps,
        }


def estimate_round_trip_cost(
    asset_type: str,
    side: str,
    horizon_days: float = 5.0,
    funding_rate_annualized: float | None = None,
    bnb_discount: bool = False,
) -> CostEstimate:
    """Price one round trip for a candidate trade.

    ``asset_type`` is the CLI asset type (``stock`` / ``crypto`` /
    ``crypto_spot`` / ``crypto_perp``). ``side`` is ``long`` / ``short`` /
    ``flat`` — a flat "side" prices every leg at zero. ``horizon_days`` is the
    intended holding period used to accrue carry legs; the default mirrors the
    accuracy loop's default holding horizon (5 days).

    Per asset type:

    - ``stock``: no explicit fee (zero-commission default), 5 bps slippage per
      side; shorts accrue ``STOCK_BORROW_APR_BPS`` over the horizon.
    - ``crypto`` / ``crypto_spot``: Binance spot fee per side + crypto
      slippage per side. (The ``crypto`` Yahoo path is analysis-only; it is
      priced like a spot venue, documented here.)
    - ``crypto_perp``: taker fee per side (×0.9 with ``bnb_discount``),
      crypto slippage per side, and expected funding over the horizon from
      ``funding_rate_annualized`` — SIGNED by side: a long PAYS positive
      funding, a short RECEIVES it, so the short's carry leg is negative.
    """
    if horizon_days < 0.0:
        raise ValueError(f"horizon_days must be >= 0, got {horizon_days!r}")
    side_l = str(side or "").strip().lower()
    if side_l not in ("long", "short", "flat"):
        raise ValueError(f"side must be long/short/flat, got {side!r}")
    if side_l == "flat":
        return CostEstimate()

    sign = 1.0 if side_l == "long" else -1.0

    if asset_type == "stock":
        return CostEstimate(
            entry_fee_bps=STOCK_FEE_BPS,
            exit_fee_bps=STOCK_FEE_BPS,
            entry_slippage_bps=STOCK_SLIPPAGE_BPS,
            exit_slippage_bps=STOCK_SLIPPAGE_BPS,
            borrow_bps=(
                STOCK_BORROW_APR_BPS * horizon_days / 365.0 if side_l == "short" else 0.0
            ),
        )

    if asset_type in _SPOT_LIKE_ASSET_TYPES:
        return CostEstimate(
            entry_fee_bps=BINANCE_SPOT_FEE_BPS,
            exit_fee_bps=BINANCE_SPOT_FEE_BPS,
            entry_slippage_bps=CRYPTO_SLIPPAGE_BPS,
            exit_slippage_bps=CRYPTO_SLIPPAGE_BPS,
        )

    if asset_type in _PERP_ASSET_TYPES:
        fee = BINANCE_USDT_M_TAKER_BPS * (1.0 - BNB_FEE_DISCOUNT if bnb_discount else 1.0)
        funding_bps = 0.0
        if funding_rate_annualized is not None:
            funding_bps = (
                float(funding_rate_annualized) * 1e4 * horizon_days / 365.0 * sign
            )
        return CostEstimate(
            entry_fee_bps=fee,
            exit_fee_bps=fee,
            entry_slippage_bps=CRYPTO_SLIPPAGE_BPS,
            exit_slippage_bps=CRYPTO_SLIPPAGE_BPS,
            funding_bps=funding_bps,
        )

    raise ValueError(f"unknown asset_type {asset_type!r}")


__all__ = [
    "BNB_FEE_DISCOUNT",
    "BINANCE_SPOT_FEE_BPS",
    "BINANCE_USDT_M_MAKER_BPS",
    "BINANCE_USDT_M_TAKER_BPS",
    "CRYPTO_SLIPPAGE_BPS",
    "COST_MODEL_VERSION",
    "CostEstimate",
    "STOCK_BORROW_APR_BPS",
    "STOCK_FEE_BPS",
    "STOCK_SLIPPAGE_BPS",
    "estimate_round_trip_cost",
]
