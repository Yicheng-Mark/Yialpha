from enum import Enum


class AnalystType(str, Enum):
    MARKET = "market"
    # Wire value stays "social" for saved-config and string-keyed-caller
    # back-compat; the user-facing label is "Sentiment Analyst".
    SOCIAL = "social"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"


class AssetType(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    # Binance USDT-M perpetual futures (Track A analysis-only). Opt-in via the
    # explicit --asset-type crypto_perp flag; detect_asset_type never returns
    # this (BTCUSDT auto-detects as CRYPTO spot, which is acceptable).
    CRYPTO_PERP = "crypto_perp"
