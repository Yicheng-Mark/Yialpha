"""Instrument identity & market metadata for Binance USDT-M perpetuals.

V2.1 Measurability: a persistent, point-in-time instrument registry so perp
classification (stock_perp / pure_crypto_perp / unknown_perp) survives
restarts, replay honour ``available_at <= analysis_as_of``, and unsupported
contract types (non-US equity, commodity, index) are labelled instead of
being silently treated as US stocks or pure crypto.
"""
