"""Perp-specific analytics that are neither dataflow vendors nor risk math.

V2.1: the deterministic Fair Value Bridge — converting a USD underlying
price target into a USDT contract target (USDT/USD FX via the inverted
Binance spot USDCUSDT price, plus expected basis) with the full conversion
chain recorded for audit.
"""
