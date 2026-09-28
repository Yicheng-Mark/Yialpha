"""Execution layer (Phase 4).

Turns a Portfolio Manager decision into an actual order. Two paths:
  * Broker API (preferred when available) — :class:`~yialpha.execution.binance_gateway.BinanceGateway`
    (official modular SDK, testnet by default, behind the three-switch live
    policy); not yet wired into the LangGraph graph.
  * Browser broker via Playwright/Edge (fallback when no API) — fail-closed,
    ``dry_run`` by default, behind a kill switch.
"""
