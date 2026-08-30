"""Execution layer (Phase 4).

Turns a Portfolio Manager decision into an actual order. Two paths:
  * Broker API (preferred when available) — not implemented here.
  * Browser broker via Playwright/Edge (fallback when no API) — fail-closed,
    ``dry_run`` by default, behind a kill switch.
"""
