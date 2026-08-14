"""Health endpoint backend — thin alias for the shared preflight.

The five checks live in :mod:`yiagents.monitoring.preflight` (single
implementation shared with ``scripts/run_baseline.py``). This module kept
for import stability; register the endpoint as a sync ``def`` so Starlette
runs it in a threadpool (the yfinance and DeepSeek probes block for seconds).
"""

from __future__ import annotations

from yiagents.monitoring.preflight import run_health

__all__ = ["run_health"]
