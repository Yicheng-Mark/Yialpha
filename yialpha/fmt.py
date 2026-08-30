"""Shared number formatting for report/dashboard surfaces.

2026-08-16 dedup: ``_fmt_pct`` / ``_fmt_num`` were verbatim copies in
backtest/report.py, monitoring/dashboard.py and agents/utils/valuation_tools.py.
These two are the EXACT duplicates (``"n/a"``, ``.2f``% / ``.3f``) — the
other per-module formatters genuinely differ ("N/A" vs "n/a", configurable
digits, CN yuan/share semantics, NaN handling) and are presentation
contracts of their own modules, so they stay local deliberately.
"""
from __future__ import annotations


def fmt_pct(x: float | None) -> str:
    """``0.0123 -> "1.23%"``; ``None -> "n/a"``."""
    if x is None:
        return "n/a"
    return f"{x * 100:.2f}%"


def fmt_num3(x: float | None) -> str:
    """Three-decimal number; ``None -> "n/a"``."""
    if x is None:
        return "n/a"
    return f"{x:.3f}"
