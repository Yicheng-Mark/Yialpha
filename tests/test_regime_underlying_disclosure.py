"""Stock-perp regime: a missing PIT underlying mapping must be DISCLOSED.

``compute_regime_state``'s stock-perp family used to fall back to the perp
contract code when the PIT registry produced no ``underlying_symbol``
(``underlying_symbol or ticker``) — handing "MUUSDT" (not a stock symbol)
to the equity history source, which either fails or silently returns junk.
The converged contract: without a mapping the equity leg is UNAVAILABLE —
disclosed via ``missing_inputs`` (``"underlying_symbol"`` names the root
cause, ``"underlying_history"`` the degraded leg), the contract symbol is
NEVER passed to the stock source, and the regime still computes (degraded,
not faked) from the perp-side legs. Registry as-of handling and the rest of
the as-of/context contract are out of scope here.

Zero network: every seam (klines, live depth, registry, YFinance history)
is monkeypatched with synthetic frames; dates come from ``date.today()`` so
the live-mode branch is taken without stubbing the clock.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import yialpha.regime.compute as rc
from yialpha.instruments.models import InstrumentRecord

_TODAY = date.today().isoformat()

_UPTREND = [100.0 + 0.5 * i for i in range(260)]  # close > SMA50 > SMA200


def _frame() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(_UPTREND), freq="D")
    idx.name = "Date"  # binance_klines_frame ships a Date-named index
    return pd.DataFrame(
        {
            "Open": [c * 0.999 for c in _UPTREND],
            "High": [c * 1.01 for c in _UPTREND],
            "Low": [c * 0.99 for c in _UPTREND],
            "Close": _UPTREND,
            "Volume": [1000.0] * len(_UPTREND),
        },
        index=idx,
    )


def _mock_perp_seams(monkeypatch) -> None:
    """Perp-side legs (klines incl. the index read + live depth)."""
    monkeypatch.setattr(rc, "binance_klines_frame", lambda *a, **k: _frame())
    monkeypatch.setattr(
        rc, "_fetch_depth_bands",
        lambda s: {
            "status": "ok",
            "spread_bps": 1.5,
            "bands": {
                "50": {"bid_notional": 1_000_000.0, "ask_notional": 1_000_000.0},
            },
        },
    )


def _record(underlying: str | None, onboard: str | None = "2024-06-01"):
    return InstrumentRecord(
        symbol="MUUSDT", instrument_class="stock_perp",
        classification_source="binance_exchangeinfo",
        classification_confidence=1.0,
        underlying_symbol=underlying, onboard_date=onboard,
    )


@pytest.mark.unit
def test_missing_mapping_withholds_contract_symbol_from_stock_source(monkeypatch):
    """Registry read fails -> no mapping -> YFinance never sees "MUUSDT"."""
    yfin_calls: list[tuple[str, str, str]] = []

    def fake_yfin(symbol, start, end):
        yfin_calls.append((symbol, start, end))
        return _frame()

    def raising_registry(symbol, as_of=None):  # noqa: ARG001
        raise RuntimeError("registry unavailable")

    _mock_perp_seams(monkeypatch)
    monkeypatch.setattr(rc, "classify_perp", raising_registry)
    monkeypatch.setattr(rc, "get_YFin_history_cached", fake_yfin)

    state = rc.compute_regime_state(
        "MUUSDT", "crypto_perp", "stock_perp", _TODAY, end_date=_TODAY,
    )
    # The perp contract code is never handed to the stock source.
    assert yfin_calls == []
    # The regime degrades (does not become uncomputable): the perp-side
    # legs still carry it, the equity leg is honestly missing.
    assert state is not None
    assert state.underlying_trend is None
    assert state.trend_regime == "up"
    assert state.realized_vol_pct is not None
    # Root cause AND degraded leg both disclosed.
    assert "underlying_symbol" in state.missing_inputs
    assert "underlying_history" in state.missing_inputs
    assert state.confidence_components["underlying"] == 0.0


@pytest.mark.unit
def test_registry_row_without_underlying_symbol_discloses_unavailable(monkeypatch):
    """A registry row that carries no underlying symbol behaves the same."""
    yfin_calls: list[tuple[str, str, str]] = []

    def fake_yfin(symbol, start, end):
        yfin_calls.append((symbol, start, end))
        return _frame()

    _mock_perp_seams(monkeypatch)
    monkeypatch.setattr(
        rc, "classify_perp",
        lambda symbol, as_of=None: _record(underlying=None),
    )
    monkeypatch.setattr(rc, "get_YFin_history_cached", fake_yfin)

    state = rc.compute_regime_state(
        "MUUSDT", "crypto_perp", "stock_perp", _TODAY, end_date=_TODAY,
    )
    assert yfin_calls == []
    assert state is not None
    assert state.underlying_trend is None
    assert "underlying_symbol" in state.missing_inputs
    assert "underlying_history" in state.missing_inputs
    assert state.confidence_components["underlying"] == 0.0


@pytest.mark.unit
def test_mapping_present_still_queries_the_underlying(monkeypatch):
    """Positive control: a resolved mapping queries YFinance with "MU"."""
    yfin_calls: list[tuple[str, str, str]] = []

    def fake_yfin(symbol, start, end):
        yfin_calls.append((symbol, start, end))
        return _frame()

    _mock_perp_seams(monkeypatch)
    monkeypatch.setattr(
        rc, "classify_perp",
        lambda symbol, as_of=None: _record(underlying="MU"),
    )
    monkeypatch.setattr(rc, "get_YFin_history_cached", fake_yfin)

    state = rc.compute_regime_state(
        "MUUSDT", "crypto_perp", "stock_perp", _TODAY, end_date=_TODAY,
    )
    assert [symbol for symbol, _start, _end in yfin_calls] == ["MU"]
    assert state is not None
    assert state.underlying_trend == "up"
    assert "underlying_symbol" not in state.missing_inputs
    assert "underlying_history" not in state.missing_inputs
    assert state.confidence_components["underlying"] == 1.0
