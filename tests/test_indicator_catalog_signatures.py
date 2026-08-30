"""Real-compute signature smoke for the indicator catalog.

The 46-item audit (2026-08-15) found six P0 bugs where vendor functions had
fake signatures masked by mock-based tests. The indicator catalog is the same
class of risk: a catalog entry whose stockstats column name does not actually
compute would pass every mock-based test and only fail in a live run, silently
shrinking the analyst's tool vocabulary. These tests compute EVERY catalog
entry against real stockstats on a deterministic synthetic OHLCV frame — no
mocks, no network — so an unsupported name fails here, loudly.

Also pins the MFI scale contract: stockstats returns 0–1, the catalog's
thresholds assume 0–100, and ``compute_indicator`` must bridge the two.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from stockstats import wrap

from yialpha.dataflows.feature_registry import DERIVED_FEATURES, compute_derived
from yialpha.dataflows.indicator_catalog import INDICATORS
from yialpha.dataflows.stockstats_utils import compute_indicator


def _synthetic_ohlcv(rows: int = 400, capitalized: bool = False) -> pd.DataFrame:
    """Deterministic pseudo-market OHLCV frame (seeded).

    ``capitalized`` matches load_ohlcv's shape (the derived-feature inputs);
    lowercase matches stockstats' convention (wrapped frames).
    """
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, rows)))
    frame = pd.DataFrame(
        {
            "Open": close * (1.0 + rng.normal(0.0, 0.01, rows)),
            "High": close * (1.0 + np.abs(rng.normal(0.0, 0.012, rows))),
            "Low": close * (1.0 - np.abs(rng.normal(0.0, 0.012, rows))),
            "Close": close,
            "Volume": rng.integers(1_000_000, 9_000_000, rows).astype(float),
        },
        index=pd.bdate_range("2023-01-02", periods=rows),
    )
    if not capitalized:
        frame.columns = [c.lower() for c in frame.columns]
    return frame


@pytest.mark.unit
class TestCatalogSignaturesCompute:
    def test_every_catalog_entry_computes(self):
        """Each catalog name must really compute — stockstats column or
        derived feature — never a typo masked by mocks.

        stockstats signals an unparsable column name by *raising UserWarning*,
        so warnings are escalated to errors; a name that "computes" by being
        swallowed still fails here. Derived features compute on the raw
        capitalized frame via the feature registry.
        """
        sdf = wrap(_synthetic_ohlcv())
        raw = _synthetic_ohlcv(capitalized=True)
        failures: list[str] = []
        for name in INDICATORS:
            try:
                if name in DERIVED_FEATURES:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        col = compute_derived(raw, name)
                    assert col is not None
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        col = compute_indicator(sdf, name)
            except Exception as exc:  # noqa: BLE001 — collect, then fail loudly
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                continue
            tail = col.tail(50)
            if int(tail.notna().sum()) == 0:
                failures.append(f"{name}: all-NaN in the last 50 rows")
        assert not failures, "catalog entries failed real computation:\n" + "\n".join(failures)

    def test_computed_values_vary(self):
        """A stuck column (constant regardless of data) is also a failure."""
        sdf = wrap(_synthetic_ohlcv())
        raw = _synthetic_ohlcv(capitalized=True)
        for name in INDICATORS:
            col = (
                compute_derived(raw, name)
                if name in DERIVED_FEATURES
                else compute_indicator(sdf, name)
            ).dropna()
            assert col.nunique() > 1, f"{name} is constant across 400 rows"

    def test_derived_registry_names_are_catalog_entries(self):
        """No orphan derived features: every registry key must be a catalog
        entry (the catalog is the LLM-facing vocabulary; a registry key no
        tool can name is dead code)."""
        assert set(DERIVED_FEATURES) <= set(INDICATORS)


@pytest.mark.unit
class TestMfiScaleContract:
    def test_raw_stockstats_mfi_is_0_1(self):
        """Guard the premise of the scale fix: raw stockstats mfi is 0–1."""
        sdf = wrap(_synthetic_ohlcv())
        raw = sdf["mfi"].dropna()
        assert float(raw.max()) <= 1.0
        assert float(raw.min()) >= 0.0

    def test_compute_indicator_scales_mfi_to_0_100(self):
        sdf = wrap(_synthetic_ohlcv())
        scaled = compute_indicator(sdf, "mfi")
        assert float(scaled.max()) > 50.0  # raw max was ~0.88 -> ~88
        assert float(scaled.min()) >= 0.0
        assert float(scaled.max()) <= 100.0

    def test_scale_fix_persists_on_the_frame(self):
        """Later reads of the same frame must see the scaled column too."""
        sdf = wrap(_synthetic_ohlcv())
        compute_indicator(sdf, "mfi")
        assert float(sdf["mfi"].max()) > 50.0

    def test_unscaled_indicators_pass_through_unchanged(self):
        sdf = wrap(_synthetic_ohlcv())
        raw = sdf["rsi"].copy()
        out = compute_indicator(sdf, "rsi")
        pd.testing.assert_series_equal(out, raw)
