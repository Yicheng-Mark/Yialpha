"""Cross-file consistency of the unified indicator catalog (P3-19).

The name -> description catalog used to live in three drifted copies
(y_finance tool gate, market_analyst prompt sections, Alpha Vantage
descriptions — the AV copy had already lost ``mfi``). All three now render
from yiagents/dataflows/indicator_catalog.py; these tests pin that they
stay in lockstep.
"""

from __future__ import annotations

import pytest

import yiagents.agents.analysts.market_analyst as ma
import yiagents.dataflows.alpha_vantage_indicator as avi
import yiagents.dataflows.indicator_catalog as ic
import yiagents.dataflows.y_finance as yfin


@pytest.mark.unit
class TestCatalogConsistency:
    def test_best_ind_params_matches_catalog(self):
        """Every name in best_ind_params is in the catalog, and vice versa."""
        assert set(yfin.best_ind_params) == set(ic.INDICATORS)
        for name, description in yfin.best_ind_params.items():
            assert description == ic.INDICATORS[name].description

    def test_av_descriptions_cover_the_same_set(self):
        """The AV description copy covers the whole catalog (mfi included)."""
        assert set(avi.indicator_descriptions) == set(ic.INDICATORS)
        assert "mfi" in avi.indicator_descriptions  # the pre-unification drift
        for name, description in avi.indicator_descriptions.items():
            assert description == ic.INDICATORS[name].description

    def test_av_gate_and_columns_cover_the_same_set(self):
        """The AV supported gate and column map cover the whole catalog."""
        assert set(avi.supported_indicators) == set(ic.INDICATORS)
        assert set(avi.col_name_map) | {"vwma"} == set(ic.INDICATORS)
        for name, (label, series) in avi.supported_indicators.items():
            assert label == ic.INDICATORS[name].label
            assert series == ic.INDICATORS[name].av_series_type

    def test_analyst_sections_render_from_catalog(self):
        """The market analyst's sections are the catalog's prompt-visible
        entries, in catalog order — with descriptions byte-identical."""
        assert list(ma._INDICATOR_SECTIONS) == list(ic.ANALYST_SECTIONS)
        expected = {
            name
            for name, spec in ic.INDICATORS.items()
            if spec.in_analyst_prompt
        }
        assert set(ma.INDICATOR_NAMES) == expected
        # The pinned prompt baseline does NOT advertise mfi even though the
        # indicator tools (and now Alpha Vantage) support it.
        assert "mfi" not in ma.INDICATOR_NAMES
        assert "mfi" in yfin.best_ind_params

    def test_every_entry_has_section_and_description(self):
        """Catalog hygiene: descriptions start with the label and every
        section header has at least one prompt entry."""
        for name, spec in ic.INDICATORS.items():
            assert spec.description.startswith(f"{spec.label}: "), name
        prompt_sections = {header for header, _ in ic.ANALYST_SECTIONS}
        for spec in ic.INDICATORS.values():
            if spec.in_analyst_prompt:
                assert spec.section in prompt_sections
        for header, entries in ic.ANALYST_SECTIONS:
            assert entries, header  # no empty sections in the prompt render
