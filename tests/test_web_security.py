"""Static supply-chain and sink guards for the no-build web client."""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "web" / "static" / "index.html"
APP = ROOT / "web" / "static" / "app.js"
CHARTS = ROOT / "web" / "static" / "charts.js"
PURIFY = ROOT / "web" / "static" / "vendor" / "purify.min.js"


@pytest.mark.unit
def test_vendored_dompurify_is_pinned_and_loaded_before_markdown_app():
    normalized = PURIFY.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(normalized).hexdigest() == (
        "9ab3d44d73c3e3947f9ab72e0f0bc15c7f1931d60b365ba261fc85fe59013c56"
    )
    html = INDEX.read_text(encoding="utf-8")
    assert html.index("vendor/purify.min.js") < html.index("vendor/marked.min.js")
    assert html.index("vendor/marked.min.js") < html.index("app.js")


@pytest.mark.unit
def test_markdown_sink_is_sanitized_with_active_content_forbidden():
    source = APP.read_text(encoding="utf-8")
    assert "DOMPurify.sanitize(window.marked.parse(t), MARKDOWN_SANITIZE_CONFIG)" in source
    assert 'FORBID_TAGS: ["math", "script", "style", "svg", "template"]' in source
    assert 'FORBID_ATTR: ["formaction", "srcdoc", "style", "xlink:href"]' in source
    assert "ALLOW_DATA_ATTR: false" in source


@pytest.mark.unit
def test_secondary_server_data_hrefs_are_validated_and_escaped():
    source = APP.read_text(encoding="utf-8")
    assert 'href="${st.report_url}"' not in source
    assert 'href="${esc(reportURL)}"' in source
    assert "const reportURL = safeReportURL(st.report_url)" in source
    assert "encodeURIComponent(String(dr.date" in source
    assert "<span>${esc(dr.date)}</span>" in source


@pytest.mark.unit
def test_new_analysis_asset_and_date_controls_match_server_contract():
    source = APP.read_text(encoding="utf-8")
    i18n = (ROOT / "web" / "static" / "i18n.js").read_text(encoding="utf-8")
    assert 'value="${today}" max="${today}"' in source
    assert 'option value="crypto_spot"' in source
    assert i18n.count("new_asset_crypto_spot:") == 2


@pytest.mark.unit
def test_node_performance_tooltip_escapes_server_controlled_name():
    source = CHARTS.read_text(encoding="utf-8")
    assert '"<b>" + escapeHTML(d.name) + "</b><br/>"' in source
    assert '"<b>" + d.name + "</b><br/>"' not in source


@pytest.mark.unit
def test_index_has_no_inline_script_and_loads_external_theme_initializer():
    html = INDEX.read_text(encoding="utf-8")
    assert '<script src="/static/theme-init.js?v=1"></script>' in html
    assert "<script>" not in html
