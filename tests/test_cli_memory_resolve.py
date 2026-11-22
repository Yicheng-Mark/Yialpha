"""Round-13 regression pins for ``yialpha memory-resolve``.

The sweep CLI wrapper crashed with a bare ``e["ticker"]`` KeyError on a
hand-corrupted pending entry whose ticker field was empty (``[date |  | Buy |
pending]`` parses to ``ticker=""``) — BEFORE the hardened resolver
(:func:`yialpha.graph.memory_resolution.resolve_pending_entries`, which has
skipped malformed rows since round-8) ever ran. The wrapper now tolerates
them the same way: skip + yellow notice, sweep the rest.
"""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

import yialpha.cli.main as main_mod
import yialpha.llm_clients as llm_clients_mod
import yialpha.graph.memory_resolution as memory_resolution_mod
from yialpha.cli.main import app

runner = CliRunner()

_SEP = "\n\n<!-- ENTRY_END -->\n\n"


def _write_log(tmp_path, tag_lines):
    """Write a memory log in the real text format: one ``[...]`` tag + body
    per entry, separated by the ENTRY_END marker."""
    blocks = [
        f"{tag}\n\nDECISION:\nsome decision text for {tag.split(' | ')[1] or 'MALFORMED'}"
        for tag in tag_lines
    ]
    log = tmp_path / "trading_memory.md"
    log.write_text(_SEP.join(blocks) + _SEP, encoding="utf-8")
    return log


@pytest.fixture(autouse=True)
def _stub_llm_and_resolver(monkeypatch):
    """The sweep's LLM client + resolver are heavy/offline-unfriendly; the
    pin only covers the wrapper's filtering + grouping, so stub both. The
    resolver stub records swept tickers into the injected list."""
    class _FakeLLM:
        def get_llm(self):
            return object()

    monkeypatch.setattr(
        llm_clients_mod, "create_llm_client", lambda **kw: _FakeLLM()
    )
    resolved: list[str] = []

    def _fake_resolve(memory_log, reflector, ticker, **kwargs):
        resolved.append(ticker)
        return 1

    monkeypatch.setattr(
        memory_resolution_mod, "resolve_pending_entries", _fake_resolve
    )
    return resolved


@pytest.mark.unit
def test_memory_resolve_skips_malformed_entry_and_sweeps_rest(
    tmp_path, monkeypatch, _stub_llm_and_resolver,
):
    """A pending row with an empty ticker no longer crashes the sweep."""
    log = _write_log(
        tmp_path,
        [
            "[2026-09-01 | AAPL | Buy | pending]",
            # Hand-corrupted row: the ticker field is gone (empty).
            "[2026-09-02 |  | Buy | pending]",
            "[2026-09-03 | MSFT | Sell | pending]",
        ],
    )
    monkeypatch.setitem(main_mod.DEFAULT_CONFIG, "memory_log_path", str(log))

    result = runner.invoke(app, ["memory-resolve"])
    assert result.exit_code == 0, result.output
    # The malformed row was skipped (no KeyError, no phantom "" ticker swept);
    # both valid tickers were swept.
    assert sorted(_stub_llm_and_resolver) == ["AAPL", "MSFT"]
    assert "malformed pending entry" in result.output


@pytest.mark.unit
def test_memory_resolve_ticker_filter_excludes_malformed(
    tmp_path, monkeypatch, _stub_llm_and_resolver,
):
    """--ticker filtering coexists with the malformed-row skip."""
    log = _write_log(
        tmp_path,
        [
            "[2026-09-01 | AAPL | Buy | pending]",
            "[2026-09-02 |  | Buy | pending]",
        ],
    )
    monkeypatch.setitem(main_mod.DEFAULT_CONFIG, "memory_log_path", str(log))

    result = runner.invoke(app, ["memory-resolve", "-t", "MSFT"])
    assert result.exit_code == 0, result.output
    # Only the malformed row remained after the filter; nothing matched MSFT,
    # and the run must end on the "no pending entries match" path, not a crash.
    assert "No pending entries match" in result.output
    assert _stub_llm_and_resolver == []
