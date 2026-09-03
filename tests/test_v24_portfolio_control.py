"""V2.4 — portfolio control pipeline at the overlay seam (batch L).

Mode matrix against the REAL overlay path (graph shell + stubbed loaders):

* legacy (default) — resolver never runs, no rows, byte-identical ticket;
* shadow — full pipeline computed + rendered as a SHADOW section, recorded
  on state, ZERO ticket/snapshot/position rows, ticket stays CANDIDATE;
* enforced — resolver authoritative: final_size/status/margin_mode/snapshot
  id land on the ticket, the five rule names ride risk_decision_ids, and
  the final ticket + snapshot + position commit atomically. A gross-breach
  scenario proves RESIZE; a desired_side=SHORT decision proves the signed
  path (negative signed_weight position); same-symbol replace semantics
  prove the retry invariant (one open position per symbol).

Ledger-level cases pin the retry invariant's other half — an EXACT
same-ticket replay is a full no-op — plus the all-or-nothing transaction
rollback and the instrument_class ledger roundtrip (no operator file).
"""

from __future__ import annotations

import pytest

from yialpha.graph.trading_graph import YiAlphaGraph
from yialpha.ledger.evidence import register_run
from yialpha.ledger.portfolio import commit_final_ticket, new_snapshot_id, open_positions
from yialpha.ledger.run_context import (
    reset_ledger_run_context,
    set_ledger_run_context,
)
from yialpha.tickets import TicketStatus

_TRADE_DATE = "2026-09-03"
_RUN_ID = "RV24CTRL0001"


def _make_graph(tmp_path, **config_over) -> YiAlphaGraph:
    g = YiAlphaGraph.__new__(YiAlphaGraph)
    g.config = {
        "risk_enabled": True,
        "kelly_fraction": 0.25,
        "max_single_position": 0.20,
        "max_single_sector": 0.30,
        "max_drawdown_hard_stop": 0.15,
        "atr_stop_mult": 2.0,
        "results_dir": str(tmp_path),
    }
    g.config.update(config_over)
    g._risk_overlay_degraded = False
    g.risk_manager = g._build_risk_manager()
    g.ticker = "BTCUSDT"
    return g


def _perp_state(**over) -> dict:
    base = {
        "final_trade_decision": "**Rating**: Buy\n\nThesis.",
        "pm_rating": "Buy",
        "pm_decision_fields": {"price_target": 70.0, "confidence": 0.7},
    }
    base.update(over)
    return base


def _stub_prices(g, monkeypatch) -> None:
    monkeypatch.setattr(
        g, "_latest_close_and_atr", lambda t, d, at="stock": (60.0, 1.5)
    )
    monkeypatch.setattr(g, "_trailing_funding_total", lambda t, d: None)
    monkeypatch.setattr(g, "_latest_mark_close", lambda t, d: None)


def _seed_open_position(symbol: str, weight: float, side: str = "LONG") -> None:
    commit_final_ticket(
        ticket_payload={"ticket_id": f"T-{symbol}", "status": "APPROVED"},
        ticket_id=f"T-{symbol}",
        decision_id=None,
        run_id=None,
        ticket_version="v1",
        snapshot_payload={"positions": []},
        snapshot_id=new_snapshot_id(f"seed-{symbol}"),
        position_symbol=symbol,
        position_side=side,
        position_signed_weight=weight,
        open_position=True,
    )


@pytest.fixture(autouse=True)
def _clean_run_context():
    reset_ledger_run_context()
    yield
    reset_ledger_run_context()


@pytest.mark.unit
def test_legacy_mode_runs_no_pipeline_and_writes_nothing(tmp_path, monkeypatch):
    g = _make_graph(tmp_path)
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE, _perp_state(), {"equity": 100_000},
        asset_type="crypto_perp",
    )
    assert "PORTFOLIO CONTROL" not in out["final_trade_decision"]
    assert "portfolio_control" not in out
    ticket = out["execution_ticket"]
    assert ticket["final_size"] is None
    assert ticket["status"] == TicketStatus.CANDIDATE.value
    assert ticket["risk_decision_ids"] == []
    assert open_positions() == []


@pytest.mark.unit
def test_shadow_mode_computes_and_renders_but_writes_nothing(tmp_path, monkeypatch):
    register_run(_RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TRADE_DATE)
    set_ledger_run_context(
        _RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TRADE_DATE
    )
    g = _make_graph(tmp_path, portfolio_control_mode="shadow")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE, _perp_state(), {"equity": 100_000},
        asset_type="crypto_perp",
    )
    md = out["final_trade_decision"]
    assert "[PORTFOLIO CONTROL — SHADOW, not enforced]" in md
    assert "portfolio_control_shadow" in out
    shadow = out["portfolio_control_shadow"]
    assert shadow["side"] == "LONG"
    assert shadow["proposed_size"] > 0.0
    assert shadow["resolver"]["action"] == "APPROVED"  # empty book, small size
    assert [d["rule"] for d in shadow["decisions"]] == [
        "global_gross", "asset_class", "single_concentration",
        "directional_concentration", "correlation_cluster",
    ]
    # Ticket untouched; no snapshot/position rows.
    ticket = out["execution_ticket"]
    assert ticket["status"] == TicketStatus.CANDIDATE.value
    assert ticket["final_size"] is None
    assert open_positions() == []


@pytest.mark.unit
def test_enforced_approved_commits_ticket_snapshot_position(tmp_path, monkeypatch):
    register_run(_RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TRADE_DATE)
    set_ledger_run_context(
        _RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TRADE_DATE
    )
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE, _perp_state(), {"equity": 100_000},
        asset_type="crypto_perp",
    )
    md = out["final_trade_decision"]
    assert "[PORTFOLIO CONTROL — ENFORCED]" in md
    ticket = out["execution_ticket"]
    assert ticket["status"] == TicketStatus.APPROVED.value
    assert ticket["final_size"] == pytest.approx(ticket["proposed_size"])
    assert ticket["final_size"] > 0.0
    assert ticket["margin_mode"] == "ISOLATED"
    assert len(ticket["risk_decision_ids"]) == 5
    assert ticket["portfolio_snapshot_id"].startswith("S")
    # The atomic commit landed: one open LONG position, snapshot readable.
    (position,) = open_positions()
    assert position["symbol"] == "BTCUSDT"
    assert position["side"] == "LONG"
    assert position["signed_weight"] == pytest.approx(ticket["final_size"])
    assert position["signed_weight"] > 0.0
    from yialpha.ledger.portfolio import snapshot_by_id

    snap = snapshot_by_id(ticket["portfolio_snapshot_id"])
    assert snap is not None and snap["mode"] == "enforced"
    assert len(snap["decisions"]) == 5
    # The FINAL payload won the ticket write (not the earlier CANDIDATE).
    from yialpha.ledger.sqlite import get_connection

    row = get_connection(readonly=True).execute(
        "SELECT payload FROM tickets WHERE ticket_id = ?", (ticket["ticket_id"],)
    ).fetchone()
    assert '"status":"APPROVED"' in row[0].replace(" ", "")


@pytest.mark.unit
def test_enforced_resizes_when_gross_breaches(tmp_path, monkeypatch):
    register_run(_RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TRADE_DATE)
    set_ledger_run_context(
        _RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TRADE_DATE
    )
    # Existing book near the gross cap: candidate must be clipped.
    _seed_open_position("ETHUSDT", 0.60)
    _seed_open_position("SOLUSDT", 0.30)
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE, _perp_state(), {"equity": 100_000},
        asset_type="crypto_perp",
    )
    ticket = out["execution_ticket"]
    assert ticket["status"] == TicketStatus.RESIZED.value
    assert ticket["final_size"] < ticket["proposed_size"]
    assert "global_gross" in ticket["risk_decision_ids"]


@pytest.mark.unit
def test_enforced_short_via_explicit_desired_side(tmp_path, monkeypatch):
    register_run(_RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TRADE_DATE)
    set_ledger_run_context(
        _RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TRADE_DATE
    )
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE,
        _perp_state(
            final_trade_decision="**Rating**: Sell\n\nThesis.",
            pm_rating="Sell",
            # The ONLY way a short may open: the explicit structured field.
            pm_decision_fields={"desired_side": "SHORT", "confidence": 0.6},
        ),
        {"equity": 100_000},
        asset_type="crypto_perp",
    )
    ticket = out["execution_ticket"]
    assert ticket["status"] == TicketStatus.APPROVED.value
    (position,) = open_positions()
    assert position["side"] == "SHORT"
    # Interim V2.4 short sizing: kelly_fraction(0.25) x max_single(0.20).
    assert position["signed_weight"] == pytest.approx(-0.05)


@pytest.mark.unit
def test_enforced_sell_rating_alone_never_opens_a_short(tmp_path, monkeypatch):
    register_run(_RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TRADE_DATE)
    set_ledger_run_context(
        _RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TRADE_DATE
    )
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE,
        _perp_state(
            final_trade_decision="**Rating**: Sell\n\nThesis.",
            pm_rating="Sell",
            pm_decision_fields={},
        ),
        {"equity": 100_000},
        asset_type="crypto_perp",
    )
    # Sell maps to CLOSE (FLAT entry candidate): no position opened.
    assert open_positions() == []


@pytest.mark.unit
def test_enforced_veto_zeroes_the_size_and_skips_position(
    tmp_path, monkeypatch
):
    register_run(_RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TRADE_DATE)
    set_ledger_run_context(
        _RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TRADE_DATE
    )
    import yialpha.risk.constraints as constraints_module

    def _vetoing(candidate, snapshot, limits):  # noqa: ARG001
        return [
            constraints_module.RiskDecision(
                rule="global_gross", action="VETO", multiplier=0.0,
                reasons=["test veto"], metrics={},
            )
        ]

    monkeypatch.setattr(constraints_module, "evaluate_constraints", _vetoing)
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE, _perp_state(), {"equity": 100_000},
        asset_type="crypto_perp",
    )
    ticket = out["execution_ticket"]
    assert ticket["status"] == TicketStatus.VETOED.value
    assert ticket["final_size"] == 0.0
    assert open_positions() == []


@pytest.mark.unit
def test_enforced_same_symbol_retry_replaces_not_stacks(tmp_path, monkeypatch):
    register_run(_RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TRADE_DATE)
    set_ledger_run_context(
        _RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TRADE_DATE
    )
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE, _perp_state(), {"equity": 100_000},
        asset_type="crypto_perp",
    )
    g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE, _perp_state(), {"equity": 100_000},
        asset_type="crypto_perp",
    )
    # One OPEN position per symbol (the retry closed the first), and the
    # closed row remains as audit trail.
    from yialpha.ledger.sqlite import get_connection

    open_now = open_positions()
    assert len(open_now) == 1 and open_now[0]["symbol"] == "BTCUSDT"
    total = get_connection(readonly=True).execute(
        "SELECT COUNT(*) AS n FROM positions WHERE symbol = ?", ("BTCUSDT",)
    ).fetchone()["n"]
    assert total == 2


@pytest.mark.unit
def test_enforced_exact_same_ticket_retry_is_noop(monkeypatch):
    """An EXACT same-ticket replay is a complete no-op (repro Case 2).

    Byte-identical commit arguments used twice: the second call must not
    even close-and-reopen — ``open_positions()`` and the positions table
    stay byte-identical. Runs against a private in-memory SQLite with
    ``ledger_exists`` stubbed to a CONSTANT True, proving the replay gate
    reads the positions table itself and cannot lean on the file-existence
    helper (which an isolated harness may legitimately force True).
    """
    import sqlite3
    from contextlib import contextmanager

    import yialpha.ledger.portfolio as portfolio_module

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE tickets (ticket_id TEXT PRIMARY KEY, decision_id TEXT, "
        "run_id TEXT, payload TEXT, ticket_version TEXT, written_at TEXT);"
        "CREATE TABLE portfolio_snapshots (snapshot_id TEXT PRIMARY KEY, "
        "run_id TEXT, payload TEXT, created_at TEXT);"
        "CREATE TABLE positions (position_id TEXT PRIMARY KEY, ticket_id TEXT, "
        "run_id TEXT, symbol TEXT, side TEXT, signed_weight REAL, opened_at TEXT, "
        "closed_at TEXT, payload TEXT);"
    )

    @contextmanager
    def memory_transaction():
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            yield cursor
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            cursor.close()

    monkeypatch.setattr(portfolio_module, "ledger_transaction", memory_transaction)
    monkeypatch.setattr(portfolio_module, "ledger_exists", lambda: True)
    monkeypatch.setattr(
        portfolio_module, "get_connection", lambda *, readonly=False: conn
    )
    args = {
        "ticket_payload": {
            "ticket_id": "T-RETRY-1",
            "status": "APPROVED",
            "instrument_class": "pure_crypto_perp",
        },
        "ticket_id": "T-RETRY-1",
        "decision_id": None,
        "run_id": None,
        "ticket_version": "v1",
        "snapshot_payload": {"positions": []},
        "snapshot_id": "S-RETRY-1",
        "position_symbol": "BTCUSDT",
        "position_side": "LONG",
        "position_signed_weight": 0.12,
        "open_position": True,
        "close_existing": False,
    }
    try:
        assert commit_final_ticket(**args) is True
        before = open_positions()
        assert len(before) == 1

        assert commit_final_ticket(**args) is True
        # Same single open position (not closed by its own retry), and no
        # second row stacked.
        assert open_positions() == before
        assert conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 1
    finally:
        conn.close()


@pytest.mark.unit
def test_commit_final_ticket_transaction_rollback_on_error(monkeypatch):
    """A failure inside the commit transaction leaves no residue anywhere.

    The ticket and snapshot INSERTs succeed, then the position INSERT
    explodes mid-transaction: the single ``BEGIN IMMEDIATE`` boundary must
    roll ALL three tables back (``open_positions()`` empty) and the record
    stage must return False instead of raising.
    """
    from contextlib import contextmanager

    import yialpha.ledger.portfolio as portfolio_module
    from yialpha.ledger.sqlite import get_connection

    real_transaction = portfolio_module.ledger_transaction

    class _CursorFailingOnPositionsInsert:
        """Forwards to the real cursor; explodes on the positions INSERT."""

        def __init__(self, cur):
            self._cur = cur

        def execute(self, sql, params=()):
            if "INSERT OR IGNORE INTO positions" in sql:
                raise RuntimeError("forced mid-transaction failure")
            return self._cur.execute(sql, params)

    @contextmanager
    def failing_transaction():
        with real_transaction() as cur:
            yield _CursorFailingOnPositionsInsert(cur)

    monkeypatch.setattr(portfolio_module, "ledger_transaction", failing_transaction)

    committed = commit_final_ticket(
        ticket_payload={"ticket_id": "T-BOOM-1", "status": "APPROVED"},
        ticket_id="T-BOOM-1",
        decision_id=None,
        run_id=None,
        ticket_version="v1",
        snapshot_payload={"positions": []},
        snapshot_id="S-BOOM-1",
        position_symbol="BTCUSDT",
        position_side="LONG",
        position_signed_weight=0.10,
        open_position=True,
    )
    assert committed is False
    counts = get_connection(readonly=True).execute(
        "SELECT (SELECT COUNT(*) FROM tickets) AS t, "
        "(SELECT COUNT(*) FROM portfolio_snapshots) AS s, "
        "(SELECT COUNT(*) FROM positions) AS p"
    ).fetchone()
    assert (counts["t"], counts["s"], counts["p"]) == (0, 0, 0)
    assert open_positions() == []


@pytest.mark.unit
def test_ledger_position_carries_instrument_class_without_operator_file():
    """A position's instrument class survives the ledger roundtrip alone.

    No operator positions file exists (conftest blanks
    ``portfolio_positions_file``), so the class must ride the commit itself:
    first only inside ticket_payload (the repro Case 3 mirror — read back
    with the exact graph extraction expression), then via the explicit
    ``instrument_class`` kwarg, which also stamps the class onto a ticket
    mirror whose payload carried none.
    """
    commit_final_ticket(
        ticket_payload={
            "ticket_id": "T-CLASS-PAYLOAD",
            "status": "APPROVED",
            "instrument_class": "pure_crypto_perp",
        },
        ticket_id="T-CLASS-PAYLOAD",
        decision_id=None,
        run_id=None,
        ticket_version="v1",
        snapshot_payload={"positions": []},
        snapshot_id=new_snapshot_id("class-payload"),
        position_symbol="BTCUSDT",
        position_side="LONG",
        position_signed_weight=0.10,
        open_position=True,
    )
    commit_final_ticket(
        ticket_payload={"ticket_id": "T-CLASS-KWARG", "status": "APPROVED"},
        ticket_id="T-CLASS-KWARG",
        decision_id=None,
        run_id=None,
        ticket_version="v1",
        snapshot_payload={"positions": []},
        snapshot_id=new_snapshot_id("class-kwarg"),
        position_symbol="ETHUSDT",
        position_side="LONG",
        position_signed_weight=0.08,
        open_position=True,
        instrument_class="pure_crypto_perp",
    )
    by_symbol = {p["symbol"]: p for p in open_positions()}
    # Case 3 mirror: the exact production extraction expression (fallback
    # and all) sees the payload-borne class on the next graph snapshot.
    observed = str(
        (by_symbol["BTCUSDT"].get("payload") or {}).get("instrument_class")
        or "unknown_perp"
    )
    assert observed == "pure_crypto_perp"
    # Explicit kwarg path: same landing on the position row.
    assert (
        by_symbol["ETHUSDT"]["payload"]["instrument_class"] == "pure_crypto_perp"
    )
    # The kwarg also stamped the ticket mirror (its payload had no class).
    from yialpha.ledger.sqlite import get_connection

    mirrored = get_connection(readonly=True).execute(
        "SELECT payload FROM tickets WHERE ticket_id = ?", ("T-CLASS-KWARG",)
    ).fetchone()
    assert '"instrument_class":"pure_crypto_perp"' in mirrored[0].replace(" ", "")


@pytest.mark.unit
def test_enforced_non_perp_run_stays_legacy(tmp_path, monkeypatch):
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "AAPL", _TRADE_DATE,
        {
            "final_trade_decision": "**Rating**: Buy\n\nThesis.",
            "pm_rating": "Buy",
            "pm_decision_fields": {"price_target": 220.0},
        },
        {"equity": 100_000},
        asset_type="stock",
    )
    assert "PORTFOLIO CONTROL" not in out["final_trade_decision"]
    assert open_positions() == []
