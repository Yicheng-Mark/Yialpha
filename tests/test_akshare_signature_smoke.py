"""Signature smoke tests for the akshare / baostock vendor call sites (A6).

Root cause this file guards against
-----------------------------------
Four separate incidents share one shape: the vendor code called a real library
function with a kwarg it does not accept (``stock_hsgt_individual_em(stock=...)``
-> real param is ``symbol``), or read DataFrame/server fields that do not exist
(``totalAssets``, ``npParentCompanyOwners``, ``netCFOperate`` ...). The unit
tests mocked the library with the SAME wrong names, so the mocks mirrored the
bug and everything looked green.

This file breaks that symmetry with two complementary layers:

1. **Static call-site audit** (runs everywhere, no optional deps): the exact
   kwargs the vendor source passes to every ``ak.*`` / ``bs.*`` call must equal
   a pinned table, and every statement field the renderers reference directly
   must be a member of the official field tables in
   :mod:`yialpha.dataflows.baostock_fields`. Adding a new call site or field
   without pinning it here fails the audit.
2. **Runtime signature check** (skipped unless the real package is installed):
   ``inspect.signature`` proves the pinned kwarg table is accepted by the REAL
   akshare / baostock package — the link a pure-mock test can never check.

The akshare/baostock packages are optional extras and are NOT installed in the
default dev venv, hence the ``pytest.importorskip`` guards; on machines (or CI
jobs) with the ``a-share`` extra the runtime layer executes for real.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from yialpha.dataflows import akshare_vendor as akv, baostock_fields as bsf, baostock_vendor as bsv

# ---------------------------------------------------------------------------
# Pinned call contracts: every kwarg the vendors pass to akshare functions.
# MUST be kept in sync with the call sites (the static audit enforces this).
# ---------------------------------------------------------------------------
EXPECTED_AKSHARE_KWARGS: dict[str, set[str]] = {
    "stock_news_em": {"symbol"},
    "stock_individual_fund_flow": {"stock", "market"},
    "stock_lhb_detail_em": {"start_date", "end_date"},
    # A1 regression: the real signature is ``symbol: str`` (verified in
    # akshare 1.18.83, stock_feature/stock_hsgt_em.py:1512); ``stock=`` raised
    # TypeError on every call.
    "stock_hsgt_individual_em": {"symbol"},
    "stock_sector_fund_flow_rank": {"indicator", "sector_type"},
    "stock_zh_a_spot_em": set(),
    "stock_zh_a_spot": set(),
}

# baostock functions called as ``bs.<name>(...)`` / ``self.bs.<name>(...)``
# (code/fields may be positional; only the keyword part is pinned here).
EXPECTED_BAOSTOCK_KWARGS: dict[str, set[str]] = {
    "login": set(),
    "logout": set(),
    # A3: the per-stock industry mapping (baostock 0.9.3,
    # security/sectorinfo.py:18 — params code, date).
    "query_stock_industry": {"code"},
    "query_history_k_data_plus": {"start_date", "end_date", "frequency",
                                  "adjustflag"},
}

# Statement endpoints reached through the ``getattr(bs, query_fn_name)``
# indirection in baostock_vendor._query_statement.
STATEMENT_QUERY_FNS: dict[str, set[str]] = {
    "query_profit_data": {"code", "year", "quarter"},
    "query_balance_data": {"code", "year", "quarter"},
    "query_cash_flow_data": {"code", "year", "quarter"},
}


def _dotted_name(node: ast.expr) -> str:
    """Dotted path of a Name/Attribute chain ('bs', 'self.bs', ...); '' else."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _method_calls(source: str, receiver: str) -> dict[str, set[str]]:
    """Map fn name -> keyword names for every ``<...>.receiver-ish.fn(...)``
    call in the module source. AST-based so docstrings/comments (e.g. the
    historical note about the removed ``ak.stock_board_industry_name_ths``
    call) cannot masquerade as call sites."""
    out: dict[str, set[str]] = {}
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        recv = _dotted_name(node.func.value)
        if recv != receiver and not recv.endswith("." + receiver):
            continue
        kws = {kw.arg for kw in node.keywords if kw.arg}
        out.setdefault(node.func.attr, set()).update(kws)
    return out


def _plain_calls(source: str, fn_name: str) -> list[set[str]]:
    """Keyword-name sets of every plain ``fn_name(...)`` call in the source."""
    out: list[set[str]] = []
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == fn_name):
            out.append({kw.arg for kw in node.keywords if kw.arg})
    return out


def _str_args_of(source: str, fn_name: str, arg_index: int) -> set[str]:
    """String-literal values at positional ``arg_index`` of every
    ``fn_name(...)`` call (non-literal args — e.g. spec variables — are
    skipped; those are covered by the COLUMNS-in-FIELDS check)."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == fn_name):
            continue
        if (arg_index < len(node.args)
                and isinstance(node.args[arg_index], ast.Constant)
                and isinstance(node.args[arg_index].value, str)):
            out.add(node.args[arg_index].value)
    return out


# --------------------------------------------------------------------------- #
# Layer 1: static call-site audit (no optional dependencies)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_akshare_call_sites_use_pinned_kwargs():
    """Every ``ak.<fn>(...)`` call in the vendor must pass exactly the pinned
    kwargs — an unlisted call site (new endpoint) or a renamed kwarg fails."""
    calls = _method_calls(inspect.getsource(akv), "ak")
    assert set(calls) == set(EXPECTED_AKSHARE_KWARGS), (
        "ak.* call sites and the pinned table diverge — update "
        "EXPECTED_AKSHARE_KWARGS (and verify the real signature) for: "
        f"{set(calls) ^ set(EXPECTED_AKSHARE_KWARGS)}")
    for name, kws in calls.items():
        assert kws == EXPECTED_AKSHARE_KWARGS[name], (
            f"ak.{name} call-site kwargs {sorted(kws)} diverge from the "
            f"pinned {sorted(EXPECTED_AKSHARE_KWARGS[name])}")


@pytest.mark.unit
def test_baostock_direct_call_sites_use_pinned_kwargs():
    """Every ``bs.<fn>(...)`` / ``self.bs.<fn>(...)`` call in both vendors must
    pass pinned kwargs."""
    for mod in (bsv, akv):
        calls = _method_calls(inspect.getsource(mod), "bs")
        assert set(calls) <= set(EXPECTED_BAOSTOCK_KWARGS), (
            f"{mod.__name__} calls unlisted baostock functions: "
            f"{set(calls) - set(EXPECTED_BAOSTOCK_KWARGS)}")
        for name, kws in calls.items():
            assert kws == EXPECTED_BAOSTOCK_KWARGS[name], (
                f"{mod.__name__}: bs.{name} kwargs {sorted(kws)} diverge "
                f"from the pinned {sorted(EXPECTED_BAOSTOCK_KWARGS[name])}")


@pytest.mark.unit
def test_baostock_statement_indirection_pinned():
    """The ``getattr(bs, query_fn_name)`` indirection must only reach the three
    statement endpoints, and the ``query_fn(...)`` call must pass exactly
    code/year/quarter (the params baostock 0.9.3 actually accepts)."""
    src = inspect.getsource(bsv)
    referenced = _str_args_of(src, "_statement_rows", 1)
    assert referenced == set(STATEMENT_QUERY_FNS), (
        "statement endpoint names diverge from the pinned table: "
        f"{referenced ^ set(STATEMENT_QUERY_FNS)}")
    kw_sets = _plain_calls(src, "query_fn")
    assert kw_sets == [{"code", "year", "quarter"}], kw_sets


@pytest.mark.unit
def test_baostock_renderer_direct_fields_in_field_tables():
    """Fields the renderers reference directly (summary lines, ``_fmt_cell``
    calls) must exist in the official field tables — a fabricated field name
    silently renders 'n/a' everywhere (the all-n/a statement-table bug)."""
    src = inspect.getsource(bsv)
    used = _str_args_of(src, "_fmt_cell", 1)
    all_fields = (set(bsf.PROFIT_DATA_FIELDS) | set(bsf.BALANCE_DATA_FIELDS)
                  | set(bsf.CASH_FLOW_DATA_FIELDS))
    assert used <= all_fields, (
        f"renderers reference non-existent baostock fields: {used - all_fields}")
    # And every rendered column spec is itself backed by the field tables
    # (duplicated here so a spec/table drift fails even if only this file runs).
    for columns, fields in (
        (bsf.PROFIT_COLUMNS, bsf.PROFIT_DATA_FIELDS),
        (bsf.BALANCE_COLUMNS, bsf.BALANCE_DATA_FIELDS),
        (bsf.CASH_FLOW_COLUMNS, bsf.CASH_FLOW_DATA_FIELDS),
    ):
        for col in columns:
            assert col.field in fields, col.field


# --------------------------------------------------------------------------- #
# Layer 2: runtime signature check against the REAL installed packages
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_real_akshare_signatures_accept_pinned_kwargs():
    """The pinned kwarg table must be accepted by the real akshare package.
    Skipped when akshare (optional extra) is not installed."""
    ak = pytest.importorskip("akshare")
    for name, kwargs in EXPECTED_AKSHARE_KWARGS.items():
        fn = getattr(ak, name, None)
        assert callable(fn), f"akshare no longer exports {name}"
        params = set(inspect.signature(fn).parameters)
        missing = kwargs - params
        assert not missing, (
            f"akshare.{name} does not accept {missing} "
            f"(params: {sorted(params)}) — the pinned table is stale")


@pytest.mark.unit
def test_real_baostock_signatures_accept_pinned_kwargs():
    """The pinned kwarg table must be accepted by the real baostock package.
    Skipped when baostock (optional extra) is not installed."""
    bs = pytest.importorskip("baostock")
    for table in (EXPECTED_BAOSTOCK_KWARGS, STATEMENT_QUERY_FNS):
        for name, kwargs in table.items():
            fn = getattr(bs, name, None)
            assert callable(fn), f"baostock no longer exports {name}"
            params = set(inspect.signature(fn).parameters)
            missing = kwargs - params
            assert not missing, (
                f"baostock.{name} does not accept {missing} "
                f"(params: {sorted(params)}) — the pinned table is stale")
