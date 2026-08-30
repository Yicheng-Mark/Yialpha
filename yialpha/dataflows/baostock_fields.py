"""BaoStock quarterly-statement field tables (server-delivered field names).

BaoStock's ``query_*_data`` statement endpoints deliver their field list from
the server (the client library has no local field table), so the exact field
names must be pinned here against the official documentation
(https://www.baostock.com/baostock/index.php/Python%20API%E6%96%87%E6%A1%A3,
季频盈利能力 / 季频偿债能力 / 季频现金流量 sections). A renderer reading a
field that the endpoint never returns silently shows ``n/a`` in every cell —
exactly the all-``n/a`` table bug this module exists to prevent.

The three ``*_FIELDS`` tuples below mirror the official return tables verbatim
(all three endpoints also deliver ``code`` / ``pubDate`` / ``statDate``); the
``*_COLUMNS`` specs are the ONLY field references the renderers in
:mod:`yialpha.dataflows.baostock_vendor` may use. The consistency test in
``tests/test_a_share_native.py`` asserts every spec field is a member of its
statement's field table.

Units (from the official sample rows):

* ratio/margin fields (``roeAvg`` 0.074617, ``npMargin`` 0.342179,
  ``liabilityToAsset`` 0.933703, ``YOYLiability`` 0.100020, ``CFOToOR``
  -3.071550, ...) are **decimal fractions** — renderers multiply by 100 and
  append "%" (``kind="pct"``);
* multiple fields (``currentRatio``, ``assetToEquity`` 15.083598,
  ``ebitToInterest``, ``CFOToNP`` -8.976439, ...) are dimensionless
  multiples — rendered as-is with an "x" suffix (``kind="x"``);
* CNY amounts (``netProfit``, ``MBRevenue``) are raw 元; share counts
  (``totalShare``, ``liqaShare``) are raw 股; EPS is 元 (``kind="cny"`` /
  ``kind="num"``).
"""

from __future__ import annotations

from typing import NamedTuple


class StatementColumn(NamedTuple):
    """One rendered statement column: BaoStock field -> table label + format."""

    field: str          # exact BaoStock field name (must be in *_FIELDS)
    label: str          # rendered table header
    kind: str           # "cny" | "pct" | "x" | "num" (see module docstring)


# --- query_profit_data (季频盈利能力) ------------------------------------- #
# Official fields: code, pubDate, statDate, roeAvg, npMargin, gpMargin,
# netProfit, epsTTM, MBRevenue, totalShare, liqaShare.
PROFIT_DATA_FIELDS: tuple[str, ...] = (
    "code", "pubDate", "statDate", "roeAvg", "npMargin", "gpMargin",
    "netProfit", "epsTTM", "MBRevenue", "totalShare", "liqaShare",
)

PROFIT_COLUMNS: tuple[StatementColumn, ...] = (
    StatementColumn("MBRevenue", "Revenue", "cny"),
    StatementColumn("netProfit", "NetProfit", "cny"),
    StatementColumn("gpMargin", "GrossMargin%", "pct"),
    StatementColumn("npMargin", "NetMargin%", "pct"),
    StatementColumn("roeAvg", "ROE%", "pct"),
    StatementColumn("epsTTM", "EPS-TTM", "num"),
    StatementColumn("totalShare", "TotalShares", "cny"),
    StatementColumn("liqaShare", "LiqaShares", "cny"),
)


# --- query_balance_data (季频偿债能力) ------------------------------------ #
# Official fields: code, pubDate, statDate, currentRatio, quickRatio,
# cashRatio, YOYLiability, liabilityToAsset, assetToEquity. NOTE: this
# endpoint returns SOLVENCY RATIOS only — no balance-sheet stocks such as
# totalAssets/totalLiabilities/equity exist, so the renderer must not pretend
# to show them (see the renderer docstring).
BALANCE_DATA_FIELDS: tuple[str, ...] = (
    "code", "pubDate", "statDate", "currentRatio", "quickRatio", "cashRatio",
    "YOYLiability", "liabilityToAsset", "assetToEquity",
)

BALANCE_COLUMNS: tuple[StatementColumn, ...] = (
    StatementColumn("currentRatio", "CurrentRatio", "x"),
    StatementColumn("quickRatio", "QuickRatio", "x"),
    StatementColumn("cashRatio", "CashRatio", "x"),
    StatementColumn("YOYLiability", "LiabYoY%", "pct"),
    StatementColumn("liabilityToAsset", "LiabToAsset%", "pct"),
    StatementColumn("assetToEquity", "AssetToEquity", "x"),
)


# --- query_cash_flow_data (季频现金流量) ---------------------------------- #
# Official fields: code, pubDate, statDate, CAToAsset, NCAToAsset,
# tangibleAssetToAsset, ebitToInterest, CFOToOR, CFOToNP, CFOToGr. NOTE:
# like query_balance_data this returns QUALITY RATIOS only — no absolute
# operating/investing/financing cash-flow amounts exist.
CASH_FLOW_DATA_FIELDS: tuple[str, ...] = (
    "code", "pubDate", "statDate", "CAToAsset", "NCAToAsset",
    "tangibleAssetToAsset", "ebitToInterest", "CFOToOR", "CFOToNP", "CFOToGr",
)

CASH_FLOW_COLUMNS: tuple[StatementColumn, ...] = (
    StatementColumn("CAToAsset", "CA/Asset%", "pct"),
    StatementColumn("NCAToAsset", "NCA/Asset%", "pct"),
    StatementColumn("tangibleAssetToAsset", "TangAsset%", "pct"),
    StatementColumn("ebitToInterest", "EBIT/Interest", "x"),
    StatementColumn("CFOToOR", "CFO/Revenue%", "pct"),
    StatementColumn("CFOToNP", "CFO/NetProfit", "x"),
    StatementColumn("CFOToGr", "CFO/GrRev%", "pct"),
)
