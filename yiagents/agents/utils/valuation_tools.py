"""LangChain ``@tool`` wrapper over the deterministic valuation engine.

The fundamentals analyst already pulls line items via ``get_fundamentals`` /
``get_balance_sheet`` / ``get_income_statement``. This tool is a Program-of-
Thought (REFERENCES.md #19) sink for those figures: the analyst passes the
numbers it has gathered and Python computes every intrinsic-value metric they
support -- Graham number, NCAV, PEG, owner earnings, two-stage DCF, WACC, and
the margin of safety versus the market price -- and returns them as a
deterministic markdown table. Delegating the arithmetic to Python removes the
class of LLM confabulation where a "margin of safety" is cited that the
reported figures do not support.

The tool is gated onto the fundamentals analyst by ``YIAGENTS_VALUATION_TOOLS``
(off by default = the analyst's tool list, and therefore its prompt, is
byte-for-byte unchanged when the flag is unset).
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool

from yiagents.dataflows.valuation_methods import summarize


def _fmt(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.4f}"


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.2f}%"


@tool
def get_valuation_metrics(
    price: Annotated[float, "current market price per share"],
    eps: Annotated[float | None, "diluted earnings per share (most recent reported period)"] = None,
    book_value_per_share: Annotated[float | None, "book value per share"] = None,
    current_assets: Annotated[float | None, "total current assets"] = None,
    total_liabilities: Annotated[float | None, "total liabilities"] = None,
    shares_outstanding: Annotated[float | None, "diluted shares outstanding"] = None,
    earnings_growth_pct: Annotated[float | None, "expected annual EPS growth in percentage points (e.g. 15 for 15%)"] = None,
    net_income: Annotated[float | None, "net income"] = None,
    depreciation_amortization: Annotated[float | None, "depreciation & amortization"] = None,
    maintenance_capex: Annotated[float | None, "maintenance capital expenditure (approximate with total capex if not disclosed)"] = None,
    free_cash_flow_per_share: Annotated[float | None, "free cash flow per share"] = None,
    growth_rate: Annotated[float | None, "DCF high-growth stage rate as a decimal (e.g. 0.10 for 10%)"] = None,
    discount_rate: Annotated[float | None, "DCF discount rate (WACC) as a decimal (e.g. 0.09 for 9%)"] = None,
    terminal_growth_rate: Annotated[float | None, "DCF terminal perpetuity growth as a decimal (e.g. 0.03 for 3%)"] = None,
    high_growth_years: Annotated[int, "DCF high-growth stage length in years"] = 10,
    market_value_equity: Annotated[float | None, "market capitalization"] = None,
    market_value_debt: Annotated[float | None, "market value of debt"] = None,
    cost_of_equity: Annotated[float | None, "cost of equity as a decimal"] = None,
    after_tax_cost_of_debt: Annotated[float | None, "after-tax cost of debt as a decimal"] = None,
) -> str:
    """Compute deterministic intrinsic-value metrics from supplied line items.

    Pass only the figures you have pulled from the fundamentals tools; every
    metric whose required inputs are missing returns ``n/a`` rather than a
    guess. Treat the returned table as the source of truth for any exact
    valuation number cited downstream (Graham number, PEG, DCF intrinsic value,
    margin of safety). Methods are public-textbook formulas (Graham, Lynch,
    Buffett owner earnings, Damodaran DCF/WACC).
    """
    out = summarize(
        price=price, eps=eps, book_value_per_share=book_value_per_share,
        current_assets=current_assets, total_liabilities=total_liabilities,
        shares_outstanding=shares_outstanding,
        earnings_growth_pct=earnings_growth_pct, net_income=net_income,
        depreciation_amortization=depreciation_amortization,
        maintenance_capex=maintenance_capex,
        free_cash_flow_per_share=free_cash_flow_per_share, growth_rate=growth_rate,
        discount_rate=discount_rate, terminal_growth_rate=terminal_growth_rate,
        high_growth_years=high_growth_years,
        market_value_equity=market_value_equity, market_value_debt=market_value_debt,
        cost_of_equity=cost_of_equity, after_tax_cost_of_debt=after_tax_cost_of_debt,
    )

    rows = [
        ("P/E ratio", _fmt(out["pe_ratio"])),
        ("Earnings yield", _fmt_pct(out["earnings_yield"])),
        ("Graham number", _fmt(out["graham_number"])),
        ("NCAV per share (net-net)", _fmt(out["ncav_per_share"])),
        ("PEG ratio", _fmt(out["peg_ratio"])),
        ("Owner earnings", _fmt(out["owner_earnings"])),
        ("DCF intrinsic value / share", _fmt(out["intrinsic_value_dcf"])),
        ("Margin of safety (vs DCF)", _fmt_pct(out["margin_of_safety_dcf"])),
        ("WACC", _fmt_pct(out["wacc"])),
    ]

    lines = [
        "## Deterministic valuation metrics",
        "",
        f"- Market price / share: {_fmt(price)}",
        "- Every metric is computed only from the line items supplied; "
        "``n/a`` means a required input was not provided (not zero).",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name, value in rows:
        lines.append(f"| {name} | {value} |")
    lines += [
        "",
        "Notes: Graham number requires positive EPS and book value; PEG "
        "requires positive earnings growth; DCF requires discount_rate > "
        "terminal_growth_rate. A positive margin of safety means the price is "
        "below the estimated intrinsic value.",
    ]
    return "\n".join(lines)
