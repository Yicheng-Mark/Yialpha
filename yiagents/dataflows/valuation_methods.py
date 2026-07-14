"""Deterministic intrinsic-value math (pure functions, no I/O, no LLM).

The fundamentals analyst is an LLM, and LLMs confabulate arithmetic -- citing a
"Graham number" or a "margin of safety" that does not follow from the reported
figures. This module is the deterministic counterpart: the canonical intrinsic-
value formulas of public, textbook finance (Graham's *Intelligent Investor*,
Lynch's *One Up on Wall Street*, Buffett's 1986 owner-earnings letter,
Damodaran's DCF/WACC), each as a pure function of explicit numeric inputs.

Everything is in ``float | None``: a guard that cannot be satisfied (negative
book value, ``r <= g`` in a terminal model, non-positive price) returns ``None``
rather than a misleading number, so callers surface "n/a" instead of a
fabricated value. No vendor code, no network, no caching -- the math only.

These are intentionally NOT named after any investor (the persona-prompting
literature, REFERENCES.md #18, shows named-persona framing hurts reasoning);
the formulas are credited to their sources in the docstrings and in
REFERENCES.md, never in the call surface.
"""

from __future__ import annotations

import math

__all__ = [
    "pe_ratio",
    "earnings_yield",
    "graham_number",
    "net_current_asset_value_per_share",
    "peg_ratio",
    "owner_earnings",
    "intrinsic_value_two_stage_dcf",
    "weighted_average_cost_of_capital",
    "margin_of_safety",
]


def _is_finite_number(*xs: float) -> bool:
    """True iff every argument is a real finite number (rejects NaN/inf/None)."""
    for x in xs:
        if x is None:
            return False
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(xf):
            return False
    return True


def pe_ratio(price: float, eps: float) -> float | None:
    """Price-to-earnings ratio ``price / eps``.

    Returns ``None`` when ``eps <= 0`` (P/E is undefined for loss-makers; a
    negative P/E is not a valuation signal, it is a flag).
    """
    if not _is_finite_number(price, eps) or eps <= 0 or price <= 0:
        return None
    return float(price) / float(eps)


def earnings_yield(price: float, eps: float) -> float | None:
    """Inverse P/E, ``eps / price`` -- the earnings return on the price paid.

    Returns ``None`` when ``price <= 0``; a negative ``eps`` is preserved (a
    negative yield is itself a signal, unlike a negative P/E).
    """
    if not _is_finite_number(price, eps) or price <= 0:
        return None
    return float(eps) / float(price)


def graham_number(eps: float, book_value_per_share: float) -> float | None:
    """Benjamin Graham's original number: ``sqrt(22.5 * EPS * BVPS)``.

    The 22.5 factor is Graham's ceiling of 15x earnings times 1.5x book value
    (*The Intelligent Investor*, revised ch. 14 / *Security Analysis*). The
    formula is defined only when both earnings and book value are positive;
    otherwise it returns ``None`` (Graham's own precondition -- the number is a
    "screen", not a rescue for impaired balance sheets).
    """
    if (not _is_finite_number(eps, book_value_per_share)
            or eps <= 0 or book_value_per_share <= 0):
        return None
    return math.sqrt(22.5 * float(eps) * float(book_value_per_share))


def net_current_asset_value_per_share(
    current_assets: float,
    total_liabilities: float,
    shares_outstanding: float,
) -> float | None:
    """Graham's "net-net": ``(current_assets - total_liabilities) / shares``.

    Graham's deepest-value screen bought stocks at <= 2/3 of net current asset
    value (current assets minus *all* liabilities), treating fixed assets as
    worth zero (*Security Analysis*). Per-share form lets the caller compare to
    price directly. Returns ``None`` when shares are not a positive number.
    """
    if (not _is_finite_number(current_assets, total_liabilities, shares_outstanding)
            or shares_outstanding <= 0):
        return None
    return (float(current_assets) - float(total_liabilities)) / float(shares_outstanding)


def peg_ratio(pe_ratio_value: float, earnings_growth_pct: float) -> float | None:
    """Peter Lynch's PEG: ``PE / earnings_growth_%``.

    Growth is expressed in percentage points (e.g. 15 for 15%); PEG < 1 is
    Lynch's "reasonably attractive" line (*One Up on Wall Street*, ch. 15).
    Returns ``None`` when growth is <= 0 (PEG is meaningless for shrinking or
    flat earnings -- a negative-growth PEG inverts the signal) or PE is invalid.
    """
    if (not _is_finite_number(pe_ratio_value, earnings_growth_pct)
            or pe_ratio_value <= 0 or earnings_growth_pct <= 0):
        return None
    return float(pe_ratio_value) / float(earnings_growth_pct)


def owner_earnings(
    net_income: float,
    depreciation_amortization: float,
    maintenance_capex: float,
) -> float | None:
    """Warren Buffett's owner earnings: ``net_income + D&A - maintenance_capex``.

    From Buffett's 1986 Berkshire letter: accounting earnings overstate the
   cash a business actually generates for its owners by the depreciation it
    adds back but the maintenance capex it buries. This is the number a DCF
    should discount, not reported net income. ``maintenance_capex`` is often
    approximated by total capex when the former is not disclosed.
    """
    if not _is_finite_number(net_income, depreciation_amortization, maintenance_capex):
        return None
    return (float(net_income) + float(depreciation_amortization)
            - float(maintenance_capex))


def intrinsic_value_two_stage_dcf(
    free_cash_flow_per_share: float,
    growth_rate: float,
    discount_rate: float,
    terminal_growth_rate: float,
    high_growth_years: int = 10,
) -> float | None:
    """Two-stage discounted-cash-flow intrinsic value per share (Damodaran).

    Discounts ``high_growth_years`` of FCF growing at ``growth_rate``, then a
    Gordon-growth terminal at ``terminal_growth_rate``, all at ``discount_rate``
    (all rates as decimals, e.g. 0.10 for 10%). Returns ``None`` when the model
    is ill-posed: ``discount_rate <= terminal_growth_rate`` (the terminal sum
    diverges), a non-positive horizon, or any non-finite input.
    """
    if (not _is_finite_number(free_cash_flow_per_share, growth_rate,
                             discount_rate, terminal_growth_rate)
            or discount_rate <= terminal_growth_rate
            or not isinstance(high_growth_years, int)
            or high_growth_years <= 0):
        return None

    fcf = float(free_cash_flow_per_share)
    g = float(growth_rate)
    r = float(discount_rate)
    gt = float(terminal_growth_rate)
    n = high_growth_years

    pv_explicit = 0.0
    flow = fcf
    for year in range(1, n + 1):
        flow *= (1.0 + g)
        pv_explicit += flow / ((1.0 + r) ** year)

    terminal_flow = fcf * ((1.0 + g) ** n) * (1.0 + gt)
    terminal_value = terminal_flow / (r - gt)
    pv_terminal = terminal_value / ((1.0 + r) ** n)

    return pv_explicit + pv_terminal


def weighted_average_cost_of_capital(
    market_value_equity: float,
    market_value_debt: float,
    cost_of_equity: float,
    after_tax_cost_of_debt: float,
) -> float | None:
    """WACC = ``E/(E+D) * Re + D/(E+D) * Rd`` (Damodaran).

    ``after_tax_cost_of_debt`` is already tax-adjusted (``Rd * (1 - tax)``) so
    the caller controls the tax rate. Returns ``None`` when total capital is
    non-positive.
    """
    if (not _is_finite_number(market_value_equity, market_value_debt,
                             cost_of_equity, after_tax_cost_of_debt)):
        return None
    total = float(market_value_equity) + float(market_value_debt)
    if total <= 0:
        return None
    e = float(market_value_equity)
    d = float(market_value_debt)
    return (e / total) * float(cost_of_equity) + (d / total) * float(after_tax_cost_of_debt)


def margin_of_safety(intrinsic_value: float, price: float) -> float | None:
    """ ``(intrinsic_value - price) / intrinsic_value``.

    Positive means the asset trades below its estimated intrinsic value (a
    discount / margin of safety); negative means a premium. Returns ``None``
    when ``intrinsic_value`` is zero (the ratio is undefined).
    """
    if not _is_finite_number(intrinsic_value, price) or intrinsic_value == 0:
        return None
    return (float(intrinsic_value) - float(price)) / float(intrinsic_value)


def summarize(
    price: float,
    eps: float | None = None,
    book_value_per_share: float | None = None,
    current_assets: float | None = None,
    total_liabilities: float | None = None,
    shares_outstanding: float | None = None,
    earnings_growth_pct: float | None = None,
    net_income: float | None = None,
    depreciation_amortization: float | None = None,
    maintenance_capex: float | None = None,
    free_cash_flow_per_share: float | None = None,
    growth_rate: float | None = None,
    discount_rate: float | None = None,
    terminal_growth_rate: float | None = None,
    high_growth_years: int = 10,
    market_value_equity: float | None = None,
    market_value_debt: float | None = None,
    cost_of_equity: float | None = None,
    after_tax_cost_of_debt: float | None = None,
) -> dict[str, float | None]:
    """Compute every metric whose inputs are present, keyed by name.

    Each metric is computed only when all of *its* required inputs are finite;
    the rest come back as ``None``. This is the single call surface the
    valuation tool exposes: the analyst supplies the line items it has, and
    gets back the subset of metrics they support -- never a fabricated number.
    """
    out: dict[str, float | None] = {}
    pe = pe_ratio(price, eps) if _is_finite_number(price, eps) else None
    out["pe_ratio"] = pe
    out["earnings_yield"] = (
        earnings_yield(price, eps) if _is_finite_number(price, eps) else None
    )
    out["graham_number"] = (
        graham_number(eps, book_value_per_share)
        if _is_finite_number(eps, book_value_per_share) else None
    )
    out["ncav_per_share"] = (
        net_current_asset_value_per_share(
            current_assets, total_liabilities, shares_outstanding)
        if _is_finite_number(current_assets, total_liabilities, shares_outstanding)
        else None
    )
    out["peg_ratio"] = (
        peg_ratio(pe if pe is not None else 0.0, earnings_growth_pct)
        if (pe is not None and _is_finite_number(earnings_growth_pct)) else None
    )
    out["owner_earnings"] = (
        owner_earnings(net_income, depreciation_amortization, maintenance_capex)
        if _is_finite_number(net_income, depreciation_amortization, maintenance_capex)
        else None
    )
    out["intrinsic_value_dcf"] = (
        intrinsic_value_two_stage_dcf(
            free_cash_flow_per_share, growth_rate, discount_rate,
            terminal_growth_rate, high_growth_years)
        if _is_finite_number(free_cash_flow_per_share, growth_rate,
                             discount_rate, terminal_growth_rate)
        else None
    )
    out["wacc"] = (
        weighted_average_cost_of_capital(
            market_value_equity, market_value_debt, cost_of_equity,
            after_tax_cost_of_debt)
        if _is_finite_number(market_value_equity, market_value_debt,
                             cost_of_equity, after_tax_cost_of_debt)
        else None
    )
    iv = out["intrinsic_value_dcf"]
    out["margin_of_safety_dcf"] = (
        margin_of_safety(iv, price) if (iv is not None) else None
    )
    return out
