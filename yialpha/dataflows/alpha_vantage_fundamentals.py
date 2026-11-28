import json

from .alpha_vantage_common import _make_api_request
from .symbol_utils import NoMarketDataError, normalize_symbol
from .utils import current_pit_end, is_filing_public, overview_would_leak_future


def _api_text(function_name: str, params: dict, ticker: str) -> dict | str:
    """One Alpha Vantage call whose blank body is a typed no-data, not a payload.

    ``_make_api_request`` returns whatever text the endpoint sent, so an empty
    200 body used to travel on as a SUCCESSFUL result: the router recorded a
    success for the statement and the tool handed the model an empty string —
    a fundamentals-blind read that nothing flagged. Raising lets the router
    record its sentinel and try the next vendor in the chain.
    """
    result = _make_api_request(function_name, params)
    if isinstance(result, str) and not result.strip():
        raise NoMarketDataError(
            ticker, detail=f"Alpha Vantage returned an empty {function_name} body",
        )
    return result


def _filter_reports_by_date(result: dict | str, curr_date: str | None) -> dict | str:
    """Drop annual/quarterly reports not yet public on ``curr_date``.

    A report whose fiscal period ends on/before ``curr_date`` is not necessarily
    public then -- it is filed days to weeks later (see
    :func:`yialpha.dataflows.utils.is_filing_public`). Keeping only reports
    whose period end + filing lag is on/before ``curr_date`` prevents a backtest
    from reading a report the market could not yet have seen.

    ``_make_api_request`` always returns the raw response *text* (``str``),
    so a JSON statement payload arrives here as a string. It is parsed with
    ``json.loads`` and the filtered ``dict`` is returned; a string that is not
    valid JSON (a CSV dataset, or an error/notice body) is returned unchanged —
    there is nothing statement-shaped to filter in it. The returned dict is
    consumed downstream exactly like a dict response (the tool layer
    JSON-serializes non-str tool outputs).
    """
    # PIT clamp (round-11, same defense as y_finance's statement entries):
    # is_filing_public trusts curr_date verbatim, so a curr_date beyond the
    # run's pinned analysis date (e.g. an LLM echoing a cached header's
    # "today") must be clamped before it filters. No-op in live mode.
    curr_date = current_pit_end(curr_date) or curr_date
    if not curr_date:
        return result
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return result
    if not isinstance(result, dict):
        return result
    for key in ("annualReports", "quarterlyReports"):
        if key in result:
            result[key] = [
                r for r in result[key]
                if is_filing_public(r.get("fiscalDateEnding", ""), curr_date)
            ]
    return result


def get_fundamentals(ticker: str, curr_date: str | None = None) -> dict | str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol using Alpha Vantage.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd

    Returns:
        str: Company overview data including financial ratios and key metrics

    Raises:
        NoMarketDataError: when ``curr_date`` is an explicit past date -- the
            OVERVIEW endpoint is a current-point snapshot with no date
            dimension, so its values (PE, marketCap, EPS, ...) are always
            *today's* and would leak the future on a past backtest date. The
            router turns this into the NO_DATA_AVAILABLE sentinel.
    """
    # PIT clamp (round-13, same gate-side defense as y_finance.get_fundamentals
    # round-8; the statement entries below got it in round-11): a curr_date
    # beyond the run's pinned analysis date (e.g. an LLM echoing a cached
    # header's "today") must be clamped before the leak gate judges it —
    # unclamped, an inflated "today-ish" date passed as live mode and served
    # the OVERVIEW snapshot into a historical run. No-op in live mode.
    curr_date = current_pit_end(curr_date) or curr_date
    if overview_would_leak_future(curr_date):
        canonical = normalize_symbol(ticker)
        raise NoMarketDataError(
            ticker, canonical,
            f"OVERVIEW snapshot is point-in-time (today only); not valid as of {curr_date}",
        )

    params = {
        "symbol": ticker,
    }

    return _api_text("OVERVIEW", params, ticker)


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> dict | str:
    """Retrieve balance sheet data for a given ticker symbol using Alpha Vantage."""
    result = _api_text("BALANCE_SHEET", {"symbol": ticker}, ticker)
    return _filter_reports_by_date(result, curr_date)


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> dict | str:
    """Retrieve cash flow statement data for a given ticker symbol using Alpha Vantage."""
    result = _api_text("CASH_FLOW", {"symbol": ticker}, ticker)
    return _filter_reports_by_date(result, curr_date)


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> dict | str:
    """Retrieve income statement data for a given ticker symbol using Alpha Vantage."""
    result = _api_text("INCOME_STATEMENT", {"symbol": ticker}, ticker)
    return _filter_reports_by_date(result, curr_date)

