from .alpha_vantage_common import _make_api_request
from .symbol_utils import NoMarketDataError, normalize_symbol
from .utils import is_filing_public, overview_would_leak_future


def _filter_reports_by_date(result, curr_date: str):
    """Drop annual/quarterly reports not yet public on ``curr_date``.

    A report whose fiscal period ends on/before ``curr_date`` is not necessarily
    public then -- it is filed days to weeks later (see
    :func:`yiagents.dataflows.utils.is_filing_public`). Keeping only reports
    whose period end + filing lag is on/before ``curr_date`` prevents a backtest
    from reading a report the market could not yet have seen.
    """
    if not curr_date or not isinstance(result, dict):
        return result
    for key in ("annualReports", "quarterlyReports"):
        if key in result:
            result[key] = [
                r for r in result[key]
                if is_filing_public(r.get("fiscalDateEnding", ""), curr_date)
            ]
    return result


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
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
    if overview_would_leak_future(curr_date):
        canonical = normalize_symbol(ticker)
        raise NoMarketDataError(
            ticker, canonical,
            f"OVERVIEW snapshot is point-in-time (today only); not valid as of {curr_date}",
        )

    params = {
        "symbol": ticker,
    }

    return _make_api_request("OVERVIEW", params)


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve balance sheet data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("BALANCE_SHEET", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve cash flow statement data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("CASH_FLOW", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve income statement data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("INCOME_STATEMENT", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)

