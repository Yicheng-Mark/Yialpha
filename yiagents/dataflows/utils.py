import os
import re
from datetime import date, datetime, timedelta
from typing import Annotated

import pandas as pd

SavePathType = Annotated[str, "File path to save data. If None, data is not saved."]


# ---------------------------------------------------------------------------
# Point-in-time (PIT) guards for fundamental data.
#
# A backtest may only see data that was actually public on the simulated date.
# Two distinct lookahead leaks are guarded here:
#
#   1. Financial statements (balance sheet / cash flow / income statement).
#      yfinance and Alpha Vantage key these by ``fiscalDateEnding`` (the fiscal
#      PERIOD end), but a period ending e.g. Sep 30 is NOT public on Oct 1 --
#      the 10-Q is filed days to weeks later (SEC large accelerated filers:
#      10-K ~60 days, 10-Q ~40 days after period end). Filtering on
#      ``fiscalDateEnding <= curr_date`` lets a backtest read a report the
#      market could not yet have seen. ``is_filing_public`` adds a filing lag
#      so a period is visible only once it was plausibly filed.
#
#   2. Overview snapshots (yfinance ``.info`` / Alpha Vantage ``OVERVIEW``).
#      These are single current-point values (PE, marketCap, EPS, beta) with NO
#      date dimension -- they are always *today's* values. Surfacing them on a
#      past backtest date leaks the future wholesale. ``overview_would_leak_future``
#      flags this so the vendor can refuse (the router turns the resulting
#      NoMarketDataError into the NO_DATA_AVAILABLE sentinel the fundamentals
#      analyst is grounded to handle).
#
# These are correctness fixes, not opt-in enhancements: they apply by default.
# The filing lag is tunable via env for conservatism (set to 0 to revert to the
# old lookahead-leaking behaviour).
# ---------------------------------------------------------------------------
_FILING_LAG_ENV = os.environ.get("YIAGENTS_FUNDAMENTALS_FILING_LAG_DAYS")
try:
    FUNDAMENTALS_FILING_LAG_DAYS: int = (
        int(_FILING_LAG_ENV) if _FILING_LAG_ENV not in (None, "") else 45
    )
except ValueError:
    FUNDAMENTALS_FILING_LAG_DAYS = 45


def is_filing_public(
    fiscal_period_end,
    curr_date: str,
    lag_days: int = FUNDAMENTALS_FILING_LAG_DAYS,
) -> bool:
    """True iff a report for the fiscal period ending ``fiscal_period_end`` was
    plausibly public by ``curr_date`` (period end + ``lag_days``).

    Conservative on parse failure: if we cannot prove a report was public, drop
    it rather than risk lookahead. ``curr_date`` empty/None means live mode (no
    as-of constraint) -> keep everything.
    """
    if not curr_date:
        return True
    try:
        period_end = datetime.strptime(str(fiscal_period_end)[:10], "%Y-%m-%d")
        as_of = datetime.strptime(str(curr_date)[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return False
    return period_end + timedelta(days=lag_days) <= as_of


def overview_would_leak_future(curr_date: str) -> bool:
    """True iff ``curr_date`` is an explicit past date, for which a vendor's
    current-point overview snapshot would leak future information.

    Such snapshots (yfinance ``.info`` / Alpha Vantage ``OVERVIEW``) carry no
    date dimension, so they are only valid when ``curr_date`` is empty (live,
    no as-of constraint) or today/future.
    """
    if not curr_date:
        return False
    try:
        as_of = datetime.strptime(str(curr_date)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return False
    return as_of < date.today()


# Tickers can contain letters, digits, dot, dash, underscore, caret
# (index symbols like ^GSPC), equals (futures like GC=F), and plus
# (forex/CFD symbols like XAUUSD+). None of these enable directory
# traversal, so the value never escapes a containing directory when
# interpolated into a path. Anything else is rejected.
_TICKER_PATH_RE = re.compile(r"^[A-Za-z0-9._\-\^=+]+$")


def safe_ticker_component(value: str, *, max_len: int = 32) -> str:
    """Validate ``value`` is safe to interpolate into a filesystem path.

    Tickers come from user CLI input or from LLM tool calls, both of which
    can be influenced by attacker-controlled content (e.g. prompt injection
    embedded in fetched news). Without validation, a value like
    ``"../../../etc/foo"`` flows into ``os.path.join`` / ``Path /`` and
    escapes the configured cache, checkpoint, or results directory.

    Returns ``value`` unchanged when it matches the allowed pattern; raises
    ``ValueError`` otherwise.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"ticker must be a non-empty string, got {value!r}")
    if len(value) > max_len:
        raise ValueError(f"ticker exceeds {max_len} chars: {value!r}")
    if not _TICKER_PATH_RE.fullmatch(value):
        raise ValueError(
            f"ticker contains characters not allowed in a filesystem path: {value!r}"
        )
    # The regex above allows '.', so values like '.', '..', '...' would pass,
    # and as a path component they traverse the parent directory. Reject any
    # value that's only dots.
    if set(value) == {"."}:
        raise ValueError(f"ticker cannot consist solely of dots: {value!r}")
    return value


def save_output(data: pd.DataFrame, tag: str, save_path: SavePathType = None) -> None:
    if save_path:
        data.to_csv(save_path, encoding="utf-8")
        print(f"{tag} saved to {save_path}")


def get_current_date():
    return date.today().strftime("%Y-%m-%d")


def decorate_all_methods(decorator):
    def class_decorator(cls):
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value):
                setattr(cls, attr_name, decorator(attr_value))
        return cls

    return class_decorator


def get_next_weekday(date):

    if not isinstance(date, datetime):
        date = datetime.strptime(date, "%Y-%m-%d")

    if date.weekday() >= 5:
        days_to_add = 7 - date.weekday()
        next_weekday = date + timedelta(days=days_to_add)
        return next_weekday
    else:
        return date
