"""Fama-French factor attribution for the YiAgents backtest harness.

Decomposes a strategy's daily returns into common-risk-factor exposure
(``Mkt-RF``, ``SMB``, ``HML`` and, for the 5-factor model, ``RMW``, ``CMA``)
plus a residual *Jensen's alpha* — the return left over after stripping market,
size, value, profitability and investment beta. This is what
:mod:`yiagents.backtest.metrics`' naive ``alpha_vs_buyhold`` (a plain
strategy-minus-buy&-hold mean difference) cannot tell you: a levered index fund
shows positive ``alpha_vs_buyhold`` but ~zero factor alpha, because its edge is
just market beta.

Three decoupled pieces:

* :func:`_parse_french_zip` — pure parser for a Kenneth French Data Library
  daily-factor ``.zip`` (no network). Column names are assigned *by position*
  rather than parsed from the header, because French's header line spacing /
  leading-comma varies across files but the column order is fixed.
* :func:`load_factor_returns` — downloads (via ``requests``, which honours the
  project's ``HTTP_PROXY`` / ``HTTPS_PROXY`` SOCKS5 settings, same path as
  yfinance / Binance), caches the raw zip bytes on disk (mtime TTL ~1 day since
  the daily files update each trading evening), and slices to ``[start, end]``.
  **Fail-open**: any network / parse error logs a warning and returns ``None``,
  so a missing factor file never breaks a backtest.
* :func:`factor_attribution` — pure OLS (``numpy.linalg.lstsq``, no scipy
  dependency) of strategy excess returns on the factor matrix; returns alpha
  (annualized), per-factor betas, R² and the observation count.

Point-in-time: French daily factors are realised, market-wide, published with
~1-day lag. Using them to attribute a backtest whose holding window has already
closed introduces no look-ahead.
"""

from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ken French Data Library daily-factor zips. Column order is fixed per file, so
# the parser assigns names by position rather than trusting the header line.
_FACTOR_FILES: dict[str, str] = {
    "3": "F-F_Research_Data_Factors_daily_CSV.zip",
    "5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
}
_FACTOR_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
_EXPECTED_COLS: dict[str, list[str]] = {
    "3": ["Mkt-RF", "SMB", "HML", "RF"],
    "5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
}
_FACTOR_LABEL = {"3": "FF3", "5": "FF5"}

# Fewer than this many aligned strategy/factor observations is too thin to
# trust an OLS fit; attribution returns None instead.
_MIN_OBS = 10


@dataclass
class FactorAttribution:
    """Result of regressing strategy excess returns on a Fama-French model."""

    model: str                       # "FF3" / "FF5"
    alpha_annual: float              # per-period intercept * periods_per_year
    betas: dict[str, float] = field(default_factory=dict)
    r_squared: float = 0.0
    n_obs: int = 0


# ---------------------------------------------------------------------------
# Pure parser (no network) — testable with a synthetic zip fixture.
# ---------------------------------------------------------------------------
def _parse_french_zip(raw: bytes, model: str) -> pd.DataFrame:
    """Parse a French daily-factor zip into a date-indexed DataFrame.

    Factor values in the source file are in **percent** and are converted to
    decimals here. Rows whose first token is not an 8-digit ``YYYYMMDD`` date
    (the header, the trailing copyright line, blank lines) are skipped or stop
    parsing. Returns an ascending-date DataFrame with columns
    ``_EXPECTED_COLS[model]``.
    """
    if model not in _EXPECTED_COLS:
        raise ValueError(f"unknown factor model {model!r}; want one of {list(_EXPECTED_COLS)}")
    expected = _EXPECTED_COLS[model]

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("zip contains no .csv member")
        text = zf.read(csv_names[0]).decode("utf-8", errors="replace")

    dates: list[pd.Timestamp] = []
    rows: list[list[float]] = []
    for line in text.splitlines():
        tokens = line.replace(",", " ").split()
        if not tokens:
            continue
        # A data row's first token is an 8-digit YYYYMMDD integer; everything
        # after must be numeric. The header / footer fail this test and are
        # skipped. Extra trailing tokens (e.g. an averaged annual row) are
        # rejected by the count check.
        first = tokens[0]
        if not (first.isdigit() and len(first) == 8):
            continue
        numeric_tail = tokens[1:]
        if len(numeric_tail) != len(expected):
            continue
        try:
            values = [float(t) for t in numeric_tail]
        except ValueError:
            continue
        dates.append(pd.Timestamp(first))
        rows.append(values)

    if not dates:
        raise ValueError("no data rows parsed from French factor zip")

    df = pd.DataFrame(rows, columns=expected, index=pd.DatetimeIndex(dates, name="date"))
    df = df.sort_index()
    # Percent -> decimal. Replace any sentinel -99.99 (French's "missing") with NaN,
    # then drop rows that lost data so the regression stays clean.
    df = df.replace(-99.99, np.nan) / 100.0
    df = df.dropna()
    return df


# ---------------------------------------------------------------------------
# Network fetcher with on-disk caching. Fail-open: returns None on any error.
# ---------------------------------------------------------------------------
def _default_cache_dir() -> Path:
    base = os.getenv("YIAGENTS_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".yiagents", "cache"))
    return Path(base) / "factors"


def _cached_or_download(path: Path, url: str, ttl_days: float = 1.0) -> bytes | None:
    """Return the factor zip bytes from a fresh cache or a new download.

    On download failure, falls back to a stale cache file if one exists (a
    slightly-old factor file is far better than no attribution). Returns
    ``None`` only when neither cache nor network can supply the bytes.
    """
    import requests

    if path.exists():
        age = time.time() - path.stat().st_mtime
        if age < ttl_days * 86_400.0:
            try:
                return path.read_bytes()
            except OSError:
                pass  # fall through to re-download

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        raw = resp.content
    except Exception as exc:  # noqa: BLE001 -- fail-open, try stale cache
        logger.warning("factor_model: download failed for %s: %s", url, exc)
        if path.exists():
            try:
                return path.read_bytes()
            except OSError:
                pass
        return None

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    except OSError as exc:  # noqa: BLE001 -- caching is best-effort
        logger.warning("factor_model: could not write cache %s: %s", path, exc)
    return raw


def load_factor_returns(
    start: str,
    end: str,
    model: str = "3",
    cache_dir: str | os.PathLike[str] | None = None,
) -> pd.DataFrame | None:
    """Load Fama-French daily factor returns for ``[start, end]``.

    Returns a date-indexed DataFrame (decimals) sliced to the window, or
    ``None`` if the model is unknown, the download/parse failed, or the window
    is empty. Never raises — callers (the backtest engine) treat ``None`` as
    "skip attribution".
    """
    if model not in _FACTOR_FILES:
        logger.warning("factor_model: unknown model %r; skipping attribution", model)
        return None
    cdir = Path(cache_dir) if cache_dir else _default_cache_dir()
    fname = _FACTOR_FILES[model]
    raw = _cached_or_download(cdir / fname, _FACTOR_BASE_URL + fname)
    if raw is None:
        return None
    try:
        df = _parse_french_zip(raw, model)
    except Exception as exc:  # noqa: BLE001 -- fail-open
        logger.warning("factor_model: parse failed for %s: %s", fname, exc)
        return None
    return df.loc[str(start):str(end)]


# ---------------------------------------------------------------------------
# Pure OLS attribution.
# ---------------------------------------------------------------------------
def factor_attribution(
    strategy_returns: "pd.Series",
    factor_returns: pd.DataFrame,
    periods_per_year: int = 252,
    model: str = "FF3",
) -> FactorAttribution | None:
    """Regress strategy excess returns on the factor matrix (OLS, numpy only).

    ``strategy_returns`` and ``factor_returns`` need not share an index; they
    are inner-joined on date. Excess return = strategy - ``RF``. Returns
    ``None`` when fewer than :data:`_MIN_OBS` aligned observations remain or the
    factor frame lacks an ``RF`` column — the caller then skips attribution
    rather than reporting a spurious fit.
    """
    sr = pd.Series(strategy_returns, dtype="float")
    sr.index = pd.to_datetime(sr.index).normalize()
    fr = factor_returns.copy()
    fr.index = pd.to_datetime(fr.index).normalize()

    common = sr.index.intersection(fr.index)
    if len(common) < _MIN_OBS:
        return None
    if "RF" not in fr.columns:
        return None

    sr_c = sr.loc[common].astype(float)
    fr_c = fr.loc[common].astype(float)
    excess = sr_c.values - fr_c["RF"].values

    factor_cols = [c for c in fr_c.columns if c != "RF"]
    if not factor_cols:
        return None
    X = fr_c[factor_cols].values
    design = np.column_stack([np.ones(len(X)), X])
    coeffs, _residuals, _rank, _sv = np.linalg.lstsq(design, excess, rcond=None)

    alpha_per = float(coeffs[0])
    betas = {c: float(b) for c, b in zip(factor_cols, coeffs[1:])}

    predicted = design @ coeffs
    ss_res = float(np.sum((excess - predicted) ** 2))
    ss_tot = float(np.sum((excess - excess.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    return FactorAttribution(
        model=model,
        alpha_annual=alpha_per * periods_per_year,
        betas=betas,
        r_squared=r_squared,
        n_obs=int(len(common)),
    )


def label_for(model: str) -> str:
    """Human label (``FF3`` / ``FF5``) for a model key, defaulting to FF3."""
    return _FACTOR_LABEL.get(str(model), "FF3")
