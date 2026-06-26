"""On-disk memoization of ``graph.propagate`` decisions for backtesting.

A single backtest calls the LLM-driven graph once per ``(ticker, date)``. Those
calls dominate cost and wall-clock time, and they are the one non-deterministic
ingredient. This cache stores the *realized* decision for each key so that:

* Re-running the same backtest (e.g. after changing the position-sizing logic in
  the risk layer) replays identical decisions instead of re-billing the LLM.
* The multi-run distribution the plan asks for ("run N>=5 times and take the
  distribution") is produced by varying ``run_tag`` across real LLM passes, then
  replaying each tag deterministically.

Only the fields a backtest needs are persisted: the final rating and the
Portfolio Manager's rendered markdown decision. The full agent state is not
cached (it is large and already written to ``results_dir`` by the graph itself).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedDecision:
    """A single realized decision, as replayed from cache."""

    ticker: str
    date: str
    run_tag: str
    rating: str
    final_decision: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "run_tag": self.run_tag,
            "rating": self.rating,
            "final_decision": self.final_decision,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CachedDecision:
        return cls(
            ticker=d["ticker"],
            date=d["date"],
            run_tag=d.get("run_tag", "default"),
            rating=d["rating"],
            final_decision=d["final_decision"],
        )


def _safe_component(value: str) -> str:
    """Hash a free-form string into a filename-safe component.

    Tickers / dates are usually safe, but ``run_tag`` is user-supplied and the
    ticker can contain a dot (``BRK.B``). Hashing keeps the cache dir flat and
    avoids any path-escape or collision surprises.
    """
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:16]
    return digest


class DecisionCache:
    """Disk-backed memo of ``(ticker, date, run_tag)`` -> realized decision.

    The store is one JSON file per key under ``cache_dir``. File I/O is
    fault-tolerant: a corrupt or unreadable entry is treated as a miss and
    removed, so a bad cache can never break a backtest.
    """

    def __init__(self, cache_dir: str | os.PathLike[str] | None = None, enabled: bool = True):
        self.enabled = enabled
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.enabled and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: dict[str, Any], enabled: bool = True) -> DecisionCache:
        """Build a cache rooted at the project's data cache dir."""
        base = Path(config.get("data_cache_dir", ".")) / "backtest_decisions"
        return cls(base, enabled=enabled)

    def _key_path(self, ticker: str, date: str, run_tag: str) -> Path | None:
        if self.cache_dir is None:
            return None
        name = f"{_safe_component(ticker)}_{_safe_component(date)}_{_safe_component(run_tag)}.json"
        return self.cache_dir / name

    def get(self, ticker: str, date: str, run_tag: str = "default") -> CachedDecision | None:
        if not self.enabled:
            return None
        path = self._key_path(ticker, date, run_tag)
        if path is None or not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                return CachedDecision.from_dict(json.load(fh))
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Corrupt backtest cache entry %s (%s); removing", path, exc)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def put(self, decision: CachedDecision) -> None:
        if not self.enabled:
            return
        path = self._key_path(decision.ticker, decision.date, decision.run_tag)
        if path is None:
            return
        tmp = path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(decision.to_dict(), fh, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("Could not write backtest cache entry %s (%s)", path, exc)

    def remember(
        self,
        ticker: str,
        date: str,
        rating: str,
        final_decision: str,
        run_tag: str = "default",
    ) -> CachedDecision:
        """Store a realized decision and return the cached record."""
        decision = CachedDecision(
            ticker=ticker,
            date=str(date),
            run_tag=run_tag,
            rating=rating,
            final_decision=final_decision,
        )
        self.put(decision)
        return decision
