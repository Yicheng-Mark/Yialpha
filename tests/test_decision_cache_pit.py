"""Point-in-time schema guards for the disk backtest decision cache."""

from __future__ import annotations

import json

from yialpha.backtest.cache import DecisionCache


def test_legacy_cache_entry_is_miss_and_evicted(tmp_path):
    cache = DecisionCache(tmp_path)
    cache.remember("AAPL", "2020-01-02", "Buy", "Rating: Buy", run_tag="r1")
    path = next(tmp_path.glob("*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("schema_version")
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert cache.get("AAPL", "2020-01-02", "r1") is None
    assert not path.exists()
