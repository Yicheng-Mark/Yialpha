"""HTTP contract and browser security-header tests for the local Web UI."""

import time
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from web import runner
from web.app import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(runner, "registry", runner.TaskRegistry())
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.unit
def test_all_responses_include_strict_script_csp(client):
    response = client.get("/")
    assert response.status_code == 200
    policy = response.headers["content-security-policy"]
    assert "script-src 'self'" in policy
    assert "'unsafe-inline'" not in policy.split("script-src", 1)[1].split(";", 1)[0]
    for directive in (
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "frame-ancestors 'none'",
    ):
        assert directive in policy


@pytest.mark.unit
def test_analyze_accepts_crypto_spot(client, monkeypatch):
    observed = {}

    async def fake_spawn(ticker, analysis_date, asset_type, language):
        observed.update(
            ticker=ticker,
            date=analysis_date,
            asset_type=asset_type,
            language=language,
        )
        task_id = "test-task"
        runner.registry.tasks[task_id] = runner.TaskState(
            task_id=task_id,
            ticker=ticker,
            date=analysis_date,
            asset_type=asset_type,
            started_at=time.time(),
            language=language,
        )
        return task_id

    monkeypatch.setattr(runner, "spawn", fake_spawn)
    response = client.post(
        "/api/analyze",
        json={
            "ticker": "BTCUSDT",
            "date": date.today().isoformat(),
            "asset_type": "crypto_spot",
            "language": "zh",
        },
    )
    assert response.status_code == 200
    assert response.json()["task_id"] == "test-task"
    assert observed["asset_type"] == "crypto_spot"


@pytest.mark.unit
def test_analyze_rejects_future_date_before_spawn(client, monkeypatch):
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("future-dated request reached process spawn")

    monkeypatch.setattr(runner, "spawn", fail_if_called)
    response = client.post(
        "/api/analyze",
        json={
            "ticker": "AAPL",
            "date": (date.today() + timedelta(days=1)).isoformat(),
            "asset_type": "stock",
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "analysis date cannot be in the future"
