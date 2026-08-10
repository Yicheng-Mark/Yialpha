"""CLI config precedence (#976, #977).

An explicit environment override for the debate/risk round counts, or the
checkpoint flag, must win over the interactive research-depth selection — the CLI
must not clobber an env-configured value back to a prompt/flag default.
"""

from unittest import mock

import pytest

import yiagents.cli.main as m

# Minimal selections dict shaped like get_user_selections()'s return value.
SELECTIONS = {
    "research_depth": 5,
    "shallow_thinker": "gpt-5.4-mini",
    "deep_thinker": "gpt-5.5",
    "backend_url": None,
    "llm_provider": "openai",
    "google_thinking_level": None,
    "openai_reasoning_effort": None,
    "anthropic_effort": None,
    "output_language": "English",
}


def test_research_depth_sets_both_rounds_without_env(monkeypatch):
    for var in ("YIAGENTS_MAX_DEBATE_ROUNDS", "YIAGENTS_MAX_RISK_ROUNDS"):
        monkeypatch.delenv(var, raising=False)
    cfg = m._build_run_config(SELECTIONS, checkpoint=None)
    assert cfg["max_debate_rounds"] == 5
    assert cfg["max_risk_discuss_rounds"] == 5


def test_env_round_counts_win_over_selection(monkeypatch):
    monkeypatch.setenv("YIAGENTS_MAX_DEBATE_ROUNDS", "2")
    monkeypatch.setenv("YIAGENTS_MAX_RISK_ROUNDS", "4")
    # DEFAULT_CONFIG already reflects the env (applied at import); emulate that.
    patched = dict(m.DEFAULT_CONFIG, max_debate_rounds=2, max_risk_discuss_rounds=4)
    with mock.patch.object(m, "DEFAULT_CONFIG", patched):
        cfg = m._build_run_config(SELECTIONS, checkpoint=None)
    assert cfg["max_debate_rounds"] == 2  # env value, not research_depth=5
    assert cfg["max_risk_discuss_rounds"] == 4


def test_partial_env_only_overrides_that_count(monkeypatch):
    monkeypatch.setenv("YIAGENTS_MAX_DEBATE_ROUNDS", "2")
    monkeypatch.delenv("YIAGENTS_MAX_RISK_ROUNDS", raising=False)
    patched = dict(m.DEFAULT_CONFIG, max_debate_rounds=2)
    with mock.patch.object(m, "DEFAULT_CONFIG", patched):
        cfg = m._build_run_config(SELECTIONS, checkpoint=None)
    assert cfg["max_debate_rounds"] == 2  # env wins
    assert cfg["max_risk_discuss_rounds"] == 5  # falls through to research_depth


def test_checkpoint_none_preserves_env_default():
    patched = dict(m.DEFAULT_CONFIG, checkpoint_enabled=True)  # e.g. env-enabled
    with mock.patch.object(m, "DEFAULT_CONFIG", patched):
        cfg = m._build_run_config(SELECTIONS, checkpoint=None)
    assert cfg["checkpoint_enabled"] is True  # not clobbered back to False


@pytest.mark.parametrize("flag", [True, False])
def test_checkpoint_flag_overrides_env(flag):
    patched = dict(m.DEFAULT_CONFIG, checkpoint_enabled=not flag)
    with mock.patch.object(m, "DEFAULT_CONFIG", patched):
        cfg = m._build_run_config(SELECTIONS, checkpoint=flag)
    assert cfg["checkpoint_enabled"] is flag


def test_explicit_batch_workers_enable_concurrency_over_default_off():
    cfg = {"batch_concurrency": False, "batch_workers": 3}
    result = m._apply_batch_worker_override(cfg, workers=4)
    assert result["batch_concurrency"] is True


def test_omitted_batch_workers_preserve_master_switch():
    cfg = {"batch_concurrency": False, "batch_workers": 3}
    result = m._apply_batch_worker_override(cfg, workers=None)
    assert result["batch_concurrency"] is False


def test_one_batch_worker_explicitly_requests_serial_mode():
    cfg = {"batch_concurrency": True, "batch_workers": 3}
    result = m._apply_batch_worker_override(cfg, workers=1)
    assert result["batch_concurrency"] is False


class _RecordingRunner:
    """Captures the config/workers the CLI batch command hands to BatchRunner."""

    instances: list["_RecordingRunner"] = []

    def __init__(self, config, workers=None, **kwargs):
        self.config = config
        self.workers = workers
        type(self).instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, tickers, date, asset_type="stock"):
        return [
            {"ticker": t, "error": None, "elapsed": 0.0, "report_path": ""}
            for t in tickers
        ]


@pytest.fixture
def _cli_batch_runner(monkeypatch):
    """Route the CLI batch command's local BatchRunner import to a recorder."""
    _RecordingRunner.instances.clear()
    import yiagents.batch.runner as runner

    monkeypatch.setattr(runner, "BatchRunner", _RecordingRunner)
    return _RecordingRunner


def test_cli_batch_omitted_workers_is_serial_by_default(_cli_batch_runner):
    """No --workers → BatchRunner sees batch_concurrency=False (K=1).

    Guards against a misleading 'default concurrent' comment/code mismatch:
    DEFAULT_CONFIG['batch_concurrency'] is False, so the batch entry point is
    strictly serial unless the user opts in via --workers or env.
    """
    m.batch(
        tickers=["AAPL", "NVDA"],
        date="2026-01-10",
        asset_type="stock",
        workers=None,
    )
    assert len(_cli_batch_runner.instances) == 1
    assert _cli_batch_runner.instances[0].config["batch_concurrency"] is False
    assert _cli_batch_runner.instances[0].workers is None


def test_cli_batch_explicit_workers_enables_concurrency(_cli_batch_runner):
    """--workers 4 → BatchRunner sees batch_concurrency=True, workers=4."""
    m.batch(
        tickers=["AAPL", "NVDA"],
        date="2026-01-10",
        asset_type="stock",
        workers=4,
    )
    assert len(_cli_batch_runner.instances) == 1
    assert _cli_batch_runner.instances[0].config["batch_concurrency"] is True
    assert _cli_batch_runner.instances[0].workers == 4


def test_complete_report_prefers_post_overlay_portfolio_decision(monkeypatch):
    rendered = []

    class QuietConsole:
        def print(self, *args, **kwargs):
            return None

    def capture_markdown(value):
        rendered.append(value)
        return value

    monkeypatch.setattr(m, "console", QuietConsole())
    monkeypatch.setattr(m, "Markdown", capture_markdown)

    m.display_complete_report(
        {
            "risk_debate_state": {"judge_decision": "raw PM decision"},
            "final_trade_decision": "post-overlay PM decision",
        }
    )

    assert rendered == ["post-overlay PM decision"]
