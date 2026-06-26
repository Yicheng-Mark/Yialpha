"""Unit tests for the browser broker execution module.

These tests exercise the **gate logic** of :class:`BrowserBroker.place_order`
without ever launching a real browser or hitting the network. Playwright is
mocked or made absent via :func:`unittest.mock.patch` on the module's
``_get_playwright`` helper, and the kill switch is driven through
``monkeypatch.setenv``.

All tests are ``@pytest.mark.unit`` (registered in ``tests/conftest.py``).
"""

from __future__ import annotations

import pytest

from yiagents.execution import browser_broker as bb
from yiagents.execution.browser_broker import (
    BrowserBroker,
    KillSwitch,
    OrderAction,
    OrderStatus,
    _coerce_bool_env,
)

# ---------------------------------------------------------------------------
# Enum round-tripping
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnums:
    def test_order_action_round_trip(self):
        assert OrderAction("buy") is OrderAction.BUY
        assert OrderAction("sell") is OrderAction.SELL
        assert OrderAction.BUY.value == "buy"

    def test_order_status_round_trip(self):
        for member in OrderStatus:
            assert OrderStatus(member.value) is member

    def test_order_status_only_one_submitted(self):
        # Invariant: exactly one status implies a live submission.
        submitted_members = [s for s in OrderStatus if "submit" in s.value and s is not OrderStatus.DRY_RUN_PREVIEW]
        assert submitted_members == [OrderStatus.SUBMITTED]


# ---------------------------------------------------------------------------
# _coerce_bool_env
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCoerceBoolEnv:
    @pytest.mark.parametrize("value", ["true", "1", "yes", "on", "TRUE", "Yes", "On"])
    def test_truthy_spellings(self, value):
        assert _coerce_bool_env(value) is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", "FALSE", "No"])
    def test_falsy_spellings(self, value):
        assert _coerce_bool_env(value) is False

    def test_none_and_empty_are_false(self):
        assert _coerce_bool_env(None) is False
        assert _coerce_bool_env("") is False
        assert _coerce_bool_env("   ") is False

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            _coerce_bool_env("treu")
        with pytest.raises(ValueError):
            _coerce_bool_env("maybe")
        with pytest.raises(ValueError):
            _coerce_bool_env("2")


# ---------------------------------------------------------------------------
# KillSwitch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKillSwitch:
    def test_unset_is_not_halted(self, monkeypatch):
        monkeypatch.delenv("YIAGENTS_KILL_SWITCH", raising=False)
        assert KillSwitch.is_halted() is False
        assert "unset" in KillSwitch.reason().lower()

    def test_truthy_values_halt(self, monkeypatch):
        for v in ("true", "1", "yes", "on"):
            monkeypatch.setenv("YIAGENTS_KILL_SWITCH", v)
            assert KillSwitch.is_halted() is True, v

    def test_falsy_values_allow(self, monkeypatch):
        for v in ("false", "0", "no", "off"):
            monkeypatch.setenv("YIAGENTS_KILL_SWITCH", v)
            assert KillSwitch.is_halted() is False, v

    def test_malformed_is_treated_as_halted(self, monkeypatch):
        # Fail-closed: a typo must not silently re-enable trading.
        monkeypatch.setenv("YIAGENTS_KILL_SWITCH", "treu")
        assert KillSwitch.is_halted() is True
        assert "halted" in KillSwitch.reason().lower()


# ---------------------------------------------------------------------------
# place_order gate logic (no browser, no network)
# ---------------------------------------------------------------------------


def _no_playwright(self):
    """Patch target: simulate Playwright not being installed."""
    return None


@pytest.mark.unit
class TestPlaceOrderGates:
    def test_kill_switch_on_blocks_before_anything(self, monkeypatch):
        # Kill switch on MUST block even with an invalid size and no playwright,
        # proving gate 1 short-circuits before later gates.
        monkeypatch.setenv("YIAGENTS_KILL_SWITCH", "true")
        broker = BrowserBroker(order_page_url="https://broker.example/order")
        result = broker.place_order("AAPL", "buy", size=-5.0)
        assert result.status is OrderStatus.BLOCKED_KILL_SWITCH
        assert result.submitted is False

    def test_kill_switch_on_submitted_false(self, monkeypatch):
        monkeypatch.setenv("YIAGENTS_KILL_SWITCH", "1")
        broker = BrowserBroker()
        result = broker.place_order("AAPL", OrderAction.BUY, size=10.0)
        assert result.status is OrderStatus.BLOCKED_KILL_SWITCH
        assert result.submitted is False
        assert "kill switch" in result.message.lower()

    def test_dry_run_default_no_playwright_no_url(self, monkeypatch):
        monkeypatch.delenv("YIAGENTS_KILL_SWITCH", raising=False)
        broker = BrowserBroker()  # dry_run_default=True, order_page_url=None
        with pytest.MonkeyPatch().context() as m:
            m.setattr(bb.BrowserBroker, "_get_playwright", _no_playwright)
            result = broker.place_order("AAPL", "buy", size=10.0)
        assert result.status is OrderStatus.DRY_RUN_PREVIEW
        assert result.submitted is False
        assert "no live broker url" in result.message.lower()
        assert result.preview_url is None

    def test_dry_run_default_no_playwright_with_url(self, monkeypatch):
        monkeypatch.delenv("YIAGENTS_KILL_SWITCH", raising=False)
        broker = BrowserBroker(order_page_url="https://broker.example/order")
        with pytest.MonkeyPatch().context() as m:
            m.setattr(bb.BrowserBroker, "_get_playwright", _no_playwright)
            result = broker.place_order("AAPL", "buy", size=10.0)
        assert result.status is OrderStatus.DRY_RUN_PREVIEW
        assert result.submitted is False
        # URL is configured so the preview URL should be surfaced.
        assert result.preview_url == "https://broker.example/order"

    def test_dry_run_false_playwright_missing_blocks(self, monkeypatch):
        monkeypatch.delenv("YIAGENTS_KILL_SWITCH", raising=False)
        broker = BrowserBroker(order_page_url="https://broker.example/order")
        with pytest.MonkeyPatch().context() as m:
            m.setattr(bb.BrowserBroker, "_get_playwright", _no_playwright)
            result = broker.place_order(
                "AAPL", "buy", size=10.0, dry_run=False
            )
        assert result.status is OrderStatus.BLOCKED_PLAYWRIGHT
        assert result.submitted is False
        assert "playwright" in result.message.lower()

    def test_dry_run_false_no_url_blocks(self, monkeypatch):
        monkeypatch.delenv("YIAGENTS_KILL_SWITCH", raising=False)
        broker = BrowserBroker(order_page_url=None)
        # Even with playwright present, no URL => blocked (live path needs target).
        result = broker.place_order(
            "AAPL", "buy", size=10.0, dry_run=False
        )
        assert result.status is OrderStatus.BLOCKED_PLAYWRIGHT
        assert result.submitted is False

    def test_size_zero_blocks_validation(self, monkeypatch):
        monkeypatch.delenv("YIAGENTS_KILL_SWITCH", raising=False)
        broker = BrowserBroker()
        result = broker.place_order("AAPL", "buy", size=0.0)
        assert result.status is OrderStatus.BLOCKED_VALIDATION
        assert result.submitted is False

    def test_negative_size_blocks_validation(self, monkeypatch):
        monkeypatch.delenv("YIAGENTS_KILL_SWITCH", raising=False)
        broker = BrowserBroker()
        result = broker.place_order("AAPL", "buy", size=-3.0)
        assert result.status is OrderStatus.BLOCKED_VALIDATION
        assert result.submitted is False

    def test_size_exceeds_equity_cap_blocks(self, monkeypatch):
        # size/equity = 1000/10000 = 0.10 > 0.01 cap.
        monkeypatch.delenv("YIAGENTS_KILL_SWITCH", raising=False)
        broker = BrowserBroker(max_order_pct_of_equity=0.01)
        result = broker.place_order(
            "AAPL", "buy", size=1000.0, equity_value=10000.0
        )
        assert result.status is OrderStatus.BLOCKED_VALIDATION
        assert result.submitted is False
        assert "exceeding" in result.message.lower()

    def test_size_within_equity_cap_passes_to_next_gate(self, monkeypatch):
        # 50/100000 = 0.05% <= 1% cap => passes validation; dry-run default
        # => DRY_RUN_PREVIEW (proves we got past the equity gate).
        monkeypatch.delenv("YIAGENTS_KILL_SWITCH", raising=False)
        broker = BrowserBroker(max_order_pct_of_equity=0.01, order_page_url="https://b.example/o")
        result = broker.place_order(
            "AAPL", "buy", size=50.0, equity_value=100000.0
        )
        assert result.status is OrderStatus.DRY_RUN_PREVIEW
        assert result.submitted is False

    def test_invalid_equity_value_blocks(self, monkeypatch):
        monkeypatch.delenv("YIAGENTS_KILL_SWITCH", raising=False)
        broker = BrowserBroker()
        result = broker.place_order(
            "AAPL", "buy", size=10.0, equity_value="not-a-number"
        )
        assert result.status is OrderStatus.BLOCKED_VALIDATION
        assert result.submitted is False

    def test_zero_equity_blocks(self, monkeypatch):
        monkeypatch.delenv("YIAGENTS_KILL_SWITCH", raising=False)
        broker = BrowserBroker()
        result = broker.place_order(
            "AAPL", "buy", size=10.0, equity_value=0.0
        )
        assert result.status is OrderStatus.BLOCKED_VALIDATION
        assert result.submitted is False

    def test_pre_submit_validator_false_blocks(self, monkeypatch):
        monkeypatch.delenv("YIAGENTS_KILL_SWITCH", raising=False)
        broker = BrowserBroker()

        def reject(ticker, action, size):
            return False

        result = broker.place_order(
            "AAPL", "buy", size=10.0, pre_submit_validator=reject
        )
        assert result.status is OrderStatus.BLOCKED_VALIDATION
        assert result.submitted is False

    def test_pre_submit_validator_raising_blocks(self, monkeypatch):
        monkeypatch.delenv("YIAGENTS_KILL_SWITCH", raising=False)
        broker = BrowserBroker()

        def explode(ticker, action, size):
            raise RuntimeError("nope")

        result = broker.place_order(
            "AAPL", "buy", size=10.0, pre_submit_validator=explode
        )
        assert result.status is OrderStatus.BLOCKED_VALIDATION
        assert result.submitted is False

    def test_pre_submit_validator_true_with_dry_run_previews(self, monkeypatch):
        monkeypatch.delenv("YIAGENTS_KILL_SWITCH", raising=False)
        broker = BrowserBroker(order_page_url="https://b.example/o")

        def accept(ticker, action, size):
            return True

        result = broker.place_order(
            "AAPL", "buy", size=10.0, pre_submit_validator=accept
        )
        assert result.status is OrderStatus.DRY_RUN_PREVIEW
        assert result.submitted is False

    def test_pre_submit_validator_none_ok_with_dry_run(self, monkeypatch):
        # A validator returning None is treated as "ok".
        monkeypatch.delenv("YIAGENTS_KILL_SWITCH", raising=False)
        broker = BrowserBroker(order_page_url="https://b.example/o")

        def neutral(ticker, action, size):
            return None

        result = broker.place_order(
            "AAPL", "buy", size=10.0, pre_submit_validator=neutral
        )
        assert result.status is OrderStatus.DRY_RUN_PREVIEW
        assert result.submitted is False

    def test_invalid_action_string_blocks_validation(self, monkeypatch):
        monkeypatch.delenv("YIAGENTS_KILL_SWITCH", raising=False)
        broker = BrowserBroker()
        result = broker.place_order("AAPL", "hold", size=10.0)
        assert result.status is OrderStatus.BLOCKED_VALIDATION
        assert result.submitted is False


# ---------------------------------------------------------------------------
# Live path: base class cannot submit (subclass not configured)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLivePathFailClosed:
    def test_base_class_fill_form_raises_not_implemented(self, monkeypatch):
        # The base broker's _fill_order_form is intentionally a stub; a live
        # attempt must fail closed with BLOCKED_PLAYWRIGHT (mapped from
        # NotImplementedError), never a submission.
        monkeypatch.delenv("YIAGENTS_KILL_SWITCH", raising=False)
        broker = BrowserBroker(order_page_url="https://b.example/o")
        result = broker.place_order(
            "AAPL", "buy", size=10.0, dry_run=False
        )
        # Playwright likely absent in CI => BLOCKED_PLAYWRIGHT before reaching
        # the form. If Playwright IS present we still hit NotImplementedError.
        assert result.status in (OrderStatus.BLOCKED_PLAYWRIGHT,)
        assert result.submitted is False

    def test_subclass_submit_never_slips_on_dom_error(self, monkeypatch):
        # Even a subclass with a buggy _fill_order_form that raises after
        # navigating must not submit; the error is caught.
        monkeypatch.delenv("YIAGENTS_KILL_SWITCH", raising=False)

        class BrokenBroker(BrowserBroker):
            def _fill_order_form(self, page, ticker, action, size):
                raise RuntimeError("DOM changed")

        # Force playwright to appear present so we reach _fill_order_form.
        broker = BrokenBroker(order_page_url="https://b.example/o")

        # We can't actually launch a real browser in unit tests, so simulate
        # the entire _do_live_order path failing closed by patching it to call
        # _fill_order_form on a dummy page.
        from unittest import mock

        with mock.patch.object(
            bb.BrowserBroker, "_get_playwright", lambda self: lambda: _FakePW()
        ):
            result = broker.place_order(
                "AAPL", "buy", size=10.0, dry_run=False
            )
        assert result.status is OrderStatus.ERROR
        assert result.submitted is False


class _FakePage:
    """Stand-in for a Playwright Page; records URL, never errors."""

    url = "https://b.example/o"

    def set_default_timeout(self, ms):
        pass

    def goto(self, url):
        self.url = url

    def click(self, *a, **k):
        pass


class _FakePWContext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakePW:
    """Minimal fake of the sync_playwright() context manager."""

    def __call__(self):
        return self

    def __enter__(self):
        chromium = mock_chromium()
        self.chromium = chromium
        return self

    def __exit__(self, *a):
        return False


def mock_chromium():
    from unittest import mock

    chromium = mock.MagicMock()
    browser = mock.MagicMock()
    context = mock.MagicMock()
    page = _FakePage()
    context.new_page.return_value = page
    browser.new_context.return_value = context
    chromium.launch.return_value = browser
    return chromium


# ---------------------------------------------------------------------------
# submitted invariant
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSubmittedInvariant:
    def test_non_submitted_statuses_never_mark_submitted(self, monkeypatch):
        monkeypatch.delenv("YIAGENTS_KILL_SWITCH", raising=False)
        broker = BrowserBroker()
        # Walk several gate outcomes; none should report submitted=True.
        for kwargs, expected_status in [
            (dict(ticker="AAPL", action="buy", size=0.0), OrderStatus.BLOCKED_VALIDATION),
            (
                dict(ticker="AAPL", action="buy", size=1000.0, equity_value=10000.0),
                OrderStatus.BLOCKED_VALIDATION,
            ),
            (dict(ticker="AAPL", action="hold", size=10.0), OrderStatus.BLOCKED_VALIDATION),
        ]:
            result = broker.place_order(**kwargs)
            assert result.status is expected_status, kwargs
            assert result.submitted is False, kwargs
