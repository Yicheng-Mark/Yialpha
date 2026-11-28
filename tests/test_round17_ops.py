"""Round-17 audit tests for the ops shard.

Hermetic: no vendor is contacted, nothing under the real ``~/.yialpha`` is
touched, and the process-wide logging configuration is put back after every
test that changes it. Each test pins ONE contract named in its docstring.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import logging
import sys
import threading
from concurrent.futures import Future
from concurrent.futures.thread import BrokenThreadPool
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import yialpha
from yialpha import accuracy as acc, default_config, logging_config as lc
from yialpha.batch import locks, runner
from yialpha.batch.runner import BatchRunner
from yialpha.dataflows import binance as bnc, y_finance as yfmod
from yialpha.dataflows.symbol_utils import normalize_symbol
from yialpha.instruments import models
from yialpha.risk import cvar, derivatives_stress as stress, kelly


# ===========================================================================
# risk.derivatives_stress — window statistics and band labels
# ===========================================================================
class TestPercentileRank:
    def test_r17_percentile_rank_unusable_windows_are_none(self):
        window = stress.MIN_WINDOW
        assert stress._percentile_rank(None) is None
        assert stress._percentile_rank(pd.Series([1.0, 2.0, 3.0])) is None  # too short
        assert stress._percentile_rank(pd.Series([np.nan] * window)) is None  # nothing usable
        assert stress._percentile_rank(pd.Series(["1.0"] * window)) is None  # text, not numbers

    @pytest.mark.parametrize(
        ("values", "expected_pct"),
        [
            ([7.0] * stress.MIN_WINDOW, 50.0),  # flat: the latest ties with everything
            (
                [float(i) for i in range(stress.MIN_WINDOW)],
                (stress.MIN_WINDOW - 1 + 0.5) / stress.MIN_WINDOW * 100.0,
            ),
            (
                [float(i) for i in range(stress.MIN_WINDOW)][::-1],
                0.5 / stress.MIN_WINDOW * 100.0,
            ),
            ([1.0] * 10 + [5.0] * 10 + [9.0] * 10, (20 + 0.5 * 10) / 30 * 100.0),  # 20 below, 10 tied
        ],
    )
    def test_r17_percentile_rank_is_a_mid_rank_that_counts_the_latest_itself(self, values, expected_pct):
        """The latest observation is part of its own window and ties count
        half: a flat window is the neutral 50, a rising one lands just under
        100 -- never a degenerate 100 or 0."""
        assert stress._percentile_rank(pd.Series(values)) == pytest.approx(
            (expected_pct, stress.MIN_WINDOW)
        )

    def test_r17_percentile_rank_drops_nan_rows_before_counting(self):
        values = [np.nan, 1.0, np.nan] + [float(i) for i in range(3, 32)]  # 30 usable of 33
        pct, n = stress._percentile_rank(pd.Series(values))
        assert n == 30
        assert pct == pytest.approx((29 + 0.5) / 30 * 100.0)


class TestZscoreLatest:
    def test_r17_zscore_latest_unusable_windows_are_none(self):
        window = stress.MIN_WINDOW
        assert stress._zscore_latest(None) is None
        assert stress._zscore_latest(pd.Series([1.0, 2.0])) is None
        assert stress._zscore_latest(pd.Series(["1.0"] * window)) is None
        assert stress._zscore_latest(pd.Series([4.0] * window)) is None  # no dispersion: no z

    def test_r17_zscore_latest_measures_the_last_value_against_its_own_window(self):
        values = np.arange(1.0, 31.0)
        expected = (30.0 - values.mean()) / values.std(ddof=1)
        assert stress._zscore_latest(pd.Series(values)) == pytest.approx((expected, 30))
        assert stress._zscore_latest(pd.Series(values[::-1]))[0] == pytest.approx(-expected)
        # NaN rows leave the window before the mean and the deviation are taken.
        noisy = pd.Series([np.nan, 1.0, np.nan] + list(values[1:]))
        assert stress._zscore_latest(noisy) == pytest.approx((expected, 30))


class TestStressLabels:
    @pytest.mark.parametrize(
        ("pct", "label"),
        [
            (100.0, "extreme_long"),
            (90.0, "extreme_long"),
            (89.99, "elevated_long"),
            (70.0, "elevated_long"),
            (69.99, "neutral"),
            (30.01, "neutral"),
            (30.0, "elevated_short"),
            (10.01, "elevated_short"),
            (10.0, "extreme_short"),
            (0.0, "extreme_short"),
        ],
    )
    def test_r17_funding_label_band_edges(self, pct, label):
        assert stress._funding_label(pct) == label

    @pytest.mark.parametrize(
        ("pct", "label"),
        [
            (100.0, "elevated_premium"),
            (85.0, "elevated_premium"),
            (84.99, "positive"),
            (50.01, "positive"),
            (50.0, "negative"),
            (15.01, "negative"),
            (15.0, "deep_discount"),
            (0.0, "deep_discount"),
        ],
    )
    def test_r17_basis_label_band_edges(self, pct, label):
        assert stress._basis_label(pct) == label


# ===========================================================================
# risk.cvar.historical_cvar — the expected tail loss
# ===========================================================================
class TestHistoricalCvar:
    _RETURNS = [-0.05, -0.03, 0.01, 0.02, 0.04]

    @pytest.mark.parametrize(
        ("confidence", "expected"),
        [
            (0.95, -0.05),  # 1 observation
            (0.79, -0.04),  # 2: ceil(1.05)
            (0.6, -0.04),  # 2: exactly 2.0
            (0.5, -0.07 / 3),  # 3
            (0.2, -0.0125),  # 4
            (0.0001, -0.002),  # every observation
        ],
    )
    def test_r17_cvar_is_the_mean_of_the_worst_tail(self, confidence, expected):
        assert cvar.historical_cvar(self._RETURNS, confidence) == pytest.approx(expected)

    def test_r17_cvar_float_dust_does_not_add_an_extra_milder_observation(self):
        """``1 - 0.95`` is a hair above 0.05 in binary: 40 x 0.05000...044
        used to round up to a third tail observation."""
        returns = [-0.10, -0.06] + [-0.02] * 38
        assert cvar.historical_cvar(returns, 0.95) == pytest.approx(-0.08)  # 2 rows, not 3

    def test_r17_cvar_tail_is_the_loss_side_however_the_sample_sits(self):
        assert cvar.historical_cvar([0.01, 0.02, 0.03], 0.5) == pytest.approx(0.015)
        assert cvar.historical_cvar(list(reversed(self._RETURNS)), 0.6) == pytest.approx(-0.04)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5, float("nan")])
    def test_r17_cvar_confidence_outside_the_open_interval_is_rejected(self, confidence):
        with pytest.raises(
            ValueError, match=r"historical_cvar: confidence must be in the open interval \(0, 1\)"
        ):
            cvar.historical_cvar(self._RETURNS, confidence)

    def test_r17_cvar_empty_or_entirely_non_finite_input_is_zero(self):
        assert cvar.historical_cvar([]) == 0.0
        assert cvar.historical_cvar(None) == 0.0
        assert cvar.historical_cvar([float("nan"), float("inf"), -float("inf")]) == 0.0
        # Non-finite rows leave before the tail size is derived from the count.
        assert cvar.historical_cvar([float("nan"), -0.02, 0.03], 0.5) == pytest.approx(-0.02)


# ===========================================================================
# risk.kelly.kelly_sizing
# ===========================================================================
class TestKellySizing:
    def test_r17_kelly_interior_value_hand_computed(self):
        """Hold (confidence 0.55, band 0..0.03), a 95 % win rate, symmetric
        payoff: p_eff = 0.5225, full Kelly = p_eff - (1 - p_eff) = 0.045,
        quarter-Kelly = 0.01125 -- inside the band, so nothing is clipped."""
        assert kelly.kelly_sizing("Hold", win_rate=0.95) == pytest.approx(0.01125)
        assert kelly.kelly_sizing("Hold", win_rate=0.95, return_raw=True) == pytest.approx(
            (0.01125, 0.01125)
        )

    def test_r17_kelly_is_capped_at_the_band_and_floored_at_the_band(self):
        # Overweight: confidence 0.72, band 0.03..0.05. b = 2, p_eff = 0.648:
        # full Kelly (2*0.648 - 0.352)/2 = 0.472, quarter = 0.118 -> capped.
        assert kelly.kelly_sizing("Overweight", win_rate=0.9, avg_win=2.0, avg_loss=1.0) == (
            pytest.approx(0.05)
        )
        assert kelly.kelly_sizing(
            "Overweight", win_rate=0.9, avg_win=2.0, avg_loss=1.0, return_raw=True
        ) == pytest.approx((0.05, 0.118))
        # A negative edge is clamped to 0 by the fraction, then floored by the band.
        assert kelly.kelly_sizing("Buy", win_rate=0.01, return_raw=True) == pytest.approx(
            (0.05, 0.0)
        )
        assert kelly.kelly_sizing("Hold", win_rate=0.01, return_raw=True) == pytest.approx(
            (0.0, 0.0)
        )

    def test_r17_kelly_sell_never_takes_a_long_position(self):
        assert kelly.kelly_sizing("Sell", win_rate=1.0) == 0.0
        assert kelly.kelly_sizing("Sell", win_rate=1.0, return_raw=True) == (0.0, 0.0)

    def test_r17_kelly_unknown_rating_never_opens_a_position(self):
        """A stray string from an LLM cannot break sizing: it reads as an
        unknown with confidence 0.5, so p_eff <= 0.5 and the fraction is never
        positive -- whatever win rate the history carries."""
        for win_rate in (0.0, 0.95, 1.0):
            assert kelly.kelly_sizing("STRONG BUY!!", win_rate=win_rate, return_raw=True) == (
                0.0, 0.0
            )

    def test_r17_kelly_payoff_ratio_that_overflows_never_becomes_a_nan_weight(self):
        """A finite ``avg_win / avg_loss`` that overflows to inf makes the
        Kelly formula divide inf by inf. The guard has to catch the NaN
        before the band clip, where ``min``/``max`` would pass it through."""
        sized = kelly.kelly_sizing("Buy", win_rate=0.6, avg_win=1e308, avg_loss=1e-308)
        assert sized == pytest.approx(0.05)
        assert kelly.kelly_sizing(
            "Buy", win_rate=0.6, avg_win=1e308, avg_loss=1e-308, return_raw=True
        ) == pytest.approx((0.05, 0.0))

    def test_r17_kelly_unusable_fraction_multiplier_and_win_rate_fall_back(self):
        assert kelly.kelly_sizing("Hold", win_rate=0.95, kelly_fraction_mult=float("nan")) == (
            pytest.approx(0.01125)
        )
        # A non-finite win rate cannot be blended: the rating's own conviction is
        # used (Hold 0.55), a negative edge at a symmetric payoff.
        assert kelly.kelly_sizing("Hold", win_rate=float("nan"), return_raw=True) == (0.0, 0.0)


# ===========================================================================
# instruments.models.InstrumentRecord.as_disclosure / default_config._coerce
# ===========================================================================
class TestInstrumentDisclosure:
    def test_r17_instrument_disclosure_lists_every_populated_field_in_order(self):
        record = models.InstrumentRecord(
            symbol="MUUSDT",
            instrument_class="stock_perp",
            classification_source="registry",
            classification_confidence=0.987,
            unsupported_reason="non-US equity",
            underlying_type="EQUITY",
            underlying_symbol="MU",
            onboard_date="2026-09-04",
            status="TRADING",
            snapshot_available_at="2026-11-20T00:00:00Z",
            quote_asset="USDT",  # not part of the one-line summary
        )
        assert record.as_disclosure() == (
            "MUUSDT: stock_perp, source=registry, confidence=0.99, reason=non-US equity, "
            "underlying_type=EQUITY, underlying=MU, onboard=2026-09-04, status=TRADING, "
            "snapshot=2026-11-20T00:00:00Z"
        )

    def test_r17_instrument_disclosure_holds_only_the_mandatory_fields(self):
        record = models.InstrumentRecord(
            symbol="BTCUSDT", instrument_class="pure_crypto_perp",
            classification_source="seed", classification_confidence=1.0,
        )
        assert record.as_disclosure() == "BTCUSDT: pure_crypto_perp, source=seed, confidence=1.00"


class TestConfigCoerce:
    @pytest.mark.parametrize(
        ("value", "reference", "expected"),
        [
            (" TRUE ", True, True),
            ("on", False, True),
            ("Off", True, False),
            ("0", True, False),
            ("42", 7, 42),
            ("1.5", 0.5, 1.5),
            ("-3", 0.5, -3.0),
            ("anything", "text", "anything"),
            ("", "text", ""),
        ],
    )
    def test_r17_config_coerce_follows_the_default_values_type(self, value, reference, expected):
        assert default_config._coerce(value, reference) == expected
        assert type(default_config._coerce(value, reference)) is type(expected)

    @pytest.mark.parametrize("bad", ["treu", "yes please", "", "2"])
    def test_r17_config_coerce_a_misspelled_boolean_fails_loudly(self, bad):
        with pytest.raises(ValueError, match=r"expected a boolean \(true/1/yes/on/false/0/no/off\)"):
            default_config._coerce(bad, False)

    @pytest.mark.parametrize(("bad", "reference"), [("four", 7), ("4.5", 7)])
    def test_r17_config_coerce_a_non_numeric_number_fails_loudly(self, bad, reference):
        with pytest.raises(ValueError):
            default_config._coerce(bad, reference)


# ===========================================================================
# logging_config.redact_secrets — the canonical scrubber
# ===========================================================================
class TestRedactSecrets:
    def test_r17_redact_empty_text_is_returned_unchanged(self):
        assert lc.redact_secrets("") == ""
        assert lc.redact_secrets(None) is None

    def test_r17_redact_scrubs_every_layer_and_is_idempotent(self):
        secret = "s3cr3t-v4lu3-n0t-1n-env"
        text = (
            f"call {secret} failed; "
            "url https://www.alphavantage.co/query?function=X&apikey=ABCDEFGH1JKLMNOP&x=1; "
            "json {'apikey': 'KLMNOPQR2STUVWXY'}; tavly tvly-abcdefghijklmnopqrstuvwxyz0123; "
            "zhipu 0123456789abcdef0123456789abcdef.AbCdEfGhIjKl; bare KLMNOPQR2STUVWXY; "
            "letter-only ABCDEFGHIJKLMNOPQ"
        )
        out = lc.redact_secrets(text, extra_secrets=[secret])
        assert out == (
            f"call {lc.REDACTED_PLACEHOLDER} failed; "
            "url https://www.alphavantage.co/query?function=X&apikey=[REDACTED]&x=1; "
            "json {'apikey': '[REDACTED]'}; tavly [REDACTED]; zhipu [REDACTED]; "
            "bare [REDACTED]; letter-only ABCDEFGHIJKLMNOPQ"
        )
        assert lc.redact_secrets(out, extra_secrets=[secret]) == out

    def test_r17_redact_leaves_ordinary_identifiers_alone(self):
        keep = "request id 0123456789abcdef0123456789abcdef, order 92, thread 7f3a9c1e2b5d, word dictionary"
        assert lc.redact_secrets(keep) == keep


# ===========================================================================
# logging_config.redact_secrets — the deployment's own credentials
# ===========================================================================
#: DeepSeek's key shape: ``sk-`` followed by 32 lower-case hex characters.
_LLM_KEY = "sk-" + "0123456789abcdef0123456789abcdef"


class TestRedactDeploymentCredentials:
    def test_r17_redact_knows_the_key_env_of_every_supported_llm_provider(self):
        """The exact-match layer is only as good as its list. The project
        switched its whole LLM chain to DeepSeek on 2026-09-30 and the list
        never followed: ``DEEPSEEK_API_KEY`` -- the one credential every run
        uses -- sat in no layer at all (not in the list, and ``sk-<32 hex>``
        matches no shape), as did the 17 other provider keys the registry in
        ``api_key_env`` names. A key in that state is redacted only by luck."""
        from yialpha.llm_clients.api_key_env import PROVIDER_API_KEY_ENV

        missing = sorted(
            name for name in PROVIDER_API_KEY_ENV.values() if name and name not in lc._KEYRING_ENVS
        )
        assert missing == []

    def test_r17_redact_catches_the_deepseek_key_shape_without_the_env(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        text = f"AuthenticationError: Incorrect API key provided: {_LLM_KEY}."
        assert lc.redact_secrets(text) == (
            "AuthenticationError: Incorrect API key provided: [REDACTED]."
        )

    def test_r17_redact_deepseek_key_from_the_env_wherever_it_is_pasted(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", _LLM_KEY)
        assert lc.redact_secrets(f"Authorization: Bearer {_LLM_KEY}") == (
            "Authorization: Bearer [REDACTED]"
        )

    @pytest.mark.parametrize("name", ["ZHIPU_CN_API_KEY", "TAVILY_API_KEY", "BINANCE_API_SECRET"])
    def test_r17_redact_a_configured_value_too_short_to_be_a_key_is_ignored(self, monkeypatch, name):
        """A two-character "credential" in a key variable is a placeholder,
        not a secret. Exact-matching it used to take every ``x1`` out of every
        line. A value the CALLER names is honoured whatever its length, and a
        key-sized configured value is still exact-matched."""
        monkeypatch.setenv(name, "x1")
        text = "attempt x1 of 3: x1 node online"
        assert lc.redact_secrets(text) == text
        assert lc.redact_secrets(text, extra_secrets=["x1"]) == (
            "attempt [REDACTED] of 3: [REDACTED] node online"
        )
        monkeypatch.setenv(name, "alpha123")
        assert lc.redact_secrets("rotated to alpha123 failed") == "rotated to [REDACTED] failed"

    @pytest.mark.parametrize(
        "keep",
        [
            f"task risk-{_LLM_KEY[3:]} ended",  # "sk-" glued to a word: not a key position
            f"{_LLM_KEY}f",  # 33 hex characters
            f"sk-{_LLM_KEY[3:].upper()}",  # upper-case hex
            "sk-0123456789abcdef0123456789abcde",  # 31
        ],
    )
    def test_r17_redact_leaves_lookalikes_of_the_deepseek_shape_alone(self, monkeypatch, keep):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        assert lc.redact_secrets(keep) == keep


class TestRedactQuotedApikey:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('{"apikey": "Zm9vYmFyQkFZ", "limit": 5}', '{"apikey": "[REDACTED]", "limit": 5}'),
            ("{'apikey': 'Zm9vYmFyQkFZ'}", "{'apikey': '[REDACTED]'}"),
            ("apikey='Zm9vYmFyQkFZ'", "apikey='[REDACTED]'"),
            ('"apikey":"Zm9vYmFyQkFZ"', '"apikey":"[REDACTED]"'),
            ("APIKEY: Zm9vYmFyQkFZ", "APIKEY: [REDACTED]"),
            ("apikey=Zm9vYmFyQkFZ&x=1", "apikey=[REDACTED]&x=1"),
        ],
    )
    def test_r17_redact_apikey_parameter_in_every_spelling(self, text, expected):
        """A credential named by position is a credential however the name is
        quoted. The parameter layer used to require ``apikey`` to touch its
        ``=`` / ``:`` directly, so the JSON spelling -- ``"apikey": "..."``,
        the way a request body is logged -- fell through to the shape layers,
        which only know three key families."""
        assert lc.redact_secrets(text) == expected

    @pytest.mark.parametrize(
        "text",
        ["pass your apikey in the query string", "apikeys are secret", "apikey", "an api-key"],
    )
    def test_r17_redact_the_word_apikey_without_a_value_is_left_alone(self, text):
        assert lc.redact_secrets(text) == text


# ===========================================================================
# logging_config.CredentialRedactionFilter — the handler-level fallback
# ===========================================================================
#: Alpha Vantage's key shape, deliberately not present in any env var.
_AV_KEY = "ABCDEFGH1JKLMNOP"
_HANDLER_KINDS = ["stream", "rich"]


def _handler(kind: str) -> logging.Handler:
    if kind == "stream":
        return logging.StreamHandler(io.StringIO())
    from rich.console import Console
    from rich.logging import RichHandler

    # What setup_logging() installs: rich tracebacks drawn from the live exception.
    return RichHandler(
        show_time=False,
        show_level=False,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        console=Console(file=io.StringIO(), force_terminal=False, width=200, legacy_windows=False),
    )


def _sink(handler: logging.Handler) -> io.StringIO:
    return handler.stream if isinstance(handler, logging.StreamHandler) else handler.console.file


def _logger_for(*handlers: logging.Handler) -> logging.Logger:
    """A private logger whose handlers are exactly ``handlers`` (each with the filter)."""
    for handler in handlers:
        handler.addFilter(lc.CredentialRedactionFilter())
    log = logging.Logger("r17_redaction_probe")
    for handler in handlers:
        log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    return log


def _emitted(handler: logging.Handler, emit) -> str:
    log = _logger_for(handler)
    emit(log)
    return _sink(handler).getvalue()


def _key_in_exception(log: logging.Logger) -> None:
    """What ``requests`` raises after ``raise_for_status()``: the URL, key and all."""
    try:
        raise RuntimeError(
            "403 Client Error: Forbidden for url: "
            f"https://www.alphavantage.co/query?function=X&apikey={_AV_KEY}"
        )
    except RuntimeError as exc:
        log.error("fetch failed", exc_info=exc)


class TestRedactionFilterMessage:
    @pytest.mark.parametrize("kind", _HANDLER_KINDS)
    def test_r17_redaction_filter_scrubs_a_key_passed_as_a_message_argument(self, kind):
        out = _emitted(_handler(kind), lambda log: log.warning("vendor rejected key %s", _AV_KEY))
        assert _AV_KEY not in out
        assert "[REDACTED]" in out and "vendor rejected key" in out

    def test_r17_redaction_filter_rewrites_the_record_once_and_clears_the_args(self):
        record = logging.LogRecord("x", logging.WARNING, "f", 1, "key %s", (_AV_KEY,), None)
        assert lc.CredentialRedactionFilter().filter(record) is True
        assert record.msg == "key [REDACTED]"
        assert record.args is None  # already interpolated: never formatted a second time
        assert record.getMessage() == "key [REDACTED]"

    def test_r17_redaction_filter_scrubs_a_traceback_text_already_on_the_record(self):
        record = logging.LogRecord("x", logging.ERROR, "f", 1, "boom", None, None)
        record.exc_text = f"Traceback (most recent call last):\nRuntimeError: denied for {_AV_KEY}"
        lc.CredentialRedactionFilter().filter(record)
        assert record.exc_text == "Traceback (most recent call last):\nRuntimeError: denied for [REDACTED]"

    def test_r17_redaction_filter_passes_a_clean_record_through_untouched(self):
        record = logging.LogRecord("x", logging.INFO, "f", 1, "loaded %d rows", (42,), None)
        assert lc.CredentialRedactionFilter().filter(record) is True
        assert (record.msg, record.args, record.getMessage()) == ("loaded %d rows", (42,), "loaded 42 rows")


class TestRedactionFilterExcInfo:
    @pytest.mark.parametrize("kind", _HANDLER_KINDS)
    def test_r17_redaction_filter_scrubs_a_key_carried_in_exc_info(self, kind):
        """``logger.exception()`` / ``exc_info=True`` is how vendor errors are
        logged, and those errors carry the key: ``requests`` puts the whole
        URL in its ``HTTPError`` message.

        A fresh record holds the live exception, not its text: the traceback
        is rendered by the HANDLER, after this filter has run -- the default
        formatter from ``exc_text``, ``RichHandler`` straight off the
        exception object. The filter only ever looked at ``record.exc_text``,
        which is still None at that point, so the key reached the sink
        unredacted on both handlers."""
        out = _emitted(_handler(kind), _key_in_exception)
        assert _AV_KEY not in out
        assert "apikey=[REDACTED]" in out
        # The traceback is still reported in full.
        assert "Traceback (most recent call last)" in out
        assert "RuntimeError: 403 Client Error: Forbidden for url" in out
        assert "in _key_in_exception" in out

    def test_r17_redaction_filter_scrubs_every_link_of_a_chained_exception(self):
        def emit(log: logging.Logger) -> None:
            try:
                try:
                    raise ValueError(f"inner cause names {_AV_KEY}")
                except ValueError as inner:
                    raise RuntimeError("outer failure") from inner
            except RuntimeError:
                log.exception("chained")

        out = _emitted(_handler("stream"), emit)
        assert _AV_KEY not in out
        assert out.count("[REDACTED]") == 1
        assert "ValueError: inner cause names [REDACTED]" in out
        assert "RuntimeError: outer failure" in out
        assert "The above exception was the direct cause" in out

    def test_r17_redaction_filter_leaves_a_clean_exception_to_its_handler(self):
        """Only records that actually carry credential material are rewritten:
        an exception without one keeps its ``exc_info`` (rich tracebacks keep
        working) and is not pre-formatted by the filter."""
        try:
            raise RuntimeError("plain failure")
        except RuntimeError:
            record = logging.LogRecord("x", logging.ERROR, "f", 1, "boom", None, sys.exc_info())
        exc_info = record.exc_info
        assert lc.CredentialRedactionFilter().filter(record) is True
        assert record.exc_info is exc_info  # the live exception stays with the record
        assert record.exc_text is None  # the handler still owns the rendering
        out = _emitted(_handler("stream"), lambda log: log.handle(record))
        assert "Traceback (most recent call last)" in out
        assert "RuntimeError: plain failure" in out


class TestRedactionFilterMalformedRecord:
    def test_r17_redaction_filter_leaves_no_raw_secret_behind_a_malformed_format(self, monkeypatch):
        """A record that cannot be formatted is reported by logging itself --
        and that report prints ``record.args`` verbatim. When the arguments
        hold a key, the filter has to make sure nothing raw is left for it."""
        monkeypatch.setattr(logging, "raiseExceptions", True)
        err = io.StringIO()
        handler = _handler("stream")
        with contextlib.redirect_stderr(err):
            _emitted(handler, lambda log: log.warning("100%% done for %s and %s", _AV_KEY))
        out = _sink(handler).getvalue()
        assert _AV_KEY not in out
        assert _AV_KEY not in err.getvalue()
        assert "[REDACTED]" in out

    def test_r17_redaction_filter_keeps_a_secret_free_malformed_format_visible(self, monkeypatch):
        """Without a secret in play the programming error stays as loud as
        logging makes it today."""
        monkeypatch.setattr(logging, "raiseExceptions", True)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            _emitted(_handler("stream"), lambda log: log.warning("100%% done for %s and %s", "x"))
        assert "--- Logging error ---" in err.getvalue()
        assert "TypeError: not enough arguments for format string" in err.getvalue()


# ===========================================================================
# logging_config.setup_logging
# ===========================================================================
class _StubRichHandler(logging.Handler):
    """Stands in for ``rich.logging.RichHandler``: records its options, emits nowhere."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.kwargs = kwargs


class TestSetupLogging:
    @pytest.fixture()
    def bare_root(self, monkeypatch: pytest.MonkeyPatch):
        """``bare_root()`` hands out an EMPTY root logger; the real one is put
        back afterwards. Other entry points install a real handler at import
        time, and the logging plugin re-attaches its capture handler for every
        test phase, so the handlers are cleared when the test asks -- not when
        the fixture is set up."""
        root = logging.getLogger()
        saved_level, saved_handlers = root.level, list(root.handlers)
        saved_levels = {name: logging.getLogger(name).level for name in lc._NOISY_LOGGERS}
        monkeypatch.delenv("YIALPHA_LOG_LEVEL", raising=False)
        monkeypatch.setattr(lc, "_get_rich_handler_cls", lambda: _StubRichHandler)

        def start() -> logging.Logger:
            root.handlers = []
            return root

        yield start
        root.setLevel(saved_level)
        root.handlers = saved_handlers
        for name, level in saved_levels.items():
            logging.getLogger(name).setLevel(level)

    def test_r17_setup_logging_installs_one_fully_equipped_handler(self, bare_root):
        root = bare_root()
        stray = logging.NullHandler()  # e.g. a library's basicConfig handler
        root.addHandler(stray)
        lc.setup_logging()
        (handler,) = root.handlers
        assert isinstance(handler, _StubRichHandler)
        assert stray not in root.handlers
        assert (root.level, handler.level) == (logging.INFO, logging.INFO)
        assert [type(f) for f in handler.filters] == [lc.CredentialRedactionFilter]
        assert handler.formatter is not None
        assert (handler.formatter._fmt, handler.formatter.datefmt) == ("%(name)s: %(message)s", "%H:%M:%S")
        assert handler.kwargs == {
            "show_time": True,
            "show_level": True,
            "show_path": False,
            "rich_tracebacks": True,
            "tracebacks_show_locals": False,
        }
        assert {logging.getLogger(name).level for name in lc._NOISY_LOGGERS} == {logging.WARNING}

    def test_r17_setup_logging_second_call_moves_the_level_without_a_second_handler(self, bare_root):
        root = bare_root()
        lc.setup_logging()
        (handler,) = root.handlers
        lc.setup_logging("DEBUG")
        assert root.handlers == [handler]
        assert (root.level, handler.level) == (logging.DEBUG, logging.DEBUG)
        # A filter someone stripped off the existing handler is put back.
        handler.filters = []
        lc.setup_logging("ERROR")
        assert root.handlers == [handler]
        assert (root.level, handler.level) == (logging.ERROR, logging.ERROR)
        assert [type(f) for f in handler.filters] == [lc.CredentialRedactionFilter]

    @pytest.mark.parametrize(
        ("env", "explicit", "expected"),
        [
            ("debug", None, logging.DEBUG),
            ("Error", None, logging.ERROR),
            ("NOTSET", None, logging.NOTSET),
            ("", None, logging.INFO),  # an empty variable reads as unset
            (None, None, logging.INFO),  # nothing set anywhere
            ("debug", "CRITICAL", logging.CRITICAL),  # the argument wins
        ],
    )
    def test_r17_setup_logging_resolves_the_level_from_the_argument_then_the_env(
        self, bare_root, monkeypatch, env, explicit, expected
    ):
        root = bare_root()
        if env is not None:
            monkeypatch.setenv("YIALPHA_LOG_LEVEL", env)
        lc.setup_logging(explicit)
        assert root.level == expected

    @pytest.mark.parametrize("bad", ["VERBOSE", "info ", "3"])
    def test_r17_setup_logging_rejects_an_unknown_level_name(self, bare_root, monkeypatch, bad):
        """A misspelled level fails at startup instead of degrading to WARNING
        -- and it fails before anything is installed."""
        root = bare_root()
        monkeypatch.setenv("YIALPHA_LOG_LEVEL", bad)
        with pytest.raises(ValueError) as exc_info:
            lc.setup_logging()
        assert str(exc_info.value) == (
            f"Invalid YIALPHA_LOG_LEVEL {bad.upper()!r}; "
            "expected one of CRITICAL/DEBUG/ERROR/INFO/NOTSET/WARNING"
        )
        assert root.handlers == []


# ===========================================================================
# yialpha/__init__ — the package must import without python-dotenv
# ===========================================================================
class TestPackageInit:
    def test_r17_package_imports_without_python_dotenv(self, monkeypatch):
        """``python-dotenv`` is a declared dependency, but its absence must
        never keep the package from importing: the ``.env`` loading is skipped
        and the version is still exposed. The module body is re-executed under
        a private name so the real package (and the test process's
        environment) is untouched."""
        monkeypatch.setitem(sys.modules, "dotenv", None)  # ``from dotenv import ...`` now fails
        spec = importlib.util.spec_from_file_location("_yialpha_init_probe", yialpha.__file__)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert "find_dotenv" not in vars(module) and "load_dotenv" not in vars(module)
        assert module.__version__ == yialpha.__version__


# ===========================================================================
# accuracy — the history scan
# ===========================================================================
def _write_log(root: Path, ticker: str, name: str, **fields: object) -> None:
    folder = root / ticker / "YiAlphaStrategy_logs"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"full_states_log_{name}.json").write_text(json.dumps(fields), encoding="utf-8")


def _record(rating: str = "Buy", trade_date: str = "2026-01-05", ticker: str = "A"):
    return acc.DecisionRecord(ticker, trade_date, rating, "stock", None, "legacy", "inferred", "p")


def _sessions(start: str, n: int) -> list[date]:
    return [d.date() for d in pd.bdate_range(start, periods=n)]


class TestScanHistory:
    def test_r17_scan_of_a_missing_directory_is_empty_not_an_error(self, tmp_path):
        assert acc.scan_history(tmp_path / "never-created") == []
        assert not (tmp_path / "never-created").exists()

    def test_r17_scan_skips_a_log_without_a_trade_date_and_keeps_the_rest(self, tmp_path):
        """The folder name is only the ticker of last resort; the trade date
        has to come from the log. One without it has nothing to score and is
        left out -- the readable logs beside it still load."""
        _write_log(tmp_path, "AAPL", "undated", company_of_interest="AAPL", pm_rating="Buy")
        _write_log(tmp_path, "AAPL", "timestamped",
                   company_of_interest="AAPL", trade_date="2026-01-05 14:30:00", pm_rating="Sell")
        _write_log(tmp_path, "MSFT", "2026-01-05", trade_date="2026-01-05",
                   final_trade_decision="**Rating**: Buy\nmore")
        records = acc.scan_history(tmp_path)
        assert [(r.ticker, r.trade_date, r.rating) for r in records] == [
            ("AAPL", "2026-01-05", "Sell"),  # a timestamped date is cut to its day
            ("MSFT", "2026-01-05", "Buy"),  # ticker from the folder, rating from the markdown
        ]

    def test_r17_default_results_dir_is_the_configured_one(self, monkeypatch, tmp_path):
        with monkeypatch.context() as patch:
            patch.setitem(default_config.DEFAULT_CONFIG, "results_dir", str(tmp_path / "reports"))
            assert acc._default_results_dir() == Path(tmp_path / "reports")


# ===========================================================================
# accuracy — the price seams
# ===========================================================================
def _ohlc(closes: list[float], days: list[date]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": [1.0] * len(closes)},
        index=pd.DatetimeIndex(pd.to_datetime(days), name="Date"),
    )


_FAR_NOW_MS = 1_800_000_000_000  # 2027-01-15T08:00:00Z: later than every date used below


def _ms_of(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)


class TestDatedClose:
    def test_r17_dated_close_crypto_asks_for_closed_bars_on_the_native_venue(self, monkeypatch):
        days = _sessions("2026-01-05", 6)
        seen: dict[str, object] = {}

        def klines(ticker, start, end, interval, venue, closed_as_of):
            seen.update(ticker=ticker, start=start, end=end, interval=interval, venue=venue,
                        closed_as_of=closed_as_of)
            return _ohlc([100.0 + i for i in range(6)], days)

        monkeypatch.setattr(bnc, "binance_klines_frame", klines)
        monkeypatch.setattr(bnc, "_now_ms", lambda: _FAR_NOW_MS)
        series = acc._dated_close("BTCUSDT", "crypto_perp", "2026-01-05", "2026-01-20")
        assert seen == {
            "ticker": "BTCUSDT", "start": "2026-01-05", "end": "2026-01-20", "interval": "1d",
            "venue": "binance_perp", "closed_as_of": _FAR_NOW_MS,  # the forming bar is dropped upstream
        }
        assert list(series.index) == days
        assert series.tolist() == [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]

    @pytest.mark.parametrize(("asset_type", "venue"), [("crypto_spot", "binance_spot"),
                                                       ("crypto_perp", "binance_perp")])
    def test_r17_dated_close_crypto_cutoff_caps_visibility_at_that_days_close(
        self, monkeypatch, asset_type, venue
    ):
        seen: dict[str, object] = {}
        monkeypatch.setattr(bnc, "binance_klines_frame",
                            lambda *a, **k: seen.update(closed_as_of=k["closed_as_of"],
                                                    venue=k["venue"], end=a[-1]))
        monkeypatch.setattr(bnc, "_now_ms", lambda: _FAR_NOW_MS)
        acc._crypto_daily_frame("ETHUSDT", asset_type, "2026-01-05", "2026-02-20",
                                cutoff=datetime(2026, 1, 10))
        # min(wall clock, cutoff day 23:59:59.999 UTC). The cutoff is a NAIVE
        # date: it is read as UTC, so the instant is the same on every host.
        assert seen["closed_as_of"] == _ms_of(date(2026, 1, 11)) - 1
        assert seen["venue"] == venue
        # A cutoff later than the wall clock does not make future bars visible.
        acc._crypto_daily_frame("ETHUSDT", asset_type, "2026-01-05", "2026-06-20",
                                cutoff=datetime(2027, 6, 1))
        assert seen["closed_as_of"] == _FAR_NOW_MS
        assert seen["end"] == "2026-06-20"  # the window itself is not trimmed

    def test_r17_dated_close_stock_never_requests_todays_forming_bar(self, monkeypatch):
        seen: dict[str, object] = {}

        def history(symbol, start, end):
            seen.update(symbol=symbol, start=start, end=end)
            return _ohlc([1.0, 2.0], _sessions("2026-01-05", 2))

        monkeypatch.setattr(yfmod, "get_YFin_history_cached", history)
        monkeypatch.setattr(acc, "_us_equity_today", lambda: date(2026, 1, 9))
        acc._dated_close("aapl", "stock", "2026-01-05", "2026-02-20")
        # yfinance's end is exclusive: ending ON today leaves today's row out.
        assert seen == {"symbol": normalize_symbol("aapl"), "start": "2026-01-05", "end": "2026-01-09"}
        # An end date already in the past is left alone.
        acc._dated_close("aapl", "stock", "2025-12-01", "2025-12-20")
        assert seen == {"symbol": normalize_symbol("aapl"), "start": "2025-12-01",
                        "end": "2025-12-20"}

    @pytest.mark.parametrize("asset_type", ["stock", "crypto_perp"])
    def test_r17_dated_close_vendor_failure_is_an_empty_series_not_a_crash(
        self, monkeypatch, caplog, asset_type
    ):
        def down(*args, **kwargs):
            raise RuntimeError("delisted")

        monkeypatch.setattr(bnc, "binance_klines_frame", down)
        monkeypatch.setattr(yfmod, "get_YFin_history_cached", down)
        with caplog.at_level(logging.WARNING, logger="yialpha.accuracy"):
            series = acc._dated_close("X", asset_type, "2026-01-05", "2026-01-20")
        assert series.empty and series.dtype == float
        assert f"Forward-price fetch failed for X ({asset_type}): delisted" in caplog.text


# ===========================================================================
# accuracy — forward_outcome and the report
# ===========================================================================
class TestForwardOutcome:
    def test_r17_forward_outcome_counts_sessions_from_the_decision_date_close(self, monkeypatch):
        """Baseline = the decision date's own close; the endpoint is
        ``holding_days`` SESSIONS later -- the weekend after the first bar is
        not a day. This is the anchor the rendered report names."""
        days = _sessions("2026-01-05", 12)  # 01-05 (Mon) .. 01-09, then 01-12 ..
        assert days[5] == date(2026, 1, 12)
        closes = pd.Series([100.0 + 10 * i for i in range(12)], index=days, dtype=float)
        asked: list[tuple] = []

        def dated(ticker, asset_type, start, end):
            asked.append((ticker, asset_type, start, end))
            return closes

        monkeypatch.setattr(acc, "_dated_close", dated)
        assert acc.forward_outcome(_record(), holding_days=5) == {
            "raw_return": 0.5, "baseline_close": 100.0, "end_close": 150.0, "days": 5,
        }
        # The window asked of the vendor: holding_days * 2 + 10 calendar days.
        assert asked == [("A", "stock", "2026-01-05", "2026-01-25")]

    def test_r17_forward_outcome_pending_until_the_whole_horizon_has_closed(self, monkeypatch):
        days = _sessions("2026-01-05", 8)
        closes = pd.Series([100.0 + i for i in range(8)], index=days, dtype=float)
        # Exactly holding_days + 1 bars: the last window that can be scored.
        monkeypatch.setattr(acc, "_dated_close", lambda *a, **k: closes.iloc[:6])
        assert acc.forward_outcome(_record(), holding_days=5)["raw_return"] == pytest.approx(0.05)
        # One bar fewer: not elapsed. Pending -- never a partial score, never a miss.
        monkeypatch.setattr(acc, "_dated_close", lambda *a, **k: closes.iloc[:5])
        assert acc.forward_outcome(_record(), holding_days=5) is None
        monkeypatch.setattr(acc, "_dated_close", lambda *a, **k: pd.Series(dtype=float))
        assert acc.forward_outcome(_record(), holding_days=5) is None

    def test_r17_forward_outcome_an_unreadable_trade_date_is_pending(self, monkeypatch):
        monkeypatch.setattr(acc, "_dated_close", lambda *a, **k: pytest.fail("no fetch for a bad date"))
        assert acc.forward_outcome(_record(trade_date="2026/01/05")) is None

    @pytest.mark.parametrize(
        ("closes", "holding_days", "expected"),
        [
            ([100.0, 101.0, float("nan"), 103.0], 2,
             {"raw_return": None, "unscorable": "non-finite close in the forward window"}),
            ([100.0, 101.0, float("inf"), 103.0], 2,
             {"raw_return": None, "unscorable": "non-finite close in the forward window"}),
            ([float("nan"), 101.0, 102.0, 103.0], 2,
             {"raw_return": None, "unscorable": "non-finite close in the forward window"}),
            ([100.0, float("nan"), 102.0, 103.0], 2,
             {"raw_return": 0.02, "baseline_close": 100.0, "end_close": 102.0, "days": 2}),
        ],
    )
    def test_r17_forward_outcome_a_non_finite_close_at_either_end_is_unscorable(
        self, monkeypatch, closes, holding_days, expected
    ):
        """A NaN endpoint used to flow through as ``raw_return=NaN`` and get
        labelled ``miss`` (NaN compares False). Only the two ends matter: a
        hole strictly between them is not part of the return."""
        series = pd.Series(closes, index=_sessions("2026-01-05", len(closes)), dtype=float)
        monkeypatch.setattr(acc, "_dated_close", lambda *a, **k: series)
        assert acc.forward_outcome(_record(), holding_days=holding_days) == pytest.approx(expected)

    @pytest.mark.parametrize("baseline", [0.0, -5.0])
    def test_r17_forward_outcome_a_non_positive_baseline_has_no_return(self, monkeypatch, baseline):
        series = pd.Series([baseline, 101.0, 102.0], index=_sessions("2026-01-05", 3), dtype=float)
        monkeypatch.setattr(acc, "_dated_close", lambda *a, **k: series)
        assert acc.forward_outcome(_record(), holding_days=2) is None


class TestOutcomeLabel:
    @pytest.mark.parametrize(
        ("rating", "raw", "label"),
        [
            ("Buy", 0.01, "hit"), ("Buy", -0.01, "miss"), ("Overweight", 0.0, "miss"),
            ("Sell", -0.01, "hit"), ("Sell", 0.01, "miss"), ("Underweight", 0.0, "miss"),
            ("Buy", None, "pending"), ("Hold", 0.5, "neutral"), ("Hold", -0.5, "neutral"),
        ],
    )
    def test_r17_outcome_label(self, rating, raw, label):
        assert acc._outcome_label(rating, raw) == label


class TestAccuracyReport:
    def test_r17_report_without_a_directional_sample_has_no_hit_rate(self, monkeypatch):
        """Nothing to divide by: the rate is None and the report says so,
        rather than printing 0/0 or NaN."""
        monkeypatch.setattr(acc, "forward_outcome", lambda record, holding_days=5: None)
        report = acc.build_accuracy_report([_record("Buy"), _record("Sell", trade_date="2026-01-06")])
        assert report["direction"] == {"n": 0, "hits": 0, "hit_rate": None}
        assert (report["total_runs"], report["scored"], report["pending"]) == (2, 0, 2)
        assert report["hold_mean_return"] is None and report["hold_n"] == 0
        assert report["by_rating"] == {}  # nothing scored, so no bucket opens
        markdown = acc.render_markdown(report)
        assert "- No fully-elapsed directional decisions yet." in markdown
        assert "0/0" not in markdown and "nan%" not in markdown and "NaN" not in markdown

    def test_r17_report_keeps_pending_hold_and_unscorable_out_of_the_directional_count(
        self, monkeypatch
    ):
        def scored(raw_return: float) -> dict:
            return {"raw_return": raw_return, "baseline_close": 100.0,
                    "end_close": 100.0 * (1.0 + raw_return), "days": 5}

        outcomes = {
            "2026-01-01": None,  # Buy, horizon not elapsed
            "2026-01-02": scored(0.10),  # Buy hit
            "2026-01-03": scored(-0.04),  # Buy miss
            "2026-01-04": scored(0.02),  # Sell miss
            "2026-01-05": scored(-0.03),  # Sell hit
            "2026-01-06": scored(0.07),  # Hold: no direction to score
            "2026-01-07": {"raw_return": None,
                           "unscorable": "non-finite close in the forward window"},  # Buy
            "2026-01-08": None,  # Hold, horizon not elapsed
        }
        ratings = {"2026-01-01": "Buy", "2026-01-02": "Buy", "2026-01-03": "Buy",
                   "2026-01-04": "Sell", "2026-01-05": "Sell", "2026-01-06": "Hold",
                   "2026-01-07": "Buy", "2026-01-08": "Hold"}
        records = [_record(ratings[d], trade_date=d) for d in sorted(ratings)]
        monkeypatch.setattr(acc, "forward_outcome",
                            lambda record, holding_days=5: outcomes[record.trade_date])
        report = acc.build_accuracy_report(records, holding_days=5)
        assert report["direction"] == {"n": 4, "hits": 2, "hit_rate": 0.5}
        assert (report["total_runs"], report["scored"], report["pending"],
                report["excluded_unscorable"]) == (8, 5, 2, 1)
        assert report["hold_mean_return"] == pytest.approx(0.07) and report["hold_n"] == 1
        assert report["by_rating"]["Buy"] == {
            "n": 2, "mean_return": pytest.approx(0.03), "directional_n": 2, "hits": 1,
            "hit_rate": 0.5,
        }
        assert report["by_rating"]["Sell"] == {
            "n": 2, "mean_return": pytest.approx(-0.005), "directional_n": 2, "hits": 1,
            "hit_rate": 0.5,
        }
        assert report["by_rating"]["Hold"] == {
            "n": 1, "mean_return": pytest.approx(0.07), "directional_n": 0, "hits": 0,
            "hit_rate": None,
        }
        rows = {r["trade_date"]: r for r in report["records"]}
        assert (rows["2026-01-01"]["outcome"], rows["2026-01-01"]["forward_return"]) == (
            "pending", None,
        )  # a pending horizon is not a miss
        assert (rows["2026-01-07"]["outcome"], rows["2026-01-07"]["forward_return"]) == (
            "unscorable", None,
        )
        assert (rows["2026-01-06"]["outcome"], rows["2026-01-06"]["forward_return"]) == (
            "neutral", 0.07,
        )
        assert "baseline_close" not in rows["2026-01-01"]
        assert rows["2026-01-02"]["baseline_close"] == 100.0
        markdown = acc.render_markdown(report)
        assert "**2/4 correct (50.0%)**" in markdown
        assert "unscorable 1 (non-finite forward close — excluded from every aggregate)" in markdown
        assert ("Hold entries: n=1, mean forward return +7.00% (opportunity-cost view, "
                "not scored for direction)") in markdown
        assert "Baseline: decision-date close" in markdown


# ===========================================================================
# accuracy.fetch_returns_yf — the memory-resolution seam
# ===========================================================================
def _closes_frame(closes: list[float], days: list[date]) -> pd.DataFrame:
    return pd.DataFrame({"Close": closes}, index=pd.DatetimeIndex(pd.to_datetime(days)))


class TestFetchReturns:
    @pytest.fixture()
    def vendor(self, monkeypatch: pytest.MonkeyPatch):
        """``vendor({symbol: closes})`` installs a yfinance stand-in on
        business days from 2026-01-05 and returns the request log."""
        asked: list[tuple[str, str, str]] = []

        def install(table: dict[str, list[float]]) -> list[tuple[str, str, str]]:
            def history(symbol, start, end):
                asked.append((symbol, start, end))
                return _closes_frame(table[symbol], _sessions("2026-01-05", len(table[symbol])))
            monkeypatch.setattr(yfmod, "get_YFin_history_cached", history)
            return asked

        monkeypatch.setattr(acc, "_us_equity_today", lambda: date(2026, 6, 1))
        return install

    def test_r17_fetch_returns_an_as_of_on_or_before_the_decision_leaves_it_pending(self, vendor):
        asked = vendor({"AAPL": [100.0] * 8, "SPY": [500.0] * 8})
        assert acc.fetch_returns_yf("AAPL", "2026-01-05", as_of_date="2026-01-05") == (None, None, None)
        assert acc.fetch_returns_yf("AAPL", "2026-01-05", as_of_date="2026-01-04") == (None, None, None)
        assert asked == []  # nothing has elapsed: no vendor is even asked

    def test_r17_fetch_returns_live_counts_sessions_and_the_benchmark(self, vendor):
        asked = vendor({"AAPL": [100.0 + i for i in range(12)], "SPY": [500.0 + i for i in range(12)]})
        raw, alpha, days = acc.fetch_returns_yf("AAPL", "2026-01-05", holding_days=5)
        assert raw == pytest.approx(105.0 / 100.0 - 1.0)
        assert alpha == pytest.approx((105.0 / 100.0 - 1.0) - (505.0 / 500.0 - 1.0))
        assert days == 5
        # Window = holding_days + 7 calendar days, identical for both legs.
        assert asked == [("AAPL", "2026-01-05", "2026-01-17"), ("SPY", "2026-01-05", "2026-01-17")]

    def test_r17_fetch_returns_live_with_a_short_tape_reports_the_days_it_did_get(self, vendor):
        """Without an as-of date a short history is scored over the sessions
        that exist, and the count is returned so the caller can see it."""
        vendor({"AAPL": [100.0, 101.0, 102.0], "SPY": [500.0] * 3})
        assert acc.fetch_returns_yf("AAPL", "2026-01-05", holding_days=5) == (
            pytest.approx(0.02), pytest.approx(0.02), 2,
        )

    @pytest.mark.parametrize(
        "table",
        [
            {"AAPL": [], "SPY": [500.0] * 8},  # no asset rows
            {"AAPL": [100.0] * 8, "SPY": []},  # no benchmark rows
            {"AAPL": [100.0], "SPY": [500.0] * 8},  # one row cannot form a return
            {"AAPL": [100.0] * 8, "SPY": [500.0]},
        ],
    )
    def test_r17_fetch_returns_needs_two_rows_on_both_legs(self, vendor, table):
        vendor(table)
        assert acc.fetch_returns_yf("AAPL", "2026-01-05", holding_days=5) == (None, None, None)

    def test_r17_fetch_returns_with_an_as_of_needs_the_whole_horizon(self, vendor):
        asked = vendor({"AAPL": [100.0 + i for i in range(20)], "SPY": [500.0 + i for i in range(20)]})
        # 01-12 is the 6th session (01-05, 06, 07, 08, 09, 12): one session short.
        assert acc.fetch_returns_yf("AAPL", "2026-01-05", holding_days=5, as_of_date="2026-01-09") == (
            None, None, None,
        )
        raw, alpha, days = acc.fetch_returns_yf("AAPL", "2026-01-05", holding_days=5,
                                                as_of_date="2026-01-12")
        assert (raw, alpha, days) == (pytest.approx(0.05), pytest.approx(0.05 - 0.01), 5)
        # The vendor window ends the day AFTER the as-of date (yfinance's end is exclusive).
        assert asked[-2:] == [("AAPL", "2026-01-05", "2026-01-13"), ("SPY", "2026-01-05", "2026-01-13")]

    def test_r17_fetch_returns_trims_a_vendor_that_serves_bars_past_the_as_of_date(self, vendor):
        """PIT belt: rows after the as-of date are dropped on the client even
        when the vendor returns them, so the endpoint is the as-of session."""
        vendor({"AAPL": [100.0 + i for i in range(20)], "SPY": [500.0 + i for i in range(20)]})
        raw, _alpha, days = acc.fetch_returns_yf("AAPL", "2026-01-05", holding_days=12,
                                                 as_of_date="2026-01-12")
        # Only 6 sessions are visible (01-05 .. 01-12): a 12-session outcome is pending.
        assert (raw, days) == (None, None)
        raw, _alpha, days = acc.fetch_returns_yf("AAPL", "2026-01-05", holding_days=5,
                                                 as_of_date="2026-01-12")
        assert raw == pytest.approx(105.0 / 100.0 - 1.0)  # close[5], not close[12]

    def test_r17_fetch_returns_as_of_today_leaves_todays_forming_bar_out(self, monkeypatch, vendor):
        """A live rerun on the boundary day must not resolve an entry against
        a session that is still trading."""
        vendor({"AAPL": [100.0 + i for i in range(20)], "SPY": [500.0 + i for i in range(20)]})
        monkeypatch.setattr(acc, "_us_equity_today", lambda: date(2026, 1, 12))
        assert acc.fetch_returns_yf("AAPL", "2026-01-05", holding_days=5, as_of_date="2026-01-12") == (
            None, None, None,
        )
        # The same cutoff once the session has closed scores normally.
        monkeypatch.setattr(acc, "_us_equity_today", lambda: date(2026, 1, 13))
        raw, _alpha, days = acc.fetch_returns_yf("AAPL", "2026-01-05", holding_days=5,
                                                 as_of_date="2026-01-12")
        assert (raw, days) == (pytest.approx(0.05), 5)

    def test_r17_fetch_returns_crypto_leg_is_priced_on_its_own_venue(self, vendor, monkeypatch):
        asked = vendor({"SPY": [500.0 + i for i in range(12)]})
        seen: dict[str, object] = {}

        def klines(ticker, start, end, interval, venue, closed_as_of):
            seen.update(ticker=ticker, start=start, end=end, venue=venue, closed_as_of=closed_as_of)
            return _closes_frame([100.0 + i for i in range(12)], _sessions("2026-01-05", 12))

        monkeypatch.setattr(bnc, "binance_klines_frame", klines)
        monkeypatch.setattr(bnc, "_now_ms", lambda: _FAR_NOW_MS)
        raw, alpha, days = acc.fetch_returns_yf(
            "BTCUSDT", "2026-01-05", holding_days=5, as_of_date="2026-01-20",
            asset_type="crypto_perp",
        )
        assert (raw, alpha, days) == (pytest.approx(0.05), pytest.approx(0.05 - 0.01), 5)
        assert seen["venue"] == "binance_perp"  # never the Yahoo spot pair
        assert seen["closed_as_of"] == _ms_of(date(2026, 1, 21)) - 1
        assert asked == [("SPY", "2026-01-05", "2026-01-17")]  # the benchmark stays on yfinance

    def test_r17_fetch_returns_a_vendor_error_is_pending_not_a_crash(self, monkeypatch, caplog):
        def down(*args, **kwargs):
            raise RuntimeError("rate limited")

        monkeypatch.setattr(yfmod, "get_YFin_history_cached", down)
        with caplog.at_level(logging.WARNING, logger="yialpha.accuracy"):
            assert acc.fetch_returns_yf("AAPL", "2026-01-05") == (None, None, None)
        assert "Could not resolve outcome for AAPL on 2026-01-05 vs SPY: rate limited" in caplog.text


# ===========================================================================
# batch.runner — analyst selection and the run lock
# ===========================================================================
_BASE_ANALYSTS = ("market", "social", "news", "fundamentals")


def _batch_config(tmp_path: Path, **extra) -> dict:
    """A runner config whose lock roots live in the per-test tmp dir."""
    config = {"data_cache_dir": str(tmp_path / "cache"), "results_dir": str(tmp_path / "results")}
    config.update(extra)
    return config


def _lock_files(tmp_path: Path) -> list[list[str]]:
    """Lock file names present under the two configured roots, cache then results."""
    return [
        sorted(p.name for p in (tmp_path / root / ".run_locks").iterdir())
        for root in ("cache", "results")
    ]


class TestBatchSelectedAnalysts:
    def test_r17_batch_analysts_unknown_asset_type_or_analyst_keeps_the_base(self):
        assert runner.batch_selected_analysts("futures", ["AAPL"], _BASE_ANALYSTS) == _BASE_ANALYSTS
        assert runner.batch_selected_analysts("Stock", ["AAPL"], _BASE_ANALYSTS) == _BASE_ANALYSTS
        assert runner.batch_selected_analysts("stock", ["AAPL"], ("market", "wizard")) == (
            "market", "wizard",
        )

    def test_r17_batch_analysts_stock_batch_keeps_every_analyst(self):
        assert runner.batch_selected_analysts("stock", ["AAPL", "MSFT"], _BASE_ANALYSTS) == (
            _BASE_ANALYSTS
        )

    def test_r17_batch_analysts_perp_batch_drops_fundamentals_unless_a_stock_perp_is_in_it(
        self, monkeypatch
    ):
        monkeypatch.setattr(bnc, "warm_equity_perp_bases", lambda: None)  # static seed only
        assert runner.batch_selected_analysts(
            "crypto_perp", ["BTCUSDT", "ETHUSDT"], _BASE_ANALYSTS,
        ) == ("market", "social", "news")
        # A tokenized-stock perp reads the underlying's fundamentals: the
        # batch stays over-inclusive rather than starving that one ticker.
        assert runner.batch_selected_analysts(
            "crypto_perp", ["BTCUSDT", "MUUSDT"], _BASE_ANALYSTS,
        ) == _BASE_ANALYSTS


class TestSerializedRun:
    def test_r17_serialized_run_refuses_a_config_without_a_lock_root(self, tmp_path):
        for config, missing in (
            ({"results_dir": str(tmp_path)}, "data_cache_dir"),
            ({"data_cache_dir": str(tmp_path)}, "results_dir"),
            ({"data_cache_dir": "", "results_dir": str(tmp_path)}, "data_cache_dir"),
        ):
            with (
                pytest.raises(ValueError, match=f"{missing} is required for batch run locking"),
                runner.serialized_run(config, "AAPL", "2026-01-05", "stock"),
            ):
                pytest.fail("the body must not run without a lock root")

    def test_r17_serialized_run_lock_key_is_the_ticker_and_date_only(self, tmp_path):
        config = _batch_config(tmp_path)
        held: dict[str, list[list[str]]] = {}
        with runner.serialized_run(config, " brk.b ", "2026-01-05", "stock"):
            held["stock"] = _lock_files(tmp_path)
        # The same ticker and date under another asset mode: the same key.
        # Unlike asset modes still serialize on the shared report/checkpoint paths.
        with runner.serialized_run(config, "BRK.B", "2026-01-05", "crypto_perp"):
            held["perp"] = _lock_files(tmp_path)
        key = json.dumps(["BRK.B", "2026-01-05"], ensure_ascii=True, separators=(",", ":"))
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        name = f"BRK.B-{digest}.lock"
        assert held == {"stock": [[name], [name]], "perp": [[name], [name]]}

    def test_r17_serialized_run_rejects_a_ticker_that_cannot_be_a_file_name(self, tmp_path):
        with (
            pytest.raises(ValueError),
            runner.serialized_run(_batch_config(tmp_path), "../../escape", "2026-01-05", "stock"),
        ):
            pytest.fail("an unsafe ticker must not reach the filesystem")

    def test_r17_serialized_run_releases_both_locks_when_the_body_raises(self, tmp_path):
        config = _batch_config(tmp_path)
        with (
            pytest.raises(RuntimeError, match="boom"),
            runner.serialized_run(config, "AAPL", "2026-01-05", "stock"),
        ):
            raise RuntimeError("boom")
        # The next thread -- of any asset mode, which is no part of the key --
        # gets in without waiting on anything the failed run left behind.
        entered = threading.Event()

        def next_run() -> None:
            with runner.serialized_run(config, "AAPL", "2026-01-05", "crypto_perp"):
                entered.set()

        waiter = threading.Thread(target=next_run, daemon=True)
        waiter.start()
        assert entered.wait(5.0)
        waiter.join(5.0)
        assert not waiter.is_alive()


# ===========================================================================
# batch.locks.shared_file_lock / FileLock
# ===========================================================================
class TestSharedFileLock:
    def test_r17_shared_file_lock_disabled_is_a_noop_that_creates_nothing(self, tmp_path):
        path = tmp_path / "memory.lock"
        with locks.shared_file_lock(path, enabled=False):
            assert list(tmp_path.iterdir()) == []
        guard = locks.shared_file_lock(path, enabled=True)
        assert isinstance(guard, locks.FileLock)
        with guard:
            pass

    def test_r17_shared_file_lock_passes_to_the_next_thread_after_a_crash(self, tmp_path):
        path = tmp_path / "run.lock"
        held, release, taken = threading.Event(), threading.Event(), threading.Event()

        def crashing_holder() -> None:
            try:
                with locks.shared_file_lock(path):
                    held.set()
                    release.wait(5.0)
                    raise RuntimeError("holder crashed")
            except RuntimeError:
                pass

        def next_holder() -> None:
            with locks.shared_file_lock(path):
                taken.set()

        first = threading.Thread(target=crashing_holder, daemon=True)
        second = threading.Thread(target=next_holder, daemon=True)
        first.start()
        assert held.wait(5.0)
        second.start()
        assert not taken.wait(0.3)  # held: the second thread has to wait
        release.set()
        first.join(5.0)
        assert taken.wait(5.0)  # ... and gets in as soon as the first one is gone
        second.join(5.0)
        assert not first.is_alive() and not second.is_alive()


# ===========================================================================
# batch.runner.BatchRunner — construction and the concurrent loop
# ===========================================================================
class _FakeGraph:
    def __init__(self, config: dict) -> None:
        self.config = config

    def propagate(self, ticker: str, trade_date: str, asset_type: str = "stock"):
        return {"company_of_interest": ticker}, f"signal:{ticker}"

    def save_reports(self, final_state: dict, ticker: str):
        return None


class _CrashGraph(_FakeGraph):
    def propagate(self, ticker: str, trade_date: str, asset_type: str = "stock"):
        if ticker == "BOOM":
            raise RuntimeError("LLM stream died mid-response")
        return super().propagate(ticker, trade_date, asset_type=asset_type)


def _row(ticker: str, error: BaseException | None = None) -> dict:
    ok = error is None
    return {
        "ticker": ticker,
        "state": {"company_of_interest": ticker} if ok else None,
        "signal": f"signal:{ticker}" if ok else None,
        "report_path": None,
        "elapsed": 0.0,
        "error": error,
    }


class TestBatchRunnerConstruction:
    def test_r17_default_graph_factory_builds_the_real_graph_class_without_debug(self, monkeypatch, tmp_path):
        built: list[tuple] = []

        class _Graph:
            def __init__(self, analysts, debug, config):
                built.append((analysts, debug, config))
                self.config = config

        monkeypatch.setattr(runner, "YiAlphaGraph", _Graph)
        config = _batch_config(tmp_path)
        with BatchRunner(config, selected_analysts=("market", "news"), workers=1,
                         progress=False) as runner_instance:
            assert runner_instance._graph_factory.__name__ == "_default_graph_factory"
        assert built == [(("market", "news"), False, config)]

    @pytest.mark.parametrize("workers", [0, -5])
    def test_r17_batch_runner_a_non_positive_worker_count_runs_serially(self, tmp_path, workers):
        config = _batch_config(tmp_path, batch_concurrency=True)
        with BatchRunner(config, graph_factory=_FakeGraph, workers=workers, progress=False) as run:
            assert run.workers == 1
            rows = run.run(["AAPL", "MSFT"], "2026-01-05")
        assert [r["signal"] for r in rows] == ["signal:AAPL", "signal:MSFT"]


class TestRunConcurrentLoop:
    def test_r17_concurrent_fail_fast_cancels_what_never_started_and_collects_the_rest(
        self, tmp_path, monkeypatch
    ):
        """The loop's bookkeeping, driven with canned futures so the three
        states a future can be in when the batch gives up are exact:
        a finished failure, work a thread already owns (it cannot be
        cancelled; its result -- or its late exception -- is still
        collected), and work that never started (cancelled, and recorded)."""
        boom, running, late_crash, never_started = Future(), Future(), Future(), Future()
        boom.set_result(_row("BOOM", error=RuntimeError("LLM stream died mid-response")))
        running.set_running_or_notify_cancel()
        late_crash.set_running_or_notify_cancel()
        by_ticker = {"BOOM": boom, "RUNNING": running, "LATE_CRASH": late_crash,
                     "NEVER": never_started}
        monkeypatch.setattr(runner, "submit_with_context",
                            lambda executor, fn, ticker, *args: by_ticker[ticker])
        timers = [
            threading.Timer(0.3, running.set_result, args=(_row("RUNNING"),)),
            threading.Timer(0.3, late_crash.set_exception, args=(RuntimeError("late crash"),)),
        ]
        for timer in timers:
            timer.start()
        try:
            config = _batch_config(tmp_path, batch_concurrency=True, batch_fail_fast=True)
            with BatchRunner(config, graph_factory=_FakeGraph, workers=2, progress=True) as run:
                rows = run.run(["BOOM", "RUNNING", "LATE_CRASH", "NEVER"], "2026-01-05")
        finally:
            for timer in timers:
                timer.cancel()
        assert [r["ticker"] for r in rows] == ["BOOM", "RUNNING", "LATE_CRASH", "NEVER"]
        assert type(rows[0]["error"]) is RuntimeError
        assert str(rows[0]["error"]) == "LLM stream died mid-response"
        assert rows[1] == _row("RUNNING")
        assert type(rows[2]["error"]) is RuntimeError and str(rows[2]["error"]) == "late crash"
        assert rows[2]["state"] is None and rows[2]["elapsed"] == 0.0
        assert type(rows[3]["error"]) is RuntimeError
        assert str(rows[3]["error"]) == "cancelled by batch_fail_fast after BOOM failed"

    def test_r17_concurrent_a_broken_pool_is_recorded_not_raised(self, tmp_path, monkeypatch):
        broken = Future()
        broken.set_exception(BrokenThreadPool())
        monkeypatch.setattr(runner, "submit_with_context",
                            lambda executor, fn, ticker, *args: broken)
        config = _batch_config(tmp_path, batch_concurrency=True)
        with BatchRunner(config, graph_factory=_FakeGraph, workers=2, progress=False) as run:
            rows = run.run(["AAPL"], "2026-01-05")
        assert len(rows) == 1
        assert type(rows[0]["error"]) is BrokenThreadPool
        assert (rows[0]["ticker"], rows[0]["state"], rows[0]["elapsed"]) == ("AAPL", None, 0.0)

    def test_r17_concurrent_fail_fast_with_real_threads_still_returns_every_ticker(self, tmp_path):
        config = _batch_config(tmp_path, batch_concurrency=True, batch_fail_fast=True)
        with BatchRunner(config, graph_factory=_CrashGraph, workers=2, progress=False) as run:
            rows = run.run(["BOOM", "NVDA"], "2026-01-05")
        assert [r["ticker"] for r in rows] == ["BOOM", "NVDA"]
        assert isinstance(rows[0]["error"], RuntimeError)
        assert rows[0]["signal"] is None
        # The ticker already in flight when the batch gave up is still collected.
        assert rows[1]["error"] is None and rows[1]["signal"] == "signal:NVDA"

    def test_r17_concurrent_duplicates_each_get_their_own_row(self, tmp_path):
        """Results are keyed by input position, never by ticker: with the
        dedup off, two runs of the same ticker come back as two rows."""
        config = _batch_config(tmp_path, batch_concurrency=True, batch_dedup_tickers=False)
        with BatchRunner(config, graph_factory=_FakeGraph, workers=2, progress=False) as run:
            rows = run.run(["AAPL", "AAPL", "MSFT"], "2026-01-05")
        assert [(r["ticker"], r["signal"]) for r in rows] == [
            ("AAPL", "signal:AAPL"), ("AAPL", "signal:AAPL"), ("MSFT", "signal:MSFT"),
        ]
        assert all(r["error"] is None for r in rows)

    def test_r17_concurrent_without_fail_fast_one_failure_does_not_stop_the_batch(self, tmp_path):
        config = _batch_config(tmp_path, batch_concurrency=True)
        with BatchRunner(config, graph_factory=_CrashGraph, workers=2, progress=True) as run:
            rows = run.run(["BOOM", "NVDA", "MSFT", "AAPL"], "2026-01-05")
        assert [r["ticker"] for r in rows] == ["BOOM", "NVDA", "MSFT", "AAPL"]
        assert isinstance(rows[0]["error"], RuntimeError)
        assert [r["signal"] for r in rows[1:]] == ["signal:NVDA", "signal:MSFT", "signal:AAPL"]
