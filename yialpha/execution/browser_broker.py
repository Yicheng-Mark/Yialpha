"""Browser broker execution (Phase 4 fallback path).

When a broker has no first-class API, this module drives the broker's Web
order-entry UI via Playwright with Microsoft Edge. It is intentionally
**fail-closed and dry-run by default**, and gated behind an analysis-only
boundary, two explicit enable switches, and an environment kill switch. The
preferred execution path is always a real broker API; this is the
fallback for the brokers that only expose a Web UI.

Design invariants (every code path must preserve these):

* **Lazy/optional Playwright.** Playwright is imported *inside* methods, never
  at module top level. Importing this module must succeed even when Playwright
  is not installed. A missing Playwright never raises at import time; it turns
  into a ``BLOCKED_PLAYWRIGHT`` / ``DRY_RUN_PREVIEW`` result instead.
* **Fail-closed.** Any uncertainty — kill switch on, validator unavailable or
  rejecting, page anomaly, dry run, missing Playwright, missing broker URL,
  subclass not configured — yields a non-submitted result. The *only* path to
  ``submitted=True`` is: ``dry_run`` explicitly ``False`` AND all three live
  policy values are explicitly armed AND the kill switch is off AND validation
  passed AND Playwright imported AND an order-page URL is configured AND
  ``_fill_order_form`` succeeded AND the submit click returned without raising.
* **Standalone.** Safety switches are read straight from ``os.environ`` on
  every attempt; this module deliberately
  does *not* couple to the project's global config system so it can be reasoned
  about and tested in isolation.
* **Broker-specific DOM is subclassed.** The base class cannot, by design,
  place a live order: ``_fill_order_form`` raises ``NotImplementedError`` and
  ``_click_submit`` is a no-op stub. A concrete broker subclasses and overrides
  both. Until that happens every "live" attempt fails closed.
"""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

# Truthy spellings the kill switch (and other boolean env knobs) accept.
# Comparison is case-insensitive. Anything outside this set raises — we never
# silently treat a typo as "off".
_TRUTHY = frozenset({"true", "1", "yes", "on"})
_FALSY = frozenset({"false", "0", "no", "off"})


def _coerce_bool_env(value: str | None) -> bool:
    """Coerce a raw environment-string value to a strict bool.

    Accepts ``"true"/"1"/"yes"/"on"`` (case-insensitive) as ``True`` and
    ``"false"/"0"/"no"/"off"`` (plus ``None`` and ``""``) as ``False``.
    Any other value raises ``ValueError`` rather than guessing — a mistyped
    kill switch must be loud, not silently "off".

    ``None`` and the empty string mean "unset / blank" and map to ``False``.
    """
    if value is None:
        return False
    text = value.strip().lower()
    if text == "":
        return False
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    raise ValueError(
        f"Unrecognized boolean environment value {value!r}; "
        f"expected one of true/1/yes/on or false/0/no/off (case-insensitive)."
    )


class KillSwitch:
    """Reads ``YIALPHA_KILL_SWITCH`` directly from the environment.

    Kept as a thin static class (not an instance) on purpose: it is a global
    safety control and reading it should not require plumbing state through
    the call graph. Reading via ``os.environ.get`` keeps this module decoupled
    from the project config system.
    """

    _ENV_VAR = "YIALPHA_KILL_SWITCH"

    @staticmethod
    def is_halted() -> bool:
        """True iff the kill switch is set to a truthy value (trading stopped)."""
        raw = os.environ.get(KillSwitch._ENV_VAR)
        try:
            return _coerce_bool_env(raw)
        except ValueError:
            # A malformed value is treated as *halted*: never let a typo
            # silently re-enable trading. The reason() call surfaces the value.
            return True

    @staticmethod
    def reason() -> str:
        """Human-readable explanation of the current kill-switch state."""
        raw = os.environ.get(KillSwitch._ENV_VAR)
        if raw is None or raw.strip() == "":
            return f"{KillSwitch._ENV_VAR} is unset (trading allowed)."
        try:
            halted = _coerce_bool_env(raw)
        except ValueError:
            return (
                f"{KillSwitch._ENV_VAR}={raw!r} is not a recognized boolean; "
                "treated as halted (fail-closed)."
            )
        if halted:
            return f"{KillSwitch._ENV_VAR}={raw!r} -> trading HALTED."
        return f"{KillSwitch._ENV_VAR}={raw!r} -> trading allowed."


class LiveExecutionSwitch:
    """Global, dynamically-read opt-in policy for any live order path.

    YiAlpha is analysis-only by default.  Arming live execution therefore
    requires three explicit environment settings at the same time:

    * ``YIALPHA_ANALYSIS_ONLY=false``
    * ``YIALPHA_LIVE_EXECUTION_ENABLED=true``
    * ``YIALPHA_EXECUTION_ENABLED=true`` (legacy compatibility switch)

    Requiring both enable switches keeps the existing public knob available
    while preventing a stale legacy environment variable from silently
    defeating the new analysis-only boundary.  Values are read on every call,
    so disabling any switch after connecting halts subsequent submissions.
    Missing or malformed values always fail closed.
    """

    _ANALYSIS_ONLY_ENV = "YIALPHA_ANALYSIS_ONLY"
    _LIVE_ENABLED_ENV = "YIALPHA_LIVE_EXECUTION_ENABLED"
    _LEGACY_ENABLED_ENV = "YIALPHA_EXECUTION_ENABLED"

    @staticmethod
    def _read(name: str, *, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            return default
        try:
            return _coerce_bool_env(raw)
        except ValueError:
            # For analysis_only, fail-closed means staying in analysis-only
            # mode.  For an enable flag it means staying disabled.
            return default

    @classmethod
    def is_enabled(cls) -> bool:
        analysis_only = cls._read(cls._ANALYSIS_ONLY_ENV, default=True)
        live_enabled = cls._read(cls._LIVE_ENABLED_ENV, default=False)
        legacy_enabled = cls._read(cls._LEGACY_ENABLED_ENV, default=False)
        return not analysis_only and live_enabled and legacy_enabled

    @classmethod
    def reason(cls) -> str:
        if cls._read(cls._ANALYSIS_ONLY_ENV, default=True):
            return (
                f"{cls._ANALYSIS_ONLY_ENV} is enabled or invalid/unset; "
                "the project is analysis-only."
            )
        if not cls._read(cls._LIVE_ENABLED_ENV, default=False):
            return f"{cls._LIVE_ENABLED_ENV} is disabled, invalid, or unset."
        if not cls._read(cls._LEGACY_ENABLED_ENV, default=False):
            return f"{cls._LEGACY_ENABLED_ENV} is disabled, invalid, or unset."
        return "All live-execution opt-in switches are enabled."


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class OrderAction(StrEnum):
    """Direction of an order. ``str`` mixin so values serialize cleanly."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(StrEnum):
    """Outcome of a :meth:`BrowserBroker.place_order` call.

    Only ``SUBMITTED`` implies a live order reached the broker. Everything
    else is a non-submitted outcome (preview, blocked, or error).
    """

    DRY_RUN_PREVIEW = "dry_run_preview"  # stopped at preview, never submitted
    SUBMITTED = "submitted"  # actually placed (only when dry_run=False + all gates pass)
    BLOCKED_EXECUTION_POLICY = "blocked_execution_policy"
    BLOCKED_KILL_SWITCH = "blocked_kill_switch"
    BLOCKED_VALIDATION = "blocked_validation"
    BLOCKED_PLAYWRIGHT = "blocked_playwright"  # playwright missing / page error
    ERROR = "error"


@dataclass
class OrderResult:
    """Immutable-ish outcome of an order attempt.

    ``submitted`` is a redundant hard boolean so callers cannot accidentally
    interpret a blocked/error status as actionable. It is ``True`` *only* when
    ``status == OrderStatus.SUBMITTED``.
    """

    status: OrderStatus
    ticker: str
    action: OrderAction
    size: float
    message: str
    submitted: bool = False
    preview_url: str | None = None


# Type alias for the optional pre-submit validator hook.
PreSubmitValidator = Callable[[str, OrderAction, float], bool | None]


# ---------------------------------------------------------------------------
# Broker driver
# ---------------------------------------------------------------------------


class BrowserBroker:
    """Drive a broker's Web order-entry UI via Playwright/Edge.

    The base class is fail-closed and broker-agnostic: it cannot place a live
    order until a concrete broker subclasses it and overrides
    :meth:`_fill_order_form` and :meth:`_click_submit`. Until then any
    ``dry_run=False`` attempt fails closed with ``BLOCKED_PLAYWRIGHT`` or
    ``ERROR`` — never a live submission slipping through.
    """

    def __init__(
        self,
        channel: str = "msedge",
        headless: bool = False,
        max_order_pct_of_equity: float = 0.01,
        nav_timeout_ms: int = 30000,
        order_page_url: str | None = None,
        dry_run_default: bool = True,
    ) -> None:
        # Broker UIs typically need a visible browser (anti-bot heuristics,
        # 2FA, etc.), so headless defaults to False.
        self.channel = channel
        self.headless = headless
        # Hard cap: a single order may not exceed this fraction of net equity.
        # A small-start guardrail so a noisy PM decision cannot move the whole
        # account in one click.
        self.max_order_pct_of_equity = float(max_order_pct_of_equity)
        self.nav_timeout_ms = int(nav_timeout_ms)
        self.order_page_url = order_page_url
        self.dry_run_default = bool(dry_run_default)

    # -- Playwright bootstrap -------------------------------------------------

    def _get_playwright(self):
        """Lazily import Playwright's sync entrypoint.

        Returns the ``sync_playwright`` callable on success, or ``None`` if
        Playwright is not importable. Importing lazily (inside this method)
        keeps the module importable without Playwright installed.
        """
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except ImportError:
            return None
        return sync_playwright

    # -- Broker-specific hooks (must be subclassed) ---------------------------

    def _fill_order_form(self, page: Any, ticker: str, action: OrderAction, size: float) -> None:
        """Fill the broker's order-entry DOM.

        The base class is intentionally broker-agnostic and **cannot** place a
        live order. Subclasses override this to:

        * locate the symbol field and type ``ticker``,
        * select buy/sell per ``action``,
        * enter quantity ``size``,
        * trigger the broker's "preview" so a confirmation screen appears.

        The default implementation raises :class:`NotImplementedError` so a
        misconfigured live attempt fails loudly and closed rather than clicking
        random DOM. Concrete brokers MUST override this for real use.
        """
        raise NotImplementedError(
            "BrowserBroker._fill_order_form is broker-specific and not "
            "configured. Subclass BrowserBroker and override _fill_order_form "
            "(and _click_submit) to drive your broker's order-entry page. "
            "Until then no live order can be placed."
        )

    def _click_submit(self, page: Any) -> None:
        """Click the broker's final submit/confirm button.

        Default implementation raises :class:`NotImplementedError`; concrete
        brokers override to actually place the order. Called only on the
        ``dry_run=False`` path after :meth:`_fill_order_form` succeeded.
        """
        raise NotImplementedError(
            "BrowserBroker._click_submit is broker-specific and not "
            "configured. Subclass and override to submit on your broker."
        )

    # -- Main entrypoint ------------------------------------------------------

    def place_order(
        self,
        ticker: str,
        action: OrderAction | str,
        size: float,
        equity_value: float | None = None,
        dry_run: bool | None = None,
        pre_submit_validator: PreSubmitValidator | None = None,
        reference_price: float | None = None,
    ) -> OrderResult:
        """Place (or preview) an order via the broker Web UI.

        Gates are evaluated **in order**, fail-closed:

        1. :class:`KillSwitch` halted -> ``BLOCKED_KILL_SWITCH``.
        2. ``dry_run`` resolution: ``dry_run_default`` if ``dry_run is None``.
        3. A non-dry-run request requires :class:`LiveExecutionSwitch` to be
           fully armed. Dry-run analysis/preview remains available.
        4. Size sanity: ``size`` must be ``> 0``. If ``equity_value`` is
           given, a ``reference_price`` must also be given, and the order's
           NOTIONAL exposure ``size * reference_price / equity_value`` must be
           ``<= max_order_pct_of_equity``. (``size`` is shares, ``equity`` is
           dollars — the former ``size / equity`` heuristic scaled with price
           and let a 50%-notional order pass a 1% cap. Fail-closed: without a
           price the notional cap cannot be verified, so the order blocks.)
        5. ``pre_submit_validator(ticker, action, size)`` if provided must
           return ``True`` or ``None`` (ok); ``False`` or raising ->
           ``BLOCKED_VALIDATION``.
        6. Playwright importable (lazy). If not -> ``BLOCKED_PLAYWRIGHT`` (no
           live attempt).
        7. ``order_page_url`` set? If ``None`` -> ``BLOCKED_PLAYWRIGHT``, unless
           ``dry_run`` in which case ``DRY_RUN_PREVIEW`` noting no URL is
           configured (still never submit).
        8. Launch Edge, navigate, fill via :meth:`_fill_order_form`, capture
           preview. If ``dry_run`` -> stop, return ``DRY_RUN_PREVIEW``.
           Else -> :meth:`_click_submit` and return ``SUBMITTED``.

        Every browser interaction is wrapped so any exception becomes
        ``ERROR`` or ``BLOCKED_PLAYWRIGHT`` — a live submission can never slip
        through an exception path.
        """
        # Normalize the action to the enum up front so failures are typed.
        try:
            action_enum = self._coerce_action(action)
        except ValueError as exc:
            return OrderResult(
                status=OrderStatus.BLOCKED_VALIDATION,
                ticker=str(ticker),
                action=OrderAction.BUY,  # placeholder; rejected before any use
                size=float(size),
                message=f"Invalid action {action!r}: {exc}",
                submitted=False,
            )

        # Gate 1: kill switch.
        if KillSwitch.is_halted():
            return OrderResult(
                status=OrderStatus.BLOCKED_KILL_SWITCH,
                ticker=ticker,
                action=action_enum,
                size=size,
                message=f"Order blocked: kill switch engaged. {KillSwitch.reason()}",
                submitted=False,
            )

        # Gate 2: resolve dry_run.
        is_dry_run = self.dry_run_default if dry_run is None else bool(dry_run)

        # Gate 3: dry-run previews are analysis output; only a real submit needs
        # the explicit live-execution policy. Read dynamically so an operator
        # can stop submissions after a browser session has already opened.
        if not is_dry_run and not LiveExecutionSwitch.is_enabled():
            return OrderResult(
                status=OrderStatus.BLOCKED_EXECUTION_POLICY,
                ticker=ticker,
                action=action_enum,
                size=size,
                message=(
                    "Order blocked by analysis-only/live-execution policy. "
                    f"{LiveExecutionSwitch.reason()}"
                ),
                submitted=False,
            )

        # Gate 4: size / equity sanity.
        try:
            size_f = float(size)
        except (TypeError, ValueError):
            return OrderResult(
                status=OrderStatus.BLOCKED_VALIDATION,
                ticker=ticker,
                action=action_enum,
                size=size,
                message=f"Order blocked: size {size!r} is not a number.",
                submitted=False,
            )
        if not (size_f > 0.0):
            return OrderResult(
                status=OrderStatus.BLOCKED_VALIDATION,
                ticker=ticker,
                action=action_enum,
                size=size_f,
                message=f"Order blocked: size must be > 0 (got {size_f}).",
                submitted=False,
            )
        if equity_value is not None:
            try:
                equity_f = float(equity_value)
            except (TypeError, ValueError):
                return OrderResult(
                    status=OrderStatus.BLOCKED_VALIDATION,
                    ticker=ticker,
                    action=action_enum,
                    size=size_f,
                    message=f"Order blocked: equity_value {equity_value!r} is not a number.",
                    submitted=False,
                )
            if equity_f <= 0.0:
                return OrderResult(
                    status=OrderStatus.BLOCKED_VALIDATION,
                    ticker=ticker,
                    action=action_enum,
                    size=size_f,
                    message=f"Order blocked: equity_value must be > 0 (got {equity_f}).",
                    submitted=False,
                )
            # Notional-exposure cap. ``size`` is SHARES and ``equity_value``
            # is DOLLARS; a shares/equity ratio is not a fraction of equity
            # (100 shares of a $500 stock on $100k equity is a 50% notional
            # position, not 0.1%). Convert with ``reference_price`` and cap
            # the NOTIONAL. Fail-closed: without a usable price the cap
            # cannot be verified, so the order is rejected — consistent with
            # this class's invariant that uncertainty never submits. Callers
            # that do not want the cap should omit ``equity_value`` entirely.
            if reference_price is None:
                return OrderResult(
                    status=OrderStatus.BLOCKED_VALIDATION,
                    ticker=ticker,
                    action=action_enum,
                    size=size_f,
                    message=(
                        "Order blocked: equity_value given without reference_price; "
                        "notional exposure (size * price / equity) cannot be "
                        "verified. Pass a positive finite reference_price."
                    ),
                    submitted=False,
                )
            try:
                price_f = float(reference_price)
            except (TypeError, ValueError):
                return OrderResult(
                    status=OrderStatus.BLOCKED_VALIDATION,
                    ticker=ticker,
                    action=action_enum,
                    size=size_f,
                    message=(
                        f"Order blocked: reference_price {reference_price!r} "
                        "is not a number."
                    ),
                    submitted=False,
                )
            if not (price_f > 0.0) or not math.isfinite(price_f):
                return OrderResult(
                    status=OrderStatus.BLOCKED_VALIDATION,
                    ticker=ticker,
                    action=action_enum,
                    size=size_f,
                    message=(
                        f"Order blocked: reference_price must be a positive "
                        f"finite number (got {price_f!r})."
                    ),
                    submitted=False,
                )
            notional = size_f * price_f
            pct = notional / equity_f
            if pct > self.max_order_pct_of_equity:
                return OrderResult(
                    status=OrderStatus.BLOCKED_VALIDATION,
                    ticker=ticker,
                    action=action_enum,
                    size=size_f,
                    message=(
                        f"Order blocked: size {size_f} @ {price_f} is a notional "
                        f"{notional:,.0f} = {pct:.2%} of equity {equity_f}, "
                        f"exceeding the {self.max_order_pct_of_equity:.2%} "
                        "single-order cap."
                    ),
                    submitted=False,
                )

        # Gate 4: optional pre-submit validator.
        if pre_submit_validator is not None:
            try:
                verdict = pre_submit_validator(ticker, action_enum, size_f)
            except Exception as exc:  # noqa: BLE001 - any validator failure is a block
                return OrderResult(
                    status=OrderStatus.BLOCKED_VALIDATION,
                    ticker=ticker,
                    action=action_enum,
                    size=size_f,
                    message=f"Order blocked: pre_submit_validator raised: {exc}",
                    submitted=False,
                )
            if verdict is False:
                return OrderResult(
                    status=OrderStatus.BLOCKED_VALIDATION,
                    ticker=ticker,
                    action=action_enum,
                    size=size_f,
                    message="Order blocked: pre_submit_validator returned False.",
                    submitted=False,
                )

        # Gate 5: Playwright must be importable for anything beyond a pure
        # dry-run preview. For dry-run we can short-circuit before needing it.
        if is_dry_run:
            # Dry run never touches the browser if we have nowhere to go, and
            # even if we do, we stop at the preview. We still try to launch the
            # browser to produce a real preview URL when possible; if Playwright
            # is missing we report a preview without a live URL.
            return self._do_dry_run(ticker, action_enum, size_f)

        # Live path begins here. From this point, dry_run is False.
        sync_playwright = self._get_playwright()
        if sync_playwright is None:
            return OrderResult(
                status=OrderStatus.BLOCKED_PLAYWRIGHT,
                ticker=ticker,
                action=action_enum,
                size=size_f,
                message="Order blocked: Playwright is not installed (dry_run=False requires it).",
                submitted=False,
            )

        # Gate 6: must know where to navigate for a live order.
        if not self.order_page_url:
            return OrderResult(
                status=OrderStatus.BLOCKED_PLAYWRIGHT,
                ticker=ticker,
                action=action_enum,
                size=size_f,
                message=(
                    "Order blocked: order_page_url is not configured "
                    "(cannot navigate to broker for a live order)."
                ),
                submitted=False,
            )

        # Gate 7: launch Edge, fill, submit. Any exception -> ERROR, never a
        # submission leaking through.
        try:
            return self._do_live_order(
                sync_playwright, ticker, action_enum, size_f
            )
        except NotImplementedError as exc:
            # Subclass not configured — fail closed, no submission possible.
            return OrderResult(
                status=OrderStatus.BLOCKED_PLAYWRIGHT,
                ticker=ticker,
                action=action_enum,
                size=size_f,
                message=f"Order blocked: broker DOM not configured. {exc}",
                submitted=False,
            )
        except Exception as exc:  # noqa: BLE001 - any browser error is non-fatal to the caller
            return OrderResult(
                status=OrderStatus.ERROR,
                ticker=ticker,
                action=action_enum,
                size=size_f,
                message=f"Order error during browser interaction: {exc}",
                submitted=False,
            )

    # -- Action coercion ------------------------------------------------------

    @staticmethod
    def _coerce_action(action: OrderAction | str) -> OrderAction:
        """Accept either an :class:`OrderAction` or a raw string like ``"buy"``."""
        if isinstance(action, OrderAction):
            return action
        if isinstance(action, str):
            text = action.strip().lower()
            try:
                return OrderAction(text)
            except ValueError as exc:
                raise ValueError(
                    f"{action!r} is not a valid OrderAction "
                    f"(expected one of {[a.value for a in OrderAction]})."
                ) from exc
        raise ValueError(
            f"Action must be an OrderAction or string, got {type(action).__name__}."
        )

    # -- Dry-run path ---------------------------------------------------------

    def _do_dry_run(
        self, ticker: str, action: OrderAction, size: float
    ) -> OrderResult:
        """Execute the dry-run preview path.

        We never submit. If Playwright and a URL are available we *could*
        launch the browser to capture a real preview, but to keep the dry-run
        path side-effect-free and unit-testable without a browser we do not
        actually launch here. We simply report what *would* happen.
        """
        sync_playwright = self._get_playwright()
        if not self.order_page_url:
            return OrderResult(
                status=OrderStatus.DRY_RUN_PREVIEW,
                ticker=ticker,
                action=action,
                size=size,
                message=(
                    "Dry-run preview: no live broker URL is configured "
                    "(order_page_url is None). Would have attempted a "
                    f"{action.value} of {size} {ticker}. NOT submitted."
                ),
                submitted=False,
                preview_url=None,
            )
        if sync_playwright is None:
            return OrderResult(
                status=OrderStatus.DRY_RUN_PREVIEW,
                ticker=ticker,
                action=action,
                size=size,
                message=(
                    "Dry-run preview: Playwright not installed, so no browser "
                    f"preview captured. Would have {action.value}ed {size} {ticker} "
                    f"at {self.order_page_url}. NOT submitted."
                ),
                submitted=False,
                preview_url=self.order_page_url,
            )
        # Playwright + URL both available: report the target URL as the preview.
        # (We intentionally do NOT launch the browser on the dry-run path to
        # keep dry runs free of side effects and usable without a live session.)
        return OrderResult(
            status=OrderStatus.DRY_RUN_PREVIEW,
            ticker=ticker,
            action=action,
            size=size,
            message=(
                f"Dry-run preview: would {action.value} {size} {ticker} at "
                f"{self.order_page_url}. Stopped before submit. NOT submitted."
            ),
            submitted=False,
            preview_url=self.order_page_url,
        )

    # -- Live path ------------------------------------------------------------

    def _final_guard(
        self, ticker: str, action: OrderAction, size: float, preview_url: str | None = None,
    ) -> OrderResult | None:
        """Re-check the kill switch and live-execution policy at the submit edge.

        The entry gates in :meth:`place_order` (Gate 1/3) read the switches
        once, long before the browser is launched. Between then and the final
        submit click, an operator may engage the kill switch or flip the
        analysis-only policy. This guard is invoked immediately before
        :meth:`_click_submit` so a stop invoked during the form-fill window is
        honored — the order is NOT submitted. Returns an ``OrderResult`` (a
        block) when the order must stop, or ``None`` when it may proceed.
        """
        if KillSwitch.is_halted():
            return OrderResult(
                status=OrderStatus.BLOCKED_KILL_SWITCH,
                ticker=ticker,
                action=action,
                size=size,
                message=(
                    "Order blocked at submit edge: kill switch engaged after "
                    f"form fill. {KillSwitch.reason()}"
                ),
                submitted=False,
                preview_url=preview_url,
            )
        if not LiveExecutionSwitch.is_enabled():
            return OrderResult(
                status=OrderStatus.BLOCKED_EXECUTION_POLICY,
                ticker=ticker,
                action=action,
                size=size,
                message=(
                    "Order blocked at submit edge: analysis-only/live-execution "
                    f"policy flipped after form fill. {LiveExecutionSwitch.reason()}"
                ),
                submitted=False,
                preview_url=preview_url,
            )
        return None

    def _do_live_order(
        self,
        sync_playwright: Callable[[], Any],
        ticker: str,
        action: OrderAction,
        size: float,
    ) -> OrderResult:
        """Launch Edge, fill the form, and submit (``dry_run=False`` only).

        Wrapped by :meth:`place_order` so any exception (including the
        ``NotImplementedError`` from an unconfigured subclass) becomes a
        non-submitted result rather than a live order leaking through.
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(
                channel=self.channel, headless=self.headless
            )
            try:
                context = browser.new_context()
                page = context.new_page()
                page.set_default_timeout(self.nav_timeout_ms)
                page.goto(self.order_page_url)

                # Broker-specific DOM filling (subclass override). Raises
                # NotImplementedError in the base — which the caller maps to
                # BLOCKED_PLAYWRIGHT, never a submission.
                self._fill_order_form(page, ticker, action, size)

                # Capture preview state. The concrete broker may expose a
                # confirmation URL; we record it if present, else the current URL.
                try:
                    preview_url = page.url
                except Exception:  # noqa: BLE001 - non-fatal preview metadata
                    preview_url = self.order_page_url

                # Final guard: re-check kill switch + live-execution policy at
                # the submit edge. The entry gates (Gate 1/3) read the switches
                # once before the browser launched; an operator may have engaged
                # the kill switch DURING the form-fill window. Honor it here so
                # the order is NOT submitted. Returns a block result or None.
                block = self._final_guard(ticker, action, size, preview_url)
                if block is not None:
                    return block

                # Final submit. Base class raises NotImplementedError here too;
                # concrete brokers override to click the real confirm button.
                self._click_submit(page)

                return OrderResult(
                    status=OrderStatus.SUBMITTED,
                    ticker=ticker,
                    action=action,
                    size=size,
                    message=(
                        f"Order SUBMITTED via broker UI: {action.value} {size} "
                        f"{ticker} at {self.order_page_url}."
                    ),
                    submitted=True,
                    preview_url=preview_url,
                )
            finally:
                with contextlib.suppress(Exception):  # noqa: BLE001 - cleanup must not mask errors
                    browser.close()
