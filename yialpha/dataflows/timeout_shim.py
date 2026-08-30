"""A default HTTP read timeout for SDKs that expose no timeout parameter.

AKShare and Tushare own their HTTP sessions internally — their public call
signatures accept no ``timeout``, so a hung domestic endpoint (Eastmoney/Sina
socket that never responds) blocks until run_robust's OS-level watchdog kills
the process. This shim patches :meth:`requests.Session.request` for the
duration of a context window and fills in a default timeout **only for calls
that pass none** (``timeout=None``, requests' own default = block forever).

Safety properties:

* Callers that pass an explicit timeout are untouched — every other vendor in
  this project passes one, so concurrent traffic during the patch window is
  unaffected. (yfinance does not go through ``requests`` at all on current
  versions — curl_cffi — so it never sees the patch.)
* The patch window is always exited via ``finally``; nesting is
  reference-counted, and the original method is restored from the moment the
  outermost window closes.
* AKShare/Tushare call sites hold their vendor's serial call lock across the
  window, so two patch windows never interleave with conflicting values from
  this package.

Positional ``timeout`` arguments are not intercepted (requests' own signature
has it ~15 params deep; no caller in the wild passes it positionally).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_state_lock = threading.Lock()
_depth = 0
_original_request = None


def _wrap(original, seconds: float):
    def request_with_default_timeout(self, method, url, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = seconds
        return original(self, method, url, **kwargs)

    return request_with_default_timeout


@contextmanager
def default_request_timeout(seconds: float) -> Iterator[None]:
    """For the duration of this block, requests calls without a timeout get one.

    ``seconds`` must be positive; the value of the OUTERMOST active window
    applies (nested windows do not re-patch).
    """
    global _depth, _original_request

    if seconds <= 0:
        raise ValueError(f"timeout must be positive, got {seconds!r}")

    import requests.sessions

    with _state_lock:
        if _depth == 0:
            _original_request = requests.sessions.Session.request
            requests.sessions.Session.request = _wrap(_original_request, seconds)
        _depth += 1
    try:
        yield
    finally:
        with _state_lock:
            _depth -= 1
            if _depth == 0 and _original_request is not None:
                requests.sessions.Session.request = _original_request
                _original_request = None
