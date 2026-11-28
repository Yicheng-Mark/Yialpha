"""pytest plugin: scramble the order in which directory entries are listed.

``Path.glob``, ``Path.iterdir``, ``os.scandir``, ``os.listdir`` and ``os.walk``
hand entries back in whatever order the file system uses -- sorted by name on
NTFS, creation order on a freshly written WSL disk, neither on a hosted
runner's ext4.  A test whose outcome depends on that order passes wherever the
order happens to match and fails elsewhere; the first CI run on the round-17
tree failed one such test on both ubuntu lanes while every local lane was
green.  This plugin permutes every listing with a seed taken from the
directory path and ``$SHUFFLE_SEED``, so such tests fail here, on purpose.

Three entry points have to be covered.  Python 3.11 and 3.12 list a directory
for ``Path.glob`` through ``os.scandir``; 3.13 through
``glob._StringGlobber.scandir``, which is bound to the original ``os.scandir``
at import time and would not see a later replacement of ``os.scandir``.
``os.listdir`` is covered for callers that go through the ``os`` module.  The
import system does not, so module import keeps its order.

Usage (this directory on ``PYTHONPATH``)::

    SHUFFLE_SEED=7 PYTHONPATH=tests/_plugins pytest -p shuffle_plugin -q

``SHUFFLE_SEED`` defaults to 1.  CI's weekly ``tests, scrambled directory
order`` job uses the run id as the seed and prints it, so a failure there is
reproduced with that number.
"""
from __future__ import annotations

import contextlib
import os
import random

_real_scandir = os.scandir
_real_listdir = os.listdir
SEED = int(os.environ.get("SHUFFLE_SEED", "1"))


class _Entries:
    """A list dressed as the iterator ``os.scandir`` returns (usable with ``with``)."""

    def __init__(self, entries: list):
        self._iter = iter(entries)

    def __iter__(self) -> _Entries:
        return self

    def __next__(self):
        return next(self._iter)

    def __enter__(self) -> _Entries:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._iter = iter(())


def _rng(path) -> random.Random:
    """The same directory path and seed always give the same permutation."""
    try:
        key = os.fsdecode(path) if isinstance(path, (bytes, bytearray)) else os.fspath(path)
    except TypeError:
        key = str(path)
    return random.Random(f"{SEED}\0{key}")


def _scrambled_scandir(path=".", *args, **kwargs):
    with _real_scandir(path, *args, **kwargs) as it:
        entries = list(it)
    _rng(path).shuffle(entries)
    return _Entries(entries)


def _scrambled_listdir(path="."):
    names = _real_listdir(path)
    _rng(path).shuffle(names)
    return names


os.scandir = _scrambled_scandir
os.listdir = _scrambled_listdir
with contextlib.suppress(Exception):  # an interpreter without that class needs nothing more
    import glob as _glob

    if hasattr(_glob, "_StringGlobber"):
        _glob._StringGlobber.scandir = staticmethod(_scrambled_scandir)
