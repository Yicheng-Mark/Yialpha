"""Shared bootstrap for ``scripts/*.py`` entry points (round-11 unification).

Every entry script used to hand-copy the ``sys.path`` project-root insert +
Windows GBK UTF-8 reconfigure header, and the copies had drifted into three
spellings (``_PROJECT_ROOT`` as str or Path; reconfigure as a getattr-guarded
loop / bare call / broad try-except). This module is the single canonical
implementation; entry scripts keep only the minimal inline shim that makes
``_bootstrap`` importable both when run directly (``sys.path[0]`` is
``scripts/``) and under importlib-based test loading, where
``spec_from_file_location`` does NOT put ``scripts/`` on ``sys.path``.
"""

from __future__ import annotations

import sys
from contextlib import suppress
from pathlib import Path


def bootstrap() -> None:
    """Idempotent project-root ``sys.path`` insert + UTF-8 console setup.

    The Windows console defaults to GBK (cp936); printing the ✅/❌ glyphs or
    CJK text raises UnicodeEncodeError mid-run — often AFTER all the work is
    done. Some streams (captured pytest output, replaced stdio) have no
    ``reconfigure``, hence the getattr + callable guard and the targeted
    suppress; ``errors="replace"`` degrades a non-encodable character instead
    of killing the script.
    """
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with suppress(AttributeError, ValueError):
                reconfigure(encoding="utf-8", errors="replace")
