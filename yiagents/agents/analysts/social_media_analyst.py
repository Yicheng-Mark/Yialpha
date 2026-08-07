"""Backwards-compatibility shim for the renamed module.

The agent is now ``sentiment_analyst`` and aggregates Yahoo Finance news,
StockTwits cashtag streams, and Reddit posts into a single sentiment
report. Import from ``yiagents.agents.analysts.sentiment_analyst``
going forward; this module will be removed in a future release.

This shim backs only the legacy *module* import path
(``from ...social_media_analyst import create_social_media_analyst``); the
legacy *function* name is aliased in ``sentiment_analyst`` itself. The two are
not duplicates -- they cover distinct renames (module name vs function name).
The new name ``create_sentiment_analyst`` is intentionally NOT re-exported
here: the new name should come from the new module, not the deprecated one.
"""

import warnings as _warnings

from yiagents.agents.analysts.sentiment_analyst import (  # noqa: F401
    create_social_media_analyst,
)

_warnings.warn(
    "yiagents.agents.analysts.social_media_analyst is deprecated. "
    "Import from yiagents.agents.analysts.sentiment_analyst instead.",
    DeprecationWarning,
    stacklevel=2,
)
