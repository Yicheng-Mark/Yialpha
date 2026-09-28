"""Shared 5-tier rating vocabulary and a deterministic heuristic parser.

The same five-tier scale (Buy, Overweight, Hold, Underweight, Sell) is used by:
- The Research Manager (investment plan recommendation)
- The Portfolio Manager (final position decision)
- The signal processor (rating extracted for downstream consumers)
- The memory log (rating tag stored alongside each decision entry)

Centralising it here avoids drift between those call sites.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Canonical, ordered 5-tier scale (most bullish to most bearish).
RATINGS_5_TIER: tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

# Matches "Rating: X" / "rating - X" / "Rating: **X**" — tolerates markdown
# bold wrappers and either a colon or hyphen separator.
_RATING_LABEL_RE = re.compile(r"rating.*?[:\-][\s*]*(\w+)", re.IGNORECASE)

# Negation words that flip the meaning of a rating word following within a
# small window ("argues against Sell", "counter to Sell", "not Sell").
# Known uncovered shape (review P3-5, accepted): "rather Sell than Buy" is a
# PREFERENCE for Sell, but a backward window cannot tell it apart from the
# negating "rather than Sell" — the heuristic keeps "rather" because the
# negating phrasing dominates PM prose.
_NEGATORS = frozenset({
    "not", "no", "never", "neither", "nor", "against", "counter",
    "counters", "countered", "contrary", "opposed", "versus", "vs",
    "instead", "rather", "despite", "anti",
})


def _negated_before(tokens: list[str], idx: int, window: int = 2) -> bool:
    """True when a negator sits within ``window`` tokens before ``tokens[idx]``."""
    return any(t in _NEGATORS for t in tokens[max(0, idx - window):idx])


def parse_rating(
    text: str, default: str = "Hold", *, warn_on_default: bool = False,
) -> str:
    """Heuristically extract a 5-tier rating from prose text.

    Two-pass strategy:
    1. Look for an explicit "Rating: X" label (tolerant of markdown bold).
    2. Fall back to the first 5-tier rating word found anywhere in the text
       that is not preceded by a nearby negator — "argues against Sell" must
       not parse as Sell. A 2-token backward window covers "against Sell",
       "counter to Sell", "not Sell", "rather than Sell". The label pass
       needs no guard: the regex captures the word immediately after the
       separator, so a negator cannot slip in between.

    Returns a Title-cased rating string, or ``default`` if no rating word appears.
    When ``warn_on_default`` is True, a ``logger.warning`` is emitted on fallback
    so callers in critical paths (risk overlay, backtest engine) can surface
    silent degradations instead of treating a failed parse as a genuine "Hold".
    """
    for line in text.splitlines():
        m = _RATING_LABEL_RE.search(line)
        if m and m.group(1).lower() in _RATING_SET:
            return m.group(1).capitalize()

    for line in text.splitlines():
        tokens = [w.strip("*:.,") for w in line.lower().split()]
        for i, t in enumerate(tokens):
            if t in _RATING_SET and not _negated_before(tokens, i):
                return t.capitalize()

    if warn_on_default:
        snippet = (text or "").strip().replace("\n", " ")[:120]
        logger.warning(
            "parse_rating: no 5-tier rating found in text, defaulting to '%s'. "
            "This may be a degraded decision (LLM output unparseable). "
            "Text excerpt: %s",
            default, snippet,
        )
    return default
