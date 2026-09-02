"""V2 version stamps for every schema and ledger object (frozen baseline).

The V2 baseline (``docs/V2_BASELINE.md``) requires that persisted objects
carry their semantic version in code, not just in git history: a
``portfolio.db`` row written today must still be interpretable after the
definition behind it changes. Each constant below versions ONE contract;
bump the affected constant whenever that contract's field semantics change
(adding an optional field with a default is additive and does NOT require a
bump; renaming, removing, or redefining the meaning of a field does).

Which constant versions what:

``SCHEMA_VERSION``
    Agent structured-output shapes (``PortfolioDecision`` and, from V2.1,
    the analyst prediction schemas).
``COST_MODEL_VERSION``
    Fee/slippage/funding constants and the composition rules in
    :mod:`yialpha.risk.cost_model`.
``TICKET_VERSION``
    :class:`yialpha.tickets.ExecutionTicket` field semantics.
``FEATURE_VERSION``
    v2 (since V2.1 Measurability): the forecast-feature contract that
    attribution depends on — the frozen horizon ladder
    ``HORIZON_LADDER_DAYS = (1, 5, 21)`` (days on the instrument's own
    session calendar), ``direction in {up, down, flat}`` with ``prob_up`` =
    P(return > 0) over the horizon, and all return legs (expected_return /
    price / funding / basis / fees / slippage) quoted as signed fractions of
    notional, with funding PnL computed as ``-sign x notional x rate``.
    v1 had no consumers (reserved only).
``REGIME_VERSION``
    Reserved for V2.2: the ``RegimeState`` definitions (trend/volatility/
    positioning classifiers). Changing a classifier's thresholds or inputs
    bumps this so historical regime tags never silently mix definitions.
"""

from __future__ import annotations

SCHEMA_VERSION = "v1"
COST_MODEL_VERSION = "v1"
TICKET_VERSION = "v1"
FEATURE_VERSION = "v2"
REGIME_VERSION = "v1"
