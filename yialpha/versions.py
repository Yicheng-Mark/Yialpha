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
    v2 (since V2.2 Context): the frozen ``RegimeState`` classifier
    definitions in :mod:`yialpha.regime.state` — the contract this stamp
    protects. Changing ANY definition below bumps this so historical regime
    tags never silently mix definitions (the id preimage starts with the
    version string, so a bump also mints disjoint ids). Frozen definitions:

    * **trend / underlying_trend** — daily close vs SMA50/SMA200 on the
      series' own candles: ``up`` iff close > both SMAs, ``down`` iff below
      both, else ``range``; requires 200 rows.
    * **realized_vol_pct** — stdev (ddof=1) of the last 20 daily returns of
      the perp close series, in percent, NOT annualized.
    * **funding_pct** — net trailing-7-day funding settlement sum (signed
      fraction).
    * **oi_pct** — latest OI's percentile in the trailing 30-day window.
    * **lsr_crowding** — latest global-account long/short ratio;
      **taker_aggression** — latest daily taker buy/sell ratio.
    * **spot_perp_basis_bps / index_mark_basis_bps** — (perp/spot − 1)×1e4
      and (last/index − 1)×1e4 on 8/35-day windows.
    * **overnight_gap_bps** — (latest open / prior close − 1)×1e4.
    * **spread/depth/liquidity regimes** — live book: spread tight ≤ 2 bps /
      normal ≤ 10 / wide; ±50bps band notional deep ≥ $2M / normal ≥
      $500k / thin; combined ample/constrained/normal per the documented
      rule.
    * **market_stress** — any-trigger composite: |funding| ≥ 1%/7d,
      |spot basis| ≥ 50 bps, |index basis| ≥ 50 bps, |overnight gap| ≥
      100 bps, or OI ≥ 90th pct with +10% 1d build.
    * **session_state** — NYSE-equivalent Eastern buckets (pre_market
      04:00–09:30 / regular 09:30–16:00 / post_market 16:00–20:00 / closed;
      date-only as-of names the weekday session, weekends closed).
    * **listing_age_days** — calendar days from the registry PIT
      ``onboard_date``; **confidence_components** — per-family availability
      weights in [0, 1].

    v1 was reserved only (no consumers).
"""

from __future__ import annotations

SCHEMA_VERSION = "v1"
COST_MODEL_VERSION = "v1"
TICKET_VERSION = "v1"
FEATURE_VERSION = "v2"
REGIME_VERSION = "v2"
