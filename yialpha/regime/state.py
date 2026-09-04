"""The frozen ``RegimeState`` vocabulary + its deterministic content id.

V2.2 Context stage: before any specialization logic (V2.3) may condition on
"what kind of market am I in?", that question needs ONE versioned, hashable
answer. This module is that answer:

* :class:`RegimeState` — the frozen per-run regime record. Three field
  groups: perp COMMON (every ``crypto_perp`` run), PURE-CRYPTO (token-free
  contracts), and STOCK-PERP (tokenized US-equity contracts). Fields outside
  a run's class stay ``None`` and are NOT missing inputs — structural
  absence is not a data gap.
* :func:`compute_regime_id` — ``"G" + 12 hex`` of
  ``sha256(REGIME_VERSION + canonical_json(input fields))``. Determinism is
  the dedupe AND the linkage mechanism: the same classifier inputs always
  produce the same id, so a prediction row carrying ``regime_id`` names the
  exact context it was decided under, and re-running the compute for the
  same ``analysis_as_of`` lands on the same row (``INSERT OR IGNORE``).
* :func:`render_regime_block` — the markdown disclosure injected into the
  market analyst as explicitly marked external evidence (only non-``None``
  fields render; ``missing_inputs`` is always listed — a partial regime
  never masquerades as a complete one).

Frozen classifier definitions (REGIME_VERSION v3 — bump
:data:`yialpha.versions.REGIME_VERSION` when ANY of these changes, so
historical regime tags never silently mix definitions; v3 redefined ONLY
the stock-perp ``session_state`` bucket to the klines-verified 24/7
calendar):

* **trend_regime / underlying_trend** — daily close vs SMA50/SMA200 on the
  series' own candles: ``up`` when close > SMA50 AND close > SMA200,
  ``down`` when close < SMA50 AND close < SMA200, else ``range``. Requires
  200 rows; fewer leaves the field ``None``.
* **realized_vol_pct** — stdev (ddof=1) of the last 20 daily simple returns
  of the perp close series, expressed in percent (3.2 = 3.2% daily). NOT
  annualized — the raw daily figure is venue-independent and never smuggles
  an annualization convention into the id.
* **funding_pct** — net trailing-7-day funding settlement sum as a signed
  fraction (+0.0012 = longs paid 0.12% over 7d).
* **oi_pct** — latest open interest's percentile (0..100) in the trailing
  30-calendar-day daily OI window (the /futures/data retention ceiling).
* **lsr_crowding** — latest global-account long/short account ratio
  (>1 = leveraged crowd long-heavy).
* **taker_aggression** — latest daily taker buy/sell volume ratio (>1 =
  aggressive buying).
* **spot_perp_basis_bps** — (perp close / Binance spot close − 1) × 1e4 on
  the same 8-day window.
* **index_mark_basis_bps** — (last close / index close − 1) × 1e4 from the
  perp's own last/index kline legs.
* **overnight_gap_bps** — (latest open / prior close − 1) × 1e4 on the perp
  series (stock-perp session-gap reading).
* **spread_depth_regime** — live top-of-book spread: ``tight`` ≤ 2 bps,
  ``normal`` ≤ 10 bps, ``wide`` above.
* **contract_liquidity** — combined ±50 bps band notional (live book):
  ``deep`` ≥ $2M, ``normal`` ≥ $500k, ``thin`` below.
* **liquidity_regime** — the combined summary: ``ample`` when spread is
  tight AND band notional ≥ $500k; ``constrained`` when spread is wide OR
  band notional < $500k; ``normal`` otherwise.
* **market_stress** — True iff ANY available trigger fires:
  |funding_pct| ≥ 0.01 (1% per 7d), |spot_perp_basis_bps| ≥ 50,
  |index_mark_basis_bps| ≥ 50, or oi_pct ≥ 90 with 1d OI change > +10%.
  ``None`` when no trigger input was computable at all.
* **liquidation_cascade_risk** — ``elevated`` when stressed AND the book is
  thin/wide, ``moderate`` when stressed with a normal book, ``low`` when not
  stressed (live depth feeds the book half; historical runs answer from the
  PIT stress legs alone).
* **session_state** — continuous 24/7 bucket since REGIME_VERSION v3
  (klines machine evidence 2026-09-04: Binance stock perps trade through
  weekends and full US-market holidays with volume), so the value is
  always ``continuous_24_7``
  (:data:`yialpha.instruments.sessions.SESSION_CONTINUOUS`) and NEVER
  ``closed`` — the as-of instant no longer matters. The former
  NYSE-equivalent America/New_York vocabulary (``pre_market`` 04:00–09:30,
  ``regular`` 09:30–16:00, ``post_market`` 16:00–20:00, ``closed``
  otherwise incl. weekends; date-only as-of = that weekday's regular
  session / weekend closed) is DORMANT: its deterministic hand-rolled
  ET/DST machinery is retained in
  :func:`yialpha.regime.compute._et_session_state` for any future class
  with a genuine session calendar.
* **listing_age_days** — calendar days from the registry's PIT
  ``onboard_date`` to the analysis date.
* **earnings_window / sector_index_trend** — honest gaps this version:
  ``None`` with the matching ``missing_inputs`` entry (no PIT earnings
  calendar or sector-index source is wired yet; a fake value would poison
  the id).

``computed_at`` is deliberately EXCLUDED from the id preimage: it is a clock
reading, not a classifier input, and including it would give the same regime
a new id every recompute (breaking the INSERT OR IGNORE dedupe).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any

from yialpha.versions import REGIME_VERSION

#: Fields excluded from the regime-id preimage (identity/meta, not inputs).
_NON_INPUT_FIELDS = frozenset({"regime_id", "regime_version", "computed_at"})


@dataclass(frozen=True)
class RegimeState:
    """One run's versioned point-in-time market-regime record.

    Field groups (a field outside the run's instrument class stays ``None``
    and is never a missing input):

    * COMMON (every perp): ``trend_regime`` / ``realized_vol_pct`` /
      ``liquidity_regime`` / ``spread_depth_regime`` / ``market_stress`` /
      ``confidence_components``;
    * PURE-CRYPTO: ``funding_pct`` / ``oi_pct`` / ``lsr_crowding`` /
      ``taker_aggression`` / ``spot_perp_basis_bps`` /
      ``liquidation_cascade_risk``;
    * STOCK-PERP: ``underlying_trend`` / ``earnings_window`` /
      ``sector_index_trend`` / ``session_state`` / ``overnight_gap_bps`` /
      ``index_mark_basis_bps`` / ``contract_liquidity`` / ``listing_age_days``.

    ``missing_inputs`` names every input family that SHOULD exist for this
    class but could not be sourced (fail-soft per family); it is part of the
    id preimage — a regime computed with a missing leg is a DIFFERENT regime
    from the same date with the leg present, and must not silently merge.
    """

    # --- perp COMMON -------------------------------------------------------
    trend_regime: str | None = None
    realized_vol_pct: float | None = None
    liquidity_regime: str | None = None
    spread_depth_regime: str | None = None
    market_stress: bool | None = None
    confidence_components: dict[str, float] = field(default_factory=dict)
    # --- PURE-CRYPTO -------------------------------------------------------
    funding_pct: float | None = None
    oi_pct: float | None = None
    lsr_crowding: float | None = None
    taker_aggression: float | None = None
    spot_perp_basis_bps: float | None = None
    liquidation_cascade_risk: str | None = None
    # --- STOCK-PERP --------------------------------------------------------
    underlying_trend: str | None = None
    earnings_window: str | None = None
    sector_index_trend: str | None = None
    session_state: str | None = None
    overnight_gap_bps: float | None = None
    index_mark_basis_bps: float | None = None
    contract_liquidity: str | None = None
    listing_age_days: int | None = None
    # --- identity / meta ----------------------------------------------------
    regime_id: str = ""
    regime_version: str = REGIME_VERSION
    analysis_as_of: str = ""
    computed_at: str = ""
    missing_inputs: tuple[str, ...] = ()


def compute_regime_id(
    state: RegimeState | Mapping[str, Any],
    regime_version: str = REGIME_VERSION,
) -> str:
    """Deterministic regime id: ``"G" + 12 hex`` over version + input fields.

    Preimage: ``{regime_version}`` followed by the canonical JSON
    (sorted keys, no whitespace) of every :class:`RegimeState` field except
    ``regime_id`` / ``regime_version`` / ``computed_at``. The version prefix
    means a classifier redefinition (REGIME_VERSION bump) can never collide
    with ids minted under the old definitions — same inputs, different
    version, different id. Accepts a plain mapping (e.g. a stored payload)
    so readers can re-derive an id without rebuilding the dataclass.
    """
    data = asdict(state) if isinstance(state, RegimeState) else dict(state)
    payload = {key: value for key, value in data.items() if key not in _NON_INPUT_FIELDS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "G" + sha256((regime_version + canonical).encode("utf-8")).hexdigest()[:12]


def regime_state_from_mapping(data: Mapping[str, Any]) -> RegimeState:
    """Coerce a stored payload (JSON dict) back into a :class:`RegimeState`.

    ``missing_inputs`` comes back from JSON as a list and is restored to the
    canonical sorted tuple; unknown keys (e.g. the ``ticker`` /
    ``instrument_class`` ledger columns that ride alongside the payload) are
    ignored. Pure local coercion — no validation of classifier values.
    """
    known = set(RegimeState.__dataclass_fields__)
    kwargs: dict[str, Any] = {key: value for key, value in dict(data).items() if key in known}
    raw_missing = kwargs.get("missing_inputs") or ()
    kwargs["missing_inputs"] = tuple(sorted(str(item) for item in raw_missing))
    components = kwargs.get("confidence_components")
    kwargs["confidence_components"] = (
        {str(k): float(v) for k, v in dict(components).items()}
        if isinstance(components, Mapping)
        else {}
    )
    return RegimeState(**kwargs)


def render_regime_block(state: RegimeState) -> str:
    """Markdown disclosure of one regime; ``None`` fields never render.

    Deterministic lines only (no prose interpolation of values beyond
    formatting); ``missing_inputs`` is ALWAYS listed when non-empty so a
    partial regime reads as partial. ``market_stress`` renders as
    ``true``/``false`` for log-greppability.
    """
    lines: list[str] = [
        f"### Regime State — version {state.regime_version}, "
        f"as of {state.analysis_as_of or 'n/a'}",
    ]
    if state.trend_regime is not None:
        lines.append(f"- **Trend**: {state.trend_regime} (close vs SMA50/SMA200, daily)")
    if state.realized_vol_pct is not None:
        lines.append(f"- **Realized vol (20d)**: {state.realized_vol_pct:.3f}% daily")
    if state.market_stress is not None:
        lines.append(f"- **Market stress**: {'true' if state.market_stress else 'false'}")
    if state.liquidity_regime is not None:
        lines.append(f"- **Liquidity**: {state.liquidity_regime}")
    if state.spread_depth_regime is not None:
        lines.append(f"- **Spread regime**: {state.spread_depth_regime} (live book)")
    if state.liquidation_cascade_risk is not None:
        lines.append(
            f"- **Liquidation cascade risk**: {state.liquidation_cascade_risk}"
        )
    if state.funding_pct is not None:
        lines.append(f"- **Funding (7d net)**: {state.funding_pct:+.4%} (longs pay when +)")
    if state.oi_pct is not None:
        lines.append(
            f"- **Open interest**: {state.oi_pct:.0f}th pct of trailing 30d"
        )
    if state.lsr_crowding is not None:
        lines.append(
            f"- **LSR crowding**: {state.lsr_crowding:.2f} global accounts "
            "(>1 = long-heavy)"
        )
    if state.taker_aggression is not None:
        lines.append(
            f"- **Taker aggression**: {state.taker_aggression:.2f} buy/sell "
            "(>1 = aggressive buying)"
        )
    if state.spot_perp_basis_bps is not None:
        lines.append(f"- **Spot-perp basis**: {state.spot_perp_basis_bps:+.1f} bps")
    if state.underlying_trend is not None:
        lines.append(
            f"- **Underlying trend**: {state.underlying_trend} "
            "(equity close vs SMA50/SMA200)"
        )
    if state.overnight_gap_bps is not None:
        lines.append(f"- **Overnight gap**: {state.overnight_gap_bps:+.1f} bps")
    if state.index_mark_basis_bps is not None:
        lines.append(f"- **Last-index basis**: {state.index_mark_basis_bps:+.1f} bps")
    if state.contract_liquidity is not None:
        lines.append(f"- **Contract liquidity**: {state.contract_liquidity} (±50bps depth)")
    if state.session_state is not None:
        lines.append(
            f"- **Session state**: {state.session_state} "
            "(24/7 continuous — klines-verified 2026-09-04)"
        )
    if state.listing_age_days is not None:
        lines.append(f"- **Listing age**: {state.listing_age_days} days")
    if state.earnings_window is not None:
        lines.append(f"- **Earnings window**: {state.earnings_window}")
    if state.sector_index_trend is not None:
        lines.append(f"- **Sector index trend**: {state.sector_index_trend}")
    if state.confidence_components:
        rendered = ", ".join(
            f"{name}={weight:.2f}"
            for name, weight in sorted(state.confidence_components.items())
        )
        lines.append(f"- **Input coverage**: {rendered}")
    if state.missing_inputs:
        lines.append(
            "- **Missing inputs**: " + ", ".join(state.missing_inputs)
        )
    lines.append(f"- **Regime id**: {state.regime_id or 'n/a'}")
    return "\n".join(lines)
