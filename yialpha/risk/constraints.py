"""The five frozen hard portfolio constraints + advisory metrics (V2.4).

docs/V2_BASELINE.md freezes the V2.4 portfolio-control core at exactly FIVE
hard constraints in exactly this order:

    global_gross -> asset_class -> single_concentration
    -> directional_concentration -> correlation_cluster

The ORDER is part of the contract (:data:`FIVE_HARD_CONSTRAINTS`), because
the report renders the decisions in that sequence and the resolver's
min-multiplier composition must be reproducible from the log.

Shared semantics (every constraint function):

* The candidate REPLACES an existing position with the same symbol in the
  snapshot (re-sizing a holding), else it is ADDED — the metric is always
  computed on the post-candidate book.
* ``weight`` is a SIGNED fraction of equity; exposure metrics (gross, asset
  class, single, cluster) use ``abs(weight)``.
* Over the limit -> RESIZE into the UNUSED headroom, never onto the old
  book: with ``others`` the post-replacement book's metric minus the
  candidate's own contribution and ``contribution`` what the candidate
  adds at the proposed weight, ``multiplier = clip((limit - others) /
  contribution, 0, 1)`` — the resized candidate lands the book exactly on
  the limit. At or under -> PASS with multiplier 1.0. An old book already
  at/over the limit leaves zero headroom -> RESIZE with multiplier 0.0 and
  reason ``book_already_over_limit`` (the risk layer shrinks the CANDIDATE,
  never the existing book). A lone position over ``max_single`` is still a
  RESIZE (the headroom form degenerates to ``limit / |proposed|``), NOT
  a VETO — VETO is reserved for hard refusals: non-finite inputs,
  non-positive equity, a degenerate candidate (weight NaN/inf), and a
  candidate NO admissible size can make compliant
  (``opposite_direction_exceeds_limit`` in :func:`directional_concentration`:
  shrinking a weakened offsetting position only deepens the net exposure it
  used to offset, so the trade is refused and the pre-trade book stands).
  The risk layer never refuses a trade it can simply shrink.
* Fail-closed: :func:`evaluate_constraints` never raises — a crashing
  constraint yields a VETO with reason ``constraint_error``.

Advisory metrics (:func:`advisory_metrics`) are computed and DISPLAYED but
NEVER resize, per the baseline freeze: VaR / CVaR / beta / correlation
matrix are calculate-display-archive only in v1.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from yialpha.risk.cvar import historical_cvar
from yialpha.risk.signed_math import Side, position_sign

#: Advisory-metric value union: a scalar, a flat dict, a nested correlation
#: matrix, or None when the input needed to compute the value honestly is
#: absent (a refusal to fabricate, not an error).
AdvisoryValue = float | dict[str, float] | dict[str, dict[str, float]] | None

#: Minimum finite return observations before var_95 / cvar_95 are displayed;
#: aligned with ``cvar_position_multiplier``'s default ``min_observations``
#: so both layers refuse to read distribution estimates off thin samples.
MIN_ADVISORY_HISTORY = 30

#: Stress-shock magnitudes (advisory only, frozen for reproducible reports).
USDT_DEPEG_SHOCK = 0.10          # USDT/USD -10%
OUTAGE_VOL_MULTIPLE = 2.0        # one untradeable day costs 2x daily vol
BTC_CRASH_MAGNITUDE = 0.20       # BTC -20%
EARNINGS_GAP_MAGNITUDE = 0.08    # +8% overnight gap against stock_perps


@dataclass(frozen=True)
class RiskDecision:
    """One hard constraint's verdict on a candidate.

    ``multiplier`` is ALWAYS a finite float in [0.0, 1.0] (validated); VETO
    additionally requires exactly 0.0. ``reasons`` are stable
    machine-readable strings (``within_limit`` / ``exceeds_limit`` /
    ``book_already_over_limit`` / veto causes / ``constraint_error``);
    ``metrics`` carries the post-candidate metric and the limit it was
    judged against, under rule-specific keys.
    """

    rule: str
    action: Literal["PASS", "RESIZE", "VETO"]
    multiplier: float
    reasons: list[str]
    metrics: dict[str, float]

    def __post_init__(self) -> None:
        if self.action not in ("PASS", "RESIZE", "VETO"):
            raise ValueError(
                f"RiskDecision.action must be PASS/RESIZE/VETO, got {self.action!r}"
            )
        if not math.isfinite(self.multiplier) or not 0.0 <= self.multiplier <= 1.0:
            raise ValueError(
                f"RiskDecision.multiplier must be a finite float in [0.0, 1.0], "
                f"got {self.multiplier!r}"
            )
        if self.action == "VETO" and self.multiplier != 0.0:
            raise ValueError("a VETO must carry multiplier == 0.0")


@dataclass(frozen=True)
class PositionView:
    """One open position as the constraint layer sees it.

    ``weight`` is the SIGNED fraction of equity (long positive, short
    negative); ``side`` restates the direction in the :data:`Side`
    vocabulary the resolver echoes. ``instrument_class`` is the routing
    vocabulary (``PerpInstrumentClass`` values — kept as ``str`` here so
    this module owns no dependency on the instruments package).
    """

    symbol: str
    instrument_class: str
    side: Side
    weight: float


@dataclass(frozen=True)
class PortfolioSnapshot:
    """The book the candidate is previewed against.

    ``positions`` are the open positions BEFORE the candidate (a same-symbol
    entry is replaced, not added). ``returns_history`` is the portfolio's
    daily return series feeding the ADVISORY VaR/CVaR/stress metrics only —
    never a hard resize.
    """

    equity: float
    positions: tuple[PositionView, ...]
    returns_history: tuple[float, ...] = ()


@dataclass(frozen=True)
class PortfolioLimits:
    """The five caps + the cluster map. Defaults are the frozen V2.4 values.

    ``cluster_of`` maps symbol -> cluster id; symbols absent from the map
    form their own single-symbol cluster. When ``cluster_of`` is None every
    symbol is its own cluster, which makes ``max_cluster`` behave as a
    per-symbol SECOND cap alongside ``max_single`` — harmless (the tighter
    of the two binds) and honest: v1 has no correlation grouping data yet.
    """

    max_gross: float = 1.0
    max_asset_class: float = 0.6
    max_single: float = 0.2
    max_directional: float = 0.8
    max_cluster: float = 0.4
    cluster_of: Mapping[str, str] | None = None


# --- shared helpers ---------------------------------------------------------


def _veto(rule: str, reason: str) -> RiskDecision:
    """A hard refusal: multiplier 0.0, one stable machine-readable reason."""
    return RiskDecision(rule=rule, action="VETO", multiplier=0.0, reasons=[reason], metrics={})


def _post_positions(
    candidate: PositionView, snapshot: PortfolioSnapshot
) -> tuple[PositionView, ...]:
    """The post-candidate book: same-symbol position replaced, else appended."""
    return tuple(p for p in snapshot.positions if p.symbol != candidate.symbol) + (
        candidate,
    )


def _degenerate_veto(
    rule: str, candidate: PositionView, snapshot: PortfolioSnapshot
) -> RiskDecision | None:
    """VETO-worthy garbage the constraint cannot honestly judge, else None.

    Hard refusals only: a non-finite candidate weight, a non-finite or
    non-positive equity, or a non-finite existing position weight. Anything
    else is a number the limit math can clip.
    """
    if not math.isfinite(candidate.weight):
        return _veto(rule, "non_finite_candidate_weight")
    if not math.isfinite(snapshot.equity) or snapshot.equity <= 0.0:
        return _veto(rule, "non_positive_equity")
    for p in snapshot.positions:
        if not math.isfinite(p.weight):
            return _veto(rule, "non_finite_position_weight")
    return None


def _limit_decision(
    rule: str,
    metric_key: str,
    limit_key: str,
    others: float,
    contribution: float,
    limit: float,
    *,
    post_metric: float | None = None,
) -> RiskDecision:
    """Size the candidate into the UNUSED headroom, never onto the old book.

    ``others`` is the post-replacement book's metric MINUS the candidate's
    own contribution; ``contribution`` is what the candidate adds to the
    metric at the proposed weight. A non-positive contribution adds nothing
    and cannot be shrunk any further (the candidate is already zero), so it
    PASSes UNCONDITIONALLY — this check must precede the at/over-limit branch
    below, otherwise a risk-reducing CLOSE whose remaining book sits at the
    cap is stranded as a zero-multiplier RESIZE the upstream graph will never
    execute, even though the candidate contributes nothing the limit math
    could clip. Otherwise, an old book already at/over the limit (tolerance
    1e-12) pins the multiplier at 0.0 with reason ``book_already_over_limit``
    — the risk layer shrinks the candidate, never the existing book. The
    candidate fits whenever ``limit - others >= contribution`` (PASS, 1.0)
    and is otherwise clipped to the headroom ratio
    ``clip((limit - others) / contribution, 0, 1)`` — the resized candidate
    lands the book exactly on the limit. ``post_metric`` overrides the
    reported metric (default ``others + contribution``) for rules whose post
    book nets below the simple sum (directional hedges).
    """
    if post_metric is None:
        post_metric = others + contribution
    metrics = {metric_key: post_metric, limit_key: limit}
    if not math.isfinite(limit):
        return _veto(rule, "non_finite_limit")
    if contribution <= 0.0:
        return RiskDecision(
            rule=rule, action="PASS", multiplier=1.0, reasons=["within_limit"],
            metrics=metrics,
        )
    if others >= limit - 1e-12:
        return RiskDecision(
            rule=rule, action="RESIZE", multiplier=0.0,
            reasons=["book_already_over_limit"], metrics=metrics,
        )
    if limit - others >= contribution:
        return RiskDecision(
            rule=rule, action="PASS", multiplier=1.0, reasons=["within_limit"],
            metrics=metrics,
        )
    multiplier = min(max((limit - others) / contribution, 0.0), 1.0)
    return RiskDecision(
        rule=rule, action="RESIZE", multiplier=multiplier, reasons=["exceeds_limit"],
        metrics=metrics,
    )


def _cluster_id(symbol: str, cluster_of: Mapping[str, str] | None) -> str:
    """Cluster id of ``symbol``; unmapped symbols cluster alone (the default)."""
    if cluster_of is None:
        return symbol
    return cluster_of.get(symbol, symbol)


# --- the five frozen hard constraints (order is the contract) ---------------


def global_gross(
    candidate: PositionView, snapshot: PortfolioSnapshot, limits: PortfolioLimits
) -> RiskDecision:
    """Rule ``global_gross``: sum of |weight| over the post-candidate book.

    The whole-portfolio leverage cap. The candidate replaces a same-symbol
    position; gross = ``sum(abs(w))`` so a long and a short BOTH consume
    gross (offsetting is directional exposure's job, not gross's). The
    candidate's contribution is ``abs(proposed)``; ``others`` is the rest
    of the book's gross — only the unused global headroom is on offer.
    """
    veto = _degenerate_veto("global_gross", candidate, snapshot)
    if veto is not None:
        return veto
    positions = _post_positions(candidate, snapshot)
    others = sum(abs(p.weight) for p in positions if p.symbol != candidate.symbol)
    return _limit_decision(
        "global_gross", "post_gross", "max_gross",
        others, abs(candidate.weight), limits.max_gross,
    )


def asset_class(
    candidate: PositionView, snapshot: PortfolioSnapshot, limits: PortfolioLimits
) -> RiskDecision:
    """Rule ``asset_class``: gross |weight| within the candidate's class.

    Groups the post-candidate book by ``instrument_class``
    (``stock_perp`` / ``pure_crypto_perp`` / ``unknown_perp``) and caps the
    candidate's own class — the stock-perp sleeve cannot quietly swallow the
    pure-crypto sleeve's budget. ``others`` is the same-class gross of the
    rest of the book; the candidate only gets that class's unused headroom.
    """
    veto = _degenerate_veto("asset_class", candidate, snapshot)
    if veto is not None:
        return veto
    positions = _post_positions(candidate, snapshot)
    others = sum(
        abs(p.weight)
        for p in positions
        if p.symbol != candidate.symbol
        and p.instrument_class == candidate.instrument_class
    )
    return _limit_decision(
        "asset_class", "post_class_gross", "max_asset_class",
        others, abs(candidate.weight), limits.max_asset_class,
    )


def single_concentration(
    candidate: PositionView, snapshot: PortfolioSnapshot, limits: PortfolioLimits
) -> RiskDecision:
    """Rule ``single_concentration``: |weight| of the candidate alone.

    Post-replacement the candidate IS the symbol's position, so the rest
    of the book contributes nothing (``others`` = 0) and the headroom form
    degenerates to ``limit / |proposed|`` — numerically identical to the
    old whole-book ratio. A lone position over ``max_single`` is a RESIZE
    (e.g. 0.25 over a 0.20 cap resizes x0.8), NEVER a VETO: shrinking is
    always available, and VETO is reserved for degenerate inputs.
    """
    veto = _degenerate_veto("single_concentration", candidate, snapshot)
    if veto is not None:
        return veto
    return _limit_decision(
        "single_concentration", "post_single", "max_single",
        0.0, abs(candidate.weight), limits.max_single,
    )


def directional_concentration(
    candidate: PositionView, snapshot: PortfolioSnapshot, limits: PortfolioLimits
) -> RiskDecision:
    """Rule ``directional_concentration``: NET exposure on the candidate's side.

    The metric is the post-candidate book's NET signed weight, taken in the
    candidate's direction: ``max(sign(side) * sum(w), 0)`` — same-side gross
    NETS against opposite-side positions (a 0.4 long candidate against a
    0.7 short book has only 0.0 net long at risk in this direction), and a
    book that nets the OTHER way exposes 0.0 in the candidate's direction.
    A FLAT candidate adds no direction -> metric 0.0 -> PASS.

    Headroom form: ``others`` is the old book's net exposure IN the
    candidate's direction (0.0 when the book nets the other way — a hedge
    gets no budget credit from the exposure it offsets) and the
    contribution is ``abs(proposed)``; the reported metric stays the true
    post-candidate net, which nets below ``others + contribution`` when
    the candidate is itself partially hedged.

    Opposite-direction feasibility (the ACTUAL final book): the cap bounds
    the post-candidate book's net exposure in BOTH directions, but the
    headroom form above judges only the candidate's OWN direction. A
    same-symbol replacement that weakens an offsetting position — a -0.10
    short hedge re-proposed at -0.01 on a book that nets long — passes its
    own direction at 0.0 while the final book's net LONG breaches the cap.
    So after the own-side form, the final book's opposite-direction net
    exposure (``max(-sign * net, 0)``) is validated too. When NO admissible
    size can restore compliance — the candidate's weight points along its
    own side, so every shrink moves net FURTHER toward the opposite
    direction and even the full proposed size violates — the rule VETOs
    with reason ``opposite_direction_exceeds_limit``: the trade is refused
    outright (the pre-trade book, offsetting position intact, stands) and
    the candidate is never enlarged to manufacture the hedge back. A weight
    pointing AGAINST its declared side (side/weight disagreement — produced
    nowhere upstream) instead GROWS the opposite exposure with size, so it
    is resized into the opposite headroom like any other contribution.
    """
    veto = _degenerate_veto("directional_concentration", candidate, snapshot)
    if veto is not None:
        return veto
    sign = position_sign(candidate.side)
    if sign == 0:
        return RiskDecision(
            rule="directional_concentration", action="PASS", multiplier=1.0,
            reasons=["within_limit"],
            metrics={"post_directional": 0.0, "max_directional": limits.max_directional},
        )
    positions = _post_positions(candidate, snapshot)
    others_net = sum(p.weight for p in positions if p.symbol != candidate.symbol)
    net = others_net + candidate.weight
    others = max(sign * others_net, 0.0)
    decision = _limit_decision(
        "directional_concentration", "post_directional", "max_directional",
        others, abs(candidate.weight), limits.max_directional,
        post_metric=max(sign * net, 0.0),
    )
    if decision.action == "VETO":
        return decision
    opp_exposure = max(-sign * net, 0.0)
    if opp_exposure <= limits.max_directional + 1e-12:
        return decision
    if sign * candidate.weight > 0.0:
        # Shrinking only deepens the opposite net; the full proposed size is
        # the most compliant size available and it still violates -> refuse.
        return RiskDecision(
            rule="directional_concentration", action="VETO", multiplier=0.0,
            reasons=["opposite_direction_exceeds_limit"],
            metrics={
                "post_directional": max(sign * net, 0.0),
                "opposite_directional": opp_exposure,
                "max_directional": limits.max_directional,
            },
        )
    # Defensive: side/weight disagree, so the opposite exposure GROWS with
    # size — shrink into the opposite headroom, keeping the tighter of the
    # two directional multipliers.
    opp_decision = _limit_decision(
        "directional_concentration", "post_directional", "max_directional",
        max(-sign * others_net, 0.0), abs(candidate.weight),
        limits.max_directional,
        post_metric=opp_exposure,
    )
    return opp_decision if opp_decision.multiplier < decision.multiplier else decision


def correlation_cluster(
    candidate: PositionView, snapshot: PortfolioSnapshot, limits: PortfolioLimits
) -> RiskDecision:
    """Rule ``correlation_cluster``: gross |weight| in the candidate's cluster.

    Clusters come from ``limits.cluster_of`` (symbol -> cluster id); symbols
    absent from the map cluster alone. With ``cluster_of=None`` every symbol
    is its own cluster, so this degenerates to a per-symbol SECOND cap on
    ``abs(candidate.weight)`` next to ``max_single`` (documented default —
    the tighter cap binds; real correlation grouping arrives with data).
    ``others`` is the rest of the cluster's gross; only the cluster's
    unused headroom is on offer.
    """
    veto = _degenerate_veto("correlation_cluster", candidate, snapshot)
    if veto is not None:
        return veto
    positions = _post_positions(candidate, snapshot)
    cluster = _cluster_id(candidate.symbol, limits.cluster_of)
    others = sum(
        abs(p.weight)
        for p in positions
        if p.symbol != candidate.symbol
        and _cluster_id(p.symbol, limits.cluster_of) == cluster
    )
    return _limit_decision(
        "correlation_cluster", "post_cluster", "max_cluster",
        others, abs(candidate.weight), limits.max_cluster,
    )


#: The five frozen hard constraints in the FROZEN EXECUTION ORDER — the order
#: itself is part of the V2.4 contract (report rendering + reproducible logs).
ConstraintFn = Callable[
    [PositionView, PortfolioSnapshot, PortfolioLimits], RiskDecision
]

FIVE_HARD_CONSTRAINTS: tuple[ConstraintFn, ...] = (
    global_gross,
    asset_class,
    single_concentration,
    directional_concentration,
    correlation_cluster,
)


def evaluate_constraints(
    candidate: PositionView, snapshot: PortfolioSnapshot, limits: PortfolioLimits
) -> list[RiskDecision]:
    """Run all five constraints in frozen order; NEVER raises (fail closed).

    A constraint that crashes yields ``rule=<its name>, action=VETO,
    multiplier=0.0, reasons=["constraint_error"]`` — a broken risk check
    refuses the trade instead of waving it through. The remaining
    constraints still run so the report shows the full book diagnosis.
    """
    decisions: list[RiskDecision] = []
    for fn in FIVE_HARD_CONSTRAINTS:
        rule = str(getattr(fn, "__name__", "unknown"))
        try:
            decisions.append(fn(candidate, snapshot, limits))
        except Exception:
            decisions.append(
                _veto(rule, "constraint_error")
            )
    return decisions


# --- advisory metrics (calculate + display + archive; NEVER resize) ---------


def _finite_floats(values: Sequence[float]) -> list[float]:
    """Finite-only copy of a return series (NaN/inf dropped, order kept)."""
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def _percentile_linear(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list.

    Matches ``numpy.percentile``'s default (linear) method so advisory VaR
    reproduces across environments; ``q`` in [0, 1].
    """
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _sample_stdev(vals: list[float]) -> float | None:
    """ddof=1 stdev; None below two observations."""
    n = len(vals)
    if n < 2:
        return None
    mean = sum(vals) / n
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))


def _beta(portfolio: list[float], bench: list[float]) -> float | None:
    """``cov(p, b) / var(b)``; None when misaligned, <2 points, or flat bench.

    Series must be the SAME LENGTH after finite filtering (unaligned daily
    series would fabricate a beta); a zero-variance bench has no beta to
    estimate, which is reported as None rather than a division blow-up.
    """
    if len(portfolio) != len(bench) or len(portfolio) < 2:
        return None
    n = len(portfolio)
    mp = sum(portfolio) / n
    mb = sum(bench) / n
    var_b = sum((b - mb) ** 2 for b in bench)
    if var_b <= 0.0:
        return None
    cov = sum((p - mp) * (b - mb) for p, b in zip(portfolio, bench, strict=True))
    return cov / var_b


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation; None when misaligned, <2 points, or zero variance."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    var_x = sum((x - mx) ** 2 for x in xs)
    var_y = sum((y - my) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return cov / math.sqrt(var_x * var_y)


def _correlation_matrix(
    position_returns: Mapping[str, Sequence[float]] | None,
) -> dict[str, dict[str, float]]:
    """Symmetric pairwise-Pearson matrix over per-symbol return series.

    ``{}`` when no series were provided (the honest default — v1 usually
    has none). Symbols are iterated in sorted order for determinism; the
    diagonal is 1.0 for any series with >=2 finite observations, and a pair
    is OMITTED (not fabricated) when its series are misaligned or one is
    zero-variance.
    """
    if not position_returns:
        return {}
    symbols = sorted(position_returns)
    series = {s: _finite_floats(position_returns[s]) for s in symbols}
    matrix: dict[str, dict[str, float]] = {}
    for a in symbols:
        row: dict[str, float] = {}
        if len(series[a]) >= 2:
            row[a] = 1.0
        for b in symbols:
            if b == a:
                continue
            rho = _pearson(series[a], series[b])
            if rho is not None:
                row[b] = rho
        matrix[a] = row
    return matrix


def advisory_metrics(
    candidate: PositionView,
    snapshot: PortfolioSnapshot,
    limits: PortfolioLimits,
    *,
    btc_returns: Sequence[float] | None = None,
    position_returns: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, AdvisoryValue]:
    """Advisory-only portfolio diagnostics — computed, displayed, archived,
    and NEVER fed into a resize (baseline freeze: VaR/CVaR/beta/相关矩阵
    第一版只计算+展示+入档).

    Keys and semantics (values are floats, nested dicts, or None when the
    input needed to compute them honestly is absent — a refusal to
    fabricate, not an error):

    * ``usdt_collateral_concentration`` — fraction of gross in USDT-quoted
      instruments. PLACEHOLDER: every V2.4 position is a USDT-M perp, so
      this is 1.0 whenever gross > 0 (0.0 on a flat book); it becomes a
      real concentration the day a non-USDT-quote instrument enters scope.
    * ``venue_exposure`` — 1.0 single-venue PLACEHOLDER (Binance only in
      v1; no second venue exists to spread or concentrate).
    * ``var_95`` / ``cvar_95`` — historical 5th-percentile VaR and tail
      mean from ``snapshot.returns_history`` (CVaR reuses
      :func:`yialpha.risk.cvar.historical_cvar`); None below
      :data:`MIN_ADVISORY_HISTORY` finite observations.
    * ``beta_vs_btc`` — ``cov/var`` of portfolio vs ``btc_returns``; None
      without the optional series (or misaligned/degenerate input).
    * ``correlation_matrix`` — pairwise Pearson over ``position_returns``
      (symbol -> series); ``{}`` when absent.
    * ``usdt_depeg_shock`` — post-shock equity delta assuming USDT/USD
      -10% on a fully-USDT-collateralized book (``-0.10 x equity``).
    * ``binance_outage_shock`` — equity delta of the perp gross being
      untradeable for one day at 2x daily vol
      (``-equity x gross x 2 x stdev(returns)``); None with <2 returns.
    * ``btc_crash_x_earnings_gap`` — equity delta of BTC -20% x beta plus
      an 8% overnight gap against stock_perp gross; None when no beta can
      be estimated (the shock is compound — refusing half of it would
      understate the tail).

    ``limits`` is accepted for signature symmetry with the hard-constraint
    entry point; no advisory metric reads a limit in v1.
    """
    positions = _post_positions(candidate, snapshot)
    gross = sum(abs(p.weight) for p in positions)
    stock_perp_gross = sum(
        abs(p.weight) for p in positions if p.instrument_class == "stock_perp"
    )
    finite_returns = _finite_floats(snapshot.returns_history)

    out: dict[str, AdvisoryValue] = {}

    # Concentration placeholders (documented above).
    out["usdt_collateral_concentration"] = 1.0 if gross > 0.0 else 0.0
    out["venue_exposure"] = 1.0

    # Historical VaR / CVaR (display-only in v1).
    if len(finite_returns) >= MIN_ADVISORY_HISTORY:
        ordered = sorted(finite_returns)
        out["var_95"] = _percentile_linear(ordered, 0.05)
        out["cvar_95"] = historical_cvar(finite_returns, confidence=0.95)
    else:
        out["var_95"] = None
        out["cvar_95"] = None

    # Beta vs BTC (None-safe without the bench series).
    beta: float | None = None
    if btc_returns is not None:
        beta = _beta(finite_returns, _finite_floats(btc_returns))
    out["beta_vs_btc"] = beta

    # Pairwise correlation matrix ({} when no series provided).
    out["correlation_matrix"] = _correlation_matrix(position_returns)

    # Stress shocks: post-shock EQUITY deltas (negative = loss), advisory.
    out["usdt_depeg_shock"] = -USDT_DEPEG_SHOCK * snapshot.equity
    vol = _sample_stdev(finite_returns)
    out["binance_outage_shock"] = (
        None if vol is None else -snapshot.equity * gross * OUTAGE_VOL_MULTIPLE * vol
    )
    out["btc_crash_x_earnings_gap"] = (
        None
        if beta is None
        else snapshot.equity
        * (-BTC_CRASH_MAGNITUDE * beta - EARNINGS_GAP_MAGNITUDE * stock_perp_gross)
    )
    return out
