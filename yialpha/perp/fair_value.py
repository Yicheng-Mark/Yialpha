"""Fair Value Bridge: USD underlying target -> USDT perp contract target.

Stock perps quote the CONTRACT in USDT while the PM's price target lives on
the UNDERLYING equity in USD. The bridge converts one into the other in a
single deterministic, fully disclosed step:

    contract_target_usdt = underlying_target_usd / usdt_usd
                           * (1.0 + expected_basis)

Direction: ``usdt_usd`` is USD per 1 USDT, so dividing a USD amount by it
yields the USDT-denominated equivalent (100 USD at 0.9993 USD/USDT ≈
100.07 USDT). ``expected_basis`` is the anticipated contract premium vs
index at the horizon; the default ``0.0`` is the disclosed convergence
assumption (perp tracks its index), so the default bridge is a pure FX
conversion. ``current_basis`` does NOT enter the formula — it is the
observed last-vs-index fraction recorded for audit alongside the
assumption the conversion actually used.

Purity: no I/O, no LLM, no clock — every input arrives as an argument, so
the same inputs always produce the same result and the chain of
human-readable steps can be replayed/re-verified from the logged snapshot.

Call sites are gated by the ``stock_perp_fair_value`` config flag
(production default on, held off in tests) — this module never reads the
flag itself; it is wired by the overlay in a later batch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Plausibility band on the fx input (USD per 1 USDT). Wider than the
#: fetch-side depeg band [0.95, 1.05] because this module may be fed by any
#: caller — a rate at/beyond the edges of a genuine stablecoin depeg must
#: be refused, not converted through.
_FX_BAND_LOW, _FX_BAND_HIGH = 0.9, 1.1

#: ``missing_inputs`` vocabulary (frozen): each key names one absent or
#: unusable leg of the bridge.
MISSING_UNDERLYING = "underlying_target"
MISSING_FX = "fx"
MISSING_FX_OUT_OF_BAND = "fx_out_of_band"
MISSING_CURRENT_BASIS = "current_basis"

#: Human phrasing per missing-input key for the disclosure block.
_MISSING_EXPLANATIONS: dict[str, str] = {
    MISSING_UNDERLYING: "underlying USD price target not provided",
    MISSING_FX: "USDT/USD fx rate unavailable (live-only feed; never defaulted)",
    MISSING_FX_OUT_OF_BAND: (
        f"USDT/USD fx outside [{_FX_BAND_LOW:.2f}, {_FX_BAND_HIGH:.2f}] "
        "(depegged feed; refused)"
    ),
    MISSING_CURRENT_BASIS: (
        "current basis snapshot not provided (diagnostic only; conversion "
        "proceeded without it)"
    ),
}

#: Heading of the markdown disclosure block (rendered verbatim by
#: :func:`render_bridge_block`; pinned by tests).
BRIDGE_HEADING = "### Fair Value Bridge (USD → USDT contract target)"


@dataclass(frozen=True)
class FairValueResult:
    """Outcome of one bridge evaluation.

    ``contract_target_usdt`` carries FULL precision — never rounded here;
    only rendered strings are formatted. ``missing_inputs`` is a subset of
    ``{"underlying_target", "fx", "fx_out_of_band", "current_basis"}`` in
    bridge order (underlying, fx, fx band, basis); ``chain`` holds the
    ordered human-readable conversion steps and is empty whenever the
    target could not be computed — the warnings replace it.
    """

    underlying_target_usd: float | None
    usdt_usd: float | None
    current_basis: float | None
    expected_basis: float = 0.0
    contract_target_usdt: float | None = None
    missing_inputs: tuple[str, ...] = ()
    chain: tuple[str, ...] = ()


def fair_value_bridge(
    underlying_target_usd: float | None,
    usdt_usd: float | None,
    current_basis: float | None,
    expected_basis: float = 0.0,
) -> FairValueResult:
    """Convert a USD underlying target into a USDT contract target.

    Pure function: ``contract_target_usdt = underlying_target_usd /
    usdt_usd * (1.0 + expected_basis)`` with ``usdt_usd`` in USD per 1
    USDT (dividing a USD target by USD-per-USDT yields the
    USDT-denominated equivalent) and ``expected_basis`` the anticipated
    contract premium vs index at the horizon — default ``0.0`` is the
    disclosed convergence assumption. ``current_basis`` never enters the
    math: it is the audit snapshot recorded alongside the assumption, so a
    missing basis is DIAGNOSTIC ONLY (named in ``missing_inputs`` and
    rendered as a warning) and does not block the conversion.

    Validation (every failure keeps the inputs on the result so the caller
    can log what it had): ``underlying_target_usd`` None or non-finite →
    ``contract_target_usdt`` is None plus the matching
    ``missing_inputs`` entry; ``usdt_usd`` None → the same; ``usdt_usd``
    that is non-finite, ``<= 0``, or outside [0.9, 1.1] →
    ``"fx_out_of_band"`` (and contract None). The
    chain is emitted with full-precision operands rendered to fixed
    decimals — the STORED target is never rounded.
    """
    missing: list[str] = []
    underlying_unusable = underlying_target_usd is None or not math.isfinite(
        underlying_target_usd
    )
    if underlying_unusable:
        missing.append(MISSING_UNDERLYING)
    fx_out_of_band = False
    if usdt_usd is None:
        missing.append(MISSING_FX)
    elif (
        not math.isfinite(usdt_usd)
        or usdt_usd <= 0.0
        or usdt_usd < _FX_BAND_LOW
        or usdt_usd > _FX_BAND_HIGH
    ):
        missing.append(MISSING_FX_OUT_OF_BAND)
        fx_out_of_band = True
    if current_basis is None:
        missing.append(MISSING_CURRENT_BASIS)

    computable = (
        underlying_target_usd is not None
        and not underlying_unusable
        and usdt_usd is not None
        and not fx_out_of_band
    )
    if not computable:
        return FairValueResult(
            underlying_target_usd=underlying_target_usd,
            usdt_usd=usdt_usd,
            current_basis=current_basis,
            expected_basis=expected_basis,
            contract_target_usdt=None,
            missing_inputs=tuple(missing),
            chain=(),
        )

    assert underlying_target_usd is not None and usdt_usd is not None  # narrowed above
    contract_target = underlying_target_usd / usdt_usd * (1.0 + expected_basis)
    chain = (
        f"underlying target {underlying_target_usd:.2f} USD",
        f"÷ USDT/USD {usdt_usd:.6f}",
        f"× (1 + expected basis {expected_basis:.6f})",
        f"= contract target {contract_target:.6f} USDT",
    )
    return FairValueResult(
        underlying_target_usd=underlying_target_usd,
        usdt_usd=usdt_usd,
        current_basis=current_basis,
        expected_basis=expected_basis,
        contract_target_usdt=contract_target,
        missing_inputs=tuple(missing),
        chain=chain,
    )


def render_bridge_block(result: FairValueResult) -> str:
    """Markdown disclosure block for the bridge (report/decision section).

    Heading, then the chain steps as bullets (only when the target was
    computed), then one ``⚠``-prefixed warning line per missing input. The
    computed target is only ever shown via the chain's fixed-decimal
    rendering — the stored float keeps full precision and is not re-rounded
    here.
    """
    lines = [BRIDGE_HEADING]
    lines.extend(f"- {step}" for step in result.chain)
    for key in result.missing_inputs:
        explanation = _MISSING_EXPLANATIONS.get(key, "required input missing")
        lines.append(f"- ⚠ missing input: {key} ({explanation})")
    return "\n".join(lines)
