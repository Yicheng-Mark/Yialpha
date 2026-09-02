"""V2.2 Context stage: the versioned point-in-time ``RegimeState``.

One frozen classifier vocabulary (``state.RegimeState``), one deterministic
assembly pass over the existing PIT data seams (``compute.compute_regime_state``),
and one content-addressed regime id that ties predictions / outcomes / tickets
to the exact market context they were decided under. Everything here is
record-stage: it describes and discloses, it never vetoes.
"""
