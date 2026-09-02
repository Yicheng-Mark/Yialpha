"""V2 ledgers: evidence, blind predictions, outcomes, tickets, instruments.

Package layout (V2.1 Measurability, docs/V2_BASELINE.md):

* :mod:`yialpha.ledger.sqlite` — the single central append-first SQLite
  database (connection/migration seam shared by every ledger).
* :mod:`yialpha.ledger.models` — record dataclasses + ID helpers.
* :mod:`yialpha.ledger.evidence` — run evidence records ([EXTERNAL EVIDENCE]
  injections become auditable, replayability-tagged rows).
* :mod:`yialpha.ledger.predictions` — immutable blind analyst predictions
  (submitted pre-debate via the submit_prediction tool).
* :mod:`yialpha.ledger.outcomes` — forward outcome rows + net-return
  attribution legs (price / funding / basis / fees / slippage).
"""
