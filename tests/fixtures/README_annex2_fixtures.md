# The Annex 2 delivery fixtures

Three committed files, one governed run behind all of them.

| File | What it is |
|---|---|
| `annex2_projected_client_b.csv` | The Gate 4 projection from the real Client B rehearsal — 30 exposures, 107 Annex 2 columns — **exactly as the projector wrote it**, including the five account-status words the lender uses ("Live", "In possession", "Moved to LTC", "Probate - awaiting sale", "Redeemed"). |
| `annex2_effective_contract_client_b.yaml` | The effective Annex 2 contract that same run materialised: the contract derived from the authoritative sources, plus that portfolio's approved operator decisions — among them the five account-status translations. This is what OCC passed to Gate 4b as `--rules`. |
| `annex2_projected_ci.csv` | The same projection with those five approved translations already applied to `RREL69`, so it is complete against the **generic** derived contract and needs no run-scoped contract to prepare. |

## Why the third file exists

Gate 4b can receive a client's approved enum translations two ways: at
projection, through the client configuration's `enum_overrides`, or at delivery
preparation, through the run's materialised contract. Client B's
`amortisation_type` went the first way and its `account_status` went the second,
so its raw projection still carries lender words and only makes sense **paired**
with its own effective contract.

The delivery agent's HTTP routes deliberately do not let a caller inject a
contract, so a fixture used through them has to stand on its own. `..._ci.csv`
is that fixture: the same governed rows, with the same approved translations,
resolved at the earlier of the two points. Nothing is invented — every RREL69
value in it is the target of an operator decision recorded in
`annex2_effective_contract_client_b.yaml`.

## What they prove

Prepared through the delivery agent, `annex2_projected_ci.csv` reaches
`PREPARED_WITH_WARNINGS` with **XSD PASSED, zero blocking errors**, 30 records
and 104 delivered fields — the three unreported codes being those auth.099
carries as a currency attribute of the amount they qualify rather than as an
element of their own (RREC22, RREL18, RREL28).
