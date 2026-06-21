# Funded MI — missing-dimension trace (client_001/mi_2025_10 (rich pack))

raw source → mapping → period eligibility → central tape → MI prep → React.

| canonical_field | dimension | source file:col | period_eligible | in_enrich_cfg | in_tape (non-null) | dim_available | status | reason |
|---|---|---|---|---|---|---|---|---|
| `youngest_borrower_age` | `age_bucket` | LoanExtract One.csv:Youngest Age | True | True | True (33) | True | **available** | `promoted_and_dimension_available` |
| `geographic_region_obligor` | `geographic_region_obligor` | :— | None | True | True (33) | True | **available** | `promoted_and_dimension_available` |
| `collateral_geography` | `geographic_region_obligor` | Collateral Extract.csv:property region | True | True | True (33) | True | **available** | `promoted_and_dimension_available` |
| `current_valuation_amount` | `ltv_bucket` | :— | None | True | True (33) | True | **available** | `promoted_and_dimension_available` |
| `current_loan_to_value` | `ltv_bucket` | :— | None | True | False (0) | True | **available** | `dimension_available_via_derivation` |
| `original_valuation_amount` | `original_ltv_bucket` | :— | None | True | True (33) | True | **available** | `promoted_and_dimension_available` |
| `original_principal_balance` | `original_ltv_bucket` | LoanExtract One.csv:Original Principal Balance | True | True | True (33) | True | **available** | `promoted_and_dimension_available` |
| `original_loan_to_value` | `original_ltv_bucket` | :— | None | True | False (0) | True | **available** | `dimension_available_via_derivation` |
| `origination_channel` | `origination_channel` | :— | None | True | True (33) | True | **available** | `promoted_and_dimension_available` |
| `broker_channel` | `origination_channel` | LoanExtract One.csv:broker | True | True | True (33) | True | **available** | `promoted_and_dimension_available` |

## Reason codes
- `raw_not_found` — no source column maps to the canonical field.
- `mapped_but_out_of_scope` — mapped but excluded by registry scope and not in `central_lender_tape.mi_enrichment_fields`.
- `source_period_ineligible` — the mapping source is not period-eligible for the run.
- `join_failed` — eligible + in enrichment config but did not join the funded universe.
- `not_in_central_tape` — absent from the promoted tape for another reason.
- `derivation_inputs_missing` — an LTV bucket whose balance/valuation inputs are absent.
- `promoted` / `available` — field reached the tape / its dimension is prepared.
