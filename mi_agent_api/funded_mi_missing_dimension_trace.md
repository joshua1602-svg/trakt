# Funded MI — missing-dimension trace (client_001/mi_2026_01)

**Framing.** MI availability is decided by the active MI target contract + MI enrichment configuration (`central_lender_tape.mi_enrichment_fields`) and the source fields actually present — NOT by the registry category/layer. A field that is regulatory/collateral in the registry can still be an MI dimension (this is MI contract enrichment using source fields that may also be relevant to regulatory reporting — not contract leakage).

raw source → mapping → MI contract/scope → period eligibility → entity-key join → central tape → MI prep → React health.

| canonical_field | dimension | source file:col | period_eligible | in_enrich_cfg | in_tape (non-null) | dim_available | status | reason |
|---|---|---|---|---|---|---|---|---|
| `youngest_borrower_age` | `age_bucket` | :— | None | True | False (0) | False | **unavailable** | `raw_not_found` |
| `geographic_region_obligor` | `geographic_region_obligor` | :— | None | True | False (0) | True | **available** | `dimension_available` |
| `collateral_geography` | `geographic_region_obligor` | monthly_collateral_report.csv:collateral_region | True | True | True (8) | True | **available** | `promoted_and_dimension_available` |
| `current_valuation_amount` | `ltv_bucket` | :— | None | True | True (8) | True | **available** | `promoted_and_dimension_available` |
| `current_loan_to_value` | `ltv_bucket` | :— | None | True | True (8) | True | **available** | `promoted_and_dimension_available` |
| `original_valuation_amount` | `original_ltv_bucket` | :— | None | True | False (0) | False | **unavailable** | `raw_not_found` |
| `original_principal_balance` | `original_ltv_bucket` | monthly_loan_report.csv:original_principal_balance | True | True | True (8) | False | **promoted** | `in_central_tape` |
| `original_loan_to_value` | `original_ltv_bucket` | :— | None | True | False (0) | False | **unavailable** | `derivation_inputs_missing` |
| `origination_channel` | `origination_channel` | :— | None | True | False (0) | False | **unavailable** | `raw_not_found` |
| `broker_channel` | `origination_channel` | :— | None | True | False (0) | False | **unavailable** | `raw_not_found` |

## Reason codes
- `raw_not_found` — no source column maps to the canonical field.
- `mapped_but_out_of_scope` — mapped but excluded by registry scope and not in `central_lender_tape.mi_enrichment_fields`.
- `source_period_ineligible` — the mapping source is not period-eligible for the run.
- `join_failed` — eligible + in enrichment config but did not join the funded universe.
- `not_in_central_tape` — absent from the promoted tape for another reason.
- `derivation_inputs_missing` — an LTV bucket whose balance/valuation inputs are absent.
- `promoted` / `available` — field reached the tape / its dimension is prepared.
