# Funded MI data path — central tape → MI-prepared dataset

Generated from `mi_agent_api/funded_prep.prepare_funded_mi_dataset` over a promoted
funded central lender tape that includes config-driven MI enrichment fields
(`central_lender_tape.mi_enrichment_fields`).

## 1. Fields present in 18_central_lender_tape.csv (after enrichment)
- `loan_identifier`
- `current_outstanding_balance`
- `current_valuation_amount`
- `original_principal_balance`
- `original_valuation_amount`
- `current_interest_rate`
- `youngest_borrower_age`
- `geographic_region_obligor`
- `origination_channel`
- `broker_channel`
- `origination_date`
- `reporting_date`
- `data_cut_off_date`
- `exposure_currency_denomination`

## 2. Fields ADDED by MI preparation (analytics_lib.buckets + config/mi/buckets.yaml)
Derived: `current_loan_to_value`, `original_loan_to_value`, `vintage_year`, `months_on_book`

LTV derivation basis (product rule — derive when no raw LTV column):
- `current_loan_to_value`: method=`derived_ratio` "" num=`current_outstanding_balance` den=`current_valuation_amount` conf=0.9
- `original_loan_to_value`: method=`derived_ratio` num=`original_principal_balance` den=`original_valuation_amount` conf=0.9

Bucket dimensions materialised:
- `ltv_bucket` ← `ltv_bucket`
- `original_ltv_bucket` ← `original_ltv_bucket`
- `age_bucket` ← `borrower_age_bucket`
- `age_bucket` ← `youngest_borrower_age_bucket`
- `interest_rate_bucket` ← `interest_rate_bucket`
- `ticket_bucket` ← `balance_band`
- `time_on_book_bucket` ← `time_on_book_bucket`

Dimensions available: `age_bucket`, `geographic_region_obligor`, `interest_rate_bucket`, `ltv_bucket`, `original_ltv_bucket`, `origination_channel`, `ticket_bucket`, `time_on_book_bucket`, `vintage_year`

## 3. Dimensions still unavailable (reason-coded)
- (none — all core funded dimensions available)

## 4. Raw vs prepared
- `MI_AGENT_ANALYTICS_DATASET` → served as-is (`funded_mi_prepared_dataset`).
- `MI_AGENT_CENTRAL_TAPE` / output-root+client/run → prepared by default.
- `MI_AGENT_DISABLE_PREP=1` → raw thin tape (`funded_central_lender_tape_raw`).
- LTV is DERIVED from balance/valuation (current) and original balance/valuation
  (original) when no raw LTV column exists; an explicit, valid source LTV is preferred.
  Missing valuation → `derivation_inputs_missing` (no misleading LTV).

See `funded_mi_missing_dimension_trace.md` for the per-field raw→React trace.
