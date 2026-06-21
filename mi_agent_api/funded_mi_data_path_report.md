# Funded MI data path — central tape → MI-prepared dataset

Generated from `mi_agent_api/funded_prep.prepare_funded_mi_dataset` over a
promoted funded central lender tape.

## 1. Fields present in `18_central_lender_tape.csv` (canonical funded tape)
- `loan_identifier`
- `current_outstanding_balance`
- `current_valuation_amount`
- `current_interest_rate`
- `current_principal_balance`
- `origination_date`
- `reporting_date`
- `data_cut_off_date`
- `exposure_currency_denomination`

## 2. Fields ADDED by MI preparation (reusing analytics_lib.buckets + config/mi/buckets.yaml)
Derived source fields: `current_loan_to_value`, `vintage_year`, `months_on_book`
Bucket dimensions materialised:
- `ltv_bucket`  ← `ltv_bucket`
- `interest_rate_bucket`  ← `interest_rate_bucket`
- `ticket_bucket`  ← `balance_band`
- `time_on_book_bucket`  ← `time_on_book_bucket`

Dimensions available for stratification: `interest_rate_bucket`, `ltv_bucket`, `ticket_bucket`, `time_on_book_bucket`, `vintage_year`

## 3. Dimensions still UNAVAILABLE (and why)
- `age_bucket` — no youngest_borrower_age in funded tape
- `original_ltv_bucket` — no original_principal_balance / original_valuation_amount
- `geographic_region_obligor` — no obligor/collateral geography (NUTS/ITL) field
- `origination_channel` — no broker/channel field in funded tape

## 4. Raw vs prepared
- `MI_AGENT_ANALYTICS_DATASET` → served as-is (kind `funded_mi_prepared_dataset`).
- `MI_AGENT_CENTRAL_TAPE` / output-root+client/run → **prepared** by default
  (kind `funded_mi_prepared_dataset`, `preparationApplied: true`).
- `MI_AGENT_DISABLE_PREP=1` → raw thin tape, KPI-only (kind
  `funded_central_lender_tape_raw`, `preparationApplied: false`).
- React consumes whatever `/mi/query` returns; with prep ON it renders funded KPIs
  **and** LTV / rate / ticket / time-on-book stratifications.
