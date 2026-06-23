# Pipeline MI — Single Source of Truth Review (Phase 1)

> Required architecture/field review (Part 0) completed **before** implementation.
> This documents the existing repo state, what is reused, and what is genuinely
> new, so Pipeline MI extends the funded MI architecture rather than forking it.

## TL;DR

The repo **already** has the raw pipeline spine and the forecast vocabulary:

| Concern | Funded book | Pipeline (pre-existing) | Pipeline (added this PR) |
| --- | --- | --- | --- |
| Governed raw tape | `18_central_lender_tape.csv` | `18a_central_pipeline_tape.csv` (built by `central_tape_builder`) | — |
| Prepared MI dataset | `funded_prep.prepare_funded_mi_dataset` | *(none)* | **`pipeline_prep.prepare_pipeline_mi_dataset` → `20_prepared_pipeline_mi.csv`** |
| Dataset contract | `mi_dataset_contract.build_dataset_contract` | *(none)* | **`pipeline_contract.build_pipeline_dataset_contract`** |
| Snapshot + API | `snapshots.compute_funded_snapshot`, `GET /mi/snapshot` | *(none)* | **`pipeline_contract.compute_pipeline_snapshot`, `GET /mi/pipeline/snapshot`** |
| Forecast assembly | — | `states.assembler.total_forecast_funded` (formula already implemented) | forecast-**ready** dataset wired to it |

So Phase 1 is the **prepared pipeline MI dataset + dataset contract + API
metadata** layer that was missing — built by extending the funded patterns and
the existing config/registry/state layers. No parallel architecture, no NL
parser changes, no scenario-query logic.

---

## 1. M2L KFI / pipeline files in the fixture pack

The task's referenced path did not previously exist; it is created in this PR:

```
tests/fixtures/client_001_mi_pack/pipeline/2025-10-01/M2L_KFI_and_Pipeline_2025_10_01.csv   (8 cases)
tests/fixtures/client_001_mi_pack/pipeline/2025-11-01/M2L_KFI_and_Pipeline_2025_11_01.csv   (10 cases)
```

The schema reuses the realistic committed header from
`tests/fixtures/kfi_pipeline_headers.csv` (already used by
`tests/test_onboarding_kfi_pipeline_mapping.py`). CSV is used (not `.xlsx`) to
match the funded `18_central_lender_tape.csv` convention and to keep tests
runnable without the optional `openpyxl` dependency; the reader
(`pipeline_contract._read_source`) handles both `.csv` and `.xlsx`.

The Nov snapshot shows realistic stage progression vs Oct (e.g. ACC1001
Offer→Completed, ACC1003 KFI→Application) plus new cases, so it exercises
month-on-month pipeline movement.

### Sheets / columns present

One sheet (flat extract). Columns:

`Company, Pool, Account Number, KFI Number, Broker, KFI Submitted Date, DOB App 1,
Gender APP 1, DOB App 2, Gender APP 2, Loan Amount, Estimated Value, Product,
Product Rate, Loan Plan, Facility, Max Facility, Max Entitlement, Property Region,
PEG Percentage, Fees Added, Property Value, Loan Purpose, Loan Purpose Detail,
Status, DPR Status, Application Submitted Date, Offer Date, Date Funds Released,
Rejection Reason A/B, KFI Used For App, Contracted Payment Period,
Interest Payment Percentage`.

## 2. Existing semantic registry / config coverage

- `config/system/fields_registry.yaml` (5,741 lines) — canonical field registry.
- `mi_agent/mi_semantics_field_registry.yaml` — MI semantic registry. **Already
  contains pipeline/forecast fields**: `pipeline_stage`, `pipeline_snapshot_date`,
  `funded_status`, `forecast_funded_balance`, `forecast_funding_date`,
  `forecast_funding_probability`.
- `config/system/aliases_onboarding_kfi.yaml` — audited KFI aliases:
  `product rate → current_interest_rate`, `estimated value →
  current_valuation_amount`, `property region → collateral_geography`.
- `config/mi/buckets.yaml` — the single bucket engine config (ltv_bucket,
  age_bucket, ticket_bucket/balance_band, interest_rate_bucket, …).
- `config/client/pipeline_expected_funding.yaml` — **stage→probability** and
  **stage→days-to-fund** assumptions (`KFI 0.20 / APPLICATION 0.45 / OFFER 0.75 /
  COMPLETED 1.00`). Probabilities are read from here, never invented.

## 3. Funded ↔ pipeline field correlation

| Pipeline concept | M2L KFI source | Canonical (reused funded name) | Funded correlation |
| --- | --- | --- | --- |
| Loan / advance amount | `Loan Amount` | `current_outstanding_balance` | `current_outstanding_balance`, `original_principal_balance` |
| Valuation | `Estimated Value` / `Property Value` | `current_valuation_amount` | `current_valuation_amount`, `original_valuation_amount` |
| LTV | derived (amount/value) | `current_loan_to_value` | `current_loan_to_value`, `original_loan_to_value` |
| Rate | `Product Rate` | `current_interest_rate` | `current_interest_rate` |
| Borrower age | derived from `DOB App 1/2` | `youngest_borrower_age` | `youngest_borrower_age` |
| Region | `Property Region` | `collateral_geography` (→ `geographic_region_obligor`) | `geographic_region_obligor`, `geographic_region_collateral` |
| Broker / channel | `Broker` | `broker_channel` (→ `origination_channel`) | `broker_channel`, `origination_channel` |
| Product | `Product` | `product_type` | `erm_product_type`, `erm_sub_product_type` |

Storing economic fields under the funded canonical names is deliberate: it lets
the **same** bucket engine (`analytics_lib.buckets`), dataset profile
(`mi_dataset_profile`) and contract builder run on the pipeline frame, and makes
funded↔pipeline directly comparable for forecasting.

## 4. Pipeline-specific fields (funnel + timing + forecast inputs)

`pipeline_stage` (normalised KFI/APPLICATION/OFFER/COMPLETED/WITHDRAWN),
`pipeline_status` (funded/pipeline/withdrawn — funded-status vocab),
`pipeline_stage_date`, `kfi_date`, `application_date`, `offer_date`,
`expected_completion_date` (Date Funds Released, else derived from
stage_days_to_fund), `completion_probability` + `stage_conversion_probability`
(config stage probability), `expected_funded_amount`,
`weighted_expected_funded_amount`, `pipeline_case_age_days`,
`days_to_expected_completion`, `pipeline_source_file`, `pipeline_reporting_date`;
plus derived dimensions `expected_completion_month`, `pipeline_stage_bucket`.

## 5. Field classification

- **Already in the semantic registry:** `pipeline_stage`, `pipeline_snapshot_date`
  (↔ `pipeline_reporting_date`), `funded_status` (↔ `pipeline_status`),
  `forecast_funded_balance`/`_date`/`_probability` (↔ `expected_funded_amount` /
  `expected_completion_date` / `completion_probability`), and every reused funded
  economic field.
- **Aliases only (already audited):** `product rate`, `estimated value`,
  `property region` (in `aliases_onboarding_kfi.yaml`). The pipeline field
  contract adds the remaining M2L KFI synonyms in
  `config/mi/pipeline_field_contract.yaml`.
- **Genuinely new canonical fields:** none required at the registry level — the
  pipeline-specific names are declared in the pipeline field contract and the
  forecast state vocabulary already exists. `application/offer/KFI date` remain
  `registry_target_missing` in the onboarding parity audit (documented in the
  alias file) and are captured by the pipeline contract rather than forced into
  the regulatory registry.

## 6. Reused funded functions / contracts

- `analytics_lib.buckets.materialise_buckets` + `config/mi/buckets.yaml` (ltv /
  age / ticket / interest-rate buckets) — identical to funded prep.
- `analytics_lib.numeric.coerce_numeric` — same deterministic amount parser.
- `mi_agent_api.mi_dataset_contract.build_dataset_contract` +
  `mi_agent.mi_dataset_profile.profile_dataset` — per-field contract.
- `mi_agent.states.models` stage/status vocabularies; `mi_agent.states.forecast`
  / `states.assembler.total_forecast_funded` for the forecast layer.
- `config/client/pipeline_expected_funding.yaml` for stage assumptions.

## 7. What stays separate from the funded book

The prepared pipeline MI dataset is a **separate artefact**
(`20_prepared_pipeline_mi.csv`); every row carries `record_type == "pipeline"`.
`pipeline_prep` reads **only** pipeline/KFI sources and never the funded central
lender tape, and the funded prep carries no pipeline columns. Pipeline rows are
never merged into `18_central_lender_tape.csv` (the onboarding builder already
enforces this; the existing
`test_funded_realistic_and_pipeline.TestPipelineOnlyEnrichment` continues to pass).

## 8. How pipeline connects to funded for later forecast

The pipeline dataset is **forecast-ready** (metadata in
`config/mi/pipeline_field_contract.yaml → forecast_readiness` and surfaced on
`GET /mi/pipeline/snapshot → forecastReadiness`):

```
forecast_funded_balance
    = current_funded_balance
    + sum(pipeline_expected_funded_amount * completion_probability)
```

- economic amount field: `expected_funded_amount`
- baseline probability field: `completion_probability` (config stage lookup)
- expected completion: `expected_completion_date` / `expected_completion_month`
- correlation axes: amount, LTV, valuation, region, broker/channel, rate,
  borrower age, product
- unique-key candidates: `pipeline_case_identifier`, `application_identifier`,
  `linked_loan_identifier` (18a tape link), borrower/property match fields.

The assembler `mi_agent.states.assembler.total_forecast_funded` already
implements this exact formula. **Scenario-query logic (e.g. "if completion rate
+5%") is intentionally NOT built in this PR.**
