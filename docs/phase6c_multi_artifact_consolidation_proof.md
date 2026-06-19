# Phase 6C — Multi-Artefact Consolidation Proof (MI runtime)

**Status:** Implemented (synthetic fixture + deterministic consolidation helper +
lineage + tests + docs). Deterministic synthetic proof only — **not** a
production consolidation engine. No onboarding orchestration, no Azure, no Event
Grid/blob ingestion, no Streamlit migration, no M&A Agent runtime, no Annex
2/regulatory changes, no new chart types, no LLM.

**Dependency order:** Phase 0 → 0B → 1 → 2 → 3 → 4 → 5 → 6 → 6B → **6C**.

**Date:** 2026-06-18

---

## 1. What this proof demonstrates

That Trakt can take **fragmented source artefacts**, deterministically
**consolidate** them into canonical MI snapshot frames (one per reporting date),
**register** them in `LocalFsSnapshotStore`, and then run the **same governed MI
runtime** (`run_mi_query`) over the consolidated history — funded/pipeline/
forecast states, temporal compare & trend, stratification/concentration, and
risk migration — with the flat single-CSV path still working unchanged.

The snapshot layer is never bypassed: every state/temporal/risk query resolves
and loads through the store.

## 2. Artefacts included

Synthetic fixtures under `tests/fixtures/phase6c_multi_artifact/`, across three
reporting dates (`2024-01-31`, `2024-02-29`, `2024-03-31`):

| Artefact | Key | Contributes |
|---|---|---|
| `borrowers.csv` | `borrower_id` | borrower_age, borrower_structure, region (obligor) |
| `loans.csv` | `loan_id` (+ `borrower_id`, `reporting_date`) | balance, interest_rate, funded_status, product_type, amortisation_type, origination/funding dates, internal_risk_grade, ifrs9_stage, pd_bucket |
| `collateral.csv` | `loan_id` | property_value, current_ltv, property_region (+ an orphan `F9` to exercise unmatched detection) |
| `cashflows.csv` | `loan_id` + `reporting_date` | interest/principal due & paid, arrears_balance, arrears_status |
| `portfolio_map.csv` | `loan_id` | portfolio_id, portfolio_name, spv_id, acquired_portfolio_id |
| `pipeline.csv` | `opportunity_id` (+ `borrower_id`, `reporting_date`) | expected_balance, pipeline_stage, forecast_funding_probability, broker_channel, origination_channel, product_type, interest_rate |

## 3. How they consolidate into canonical MI snapshots

`tests/helpers/phase6c_consolidation.py` (loaded by file path in the test) joins,
per reporting date:
- `borrowers → loans` on `borrower_id`;
- `loans → collateral` on `loan_id`;
- `loans → cashflows` on `loan_id + reporting_date`;
- `loans → portfolio_map` on `loan_id`;
- `pipeline` opportunities kept in a **distinct namespace** (`opportunity_id`,
  never merged into funded `loan_id`); pipeline `borrower_id` joins borrower
  fields where available.

Funded and pipeline rows are concatenated into **one canonical frame per
reporting date** with canonical column names the runtime expects
(`current_outstanding_balance`, `current_interest_rate`,
`current_loan_to_value`, `geographic_region_obligor`, `erm_product_type`,
`youngest_borrower_age`, `funded_status`, `pipeline_stage`,
`forecast_funding_probability`, `forecast_funded_balance` (derived =
expected_balance × probability), `internal_risk_grade`, `ifrs9_stage`,
`pd_bucket`, `arrears_status`, `arrears_balance`, `portfolio_id`/`_name`,
`spv_id`, `acquired_portfolio_id`, `months_on_book` (derived), plus
`loan_id`/`opportunity_id`/`source_record_id`/`stable_entity_id`). Frames are
registered through `LocalFsSnapshotStore.register_snapshot` with explicit
`reporting_date`/`cut_off_date`/`upload_timestamp`.

This join logic is **hard-wired for the synthetic fixture** — there is no schema
inference, no mapping config, and no generic mapper.

## 4. Lineage metadata captured

A static `LINEAGE` dict records the source artefact for each key consolidated
field, e.g. `current_outstanding_balance ← loans.csv/pipeline.csv`,
`current_loan_to_value ← collateral.csv`, `geographic_region_obligor ←
borrowers.csv`, `arrears_status ← cashflows.csv`, `portfolio_id ←
portfolio_map.csv`, `pipeline_stage ← pipeline.csv`, `forecast_funded_balance ←
derived`. Tests assert lineage exists for the key fields. This is a simple dict,
not a production lineage engine.

## 5. MI runtime questions proven (via `run_mi_query`)

- **States (latest):** `total_funded` (3 rows / 620k), `total_pipeline` (1 / 50k),
  `total_forecast_funded` (645k).
- **Temporal:** funded compare 300k→620k (new 1 / retained 2); funded trend
  [300k, 620k, 620k]; pipeline trend [90k, 90k, 50k]; forecast-funded trend
  [335k, 663k, 645k].
- **Stratification / concentration:** funded by `portfolio_id` (PF_001=400k,
  PF_002=220k); funded by region (North=400k, South=220k); pipeline by stage
  (OFFER/APPLICATION at the S2 cut); forecast-funded by region (North=425k);
  concentration warning by region (North 64.5% → `status=red`).
- **Risk migration:** grade B→C, IFRS 9 Stage 1→Stage 2, PD bucket — all
  `deteriorated` (loan F2 across baseline→current).
- **Backward compatibility:** the existing flat single-CSV MI query still works.

## 6. Structured issues

`missing_required_artifact`, `missing_optional_artifact`, `missing_join_key`,
`unmatched_artifact_rows` (the orphan `F9` collateral row is flagged),
`duplicate_join_key`, `missing_lineage_for_field`,
`missing_optional_consolidated_field`, `snapshot_registration_failed`,
`multi_artifact_consolidation_warning`. Optional artefact gaps are flagged and do
**not** crash: a directory with only the two required artefacts still
consolidates into three snapshots, with the missing optional artefacts recorded
as warnings.

## 7. What remains unproven / deferred

- Real-world schema drift, column-name mapping, and multi-file fan-in per
  artefact type (a production mapper) — out of scope.
- Onboarding orchestration that *produces* these artefacts and Trakt portfolio
  references — deferred.
- Idempotent re-consolidation, reconciliation of pipeline→funded conversion, and
  cross-snapshot entity stitching beyond the stable `loan_id`/`opportunity_id`
  namespaces.
- Non-flat chart rendering (the runtime returns a governed `chart_instruction`).

## 8. Why this is not a production consolidation engine

The joins, field mappings, and lineage are **hard-coded for the synthetic
fixture** to prove the end-to-end path deterministically. There is no
configuration-driven mapping, no source-schema discovery, no validation of
arbitrary client tapes, and no performance/scale handling. It is a proof that the
*shape* of consolidation → snapshot → governed MI runtime works.

## 9. How this prepares future onboarding/orchestration work

It fixes the **target canonical contract** the future onboarding/consolidation
engine must emit (the canonical column set + reserved snapshot fields the
runtime consumes), demonstrates the **artefact → join → lineage → snapshot**
flow, and proves the governed MI runtime is agnostic to *how* the snapshot was
assembled. A production engine can replace the hard-wired helper while keeping
the same `SnapshotStore` registration boundary and `run_mi_query` surface.

## 10. Files changed

- `tests/fixtures/phase6c_multi_artifact/{borrowers,loans,collateral,cashflows,portfolio_map,pipeline}.csv` (new fixtures)
- `tests/helpers/phase6c_consolidation.py` (new helper)
- `tests/test_phase6c_multi_artifact_consolidation.py` (new, 28 tests)
- `docs/phase6c_multi_artifact_consolidation_proof.md` (this file)

## 11. Tests run

`tests/test_phase6c_multi_artifact_consolidation.py` — **28 passed**. MI Agent +
Phase 6/6B/6C suites — **226 passed**.

## 12. Limitations / deferred items

- Production consolidation engine, onboarding orchestration, Azure, Streamlit,
  M&A Agent runtime, and UI remain future phases.
- Forecast-funded-by-broker is not asserted (funded rows carry no broker in the
  source artefacts); forecast-funded-by-region is proven instead.
