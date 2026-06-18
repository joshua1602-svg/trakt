# Phase 3 — MI State Assembler Foundations

**Status:** Implemented (pure assembler + selectors + route-eligibility + tests +
docs). No orchestration, no MI Agent runtime wiring, no LLM query routing, no M&A
agent, no risk-monitor runtime, no temporal-trend/migration runtime, no Azure, no
Streamlit/chart migration, no legacy `analytics/` imports, no Annex 2/regulatory
changes.

**Dependency order:** Phase 0 → Phase 0B → Phase 1 → Phase 2 → **Phase 3**.

**Date:** 2026-06-18

This phase turns the declarative Phase 0B state library into deterministic,
pure **state-assembly functions** that consume Phase 2 `SnapshotStore` outputs
(or already-loaded DataFrames) and the Phase 1 `analytics_lib`, and produce
analytical DataFrames (`StateResult`) ready for later MI queries.

---

## 1. What was built

A new `mi_agent/states/` package:

| File | Responsibility |
|---|---|
| `models.py` | `StateResult` dataclass, the structured-issue model + codes, and the funded/pipeline value vocabularies (`classify_funded_value`). |
| `selectors.py` | `SnapshotSelector` (latest / as_of / range / compare) delegating to the Phase 2 `SnapshotStore` resolvers. |
| `route_contracts.py` | Lightweight, testable `validate_state_for_route` / `is_state_allowed` reading `config/routes/<route>_route.yaml` + alias resolution. |
| `assembler.py` | The deterministic state functions + an `assemble_state(name, source, …)` dispatcher. |
| `__init__.py` | Curated public API. |

### 1.1 Core states (from `config/mi/state_library.yaml`)
- **`total_funded`** — funded book at a selected snapshot. Uses `funded_status`
  where present; else derives from `pipeline_stage`; else (neither present)
  applies the documented v1 fallback "the loaded frame is the funded book",
  recording a `missing_funded_status` issue. Records the selection method.
- **`total_pipeline`** — in-pipeline records. Uses `funded_status` /
  `pipeline_stage`; **never** falls back to "all rows are pipeline" (unsafe) —
  with neither field it returns an empty frame plus a `missing_pipeline_stage`
  issue.
- **`total_forecast_funded`** — funded book + expected-converted pipeline. Per
  pipeline row the expected funded balance is `forecast_funded_balance` where
  populated, else `current_outstanding_balance × forecast_funding_probability`
  where both exist. **Probabilities are never invented.** Rows with no usable
  forecast are flagged (`missing_forecast_probability`) and, by default,
  retained with a null `forecast_contribution` (`include_unforecastable=False`
  drops them). Output tags each row `state_component` (`funded` /
  `forecast_pipeline`) with a numeric `forecast_contribution`.
- **Cohort states** — `cohort_by_date` (parameterised by `date_field`:
  `origination_date` / `funding_date` / `acquisition_date`), plus segmentation
  wrappers `cohort_by_portfolio` / `cohort_by_spv` /
  `cohort_by_acquired_portfolio`. The descriptive names
  `cohort_by_origination_date` / `cohort_by_funding_date` /
  `cohort_by_acquisition_date` are accepted via `assemble_state` and map to
  `cohort_by_date` with the right `date_field`. Cohorts add a
  `<date_field>_cohort` column, optional `months_on_book` (against a
  reporting/as-of date), an optional segmentation-presence check, and optional
  bucket materialisation — all **frame-level**, suitable for later
  stratification (no aggregation/charting here).

### 1.2 Selectors
`SnapshotSelector` resolves **which** snapshot(s) to assemble from:
`latest` / `as_of` (single, used by the Phase 3 point-in-time states) and
`range` / `compare` (provided to *prepare* Phase 4 — the trend/migration runtime
is **not** built here). Single resolution uses
`SnapshotStore.resolve_latest` / `resolve_as_of`.

### 1.3 Route eligibility (no orchestration)
`validate_state_for_route(state, route)` reads the Phase 0B route contract and
returns an `unsupported_state_for_route` issue when a state is not in the
route's `allowed_states` (aliases resolved to canonical first). Assembler
functions accept a `route=` argument; when supplied and the state is disallowed,
they return early with `metadata["assembled"] = False` and the issue — they do
**not** assemble. This is a pure config read + membership check, not a runtime
route resolver.

### 1.4 Structured issues
`StateResult.issues` uses the shared convention (`code` / `severity` /
`message` / `field`). Codes: `missing_required_state_field`,
`missing_optional_state_field`, `unavailable_state`, `unavailable_dimension`,
`missing_snapshot`, `empty_state_frame`, `missing_forecast_probability`,
`missing_balance_field`, `missing_funded_status`, `missing_pipeline_stage`,
`unsupported_state_for_route`, `invalid_date`. Optional-field gaps never crash.

---

## 2. What was intentionally NOT built

- No orchestration / MI Agent runtime wiring; nothing calls the assembler from
  the live agent.
- No LLM query routing, no M&A agent, no risk-monitor runtime.
- No temporal MI **trend** runtime and no snapshot-to-snapshot **migration**
  (Phase 4 / Phase 5); `selectors` expose `range`/`compare` only to prepare.
- No Azure, no Streamlit/chart output, no legacy `analytics/` imports, no
  Annex 2/regulatory changes.

---

## 3. How it consumes earlier phases

- **Phase 0B (route/state configs):** `route_contracts.py` reads
  `config/routes/*_route.yaml` `allowed_states`; the canonical state names match
  `config/mi/state_library.yaml`. The reserved column names match the Phase 0B
  virtual semantic fields (`funded_status`, `pipeline_stage`,
  `forecast_funding_probability`, `forecast_funded_balance`, segmentation ids,
  event dates).
- **Phase 1 (`analytics_lib`):** cohort derivation uses
  `analytics_lib.cohort.add_cohort_period` / `months_on_book`; optional bucket
  materialisation uses `analytics_lib.materialise_buckets`. (`stratify` is
  available for downstream callers; Phase 3 stays frame-level and does not
  aggregate.)
- **Phase 2 (`SnapshotStore`):** `selectors.py` calls the resolvers; the
  assembler loads loan frames via `load_loans`, and a missing snapshot becomes a
  `missing_snapshot` issue rather than an exception.

---

## 4. How this prepares Phase 4 (temporal MI/trends) and Phase 5 (risk monitor)

- **Phase 4 (temporal/trends):** `SnapshotSelector` already models `range` and
  `compare`; Phase 4 will iterate a selector's resolved snapshots, assemble each
  via these same pure functions, and compute period-over-period deltas / cohort
  trends — the assembler needs no change. The `forecast_contribution` /
  `state_component` columns and cohort columns are the building blocks for
  movement views.
- **Phase 5 (risk monitor):** funded / forecast-funded frames feed
  `analytics_lib.concentration`; two snapshots resolved via `compare` feed the
  (future) migration engine. Route eligibility and structured issues give the
  monitor a consistent, fail-soft substrate.

---

## 5. Files changed

- `mi_agent/states/__init__.py`, `models.py`, `selectors.py`,
  `route_contracts.py`, `assembler.py` (new package).
- `tests/test_phase3_mi_state_assembler.py` (new, 30 tests).
- `docs/phase3_mi_state_assembler.md` (this file).

## 6. Tests run

`tests/test_phase3_mi_state_assembler.py` — **30 passed**. Combined
Phase 0B/1/2/3 + `mi_agent/` suites — **268 passed**.

## 7. Limitations / deferred items

- Funded/pipeline classification relies on a configurable value vocabulary;
  unusual client labels may need per-client extension in a later phase.
- The forecast component reads per-row `forecast_funding_probability` /
  `forecast_funded_balance`; a config-driven stage→probability model
  (`config/client/pipeline_expected_funding.yaml`) is left for Phase 4.
- `range` / `compare` selectors resolve snapshots but the trend/migration
  **runtime** is deferred to Phase 4/5.
- The assembler is not wired into the MI Agent runtime (by design).
