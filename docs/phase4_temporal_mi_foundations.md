# Phase 4 — Temporal MI / Trend & Compare Foundations

**Status:** Implemented (temporal compare/trend + config forecast fallback +
tests + docs). No full orchestration, no MI Agent runtime wiring, no LLM query
routing, no M&A agent runtime, no full risk monitor, no Azure, no Streamlit/chart
migration, no legacy `analytics/` imports, no Annex 2/regulatory changes.

**Dependency order:** Phase 0 → Phase 0B → Phase 1 → Phase 2 → Phase 3 →
**Phase 4**.

**Date:** 2026-06-18

Phase 3 assembles a single snapshot into an MI state. Phase 4 adds the two
recurring-MI capabilities on top: **compare** (two snapshots) and **trend**
(an ordered range), deterministically and frame-in/frame-out.

---

## 1. What was built

`mi_agent/states/temporal.py` (plus small extensions to existing modules):

| Item | Responsibility |
|---|---|
| `compare(...)` | Compare a state between a baseline and a current snapshot. |
| `trend(...)` | Assemble a state across an ordered range of snapshots. |
| `assemble_temporal(...)` | Dispatcher by mode (`compare` / `trend`). |
| `TemporalResult` | `mode` / `state` / `frame` / `issues` / `metadata` (+ `ok`, `row_count`). |
| `mi_agent/states/forecast.py` | `load_stage_probabilities()` — reads `config/client/pipeline_expected_funding.yaml`. |
| `route_contracts.validate_temporal_request` | Validates state **and** temporal mode against the route contract. |
| `assembler.total_forecast_funded` (extended) | Optional `stage_probabilities` config fallback + `forecast_source` column. |

### 1.1 Compare
Resolves baseline/current snapshots via `SnapshotStore.resolve_as_of`, assembles
each via the Phase 3 state functions, and emits:
- current / baseline **value** (balance), **absolute change**, **percentage
  change** (with explicit divide-by-zero handling), **count change**;
- **new / exited / retained** record counts using the stable `loan_id` key from
  the snapshot layer (issue if the key is absent);
- optional **stratified** compare (`stratify_by=` / `segment=`) producing
  per-bucket baseline/current/change rows (reusing `analytics_lib.stratify`).

Supported states: `total_funded`, `total_pipeline`, `total_forecast_funded`,
`cohort_by_origination_date` / `_funding_date` / `_acquisition_date`,
`cohort_by_portfolio` / `_spv` / `_acquired_portfolio`.

### 1.2 Trend
Resolves an ordered range via `SnapshotStore.resolve_range` and produces one row
per snapshot (or per snapshot × group when `stratify_by`/`segment` is given) with
`reporting_date`, `snapshot_id`, `state`, `count`, `balance`, and the optional
group column. Works for funded / pipeline / forecast-funded / cohort states and
for portfolio/SPV/acquired-portfolio segments where those fields exist.

### 1.3 Forecast stage-probability config (Phase 3 deferral closed)
`total_forecast_funded` now resolves a pipeline row's expected funded balance in
priority order:
1. `forecast_funded_balance` (→ `forecast_source = explicit_balance`);
2. `current_outstanding_balance × forecast_funding_probability`
   (→ `row_probability`);
3. `current_outstanding_balance × config stage→probability` lookup on
   `pipeline_stage` (→ `config_stage_probability`, recorded as
   `forecast_probability_from_config`).
Probabilities are **never invented**: a row with none of the above is flagged
(`missing_forecast_probability`) and retained with a null contribution (or
excluded on request). Compare/trend pass the config through via
`stage_probabilities=` or `forecast_config_path=`.

### 1.4 Structured issues
`missing_baseline_snapshot`, `missing_current_snapshot`,
`insufficient_snapshots_for_trend`, `missing_stable_key_for_movement`,
`missing_probability_for_forecast` / `missing_forecast_probability`,
`forecast_probability_from_config`, `unavailable_temporal_mode`,
`unsupported_temporal_state`, `empty_temporal_result`,
`percentage_change_divide_by_zero`. Optional gaps never crash.

### 1.5 Route eligibility
`validate_temporal_request(state, route, mode)` checks the requested state is in
the route's `allowed_states` **and** the mode is in the route's
`temporal_modes`. MI permits `single`/`compare`/`trend`; M&A and Regulatory are
`single` only, so compare/trend are rejected there (M&A with a non-MI state is
rejected as `unsupported_temporal_state`; with a valid state but compare/trend
mode as `unavailable_temporal_mode`; Regulatory has no MI states so any temporal
MI request is rejected).

---

## 2. What was intentionally NOT built

- No full orchestration, no MI Agent runtime wiring, no LLM query routing.
- No M&A agent runtime, no **full** risk monitor.
- **No risk-grade / PD migration matrices** — that is Phase 5 (the snapshot join
  primitives and movement counts here are the substrate it will build on).
- No Azure, no Streamlit/chart output, no legacy `analytics/` imports, no
  Annex 2/regulatory changes.

---

## 3. How it builds on earlier phases

- **Phase 2 (`SnapshotStore`):** compare uses `resolve_as_of`; trend uses
  `resolve_range`; both load frames via `load_loans`. Movement counts use the
  Phase 2 stable `loan_id` (funded) / `OPP_` opportunity-id (pipeline) keys.
- **Phase 3 (state assembler):** every temporal point reuses `assemble_state`
  unchanged (single-snapshot assembly). The only Phase 3 change is the additive,
  backward-compatible `stage_probabilities` parameter + `forecast_source` column
  on `total_forecast_funded`.
- **Phase 1 (`analytics_lib`):** stratified compare/trend reuse
  `analytics_lib.stratify`; bucket dimensions are materialised on demand via
  `analytics_lib.materialise_buckets`.
- **Phase 0B (route configs):** `allowed_states` + `temporal_modes` drive
  eligibility.

---

## 4. How it prepares Phase 5 (risk monitor)

- The compare path already performs the **two-snapshot join on a stable key**
  (new/exited/retained); the risk monitor's grade/PD/IFRS-9 **migration
  matrices** are the same join specialised to transition counts — they slot in
  beside `_movement` without changing the resolver or assembler layers.
- `total_funded` / `total_forecast_funded` frames + `analytics_lib.concentration`
  give the monitor its funded-vs-forecast concentration and limit-usage inputs;
  trend rows feed concentration-movement (early-warning) trajectories.

---

## 5. How it supports recurring MI without full orchestration

Compare and trend are pure functions over a `SnapshotStore` — a caller (a future
orchestration/agent layer) supplies the store, the route, a state and a temporal
selector and gets a deterministic DataFrame + issues back. Nothing here schedules
ingestion, routes questions, or renders output; recurring MI is *expressible*
(weekly/monthly snapshot-to-snapshot deltas and trends) without any runtime
wiring.

---

## 6. Files changed

- `mi_agent/states/temporal.py` (new), `mi_agent/states/forecast.py` (new).
- `mi_agent/states/assembler.py` (additive `stage_probabilities` +
  `forecast_source`), `mi_agent/states/models.py` (temporal issue codes),
  `mi_agent/states/route_contracts.py` (`temporal_modes` validation),
  `mi_agent/states/__init__.py` (exports).
- `tests/test_phase4_temporal_mi.py` (new, 25 tests).
- `docs/phase4_temporal_mi_foundations.md` (this file).

## 7. Tests run

`tests/test_phase4_temporal_mi.py` — **25 passed**. Combined
Phase 0B/1/2/3/4 + `mi_agent/` suites — **293 passed**.

## 8. Limitations / deferred items

- Compare movement counts are computed at the total level; per-bucket movement
  (new/exited/retained within a stratum) is deferred.
- The config forecast model uses only the `stage_probabilities` block; broker /
  product adjustments and stage-days-to-fund in the config are not yet applied.
- `compare` aligns snapshots via `resolve_as_of` (latest on/before the date);
  exact-date-only matching is not enforced.
- Risk-grade/PD migration matrices and the full risk monitor are Phase 5.
- Not wired into the MI Agent runtime (by design).
