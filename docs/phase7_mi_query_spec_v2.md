# Phase 7 — MIQuerySpec v2 & MI Interpretation Contract Foundations

**Status:** Implemented (additive spec expansion + spec-level validator +
interpretation contract + examples + tests + docs). No external LLM calls, no
onboarding orchestration, no Azure, no Streamlit migration, no M&A runtime, no
Annex 2/regulatory changes, no new chart types, no loosened validation.

**Dependency order:** Phases 0–6E → **Phase 7**.

**Date:** 2026-06-18

Phase 7 evolves the **existing `MIQuerySpec`** into the *single governed query
contract* that expresses both (1) the flat single-CSV MI Agent path and (2) the
snapshot/state/temporal/risk runtime path — backward-compatibly.

---

## 1. What changed

- **`mi_agent/mi_query_spec.py` — additive v2 expansion.** New optional/defaulted
  fields covering state, snapshot/segmentation, temporal, forecast, risk,
  buckets, chart/output and governance (e.g. `query_id`, `reporting_date`,
  `cut_off_date`, `portfolio_id`/`spv_id`/`acquired_portfolio_id`,
  `comparison_basis`, `trend_grain`, `forecast_probability_source`,
  `risk_monitor_mode`, `migration_dimension`/`concentration_dimension`/
  `risk_dimension`, `bucket_strategy`/`bucket_count`/`bucket_field`,
  `output_type`, `require_structured_issues`, `strict_mode`, …). Controlled
  vocabularies (`STATES`, `TEMPORAL_MODES`, `RISK_MONITOR_MODES`,
  `BUCKET_STRATEGIES`, …) are module constants shared with the validator and the
  interpretation contract.
- **`MIQuerySpec.effective_execution_mode()` + `normalized()`.** Mode inference
  mirrors `mi_runtime.infer_execution_mode`; `normalized()` maps v2 convenience
  fields onto the canonical runtime fields (`risk_monitor_mode` → `risk_monitor`,
  `migration/concentration/risk_dimension` → `dimension`, `reporting_date` →
  `as_of_date` for `as_of`), and is **idempotent for v1/Phase-6 specs**.
- **`mi_agent/mi_spec_validation.py` — new spec-level validator.**
  `validate_query_spec(spec, …)` returns a `SpecValidationResult` of structured
  issues for mode requirements, route gating, ambiguous dimensions, and flat-mode
  virtual-field misuse. It **complements** (does not replace or loosen) the v1
  chart-structure validator in `mi_query_validator.py`.
- **`mi_agent/mi_query_spec_v2_examples.py` — canonical examples.** A catalogue of
  valid (`EXAMPLES`) and must-fail (`INVALID_EXAMPLES`) specs, machine-checked by
  the tests so the docs and code never drift.
- **Docs:** `docs/mi_query_spec_v2_interpretation_contract.md` (LLM boundary) and
  this summary.

## 2. What stayed backward-compatible

- Every new field is optional/defaulted; `route_id` defaults to `mi`,
  `execution_mode` to `None` (→ `flat`). A v1 spec is unchanged in behaviour.
- `referenced_fields()` and the v1 validator are untouched — the new fields are
  **not** semantic-field slots and do not affect v1 chart validation.
- `from_dict` still silently drops unknown keys; `to_dict` round-trips.
- **`run_mi_query` is unchanged** — no runtime rewrite. The existing flat,
  Phase 6, 6B and 6C suites pass unchanged (271 combined with Phase 7).

## 3. How MIQuerySpec v2 relates to `run_mi_query`

The runtime reads canonical fields (`route_id`, `execution_mode`, `state`,
`temporal_mode`, date selectors, `risk_monitor`, `dimension`,
`snapshot_client_id`, `snapshot_store_root`). v2 adds **convenience** fields
(e.g. `risk_monitor_mode`, `migration_dimension`); `spec.normalized()` maps them
onto those canonical fields, so a v2 spec runs through the *unchanged* runtime
after normalisation. A Phase 7 test proves a normalised v2 risk-migration spec
executes end-to-end via `run_mi_query` over a `LocalFsSnapshotStore`.

> Naming alignment: where Phase 6 already had a field (e.g. `risk_monitor`,
> `dimension`, `as_of_date`), v2 keeps it as the canonical target and treats the
> new descriptive fields as aliases resolved by `normalized()` — no duplicate
> competing concepts at the runtime.

## 4. How it prepares the LLM interpretation layer

`docs/mi_query_spec_v2_interpretation_contract.md` defines the **only** thing an
interpreter/LLM may emit: MIQuerySpec-v2 JSON drawn from the controlled
vocabularies — never code, SQL, chart specs, computed numbers, or unlisted
fields. It specifies the natural-language term mappings (portfolio → Trakt
`portfolio_id`; acquired → `acquired_portfolio_id`; SPV → `spv_id`; stage →
`pipeline_stage` only in pipeline context; IFRS/risk stage variants; quantile
buckets by default) and the clarification behaviour for ambiguous requests. The
validator enforces these rules deterministically, so a future parser can be
graded against them **without any LLM in the test suite**.

## 5. Validation rules implemented

- Flat mode rejects snapshot-only **virtual** fields not present in the provided
  DataFrame (`virtual_field_not_available_in_flat_mode`).
- State mode requires `state` + `snapshot_client_id`.
- Temporal **compare** requires `baseline_date` + `current_date`; **trend**
  requires `start_date` + `end_date` (`temporal_selector_incomplete`).
- Risk mode requires a valid `risk_monitor_mode` + the dimension/dates that mode
  needs (`invalid_risk_monitor_spec`).
- **M&A** route rejects MI pipeline/forecast states (`invalid_route_for_state`),
  temporal compare/trend (`invalid_temporal_mode_for_route`), and the risk
  monitor (`risk_monitor_not_enabled`).
- **Regulatory** route rejects all MI state/temporal/risk execution.
- Bare ambiguous dimensions (`stage`/`portfolio`/`region`/`rate`/`balance`) are
  rejected (`ambiguous_dimension`).
- Bad enum values are rejected (`invalid_enum_value`). Invalid combinations
  return structured issues, never crashes.

## 6. Files changed

- `mi_agent/mi_query_spec.py` (additive fields + vocab constants + `normalized()`)
- `mi_agent/mi_spec_validation.py` (new spec validator)
- `mi_agent/mi_query_spec_v2_examples.py` (new examples catalogue)
- `tests/test_phase7_mi_query_spec_v2.py` (new, 45 tests)
- `docs/mi_query_spec_v2_interpretation_contract.md`, `docs/phase7_mi_query_spec_v2.md`

## 7. Tests run

`tests/test_phase7_mi_query_spec_v2.py` — **45 passed**. MI Agent + Phase 6/6B/6C
+ Phase 7 — **271 passed** (no regression; flat path and runtime unchanged).

## 8. What remains deferred

- **The interpreter/LLM itself is not built** — Phase 7 defines the contract and
  the deterministic validator only; tests construct specs directly.
- **Runtime wiring of `normalized()`** — `run_mi_query` is intentionally
  unchanged; callers normalise v2 convenience specs before dispatch (a future
  phase may have the runtime call `normalized()` at entry).
- **Quantile bucketing in the runtime** — `bucket_strategy: quantile` is
  expressible and validated, but materialising quantile bands inside grouped
  runtime paths remains a follow-up (the engine exists and is unit-tested).
- Onboarding orchestration, Azure adapter, UI, and M&A Agent runtime remain out
  of scope.
