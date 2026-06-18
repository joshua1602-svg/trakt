# Phase 6 — MI Agent Runtime Integration (read-only / query-only)

**Status:** Implemented (Step 0 semantic fixes + governed runtime dispatch +
tests + docs). Read-only/query-only. No onboarding orchestration, no Azure, no
Event Grid/blob ingestion, no Streamlit migration, no M&A Agent runtime, no
legacy `analytics/` imports, no Annex 2/XML/regulatory changes, no new chart
types, no UI rebuild, no LLM in tests.

**Dependency order:** Phase 0 → 0B → 1 → 2 → 3 → 4 → 5 → 5A audit → **Phase 6**.

**Date:** 2026-06-18

Phase 6 wires the Phase 2/3/4/5 foundations into the MI Agent behind **one
explicit runtime boundary** (`mi_agent/mi_runtime.py`), after first resolving the
Phase 5A audit's semantic hazards (Step 0). The existing flat single-CSV MI Agent
is preserved exactly.

---

## 1. What was built

| File | Responsibility |
|---|---|
| `mi_agent/mi_query_spec.py` | **Additive** optional runtime fields on `MIQuerySpec` (no breaking change). |
| `mi_agent/portfolio_reference.py` | Trakt portfolio reference config model + loader (Step 0 #1). |
| `mi_agent/semantic_resolver.py` | Governed term resolution: portfolio/SPV/stage/buckets with structured issues (Step 0 #1, #2). |
| `mi_agent/quantile_buckets.py` | Asset-agnostic quantile (quartile) bucketing for balance/rate/time-on-book (Step 0 #3). |
| `mi_agent/mi_runtime.py` | Runtime dispatch: flat vs state vs temporal vs risk; route gating; `RuntimeResult`; governed chart instruction. |
| `config/client/portfolio_reference_example.yaml` | Illustrative portfolio reference config shape. |

---

## 2. Phase 6 Step 0 — compatibility/semantic fixes (done first)

### 2.1 Trakt portfolio reference model
"portfolio" is **not** hard-coded to any source field. Trakt mints its own
portfolio references per client during onboarding
(`portfolio_reference_pattern: {client_slug}_{sequence:03d}` →
`client_a_001`, …). Resolution:
- **"portfolio"** → Trakt `portfolio_id` (requires a portfolio reference
  config); with no config → `missing_portfolio_reference_config` issue,
  **never** `acquired_portfolio_id`.
- **"acquired portfolio"** → `acquired_portfolio_id`.
- **"SPV"** → `spv_id`.

The config shape (`portfolio_reference.py` / example YAML) carries `client_id`,
`client_name`, `portfolio_reference_pattern`, and per-portfolio `portfolio_id`,
`portfolio_name`, optional `source_portfolio_label`/`source_portfolio_field`,
`spv_id`, `acquired_portfolio_id`. A `mint_reference(n)` helper assigns
references deterministically. Full onboarding population is out of scope.

### 2.2 "stage" means pipeline stage
Bare **"stage"** → `pipeline_stage` (KFI / application / offer / completion /
funded), **only** in a pipeline context (context `pipeline`/`origination` or
state `total_pipeline`/`total_forecast_funded`). Elsewhere
(funded-only / M&A / regulatory / risk) it returns `invalid_stage_context`.
"pipeline stage" → `pipeline_stage`; "IFRS stage"/"IFRS 9 stage" →
`ifrs9_stage`; "risk stage"/"internal risk stage" → `internal_risk_stage`.
This corrects the audit hazard where "stage" resolved to `ifrs9_stage`.

### 2.3 Asset-agnostic quantile buckets
Balance, interest-rate and time-on-book buckets **default to quartiles over the
selected population** (`quantile_buckets.py`), not product-specific fixed bands.
Insufficient distinct data → `quantile_bucket_insufficient_data` (no silent
fallback to arbitrary bands). Asset-specific bands already defined for LTV /
borrower age (in `config/mi/buckets.yaml` via `analytics_lib.buckets`) are
preserved untouched.

### 2.4 Preserve the existing MI Agent
Flat single-CSV mode behaves exactly as before. Virtual snapshot-only fields do
not crash flat mode — they are rejected with
`virtual_field_not_available_in_flat_mode`. Route `allowed_dimensions` now all
resolve (the three previously-unregistered bucket dims resolve as quantile
dimensions), closing audit finding F1 for the runtime path.

---

## 3. Runtime integration

### 3.1 Additive spec fields
`route_id` (default `mi`), `execution_mode`, `state`, `temporal_mode`,
`as_of_date`, `baseline_date`, `current_date`, `start_date`, `end_date`,
`segment`, `risk_monitor`, `snapshot_client_id`, `snapshot_store_root`. All
optional and defaulted; existing v1 specs are unchanged and
`referenced_fields()` / v1 validation are unaffected.

### 3.2 Dispatch (`run_mi_query`)
Infers the mode (`flat` default; `risk` if `risk_monitor`; `temporal` if
`temporal_mode ∈ {compare, trend}`; `state` if `state` set) and routes to:
- **flat** → existing `execute_mi_query` over the loaded DataFrame;
- **state** → `mi_agent.states.assemble_state` (latest/as_of selector);
- **temporal** → `mi_agent.states.temporal.compare` / `trend`;
- **risk** → `mi_agent.risk_monitor.run_migration` / `run_concentration` /
  `run_trajectory`.

Returns a unified `RuntimeResult(mode, result_type, data, issues, metadata,
warnings, chart_instruction)`.

### 3.3 Route-contract validation
- **MI** route: states, temporal compare/trend, and risk monitor allowed.
- **M&A** route: pipeline/forecast states rejected (`invalid_route_for_state`),
  temporal compare/trend rejected (`invalid_temporal_mode_for_route`), risk
  monitor rejected (`risk_monitor_not_enabled`) unless `allow_mna_risk=True`.
- **Regulatory** route: all MI state/temporal/risk execution rejected.

### 3.4 SnapshotStore integration
`LocalFsSnapshotStore` only. The store is passed in or built from an explicit
`store_root`/`snapshot_store_root`; missing → `snapshot_store_required` /
`snapshot_store_missing`. No Azure, no cloud auto-discovery.

### 3.5 Chart governance
Only the existing governed MI chart factory and `CHART_TYPES`
(`bar/line/scatter/bubble/heatmap/treemap/none`) are used. Flat mode renders via
`create_mi_chart`. Non-flat results carry a governed `chart_instruction`
(trend → `line`, compare/concentration → `bar`, migration → `heatmap`); an
unrenderable result falls back to table-only with
`unsupported_chart_for_result`. No multi-line cohort curve was added (deferred);
no Streamlit chart code copied.

---

## 4. Backward compatibility

The flat path is unchanged: `MIQuerySpec` defaults leave `execution_mode=None`
→ `flat`, and `run_mi_query` calls the existing validate→execute→chart pipeline.
The existing MI Agent suite (135 tests) and all phase suites pass unchanged
(**360 passed** combined). New spec fields are additive dataclass members that
do not affect `referenced_fields()` or v1 chart validation.

---

## 5. Runtime paths implemented & tested

- **A. Flat:** existing DataFrame MI query (bar, grouped) — works; chart factory
  used.
- **B. State:** `total_funded`, `total_pipeline`, `total_forecast_funded` (latest
  snapshot).
- **C. Temporal:** funded compare; funded/pipeline/forecast-funded trend.
- **D. Risk:** risk-grade migration (baseline/current); concentration warning
  (current); trajectory (range).

---

## 6. Structured issues

`snapshot_store_required`, `snapshot_store_missing`, `invalid_route_for_state`,
`invalid_temporal_mode_for_route`, `unsupported_chart_for_result`,
`state_result_empty`, `risk_monitor_not_enabled`, `temporal_selector_incomplete`,
`missing_snapshot_client_id`, `fallback_to_flat_executor`,
`virtual_field_not_available_in_flat_mode`, plus Step 0 resolver codes
(`missing_portfolio_reference_config`, `invalid_stage_context`,
`ambiguous_dimension`, `unresolved_route_dimension`,
`quantile_bucket_insufficient_data`).

---

## 7. What was intentionally NOT built

- No onboarding orchestration (only the minimal portfolio-reference config shape
  + helper).
- No Azure adapter, no Event Grid/blob ingestion.
- No Streamlit/chart migration, no UI rebuild, no M&A Agent runtime.
- No legacy `analytics/` imports, no Annex 2/XML/regulatory changes.
- No new/free-form chart types; no multi-line cohort curves.
- No LLM parsing in tests (specs are constructed directly). The deterministic
  parser was left unchanged; the governed `semantic_resolver` is the additive
  resolution surface.

---

## 8. How the foundations are wired behind one boundary

`run_mi_query` is the single entry point. Everything below it
(`snapshot/`, `mi_agent/states/`, `mi_agent/risk_monitor/`, `analytics_lib/`) is
reached only through it; the existing `mi_query_*` modules are untouched except
the additive spec fields. This keeps the flat path isolated and lets the
state/temporal/risk capabilities be enabled per query/route without disturbing
v1.

---

## 9. Files changed

- `mi_agent/mi_query_spec.py` (additive fields), `mi_agent/mi_runtime.py`,
  `mi_agent/semantic_resolver.py`, `mi_agent/portfolio_reference.py`,
  `mi_agent/quantile_buckets.py` (new),
  `config/client/portfolio_reference_example.yaml` (new example),
  `tests/test_phase6_mi_runtime.py` (new, 44 tests),
  `docs/phase6_mi_runtime_integration.md` (this file).

## 10. Tests run

`tests/test_phase6_mi_runtime.py` — **44 passed**. Combined `mi_agent/` +
Phase 0B/1/2/3/4/5/6 — **360 passed**.

## 11. Limitations / deferred items

- Onboarding population of portfolio references, and quantile-bucket
  materialisation *inside* the state/temporal aggregation, are deferred (the
  quantile engine is available and unit-tested; wiring it into every grouped
  runtime path is a follow-up).
- Non-flat results return a governed `chart_instruction` but are not auto-rendered
  through the chart factory in this phase (flat mode is).
- Multi-line cohort curves, M&A Agent runtime, Azure adapter, onboarding
  orchestration, and UI remain future phases.
