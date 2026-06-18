# Phase 5A — MI Agent Regression & Target-State Audit (read-only)

**Type:** Read-only audit. No code, configs, or tests were modified. No Phase 0–5
module was wired into the MI Agent runtime. No PR opened.

**Date:** 2026-06-18
**Audited `main`:** `0050ef6` (Phase 5 merged via #164; Phases 0–5 all present).

---

## 1. Executive summary

**The existing governed MI Agent flat-query path still works and is uncontended
by Phases 0–5.** The MI Agent test suite is fully green (135/135), a live
end-to-end smoke run (parse → validate → execute → chart factory) succeeds, and
every Phase 0–5 test suite passes. The new packages (`analytics_lib/`,
`snapshot/`, `mi_agent/states/`, `mi_agent/risk_monitor/`) are **not imported by
the MI Agent runtime** — they are genuinely additive and unwired.

The additions are **mostly target-state and correctly isolated**, with two
caveats that matter *before* Phase 6 wiring (not now):

1. **Shared runtime-consumed artifacts did change** (additively): the canonical
   field registry (+21 fields), analytics aliases, and the **regenerated MI
   semantic registry** (72→99 fields, `v0.2.2 → v0.3.0`). The MI Agent runtime
   *does* read the semantic registry, so this is not "isolated new packages"
   — it is an additive content change to a runtime input. It is backward
   compatible today (all tests pass), but it shifts deterministic parser
   resolution for some terms (§5).
2. **Route/registry naming gaps**: route `allowed_dimensions` reference three
   bucket dimensions (`interest_rate_bucket`, `time_on_book_bucket`,
   `balance_band`) that are not keys in the MI semantic registry. Harmless while
   unwired; a Phase 6 dimension validator would reject them.

No critical breakage was found. The broad repository suite shows pre-existing
environment/engine failures only (missing `openpyxl`/`lxml`, a delivery-XML
fixture issue) — **none in MI-phase or `mi_agent/` tests**.

**Recommendation:** safe to proceed to Phase 6 planning, with a short list of
should-fix items (parser/virtual-field handling, route-dimension naming) to
resolve as the first step of Phase 6 rather than as emergency fixes.

---

## 2. Tests run and results

| Suite | Result |
|---|---|
| `mi_agent/tests/` (existing MI Agent) | **135 passed** |
| `tests/test_phase0b_mi_mna_foundations.py` | passed |
| `tests/test_phase1_analytics_lib.py` | passed |
| `tests/test_phase2_snapshot_layer.py` | passed |
| `tests/test_phase3_mi_state_assembler.py` | passed |
| `tests/test_phase4_temporal_mi.py` | passed |
| `tests/test_phase5_risk_monitor.py` | passed |
| Combined MI-phase + `mi_agent/` (prior runs) | **316 passed** |
| Full `tests/` dir (minus `test_xml_builder_annex2_shape_fixes.py`, which needs `lxml`) | **1221 passed, 25 failed, 1 skipped, 20 errors** (~9 min) |

**Live smoke run** (deterministic, no LLM): `parse_user_question("ltv by region")`
→ `validate_mi_query` (ok) → `execute_mi_query` over a synthetic canonical frame
→ 3 rows (`collateral_geography`, `current_loan_to_value_weighted_avg`,
`concentration_pct`); chart factory (`create_mi_chart`) present and importable.
**The governed flat path is intact.**

### Failure attribution (broad suite)

The 20 **errors** and 25 **failures** are **pre-existing and environmental**, not
caused by Phases 0–5:

- `tests/test_onboarding_multifile_mapping_coverage.py`,
  `tests/test_onboarding_file_signature_diagnostics.py` → `ModuleNotFoundError:
  openpyxl` (declared in `requirements.txt`, not installed in this container).
- `tests/test_delivery_xml_agent_review.py` → `KeyError: 'preview_policy'`
  (delivery-XML fixture/config, untouched by MI phases).
- `tests/test_xml_builder_annex2_shape_fixes.py` → `ModuleNotFoundError: lxml`
  (excluded from the run).

Full enumeration confirms **every** failing/erroring file is in the
onboarding / transformation / enum / Annex 2 / XSD-XML / delivery engine areas —
**none in MI-phase or `mi_agent/`**:

```
FAILED (25):
  6  tests/test_onboarding_annex2_workflow.py
  4  tests/test_xsd_structured_preview.py
  3  tests/test_onboarding_target_coverage.py
  2  tests/test_xsd_structured_synthetic.py
  2  tests/test_transformation_agent_workflow.py
  2  tests/test_llm_mapper_agent.py
  1  tests/test_onboarding_review_pack_target_first.py
  1  tests/test_onboarding_multifile_mapping_coverage.py
  1  tests/test_onboarding_file_signature_diagnostics.py
  1  tests/test_onboarding_field_scope.py
  1  tests/test_enum_agent.py
  1  tests/test_annex2_field_xsd_path_map.py
ERROR (20):
 10  tests/test_delivery_xml_agent_review.py
  6  tests/test_onboarding_multifile_mapping_coverage.py
  4  tests/test_onboarding_file_signature_diagnostics.py
```

Evidence it is not MI-phase-caused:
- **Zero** MI-phase or `mi_agent/` test files appear in the failed/errored set
  (grep over the summary for `phase*|mi_agent|risk_monitor|analytics_lib|snapshot|states|temporal` → none).
- The MI phases did not modify any engine/onboarding/delivery/regulatory source
  or config (§4 file-diff).
- The canonical registry change is purely additive: **472 → 493 fields, 0
  removed** (so no engine field-count assumption is invalidated).
- Sampled root causes: missing `openpyxl` (onboarding multifile/file-signature),
  missing `lxml` (xml builder), `KeyError: 'preview_policy'` (delivery-XML
  fixture) — all pre-existing environment/engine, independent of Phases 0–5.

---

## 3. Existing MI Agent regression status

**PASS — no regression.**

| Flow element | Status |
|---|---|
| NL / deterministic spec parsing (`llm_query_parser`) | ✅ works |
| `MIQuerySpec` validation (`mi_query_validator`) | ✅ works |
| Semantic registry lookup (`load_mi_semantics`) | ✅ works (now loads the 99-field v0.3.0 registry) |
| Query execution over a single DataFrame (`mi_query_executor`) | ✅ works; raises clear `MIQueryExecutionError` on missing columns |
| Chart factory / permissible library | ✅ `create_mi_chart` import OK; `CHART_TYPES = {bar, line, scatter, bubble, heatmap, treemap, none}` unchanged |
| Existing tests | ✅ 135/135 |

**Public-API / assumption changes (breaking? no, but notable):**
- `mi_semantics_field_registry.yaml` **version 0.2.2 → 0.3.0**, **field_count
  72 → 99** (67 core / 32 extended / 14 derived / **15 virtual**). The two
  existing registry tests (`test_mi_semantics_buckets.py`,
  `test_mi_semantics_cleanup.py`) were **updated** for the new counts/version —
  i.e. a test-level assumption was deliberately revised, not silently broken.
- A new metadata key `virtual_field_count` was added (additive).
- No change to `CHART_TYPES`, `MIQuerySpec` public fields, executor/validator
  signatures.

---

## 4. Additive / isolation status

| Check | Result |
|---|---|
| `analytics_lib` imported by MI Agent runtime | ❌ not imported (isolated) |
| `snapshot` wired into runtime | ❌ not wired |
| `mi_agent/states` wired into runtime | ❌ not wired (sub-package, not imported by runtime `*.py`) |
| `mi_agent/risk_monitor` wired into runtime | ❌ not wired |
| Streamlit/legacy `analytics/` imports leaked into MI Agent | ❌ none (Streamlit only in the pre-existing `mi_agent/streamlit_mi_agent.py` UI) |
| Azure imports anywhere (`mi_agent`/`analytics_lib`/`snapshot`) | ❌ none |
| Annex 2 / regulatory **logic** modified | ❌ none |

`grep` over `mi_agent/*.py` (runtime files) for `analytics_lib|snapshot|mi_agent.states|risk_monitor`
returns nothing; `mi_agent/__init__.py` and `streamlit_mi_agent.py` likewise.

**File-diff of all MI-phase changes (base `6a3ddba` → `HEAD`):** new packages
(`analytics_lib/`, `snapshot/`, `mi_agent/states/`, `mi_agent/risk_monitor/`),
new configs (`config/mi/*`, `config/routes/*`, `config/mna/*`), docs, phase
tests, **plus three shared/runtime-consumed files**:
`config/system/fields_registry.yaml` (additive +21),
`config/system/aliases_analytics.yaml` (additive), and
`mi_agent/mi_semantics_field_registry.yaml` (regenerated), and two updated MI
test files. **The only path matching "regulatory" is the new
`config/routes/regulatory_annex2_route.yaml` skeleton — a route *contract*
reference doc, not Annex 2 delivery/XML logic.**

> **Isolation nuance:** the *packages* are fully isolated; the *config layer* is
> not — Phases 0/0B intentionally edited shared registry/alias files that the
> existing onboarding/transformation engine and the MI Agent both consume. This
> is by design and additive, but it means "isolated" is true for code modules
> and "additive-but-shared" for config.

---

## 5. Config & registry consistency findings

Automated checks (duplicate-key loader + cross-reference script):

| Check | Result |
|---|---|
| Duplicate keys in any `config/**`, registries, aliases | ✅ none |
| Semantic fields → missing canonical **without** virtual/derived flag | ✅ none |
| Derived fields lacking `derived_from` | ✅ none |
| Virtual fields that actually exist in canonical (should not be virtual) | ✅ none |
| Route `allowed_states` not in `state_library.yaml` (alias-aware) | ✅ none |
| `stratification_catalogue` bucket refs missing from `buckets.yaml` | ✅ none |
| Chart types referenced by configs outside the permissible library | ✅ none referenced |
| Canonical registry additive (no removed fields) | ✅ 472→493, 0 removed |

**Findings (non-blocking while unwired):**

- **F1 (medium) — route dimensions not in the semantic registry.**
  `config/routes/{mi,mna,regulatory_and_mi}_route.yaml` list
  `interest_rate_bucket`, `time_on_book_bucket`, and `balance_band` under
  `allowed_dimensions`, but those keys are **not** in
  `mi_agent/mi_semantics_field_registry.yaml`. `balance_band`'s registry
  equivalent is `ticket_bucket` (**naming mismatch**); `interest_rate_bucket`
  and `time_on_book_bucket` are defined only in `buckets.yaml`, not registered
  as semantic dimensions. A Phase 6 validator that checks dimensions against the
  registry would reject these.
- **F2 (low) — bucket `semantic_field` self-references.** In `buckets.yaml`,
  `interest_rate_bucket` and `time_on_book_bucket` declare
  `semantic_field: <self>`, which is not a registry entry (they are flagged
  "engine-materialised" in comments). Consistent with intent, but they are not
  yet first-class registry dimensions.
- **F3 (informational) — alias collisions.** `aliases_analytics.yaml` is
  collision-free for the new fields (per Phase 0 review); the only pre-existing
  cross-field collisions involve `number_of_days_in_*_arrears` /
  `*_valuation_date` / `origination_date` and do not touch the new risk fields.

---

## 6. Target-state classification

| Addition | Classification | Notes |
|---|---|---|
| Phase 0 canonical risk fields (`internal_risk_grade`, `ifrs9_stage`, `pd/lgd/ead`, …) | **Target-state & necessary** | Additive, 0 removals; underpins MI risk monitor & M&A risk views. |
| Phase 0 analytics aliases | **Target-state & necessary** | Additive analytics-tier aliases. |
| Phase 0B route/state/bucket/stratification/M&A config skeletons | **Target-state, partially premature/unwired** | Declarative-only; F1/F2 naming gaps to close before they are read at runtime. |
| Phase 0B MI semantic registry curation (virtual/derived fields, v0.3.0) | **Target-state but with runtime-coupling risk** | Consumed by the live validator/parser; introduces virtual fields reachable by the parser (§7 R1/R2). |
| Phase 1 `analytics_lib/` | **Target-state & necessary** | Pure, isolated, well-tested substrate. |
| Phase 2 `snapshot/` (+ local_fs adapter) | **Target-state & necessary** | Isolated; CSV dtype-loss noted (§7 R5). |
| Phase 3 `mi_agent/states/` assembler | **Target-state & necessary** | Isolated; reuses Phase 1/2. |
| Phase 4 `mi_agent/states/temporal.py` | **Target-state & necessary** | Isolated; additive `stage_probabilities` param on `total_forecast_funded` was a backward-compatible signature change. |
| Phase 5 `mi_agent/risk_monitor/` | **Target-state, target-state-but-unwired** | Isolated; depends on config orderings (§7 R7). |
| `config/mi/risk_monitor.yaml` Phase 5 extensions | **Target-state & necessary** | Additive; `example_concentration_limits` explicitly non-enforced. |

**Nothing is classified "should be removed."** The items to *gate behind the
Phase 6 runtime boundary* are the ones that read shared runtime inputs: the
semantic-registry virtual fields and the route dimension list.

---

## 7. Runtime-risk assessment (highest first)

- **R1 — Virtual fields are parser-reachable but have no data in the flat path.**
  The deterministic parser resolves `"portfolio"` → `acquired_portfolio_id`
  (**virtual**, and arguably the *wrong* portfolio field — should be
  `portfolio_id`), `"spv"` → `spv_id` (virtual), etc. In the current
  single-CSV executor these columns are absent, so such a query raises
  `MIQueryExecutionError`. **Risk:** when Phase 6 wires the runtime, queries
  that resolve to snapshot-only virtual fields will error unless the parser
  excludes virtual fields from the flat path (or the state layer supplies them).
- **R2 — `"stage"` ambiguity.** `"stage"` now matches three dimensions
  (`ifrs9_stage`, `internal_risk_stage`, `pipeline_stage`); `find_field` returns
  `ifrs9_stage` by core-tier/first-match. A user meaning pipeline stage or
  internal risk stage gets IFRS 9. Latent (no test exercises it) but a real
  resolution change from Phase 0B.
- **R3 — Semantic registry version/count coupling.** Anything asserting the old
  72/`0.2.2` would break; the two in-repo tests were updated, but external
  consumers/snapshots of the registry should be re-checked before Phase 6.
- **R4 — Route-config dimension assumptions (F1).** Phase 6 route validation
  must reconcile `interest_rate_bucket`/`time_on_book_bucket`/`balance_band`
  with the registry (register them, or rename to `ticket_bucket`, or treat
  bucket keys as a separate namespace).
- **R5 — Snapshot CSV dtype loss.** `LocalFsSnapshotStore` round-trips loan rows
  as CSV; dtypes are re-inferred on load. Downstream numeric/date handling must
  coerce defensively (the analytics/state layers already use `to_numeric`/
  `to_datetime`), but mixed-type columns could surprise a naive consumer.
- **R6 — Forecast probability fallback.** `total_forecast_funded` priority is
  explicit (`forecast_funded_balance` → row prob → config stage prob → flagged
  missing) and never invents values; low risk, but config stage labels must match
  `pipeline_stage` casing (handled via lowercasing).
- **R7 — Risk-monitor ordering assumptions.** Improve/deteriorate depends on
  `deterioration_orderings`; unordered dims are `changed`/`unchanged` only.
  `internal_risk_stage` is intentionally unordered — fine, but any client whose
  grade scale differs from `A..G` needs config, or migrations will read as
  `changed`.
- **R8 — Chart factory compatibility.** No new chart type was added; cohort/
  vintage multi-line remains a *future additive* enhancement. Low risk.
- **R9 — Single-DataFrame assumption.** The legacy executor assumes one loaded
  frame; the state/temporal/risk layers assume snapshot frames. Phase 6 must
  bridge these explicitly (state assembler → frame → existing executor), not
  conflate them.

---

## 8. Recommended next actions

**Must-fix before Phase 6** *(none are emergencies; do them as Phase 6 step 0)*
- Nothing blocks the current MI Agent. No must-fix to keep today's flow working.

**Should-fix before Phase 6 runtime wiring**
1. **Parser virtual-field handling (R1/R2):** exclude snapshot-only *virtual*
   fields from the flat single-CSV resolution path (or require the state layer to
   provide them), and disambiguate `"portfolio"` → `portfolio_id` and `"stage"`.
2. **Reconcile route dimensions with the registry (F1/R4):** register
   `interest_rate_bucket`/`time_on_book_bucket` as derived semantic dimensions,
   and align `balance_band` ↔ `ticket_bucket` naming (or define a documented
   bucket-key namespace the route validator understands).
3. **Re-confirm no external consumer pins the old registry count/version (R3).**

**Acceptable to defer**
- F2 bucket `semantic_field` self-references (resolve when those buckets become
  first-class dimensions).
- Snapshot Parquet/typed-store upgrade (R5).
- Per-bucket migration movement, broker/product forecast adjustments.

**Safe to proceed**
- Phase 6 planning and design. The foundations are isolated, additive, tested,
  and do not regress the existing MI Agent. Wire them in behind an explicit
  runtime boundary (state assembler → frame → existing governed executor/chart
  factory), resolving the should-fix items as the first wiring step.

---

## 9. Environment note

Two declared dependencies (`openpyxl`, `lxml`) are absent in the audit container,
which accounts for the onboarding/XML test errors. These are **pre-existing**
and unrelated to Phases 0–5; the MI Agent and all MI-phase suites run fully on
the installed dependency set (`pandas`, `numpy`, `pyyaml`, `plotly`,
`rapidfuzz`, `pytest`).

*Read-only audit — no code, configs, or tests were modified; no PR opened.*
