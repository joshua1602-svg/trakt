# Phase 6E — MI Architecture Walkthrough, Synthetic Testing & Owner Review Pack

**Type:** Review / comprehension only. No features built, no code refactored, no
configs or tests changed. This document explains what Phases 0–6D actually
built, how it runs, what is genuinely proven, and what to inspect before
trusting it.

**Date:** 2026-06-18
**Reviewed `main`:** `bffa0a5` (Phases 0–6D all merged).

---

## 1. Executive summary

After Phases 0–6D, the MI stack has **two query paths** plus a **synthetic proof**
that the new path can be fed from realistic, fragmented data — and a clearly
**deferred** production tail.

- **Existing flat MI Agent path (unchanged, production-shaped).** The original
  `question → MIQuerySpec → validate → execute → chart` pipeline over a single
  loaded CSV/DataFrame still works exactly as before. It is governed, tested
  (135 tests), and untouched in behaviour.
- **New snapshot/state/temporal/risk runtime path (local-only).** A new boundary,
  `mi_agent/mi_runtime.run_mi_query`, can take a snapshot store and answer
  *recurring* MI: portfolio states (funded / pipeline / forecast-funded),
  period-over-period compare & trend, concentration, and risk migration —
  reading history from `LocalFsSnapshotStore`. It is deterministic and tested,
  but **local filesystem only** and **not wired into a UI or scheduler**.
- **Synthetic multi-artefact proof (synthetic-only).** Phase 6C shows that six
  fragmented source files can be deterministically consolidated into canonical
  snapshots and queried through the same runtime. The consolidation joins are
  **hard-wired for the synthetic fixture** — it proves the *shape* of the flow,
  not production data handling.
- **Still deferred.** Production onboarding/consolidation engine, Azure/cloud
  ingestion, a user interface, and the M&A Agent runtime are **not built**.

**One-line truth:** *the analytics engine and its governed runtime exist and are
well-tested in isolation; the "from real client files to MI" path is proven only
on synthetic data with hard-wired joins, and the user-facing/production tail is
not built.*

---

## 2. Phase-by-phase inventory

| Phase | Purpose | Key files added/changed | Capability introduced | Deliberately NOT built | Maturity |
|---|---|---|---|---|---|
| **0** | Canonical risk fields + aliases | `config/system/fields_registry.yaml` (+21 fields → 493), `aliases_analytics.yaml` | Asset-neutral risk/PD/LGD/IFRS9 fields + analytics aliases | Any code, charts, snapshot, agents | **Config-only** (production registry) |
| **0B** | MI/M&A semantic registry + route/config skeletons | `mi_agent/mi_semantics_field_registry.yaml` (72→99, v0.3.0), `config/routes/*`, `config/mi/*`, `config/mna/diligence_scorecard.yaml` | Curated MI semantic dimensions (incl. 15 *virtual* snapshot fields), declarative route/state/bucket/risk configs | Any runtime that reads them | **Config-only** |
| **1** | Pure shared analytics library | `analytics_lib/{buckets,stratify,concentration,cohort,migration,config_loader}.py` | Deterministic bucketing, stratification, concentration, cohort maths | Orchestration, IO, UI, charts | **Local-only, production-shaped** (pure functions) |
| **2** | Snapshot/history layer | `snapshot/{model,keys,store}.py`, `snapshot/adapters/local_fs.py` | Storage-neutral `SnapshotStore` + `LocalFsSnapshotStore`; deterministic keys; date separation | Azure adapter, ingestion triggers | **Local-only** (interface is production-shaped; only FS adapter exists) |
| **3** | MI state assembler | `mi_agent/states/{assembler,selectors,route_contracts,models}.py` | `total_funded/pipeline/forecast_funded`, cohort states; route-state eligibility | Temporal, risk, runtime wiring | **Local-only** |
| **4** | Temporal MI | `mi_agent/states/temporal.py`, `forecast.py` | compare (2 snapshots) + trend (range); config stage→probability fallback | Migration matrices, risk monitor | **Local-only** |
| **5** | Risk monitor | `mi_agent/risk_monitor/{migration,concentration,monitor,models}.py`, extended `config/mi/risk_monitor.yaml` | Migration matrices, per-loan flags, concentration RAG, trajectory | Runtime wiring, UI | **Local-only** |
| **5A** | Read-only audit | `docs/phase5a_*` | Regression + isolation audit; found 2 semantic hazards | (audit only) | **Doc-only** |
| **6** | MI runtime integration + Step 0 fixes | `mi_agent/mi_runtime.py`, `semantic_resolver.py`, `portfolio_reference.py`, `quantile_buckets.py`, additive `mi_query_spec.py`, `config/client/portfolio_reference_example.yaml` | `run_mi_query` dispatch (flat/state/temporal/risk) + route gating + governed chart instruction; portfolio/stage/quantile resolvers | Onboarding, Azure, UI, M&A, new chart types | **Local-only** |
| **6B** | Runtime smoke pack | `tests/test_phase6b_*`, `tests/fixtures/phase6b_flat_canonical.csv` | End-to-end proof over canonical snapshots | Multi-artefact consolidation | **Synthetic-only** |
| **6C** | Multi-artefact consolidation proof | `tests/test_phase6c_*`, `tests/helpers/phase6c_consolidation.py`, `tests/fixtures/phase6c_multi_artifact/*.csv` | Fragmented artefacts → canonical snapshots → runtime | Production mapper, onboarding | **Synthetic-only** |
| **6D** | Product proof pack | `docs/mi_runtime_product_proof_pack.md` | Business-readable proof summary | — | **Doc-only** |

> Tests by suite (collected): phase0b 53, phase1 26, phase2 24, phase3 30,
> phase4 25, phase5 23, phase6 44, 6B 19, 6C 28 (= **272** MI-phase tests) plus
> the existing **135** `mi_agent/` tests.

---

## 3. Architecture map (what each layer owns / must not own)

```
   CONFIG / CONTRACTS (declarative; data, not logic)
   ├─ config/system/fields_registry.yaml      canonical field universe (493). OWNS: the
   │                                           regulatory+analytics field vocabulary.
   │                                           MUST NOT: contain MI runtime logic.
   ├─ config/system/aliases_analytics.yaml     source-label → canonical aliases (analytics tier).
   ├─ mi_agent/mi_semantics_field_registry.yaml MI semantic layer (99; 15 virtual). OWNS: which
   │                                           fields are MI dimensions/metrics + roles. MUST NOT:
   │                                           hold data or per-client values.
   ├─ config/routes/*.yaml                      route contracts (allowed states/dims/temporal modes,
   │                                           risk_monitor on/off). OWNS: what each route may do.
   └─ config/mi/*.yaml, config/mna/*.yaml       state library, buckets, stratification, risk
                                               orderings/thresholds, M&A scorecard shape.

   PURE ANALYTICS (frame-in / frame-out; no IO, no UI, no cloud)
   └─ analytics_lib/*                           buckets, stratify, concentration, cohort. OWNS:
                                               deterministic maths. MUST NOT: read stores, render
                                               charts, call cloud/LLM, import legacy analytics/.

   HISTORY (storage-neutral)
   └─ snapshot/*  (+ adapters/local_fs.py)      SnapshotHeader/keys + SnapshotStore interface +
                                               local FS adapter. OWNS: persisting/resolving
                                               timestamped snapshots, date separation, stable keys.
                                               MUST NOT: contain analytics or Azure SDK calls.

   MI ENGINES (compose analytics over snapshots)
   ├─ mi_agent/states/*                         state assembly, selectors, route eligibility,
   │                                           temporal compare/trend, forecast fallback. OWNS:
   │                                           turning snapshot frames into MI states/series.
   └─ mi_agent/risk_monitor/*                   migration matrices, concentration, trajectory.

   RUNTIME BOUNDARY (the one entry point)
   └─ mi_agent/mi_runtime.py  run_mi_query      inspects a spec; dispatches flat | state | temporal
                                               | risk; gates by route; returns RuntimeResult +
                                               governed chart_instruction. OWNS: dispatch + gating.
                                               MUST NOT: implement analytics or render UI.

   STEP-0 GOVERNED HELPERS (library; caller-invoked, NOT auto-wired — see §6/§9)
   ├─ mi_agent/semantic_resolver.py             term → field (portfolio/stage/quantile dims) + issues
   ├─ mi_agent/portfolio_reference.py           Trakt portfolio reference config model
   └─ mi_agent/quantile_buckets.py              asset-agnostic quartile bucketing

   PROOF FIXTURES / TESTS (synthetic)
   ├─ tests/fixtures/phase6b_flat_canonical.csv             flat-path canonical CSV
   ├─ tests/fixtures/phase6c_multi_artifact/*.csv           6 source artefacts × 3 dates
   ├─ tests/helpers/phase6c_consolidation.py                hard-wired synthetic consolidation
   └─ tests/test_phase6{,b,c}_*.py                          runtime + consolidation proofs
```

The existing **`mi_agent/mi_query_{spec,validator,executor}.py` + `mi_chart_factory.py`**
remain the flat path and the **only** chart renderer.

---

## 4. Runtime walkthroughs

Entry point for all new-path queries is `run_mi_query(spec, *, semantics, data=None,
store=None, risk_config=None, ...)`. Mode is inferred:
`risk_monitor` set → **risk**; `temporal_mode ∈ {compare,trend}` → **temporal**;
`state` set → **state**; otherwise → **flat**.

1. **Flat single-CSV** (`MIQuerySpec(chart_type=bar, metric=…, dimension=…)`, `data=df`)
   → mode `flat` → **virtual-field guard** (reject snapshot-only fields with
   `virtual_field_not_available_in_flat_mode`) → existing `execute_mi_query`
   (validate → group/aggregate) → `MIQueryResult` (table) → governed
   `chart_instruction={chart_type: bar}`; with `build_chart=True` also renders via
   `create_mi_chart`. No store, no route check.

2. **total_funded latest** (`state=total_funded`, `snapshot_client_id`, `store`)
   → mode `state` → route gate `validate_state_for_route` (regulatory rejected) →
   require store + client → `SnapshotSelector.latest` → `assemble_state` →
   loan-level funded frame → `RuntimeResult(result_type="state")`; chart None;
   empty → `state_result_empty`.

3. **total_pipeline latest** — as (2) with `state=total_pipeline`; pipeline rows
   selected by `funded_status`/`pipeline_stage`.

4. **total_forecast_funded latest** — as (2); funded balance + per-pipeline-row
   forecast contribution (`forecast_funded_balance`, else balance×probability);
   metadata `forecast_funded_total`. Missing probability → `missing_forecast_probability`.

5. **funded trend (3 snapshots)** (`state=total_funded, temporal_mode=trend,
   start_date, end_date`) → mode `temporal` → `validate_temporal_request`
   (route must allow `trend`) → `resolve_range` → per-snapshot assemble + aggregate
   → rows `(reporting_date, count, balance)` → `chart_instruction={line}`;
   `<2` snapshots → `insufficient_snapshots_for_trend`.

6. **funded compare baseline/current** (`temporal_mode=compare, baseline_date,
   current_date`) → `resolve_as_of` ×2 → assemble both → balances, change, %,
   and new/exited/retained via stable `loan_id` → one-row table →
   `chart_instruction={bar}`; missing dates → `temporal_selector_incomplete`;
   missing snapshot → `missing_baseline/current_snapshot`.

7. **pipeline by stage** (`risk_monitor=concentration, state=total_pipeline,
   dimension=pipeline_stage, as_of_date`) → mode `risk` →
   `validate_risk_monitor_route` → assemble `total_pipeline` → `run_concentration`
   groups by `pipeline_stage` → table `(pipeline_stage, balance_sum,
   balance_share, status)` → `chart_instruction={bar}`.

8. **funded by portfolio** (`risk_monitor=concentration, state=total_funded,
   dimension=portfolio_id`) → as (7) over funded → groups by `portfolio_id`.
   *(Note: the caller passes `dimension="portfolio_id"`; the term "portfolio" is
   resolved by `semantic_resolver` only if the caller invokes it first — see §6/§9.)*

9. **concentration warning** — same path as (7)/(8); the `status` column is RAG
   (amber/red from `concentration_thresholds`), and `approaching_limit` flags
   near-threshold groups. Below-minimum groups → `concentration_below_minimum_threshold`.

10. **risk grade migration** (`risk_monitor=migration, dimension=internal_risk_grade,
    baseline_date, current_date`, `risk_config`) → mode `risk` →
    `run_migration` resolves both snapshots, joins raw frames on `loan_id`, classifies
    transitions using config ordering → matrix `(from, to, loan_count, balance_sum,
    movement_type)` → `chart_instruction={heatmap}`; missing key →
    `missing_stable_key_for_migration`; no ordering → `unordered_migration_dimension`.

11. **IFRS9 migration** — as (10) with `dimension=ifrs9_stage` (ordering Stage 1→2→3).

12. **PD bucket migration** — as (10) with `dimension=pd_bucket` (ordering from buckets).

---

## 5. Synthetic data walkthrough

### Phase 6B (canonical snapshots — `tests/test_phase6b_*` + `phase6b_flat_canonical.csv`)
Three canonical snapshots are built **directly in the test** (not from artefacts)
and registered for client `smoke` across `2024-01-31/02-29/03-31`. Each row is
already canonical (funded_status, pipeline_stage, balance, region, broker,
portfolio_id, grade, ifrs9_stage, pd_bucket, forecast_funding_probability).
Values prove movement: funded **300→620→620**, pipeline **90→90→50**,
forecast-funded **335→663→645**, and loan **F2 deteriorates** (grade B→C, IFRS
1→2, PD bucket worsens). The flat CSV fixture proves the legacy path.

### Phase 6C (fragmented artefacts — `tests/fixtures/phase6c_multi_artifact/`)
Six source files (synthetic):
- **borrowers.csv** (borrower_id, age, structure, region),
- **loans.csv** (loan_id, borrower_id, reporting_date, balance, rate, funded_status,
  product, amortisation, dates, grade, ifrs9_stage, pd_bucket),
- **collateral.csv** (loan_id, property_value, current_ltv, property_region — plus an
  orphan `F9`),
- **cashflows.csv** (loan_id + reporting_date, due/paid, arrears_balance, arrears_status),
- **portfolio_map.csv** (loan_id → portfolio_id/name, spv_id, acquired_portfolio_id),
- **pipeline.csv** (opportunity_id, borrower_id, expected_balance, stage, probability,
  broker/channel, product, rate).

`tests/helpers/phase6c_consolidation.py` joins them deterministically:
`borrowers→loans` on `borrower_id`; `collateral`/`portfolio_map`→`loans` on `loan_id`;
`cashflows`→`loans` on `loan_id + reporting_date`; **pipeline opportunities kept in a
distinct namespace** from funded `loan_id`. Output = **one canonical snapshot per
reporting date** with the runtime's canonical columns + derived
`forecast_funded_balance` and `months_on_book`, registered through
`LocalFsSnapshotStore`. A `LINEAGE` dict records each key field's source artefact
(e.g. LTV ← collateral, arrears ← cashflows, portfolio ← portfolio_map). The
numbers mirror 6B so the same runtime assertions hold.

**What remains hard-coded/synthetic:** column names, join keys, lineage, and the
"which artefact owns which field" mapping are all fixed for this fixture; there is
no schema inference, no mapping config, no multi-file fan-in, and no real client data.

---

## 6. What is genuinely proven vs not

| Claim | Evidence / test | Status | Caveat |
|---|---|---|---|
| Flat MI still works | `mi_agent/tests` (135) + `test_phase6{,b,c}` flat tests | **Proven** | Behaviour unchanged; spec only gained optional fields |
| `LocalFsSnapshotStore` works | `test_phase2_snapshot_layer` (24) | **Proven** | FS adapter only; CSV dtype re-inferred on load (§9) |
| `run_mi_query` dispatches to states | `test_phase6` + 6B/6C state tests | **Proven** | State path returns **loan-level frames**, not aggregated/charted |
| `run_mi_query` dispatches to temporal trend/compare | 6/6B/6C temporal tests | **Proven** | Trend/compare correct on synthetic data |
| `run_mi_query` dispatches to risk monitor | 6/6B/6C risk tests | **Proven** | Migration/concentration correct on synthetic data |
| Multi-artefact consolidation → canonical snapshots | `test_phase6c` (28) | **Partially proven** | Synthetic fixture, hard-wired joins; not a mapper |
| "Funded/pipeline by dimension" | 6B/6C concentration tests | **Partially proven** | Implemented via the **concentration** path, not a dedicated state-stratification output |
| Term semantics (portfolio/stage/quantile) | `test_phase6` resolver unit tests | **Partially proven** | Resolvers exist & pass, but **`run_mi_query` does not call them** — callers pass resolved fields (§9) |
| Chart governance preserved | flat render tests; only `bar/line/heatmap` instructions | **Proven (flat) / partial (non-flat)** | Non-flat returns a chart *instruction* but is **not auto-rendered** |
| No Streamlit dependency (new code) | import guards in phase tests | **Proven** | Legacy `mi_agent/streamlit_mi_agent.py` still exists (untouched) |
| No Azure dependency | import guards | **Proven** | — |
| No Annex 2 impact | git-diff guards in phase tests | **Proven** | — |
| Production onboarding | — | **Not proven** | Not built |
| Production mapping engine | — | **Not proven** | Consolidation is synthetic/hard-wired |
| UI | — | **Not proven** | Not built |
| M&A Agent runtime | — | **Not proven** | Config skeleton only |

---

## 7. Regression & safety checklist

All commands run from the repo root. (Environment needs `pandas`, `numpy`,
`pyyaml`, `plotly`, `rapidfuzz`, `pytest`; `openpyxl`/`lxml` are unrelated
pre-existing gaps for onboarding/XML tests only.)

| Check | Command | Expected |
|---|---|---|
| Existing flat MI Agent unbroken | `python3 -m pytest mi_agent/tests/ -q` | **135 passed** |
| Semantic registry integrity (counts/version) | `python3 -m pytest mi_agent/tests/test_mi_semantics_cleanup.py mi_agent/tests/test_mi_semantics_buckets.py -q` | passed (v0.3.0 / 99 fields) |
| New runtime unbroken | `python3 -m pytest tests/test_phase6_mi_runtime.py -q` | **44 passed** |
| Full MI-phase + agent sweep | `python3 -m pytest mi_agent/tests/ tests/test_phase0b_mi_mna_foundations.py tests/test_phase1_analytics_lib.py tests/test_phase2_snapshot_layer.py tests/test_phase3_mi_state_assembler.py tests/test_phase4_temporal_mi.py tests/test_phase5_risk_monitor.py tests/test_phase6_mi_runtime.py tests/test_phase6b_mi_runtime_smoke_pack.py tests/test_phase6c_multi_artifact_consolidation.py -q` | **407 passed** |
| No Annex 2/regulatory files touched | included as guard tests in each phase suite | guard tests pass |

---

## 8. Owner test script (run these yourself)

| # | Command | What it tests | Expected output | A failure would mean |
|---|---|---|---|---|
| 1 | `python3 -m pytest mi_agent/tests/ -q` | The original flat MI Agent (parse/validate/execute/chart) | `135 passed` | The legacy MI path regressed — stop and investigate |
| 2 | `python3 -m pytest tests/test_phase6_mi_runtime.py -q` | The runtime boundary: dispatch, route gating, Step 0 resolvers | `44 passed` | Dispatch/gating or resolver semantics broken |
| 3 | `python3 -m pytest tests/test_phase6b_mi_runtime_smoke_pack.py -q` | End-to-end MI over canonical snapshots | `19 passed` | The state/temporal/risk runtime is broken on clean data |
| 4 | `python3 -m pytest tests/test_phase6c_multi_artifact_consolidation.py -q` | Fragmented artefacts → consolidation → snapshots → runtime | `28 passed` | Consolidation/lineage/registration or downstream MI broke |
| 5 | `python3 -m pytest tests/test_phase5_risk_monitor.py tests/test_phase4_temporal_mi.py tests/test_phase3_mi_state_assembler.py -q` | Risk monitor, temporal, state assembler in isolation | all passed | A specific engine layer is broken |
| 6 | `python3 -c "import yaml;d=yaml.safe_load(open('mi_agent/mi_semantics_field_registry.yaml'));print(d['metadata']['version'],d['metadata']['field_count'])"` | Registry version/count is what you expect | `0.3.0 99` | Registry drift — downstream resolution may shift |

To watch a single query run, read `tests/test_phase6b_mi_runtime_smoke_pack.py`
(each test is a self-contained `run_mi_query` example) and
`tests/helpers/phase6c_consolidation.py` (the consolidation walkthrough).

---

## 9. Red flags / areas to inspect manually

1. **Semantic resolver is not wired into the runtime.** `run_mi_query` consumes
   `spec.dimension`/`spec.state` literally; it does **not** call
   `semantic_resolver`. The "portfolio → portfolio_id", "stage → pipeline_stage",
   quantile-bucket behaviours are proven only at the **library unit level** and
   only take effect if a caller resolves terms first (the 6B/6C tests do this
   manually). **Inspect:** whether you expect natural-language term resolution to
   be part of the runtime, or a deliberate separate step.
2. **Portfolio reference config is example-only.** `portfolio_reference_example.yaml`
   is illustrative; there is no per-client config populated and the runtime does
   not load it. "portfolio" without config correctly raises
   `missing_portfolio_reference_config` — but only when the resolver is called.
3. **Stage semantics depend on context being passed.** "stage" → `pipeline_stage`
   only in a pipeline context; in funded/M&A/regulatory contexts it raises
   `invalid_stage_context`. Again, only when the resolver is invoked.
4. **Virtual fields (15).** `portfolio_id, spv_id, acquired_portfolio_id,
   funded_status, pipeline_stage, reporting_date, cut_off_date, upload_timestamp,
   acquisition_date, spv_transfer_date, forecast_funding_date,
   forecast_funding_probability, forecast_funded_balance, number_of_borrowers,
   months_on_book` are flagged virtual: rejected in flat mode, expected to be
   materialised by snapshots. **Inspect:** that every snapshot you care about
   actually materialises the ones you query.
5. **Quantile buckets are not exercised by any runtime path.** `quantile_buckets.py`
   is unit-tested but no `run_mi_query` path calls it; the 6B/6C "by region/
   portfolio/stage" views group by **raw categorical columns**, not quantile
   bands. **Inspect:** whether asset-agnostic quantile banding is needed in the
   runtime before relying on it.
6. **Forecast probability source.** Priority is `forecast_funded_balance` →
   row `forecast_funding_probability` → config stage→probability → flagged
   missing. The config path is only used if `stage_probabilities`/`risk_config`
   is passed. **Inspect:** which source your data will actually use, and that
   probabilities are never invented.
7. **Risk monitor orderings.** Improve/deteriorate depends on
   `config/mi/risk_monitor.yaml deterioration_orderings`. `internal_risk_stage`
   is intentionally unordered (→ "changed"). **Inspect:** that your client's grade
   scale matches the configured ordering, or migrations read as "changed".
8. **Snapshot CSV dtype caveat.** `LocalFsSnapshotStore` round-trips loan rows as
   CSV; dtypes are re-inferred on load. Downstream code coerces numerics/dates
   defensively, but mixed-type columns could surprise a naive consumer. A typed
   (Parquet) store is deferred.
9. **Synthetic consolidation boundary.** `phase6c_consolidation.py` is a proof
   helper: hard-wired joins, no schema inference, an intentional orphan row to
   exercise issues. **Do not** read it as a production mapper.

---

## 10. Recommended next decision (options, not a default)

This is a decision for you — each option below is the right move under a
different condition.

| Option | Choose it when… |
|---|---|
| **Pause & review only** | You want to internalise the architecture and validate the synthetic results before any further spend. (This document supports that.) |
| **Produce screenshots / HTML demo** | You need to show stakeholders something tangible *now* and are comfortable it’s synthetic; renders trend/concentration via the governed chart factory. Low risk, no new engine. |
| **Harden runtime tests** | You want more confidence before production (e.g. wire the semantic resolver into `run_mi_query`, add quantile-bucket runtime paths, property/edge tests, multi-client snapshots) — the cheapest way to raise the "partially proven" rows to "proven". |
| **Build production onboarding/consolidation** | You have real client artefacts and need the hard-wired 6C helper replaced by a config-driven mapper + data-quality workflow. Highest value toward "real MI", highest effort. |
| **Build Azure adapter** | Deployment target is decided and you need cloud ingestion/storage; implement `AzureBlobSnapshotStore` behind the existing interface + thin ingress. Do after onboarding is real. |
| **Build M&A Agent runtime** | There is near-term M&A diligence demand; reuse the funded/concentration paths point-in-time behind the M&A route contract. |
| **Build UI / export layer** | Stakeholders need self-serve access; wrap `run_mi_query` outputs + governed charts into an app/export. Do after at least one production-data path exists. |

**Suggested sequencing if you do proceed:** *harden runtime tests* (cheap, de-risks)
→ *production onboarding/consolidation* (unlocks real data) → *Azure* (deploy) →
*UI/export* (access) → *M&A runtime* (parallel, demand-driven).

---

### Document section summary
1. Executive summary · 2. Phase 0–6D inventory table · 3. Architecture map +
ownership · 4. Twelve runtime walkthroughs · 5. 6B/6C synthetic data · 6.
Proven-vs-not table · 7. Regression checklist + commands · 8. Owner test script ·
9. Nine red flags · 10. Seven next-step options.

*Review/comprehension only — no code, configs, or tests were changed.*
