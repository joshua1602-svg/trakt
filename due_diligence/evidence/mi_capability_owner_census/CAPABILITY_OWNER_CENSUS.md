# MI CAPABILITY-TO-OWNER CENSUS

> **READ-ONLY.** No production file was modified. No live model call was made. Nothing was deployed. Every figure below was either traced in source or measured by a deterministic offline run against committed corpora and fixtures.

```
CENSUS_SHA = 4337d71de1e0c92e66d09511bb5471a9a4a6d475
BRANCH     = claude/mi-capability-owner-census-9at6it
WORKING TREE AT CENSUS TIME = clean (git status --porcelain: 0 entries)
```

## 0. What the brief's terms map onto in this repository

The census must not invent vocabulary, so these bindings are stated before anything is counted.

| Brief's term | This repository |
|---|---|
| **GovernedQueryPlan** | mi_agent/query_plan.py — QueryPlan / AnalyticalScope / PlannedOutput / ScopeDelta / Predicate. The repository contains no symbol literally named `GovernedQueryPlan`; this is the contract the brief refers to. |
| **the new Opus path** | mi_agent.llm_query_parser (default model claude-opus-5, mi_agent/mi_agent_config.py:41) proposes an MIQuerySpec. mi_agent/parsed_question.py:152 then LIFTS that spec into a QueryPlan via query_plan_adapter.compiled_spec_for and executes the COMPILED spec. Opus never emits a QueryPlan directly; the plan is a lift of the parser's spec, deterministic and model-independent. |
| **CONNECTED** | The lift succeeds (plan_from_spec returns a plan) AND the request is executed by the generic deterministic executor that the compiler targets. A liftable spec claimed by a specialist route is NOT connected: the plan was the parse's semantic contract, but the route executes through its own owner, which no plan reaches. |
| **second plan layer** | mi_agent_api/analytical_plan.py carries a DIFFERENT plan artefact (Plan/Step over derived primitives), driven by question_interpretation.schema.QuestionInterpretation. Six routes are converted onto it. It is not the GovernedQueryPlan and the two do not compose today. |

There is **no symbol named `GovernedQueryPlan`** in the codebase. The contract the brief describes is `mi_agent/query_plan.py::QueryPlan` + `AnalyticalScope` + `PlannedOutput` + `ScopeDelta` + `Predicate`. Throughout this document *the plan* means that contract.

A second, unrelated plan artefact exists — `mi_agent_api/analytical_plan.py::Plan`, driven by `question_interpretation.schema.QuestionInterpretation`. Six routes are already converted onto it (Conversions 1, 2, 4, 5, geo, C7). **The two plan layers do not compose today**, and several capabilities are 'migrated' onto one while being 'not connected' to the other. Both are recorded; where the distinction matters it is called out.

---

## 1. Measured baseline

Method: `probes/lift_census_probe.py` and `probes/route_claim_probe.py`, read-only, offline, deterministic parser (`llm_enabled=False`), no network and no model call. Raw output in `measurements/`.

| Measure | Value |
|---|---|
| Distinct corpus questions (stage 1 + stage 2) | **882** |
| Liftable into a `QueryPlan` (`plan_from_spec` returns a plan) | **660 (74.8%)** |
| Not liftable | 222 |
| Claimed by a specialist recogniser | 168 |
| Fall through to the generic deterministic executor | 714 |
| **Generic AND liftable — the plan is the contract *and* reaches the owner** | **607 (68.8%)** |
| `MIQuerySpec` fields | 76 |
| …modelled by `AnalyticalScope`/`PlannedOutput` | 18 |
| …unmodelled (any one at a non-default value declines the lift) | 58 |

### Why the 222 declines decline

| Reason | Questions |
|---|---|
| ranking / superlative (ranking_mode, sort_by, top_n, limit, sort_direction) | 109 |
| time axis / series (line chart whose x is a date, not a grouping dim) | 35 |
| forecast (forecast_mode, forecast_question, forecast_target_value) | 28 |
| risk limits (risk_limit_query, risk_limit_category, risk_monitor_mode) | 20 |
| unavailable_filters (a DISCLOSURE field, not a semantic) | 12 |
| temporal compare (temporal_mode, compare_periods) | 7 |
| share aggregation (not in query_plan.OPERATIONS) | 5 |
| funded bridge (bridge_query) | 3 |
| cohort progression (cohort_progression) | 3 |

**Read that table carefully.** Not one line says *Trakt cannot calculate this*. Every line names a **slot the plan does not have**, for arithmetic the executor already performs.

### Route claims, measured

Offline recognition without data roots, so routes whose recognition needs a resolved dataset or a frame under-report; those are covered by name in `probes/targeted_route_probe.py`.

```
CLAIMING ROUTE                         N   liftable   not
<generic_mi_workflow>                714      607    107
analytical_composition                35       29      6
evolution                             33        0     33
forecast_extrapolation                26        0     26
concentration_analysis                19       10      9
risk_limits                           18        0     18
geo_exposure                          12        4      8
period_change_analysis                 7        2      5
portfolio_summary                      4        4      0
cohort_conversion                      4        2      2
temporal_compare                       4        0      4
funded_bridge                          3        0      3
portfolio_risk_comparison              2        2      0
scenario                               1        0      1
```

### Deterministic execution proofs

**Pipeline AMOUNT BY STAGE is owned by the generic executor and is carried by the QueryPlan on the live path.**

- *How:* ParsedQuestion.parse('How much is in the pipeline by stage?') -> meta['semantic_contract'] == 'query_plan'; execute_mi_query over pipeline_contract.load_prepared_pipeline(<committed fixture>).
- *Fixture:* `tests/fixtures/client_001_mi_pack/pipeline/2025-11-01/M2L_KFI_and_Pipeline_2025_11_01.csv`
- *Result:* table, 4 rows: APPLICATION 485,000 / COMPLETED 450,000 / OFFER 420,000 / KFI 400,000; group_field_keys=['pipeline_stage']; a full reconciliation block was emitted.
- *Verdict:* CONFIRMED — the brief's worked example reproduces.

**Grouped TOP-N ranking already executes inside the same owner; it is the PLAN that cannot carry it.**

- *How:* Same fixture. 'Top 2 stages by amount.' parses to ranking_mode='grouped', sort_by='current_outstanding_balance', top_n=2.
- *Result:* execute_mi_query returned exactly 2 ranked rows. plan_from_spec returned None; meta['semantic_contract'] is None.
- *Verdict:* CONFIRMED — capability EXISTS, plan NOT representable.

**A pipeline-stage NARROWING is applied from raw text AFTER the plan, not from the plan.**

- *How:* Same fixture. 'What is the amount for offer stage cases?' parses with spec.filters == {} and executes ungrouped-by-request over all four stages when execute_mi_query is called directly.
- *Result:* The narrowing is injected later by mi_agent_workflow from question_interpretation.lexical.pipeline_stage_request(question).
- *Verdict:* CONFIRMED — raw-text coupling on the CONNECTED path.

**The plan layer's own contract tests pass at CENSUS_SHA.**

- *How:* python -m pytest mi_agent/tests/test_query_plan_adapter.py test_query_plan_compiler.py test_query_plan_contracts.py test_shadow_replay.py test_query_plan_execution.py test_query_plan_is_the_live_contract.py test_query_plan_reconciliation.py
- *Result:* 80 passed, 114 subtests passed. No product file modified.
- *Verdict:* CONFIRMED

---

## OUTPUT 1 — EXECUTIVE CAPABILITY MAP

| # | Capability | Dataset | Deterministic calculation owner | Current | Temporal | Filters | Dimensions | Plan repr. | New-plan connectivity | Migration complexity |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | generic_analysis — funded current aggregation | funded | `mi_agent.mi_query_executor.execute_mi_query` | yes | current | YES_GENERIC | YES_GENERIC | YES | CONNECTED | TRIVIAL |
| 2 | generic_analysis — pipeline current aggregation (incl. pipeline stage analysis) | pipeline | `mi_agent.mi_query_executor.execute_mi_query` | yes | current | YES_GENERIC | YES_GENERIC | YES | CONNECTED | SMALL |
| 3 | forecast-view analysis (derived funded + weighted pipeline frame) | derived | `mi_agent.mi_query_executor.execute_mi_query` | yes | current | YES_GENERIC | YES_LIMITED | YES | CONNECTED | SMALL |
| 4 | superlative / ranking (grouped top-N and loan-level) | funded\|pipeline | `mi_agent.mi_query_executor._apply_top_n` | yes | current | YES_GENERIC | YES_GENERIC | NO | NOT_CONNECTED | SMALL |
| 5 | share / contribution analysis | funded\|pipeline | `mi_agent.mi_query_executor._execute_share` | yes | current | YES_GENERIC | YES_GENERIC | NO | NOT_CONNECTED | MEDIUM |
| 6 | loan-level listing, scatter and bubble | funded\|pipeline | `mi_agent.mi_query_executor._execute_loan_level` | yes | current | YES_GENERIC | axes rather than groupings … | NO | NOT_CONNECTED | MEDIUM |
| 7 | stratification / bucketed dimensions | funded\|pipeline | `mi_agent.mi_query_executor.execute_mi_query` | yes | current | YES_GENERIC | YES_GENERIC | YES | CONNECTED | TRIVIAL |
| 8 | funded temporal analysis / evolution (time series) | funded | `mi_agent_api.temporal_query.execute_temporal` | — | series, weekly, monthly | YES_GENERIC | YES_GENERIC | NO | NOT_CONNECTED | MEDIUM |
| 9 | pipeline evolution (weekly history, funnel evolution) | pipeline | `mi_agent_api.evolution.pipeline_evolution` | — | series, weekly, movement | YES_LIMITED | YES_LIMITED | NO | NOT_CONNECTED | MEDIUM |
| 10 | temporal comparison (two named reporting periods) | both | `mi_agent_api.temporal_compare.compare_periods` | — | compare | NO | NO | NO | NOT_CONNECTED | MEDIUM |
| 11 | portfolio_summary (funded headline position) | funded | `mi_agent_api.movement_summary.portfolio_summary` | yes | current | NO | YES_LIMITED | PARTIAL | NOT_CONNECTED | MEDIUM |
| 12 | pipeline_summary (pipeline headline position) | pipeline | `mi_agent_api.pipeline_contract.compute_pipeline_snapshot` | yes | current | NO | YES_LIMITED | PARTIAL | PARTIAL | SMALL |
| 13 | period_movement (month-on-month with attribution) | funded | `mi_agent_api.movement_summary.period_movement` | — | compare, movement | NO | YES_LIMITED | PARTIAL | NOT_CONNECTED | MEDIUM |
| 14 | period change analysis (Business-Semantics-Registry driven) | funded (two snapshots) | `mi_agent.period_change.workflow.run_period_change_analysis` | — | compare, movement | YES_LIMITED | YES_GENERIC | NO | NOT_CONNECTED | LARGE |
| 15 | funded bridge / balance attribution waterfall | funded (period range) | `mi_agent_api.evolution.funded_bridge` | — | movement, compare | YES_LIMITED | YES_GENERIC | NO | NOT_CONNECTED | LARGE |
| 16 | cohort / vintage analysis (static pool) | funded | `mi_agent_api.cohorts.cohort_analysis` | yes | current (point-in-time static poo… | YES_LIMITED | vintage (year or configured… | YES | PARTIAL | SMALL |
| 17 | cohort progression (metric progression by vintage across reporting dates) | funded (multi-period) | `mi_agent_api.evolution.funded_cohort_progression` | — | series, movement | YES_LIMITED | vintage x reporting date | NO | NOT_CONNECTED | LARGE |
| 18 | cohort conversion (cumulative KFI -> Funded) | pipeline history | `mi_agent_api.pipeline_history.build_historical_completion_model` | — | series (cohort cumulative) | NO | stage | NO | NOT_CONNECTED | LARGE |
| 19 | pipeline stage movement / transitions | pipeline (two weekly extracts) | `mi_agent_api.movement_detail.stage_transition_events` | — | movement (two governed weekly ext… | YES_LIMITED | stage x stage (the transiti… | NO | NOT_CONNECTED | LARGE |
| 20 | pipeline movement summary (all stages, one interval) | pipeline (two weekly extracts) | `mi_agent_api.pipeline_movement_summary.build` | — | movement | NO | all stages | NO | NOT_CONNECTED | LARGE |
| 21 | geography analysis (UK ITL3 exposure) | funded | `mi_agent_api.geo.exposure_by_itl3` | yes | current | YES_LIMITED | ITL3 area | PARTIAL | PARTIAL | MEDIUM |
| 22 | concentration analysis (governed dimension distribution) | funded | `mi_workflows.engine.ranked_distribution` | yes | current | YES_LIMITED | YES_GENERIC | PARTIAL | NOT_CONNECTED | MEDIUM |
| 23 | limit assessment / approved concentration tests and headroom | funded + approved limit configuration | `mi_agent.concentration_tests.evaluation.evaluate_active_tests` | yes | current; compare_concentration_pe… | YES_LIMITED | the tested dimension per ap… | NO | NOT_CONNECTED | LARGE |
| 24 | borrowing base / facility utilisation / eligibility | funded + facility configuration + governed concentration results | `mi_agent.borrowing_base.calculator.calculate` | yes | current | YES_LIMITED | ineligibility reason; conce… | NO | NOT_CONNECTED | LARGE |
| 25 | forecast / expected funding — run-rate extrapolation and milestones | funded history + pipeline | `mi_agent_api.forecast_extrapolation.run_rate_model` | — | series (forward), other (mileston… | NO | NO | NO | NOT_CONNECTED | LARGE |
| 26 | forecast bridge (funded + probability-weighted pipeline) | funded + pipeline (aggregate composition) | `mi_agent_api.forecast_bridge.compute_forecast_bridge` | yes | current + forward | NO | portfolio (projections) | NO | NOT_CONNECTED | LARGE |
| 27 | scenario / what-if on the completion run-rate | derived from the forecast base | `mi_agent_api.scenario.apply_scenario` | — | other (forward, perturbed) | NO | NO | NO | NOT_CONNECTED | LARGE |
| 28 | portfolio risk comparison (two governed scopes at one date) | funded | `mi_workflows.portfolio_risk_comparison.run_portfolio_risk_comparison` | yes | current (one reporting date, two … | YES_LIMITED | the compared fields | PARTIAL | NOT_CONNECTED | LARGE |
| 29 | analytical composition layer (multi-capability plans) | varies by plan | `NONE OF ITS OWN` | — | delegated | NO | delegated | NO | NOT_CONNECTED | LARGE |
| 30 | source portfolio scope (Direct / Acquired / named source / cohort) | cross-cutting | `NOT A CALCULATION` | — | n/a | YES_GENERIC | source_portfolio_id / sourc… | PARTIAL | PARTIAL | TRIVIAL |
| 31 | seasoning / front-book vs back-book population | funded | `NOT A CALCULATION` | — | n/a | YES_GENERIC | seasoning segment | YES | PARTIAL | SMALL |
| 32 | week-on-week movement attribution (hover / drill) | pipeline (two weekly extracts) | `mi_agent_api.movement_detail.movement_components` | — | movement (two adjacent weekly ext… | YES_LIMITED | any dimension the prepared … | NO | NOT_CONNECTED | LARGE |
| 33 | FORWARD-LOOKING approved-limit projection | funded + pipeline + approved limit configuration | `NONE. This` | — | none | NO | n/a | NO | NOT_CONNECTED | LARGE |

---

## OUTPUT 2 — CONNECTIVITY MAP

_Groups 2-5 are not mutually exclusive: a capability can be partially connected AND raw-text coupled, and is listed in both. Group 5 lists only capabilities that are not already CONNECTED — the four CONNECTED capabilities also carry post-plan raw-text reads (dataset, lens, pipeline stage, missing-dimension policy), recorded in their individual records and in the raw-text audit. That is why CONNECTED here means *the plan reaches the owner*, not *the plan decides everything*._

### 1. ALREADY CONNECTED  (4)

- **generic_analysis — funded current aggregation**
- **generic_analysis — pipeline current aggregation (incl. pipeline stage analysis)**
- **forecast-view analysis (derived funded + weighted pipeline frame)**
- **stratification / bucketed dimensions**

### 2. PARTIALLY CONNECTED  (5)

- **pipeline_summary (pipeline headline position)** — The measurable half already reaches the plan through the generic path; the headline block does not, because this route claims the question before the generic path runs.
- **cohort / vintage analysis (static pool)** — A bare 'balance by vintage year' already reaches the generic executor through the plan (vintage_year is a prepared column). The /mi/cohorts SERVICE shape — book share, metricsAvailable — is a different owner nothing routes a plan to.
- **geography analysis (UK ITL3 exposure)** — A geography-basis slot on the scope. 4 of 12 corpus geo-claimed questions lift today; they lift WITHOUT the basis, which is the risk.
- **source portfolio scope (Direct / Acquired / named source / cohort)** — The seam exists and is unused. plan_from_spec already accepts dataset / portfolio_lens / period kwargs; nothing passes them.
- **seasoning / front-book vs back-book population** — Where the parser resolves the segment into spec.filters the plan carries it. Where seasoning is resolved later (the analytical layer's populations module) it does not.

### 3. REPRESENTABLE BUT NOT CONNECTED  (4)

- **portfolio_summary (funded headline position)** — The route executes through analytical_plan (the OTHER plan layer). Nothing routes a QueryPlan here. A liftable summary spec is still lifted at parse — measured 4 of 4 corpus summary questions — and then the route ignores it.
- **period_movement (month-on-month with attribution)** — As record 11 — it executes through the other plan layer.
- **concentration analysis (governed dimension distribution)** — Ordering/limit slot (record 4) plus a registry-category reference. 10 of 19 corpus concentration questions lift today and are then claimed by this route instead.
- **portfolio risk comparison (two governed scopes at one date)** — Sibling (non-nested) scopes plus a comparison relationship.

### 4. NOT YET REPRESENTABLE IN GovernedQueryPlan  (15)

- **superlative / ranking (grouped top-N and loan-level)** — One missing plan semantic: an ORDERING (measure + direction) and a LIMIT, at plan or output level. The arithmetic already exists in the executor the compiler targets.
- **share / contribution analysis** — Two slots: the operation vocabulary, and a denominator scope reference (the one direction ScopeDelta deliberately forbids).
- **loan-level listing, scatter and bubble** — No plan concept for 'the rows themselves, unaggregated'.
- **funded temporal analysis / evolution (time series)** — A temporal axis on the plan: {grain, window or explicit period list} plus the rule that a series is N executions of one scope. The compiler already does N-executions-per-plan for N populations; a series is the same shape over periods.
- **pipeline evolution (weekly history, funnel evolution)** — Temporal axis (as record 8) and a pipeline-series binding.
- **temporal comparison (two named reporting periods)** — A comparison relationship between two scopes, plus a period-pair slot. Note the compiler ALREADY executes several scopes per plan — what is missing is the statement that one is the baseline for the other.
- **period change analysis (Business-Semantics-Registry driven)** — Comparison relationship + movement + distribution-shift concepts.
- **funded bridge / balance attribution waterfall** — Movement decomposition semantics + period range.
- **cohort conversion (cumulative KFI -> Funded)** — Entity-tracking-across-snapshots semantics.
- **pipeline stage movement / transitions** — Entity-state-transition semantics.
- **pipeline movement summary (all stages, one interval)** — As record 19.
- **borrowing base / facility utilisation / eligibility** — Facility/eligibility semantics.
- **forecast bridge (funded + probability-weighted pipeline)** — Cross-dataset aggregate composition.
- **analytical composition layer (multi-capability plans)** — No relationship defined between the two plan artefacts.
- **week-on-week movement attribution (hover / drill)** — Movement decomposition semantics (shared with record 15).

### 5. RAW-TEXT-COUPLED / REQUIRES DEEPER MIGRATION  (8)

- **cohort progression (metric progression by vintage across reporting dates)** — Temporal axis + static-pool membership semantics.
- **geography analysis (UK ITL3 exposure)** — A geography-basis slot on the scope. 4 of 12 corpus geo-claimed questions lift today; they lift WITHOUT the basis, which is the risk.
- **limit assessment / approved concentration tests and headroom** — Threshold/limit semantics and a configuration reference.
- **forecast / expected funding — run-rate extrapolation and milestones** — Projection semantics.
- **scenario / what-if on the completion run-rate** — Scenario/perturbation semantics.
- **portfolio risk comparison (two governed scopes at one date)** — Sibling (non-nested) scopes plus a comparison relationship.
- **source portfolio scope (Direct / Acquired / named source / cohort)** — The seam exists and is unused. plan_from_spec already accepts dataset / portfolio_lens / period kwargs; nothing passes them.
- **seasoning / front-book vs back-book population** — Where the parser resolves the segment into spec.filters the plan carries it. Where seasoning is resolved later (the analytical layer's populations module) it does not.

### 6. GENUINE CAPABILITY GAP  (1)

- **FORWARD-LOOKING approved-limit projection** — Trakt does not calculate this anywhere; the route declines rather than substituting today's position.

---

## OUTPUT 3 — OWNER MAP

```
TOTAL_USER_FACING_CAPABILITIES  = 33
TOTAL_DISTINCT_CALCULATION_OWNERS = 21
COMPOSING LAYERS THAT CALCULATE NOTHING = 8
```

```
mi_agent.mi_query_executor.execute_mi_query
    -> funded current aggregation (count/sum/avg/weighted/median/min/max)
    -> pipeline current aggregation, including stage counts AND stage amounts
    -> forecast-view aggregation over the derived frame
    -> grouped top-N ranking (_apply_top_n) and loan-level ranking (_execute_ranked_loans)
    -> share (_execute_share) and contribution (_execute_contribution)
    -> loan-level listing, scatter and bubble (_execute_loan_level)
    -> bucketed / stratified dimensions (over materialised bucket columns)
    -> multi-measure sets over one population (the P1E path)
    -> 1, 2 and 3+ dimension grouping
    -> shared predicate execution (_apply_filters / governed_predicate_mask)
    -> EVERY per-period figure in a temporal series, via temporal_query.execute_temporal
    -> reconciliation and applied-predicate evidence for all of the above

mi_workflows.engine (aggregate / distribution / ranked_distribution / compare_values / directionality_verdict / mixed_currency_guard)
    -> concentration analysis
    -> portfolio risk comparison
    -> the analytical layer's portfolio_snapshot and population_profile
    -> a SECOND ranking implementation alongside the executor's _apply_top_n

mi_agent_api.evolution (funded_frames / assemble_funded_evolution / pipeline_evolution / pipeline_funnel_evolution / funded_bridge / funded_cohort_progression)
    -> funded dashboard time series (five metrics)
    -> pipeline weekly series and funnel evolution
    -> funded balance attribution bridge
    -> cohort progression across reporting dates
    -> the period stack every converted plan route reads

mi_agent_api.movement_summary (portfolio_summary / period_movement / _delta / _regional_exposure / _cohorts)
    -> funded headline position
    -> month-on-month movement with regional and source attribution

mi_agent.period_change (workflow.run_period_change_analysis + calculations + distribution + bridge + ranking)
    -> governed period change analysis
    -> the analytical layer's period_movement capability

mi_agent_api.movement_detail (movement_components / stage_transition_events / transition_matrix / build_stage_transition_detail)
    -> pipeline stage movement and transitions (chat)
    -> all-stage pipeline movement summary (chat)
    -> week-on-week movement attribution (hover / drill)
    -> the weekly brief's movement inputs
    -> PPTX deck movement slides

mi_agent_api.pipeline_contract (load_prepared_pipeline / compute_pipeline_snapshot / collect_weekly_history / cap_breakdown)
    -> pipeline headline position
    -> the prepared pipeline frame EVERY pipeline capability operates on
    -> the analytical layer's pipeline_stock and pipeline_completion_forecast

mi_agent_api.forecast_extrapolation (run_rate_model / kfi_conversion_model / build_extrapolation)
    -> run-rate forecast and milestone solving
    -> the analytical layer's completion_run_rate and threshold_projection
    -> the base the scenario engine perturbs

mi_agent.concentration_tests.evaluation.evaluate_active_tests
    -> approved concentration tests, headroom and breach status (chat)
    -> the Risk Limits workspace
    -> Copilot limit questions
    -> the concentration adjustment inside the borrowing base
    -> the analytical layer's concentration_limits

analytics_lib.concentration (group_shares / top_n_concentration / limit_usage / rag_status)
    -> mi_agent_api.risk_limits.compute_risk_limits
    -> mi_agent.risk_monitor.concentration
    -> the Streamlit-era risk monitor's logic, preserved

analytics_lib.cohort (cohort_table / add_cohort_period / months_on_book)
    -> vintage / static-pool analysis
    -> cohort formation and static-pool membership
    -> the months_on_book dimension the funded preparation derives

analytics_lib.buckets.materialise_buckets + mi_agent.quantile_buckets
    -> every bucketed dimension in the registry (LTV, age, ticket size, rate, time-on-book)
    -> so every 'by X bucket' question in records 1-3

mi_agent.borrowing_base.calculator.calculate
    -> borrowing base, eligibility and facility utilisation (chat)
    -> GET /mi/borrowing-base
    -> the React Eligibility & Concentrations tab

mi_agent_api.temporal_query.execute_temporal
    -> MI-Query temporal series (delegating every FIGURE to the executor)
    -> time x dimension series

mi_agent_api.temporal_compare.compare_periods
    -> two-period comparison delta / pct / direction

mi_agent_api.geo.exposure_by_itl3
    -> ITL3 geographic exposure

mi_agent_api.cohorts (cohort_analysis / cohort_formation / cohort_static_pool)
    -> vintage analysis
    -> cohort membership for progression

mi_agent_api.forecast_bridge.compute_forecast_bridge
    -> forecast funded balance (funded + weighted pipeline)
    -> pipeline watchlist
    -> the analytical layer's funded_balance_forecast

mi_agent_api.pipeline_history.build_historical_completion_model
    -> cohort conversion (KFI -> Funded)
    -> the empirical stage probabilities the pipeline forecast prefers over config

mi_agent_api.scenario.apply_scenario
    -> what-if on the completion run-rate

mi_agent.states (assembler / temporal.compare / temporal.trend / forecast)
    -> the mi_runtime state/temporal/risk engines — reachable from the CLI interpreter and the simulation harness, NOT from the HTTP serving path at this SHA
    -> assembler.total_forecast_funded, whose formula forecast_bridge reuses

```

**Layers that COMPOSE and calculate nothing** (not owners):

| Layer | Role |
|---|---|
| `mi_workflows.analytical.orchestrator` | runs registered executors; tests parse the source to prove no adapter computes a financial result |
| `mi_agent_api.analytical_plan` | interpretation -> plan -> EXISTING primitives; structurally forbidden from reading the question |
| `mi_agent.query_plan_compiler / query_plan_execution` | QueryPlan -> the specs the executor already accepts; no arithmetic |
| `mi_agent_api.pipeline_movement_summary` | composes movement_detail's payload |
| `mi_agent_api.insight_engine` | assembles the weekly brief from governed movement, concentration and funnel outputs |
| `mi_agent_api.borrowing_base_query` | answers from the governed borrowingBase envelope |
| `mi_agent_api.concentration_query` | answers from the governed concentration-test envelope |
| `mi_workflows.analytical.narrative` | findings in, prose out; may only read findings |

---

## OUTPUT 4 — MIGRATION GAP MAP

_Classification only. No sequencing, no recommendation._

| Gap class | Capabilities | Count |
|---|---|---|
| `DISPATCH_ONLY` | portfolio_summary (funded headline position); pipeline_summary (pipeline headline position); period_movement (month-on-month with attribution); cohort / vintage analysis (static pool); concentration analysis (governed dimension distribution) | 5 |
| `PLAN_TO_SPEC_ADAPTER` | portfolio_summary (funded headline position); pipeline_summary (pipeline headline position); period_movement (month-on-month with attribution); cohort / vintage analysis (static pool); concentration analysis (governed dimension distribution) | 5 |
| `DATASET_BINDING` | generic_analysis — funded current aggregation; generic_analysis — pipeline current aggregation (incl. pipeline stage analysis); forecast-view analysis (derived funded + weighted pipeline frame); pipeline evolution (weekly history, funnel evolution); forecast bridge (funded + probability-weighted pipeline); source portfolio scope (Direct / Acquired / named source / cohort) | 6 |
| `TEMPORAL_BINDING` | funded temporal analysis / evolution (time series); pipeline evolution (weekly history, funnel evolution); temporal comparison (two named reporting periods); period_movement (month-on-month with attribution); period change analysis (Business-Semantics-Registry driven); funded bridge / balance attribution waterfall; cohort progression (metric progression by vintage across reporting dates) | 7 |
| `RECEIPT_ADAPTATION` | generic_analysis — funded current aggregation; generic_analysis — pipeline current aggregation (incl. pipeline stage analysis); forecast-view analysis (derived funded + weighted pipeline frame); superlative / ranking (grouped top-N and loan-level) | 4 |
| `MISSING_PLAN_SEMANTIC` | superlative / ranking (grouped top-N and loan-level); share / contribution analysis; loan-level listing, scatter and bubble; temporal comparison (two named reporting periods); portfolio_summary (funded headline position); period_movement (month-on-month with attribution); period change analysis (Business-Semantics-Registry driven); funded bridge / balance attribution waterfall; cohort progression (metric progression by vintage across reporting dates); cohort conversion (cumulative KFI -> Funded); pipeline stage movement / transitions; pipeline movement summary (all stages, one interval); geography analysis (UK ITL3 exposure); concentration analysis (governed dimension distribution); limit assessment / approved concentration tests and headroom; borrowing base / facility utilisation / eligibility; forecast / expected funding — run-rate extrapolation and milestones; forecast bridge (funded + probability-weighted pipeline); scenario / what-if on the completion run-rate; portfolio risk comparison (two governed scopes at one date); analytical composition layer (multi-capability plans); week-on-week movement attribution (hover / drill) | 22 |
| `RAW_TEXT_COUPLING` | generic_analysis — funded current aggregation; generic_analysis — pipeline current aggregation (incl. pipeline stage analysis); cohort progression (metric progression by vintage across reporting dates); geography analysis (UK ITL3 exposure); limit assessment / approved concentration tests and headroom; forecast / expected funding — run-rate extrapolation and milestones; scenario / what-if on the completion run-rate; portfolio risk comparison (two governed scopes at one date); source portfolio scope (Direct / Acquired / named source / cohort); seasoning / front-book vs back-book population | 10 |
| `GENUINE_CAPABILITY_GAP` | FORWARD-LOOKING approved-limit projection | 1 |

A capability commonly carries more than one class, so the counts sum above the capability total.

---

## RAW-TEXT DEPENDENCY AUDIT

The test is mechanical, and deliberately so: **after an analytical intent / spec / plan exists, does any downstream code read the natural-language question again to decide dataset, measure, aggregation, filter, dimension, period, comparison, scope or capability?** Text used purely for presentation does not count.

**RAW_TEXT_DEPENDENT_CAPABILITIES = 16 of 33**

- generic_analysis — funded current aggregation
- generic_analysis — pipeline current aggregation (incl. pipeline stage analysis)
- forecast-view analysis (derived funded + weighted pipeline frame)
- funded temporal analysis / evolution (time series)
- pipeline evolution (weekly history, funnel evolution)
- pipeline_summary (pipeline headline position)
- cohort progression (metric progression by vintage across reporting dates)
- geography analysis (UK ITL3 exposure)
- limit assessment / approved concentration tests and headroom
- forecast / expected funding — run-rate extrapolation and milestones
- scenario / what-if on the completion run-rate
- portfolio risk comparison (two governed scopes at one date)
- analytical composition layer (multi-capability plans)
- source portfolio scope (Direct / Acquired / named source / cohort)
- seasoning / front-book vs back-book population
- FORWARD-LOOKING approved-limit projection

**RAW_TEXT_INDEPENDENT_CAPABILITIES = 17 of 33**

- superlative / ranking (grouped top-N and loan-level)
- share / contribution analysis
- loan-level listing, scatter and bubble
- stratification / bucketed dimensions
- temporal comparison (two named reporting periods)
- portfolio_summary (funded headline position)
- period_movement (month-on-month with attribution)
- period change analysis (Business-Semantics-Registry driven)
- funded bridge / balance attribution waterfall
- cohort / vintage analysis (static pool)
- cohort conversion (cumulative KFI -> Funded)
- pipeline stage movement / transitions
- pipeline movement summary (all stages, one interval)
- concentration analysis (governed dimension distribution)
- borrowing base / facility utilisation / eligibility
- forecast bridge (funded + probability-weighted pipeline)
- week-on-week movement attribution (hover / drill)

#### Every raw-question reader, by exact function

| Function | Decides | Called from | After the plan exists? |
|---|---|---|---|
| `mi_agent_api.workspace.resolve_dataset` | dataset | mi_agent_api/mi_service.py:487 | **YES** |
| `mi_agent.portfolio_lens.resolve_lens_with_default` | scope / population | mi_agent/mi_agent_workflow.py:763; mi_agent_api/chat_routing.py:579, 4019, 4075, 4724 | **YES** |
| `question_interpretation.lexical.pipeline_stage_request` | filter | mi_agent/mi_agent_workflow.py:808 | **YES** |
| `mi_agent.mi_agent_workflow.missing_dimension_policy_for` | aggregation coverage | mi_agent/mi_agent_workflow.py:490, used at :942 | **YES** |
| `mi_agent.mi_agent_workflow `_grouping_marker` regex` | aggregation / output shape | mi_agent/mi_agent_workflow.py:880 | **YES** |
| `mi_agent.execution_receipt.detect_requested_facets` | capability / completeness | mi_agent/mi_agent_workflow.py (receipt assembly) | **YES** |
| `mi_agent_api.concentration_query.detect_intent` | capability | mi_agent_api/chat_routing.py:_route_concentration_tests -> conc_query.answer(question, envelope) | **YES** |
| `mi_agent_api.chat_routing._scenario_multiplier` | scenario magnitude | mi_agent_api/chat_routing.py:2263 | **YES** |
| `mi_agent.period_request.requested_unit` | period | mi_agent_api/chat_routing.py:_route_forecast | **YES** |
| `mi_agent.mi_geography.stated_basis` | dimension (geography basis) | mi_agent/mi_geography.py:348, consumed by the geography contract | **YES** |
| `mi_workflows.portfolio_risk_comparison.resolve_comparison_scopes` | scope (two populations) | mi_workflows/portfolio_risk_comparison.py:524, INSIDE run_portfolio_risk_comparison | **YES** |
| `mi_agent.seasoning.resolve_population_predicate` | scope | mi_workflows/analytical/populations.py | **YES** |
| `mi_agent_api.chat_routing._projects_forward` | capability | mi_agent_api/chat_routing.py (risk_limits recogniser) | no (recognition) |
| `mi_agent_api.stage_movement_query.read` | filter / capability | recognition, memoised via remember_recognition; fallback re-read at stage_movement_query.py:555 | no (recognition) |
| `mi_agent_api.pipeline_movement_summary.read` | period | recognition, memoised; fallback at line 364 | no (recognition) |
| `mi_agent_api.borrowing_base_query.read` | capability | recognition, memoised; fallback at line 649 | no (recognition) |
| `mi_agent.period_change.rank_request.detect_rank_request` | aggregation (ranking basis / top-n) | recognition-time for the period-change route | no (recognition) |

Notes:

- `resolve_dataset` — THE dataset owner. Called before routing and before any plan slot could be filled; AnalyticalScope.dataset is left None.
- `resolve_lens_with_default` — Applies filters to the spec AFTER the QueryPlan lift.
- `pipeline_stage_request` — Injects a stage predicate into spec.filters. Proven by execution.
- `missing_dimension_policy_for` — bucket vs exclude moves grouped denominators and coverage %.
- `mi_agent_workflow `_grouping_marker` regex` — Gates whether a failed spec may be auto-recovered to a KPI/table.
- `detect_requested_facets` — Re-derives material facets FROM THE SENTENCE and can REFUSE the answer (facet state LOST). Not presentation.
- `detect_intent` — Chooses list / get / summary / compare from the sentence.
- `_scenario_multiplier` — Quantifies the perturbation from the sentence.
- `requested_unit` — The projection's temporal unit.
- `stated_basis` — Obligor vs collateral geography — which geography the answer is measured on.
- `resolve_comparison_scopes` — The only workflow that still resolves its populations from raw text inside the calculation entry point.
- `resolve_population_predicate` — Front/back book boundary.
- `_projects_forward` — A recognition-time decline test, not a post-plan read.
- `read` — Reading at recognition IS recognition. Listed for completeness.
- `read` — As above.
- `read` — As above.
- `detect_rank_request` — As above.

---

## OUTPUT 5 — KEY FINDINGS

**1. What percentage of Trakt's existing MI capability already has a deterministic calculation owner that does NOT require raw question text?**

Two honest denominators, because they answer different questions.

- *By capability:* **17 of 33 (52%)** have a calculation owner that takes structured inputs only. The remaining 16 re-read the sentence for a semantic decision **after** the parse — and in every case the coupling is in the ORCHESTRATION layer (route, workflow, receipt), never in the arithmetic itself.

- *By arithmetic:* **every one of the 21 calculation owners in Output 3 takes structured inputs only.** Not one of them performs arithmetic conditioned on the sentence. The single owner that still reads the question inside its entry point — `portfolio_risk_comparison.run_portfolio_risk_comparison` — reads it to choose two POPULATIONS and then hands both to `mi_workflows.engine`, which is text-free. What is text-bound is *which rows*, *which dataset*, *which period* and *whether to answer at all* — never *what a number means*.

That distinction is the single most important line in this census. Trakt does not have a calculation problem. It has a **scope-and-dispatch problem**.

**2. What percentage is currently reachable from the GovernedQueryPlan?**

- *By question volume:* **607 of 882 = 68.8%** — the plan is the semantic contract AND the compiled spec is what executes. A further 53 questions (6.0%) are lifted at parse but then claimed by a specialist route that executes through its own owner; for those the plan is a **contract without a consumer**.

- *By capability:* **4 of 33 CONNECTED, 5 PARTIAL, 24 NOT_CONNECTED.** The capability count is far less flattering than the volume count, because the connected capabilities are the high-frequency ones.

**3. How many apparent 'specialist capabilities' actually reuse common generic execution infrastructure?**

**At least fourteen of the thirty-three.** Named explicitly:

| Looks specialist | Actually executed by |
|---|---|
| pipeline stage analysis / amount by stage | `mi_query_executor.execute_mi_query` — proven by execution |
| superlative / ranking / top-N | `mi_query_executor._apply_top_n` — proven by execution |
| share and contribution | `mi_query_executor._execute_share` / `_execute_contribution` |
| bucketed / stratified dimensions | the executor, over a bucket column materialised in preparation |
| loan-level listing / scatter / bubble | `mi_query_executor._execute_loan_level` |
| forecast-view analysis | the executor, over a derived frame |
| funded temporal series | `temporal_query.execute_temporal` → the executor, once per period |
| vintage / static-pool analysis | `analytics_lib.cohort` primitives + ordinary grouping |
| concentration analysis | `mi_workflows.engine.ranked_distribution` |
| portfolio risk comparison | `mi_workflows.engine.aggregate` / `compare_values` |
| the analytical layer's portfolio_snapshot / population_profile | `mi_workflows.engine` |
| pipeline movement summary | `movement_detail.build_stage_transition_detail` — the SAME payload the stage route consumes |
| weekly brief insights | `movement_detail` + concentration + funnel; composes only |
| risk-limit headroom ranking | `analytics_lib.concentration.group_shares` / `top_n_concentration` |

**Ranking is implemented seven times** in the estate — `mi_query_executor._apply_top_n`, `mi_query_executor._execute_ranked_loans`, `mi_workflows.engine.ranked_distribution`, `mi_agent.period_change.ranking.rank_movement`, `mi_agent_api.movement_detail.rank_contributors`, `analytics_lib.concentration.top_n_concentration` and `mi_agent.risk_monitor.concentration.top_n_concentration`. The plan has no ordering slot at all.

**4. Which migration gaps are simply wiring?**

Five, and they are unusually cheap because the contract, the resolver and in two cases the *function parameter* already exist:

| Gap | Why it is wiring |
|---|---|
| **Dataset / lens / period binding** | `plan_from_spec` already accepts `dataset=`, `portfolio_lens=` and `period=`. `parsed_question.py:154` calls `compiled_spec_for(spec)` with **none of them**. The owners (`workspace.resolve_dataset`, `portfolio_context.resolve_context`) already produce exactly these values. |
| **Ordering + limit slot** | The arithmetic exists and is proven. Adding an ordering/limit to `PlannedOutput` connects **93 generic-path corpus questions** with no new calculation. |
| **`unavailable_filters` on the lift** | It is a DISCLOSURE field, not a semantic — it records filters that could NOT be applied. It blocks 12 corpus lifts purely because the inverted liftability test treats every unmodelled field as semantic. Carrying it across like `metric_defaulted` already is costs nothing. |
| **`pipeline_summary` dispatch** | Its measurable half is already the generic executor's; the route simply claims the question first. |
| **`vintage_analysis` dispatch** | `balance by vintage_year` already flows through the plan; only the `/mi/cohorts` service shape does not. |

**5. Which genuinely require architectural work?**

Five semantic families the plan has no vocabulary for. None of them is a missing calculation:

1. **A temporal axis** — grain, window, period list. Blocks evolution, pipeline evolution, cohort progression and the series half of every capability. (`AnalyticalScope.period` is a single optional string.)

2. **A comparison relationship between two scopes** — the compiler already executes N scopes per plan; nothing can say *this one is the baseline for that one*. Blocks temporal compare, period movement, period change, portfolio risk comparison.

3. **Sibling (non-nested) populations** — `ScopeDelta` may only NARROW, by deliberate design. Two peer populations, and the denominator a share needs, are both the direction it forbids.

4. **Entity state across snapshots** — stage transitions and cohort conversion follow one case through two frames. `AnalyticalScope` describes rows in **one** frame.

5. **Forward constructions** — projection, run-rate, milestone solving, scenario perturbation, threshold/limit status. A plan describes rows that exist.

**6. Are we at risk of rebuilding or unnecessarily narrowing existing capability?**

**Yes, and the evidence is already in the repository.**

- The brief's own worked example is the pattern: `pipeline_contract` exposes stage *counts*, so amount-by-stage was concluded absent. It is not absent; it is a one-line executor call, reproduced in this census against a committed fixture.

- **Every one of the 222 corpus declines is a plan-vocabulary gap, not a capability gap.** A programme that reads `plan_from_spec → None` as *unsupported* would conclude Trakt cannot rank, cannot compute a share, cannot draw a time series, cannot compare two months and cannot test a limit. All five are shipped, tested and — for ranking — proven executing in this document.

- **`plan_from_spec` is deliberately, aggressively conservative.** Its own docstring says it *declines more than it accepts*, and the inverted test means a field added to `MIQuerySpec` tomorrow is un-liftable until someone models it. That is excellent engineering and a **terrible capability signal**. A decline says *the plan cannot carry this*; it says nothing whatever about the executor.

- Three routes exist **only** because a broader capability had previously substituted for a narrower one — stage stock for a stage transition, today's limit status for a forward projection, the funded summary for a pipeline summary. Narrowing during migration would reintroduce each of them.

- One measured overlap to watch: **29 of the 35 questions claimed by the analytical composition layer also lift to a `QueryPlan`.** Two plan layers with overlapping claims and no composition rule between them is where a capability gets rebuilt.

**7. Does the remaining migration appear substantially smaller or larger than route count would suggest?**

**Substantially smaller.**

- Nineteen registered recognisers, thirty-three user-facing capabilities — but **21 distinct calculation owners**, and **one of them (`execute_mi_query`) already serves 69% of corpus traffic**.

- 9 of the 33 are `TRIVIAL` or `SMALL`; they account for a disproportionate share of question volume, because ranking alone is 12.4 per cent of the corpus and is `SMALL`.

- The `LARGE` items cluster into the **five semantic families** above, not into 15 separate migrations. Building the temporal axis once addresses four capabilities; building the comparison relationship once addresses four more.

- **Exactly one genuine capability gap exists in the entire estate** (forward-looking approved-limit projection), and it is already documented, already declined rather than substituted, and explicitly out of scope of any plan migration.

Route count over-states the work. **Owner count under-states how much is already done**: the biggest owner is already connected, and the largest single gap in front of it is an `ORDER BY … LIMIT`.


---

## Per-capability records

### 1. generic_analysis — funded current aggregation

**USER_INTENT_EXAMPLES**
- _What is the total funded balance?_
- _What is the balance by broker?_
- _How many loans are to borrowers over 55 with LTV above 50%?_
- _What is the weighted average LTV by region?_

| Field | Value |
|---|---|
| DATASET | funded |
| DATASET_OWNER | `mi_agent_api.datasets._resolve_query_frame / mi_agent_api.snapshots` |
| PREPARATION_OWNER | `mi_agent_api.funded_prep.prepare_funded_mi_dataset` |
| **CALCULATION_OWNER** | **`mi_agent.mi_query_executor.execute_mi_query`** |
| MEASURES_SUPPORTED | COUNT, SUM, AVG, WEIGHTED_AVG, MEDIAN, MIN, MAX over any registry metric (42 metric-role fields); plus SHARE and CONTRIBUTION, which the executor owns but query_plan.OPERATIONS does not name. |
| FILTERS_SUPPORTED | **YES_GENERIC** — mi_query_executor._apply_filters / governed_predicate_mask: eq, in, gt, ge, lt, le, between, contains, date bounds, over any registry field. |
| DIMENSIONS_SUPPORTED | YES_GENERIC — every dimension-role field in mi_agent/mi_semantics_field_registry.yaml (59 at this SHA); the authoritative set is mi_query_executor._all_group_dims over spec.dimensions then spec.dimension. 1, 2 or 3+ axes. |
| TEMPORAL_SUPPORT | current |
| POPULATION_SCOPE_SUPPORT | whole book / Direct / Acquired / named source / cohort / row-predicate population |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **YES** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> mi_agent_api.mi_service -> mi_agent.mi_agent_workflow.run_mi_agent_query:944
- mi_agent.mi_runtime.run_mi_query (CLI / simulation only — NOT on the HTTP serving path at this SHA)
- mi_agent.query_plan_execution.execute_query_plan (contract tests only; no production caller)

**Raw-question reads (exact functions)**
- mi_agent_api.workspace.resolve_dataset(question) — THE dataset, called at mi_agent_api/mi_service.py:487, downstream of nothing and upstream of everything. AnalyticalScope.dataset is left None on the live lift.
- mi_agent.portfolio_lens.resolve_lens_with_default(question, ...) — the source-portfolio POPULATION, applied to the spec at mi_agent/mi_agent_workflow.py:763 AFTER the plan lift.
- question_interpretation.lexical.pipeline_stage_request(question) — a stage FILTER injected into spec.filters at mi_agent/mi_agent_workflow.py:808.
- mi_agent.mi_agent_workflow.missing_dimension_policy_for(question) — bucket vs exclude for missing grouping values, which moves grouped denominators and coverage (mi_agent/mi_agent_workflow.py:490).
- mi_agent.mi_agent_workflow `_grouping_marker` regex at line 880 — decides whether a failed spec may be auto-recovered to a KPI/table.
- mi_agent.execution_receipt.detect_requested_facets(question, ...) — re-derives material facets FROM THE SENTENCE and can turn an answer into a refusal (facet state LOST). 5,015 lines; not presentation.

**EXECUTION_EVIDENCE**
- receipt (mi_agent.execution_receipt)
- reconciliation (mi_query_executor._build_reconciliation: input / included / excluded records + balance + coverage %)
- dataset identity + run_id (metadata.dataset, metadata.run_id)
- applied predicates (metadata.applied_predicates — field, operator, values, distinct from reconciliation.filters which echoes the spec)
- applied filter fields (metadata.applied_filter_fields)
- grouped dimensions (metadata.group_field_keys) and rejected ones
- dimension + filter invariants (mi_agent.mi_query_contract)
- query trace (mi_query_contract.build_query_trace)

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **YES** |
| NEW_PLAN_CONNECTIVITY | **CONNECTED** |
| CONNECTIVITY_GAP | NONE for the measure/filter/dimension core. The plan's dataset, portfolio_lens and period slots are NOT populated on the live path — compiled_spec_for(spec) is called with no scope kwargs at mi_agent/parsed_question.py:154 — so the frame-selecting facts remain outside the plan. |
| Gap class | `DATASET_BINDING`, `RAW_TEXT_COUPLING`, `RECEIPT_ADAPTATION` |
| MIGRATION_COMPLEXITY | **TRIVIAL** |

> 607 of 882 corpus questions (68.8%) reach this owner through the plan. This one function is the single largest MI surface in the estate.

---

### 2. generic_analysis — pipeline current aggregation (incl. pipeline stage analysis)

**USER_INTENT_EXAMPLES**
- _How much is in the pipeline by stage?_
- _How many cases are at Offer?_
- _What is the pipeline amount by broker?_
- _Pipeline exposure by region._

| Field | Value |
|---|---|
| DATASET | pipeline |
| DATASET_OWNER | `mi_agent_api.pipeline_contract.discover_pipeline_sources / resolve_pipeline_source / load_prepared_pipeline` |
| PREPARATION_OWNER | `mi_agent_api.pipeline_prep.prepare_pipeline_mi_dataset` |
| **CALCULATION_OWNER** | **`mi_agent.mi_query_executor.execute_mi_query (IDENTICAL to record 1 — one owner, two prepared frames)`** |
| MEASURES_SUPPORTED | Same governed set as record 1, over the canonical economic fields pipeline_prep maps onto the funded names (so funded and pipeline correlate). Stage amounts included. |
| FILTERS_SUPPORTED | **YES_GENERIC** — the same _apply_filters, plus pipeline_stage. |
| DIMENSIONS_SUPPORTED | YES_GENERIC — the same registry, plus pipeline_stage and the pipeline-only fields pipeline_prep derives. |
| TEMPORAL_SUPPORT | current |
| POPULATION_SCOPE_SUPPORT | pipeline (whole). The weekly extract carries no source-portfolio provenance, so a Direct/Acquired lens cannot narrow it; chat_routing._pipeline_scope_disclosure discloses this rather than silently widening. |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **YES** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query (dataset resolved to PIPELINE by workspace.resolve_dataset)
- GET /mi/pipeline/snapshot -> pipeline_contract.compute_pipeline_snapshot (a DIFFERENT owner: stage COUNTS and cap breakdown, not arbitrary measures)

**Raw-question reads (exact functions)**
- Identical to record 1. Additionally the stage narrowing is ALWAYS raw-text sourced: spec.filters carries no stage, and lexical.pipeline_stage_request(question) injects it (proved by execution — see EXECUTION_PROOFS[2]).

**EXECUTION_EVIDENCE**
- as record 1, plus pipeline source lineage from pipeline_contract (source file, extract date, weekly inventory)

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **YES** |
| NEW_PLAN_CONNECTIVITY | **CONNECTED** |
| CONNECTIVITY_GAP | The plan cannot say WHICH DATASET, and cannot carry the stage narrowing, so both survive only as post-plan text reads. Proven: 'How much is in the pipeline by stage?' stamps semantic_contract=query_plan, and the stage filter for 'the amount for offer stage cases' does not. |
| Gap class | `DATASET_BINDING`, `RAW_TEXT_COUPLING`, `RECEIPT_ADAPTATION` |
| MIGRATION_COMPLEXITY | **SMALL** |

> THE BRIEF'S WORKED EXAMPLE, reproduced at this SHA. pipeline_contract exposes stage counts; stage AMOUNTS come from the generic executor over the prepared pipeline frame. One correction: the live chain is mi_service -> mi_agent_workflow -> execute_mi_query. mi_runtime.run_mi_query is NOT on the HTTP serving path at this SHA (its only callers are scripts/ and simulation/).

---

### 3. forecast-view analysis (derived funded + weighted pipeline frame)

**USER_INTENT_EXAMPLES**
- _Forecast funded balance by region._
- _What is the forecast book by broker?_

| Field | Value |
|---|---|
| DATASET | derived |
| DATASET_OWNER | `mi_agent_api.workspace.build_forecast_view_frame` |
| PREPARATION_OWNER | `funded_prep + pipeline_prep, then the derived projection` |
| **CALCULATION_OWNER** | **`mi_agent.mi_query_executor.execute_mi_query (same owner)`** |
| MEASURES_SUPPORTED | as record 1; current_outstanding_balance carries the forecast contribution so any 'X by dimension' yields forecast X by dimension. |
| FILTERS_SUPPORTED | **YES_GENERIC** — over the projected columns only. |
| DIMENSIONS_SUPPORTED | YES_LIMITED — the derived frame keeps a subset of the book's columns; mi_agent_api/datasets.py BOOK_COLUMNS_ATTR records the source schema so an availability check does not report a field the book carries as absent. |
| TEMPORAL_SUPPORT | current |
| POPULATION_SCOPE_SUPPORT | whole book + whole pipeline |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **YES** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query where workspace.resolve_dataset returns FORECAST

**Raw-question reads (exact functions)**
- workspace.resolve_dataset(question) — the word 'forecast' anywhere selects this view (datasets.py:721 documents the substring test and its consequence).

**EXECUTION_EVIDENCE**
- as record 1, plus the derived-frame book-column stamp

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **YES** |
| NEW_PLAN_CONNECTIVITY | **CONNECTED** |
| CONNECTIVITY_GAP | Same DATASET_BINDING gap as records 1-2, and sharper here: the dataset choice changes the FRAME's columns, so a plan with dataset=None cannot state what its own scope was. |
| Gap class | `DATASET_BINDING`, `RECEIPT_ADAPTATION` |
| MIGRATION_COMPLEXITY | **SMALL** |

> 12 corpus lifts are blocked by `unavailable_filters` alone — a DISCLOSURE field recording predicates that could NOT be applied. It declines the lift only because the inverted liftability test treats every unmodelled field as semantic.

---

### 4. superlative / ranking (grouped top-N and loan-level)

**USER_INTENT_EXAMPLES**
- _Which brokers have the largest exposure?_
- _Top 5 regions by balance._
- _What is the largest single loan?_
- _Show the 10 biggest loans._

| Field | Value |
|---|---|
| DATASET | funded\|pipeline |
| DATASET_OWNER | `as records 1-2` |
| PREPARATION_OWNER | `as records 1-2` |
| **CALCULATION_OWNER** | **`mi_agent.mi_query_executor._apply_top_n (grouped) and mi_agent.mi_query_executor._execute_ranked_loans (loan-level) — both inside execute_mi_query`** |
| MEASURES_SUPPORTED | the ranked measure is any governed metric; ordering basis is spec.sort_by with rank priority (balance, count, concentration). |
| FILTERS_SUPPORTED | **YES_GENERIC** — (ranking runs after _apply_filters) |
| DIMENSIONS_SUPPORTED | YES_GENERIC |
| TEMPORAL_SUPPORT | current |
| POPULATION_SCOPE_SUPPORT | as records 1-2 |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **NO** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query (generic path)
- mi_workflows.engine.ranked_distribution (analytical layer, concentration workflow) — a SECOND ranking owner
- mi_agent.period_change.ranking.rank_movement — a THIRD, over movements rather than levels
- mi_agent_api.movement_detail.rank_contributors — a FOURTH
- analytics_lib.concentration.top_n_concentration and mi_agent.risk_monitor.concentration.top_n_concentration — two more

**EXECUTION_EVIDENCE**
- as record 1; execution_receipt.detect_unranked_superlative additionally fails closed when a superlative was asked for and no ranking ran

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | AnalyticalScope/PlannedOutput have NO ordering and NO limit slot. There is no place to say 'order by this measure, descending, keep 5' and no place to say 'the single largest row'. Five spec fields carry it today — ranking_mode, sort_by, sort_direction, top_n, limit — and all five are outside MODELLED_FIELDS. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | One missing plan semantic: an ORDERING (measure + direction) and a LIMIT, at plan or output level. The arithmetic already exists in the executor the compiler targets. |
| Gap class | `MISSING_PLAN_SEMANTIC`, `RECEIPT_ADAPTATION` |
| MIGRATION_COMPLEXITY | **SMALL** |

> THE SINGLE LARGEST CONNECTIVITY GAP IN THE ESTATE: 109 of 882 corpus questions (12.4%) — 93 of them on the generic path, i.e. they would be connected the moment the slot exists. Proven to execute correctly today (EXECUTION_PROOFS[1]). This is NOT a capability gap.

---

### 5. share / contribution analysis

**USER_INTENT_EXAMPLES**
- _What percentage of the book has LTV above 50%?_
- _What share of the book is joint borrowers?_
- _How much does each region contribute to the weighted average LTV?_

| Field | Value |
|---|---|
| DATASET | funded\|pipeline |
| DATASET_OWNER | `as records 1-2` |
| PREPARATION_OWNER | `as records 1-2` |
| **CALCULATION_OWNER** | **`mi_agent.mi_query_executor._execute_share and mi_agent.mi_query_executor._execute_contribution`** |
| MEASURES_SUPPORTED | share of loan_count or of any monetary metric; contribution is weight-field based |
| FILTERS_SUPPORTED | **YES_GENERIC** — and a share is the one path handed BOTH the filtered and the unfiltered frame, because it needs both populations. |
| DIMENSIONS_SUPPORTED | YES_GENERIC for contribution; share is whole-book denominated. |
| TEMPORAL_SUPPORT | current |
| POPULATION_SCOPE_SUPPORT | as records 1-2 |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **NO** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query (generic path)
- mi_workflows.engine.aggregate(share_basis=...) — a second owner used by the BSR workflows
- mi_agent.period_change.calculations._aggregate_share — a third

**EXECUTION_EVIDENCE**
- metadata.share_basis, metadata.contribution_weight_field, plus the record-1 reconciliation block

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | query_plan.OPERATIONS is (count, sum, avg, weighted_avg, median, min, max). `share` and `contribution` are not in it, and PlannedOutput.__post_init__ RAISES on an operation outside the set, so plan_from_spec declines. A share also needs a DENOMINATOR population, which AnalyticalScope cannot express: a ScopeDelta may only narrow, so 'this population as a fraction of the book' has no second scope to point at. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | Two slots: the operation vocabulary, and a denominator scope reference (the one direction ScopeDelta deliberately forbids). |
| Gap class | `MISSING_PLAN_SEMANTIC` |
| MIGRATION_COMPLEXITY | **MEDIUM** |

> 5 corpus questions declined on this alone. Medium rather than small because the denominator is a genuine contract question, not a field: adding a widening delta would break the asymmetry ScopeDelta exists to guarantee.

---

### 6. loan-level listing, scatter and bubble

**USER_INTENT_EXAMPLES**
- _List the loans above £500k._
- _Plot LTV against interest rate._
- _Show a bubble chart of balance by age and LTV._

| Field | Value |
|---|---|
| DATASET | funded\|pipeline |
| DATASET_OWNER | `as records 1-2` |
| PREPARATION_OWNER | `as records 1-2` |
| **CALCULATION_OWNER** | **`mi_agent.mi_query_executor._execute_loan_level`** |
| MEASURES_SUPPORTED | row-level values of any registry field; deterministic sampling (sample_seed=42) with a declared max_loan_level_rows. |
| FILTERS_SUPPORTED | **YES_GENERIC** — (no further detail) |
| DIMENSIONS_SUPPORTED | axes rather than groupings (x / y / size / color) |
| TEMPORAL_SUPPORT | current |
| POPULATION_SCOPE_SUPPORT | as records 1-2 |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **NO** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query (generic path)

**EXECUTION_EVIDENCE**
- as record 1, plus the sampling disclosure in metadata

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | AnalyticalScope models GROUPING dimensions only. x/y are liftable ONLY as restatements of dimensions[0]/[1] (query_plan_adapter._is_liftable), and `size`/`color` are unmodelled outright. A row-level answer is also not a PlannedOutput: there is no operation. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | No plan concept for 'the rows themselves, unaggregated'. |
| Gap class | `MISSING_PLAN_SEMANTIC` |
| MIGRATION_COMPLEXITY | **MEDIUM** |

> Shares the ranking gap where the listing is ordered.

---

### 7. stratification / bucketed dimensions

**USER_INTENT_EXAMPLES**
- _Balance by LTV bucket._
- _Loan count by ticket-size band._
- _Balance by age bucket._

| Field | Value |
|---|---|
| DATASET | funded\|pipeline |
| DATASET_OWNER | `as records 1-2` |
| PREPARATION_OWNER | `analytics_lib.buckets.materialise_buckets (config-driven bands) + mi_agent.quantile_buckets.materialise_quantile_bucket (asset-agnostic quartiles) — run during preparation` |
| **CALCULATION_OWNER** | **`mi_agent.mi_query_executor.execute_mi_query over the materialised bucket column (the bucket is a DIMENSION by the time the executor sees it)`** |
| MEASURES_SUPPORTED | as record 1 |
| FILTERS_SUPPORTED | **YES_GENERIC** — (no further detail) |
| DIMENSIONS_SUPPORTED | YES_GENERIC over registry-named bucket dimensions |
| TEMPORAL_SUPPORT | current |
| POPULATION_SCOPE_SUPPORT | as records 1-2 |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **NO** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query
- dashboard stratification surfaces
- analytics_lib.stratify.stratify (a second, dashboard-side owner)

**EXECUTION_EVIDENCE**
- bucket definitions are registry/config-named and appear in resolved_fields; the reconciliation block covers the bucket column

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **YES** |
| NEW_PLAN_CONNECTIVITY | **CONNECTED** |
| CONNECTIVITY_GAP | NONE — a bucket is an ordinary dimension at execution. spec.bucket_strategy / bucket_count / bucket_field are unmodelled, so a question that RESTATES the strategy in words declines; the ordinary 'by LTV bucket' form does not. |
| Gap class | — |
| MIGRATION_COMPLEXITY | **TRIVIAL** |

> A capability that looks specialist and is not: the arithmetic is record 1's.

---

### 8. funded temporal analysis / evolution (time series)

**USER_INTENT_EXAMPLES**
- _Show funded balance evolution by month._
- _How has the weighted average LTV moved over the last six months?_
- _Show monthly balance evolution by broker._

| Field | Value |
|---|---|
| DATASET | funded |
| DATASET_OWNER | `mi_agent_api.evolution.funded_frames over mi_agent_api.snapshots (the governed monthly funded runs)` |
| PREPARATION_OWNER | `mi_agent_api.funded_prep.prepare_funded_mi_dataset per period` |
| **CALCULATION_OWNER** | **`mi_agent_api.temporal_query.execute_temporal — which calls mi_agent.mi_query_executor.execute_mi_query ONCE PER PERIOD. The lowest owner of every FIGURE is the executor; execute_temporal owns only the period stacking. mi_agent_api.evolution.assemble_funded_evolution remains the owner for the DASHBOARD series (five fixed metrics).`** |
| MEASURES_SUPPORTED | MI-Query path: the full governed set, because each period is an ordinary executor call. Dashboard path: balance, loan count, WA LTV, WA interest rate, average youngest-borrower age. |
| FILTERS_SUPPORTED | **YES_GENERIC** — on the MI-Query path — and temporal_query.predicates_applied_in_every_period proves the SAME predicates ran in every period rather than assuming it. YES_LIMITED on the dashboard path. |
| DIMENSIONS_SUPPORTED | YES_GENERIC on the MI-Query path (time x dimension); temporal_query.series_by_category assembles the per-category series. |
| TEMPORAL_SUPPORT | series, weekly, monthly |
| POPULATION_SCOPE_SUPPORT | whole book / Direct / Acquired / named source |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **YES** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> chat_routing._route_evolution
- GET /mi/evolution/funded
- GET /mi/evolution/funnel, /mi/evolution/forecast

**Raw-question reads (exact functions)**
- mi_agent_api/chat_routing.py _route_evolution reads the question for the grain and for the grouped-answer shape (_grouped_evolution_answer); _resolve_lens(question, source_lens) resolves the population.

**EXECUTION_EVIDENCE**
- per-period reconciliation and lineage (run id, reporting date, source file) — the point-in-time MI standard, per period
- predicates_applied_in_every_period
- receipt

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | AnalyticalScope.period is a SINGLE optional string. There is no slot for a period RANGE, a GRAIN (weekly/monthly) or a TIME AXIS. query_plan_adapter._is_liftable explicitly refuses a spec whose `x` is a time axis rather than dimensions[0], and documents why: a plan that claimed such a question has no dimensions would misdescribe it. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | A temporal axis on the plan: {grain, window or explicit period list} plus the rule that a series is N executions of one scope. The compiler already does N-executions-per-plan for N populations; a series is the same shape over periods. |
| Gap class | `TEMPORAL_BINDING` |
| MIGRATION_COMPLEXITY | **MEDIUM** |

> 35 corpus declines are exactly this. temporal_query.py was written to make the executor the single owner of a period figure — that work means the plan only has to carry the AXIS, not the arithmetic.

---

### 9. pipeline evolution (weekly history, funnel evolution)

**USER_INTENT_EXAMPLES**
- _How has the pipeline moved over the last five weeks?_
- _Show the pipeline funnel over time._
- _What is the five-week average pipeline amount?_

| Field | Value |
|---|---|
| DATASET | pipeline |
| DATASET_OWNER | `mi_agent_api.pipeline_contract.collect_weekly_history / build_pipeline_history / weekly_extract_inventory` |
| PREPARATION_OWNER | `mi_agent_api.pipeline_prep.prepare_pipeline_mi_dataset per week` |
| **CALCULATION_OWNER** | **`mi_agent_api.evolution.pipeline_evolution and mi_agent_api.evolution.pipeline_funnel_evolution (+ five_week_average, weekly_flow)`** |
| MEASURES_SUPPORTED | pipeline_amount (= sum of current_outstanding_balance on the prepared frame), case counts, per-stage levels, weekly flow |
| FILTERS_SUPPORTED | **YES_LIMITED** — stage and the prepared-frame fields; not the executor's generic predicate set. |
| DIMENSIONS_SUPPORTED | YES_LIMITED — stage and the funnel's own levels. |
| TEMPORAL_SUPPORT | series, weekly, movement |
| POPULATION_SCOPE_SUPPORT | pipeline (whole) — no source-portfolio provenance available |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **YES** |

**ORCHESTRATION_ENTRY_POINTS**
- GET /mi/evolution/pipeline
- GET /mi/evolution/funnel
- POST /mi/query -> _route_evolution where the dataset is PIPELINE

**Raw-question reads (exact functions)**
- _route_evolution as record 8

**EXECUTION_EVIDENCE**
- weekly extract inventory and per-week lineage
- reconciliation per period

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | As record 8, plus: the funnel's stage levels are a fixed vocabulary rather than a governed dimension the plan can name. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | Temporal axis (as record 8) and a pipeline-series binding. |
| Gap class | `TEMPORAL_BINDING`, `DATASET_BINDING` |
| MIGRATION_COMPLEXITY | **MEDIUM** |

> A DIFFERENT calculation owner from record 8 despite the shared route name — kept separate deliberately.

---

### 10. temporal comparison (two named reporting periods)

**USER_INTENT_EXAMPLES**
- _Compare October and November funded balance._
- _How did pipeline amount change from last week?_
- _What was the balance last month versus this month?_

| Field | Value |
|---|---|
| DATASET | both |
| DATASET_OWNER | `mi_agent_api.workspace.resolve_dataset, then evolution.funded_frames or pipeline_contract.collect_weekly_history` |
| PREPARATION_OWNER | `funded_prep / pipeline_prep per period` |
| **CALCULATION_OWNER** | **`mi_agent_api.temporal_compare.compare_periods (delta, pct delta, direction) over the series built by temporal_compare.run_temporal_compare; the per-period FIGURE comes from the evolution assembly, not from a second engine.`** |
| MEASURES_SUPPORTED | resolve_metric_key maps (dataset, metric, aggregation) onto the governed period metric keys — a bounded set, not the full registry. |
| FILTERS_SUPPORTED | **NO** — on this owner. A narrowed comparison is not expressible here. |
| DIMENSIONS_SUPPORTED | NO — whole-population comparison only. |
| TEMPORAL_SUPPORT | compare |
| POPULATION_SCOPE_SUPPORT | whole book / dataset-wide |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **NO** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> chat_routing._route_compare
- GET /mi/evolution/compare
- mi_agent_api.analytical_plan.temporal_compare (Conversion 5 — the interpretation-driven plan, already converted)

**EXECUTION_EVIDENCE**
- per-period reconciliation
- source periods (governed source files)
- a controlled insufficient-data response when a period or metric is missing
- receipt

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | No second period, and no comparison relationship. A QueryPlan has ONE shared scope with ONE period; two periods are two scopes with no contract saying they are to be compared rather than merely both reported. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | A comparison relationship between two scopes, plus a period-pair slot. Note the compiler ALREADY executes several scopes per plan — what is missing is the statement that one is the baseline for the other. |
| Gap class | `TEMPORAL_BINDING`, `MISSING_PLAN_SEMANTIC` |
| MIGRATION_COMPLEXITY | **MEDIUM** |

> 7 corpus declines. Already converted onto the OTHER plan layer (analytical_plan), with committed equivalence evidence at migration_phase0/plan_equivalence_temporal_compare.py.

---

### 11. portfolio_summary (funded headline position)

**USER_INTENT_EXAMPLES**
- _Summarise the portfolio._
- _Give me a portfolio overview._
- _What is the current position of the book?_

| Field | Value |
|---|---|
| DATASET | funded |
| DATASET_OWNER | `mi_agent_api.evolution.funded_frames over output_root` |
| PREPARATION_OWNER | `mi_agent_api.funded_prep.prepare_funded_mi_dataset` |
| **CALCULATION_OWNER** | **`mi_agent_api.movement_summary.portfolio_summary, invoked through mi_agent_api.analytical_plan.portfolio_summary (Conversion 1). The primitives are evolution.funded_frames, evolution._scope_frame_lens, evolution.assemble_funded_evolution, movement_summary._regional_exposure and movement_summary._cohorts.`** |
| MEASURES_SUPPORTED | loan count, funded balance, WA current LTV, WA interest rate, average youngest-borrower age — a FIXED headline set, not the registry. |
| FILTERS_SUPPORTED | **NO** — a summary is not narrowed by predicates; it is narrowed by POPULATION (source scope) only. |
| DIMENSIONS_SUPPORTED | YES_LIMITED — largest regional exposures and cohort balances, fixed by the summary contract. |
| TEMPORAL_SUPPORT | current |
| POPULATION_SCOPE_SUPPORT | whole book / Direct / Acquired / named source, from QuestionInterpretation.source_scope |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **NO** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> chat_routing._route_portfolio_summary
- mi_agent_api.analytical_plan.portfolio_summary

**Raw-question reads (exact functions)**
- None. Conversion 1 is the reference conversion: _summary_population takes the population from the interpretation contract and RETURNS NOTHING when there is no contract, rather than falling back to a second reader (chat_routing.py:761-799). analytical_plan.build_plan is structurally forbidden from reading the question (assert_no_question_read).

**EXECUTION_EVIDENCE**
- receipt
- reconciliation
- governed run artefact lineage
- lens label + resolved portfolio-id list
- lensApplied disclosure (chat_routing._disclose_* )

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **PARTIAL** |
| Missing semantic slot | A QueryPlan CAN carry the five headline measures as five PlannedOutputs over one scope. It CANNOT carry the regional-exposure ranking (no ordering slot — record 4) nor the cohort block. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | The route executes through analytical_plan (the OTHER plan layer). Nothing routes a QueryPlan here. A liftable summary spec is still lifted at parse — measured 4 of 4 corpus summary questions — and then the route ignores it. |
| Gap class | `DISPATCH_ONLY`, `PLAN_TO_SPEC_ADAPTER`, `MISSING_PLAN_SEMANTIC` |
| MIGRATION_COMPLEXITY | **MEDIUM** |

> THE CLEAREST 'EXISTS=YES, REPRESENTABLE=PARTIAL, CONNECTED=NO' CASE. Also the clearest evidence that raw-text independence is achievable: this route has none.

---

### 12. pipeline_summary (pipeline headline position)

**USER_INTENT_EXAMPLES**
- _What is the current pipeline position?_
- _Summarise the pipeline._

| Field | Value |
|---|---|
| DATASET | pipeline |
| DATASET_OWNER | `mi_agent_api.pipeline_contract.resolve_pipeline_source` |
| PREPARATION_OWNER | `mi_agent_api.pipeline_prep.prepare_pipeline_mi_dataset` |
| **CALCULATION_OWNER** | **`mi_agent_api.pipeline_contract.compute_pipeline_snapshot (+ cap_breakdown)`** |
| MEASURES_SUPPORTED | case count, gross amount, per-stage counts, completion-probability summary, cap breakdown |
| FILTERS_SUPPORTED | **NO** — (no further detail) |
| DIMENSIONS_SUPPORTED | YES_LIMITED — stage |
| TEMPORAL_SUPPORT | current |
| POPULATION_SCOPE_SUPPORT | pipeline (whole); declared-provenance disclosure via pipeline_prep.declared_source_provenance |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **YES** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> chat_routing._route_pipeline_summary
- GET /mi/pipeline/snapshot

**Raw-question reads (exact functions)**
- chat_routing._is_pipeline_summary(question, spec) for the claim; chat_routing._pipeline_scope_disclosure -> _resolve_lens(question, source_lens) for the scope sentence.

**EXECUTION_EVIDENCE**
- pipeline source lineage
- declared source provenance disclosure
- receipt

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **PARTIAL** |
| Missing semantic slot | Stage counts and stage amounts ARE representable (record 2 proves it). The completion-probability summary and cap breakdown are not: they are derived facts of the preparation layer, not aggregations. |
| NEW_PLAN_CONNECTIVITY | **PARTIAL** |
| CONNECTIVITY_GAP | The measurable half already reaches the plan through the generic path; the headline block does not, because this route claims the question before the generic path runs. |
| Gap class | `DISPATCH_ONLY`, `PLAN_TO_SPEC_ADAPTER` |
| MIGRATION_COMPLEXITY | **SMALL** |

> Did not fire in the offline route probe because recognition requires the dataset owner to say PIPELINE first; covered by name in probes/targeted_route_probe.py.

---

### 13. period_movement (month-on-month with attribution)

**USER_INTENT_EXAMPLES**
- _What has changed versus last month?_
- _What moved since the prior reporting period?_
- _How much of the movement is new completions?_

| Field | Value |
|---|---|
| DATASET | funded |
| DATASET_OWNER | `mi_agent_api.evolution.funded_frames` |
| PREPARATION_OWNER | `mi_agent_api.funded_prep.prepare_funded_mi_dataset` |
| **CALCULATION_OWNER** | **`mi_agent_api.movement_summary.period_movement (+ _delta), invoked through mi_agent_api.analytical_plan.period_movement (Conversion 2)`** |
| MEASURES_SUPPORTED | the record-11 headline set, as movements; plus regional attribution of the balance movement and each source portfolio's contribution |
| FILTERS_SUPPORTED | **NO** — (no further detail) |
| DIMENSIONS_SUPPORTED | YES_LIMITED — region and source portfolio, fixed by contract |
| TEMPORAL_SUPPORT | compare, movement |
| POPULATION_SCOPE_SUPPORT | as record 11 |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **NO** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> chat_routing._route_period_movement
- mi_agent_api.analytical_plan.period_movement

**Raw-question reads (exact functions)**
- None at the calculation owner. The route reads the sentence only to claim (_is_period_movement).

**EXECUTION_EVIDENCE**
- receipt
- per-period reconciliation
- prior-period lineage
- attribution decomposition

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **PARTIAL** |
| Missing semantic slot | No comparison relationship (as record 10) and no attribution concept. The measures themselves are representable. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | As record 11 — it executes through the other plan layer. |
| Gap class | `DISPATCH_ONLY`, `PLAN_TO_SPEC_ADAPTER`, `TEMPORAL_BINDING`, `MISSING_PLAN_SEMANTIC` |
| MIGRATION_COMPLEXITY | **MEDIUM** |

> Conversion 2 generalised Conversion 1's plan artefact; the primitive vocabulary and the population step were reused unchanged.

---

### 14. period change analysis (Business-Semantics-Registry driven)

**USER_INTENT_EXAMPLES**
- _What changed in the portfolio this month?_
- _Which metrics moved most between the last two snapshots?_
- _How did the composition shift?_

| Field | Value |
|---|---|
| DATASET | funded (two snapshots) |
| DATASET_OWNER | `mi_agent_api.snapshots -> mi_agent.period_change.models.SnapshotFrame; periods resolved by period_change.periods.resolve_periods` |
| PREPARATION_OWNER | `mi_agent_api.funded_prep.prepare_funded_mi_dataset per snapshot` |
| **CALCULATION_OWNER** | **`mi_agent.period_change.workflow.run_period_change_analysis, delegating to period_change.calculations.aggregate (numeric / weighted / share), period_change.distribution.distribution_change, period_change.bridge.balance_bridge and period_change.ranking.rank_movement`** |
| MEASURES_SUPPORTED | every field the Business Semantics Registry governs (config/business_semantics_registry.yaml), selected by period_change.selection.select_fields against a declared policy (config/period_change_selection.yaml) — not a hard-coded list. |
| FILTERS_SUPPORTED | **YES_LIMITED** — scope (portfolio ids, asset classes) and the selection policy; not arbitrary row predicates. |
| DIMENSIONS_SUPPORTED | YES_GENERIC over registry dimensions, as composition shifts. |
| TEMPORAL_SUPPORT | compare, movement |
| POPULATION_SCOPE_SUPPORT | whole book / governed portfolio scope, authorisation-bounded (check_scope_access fails closed rather than intersecting) |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **NO** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> mi_agent_api.period_change_route.route_period_change
- direct API / job construction of PeriodChangeRequest

**Raw-question reads (exact functions)**
- The calculation is text-free: PeriodChangeRequest carries `question` but workflow.py reads it ONLY to echo it in to_dict(). Verified by inspection — the only three occurrences of `question` in workflow.py are the field, the docstring and the echo. The recognition read is memoised pre-claim (RouteRequest.remember_recognition) so the handler does not re-read.

**EXECUTION_EVIDENCE**
- receipt
- reconciliation
- audit block naming the registry version and selection policy
- explicit limitations and excluded candidates
- ranked movement receipt (mi_agent_api.movement_receipt)
- mixed-currency guard

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | A QueryPlan carries outputs over ONE scope. This is a two-snapshot MOVEMENT with per-unit ranking, distribution shift and a balance bridge — four relationship concepts the plan has no slot for. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | Comparison relationship + movement + distribution-shift concepts. |
| Gap class | `MISSING_PLAN_SEMANTIC`, `TEMPORAL_BINDING` |
| MIGRATION_COMPLEXITY | **LARGE** |

> Architecturally the most advanced route in the estate and the least raw-text coupled. Migrating it is about the PLAN growing up to it, not about the route changing.

---

### 15. funded bridge / balance attribution waterfall

**USER_INTENT_EXAMPLES**
- _What drove the change in funded balance?_
- _Break down the balance movement by broker._
- _Show the funded bridge for the last quarter._

| Field | Value |
|---|---|
| DATASET | funded (period range) |
| DATASET_OWNER | `mi_agent_api.evolution.funded_frames` |
| PREPARATION_OWNER | `mi_agent_api.funded_prep.prepare_funded_mi_dataset` |
| **CALCULATION_OWNER** | **`mi_agent_api.evolution.funded_bridge`** |
| MEASURES_SUPPORTED | balance movement decomposed into attribution components |
| FILTERS_SUPPORTED | **YES_LIMITED** — population scope |
| DIMENSIONS_SUPPORTED | YES_GENERIC for the bridge dimension (spec.bridge_dimension, resolved through the governed registry by Conversion 4) |
| TEMPORAL_SUPPORT | movement, compare |
| POPULATION_SCOPE_SUPPORT | whole book / Direct / Acquired / named source |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **NO** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> chat_routing._route_bridge
- mi_agent_api.analytical_plan.funded_bridge (Conversion 4)

**Raw-question reads (exact functions)**
- Conversion 4 moved the population and the grouping concept onto the interpretation contract; equivalence is evidenced at migration_phase0/plan_equivalence_funded_bridge.py, which compares the shipped and compositional paths on ROWS.

**EXECUTION_EVIDENCE**
- receipt
- reconciliation
- start period and window disclosure
- bridge component attribution

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | A bridge is a DECOMPOSITION of a movement into components. There is no plan concept for a component set, and no period range. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | Movement decomposition semantics + period range. |
| Gap class | `MISSING_PLAN_SEMANTIC`, `TEMPORAL_BINDING` |
| MIGRATION_COMPLEXITY | **LARGE** |

> 3 corpus declines on spec.bridge_query alone.

---

### 16. cohort / vintage analysis (static pool)

**USER_INTENT_EXAMPLES**
- _Show the book by origination vintage._
- _What is the balance by vintage year?_
- _What is the weighted average LTV of the 2022 vintage?_

| Field | Value |
|---|---|
| DATASET | funded |
| DATASET_OWNER | `mi_agent_api.datasets / snapshots` |
| PREPARATION_OWNER | `mi_agent_api.funded_prep (derives vintage_year and months_on_book)` |
| **CALCULATION_OWNER** | **`mi_agent_api.cohorts.cohort_analysis, over analytics_lib.cohort.cohort_table / add_cohort_period / months_on_book`** |
| MEASURES_SUPPORTED | balance, loan count, book share, balance-weighted LTV / interest rate / months-on-book per vintage. metricsAvailable declares exactly what was computed; nothing is fabricated. |
| FILTERS_SUPPORTED | **YES_LIMITED** — population |
| DIMENSIONS_SUPPORTED | vintage (year or configured grain) |
| TEMPORAL_SUPPORT | current (point-in-time static pool) |
| POPULATION_SCOPE_SUPPORT | whole book / Direct / Acquired / named source |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **NO** |

**ORCHESTRATION_ENTRY_POINTS**
- GET /mi/cohorts
- GET /mi/cohorts/vintages
- mi_workflows.analytical capability id `vintage_analysis`
- POST /mi/query via the analytical layer

**EXECUTION_EVIDENCE**
- metricsAvailable (an explicit statement of what the tape supported)
- reconciliation
- declared limitation: redemption / completion / performance curves are NOT computed in the MI path

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **YES** |
| Missing semantic slot | Vintage is an ordinary dimension and the metrics are ordinary aggregations. A plan CAN express this. |
| NEW_PLAN_CONNECTIVITY | **PARTIAL** |
| CONNECTIVITY_GAP | A bare 'balance by vintage year' already reaches the generic executor through the plan (vintage_year is a prepared column). The /mi/cohorts SERVICE shape — book share, metricsAvailable — is a different owner nothing routes a plan to. |
| Gap class | `DISPATCH_ONLY`, `PLAN_TO_SPEC_ADAPTER` |
| MIGRATION_COMPLEXITY | **SMALL** |

> A capability that looks specialist and is largely record 1 with a prepared dimension.

---

### 17. cohort progression (metric progression by vintage across reporting dates)

**USER_INTENT_EXAMPLES**
- _Track the 2023 vintage across reporting dates._
- _How has each vintage's balance progressed?_
- _Show cohort progression for the front book._

| Field | Value |
|---|---|
| DATASET | funded (multi-period) |
| DATASET_OWNER | `mi_agent_api.evolution.funded_frames` |
| PREPARATION_OWNER | `mi_agent_api.funded_prep` |
| **CALCULATION_OWNER** | **`mi_agent_api.evolution.funded_cohort_progression; cohort formation and static-pool membership from mi_agent_api.cohorts.cohort_entry_map / cohort_formation / cohort_static_pool`** |
| MEASURES_SUPPORTED | the governed metric per vintage per reporting date |
| FILTERS_SUPPORTED | **YES_LIMITED** — population, vintage, grain |
| DIMENSIONS_SUPPORTED | vintage x reporting date |
| TEMPORAL_SUPPORT | series, movement |
| POPULATION_SCOPE_SUPPORT | whole book / Direct / Acquired / named source |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **YES** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> chat_routing._route_cohort_progression
- GET /mi/cohorts/progression

**Raw-question reads (exact functions)**
- chat_routing._route_cohort_progression -> _portfolio_lens.resolve_lens_with_default(question, ...) — the population, read from the sentence after the parse.

**EXECUTION_EVIDENCE**
- per-period reconciliation and lineage
- cohort membership basis
- receipt

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | A cohort progression is a MATRIX over (vintage x period) with static-pool membership. No period axis and no membership concept. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | Temporal axis + static-pool membership semantics. |
| Gap class | `TEMPORAL_BINDING`, `MISSING_PLAN_SEMANTIC`, `RAW_TEXT_COUPLING` |
| MIGRATION_COMPLEXITY | **LARGE** |

> 3 corpus declines on spec.cohort_progression.

---

### 18. cohort conversion (cumulative KFI -> Funded)

**USER_INTENT_EXAMPLES**
- _What is our conversion rate?_
- _What proportion of KFIs complete?_
- _How many KFI cases have funded?_

| Field | Value |
|---|---|
| DATASET | pipeline history |
| DATASET_OWNER | `mi_agent_api.pipeline_contract.collect_weekly_history` |
| PREPARATION_OWNER | `mi_agent_api.pipeline_prep.prepare_pipeline_mi_dataset per week` |
| **CALCULATION_OWNER** | **`mi_agent_api.pipeline_history.build_historical_completion_model — empirical stage->completion transitions tracked case-by-case across consecutive weekly snapshots, with a MIN_OBSERVATIONS sufficiency floor below which the configured stage probability is used instead.`** |
| MEASURES_SUPPORTED | cumulative cohort conversion rate, timing per stage |
| FILTERS_SUPPORTED | **NO** — (no further detail) |
| DIMENSIONS_SUPPORTED | stage |
| TEMPORAL_SUPPORT | series (cohort cumulative) |
| POPULATION_SCOPE_SUPPORT | pipeline (whole) |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **NO** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> chat_routing._route_conversion

**EXECUTION_EVIDENCE**
- pipeline_history.historical_model_evidence — observation counts, sufficiency verdict per stage, and whether config or empirical was used
- weekly extract inventory

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | A cohort conversion is a longitudinal CASE-TRACKING calculation, not an aggregation over a frame. A QueryPlan has no concept of following one entity across snapshots. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | Entity-tracking-across-snapshots semantics. |
| Gap class | `MISSING_PLAN_SEMANTIC` |
| MIGRATION_COMPLEXITY | **LARGE** |

> Genuinely specialist. Building the model replays every retained weekly extract, which is why RouteRequest defers it behind a provider.

---

### 19. pipeline stage movement / transitions

**USER_INTENT_EXAMPLES**
- _How many cases went from KFI into Application?_
- _What moved into Offer last week?_
- _How much value transitioned out of Application?_

| Field | Value |
|---|---|
| DATASET | pipeline (two weekly extracts) |
| DATASET_OWNER | `mi_agent_api.pipeline_contract.collect_weekly_history + movement_detail.select_pair` |
| PREPARATION_OWNER | `mi_agent_api.pipeline_prep.prepare_pipeline_mi_dataset + pipeline_prep.case_stage_frame` |
| **CALCULATION_OWNER** | **`mi_agent_api.movement_detail.stage_transition_events -> transition_matrix / new_arrival_summary / stayer_summary / departure_summary / event_totals / stage_reconciliation / global_reconciliation, assembled by movement_detail.build_stage_transition_detail`** |
| MEASURES_SUPPORTED | gross case-level transitions and their amounts, per source/destination stage pair |
| FILTERS_SUPPORTED | **YES_LIMITED** — a named stage or a source/destination pair |
| DIMENSIONS_SUPPORTED | stage x stage (the transition matrix) |
| TEMPORAL_SUPPORT | movement (two governed weekly extracts) |
| POPULATION_SCOPE_SUPPORT | pipeline (whole) |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **NO** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> mi_agent_api.stage_movement_query.handle
- GET /mi/insight/movement-detail
- mi_agent_pptx deck generation

**Raw-question reads (exact functions)**
- mi_agent_api.stage_movement_query.read — recognition-time, memoised; the fallback `or read(request.question)` is the only post-claim re-read and fires only when the memo is absent.

**EXECUTION_EVIDENCE**
- stage_reconciliation and global_reconciliation — every case accounted for as arrival, stayer, transition or departure
- extract pair identity and dates
- receipt

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | A transition is a relationship between an entity's state in two snapshots. AnalyticalScope describes ROWS IN ONE FRAME. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | Entity-state-transition semantics. |
| Gap class | `MISSING_PLAN_SEMANTIC` |
| MIGRATION_COMPLEXITY | **LARGE** |

> This route exists precisely because a STOCK was being substituted for a TRANSITION: 'How many cases went from KFI into Application?' was answered with the current KFI stock. Narrowing this capability during migration would reintroduce that defect.

---

### 20. pipeline movement summary (all stages, one interval)

**USER_INTENT_EXAMPLES**
- _Give me the stage movement summary._
- _Summarise pipeline movement this week._
- _How did the whole pipeline move?_

| Field | Value |
|---|---|
| DATASET | pipeline (two weekly extracts) |
| DATASET_OWNER | `as record 19` |
| PREPARATION_OWNER | `as record 19` |
| **CALCULATION_OWNER** | **`mi_agent_api.pipeline_movement_summary.build — COMPOSES the payload movement_detail.resolve_stage_transition_detail already publishes. It defines no metric and pairs no snapshots; record 19 is the calculation owner.`** |
| MEASURES_SUPPORTED | as record 19, aggregated across all stages |
| FILTERS_SUPPORTED | **NO** — (that is the point — it names no stage) |
| DIMENSIONS_SUPPORTED | all stages |
| TEMPORAL_SUPPORT | movement |
| POPULATION_SCOPE_SUPPORT | pipeline (whole) |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **NO** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> pipeline_movement_summary.handle

**Raw-question reads (exact functions)**
- mi_agent_api.pipeline_movement_summary.read — recognition-time, memoised; fallback re-read at line 364.

**EXECUTION_EVIDENCE**
- inherits record 19's reconciliation wholesale

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | As record 19. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | As record 19. |
| Gap class | `MISSING_PLAN_SEMANTIC` |
| MIGRATION_COMPLEXITY | **LARGE** |

> SAME CALCULATION OWNER as record 19 — two routes, one owner. Recorded separately only because the user intent and the refusal behaviour genuinely differ (record 19 returns nothing without a named stage).

---

### 21. geography analysis (UK ITL3 exposure)

**USER_INTENT_EXAMPLES**
- _Where is the book concentrated geographically?_
- _Show exposure by ITL3 area._
- _What is our exposure in Bristol?_

| Field | Value |
|---|---|
| DATASET | funded |
| DATASET_OWNER | `mi_agent_api.datasets (frame_resolver)` |
| PREPARATION_OWNER | `mi_agent_api.funded_prep; ITL3 derived from the tape's geographic_region_*_itl3 field or from the property postcode via uk_itl_master_lookup_v2.csv` |
| **CALCULATION_OWNER** | **`mi_agent_api.geo.exposure_by_itl3`** |
| MEASURES_SUPPORTED | exposure (balance) and loan count per ITL3 area, with each area's share of the total |
| FILTERS_SUPPORTED | **YES_LIMITED** — population scope |
| DIMENSIONS_SUPPORTED | ITL3 area |
| TEMPORAL_SUPPORT | current |
| POPULATION_SCOPE_SUPPORT | whole book / Direct / Acquired / named source |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **YES** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> chat_routing._route_geo
- GET /mi/geo/exposure
- mi_agent_api.analytical_plan.geo_exposure (converted)

**Raw-question reads (exact functions)**
- mi_agent.mi_geography.stated_basis(question) — WHICH geography the answer is measured on (obligor vs collateral). The geography contract is resolved once at parse and carried on ParsedQuestion.geography, but stated_basis is a raw-text reader by construction.

**EXECUTION_EVIDENCE**
- geography basis disclosure (which geography the book reports on, and how that was decided — mi_agent.mi_geography)
- region_basis_block
- reconciliation
- receipt

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **PARTIAL** |
| Missing semantic slot | ITL3 area is an ordinary dimension and exposure an ordinary sum, so the aggregation is representable. The GEOGRAPHY BASIS is not: AnalyticalScope has no slot for 'measured on the collateral geography', and a plan that omitted it would describe a different analysis from the one that ran. |
| NEW_PLAN_CONNECTIVITY | **PARTIAL** |
| CONNECTIVITY_GAP | A geography-basis slot on the scope. 4 of 12 corpus geo-claimed questions lift today; they lift WITHOUT the basis, which is the risk. |
| Gap class | `MISSING_PLAN_SEMANTIC`, `RAW_TEXT_COUPLING` |
| MIGRATION_COMPLEXITY | **MEDIUM** |

> Committed equivalence evidence for the OTHER plan layer at migration_phase0/plan_equivalence_geo_exposure.py, and shipped evidence at due_diligence/evidence/mi_geography/.

---

### 22. concentration analysis (governed dimension distribution)

**USER_INTENT_EXAMPLES**
- _Where is the book concentrated?_
- _How concentrated is our broker exposure?_
- _What share does the largest region hold?_

| Field | Value |
|---|---|
| DATASET | funded |
| DATASET_OWNER | `mi_agent_api.datasets (frame_resolver) at one reporting date` |
| PREPARATION_OWNER | `mi_agent_api.funded_prep` |
| **CALCULATION_OWNER** | **`mi_workflows.engine.ranked_distribution and mi_workflows.engine.distribution, orchestrated by mi_workflows.concentration_analysis.run_concentration_analysis`** |
| MEASURES_SUPPORTED | exposure share, count share, rank, cumulative share, top-N share, and an explicit unknown block that stays in the denominator |
| FILTERS_SUPPORTED | **YES_LIMITED** — one governed portfolio scope, one reporting date |
| DIMENSIONS_SUPPORTED | YES_GENERIC over the Business Semantics Registry's declarations carrying analytical_role=dimension with the governed `concentration` category (config/business_semantics_registry.yaml) — the registry is the contract, not a list in code. |
| TEMPORAL_SUPPORT | current |
| POPULATION_SCOPE_SUPPORT | one governed portfolio scope |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **NO** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> chat_routing._route_concentration
- mi_workflows.analytical capability id `population_profile`

**Raw-question reads (exact functions)**
- The workflow REFUSES to run without a pre-claim reading (concentration_analysis.py:686 — 'NO READING, NO ANSWER'). requested_concept / requested_single_name_kind were removed from the execution path deliberately; `question` survives only in controlled-failure text.

**EXECUTION_EVIDENCE**
- every exclusion recorded with its reason
- the registry declaration behind every dimension analysed
- unknown share disclosed rather than dropped
- receipt
- reconciliation

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **PARTIAL** |
| Missing semantic slot | The distribution itself is a grouped aggregation plus a ranking — representable the moment record 4's ordering slot exists. The CONCENTRATION FRAMING (which dimensions are concentration dimensions, what the unknown block means) is a registry decision the plan has no slot to reference. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | Ordering/limit slot (record 4) plus a registry-category reference. 10 of 19 corpus concentration questions lift today and are then claimed by this route instead. |
| Gap class | `MISSING_PLAN_SEMANTIC`, `DISPATCH_ONLY`, `PLAN_TO_SPEC_ADAPTER` |
| MIGRATION_COMPLEXITY | **MEDIUM** |

> A SECOND ranking owner (engine.ranked_distribution) alongside the executor's _apply_top_n.

---

### 23. limit assessment / approved concentration tests and headroom

**USER_INTENT_EXAMPLES**
- _Are we breaching any concentration limits?_
- _What is the headroom on the top-3-broker test?_
- _Which limits are closest to breach?_
- _What changed in risk limits this month?_

| Field | Value |
|---|---|
| DATASET | funded + approved limit configuration |
| DATASET_OWNER | `mi_agent_api.snapshots + mi_agent.concentration_tests.store / library (approved configuration), with Schedule 8 extraction via mi_agent.risk_monitor.schedule8_extractor as a declared fallback` |
| PREPARATION_OWNER | `mi_agent_api.funded_prep` |
| **CALCULATION_OWNER** | **`mi_agent.concentration_tests.evaluation.evaluate_active_tests (+ _utilization, _headroom, _breach_amount, summarise), reached through mi_agent_api.concentration_tests_api.compute_concentration_tests. mi_agent_api.risk_limits.compute_risk_limits is a SECOND owner for the React Risk Limits panel, built on analytics_lib.concentration.group_shares / top_n_concentration.`** |
| MEASURES_SUPPORTED | actual, limit, headroom, utilisation, breach amount, status (green/amber/red/needs_review/unavailable), movement vs the prior funded run, source and confidence |
| FILTERS_SUPPORTED | **YES_LIMITED** — the test's own population resolver (evaluation.resolve_population) |
| DIMENSIONS_SUPPORTED | the tested dimension per approved test |
| TEMPORAL_SUPPORT | current; compare_concentration_periods for month-on-month |
| POPULATION_SCOPE_SUPPORT | whole funded book at one reporting date |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **YES** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> chat_routing._route_risk and _route_concentration_tests
- GET /mi/risk-limits
- GET /mi/concentration-tests (+ /drillthrough, /drivers, /history)
- mi_workflows.analytical capability id `concentration_limits`

**Raw-question reads (exact functions)**
- mi_agent_api.concentration_query.detect_intent(question) — chooses between list / get / summary / compare, i.e. WHICH analysis runs, from the sentence, called at chat_routing._route_concentration_tests via conc_query.answer(question, envelope).
- chat_routing._projects_forward(question, spec) — decides whether the question is FORWARD-looking and must be declined rather than answered with today's status.

**EXECUTION_EVIDENCE**
- source precedence always disclosed (approved_configuration vs extracted vs unavailable) and never silent
- missing fields per test
- a test whose input field the book lacks is reported unavailable, never as passing
- funded_attribution_status
- receipt

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | A limit test is actual-vs-threshold-with-status against an APPROVED CONFIGURATION. The plan has no threshold, no status vocabulary and no reference to a governed test library. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | Threshold/limit semantics and a configuration reference. |
| Gap class | `MISSING_PLAN_SEMANTIC`, `RAW_TEXT_COUPLING` |
| MIGRATION_COMPLEXITY | **LARGE** |

> 20 corpus declines. The FORWARD-looking variant is declined by design and is a genuine capability gap — see record 28.

---

### 24. borrowing base / facility utilisation / eligibility

**USER_INTENT_EXAMPLES**
- _What is our borrowing base?_
- _What is the facility utilisation?_
- _How much eligible collateral do we have?_
- _What is ineligible and why?_

| Field | Value |
|---|---|
| DATASET | funded + facility configuration + governed concentration results |
| DATASET_OWNER | `mi_agent_api.borrowing_base_api (resolves the governed frames and the governed concentration results)` |
| PREPARATION_OWNER | `mi_agent_api.funded_prep` |
| **CALCULATION_OWNER** | **`mi_agent.borrowing_base.calculator.calculate (+ summarise_eligibility, concentration_adjustment, nearest_concentration, _invariants); eligibility rules in mi_agent.borrowing_base.eligibility`** |
| MEASURES_SUPPORTED | eligible / ineligible balance, concentration adjustment, borrowing base, drawn, commitment, utilisation, headroom, nearest concentration test |
| FILTERS_SUPPORTED | **YES_LIMITED** — eligibility rules are the filter vocabulary |
| DIMENSIONS_SUPPORTED | ineligibility reason; concentration test |
| TEMPORAL_SUPPORT | current |
| POPULATION_SCOPE_SUPPORT | whole funded book |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **NO** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> mi_agent_api.borrowing_base_query.handle
- GET /mi/borrowing-base
- GET /mi/borrowing-base/loans
- React Eligibility & Concentrations tab

**Raw-question reads (exact functions)**
- mi_agent_api.borrowing_base_query.read(question, spec, view=...) — recognition-time; fallback re-read at line 649.

**EXECUTION_EVIDENCE**
- mi_agent.borrowing_base.receipt
- calculator._invariants (overlap and totals checks)
- eligibility rule attribution per loan
- the same envelope the dashboard renders — one calculation, three surfaces

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | Eligibility is a RULE SET, not a predicate set; a borrowing base is a contractual construction. Neither has a plan slot. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | Facility/eligibility semantics. |
| Gap class | `MISSING_PLAN_SEMANTIC` |
| MIGRATION_COMPLEXITY | **LARGE** |

> Genuinely specialist; it calculates nothing in the MI layer — borrowing_base_query answers from the governed envelope.

---

### 25. forecast / expected funding — run-rate extrapolation and milestones

**USER_INTENT_EXAMPLES**
- _When do we reach £100m?_
- _What is our completion run-rate?_
- _How long until the book doubles?_
- _What is the expected funding over the next six months?_

| Field | Value |
|---|---|
| DATASET | funded history + pipeline |
| DATASET_OWNER | `mi_agent_api.evolution.funded_frames + mi_agent_api.pipeline_contract` |
| PREPARATION_OWNER | `funded_prep + pipeline_prep` |
| **CALCULATION_OWNER** | **`mi_agent_api.forecast_extrapolation.run_rate_model (Model A), kfi_conversion_model (Model B) and build_extrapolation (assembly + milestone solving), over completion_history`** |
| MEASURES_SUPPORTED | monthly run-rate, annualised rate, projected balance series, milestone dates to declared thresholds, downside/base/upside BANDS (explicitly labelled indicative scenario bands, not statistical CIs) |
| FILTERS_SUPPORTED | **NO** — (no further detail) |
| DIMENSIONS_SUPPORTED | none (whole-book projection) |
| TEMPORAL_SUPPORT | series (forward), other (milestone solving) |
| POPULATION_SCOPE_SUPPORT | whole book |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **YES** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> chat_routing._route_forecast
- GET /mi/forecast/extrapolation
- GET /mi/evolution/forecast
- mi_workflows.analytical capability ids `completion_run_rate` and `threshold_projection`

**Raw-question reads (exact functions)**
- chat_routing._route_forecast -> mi_agent.period_request.requested_unit(question) — the temporal unit of the projection, read from the sentence after the parse.

**EXECUTION_EVIDENCE**
- the three models are kept apart and labelled; the point-in-time weighted pipeline is NOT presented as the scale-up forecast
- band basis disclosure
- horizon bound — beyond it the answer is 'not within the horizon', never an extrapolated date
- receipt

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | Projection, run-rate and milestone-solving are forward-looking constructions. A QueryPlan describes rows that EXIST. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | Projection semantics. |
| Gap class | `MISSING_PLAN_SEMANTIC`, `RAW_TEXT_COUPLING` |
| MIGRATION_COMPLEXITY | **LARGE** |

> 28 corpus declines — the second largest decline group after ranking.

---

### 26. forecast bridge (funded + probability-weighted pipeline)

**USER_INTENT_EXAMPLES**
- _What will the book be worth once the pipeline completes?_
- _What is the forecast funded balance?_
- _Which pipeline cases are on the watchlist?_

| Field | Value |
|---|---|
| DATASET | funded + pipeline (aggregate composition) |
| DATASET_OWNER | `mi_agent_api.snapshots + mi_agent_api.pipeline_contract` |
| PREPARATION_OWNER | `funded_prep + pipeline_prep (which sources completion probabilities from config/client/pipeline_expected_funding.yaml — never from the frontend)` |
| **CALCULATION_OWNER** | **`mi_agent_api.forecast_bridge.compute_forecast_bridge (+ portfolio_projections, build_pipeline_watchlist), using the same formula implemented row-level in mi_agent.states.assembler.total_forecast_funded`** |
| MEASURES_SUPPORTED | forecast_funded_balance = current_funded_balance + sum(expected_funded_amount * completion_probability); plus a deterministic watchlist |
| FILTERS_SUPPORTED | **NO** — (no further detail) |
| DIMENSIONS_SUPPORTED | portfolio (projections) |
| TEMPORAL_SUPPORT | current + forward |
| POPULATION_SCOPE_SUPPORT | whole funded book + whole pipeline |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **NO** |

**ORCHESTRATION_ENTRY_POINTS**
- GET /mi/forecast/snapshot
- mi_workflows.analytical capability id `funded_balance_forecast`
- POST /mi/query via the analytical layer

**EXECUTION_EVIDENCE**
- an AGGREGATE composition — pipeline rows are never merged into the funded book, and the module says so
- without a governed pipeline the forecast is reported unavailable and the funded balance is explicitly NOT labelled a forecast
- probability provenance

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | Two datasets composed at the AGGREGATE level, with a probability weighting. A QueryPlan has one dataset slot and no cross-dataset composition. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | Cross-dataset aggregate composition. |
| Gap class | `MISSING_PLAN_SEMANTIC`, `DATASET_BINDING` |
| MIGRATION_COMPLEXITY | **LARGE** |

> Committed hardening evidence at due_diligence/evidence/forecast_composition_hardening/.

---

### 27. scenario / what-if on the completion run-rate

**USER_INTENT_EXAMPLES**
- _If conversion improved by 10%, when do we reach £50m?_
- _What if our run-rate halved?_

| Field | Value |
|---|---|
| DATASET | derived from the forecast base |
| DATASET_OWNER | `mi_agent_api.forecast_extrapolation (resolves the base)` |
| PREPARATION_OWNER | `n/a — the engine is pure` |
| **CALCULATION_OWNER** | **`mi_agent_api.scenario.apply_scenario (+ multiplier_from_conversion_delta)`** |
| MEASURES_SUPPORTED | adjusted projected balance series and milestone date, alongside the unchanged base for comparison |
| FILTERS_SUPPORTED | **NO** — (no further detail) |
| DIMENSIONS_SUPPORTED | none |
| TEMPORAL_SUPPORT | other (forward, perturbed) |
| POPULATION_SCOPE_SUPPORT | whole book |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **YES** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> chat_routing._route_scenario

**Raw-question reads (exact functions)**
- chat_routing._scenario_multiplier(question) — the MAGNITUDE of the perturbation, quantified from the sentence at chat_routing.py:2263. The engine itself is pure and takes typed overrides; the number reaching it comes from text.

**EXECUTION_EVIDENCE**
- base and adjusted series both returned, so the perturbation is visible rather than asserted
- receipt

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | A counterfactual perturbation of a forward model. No plan slot, and no rows to scope. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | Scenario/perturbation semantics. |
| Gap class | `MISSING_PLAN_SEMANTIC`, `RAW_TEXT_COUPLING` |
| MIGRATION_COMPLEXITY | **LARGE** |

> The engine is side-effect free and would be trivially reusable; it is the MAGNITUDE extraction that is text-bound.

---

### 28. portfolio risk comparison (two governed scopes at one date)

**USER_INTENT_EXAMPLES**
- _Compare the direct and acquired books._
- _How does portfolio A compare with portfolio B on risk?_

| Field | Value |
|---|---|
| DATASET | funded |
| DATASET_OWNER | `mi_agent_api.datasets (one frame, two scopes applied)` |
| PREPARATION_OWNER | `mi_agent_api.funded_prep` |
| **CALCULATION_OWNER** | **`mi_workflows.portfolio_risk_comparison.run_portfolio_risk_comparison, over mi_workflows.engine.aggregate / compare_values / directionality_verdict / mixed_currency_guard`** |
| MEASURES_SUPPORTED | every field the Business Semantics Registry declares comparable, with its declared aggregation, weight, share basis and directionality |
| FILTERS_SUPPORTED | **YES_LIMITED** — the two governed scopes |
| DIMENSIONS_SUPPORTED | the compared fields |
| TEMPORAL_SUPPORT | current (one reporting date, two populations) |
| POPULATION_SCOPE_SUPPORT | two governed portfolio scopes |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **YES** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> chat_routing._route_portfolio_comparison

**Raw-question reads (exact functions)**
- mi_workflows.portfolio_risk_comparison.resolve_comparison_scopes(question, reg) — called INSIDE run_portfolio_risk_comparison (line 524). The two POPULATIONS are decided from raw text at execution time. Contrast concentration_analysis, whose equivalent read was moved to recognition and memoised.

**EXECUTION_EVIDENCE**
- comparability decision per field, from the registry
- shared asset class resolution
- mixed-currency guard — monetary metrics only compared where currency profiles match, no FX ever
- receipt

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **PARTIAL** |
| Missing semantic slot | TWO populations over one frame is the one thing QueryPlan was built for — but only as NARROWINGS of one shared scope. Two sibling populations that are not nested cannot both be a ScopeDelta of one shared scope, and there is no comparison relationship. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | Sibling (non-nested) scopes plus a comparison relationship. |
| Gap class | `MISSING_PLAN_SEMANTIC`, `RAW_TEXT_COUPLING` |
| MIGRATION_COMPLEXITY | **LARGE** |

> The ONE workflow whose population is still resolved from raw text inside the calculation entry point.

---

### 29. analytical composition layer (multi-capability plans)

**USER_INTENT_EXAMPLES**
- _How is the book doing and where are we most exposed?_
- _Summarise the position and tell me if we are near any limits._
- _What is the front book worth versus the back book?_

| Field | Value |
|---|---|
| DATASET | varies by plan |
| DATASET_OWNER | `mi_workflows.analytical.context (lazy, memoised; delegates every resolution to the existing service)` |
| PREPARATION_OWNER | `delegated` |
| **CALCULATION_OWNER** | **`NONE OF ITS OWN — mi_workflows.analytical.orchestrator runs each call through the capability's registered deterministic executor. tests/test_analytical_capability_layer.py parses the source to enforce that no adapter computes a financial result.`** |
| MEASURES_SUPPORTED | the union of the ten declared capabilities in mi_workflows/analytical/registry.py |
| FILTERS_SUPPORTED | **NO** — delegated |
| DIMENSIONS_SUPPORTED | delegated |
| TEMPORAL_SUPPORT | delegated |
| POPULATION_SCOPE_SUPPORT | mi_workflows.analytical.populations — the ONLY place a population is created here, and every one comes from an existing governed resolver (mi_agent.seasoning for the front/back binary partition, the portfolio registry for provenance) |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **YES** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query -> mi_workflows.analytical.route.recogniser() — registered FIRST and at a higher confidence, and declining every question a single-capability route already owns

**Raw-question reads (exact functions)**
- mi_workflows.analytical.planner.plan_for reads the structured parse plus a controlled vocabulary to build the plan. This is PLANNING, not post-plan re-reading — but it is a separate reader from the QueryPlan lift, and the two plans do not compose.

**EXECUTION_EVIDENCE**
- Finding objects carrying their own unavailability notes
- narrative may only read findings, so a sentence cannot contain a figure the deterministic layer did not produce
- per-capability limitations declared in the registry

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | This is a plan OVER capabilities; QueryPlan is a plan over one population's outputs. They are different altitudes and neither contains the other today. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | No relationship defined between the two plan artefacts. |
| Gap class | `MISSING_PLAN_SEMANTIC` |
| MIGRATION_COMPLEXITY | **LARGE** |

> 35 corpus questions claimed; 29 of them ALSO lift to a QueryPlan at parse, which is a measured sign the two layers overlap without composing.

---

### 30. source portfolio scope (Direct / Acquired / named source / cohort)

**USER_INTENT_EXAMPLES**
- _What is the balance in the acquired book?_
- _Direct originations only._
- _How does direct_001 compare?_

| Field | Value |
|---|---|
| DATASET | cross-cutting |
| DATASET_OWNER | `mi_agent_api.portfolio_context.build_registry — the SAME governed registry React renders its portfolio selector from` |
| PREPARATION_OWNER | `engine/provenance.py stamps source_portfolio_type / source_portfolio_id at onboarding` |
| **CALCULATION_OWNER** | **`NOT A CALCULATION — mi_agent.portfolio_lens.resolve_lens_with_default resolves the NAME, and mi_agent_api.portfolio_context.resolve_context resolves what it CONTAINS (an explicit portfolio-id list, so a newly onboarded member widens the group with no code change). The narrowing is then an ordinary predicate.`** |
| MEASURES_SUPPORTED | n/a |
| FILTERS_SUPPORTED | **YES_GENERIC** — realised as a filter on the provenance fields; it only narrows rows and never changes a calculation |
| DIMENSIONS_SUPPORTED | source_portfolio_id / source_portfolio_type |
| TEMPORAL_SUPPORT | n/a |
| POPULATION_SCOPE_SUPPORT | Total / Direct / Acquired / named source / cohort |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **YES** |

**ORCHESTRATION_ENTRY_POINTS**
- every lens-aware route (chat_routing._resolve_lens, four call sites)
- the generic path (mi_agent_workflow.py:763)
- QuestionInterpretation.source_scope (the converted routes)

**Raw-question reads (exact functions)**
- mi_agent.portfolio_lens.resolve_lens_with_default(question, default_lens, ...) — a question's own words OVERRIDE the caller's UI selection by design. Called at mi_agent/mi_agent_workflow.py:763 (generic path) and mi_agent_api/chat_routing.py:579/4019/4075/4724.

**EXECUTION_EVIDENCE**
- resolved portfolio-id list, never a type string, so a group is exactly the sum of its registered members
- lensApplied boolean + explicit disclosure when a non-total lens could NOT be applied
- a disclosure when the QUESTION'S WORDING WIDENED the caller's selection (chat_routing._disclose_wording_widened_the_scope)
- a warning when the requested portfolio is not in the registry and the answer therefore covers the TOTAL book

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **PARTIAL** |
| Missing semantic slot | AnalyticalScope HAS a portfolio_lens slot. It is simply not populated: mi_agent/parsed_question.py:154 calls compiled_spec_for(spec) with no scope kwargs, so the lens comes from spec.portfolio_lens, which is empty at parse and is set later. |
| NEW_PLAN_CONNECTIVITY | **PARTIAL** |
| CONNECTIVITY_GAP | The seam exists and is unused. plan_from_spec already accepts dataset / portfolio_lens / period kwargs; nothing passes them. |
| Gap class | `DATASET_BINDING`, `RAW_TEXT_COUPLING` |
| MIGRATION_COMPLEXITY | **TRIVIAL** |

> ONE OF THE CHEAPEST REAL GAPS IN THE CENSUS: the contract, the resolver and the parameter all exist. Only the call site is missing.

---

### 31. seasoning / front-book vs back-book population

**USER_INTENT_EXAMPLES**
- _What is the front book worth?_
- _Balance by seasoning segment._
- _How much was written in the last lending window?_

| Field | Value |
|---|---|
| DATASET | funded |
| DATASET_OWNER | `mi_agent_api.datasets` |
| PREPARATION_OWNER | `mi_agent_api.funded_prep (vintage_year, months_on_book); mi_agent.seasoning.months_between` |
| **CALCULATION_OWNER** | **`NOT A CALCULATION — mi_agent.seasoning.resolve_population_predicate / resolve_segment_population decide the BOUNDARY (a governed BINARY PARTITION, so naming one side makes the other available by construction). The aggregation is record 1's.`** |
| MEASURES_SUPPORTED | n/a |
| FILTERS_SUPPORTED | **YES_GENERIC** — once resolved to a predicate |
| DIMENSIONS_SUPPORTED | seasoning segment |
| TEMPORAL_SUPPORT | n/a |
| POPULATION_SCOPE_SUPPORT | front book / back book / named lending window |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **YES** |

**ORCHESTRATION_ENTRY_POINTS**
- the generic path (as a predicate)
- mi_workflows.analytical.populations

**Raw-question reads (exact functions)**
- mi_agent.seasoning.segments_named(text) / lending_windows_named(text) / resolve_population_predicate(text, ...) — the boundary is read from the sentence.

**EXECUTION_EVIDENCE**
- the configured seasoning boundary is disclosed
- mi_agent.population.apply_population emits proof the predicate ran

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **YES** |
| Missing semantic slot | A resolved seasoning predicate is an ordinary Predicate on a prepared column. |
| NEW_PLAN_CONNECTIVITY | **PARTIAL** |
| CONNECTIVITY_GAP | Where the parser resolves the segment into spec.filters the plan carries it. Where seasoning is resolved later (the analytical layer's populations module) it does not. |
| Gap class | `RAW_TEXT_COUPLING` |
| MIGRATION_COMPLEXITY | **SMALL** |

> A governed BINARY partition, which is why naming one side is safe.

---

### 32. week-on-week movement attribution (hover / drill)

**USER_INTENT_EXAMPLES**
- _What changed between these two weeks?_
- _What contributed to the pipeline movement?_

| Field | Value |
|---|---|
| DATASET | pipeline (two weekly extracts) |
| DATASET_OWNER | `mi_agent_api.movement_detail.select_pair over pipeline_contract.collect_weekly_history` |
| PREPARATION_OWNER | `mi_agent_api.pipeline_prep` |
| **CALCULATION_OWNER** | **`mi_agent_api.movement_detail.movement_components (+ component_summary, rank_contributors, reassignment_counts), assembled by build_movement_detail`** |
| MEASURES_SUPPORTED | the SAME number the chart already plots, decomposed — no metric is defined here |
| FILTERS_SUPPORTED | **YES_LIMITED** — the dimension being decomposed |
| DIMENSIONS_SUPPORTED | any dimension the prepared frame carries |
| TEMPORAL_SUPPORT | movement (two adjacent weekly extracts) |
| POPULATION_SCOPE_SUPPORT | pipeline (whole) |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **NO** |

**ORCHESTRATION_ENTRY_POINTS**
- GET /mi/insight/movement-detail
- mi_agent_api.insight_engine (weekly brief — composes, does not calculate)

**EXECUTION_EVIDENCE**
- headline recomputed from the same prepared frames the chart used
- component reconciliation
- extract pair identity

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | A decomposition of a delta between two frames. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | Movement decomposition semantics (shared with record 15). |
| Gap class | `MISSING_PLAN_SEMANTIC` |
| MIGRATION_COMPLEXITY | **LARGE** |

> Dashboard-facing; reachable from MI Query only indirectly, through records 19-20 which consume the same module.

---

### 33. FORWARD-LOOKING approved-limit projection

**USER_INTENT_EXAMPLES**
- _Do we expect to breach any concentration limits?_
- _Which concentration tests are we at risk of breaching?_

| Field | Value |
|---|---|
| DATASET | funded + pipeline + approved limit configuration |
| DATASET_OWNER | `n/a` |
| PREPARATION_OWNER | `n/a` |
| **CALCULATION_OWNER** | **`NONE. This is the census's one GENUINE_CAPABILITY_GAP.`** |
| MEASURES_SUPPORTED | none |
| FILTERS_SUPPORTED | **NO** — n/a |
| DIMENSIONS_SUPPORTED | n/a |
| TEMPORAL_SUPPORT | none |
| POPULATION_SCOPE_SUPPORT | n/a |
| RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION | **YES** |

**ORCHESTRATION_ENTRY_POINTS**
- POST /mi/query — chat_routing's `risk_limits` recogniser DECLINES when chat_routing._projects_forward(question, spec) is true, and the existing forward-projection facet produces the refusal.

**Raw-question reads (exact functions)**
- chat_routing._projects_forward(question, spec)

**EXECUTION_EVIDENCE**
- A CONTROLLED REFUSAL rather than a substitution. Measured before the gate, these questions were answered with TODAY's status ('5 passed, 6 breaches … Nearest to limit: Top 3 brokers') — current-state substitution, recorded against Q25A/B/C in the frozen readiness bank.

| Migration | |
|---|---|
| GOVERNED_QUERY_PLAN_REPRESENTABLE | **NO** |
| Missing semantic slot | n/a — there is nothing to represent. |
| NEW_PLAN_CONNECTIVITY | **NOT_CONNECTED** |
| CONNECTIVITY_GAP | n/a |
| Gap class | `GENUINE_CAPABILITY_GAP` |
| MIGRATION_COMPLEXITY | **LARGE** |

> risk_monitor.run_funded_vs_forecast is deliberately NOT wired in: it forecasts group SHARES against placeholder RAG thresholds (amber 0.20 / red 0.30), carries no approved test name and no headroom, needs a caller-supplied dimension and a SnapshotStore the MI path does not construct. Documented at migration_phase0/MI_CURRENT_VS_FORWARD_CONCENTRATION.md. THE ONLY GENUINE CAPABILITY GAP IN THIS CENSUS.

---

## Reproducing this census

```bash
git checkout 4337d71de1e0c92e66d09511bb5471a9a4a6d475
python3 due_diligence/evidence/mi_capability_owner_census/probes/lift_census_probe.py
python3 due_diligence/evidence/mi_capability_owner_census/probes/route_claim_probe.py
python3 due_diligence/evidence/mi_capability_owner_census/probes/targeted_route_probe.py
python3 -m pytest mi_agent/tests/test_query_plan_adapter.py \
    mi_agent/tests/test_query_plan_compiler.py \
    mi_agent/tests/test_query_plan_contracts.py \
    mi_agent/tests/test_query_plan_execution.py \
    mi_agent/tests/test_query_plan_is_the_live_contract.py \
    mi_agent/tests/test_query_plan_reconciliation.py \
    mi_agent/tests/test_shadow_replay.py -q
python3 due_diligence/evidence/mi_capability_owner_census/build_census.py
```

Requires `pandas`, `pyyaml`, `plotly`, `pytest`. No network access and no API key: every probe runs the deterministic parser.
