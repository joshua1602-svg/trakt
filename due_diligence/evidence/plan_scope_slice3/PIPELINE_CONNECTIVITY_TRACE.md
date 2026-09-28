# Slice 3 — pipeline connectivity: Phase 1 trace, and a stop

Traced at `425a1768`. **No product code changed. No live Opus call. Nothing deployed.**

---

## 1. The existing pipeline estate

```
PIPELINE_CURRENT_DATASET_OWNER   mi_agent_api.datasets._resolve_query_frame("pipeline", portfolio_id)
                                   -> datasets._resolve_pipeline_source(client_id, run_id)
                                   -> pipeline_contract.load_prepared_pipeline(source, historical_model=…)

PIPELINE_CURRENT_EXECUTION_OWNER   the LEGACY chat route. A prepared pipeline frame
                                   plus the legacy MIQuerySpec; envelopes carry
                                   reconciliation {dataset: "pipeline"} and NO
                                   execution receipt.

PIPELINE_TEMPORAL_DATASET_OWNER    pipeline_contract.weekly_extract_inventory over
                                   datasets._pipeline_discovery_root()   (WEEKLY extracts)

PIPELINE_TEMPORAL_EXECUTION_OWNER  mi_agent_api.evolution.pipeline_evolution(...)
                                   reached from chat_routing route
                                   "evolution_pipeline_stage" / "evolution_funnel"

PIPELINE_STAGE_OWNER               question_interpretation.lexical.PIPELINE_STAGE_FIELD
PIPELINE_HISTORY_OWNER             datasets._pipeline_history -> pipeline_contract.build_pipeline_history
```

All of these are healthy and none needs rebuilding.

---

## 2. What Opus actually produces, and where it stops

Not hypothesised — the **recorded intent** from the Slice 3 affordance run for
*"What is in the front book?"*:

```json
{"capability": "pipeline", "operation": "summary",
 "population": {"base": "pipeline"},
 "measures": [{"concept": "pipeline_amount"}], "time": {"form": "current"}}
```

Compiled, that plan carries:

```
capability          pipeline                  (specialist)
population.base     pipeline                  ✓ correct
measures[0]         pipeline_amount
                      canonical_field  null
                      statistic        "capability"
                      capability_owner "pipeline"
```

Put to the runtime, **three** things stop it, in this order:

```
1. adapter.check_eligibility(plan)
     -> (False, CAPABILITY_NOT_GENERIC,
         "capability='pipeline' is specialist; it owns its own input contract
          and arithmetic")

2. adapter.spec_for_plan(plan)
     -> raises KeyError 'capability'
        the adapter cannot express a specialist measure at all: there is no
        canonical field to bind and the statistic is the string "capability"

3. adapter.check_population_base(plan, "pipeline")
     -> (False, POPULATION_NOT_EXECUTABLE, …)
```

**The brief's premise is half right.** `population.base = pipeline` does reach
the plan correctly and does stop. But it is not the thing stopping it first, and
handing the base to a pipeline owner would not help: the plan's *capability* and
its *measure* are specialist, and `spec_for_plan` raises before any population
question is asked.

---

## 3. The real gap

There is **no governed execution owner for any specialist capability.**

```
ELIGIBLE_OPERATIONS   {point_in_time, breakdown}
check_capability      generic_analysis only
```

Slice 1 built one runtime — `generic_analysis` over canonical fields — and
Slice 2 added its temporal variant. Nothing has ever executed `pipeline`,
`borrowing_base`, `funded_bridge`, `pipeline_stage_movement`, `forecast`,
`concentration`, `limit_assessment` or `portfolio_summary` from a
`GovernedQueryPlan`.

The existing pipeline capability is real and accepted, but it lives in the
**legacy** estate: it consumes a legacy `MIQuerySpec` and the raw question, and
emits envelopes with `reconciliation {dataset: "pipeline"}` — not an execution
receipt with `applied_predicates`, which is what the governed coverage ledger
reconciles against.

So connecting it means building the first **specialist-capability governed
runtime**: plan → capability dispatch, an execution adapter over the existing
pipeline owners, and governed evidence manufactured from their reports so
requested-vs-executed can be proven. That is a third runtime tier beside Slice 1
and Slice 2, and it is the template six other capabilities would then follow.

That is not "routing/adapter integration only".

---

## 4. What IS a minimum bridge, and what is not

Measured, not asserted. A **generic_analysis** plan over the pipeline population
already compiles, is eligible, and binds a spec:

| plan | perimeter | spec bound | stopped only by |
|---|---|---|---|
| `generic_analysis` / `point_in_time`, base=pipeline, measure `loan` | eligible | yes | population gate |
| `generic_analysis` / `breakdown`, base=pipeline, dim `pipeline_stage` | eligible | yes, `dimensions=['pipeline_stage']` | population gate |
| `pipeline` / `summary`, base=pipeline, measure `pipeline_amount` | **CAPABILITY_NOT_GENERIC** | **raises** | — |

So the brief's acceptance list splits cleanly:

**Reachable by a genuine minimum bridge** — extend `EXECUTABLE_POPULATIONS`, and
let the plan drive which frame the existing dataset owner resolves:

* **C2** pipeline case count
* **C3** pipeline by stage
* any governed filter/dimension the pipeline tape carries

**Not reachable without the new specialist tier:**

* **C1 / P1** *"What is the pipeline balance?"* — `pipeline_amount` is
  capability-owned and has no canonical field
* **P5** *"Show pipeline balance each week."* — same measure, plus weekly history
* **T1 / T2 / P4** pipeline evolution — the temporal runtime is funded-only by
  construction (`snapshot_route = FUNDED_ROUTE`, monthly `SnapshotStore`), while
  pipeline history is weekly extracts under a different owner. Connecting it
  needs a second temporal runtime, not a parameter.

---

## 5. Stop

The brief says: *"If the existing pipeline capability would need to be rebuilt
or materially redesigned, STOP and report why."*

The pipeline capability does **not** need rebuilding. What is missing is the
governed execution tier that would consume it, and that tier does not exist for
any capability. Building it is Slice-1-sized work, not a bridge — and building
only the generic half would connect counts and stage breakdowns while leaving
*"What is the pipeline balance?"* refusing, which is not
`CURRENT_PIPELINE_CONNECTED = YES` by any honest reading.

```
SLICE_3_PIPELINE_CONNECTIVITY = INCONCLUSIVE — scope finding, implementation not started
SLICE_3 = NOT CLOSED
```

Nothing was changed, deployed, or asked of the model.
