# Production-path architecture trace, and the integration decision

Read-only. No product file changed, no branch merged, no model called, no
deployed API touched.

```
PRODUCTION_ENTRYPOINT = mi_agent_api/app.py :: query()   @app.post("/mi/query")
```

## Phase 1 — the actual production call graph

Every line below is a real `file :: function`. Nothing is inferred from naming.

| # | stage | file :: function | in → out | reads raw question? | narrows semantics? | selects frame? | calculates? | receipt? |
|---|---|---|---|---|---|---|---|---|
| 1 | HTTP route | `mi_agent_api/app.py :: query` | `QueryRequest` → React payload | no | no | no | no | no |
| 2 | auth / principal | `app.py :: _execution_context` (via `auth.auth_guard`, `principal_from_request`) | `Request` → `ExecutionContext` | no | no | no | no | tenant identity |
| 3 | governed service | `mi_agent_api/mi_service.py :: execute_governed_mi_query` | `MiQueryRequest` + `ExecutionContext` → `GovernedResult` | no | no | no | no | owns the governance envelope |
| 4 | **dataset resolution** | `mi_agent_api/workspace.py :: resolve_dataset(question)` (called at `mi_service.py:487`) | question → `view` | **YES** | yes — picks the population | **yes** | no | `view` reaches the scope ref |
| 5 | tenancy / portfolio authorisation | `mi_service.py` §1 (`PolicyState`, `AuthorisedPortfolio`) | context + portfolio_id → authorised portfolio | no | no | no | no | policy + snapshot ref |
| 6 | dataset approval | `mi_service.py :: deps.datasets.describe_active_dataset` → `_snapshot_ref` | descriptor → approval state | no | no | no | no | `snapshot` ref |
| 7 | **frame resolution** | `mi_service.py :: _run_analysis` → `_resolve_frame(ds, view, portfolio_id)` | `view`, portfolio → DataFrame | no | no | **YES** | no | frame identity |
| 8 | **geography binding** | `_run_analysis` → `_resolve_geography(client_id, portfolio_id, df)` → `_parser_mod.bind_geography` | client/portfolio/frame → geography contract | **no** | no | no | no | yes |
| 9 | **interpretation** | `mi_agent/parsed_question.py :: ParsedQuestion.parse(question, semantics, geography=…, available_columns=…, available_values=…, semantics_resolver=…)` | question + contracts → `ParsedQuestion` (carries `MIQuerySpec`) | **YES** | yes — the whole semantic act | no | no | `parsed.meta` |
| 10 | caller filters merged | `parsed.merge_filters(req.filters)` | HTTP filters → spec filters | no | yes | no | no | yes |
| 11 | **routing / capability selection** | `mi_agent_api/chat_routing.py :: try_route(question, *, portfolio_id, view, as_of, source_lens, parsed, …)` via `recogniser_registry.py :: RecogniserRegistry.candidates` | question + parsed → routed envelope or `None` | **YES** (see §Raw-text rereads) | yes | per-route | delegates | builds `QuestionInterpretation` |
| 12 | **point-in-time execution** | `mi_agent/mi_agent_workflow.py :: run_mi_agent_query` | question + frame + semantics → result dict | **YES** | yes | no | **YES** | `execution_receipt` |
| 13 | lens resolution + application | `mi_agent_workflow.py:~764` `portfolio_lens.resolve_lens_with_default(question, _default_lens, available_values=…)`; `portfolio_registry.resolve(df, context_id(lens))`; `portfolio_lens.apply_scope(spec, _lens, _portfolio_scope)` | question + caller lens + registry → `spec.portfolio_lens` + id list | **YES** | yes — narrows population | yes (row filter) | no | `result["portfolio_lens"]` |
| 14 | calculation | `mi_agent/mi_query_executor.py :: execute_mi_query(spec, data, semantics)` | spec + frame → `MIQueryResult` | no | no | no | **YES** | metadata |
| 15 | receipt | `mi_agent/execution_receipt.py` | result + spec → receipt | no | no | no | no | **owns it** |
| 16 | presentation | `mi_agent_api/adapters.py :: adapt_workflow_result`, `presenters.to_react_payload` | result → React envelope | no | no | no | no | carries governance block |

Two notes the code states itself. `app.py:421-423`: `datasetContext` is
*"DISPLAY CONTEXT ONLY — it does not decide which dataset the question runs
… See `mi_agent_api.workspace.resolve_dataset`."* And `app.py:2042`: the route is
*"a THIN adapter… No parsing, routing, dataset resolution, calculation,
validation, policy or provenance logic lives here."* The adapter is clean; the
semantics live in stages 4–13.

## Phase 2 — current semantic owners

### A. DATASET

```
DATASET_OWNER            = mi_agent_api/workspace.py :: resolve_dataset(question)
                           (called mi_service.py:487, BEFORE governance)
DATASET_SELECTION_INPUT  = the raw question string. The React tab
                           (`datasetContext`) is explicitly NOT an input — the
                           comment at mi_service.py:480-481 records that folding
                           it in used to serve "the same sentence from a
                           different dataset"
DATASET_SELECTION_OUTPUT = `view` (a string), consumed by
                           mi_service._resolve_frame(ds, view, portfolio_id)
                           → a DataFrame
RAW_TEXT_DEPENDENCY      = YES
DUPLICATE_OWNERS         = NO — single owner, and it runs before interpretation
```

The calculation layer does **not** know which dataset was selected:
`execute_mi_query(spec, data, semantics)` receives the frame, and `MIQuerySpec`
has no `dataset` field (verified in Phase B). Dataset identity survives only in
the governance envelope (`_scope_ref`, `_snapshot_ref`), not in the calculation
receipt.

### B. PORTFOLIO LENS

```
PORTFOLIO_LENS_OWNER    = DUPLICATED
                          (1) mi_agent/mi_agent_workflow.py:~764 — point-in-time
                          (2) mi_agent_api/chat_routing.py:606 :: _resolve_lens — routed
                          both call portfolio_lens.resolve_lens_with_default(question, default)
LENS_SELECTION_INPUT    = raw question text, with the caller's
                          `sourcePortfolioLens` as the DEFAULT
                          (`lens_from_selection(source_lens)`)
LENS_APPLICATION_STAGE  = portfolio_lens.apply_scope(spec, lens, scope) after
                          portfolio_registry.resolve(df, context_id(lens)) —
                          the filter is a resolved portfolio-id list, never a
                          type string
RAW_TEXT_DEPENDENCY     = YES
DUPLICATE_OWNERS        = YES (two resolution sites, one application contract)
```

The default is correct per the business semantic: no lens named → no narrowing.
`apply_lens`/`apply_scope` is a single, language-free application owner; only
*resolution* is duplicated.

### C. PERIOD / TEMPORAL

```
TEMPORAL_OWNER          = SPLIT, and the snapshot runtime is NOT on /mi/query
  single-period          mi_agent_workflow.run_mi_agent_query over ONE frame
                         resolved at stage 7. `as_of_date` arrives from the HTTP
                         body (app.py:2062) and is passed to chat_routing.try_route
                         as `as_of=`; mi_agent_workflow itself contains no
                         as_of/snapshot/store execution logic.
  multi-snapshot         REFUSED on this path. mi_agent_workflow.py:339-352
                         returns a governed message: a capability "measured
                         ACROSS governed snapshots" cannot execute here, "No
                         value has been computed".
  temporal comparison    mi_agent_api/temporal_compare.py, reached as the
                         `temporal_compare` recogniser route; also
                         `period_movement`, `funded_bridge`, `cohort_progression`,
                         `evolution` routes.
SNAPSHOT_SELECTION_OWNER = mi_agent/states/selectors.py :: SnapshotSelector
                           (.as_of / .latest / .range / .compare), consumed by
                           mi_agent/mi_runtime.py :: _run_state / _run_temporal.
                           **Reached only from mi_agent/interpreter/runtime_bridge.py:143
                           :: interpret_and_run_mi_query — NOT from /mi/query.**
PERIOD_PAIR_OWNER        = mi_agent/period_change/periods.py (governed
                           two-snapshot resolution over SnapshotFrame sequences)
GAP_POLICY_OWNER         = mi_agent/period_change/periods.py :: _guard_gap
                           (max_snapshot_gap_days)
RAW_TEXT_DEPENDENCY      = YES — mi_agent/period_change/recognition.py recognises
                           period language; `as_of` is additionally supplied by
                           the caller
DUPLICATE_OWNERS         = YES. Three temporal concepts coexist: the HTTP
                           `as_of_date`, raw-text period recognition, and the
                           SnapshotSelector/store path that /mi/query never
                           reaches.
```

This is the single most important structural finding of the trace, and it
explains Phase B cleanly: `/mi/query` is a **single-frame architecture**. The
snapshot machinery exists, is governed, and is wired to a different entry point.

### D. CORE GEOGRAPHY

```
CORE_GEOGRAPHY_OWNER       = config/asset/mi_geography.yaml ::
                             primary_basis_by_asset_class
                             → mi_agent/mi_geography.py :: default_primary_basis
                             → axis_fields → field_for_basis
                             (equity_release / lifetime_mortgage declare their
                             basis in config/asset/product_defaults_ERM.yaml;
                             per-portfolio override via the governed portfolio
                             registry `mi_geography.primary_basis`)
ASSET_CONFIG_BINDING_STAGE = mi_service._run_analysis, BEFORE the parse —
                             `_resolve_geography(client_id, portfolio_id, df)`
                             then `bind_geography(...)`, and the bound contract
                             is passed INTO ParsedQuestion.parse as `geography=`
RAW_TEXT_DEPENDENCY        = NO for the basis decision. `mi_geography.stated_basis
                             (question)` exists for a caller who names a basis
                             explicitly, but the governed default comes from
                             asset config, not text.
ITL3_MI_QUERY_PATH_EXISTS  = NO. ITL3 fields sit only in the `code` tier of
                             `tiers_for_basis`, never `reporting`, and
                             `field_for_basis` returns the most readable field
                             first. The ITL3 surface is `mi_agent_api/geo.py`
                             (`exposure_by_itl3`), the Funded → Geography
                             dashboard lens.
```

This is the **best-behaved** semantic in the estate: one owner, config-driven,
bound before interpretation, and already handed to the interpreter as an input.
It is the model the other three should follow.

### E. SPECIALIST CAPABILITIES

Registered recognisers (from `Recogniser(name=…)` declarations in
`mi_agent_api/`), evaluated in deterministic priority order by
`recogniser_registry.py :: RecogniserRegistry.candidates(request)`, each gated by
`resolve_capability_state(capability, context_id)`:

`scenario`, `cohort_conversion`, `forecast_extrapolation`, `funded_bridge`,
`cohort_progression`, `geo_exposure`, `concentration` (`conc_mod.WORKFLOW_ID`),
`period_movement`, `portfolio_summary`, `pipeline_summary`, `period_change`,
`temporal_compare`, `risk_limits`, `evolution`, plus the pipeline-stage route
(`prc_mod.WORKFLOW_ID`).

Every one is reached by a **recogniser reading the question**, and each delegates
to a deterministic service that owns its own arithmetic.

## Phase 3 — the old interpretation / routing layer, classified

| component | file :: function | classification | reason |
|---|---|---|---|
| dataset resolution | `workspace.py :: resolve_dataset(question)` | **REPLACE_WITH_GOVERNED_PLAN** | a semantic read of English that the plan already states |
| the parse | `parsed_question.py :: ParsedQuestion.parse` | **REPLACE_WITH_GOVERNED_PLAN** | this *is* the old control plane |
| LLM repair parser | `mi_agent/llm_query_parser.py :: parse_with_repair` | **OBSOLETE_IF_NEW_PLAN_IS_LIVE** | a second language reader with a repair loop — exactly what interpretation_v2 replaced |
| spec→plan lift | `query_plan_adapter.py :: compiled_spec_for`, `plan_from_spec` | **KEEP_BUT_CHANGE_INPUT** | the migration seam; under the target the direction inverts (plan is the source, not the lift target) |
| predicate materialisation | `mi_agent/population.py :: material_predicates` | **KEEP_DOWNSTREAM** | deterministic; consumes a spec, not English |
| routing | `chat_routing.py :: try_route` | **KEEP_BUT_CHANGE_INPUT** | capability gating, scope disclosure and envelope assembly are worth keeping; recognition from text is not |
| interpretation record | `chat_routing.py :: _qi_build(question, spec=…)` | **KEEP_BUT_CHANGE_INPUT** | should be built from the plan's own provenance |
| concept merge | `mi_agent_api/concept_merge_arm.py` | **OBSOLETE_IF_NEW_PLAN_IS_LIVE** | reconciles two readers of one sentence; with one authority there is nothing to merge |
| recogniser registry | `recogniser_registry.py :: RecogniserRegistry` | **KEEP_BUT_CHANGE_INPUT** | keep priority/gating, drive it from `plan.capability` instead of text |
| specialist recognisers | the 15 above | **REPLACE_WITH_GOVERNED_PLAN** (recognition only) | the plan names the capability; their deterministic services are KEEP_DOWNSTREAM |
| raw-text period recognition | `period_change/recognition.py` | **OBSOLETE_IF_NEW_PLAN_IS_LIVE** | the plan carries time semantics |
| period-pair resolution | `period_change/periods.py` | **KEEP_DOWNSTREAM** | deterministic resolver + governed gap policy |
| lens resolution | `portfolio_lens.resolve_lens_with_default(question, …)` | **REPLACE_WITH_GOVERNED_PLAN** | duplicated text reader |
| lens application | `portfolio_lens.apply_scope / apply_lens` | **KEEP_DOWNSTREAM** | language-free |
| geography binding | `mi_geography.*`, `_resolve_geography` | **KEEP_DOWNSTREAM** | already config-driven and pre-parse |
| calculation | `mi_query_executor.execute_mi_query` | **KEEP_DOWNSTREAM** | proven: 0 numerical errors over 93 plan-level cases |
| receipt | `execution_receipt.py` | **KEEP_BUT_CHANGE_INPUT** | must additionally record requested-vs-resolved scope |

```
CURRENT_RAW_TEXT_REREADS (after the single parse, inside try_route)
  chat_routing.py ~+37   ParsedQuestion.parse(question, semantics)   — fallback SECOND parse
  chat_routing.py ~+48   "read the question for its governed …"      — pre-recogniser semantic read
  chat_routing.py ~+94   mask_value_spans(question, values)
  chat_routing.py ~+142  _receipt.requested_dimension_terms(question, semantics, columns)
  chat_routing.py ~+161  _qi_build(question, spec=…)
  chat_routing.py ~+230  _resolve_lens(question, source_lens)
  chat_routing.py ~+278  _disclose_lens_scope(envelope, question, source_lens)
  mi_agent_workflow ~764 resolve_lens_with_default(question, …)
```

Of these, the dimension-term read and both lens reads are **semantic**, not
presentational. `mask_value_spans` and `_disclose_lens_scope` are
presentation/disclosure and could legitimately survive.

## Phase 4 — is the new plan contract sufficient?

`GovernedQueryPlan` (frozen, `mi_agent/interpretation_v2/plan.py`) carries:
`capability`, `operation`, `population` (base/lens/seasoning + scope
predicates), `outputs[]` (each with `measures` incl. statistic/weight/field,
`dimensions`, `filters`, `geography`), top-level `filters`, `geography`,
`period` (`PeriodBinding`: form, labels, grain, periods_back, contract,
resolved, owned_by_capability), `comparison_*`, `target`, and `provenance`
(`intent_claims` vs `compiler_bindings`).

| plan field | drives the existing owner today? | adapter exists? | smallest translation | binds or hands over? |
|---|---|---|---|---|
| `capability` / `operation` | YES | no | map to recogniser/route name | hands over |
| `outputs[].measures` (+ statistic, weight, `canonical_field`) | YES | partial (`query_plan_adapter`) | fill `MIQuerySpec.metric/aggregation/weight_field/measures` | hands over (field already bound by the compiler) |
| `outputs[].dimensions` | YES | partial | `spec.dimension/dimensions/x/y` | hands over |
| `filters` (+ `canonical_field`, comparator, value) | YES | partial | `spec.filters` | hands over |
| `geography` (basis/level/field) | YES | no | already the same contract the pre-parse binder produces | hands over |
| `population.lens` | **NO** | no | `lens_from_selection(plan.population.lens)` → `apply_scope` | hands over — the plan states the lens, the registry resolves its members |
| `period` (`PeriodBinding.contract`) | **NO** | no | `previous_governed_reporting_period` / `governed_reporting_period_pair` → `as_of_date` or the `temporal_compare` / `period_change` route | hands over — **the compiler has already named the governed contract** |
| dataset | **NO — absent from the plan** | n/a | see MISSING_BINDINGS | — |
| `comparison_*`, `target` | YES for routed capabilities | no | route inputs | hands over |
| `ambiguity` / refusal | YES — CLARIFY/REFUSE precede execution | n/a | return the governed refusal | hands over |
| `provenance.compiler_bindings` | YES | n/a | the receipt's "requested" half | hands over |

```
NEW_PLAN_CONTRACT_SUFFICIENT = PARTIAL

MISSING_BINDINGS
 1. DATASET. GovernedQueryPlan has NO dataset field. `CandidateIntent.population.base`
    (funded / forecast / pipeline) is the nearest governed semantic and is
    re-validated by the compiler, but it is a population base, not a dataset
    selector, and baseline v2 recorded `population.base` as an undefined
    contract semantic for forward-looking questions (NL2, NL9, Q23, Q24).
    This is the one genuine gap. It is an interpretation_v2 contract question,
    and interpretation_v2 is frozen — so the adapter must either take the
    dataset from the caller (as `datasetContext` already offers) or the contract
    must gain the field in a future, separately-signed-off sprint.
 2. PERIOD → EXECUTION ROUTE. The plan names a governed period CONTRACT, not an
    execution mode. Mapping contract → (`as_of_date` | temporal route) is a
    lookup, but which routes are admissible is a product decision not yet made.
 3. RECEIPT. No field carries "resolved" facts; the plan is entirely the
    "requested" half by design.
```

## Phase 5 — integration options

| criterion | A: plan → legacy ParsedQuestion | B: plan → thin orchestration adapter | C: plan → execute_query_plan owns scope | D: plan → `runtime_bridge` style |
|---|---|---|---|---|
| semantic duplication risk | **HIGH** — rebuilds a spec the plan already states, and `try_route` would re-read text | **LOW** | MEDIUM | MEDIUM |
| raw-text reread after plan | **YES** (all 8 sites survive) | NO | NO | partial |
| legacy code retained | most | calculation + lens application + geography + specialists + receipt | little | medium |
| new code required | small | **one adapter module** | large (scope resolution + snapshot ownership) | medium |
| preserves SnapshotStore authority | yes, unused | **yes** | **NO** — would need a second temporal owner | yes |
| preserves asset-config geography | yes | **yes** | yes | yes |
| preserves lens owner | resolution duplicated | **yes** (`apply_scope` kept) | would re-implement | yes |
| specialist compatibility | yes | **yes** — routes keep their services | poor — plan ops are only `{avg,count,max,median,min,sum,weighted_avg}` | yes |
| receipt / auditability | unchanged (no requested-vs-resolved) | **improves** — adapter knows both halves | improves | improves |
| regression / blast risk | MEDIUM | **LOW** behind a flag | **HIGH** | MEDIUM |
| reversibility | poor (legacy entangled) | **high** — one call site | low | medium |
| consistent with target architecture | **NO** — reinstates the old control plane | **YES** | no | partially |

Option C is ruled out by evidence, not preference: `execute_query_plan` has no
production caller, `MIQuerySpec` has no `dataset` field, `AnalyticalScope`'s
period is documented as caller-resolved, and the plan's operation vocabulary
cannot express ranking, distribution, summary, movement or concentration.

Option D is real — `mi_agent/interpreter/runtime_bridge.py ::
interpret_and_run_mi_query` already does question → interpret → `run_mi_query`
with a store. It is the only existing code that reaches the snapshot runtime. It
is worth keeping as the *precedent for the adapter's shape*, but it takes a
question, not a plan, so it is not the seam itself.

```
RECOMMENDED_INTEGRATION_OPTION = B
```

**WHY.** The trace shows the estate's semantics are already owned by good
deterministic modules — `mi_geography` (config-driven, pre-parse),
`portfolio_lens.apply_scope` (language-free), `period_change.periods` (governed
resolver with a gap policy), `execute_mi_query` (0 numerical errors across 93
plan-level cases), `execution_receipt`. What is duplicated and text-dependent is
only the *interpretation and routing* layer, which is precisely what the signed
interpretation_v2 replaces. Option B removes one layer and keeps every owner.
Option A would reinstate the layer being retired; Option C would build a second
temporal owner the architecture already has.

## Phase 6 — target call graph

```
POST /mi/query                                   app.py :: query            (unchanged)
  → ExecutionContext                             app.py :: _execution_context (unchanged)
  → governance / tenancy / portfolio auth         mi_service §1              (unchanged)
  → dataset + frame resolution                    mi_service._resolve_frame  (unchanged)
        dataset from the CALLER (datasetContext / MiQueryRequest), NOT from text
  → geography binding                             _resolve_geography → bind_geography (unchanged)
  → OpusInterpreter                               interpretation_v2.opus_interpreter (frozen)
  → CandidateIntent                               interpretation_v2.intent           (frozen)
  → canonical_intent + DeterministicCompiler      interpretation_v2.normalise/compiler (frozen)
  → GovernedQueryPlan | Clarify | Refuse          interpretation_v2.plan             (frozen)
  → PlanRuntimeAdapter                            **NEW, the only new module**
        ├─ plan.population.lens  → portfolio_lens.lens_from_selection → apply_scope
        ├─ plan.period.contract  → as_of_date | temporal route selection
        ├─ plan.geography        → assert equals the pre-bound contract, else refuse
        ├─ plan.capability       → recogniser_registry route, by NAME not text
        └─ plan.outputs/filters  → MIQuerySpec fields
  → existing deterministic owner                  execute_mi_query | specialist service
  → ExecutionReceipt                              execution_receipt + requested-vs-resolved
  → React payload                                 adapters / presenters     (unchanged)
```

### The one new component

**`PlanRuntimeAdapter`** (name indicative).

*Why required:* nothing today converts a `GovernedQueryPlan` into the inputs the
existing owners expect; `query_plan_adapter` goes the other way (spec → plan).

*Inputs:* `GovernedQueryPlan`, the resolved frame, the bound geography contract,
the caller's dataset/lens selection, `ExecutionContext`.

*Outputs:* either (`MIQuerySpec`, lens, scope, period instruction) for the
generic path, or a specialist route request, or a governed refusal.

*Allowed to decide:* which existing owner to call; which already-bound plan value
to place in which existing field; that an unsatisfiable plan is a refusal.

*Explicitly forbidden:* reading the question text; choosing a measure, filter,
dimension, statistic, capability, geography basis, lens or period; resolving a
date; filtering a frame; computing anything; defaulting a value the plan left
empty (that is a refusal, not a default).

```
NEW_CODE_REQUIRED       = one adapter module + one flagged call site in mi_service._run_analysis
LEGACY_CODE_BYPASSED    = workspace.resolve_dataset(question), ParsedQuestion.parse,
                          llm_query_parser.parse_with_repair, concept_merge_arm,
                          period_change.recognition, portfolio_lens.resolve_lens_with_default,
                          text-based recogniser matching
LEGACY_CODE_RETAINED    = mi_geography, portfolio_lens.apply_scope/apply_lens,
                          portfolio_registry, population.material_predicates,
                          period_change.periods, mi_query_executor, every specialist
                          service, execution_receipt, recogniser_registry gating,
                          adapters/presenters, the whole governance envelope
SEMANTIC_OWNER_DUPLICATION_AFTER_TARGET = NO (one interpreter, one compiler, one
                          owner per downstream semantic) — PROVIDED the legacy
                          path is behind a flag and not consulted concurrently
RAW_TEXT_SEMANTIC_REREAD_AFTER_PLAN     = NO by design; `mask_value_spans` and
                          `_disclose_lens_scope` may remain as presentation only
```

## Phase 7 — specialist capability matrix

| capability | production entry | recogniser | deterministic service | plan sufficient? | adapter needed? | raw-text remaining? | treatment |
|---|---|---|---|---|---|---|---|
| generic point-in-time | `mi_agent_workflow.run_mi_agent_query` | — (default path) | `mi_query_executor.execute_mi_query` | **YES** | YES (thin) | none | **FIRST SLICE** |
| funded bridge / movement | `chat_routing` route | `funded_bridge`, `period_movement`, `period_change` | `period_change/bridge.balance_bridge` (+`periods`, `calculations`) | YES — `capability`+`operation`+measure present | YES | none once `periods` gets the plan's contract | keep service, replace recogniser |
| concentration | `chat_routing` route | `conc_mod.WORKFLOW_ID` | `mi_agent_api/concentration_query.py` | PARTIAL — plan ops lack `concentration` | YES | UNKNOWN | keep service; compiler already governs `concentration_share`/`_exposure` |
| ranking | `chat_routing` route | `period_movement` rank path / `risk_limits` | `period_change/ranking.py` | PARTIAL — `rank` is a CandidateIntent operation but not a plan-executor op | YES | none | keep service, drive by `operation=rank` |
| distribution | `chat_routing` route | `period_change` | `period_change/distribution.py` | PARTIAL (as ranking) | YES | none | as ranking |
| portfolio summary | `chat_routing` route | `portfolio_summary`, `pipeline_summary` | the route's own service | YES — `operation=summary` | YES | UNKNOWN | keep service |
| borrowing base | `chat_routing` route | `borrowing_base_query` | `mi_agent/borrowing_base/calculator.calculate(df, facility)` | YES — capability + target | YES | none | keep service; facility comes from config, never text |
| forecast | `chat_routing` route | `forecast_extrapolation` (`CAP_ORIGINATION_FORECAST`) | forecast service | YES — `forecast_milestone` + `target` | YES | UNKNOWN | keep service |
| temporal compare | `chat_routing` route | `temporal_compare` | `mi_agent_api/temporal_compare.py` | PARTIAL — needs the period-pair contract | YES | period text today | keep service, feed `governed_reporting_period_pair` |
| geo exposure (ITL3) | `mi_agent_api/geo.py` | `geo_exposure` | `geo.py` | **OUT OF SCOPE** | no | n/a | leave alone — dashboard lens, not MI query |

**A single generic plan cannot dispatch everything.** The plan *identifies* every
capability (that is what `capability` + `operation` are for), but the executor's
operation vocabulary is `{avg, count, max, median, min, sum, weighted_avg}`.
Specialists must keep their own input contracts and services; the adapter
dispatches to them. No specialist arithmetic should be rewritten.

## Phase 8 — the quarantined branch, read-only

33 commits. **22 touch only evidence, tests or CI**; 11 touch product code.

```
TOTAL_MATERIAL_COMMITS = 11
KEEP                   = 4
KEEP_WITH_ADAPTATION   = 4
OBSOLETE               = 0 (whole commits) — but 3 carry parser-only parts that become obsolete
DEFER                  = 2
UNKNOWN                = 1
```

| sha | subject | product files | class | reason |
|---|---|---|---|---|
| `9abbc286` | the harmonised region, and the fifth chooser | `mi_geography.py`, `risk_limits.py` | **KEEP** | geography owner is retained wholesale in the target |
| `a4712b7e` | G6 — one geography field owner: a request measures ONE basis everywhere | `mi_geography.py`, `chat_routing.py`, `evolution.py`, `geo.py` | **KEEP** | exactly the target's geography principle; the `chat_routing` part needs re-pointing |
| `05dbe95f` | G1 — one period-pair owner | `period_change/{models,periods,recognition}.py`, `analytical_plan.py` | **KEEP_WITH_ADAPTATION** | `periods.py`/`models.py` are the retained resolver; `recognition.py` is raw-text period parsing that the plan makes obsolete |
| `2ef7fb3e` | a weighting names the WEIGHT; a parallel question is one list | `execution_receipt.py`, `llm_query_parser.py`, `statistic.py`, `chat_routing.py` | **KEEP_WITH_ADAPTATION** | receipt + statistic fixes are KEEP; the `llm_query_parser` half is obsolete |
| `8a1b4197` | G3 — the fail-closed guard reads the ledger, not the sentence | `mi_service.py` | **KEEP** | moves *away* from raw text — the target's own direction |
| `f6fa1f8e` | G3 — the claimant ASKS the owners; semantic claims ledger | `llm_query_parser.py`, `period_change/recognition.py`, `seasoning.py`, `semantic_claims.py` | **KEEP_WITH_ADAPTATION** | `semantic_claims` is a governance ledger worth keeping; its two text-reader clients are obsolete |
| `10ccb254` | capability ownership beats a generic measure collision | `borrowing_base/capacity.py`, `capability_ownership.py`, `llm_query_parser.py`, `borrowing_base_query.py` | **KEEP_WITH_ADAPTATION** | `capability_ownership` is directly useful to the adapter's dispatch; parser part obsolete |
| `8e386ce5` | G2 — one normaliser | `categorical_spans.py`, `llm_query_parser.py`, `borrowing_base_query.py`, `concentration_query.py` | **KEEP** | value normalisation is needed wherever filter values are compared, plan or not |
| `101ac7aa` | GEO-ITL3-RECOVERY | `mi_agent_api/geo.py` | **DEFER** | the ITL3 dashboard lens, outside MI-query scope by the product contract |
| `2f25afdc` | Phase 7 — vertical non-regression on the routed path | `chat_routing.py` | **DEFER** | a fix to the routed path the target re-points; re-assess after the adapter exists |
| `ae916afc` | G7-lite — an unasked count is not a claim; a model fill carries its own operation | `llm_query_parser.py`, `mi_query_spec.py`, `query_plan_adapter.py`, `concept_merge_arm.py` | **UNKNOWN** | touches the lift adapter and concept-merge, both of which change role in the target; needs a per-hunk read, not a commit-level call |

No commit was merged or cherry-picked. The earlier finding stands: **none of
these addressed the plan-scope boundary**, because that boundary is not on the
production path.

## Phase 9 — receipt ownership

```
Target invariant:  request claimed X → runtime executed Y → receipt proves X → Y
                   X unhonourable → governed refusal
```

| semantic | knows REQUESTED | knows RESOLVED | where they meet |
|---|---|---|---|
| dataset | the caller (`MiQueryRequest.dataset_context`) — and, in the target, *only* the caller | `_resolve_frame` | adapter records both; receipt carries `requested_dataset` + `resolved_dataset` |
| portfolio lens | `plan.population.lens` | `portfolio_registry.resolve` → the id list | `apply_scope` already sets `spec.portfolio_lens`; receipt adds the resolved id list |
| requested period | `plan.period` (form, labels, contract) | `SnapshotSelector` / `period_change.periods` / the frame's own reporting date | **the adapter is the only place both are known** — this is why the receipt work belongs after the seam, not before |
| resolved snapshot / reporting date | — | frame or snapshot header | receipt `resolved_period` + `snapshot_id` |
| filters | `plan.filters` (concept + canonical_field + comparator + value) | `applied_predicates`, `filtered_row_count` | already both present; receipt should name the plan's concept beside the executed field |
| geography | `plan.geography` (basis/level) | `field_for_basis` result | already resolved pre-parse; receipt should state basis **and** field |
| capability | `plan.capability` / `operation` | the route actually taken | adapter records the dispatch decision |
| measure / statistic / dimension | `plan.outputs[*]` | `aggregation`, `measures_executed`, `group_field_keys` | already both present and correct |

Ownership conclusion: **the adapter is the meeting point.** That is the decisive
argument for doing the seam before the receipt — Phase B's Option 2 would have
put requested-vs-resolved into a receipt whose "requested" half has no
authoritative source on the production path today.

## Phase 10 — the first implementation slice

**Capability class: a single-period, single-output, funded point-in-time query
with optional filters and at most two dimensions — the generic path at stage 12,
no specialist route.**

Why this one, from the evidence:

* It is the only capability where `NEW_PLAN_CONTRACT_SUFFICIENT = YES` with no
  missing binding other than dataset (which the caller already supplies).
* Its deterministic owner is the most thoroughly proven thing in the estate:
  0 numerical errors across 93 plan-level cases, 22/22 independent arithmetic
  oracle, geography 5/5.
* It needs no snapshot store, so it does not touch the temporal question that
  stopped Phase B.
* `try_route` returns `None` for it today and defers to the point-in-time path,
  so the slice can be taken **without touching any recogniser**.
* It is reversible at one call site.

Shape: in `mi_service._run_analysis`, behind an explicit flag, for plans the
adapter can satisfy completely, build the spec from the plan instead of from
`ParsedQuestion.parse`, run the same `execute_mi_query`, and compare. Shadow
mode first — run both, log divergence, serve the legacy answer — then switch.

```
ESTIMATED_BLAST_RADIUS = LOW (behind a flag, shadow-first, one call site)

KEY_BLAST_RISKS
  1. dataset — the plan has no dataset field; if the adapter ever infers one it
     creates the duplication this whole exercise removes. It must take the
     caller's.
  2. the fallback parse at chat_routing ~+37 would re-parse if `parsed` is None;
     the adapter must supply a parsed-equivalent or that path must be closed.
  3. `parsed.merge_filters(req.filters)` — caller filters must merge into the
     plan's filters with the same precedence, or populations will differ silently.
  4. `available_values` / `available_columns` are handed to the parser today;
     interpretation_v2 reaches the registry instead, so a book-specific value the
     parser could see may not resolve. Needs measuring before the switch.
  5. CLARIFY / REFUSE have no production representation; the envelope needs a
     governed shape for "the question was not answerable as asked".

REQUIRED_GATES_BEFORE_IMPLEMENTATION
  G1 shadow-mode divergence run over the existing 843-question corpus, with
     complete output capture — plan answer vs legacy answer, per question
  G2 the 22-case arithmetic oracle stays 22/22
  G3 the 65-case v2 bank and the relevant scope-bank cases unchanged
  G4 a decision on dataset authority: caller-supplied (recommended) or a
     contract change to interpretation_v2 in a separate signed sprint
  G5 a decision on the CLARIFY/REFUSE envelope shape
  G6 mi_agent/tests holds at 1404 passed / 16 known failing groups
```

## Verdict

```
PRODUCTION_PATH_ARCHITECTURE_SETTLED = YES

RECOMMENDED_NEXT_STEP =
    implement only the first bounded integration slice — generic single-period
    funded point-in-time — behind an explicit shadow flag, with the old
    production path retained as the control and serving every answer until G1-G6
    pass.
```

The principle the brief set out is achievable as stated: **replace the
interpretation/control plane, not the deterministic runtime.** The trace shows
the runtime is in good order and the control plane is where the duplication
lives — one dataset reader of English, two lens readers of English, a fallback
second parse, a concept-merge arm that exists only because two readers disagree,
and raw-text period recognition beside a governed period resolver that does not
need it.

Two open decisions are recorded rather than assumed: **dataset authority** (the
plan has no field for it, and interpretation_v2 is frozen) and the
**CLARIFY/REFUSE envelope**. Neither blocks the first slice; both must be settled
before the second.
