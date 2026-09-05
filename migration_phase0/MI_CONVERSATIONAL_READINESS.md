# Conversational analytical composition — architectural readiness review

**Verdict: CONDITIONAL GO.**

The deterministic estate can carry a conversational layer above it without any
change to how a number is calculated. Three bounded prerequisites stand between
here and a safe V1, and one of them is a silent-answer defect that exists
*today*, in the same-turn multi-output path, and would be inherited by every
follow-up turn built on top of it.

Nothing in production changed to produce this document. The evidence is
reproducible:

```
python migration_phase0/conversational_readiness_probe.py
python -m pytest mi_agent/tests/test_conversational_composition_characterisation.py
```

The probe runs the real `run_mi_agent_query` and `execute_mi_query` over the
estate's own deterministic funded tape (`mi_agent.mi_query_harness.build_fixture`,
with one NUTS3 label relabelled `London` so the worked example is expressible)
and the governed pipeline frame prepared by `pipeline_prep.prepare_pipeline_mi_dataset`
from `tests/fixtures/pipeline_transition_2w`. Its JSON is
`migration_phase0/MI_CONVERSATIONAL_READINESS.json`.

---

## A. Current architecture

### A.1 The point-in-time funded lifecycle

Traced with *“What is the funded balance for joint borrowers in the London
region with LTV above 40%?”*

| # | Stage | Owning module → function | Input contract | Output contract |
|---|---|---|---|---|
| 1 | NL entry | `mi_agent_api/app.py:1957` `@app.post("/mi/query")` → `query()` | `QueryRequest` (`app.py:413`) | `GovernedResult` → React payload |
| 2 | Channel adapter | `app.py:1973` builds `mi_service.MiQueryRequest` | HTTP body + `ExecutionContext` | `MiQueryRequest` (`mi_service.py:83`) |
| 3 | Capability | `mi_agent_api/mi_service.py:441` `execute_governed_mi_query` | `MiQueryRequest`, `ExecutionContext` | `GovernedResult[dict]` |
| 4 | Dataset resolution | `mi_agent_api/workspace.py:215` `resolve_dataset(question)` | the **question only** | `"funded" \| "pipeline" \| …` |
| 4b | Dataset re-read | `mi_service.py:~1512` under span ownership | question masked by book values | same owner, one evidence-richer answer |
| 5 | Governance | `context.require_scope`; `trakt_core/tenancy.authorise_portfolio_access`; `trakt_core/policy.evaluate_source_approval` | `ExecutionContext`, portfolio selector | `AuthorisedPortfolio`, `PolicyState`, `SnapshotRef` |
| 6 | Frame resolution | `mi_service.py:853` `_resolve_frame` → `datasets.py:743` `_resolve_query_frame` | view + portfolio id | `pd.DataFrame` \| error |
| 7 | Semantic/concept extraction | `mi_agent/parsed_question.py:93` `ParsedQuestion.parse` → `llm_query_parser.py:5688` `parse_with_repair` → `:3690` `_deterministic_parse` | question, registry, `available_columns`, `available_values` | `ParsedQuestion(spec, meta)` |
| 8 | Canonical field/value binding | `llm_query_parser._parse_filters`, `_parse_categorical_filter`, `_borrower_structure_filter:3048`, `mi_agent/categorical_spans.py`, `region_resolution.py` | masked question + book value catalogue | entries in `spec.filters` |
| 9 | **QuerySpec** | `mi_agent/mi_query_spec.py:319` `MIQuerySpec` | — | the governed request contract |
| 10 | Route selection | `mi_agent_api/chat_routing.py:4070` `try_route` + `recogniser_registry.py` | `question`, `parsed`, resolvers | routed envelope, or `None` → point-in-time |
| 10b | Contract projection | `chat_routing.py:4193` → `question_interpretation/projection.py:88` `from_parts` | **spec + facets** (never the question) | `QuestionInterpretation` (`schema.py:859`) |
| 10c | Plan (routed only) | `mi_agent_api/analytical_plan.py:131` `build_plan` | `QuestionInterpretation` | `Plan(Step…)` (`analytical_plan.py:98/113`) |
| 11 | Governed population | `mi_agent/population.py:181` `material_predicates` → `:200` `apply_population` | `spec.filters` + semantics | narrowed frame + `PopulationEvidence` |
| 12 | Deterministic execution | `mi_agent/mi_agent_workflow.py:479` `run_mi_agent_query` → `mi_query_executor.py:1570` `execute_mi_query` | `MIQuerySpec`, frame, semantics | `MIQueryResult` (`:124`) |
| 12a | Filters | `mi_query_executor.py:727` `_apply_filters` → `:538` `governed_predicate_mask` | `spec.filters` | narrowed frame + `applied_filter_fields` |
| 12b | Dimensions | `mi_query_contract.py:61` `all_group_dims`; grouped executors | `spec.dimension(s)`/`hierarchy` | `group_field_keys`, `group_columns` |
| 12c | Measures | `mi_query_executor.py:1008` `resolve_measures`; dispatch `:1646` | `spec.measures` \| `spec.metric` | `measures_requested/executed/unavailable` |
| 12d | Aggregation | registry `default_aggregation` / `allowed_aggregations` | measure key + requested agg | `ResolvedMeasure(column, label)` |
| 13 | Reconciliation | `mi_query_executor.py:791` `_build_reconciliation` | before/after frames | `metadata.reconciliation` |
| 14 | Facets & semantic guard | `execution_receipt.py:1246` `detect_requested_facets` → `:2408` `reconcile_facets` → `:2795` `build_receipt` → `:2871` `assess` | question, spec, result, semantics | `ExecutionReceipt`, verdict `ok\|clarify\|refuse` |
| 15 | Post-guards | `mi_service._fail_closed_analytical`, `_guard_temporal_honouring`, `_guard_unresolved_scope`, `_guard_unknown_category` | rendered result | same, or a controlled refusal |
| 16 | Envelope | `mi_agent_api/adapters.py:776` `adapt_workflow_result` | workflow dict | `{ok, answer, spec, executionSummary, artifacts[], reconciliation, queryTrace, semanticGuard, metadata}` |
| 17 | Renderer | `mi_agent/mi_chart_factory.py:749` `create_mi_chart(result, semantics)` | **`MIQueryResult` only** | `MIChartResult` |
| 18 | Presentation | `mi_agent_api/presenters.py:20` `to_react_payload` | `GovernedResult` | HTTP body |

### A.2 The pipeline lifecycle

**Identical from stage 3 onward.** The only divergence is stages 4 and 6:
`resolve_dataset` returns `"pipeline"`, and `_resolve_query_frame("pipeline", …)`
serves the frame `pipeline_prep.prepare_pipeline_mi_dataset` produced. The
executor, the receipt, the facet reconciler and the renderer are
dataset-parametric — `dataset=view` is threaded to `execute_mi_query` and lands
on `metadata.dataset` and `ExecutionReceipt.dataset`.

Measured on the pipeline frame:

```
For pipeline loans at the OFFER stage, give me case count, balance and WA LTV.
  metadata.dataset            = pipeline
  measures_executed           = [loan_count, current_outstanding_balance, current_loan_to_value]
  applied_filter_fields       = [pipeline_stage]
  receipt                     = Loans · Balance · Weighted-average Current LTV · Pipeline Stage = OFFER · 2 loans
```

There is no funded-only code on this path. That matters for the V1 perimeter:
funded **or** pipeline costs nothing extra; funded **↔** pipeline transition is
where the work is (§D, dialogue 7).

### A.3 The one authoritative machine-readable “population actually calculated over”

**Yes — it exists, and it is `MIQuerySpec.filters`, proven by
`MIQueryResult.metadata["applied_filter_fields"]`.**

Not inferred. The two objects, verbatim from the probe:

```json
spec.filters = {
  "geographic_region_obligor": "London",
  "current_loan_to_value": {"op": "gt", "value": 40.0},
  "borrower_type": "Joint"
}
query_result.metadata.applied_filter_fields =
  ["geographic_region_obligor", "current_loan_to_value", "borrower_type"]
query_result.metadata.reconciliation = {
  "dataset": "funded", "total_records": 400, "records_after_filters": 21,
  "filters_applied": true, "filters": { …the same dict… },
  "balance_after_filters": …, "coverage_by_balance_pct": …
}
```

Three properties make this the right object rather than merely an available one:

1. **It is the intent AND the evidence, separately.** `spec.filters` is what was
   requested; `applied_filter_fields` is what `_apply_filters` actually ran,
   collected only after `_require_column` confirmed the book carries it
   (`mi_query_executor.py:730-738`). An inheriting layer can therefore inherit
   *what executed*, never *what was parsed*.
2. **It has one normaliser.** `population.predicate_of` (`population.py:158`) is
   the single reader of the three shapes `spec.filters` uses, and both executors
   go through it.
3. **It is already published.** `adapt_workflow_result` returns the whole `spec`
   on the envelope. The client already holds the governed population.

Two things the receipt is **not**:

- `ExecutionReceipt.filters` is **prose** — `["London", "Current LTV > 40",
  "Borrower Type = Joint"]`. It is for a reader, it is not re-parseable, and a
  conversational layer that read it would be reconstructing a population from
  natural language, which is the NO-GO condition. Do not build on it.
- `ExecutionReceipt.to_dict()` omits `dataset`. The dataset is recoverable from
  `metadata.reconciliation.dataset`, which the envelope carries — but the
  receipt’s own dict does not carry it, and a scope object must not read the
  receipt alone.

---

## B. Scope reconstruction result

### **YES.**

The test is not an inspection, it is a replay. Rebuild the population from
`spec` + `query_result.metadata` alone, assemble a fresh `MIQuerySpec` carrying
**no natural language**, execute it on the unchanged `execute_mi_query`, and
compare the rows.

```
question : What is the funded balance for joint borrowers in the London region with LTV above 40%?
executed : Total Balance · London · Current LTV > 40 · Borrower Type = Joint · 21 loans
rebuilt  : {"geographic_region_obligor":"London",
            "current_loan_to_value":{"op":"gt","value":40.0},
            "borrower_type":"Joint"}

INHERIT  WA LTV over the inherited population        rows=21    ✓ same population, new measure
ADD      + LTV > 80                                  rows=0     ✓ narrower
MODIFY   borrower_type Joint -> Single               rows=21    ✓ {…,"borrower_type":"Single"}
RESET    whole funded book                           rows=400   ✓ filters {}
GROUP    inherited population by age bucket          rows=21    ✓ 3 groups, same 21 rows

INHERIT replay lands on the original population: True
```

All four follow-up operations plus grouping are **spec edits followed by the
existing executor**. No new economics, no duplicated calculation, no new
primitive.

### Slot-by-slot sufficiency

| Slot | Owner today | Populated | Note |
|---|---|---|---|
| dataset | `query_result.metadata["dataset"]` | ✅ | also `reconciliation.dataset`; **absent from `receipt.to_dict()`** |
| population filters | `spec.filters` | ✅ | canonical, field → condition |
| grouping | `spec.dimensions` / `spec.dimension` / `metadata.group_field_keys` | ✅ | executed form is on the metadata |
| requested measure | `spec.metric` + `spec.measures` | ✅ | `metadata.measures_executed` is the executed form |
| aggregation | `spec.aggregation`, per-measure on `measures_executed` | ✅ | governed by the registry, not the sentence |
| weight field | `spec.weight_field` / `ResolvedMeasure.weight_field` | ⚠️ | empty on a `sum`; resolved at execution for `weighted_avg` |
| applied evidence | `metadata.applied_filter_fields`, `metadata.populationApplied` | ✅ | two sources, one reader (`population_applied`, `execution_receipt.py:3284`) |
| reporting period | `spec.reporting_date` / `receipt.period` | ❌ **PARTIAL** | both `None` on a point-in-time answer; the period lives on the *snapshot*, in `GovernedResult.snapshot` / `sourceNotes`, not on the spec |
| portfolio / lens | `spec.portfolio_lens`, `MiQueryRequest.source_portfolio_lens`, `payload.portfolioScope` | ❌ **PARTIAL** | request-supplied; **not** echoed onto `spec` when the caller supplies it as a request field |
| comparison basis | `spec.comparison_basis` | ✅ | `None` on point-in-time, set by the compare routes |

**Verdict: COMPLETE for the analytical population; PARTIAL for the two governance
axes.** The missing fields are `reporting_period` and `portfolio_scope`, and
both are *known to the request and the envelope* — they are simply not on the
spec. They are not missing information; they are information that lives in two
places with no single carrier.

### Smallest contract extension required

Not a schema change. A **derivation function with one owner**:

```
ConversationScope.from_result(spec, query_result.metadata, governed_result) -> ConversationScope
```

reading `dataset` from `metadata.reconciliation.dataset`, the population from
`spec.filters` **gated by** `metadata.applied_filter_fields`, the period from
`GovernedResult.snapshot`, and the lens from `payload.portfolioScope`. It adds
no vocabulary; every field is an existing governed key.

---

## C. Multi-output result

Machine-readable evidence only. `metadata.measures_executed` must name every
requested output, and `metadata.applied_filter_fields` must equal the shared
population — prose mentioning several numbers is not counted.

```
total multi questions ................... 7
atoms all green ......................... 7
composition verified .................... 3
composition failed despite green atoms .. 4
blocked by a broken atom ................ 0
answered ok but not as asked (SILENT) ... 1
```

| Shape | Verdict | Outputs | Population | Note |
|---|---|---|---|---|
| A · same population, 2 measures (`joint borrowers`) | COMPOSITION_FAILED | 2/2 ✅ | ✗ dropped | fails closed, `guard=clarify` |
| B · same population, 3 measures (`joint borrowers`) | COMPOSITION_FAILED | 3/3 ✅ | ✗ dropped | fails closed |
| B2 · same population, 3 measures (`London region`) | **COMPOSITION_VERIFIED** | 3/3 ✅ | ✅ | |
| C · shared population + clause-specific filter | COMPOSITION_FAILED | **2/3** | ✗ over-narrowed | **SILENT** |
| D · grouped multi-measure | **COMPOSITION_VERIFIED** | 3/3 ✅ | ✅ | 2 groups × 3 measures |
| E · pipeline multi-measure (`OFFER stage`) | **COMPOSITION_VERIFIED** | 3/3 ✅ | ✅ | same executor, `dataset=pipeline` |
| E2 · pipeline joint-borrower multi-measure | COMPOSITION_FAILED | 2/3 | ✗ dropped | fails closed |

### What already works

`MIQuerySpec.measures: List[{field, aggregation, weight_field}]` is shipped
(`mi_query_spec.py:359`, normalised by `normalise_measures:184`), dispatched
first in the executor (`mi_query_executor.py:1646`), resolved against the
registry by `resolve_measures:1008`, and executed whole — flat by
`_execute_measure_set`, grouped by `_execute_grouped_measure_set`. It publishes
`measures_requested` / `measures_executed` / `measures_unavailable`, and
`_multi_measure_answer` (`mi_agent_workflow.py:388`) appends
`"Not available: …"` by name rather than answering 2-of-3 silently.

**Requirement 4 of the stop conditions — “a reliable way to prove that every
requested output executed” — is already met, for measures.**

### Defect 1 — the measure-set path never asks the borrower-population owner

`_measure_set_recognizer` (`llm_query_parser.py:3524`) resolves its population
with `_parse_filters` and `_parse_categorical_filter`. Neither owns the
joint/sole vocabulary. `_borrower_structure_filter` (`:3048`) does, and it has
three call sites — `:3260`, `:3840`, `:3896` — none of which is this one, and
`_measure_set_recognizer` runs *before* all three (`:3798`). Geography and
numeric thresholds survive the same path; the categorical borrower population
does not.

```
"For joint borrowers, give me loan count and funded balance."
  measures_executed = [loan_count, current_outstanding_balance]   ✓
  spec.filters      = {}                                          ✗ population lost
  ok                = False, guard = clarify                      ✓ fails closed
```

This is the estate’s recurring shape — *one concept with two owners* — and it is
the same family the Phase 0 atomic characterisation names. **It fails closed.**
It is a three-line widening at one call site. It is a finding, not a change made
here.

### Defect 2 — a clause-scoped filter is promoted to the shared population (SILENT)

`MIQuerySpec` carries **one** filter set. A predicate that belongs to only the
third clause has nowhere to live except the population all three clauses share.

```
atoms:    "How many loans are in the London region?"                     69 loans
          "What is the balance in the London region?"                    69 loans
          "What is the balance in the London region with LTV above 40%?"  39 loans

combined: "For loans in the London region, what is the loan count, the balance,
           and how much of that balance has LTV above 40%?"
  ok                    = True
  measures_executed     = [loan_count, current_outstanding_balance]   ← 2 of 3
  applied_filter_fields = [geographic_region_obligor, current_loan_to_value]
  population            = 39                                          ← the NARROW cohort
  receipt               = "Loans · Balance · London · Current LTV > 40 · 39 loans"
```

Count and balance are reported over 39 loans when the reader asked for 69, and
the third figure was never produced. The receipt names the extra filter but
cannot say it was asked for only once, because nothing in the contract records
clause scope. The facet reconciler passes because every facet raised — geography,
threshold, measures — *was* applied.

**This is the only silent shape in the review, and it is the reason the verdict
is CONDITIONAL.** It is a same-turn defect that exists now; a conversational
layer that inherits from such a turn would inherit a population that was never
asked for.

### Atomic confounders, recorded so they are not read as compositional

A bare place name beside another predicate is dropped:

```
"What is the balance in London?"                                  geo bound ✓
"What is the funded balance for joint borrowers in the London region?"  ✓
"What is the funded balance for joint borrowers in London?"       geo DROPPED, ok=False
"What is the funded balance for joint borrowers in Scotland?"     geo DROPPED, ok=False
```

Every drop **fails closed**. This belongs to the atomic remediation sprint, not
this one; it is listed because a multi-turn row that inherits nothing because
turn 1 never bound the geography is an atomic finding wearing a conversational
costume, and the review’s own worked examples had to route around it.

---

## D. Multi-turn gap analysis

Each dialogue’s turns were run **independently**, which is exactly what the
stateless API does today.

| # | Dialogue | Q1 pop | Q2 pop | Today |
|---|---|---|---|---|
| 1 | INHERIT — “what is their WA LTV?” | 33 | 400 | **SILENTLY BROADENED** |
| 2 | ADD — “how much of that has LTV above 80%?” | 33 | — | REFUSED (honest) |
| 3 | MODIFY — “what about single borrowers?” | 33 | — | REFUSED (honest) |
| 4 | NUMERIC REF — “of the £38m, what is the WA LTV?” | 33 | 400 | **SILENTLY BROADENED** — but the £38m never became a predicate |
| 5 | AMBIGUOUS — two cohorts, then “their WA LTV?” | 400 | 400 | ANSWERED WITHOUT CLARIFYING (no cohort silently picked) |
| 6 | RESET — “now show the whole funded book” | 33 | 400 | CORRECT (statelessness *is* reset) |
| 7 | DATASET — “what about the pipeline?” | 33 | — | REFUSED (honest) |
| 8 | FAILED PRIOR TURN — “what about their balance?” | 400 | 400 | CORRECT (no state to create) |
| 9 | PRESENTATION — “show that by age bucket” | 33 | 400 | **SILENTLY BROADENED** |

**Silently broadened: 3 of 9. Refused honestly: 4 of 9. Correct by
statelessness: 2 of 9.**

The failure mode today is singular and uniform: *right economics, wrong
population, disclosed only by the receipt saying “entire funded portfolio”.*
Nothing computes the wrong number. Nothing invents a filter. Nothing picks a
cohort silently.

| Operation | Status | Evidence |
|---|---|---|
| **INHERIT** | **reusable primitive exists** | replay lands on the original 21 rows; `spec.filters` + `applied_filter_fields` are sufficient |
| **ADD** | **reusable primitive exists, one invariant missing** | `spec.filters` is field → **one** condition, so a second bound on the same field must be *folded*, not appended. `between` is governed and executes (`mi_query_executor.py:484`), so `>40 ∧ <80 → between[40,80]` is expressible; `>75 ∧ <70` must refuse |
| **MODIFY** | **reusable primitive exists** | replace the dict entry; the receipt then names the new value |
| **RESET** | **reusable primitive exists** | `filters = {}`; already the default behaviour |
| **result-reference resolution** (“of the £38m”) | **missing contract** — but the dangerous half is already safe | see below |
| **ambiguity** | **missing capability** | nothing today detects that “their” has two referents; `assess` clarifies on an unapplied facet, not on an unresolved pronoun |

### Result-reference resolution is already half-solved, and it is the safe half

Governance failure mode F — a prior result value read as a new numeric filter —
is **structurally prevented today**:

```
"Of the £38m, what is the weighted average LTV?"        filters = {}   ✓
"Of the 38 million, what is the weighted average LTV?"  filters = {}   ✓
"What is the weighted average LTV of the £38m?"         filters = {}   ✓
"Of the 43 loans, what is the weighted average LTV?"    filters = {}   ✓
"Of the £38m, how many loans are there?"                filters = {}   ✓
control:
"What is the balance for loans above £250,000?"  filters = {current_outstanding_balance: gt 250000}  ✓
```

A bare number with no governed comparator phrase beside it produces no
predicate (`_FILTER_COMPARATORS` at `llm_query_parser.py:2029`, `_POSTFIX_COMPARATORS` at `:3087`). So the layer must *add* “£38m means the previous
population”; it does not have to *prevent* “£38m means a filter”. That is the
cheap direction.

### Where the four operations should be applied — ONE owner

**After parsing, before routing: a single `ScopeResolver` that takes
`(ConversationScope | None, ParsedQuestion)` and returns a `ParsedQuestion`
whose `spec` carries the merged population, called once in
`mi_service._run_analysis` immediately after `ParsedQuestion.parse` and
immediately before `parsed.merge_filters(req.filters)`
(`mi_service.py:1543` → `:1600`).**

Why that seam and no other:

- **It is already the single-owner seam.** `ParsedQuestion` exists precisely
  because the question used to be parsed twice and routing could act on a
  different spec from the one executed (`parsed_question.py` docstring). The
  merge site is the one place `ParsedQuestion.merge_filters` already runs — the
  drill-through filter channel — and that channel already proves the pattern:
  a caller-supplied population is merged onto the spec once, and every reader
  downstream sees the merged object.
- **Before parsing is wrong.** Merging into the *text* would put the population
  back into natural language, which is stop condition 2.
- **During QuerySpec composition (inside the parser) is wrong.** The parser has
  no conversation and must not acquire one; it would become a second owner of
  the population and the estate has been bitten by that exact shape four times.
- **Inside the routes is wrong.** Thirteen recognisers merging state
  independently is thirteen chances to disagree about what “that” meant. The
  P1L comment at `mi_service.py:1573-1578` says this in the estate’s own words about
  the population frame, and the same argument applies unchanged.
- **After execution is too late** — the population decides the numbers.

Because the resolver runs *before* `try_route`, routed capabilities inherit the
same merged spec, and the `QuestionInterpretation` built at
`chat_routing.py:4193` is projected from it — so the routed contract and the
point-in-time spec still cannot disagree.

**One consequence to enforce as an invariant: nothing else may merge
conversational state.** Not `chat_routing`, not a recogniser, not `adapters`,
not the Teams bot.

---

## E. Presentation readiness

**Yes — the same governed result can drive text, table and chart without
reinterpreting any economics.**

1. **Is renderer selection already independent of calculation?** Yes.
   `create_mi_chart(result, semantics)` (`mi_chart_factory.py:749`) takes a
   `MIQueryResult` and the registry — no question, no frame, no filters, no
   portfolio. It reads `result.spec.chart_type`, `result.data` and
   `result.metadata`. There is no path by which changing a renderer changes a
   number.
2. **Do result payloads carry enough dimension/measure metadata?** Yes:
   `metadata.group_field_keys`, `group_columns`, `measures_executed`
   (`{field, canonical_field, aggregation, weight_field, column, label}`),
   `percent_scale_detected`, plus `queryTrace.resultColumns` and `chartAxes` on
   the envelope.
3. **Can charts be generated without re-running the economics?** Yes — and the
   envelope already emits *several typed artifacts from one result*:
   `adapters.py:807-935` builds a `kpi`, a `table` and a `chart` artifact from
   the same `MIQueryResult` into one `artifacts[]` list.
4. **Do multi-result comparisons need a new presentation contract?** Yes, and
   only there. `adapt_workflow_result` takes exactly one workflow carrying
   exactly one `MIQueryResult`. Two governed populations side by side
   (“compare joint and single in a bar chart”) is the one presentation shape
   with no home. `artifacts[]` is already a list, so the extension is an
   envelope that carries N results, not a new renderer.
5. **Can chart/table requests inherit ConversationScope safely?** Yes — a
   presentation follow-up is a scope edit of `intent` / `chart_type` /
   `output_format` / `dimensions` with `filters` untouched, then the same
   executor. The probe’s `GROUP` replay does exactly this: same 21 rows, three
   age-bucket groups.

**One defect to fix in Stage 3:** a presentation verb reaches the *measure*
resolver today.

```
"For loans in the London region, give me the balance and put it in a table."
  ok = False — "I understood that you asked for put it, but … it is not a
                governed measure in this dataset"
```

Fails closed, and correctly refuses rather than guessing — but it shows the
presentation decision is currently taken *inside* the parser. Stage 3 must lift
it above the parser rather than teach the parser more verbs.

---

## F. Required new components

Three. Deliberately fewer owners than the brief sketched: no `QueryPlan` (the
executor already dispatches a measure set, and `analytical_plan.Plan` already
exists for the routed side), and no separate `ResultReference` object (a
reference resolves *to* a `ConversationScope`, so it is a method, not a type).

### F.1 `ConversationScope`

| | |
|---|---|
| **Responsibility** | The governed population of the last successful turn, in canonical form, plus what was produced from it. **Derived from existing contracts; owns no vocabulary of its own.** |
| **Owner** | new `mi_agent/conversation_scope.py` — beside `population.py`, in the package that owns `MIQuerySpec` and `Predicate` |
| **Inputs** | `MIQuerySpec`, `MIQueryResult.metadata`, `GovernedResult` (snapshot, portfolioScope, tenant) |
| **Outputs** | frozen dataclass + `to_dict()` / `from_dict()`; every field a canonical registry key |
| **Invariants** | ① a filter enters only if its field is in `applied_filter_fields` or `metadata.populationApplied`; ② never constructed from a turn where `ok is False` or `semanticGuard.verdict != "ok"`; ③ carries `tenant_id` + `snapshot_id` and is rejected on mismatch; ④ carries `originating_turn`, `spec`, `receipt` as provenance; ⑤ no field whose value came from prose |
| **Prod LOC** | 180–260 |
| **Test LOC** | 300–400 |
| **Depends on** | `mi_query_spec`, `population.Predicate`, `trakt_core.context` |

Recommended shape — note every value is an existing governed key:

```python
@dataclass(frozen=True)
class ConversationScope:
    dataset: str                              # metadata.reconciliation.dataset
    portfolio_scope: Optional[Dict[str, Any]] # payload.portfolioScope / portfolioCoverage
    reporting_period: Optional[str]           # GovernedResult.snapshot
    population_filters: Dict[str, Any]        # spec.filters, GATED by applied_filter_fields
    grouping: Tuple[str, ...]                 # metadata.group_field_keys
    previous_outputs: Tuple[ScopeOutput, ...] # from metadata.measures_executed + the row
    provenance: ScopeProvenance               # turn index, spec, receipt, snapshot_id, tenant_id
```

`ScopeOutput` is `{measure, aggregation, value, unit, column, result_id}` — every
field already on `measures_executed` plus the executed value.

### F.2 `ScopeResolver`

| | |
|---|---|
| **Responsibility** | THE single owner of INHERIT / ADD / MODIFY / RESET and of referent resolution. Produces the merged `MIQuerySpec` for this turn. |
| **Owner** | new `mi_agent/scope_resolution.py`, called once from `mi_service._run_analysis` |
| **Inputs** | `Optional[ConversationScope]`, `ParsedQuestion`, semantics, book value catalogue |
| **Outputs** | `(ParsedQuestion, ScopeDecision)` — the decision records op, inherited fields, dropped fields, conflicts, and any clarification demand |
| **Invariants** | ① never widens: a turn may only inherit or narrow unless the sentence explicitly resets; ② two bounds on one field are **folded** into one governed condition (`between`) or **refused**, never appended and never silently replaced; ③ a dataset change **drops** the whole inherited population unless the registry declares a `funded_correlation` for the field; ④ an unresolvable referent returns a clarification, never a choice; ⑤ every inherited predicate is disclosed on the receipt as inherited; ⑥ scope is discarded when tenant, portfolio scope or snapshot differs |
| **Prod LOC** | 350–500 |
| **Test LOC** | 700–900 |
| **Depends on** | `conversation_scope`, `parsed_question`, `population`, the semantic registry |

### F.3 `MultiResultEnvelope`

| | |
|---|---|
| **Responsibility** | Carry N governed results in one response, with per-result receipts and one composed answer. Needed for shape C (clause-scoped filter) and for two-population comparison. |
| **Owner** | `mi_agent_api/adapters.py` — extend `adapt_workflow_result` to accept a list; a second adapter would be the duplication this estate removes |
| **Inputs** | ordered `[(MIQuerySpec, MIQueryResult, ExecutionReceipt)]` |
| **Outputs** | today’s envelope plus `results[]`, each with its own `spec`, `executionSummary`, `artifacts[]`; `ok` is true only when **every** requested output executed |
| **Invariants** | ① one receipt per result, never a merged one; ② a partial set is `ok:false` with each missing output named; ③ every result declares the scope it was calculated over, so two rows with different populations cannot read as one |
| **Prod LOC** | 200–300 |
| **Test LOC** | 350–450 |
| **Depends on** | `adapters`, `execution_receipt` |

**Not required:** `QueryPlan` (the measure-set dispatcher and
`analytical_plan.Plan` cover it), a conversational store (§G), a second
semantic registry, and any new calculation primitive.

---

## G. Blast radius

### Substantially unchanged — the answer to the question you asked

| Module | Change |
|---|---|
| `mi_agent/mi_query_executor.py` (1,906 ln) | **none.** A merged spec is a spec. Verified by the replay in §B. |
| `analytics_lib/*`, `mi_agent/statistic.py`, `seasoning.py`, `quantile_buckets.py` | **none.** No calculation primitive is touched. |
| `mi_agent/mi_semantics_field_registry.yaml` + `build_mi_semantics_registry.py` | **none.** No new field, measure or aggregation. |
| `mi_agent/execution_receipt.py` (4,767 ln) | **additive only** — one facet kind (`KIND_INHERITED`) so an inherited predicate is disclosed. No reconciler logic moves. |
| `mi_agent/population.py` | **none.** `Predicate` and `material_predicates` are reused as-is. |
| `mi_agent/mi_chart_factory.py` | **none.** Renderer already reads only the result. |
| `question_interpretation/*` | **none.** `from_parts` projects from the merged spec exactly as it projects from today’s. |
| `mi_agent_api/analytical_plan.py` | **none.** It plans from the interpretation, which is projected from the merged spec. |
| `trakt_core/*` (context, tenancy, policy, audit) | **none** for V1 client-carried scope. |
| `chat_routing.py` (4,365 ln) + 13 recognisers | **none** — and this must be enforced: the resolver runs above routing precisely so no recogniser acquires conversational state. |

### Modified

| Module | Change | Est. LOC |
|---|---|---|
| `mi_agent_api/mi_service.py` | one resolver call after parse; `ConversationScope` construction on success; scope on the envelope | 60–90 |
| `mi_agent_api/app.py` | `QueryRequest.conversationScope`; echo the new scope on the response | 25–40 |
| `mi_agent_api/adapters.py` | multi-result envelope; publish `conversationScope` | 200–300 |
| `mi_agent/mi_agent_workflow.py` | thread the resolver’s `ScopeDecision` into the receipt | 40–70 |
| `mi_agent/llm_query_parser.py` | **Stage 1 only**: call `_borrower_structure_filter` from `_measure_set_recognizer`; represent a clause-scoped predicate | 60–120 |
| `mi_agent/mi_query_spec.py` | a clause-scope channel for shape C (see risk below) | 40–80 |
| `frontend/mi-agent-ui`, `teams_bot.py`, `copilot_actions.py` | carry the opaque scope back on the next request | 60–120 total |

---

## H. Delivery estimate

Grounded in the seams above, not in calendar intuition.

### Stage 1 — same-turn 2–3 output composition

Close the two composition defects and prove every requested output ran.

- **Production LOC 300–550.** The measure-set population fix is ~10 lines at one
  call site (`llm_query_parser.py:3524`). The rest is shape C: representing a
  clause-scoped predicate, and either executing N specs or carrying a
  per-clause filter — plus the `MultiResultEnvelope`.
- **Test LOC 600–900.**
- **Modules 3–4:** `llm_query_parser`, `mi_query_spec`, `adapters`, `mi_agent_workflow`.
- **Risks.** (i) The real decision is *whether shape C is one spec with clause
  scope or N specs in one envelope.* N specs is cheaper and reuses the executor
  untouched; one spec with clause scope changes the contract every reader of
  `spec.filters` depends on — and there are many. **Recommend N specs.**
  (ii) `_measure_set_recognizer` runs before three other filter branches;
  widening it changes which branch claims a sentence, so the compound canary
  bank and the 115-bank must be re-run. (iii) `MAX_MEASURES` truncation is
  already fail-closed and must stay so.

### Stage 2 — conversational scope inheritance

- **Production LOC 650–950** (`ConversationScope` 180–260, `ScopeResolver`
  350–500, wiring 120–190).
- **Test LOC 1,200–1,700.** Higher than production by design: every governance
  mode in §I needs a named test, and the resolver is the only place that can get
  a population wrong.
- **Modules 2 new + 3 modified.**
- **Risks.** (i) **Referent ambiguity is the hard part** — nothing today detects
  that “their” has two candidates; V1 should refuse whenever the prior turn was
  grouped or produced more than one population. (ii) **Predicate folding** on a
  field that already carries a bound — the dict cannot hold two. (iii) **Dataset
  transition** — `config/mi/pipeline_field_contract.yaml` declares
  `funded_correlation`, and Phase 0 found *nothing reads it at bind time*; until
  something does, a dataset change must drop the population. (iv) An inherited
  predicate must be visibly *inherited* on the receipt, or silent inheritance
  becomes indistinguishable from a stated filter.

### Stage 3 — conversational chart/table requests

- **Production LOC 150–300.**
- **Test LOC 250–400.**
- **Modules 2–3:** `scope_resolution` (presentation ops), `adapters`, and the
  presentation-verb lift out of the parser.
- **Risks.** (i) “Put that in a table” currently reaches the measure resolver —
  the fix is to decide presentation *above* the parser, not to add verbs.
  (ii) A two-population chart needs Stage 1’s multi-result envelope, so Stage 3
  depends on Stage 1, not only on Stage 2.

**Total: ~1,100–1,800 production LOC, ~2,050–3,000 test LOC, 2 new modules and
~7 modified.** No calculation code, no registry change, no executor change.

---

## I. Governance requirements

| | Failure mode | Existing guard | New invariant required |
|---|---|---|---|
| A | **Stale scope** | none — there is no state to go stale | `ScopeResolver`: scope expires with the snapshot it was built on; a turn whose `snapshot_id` differs starts fresh |
| B | **Scope leakage** across user/tenant/portfolio | `authorise_portfolio_access`, `ExecutionContext.tenant_id`, `TENANT_MISMATCH` (`mi_service.py:~495`) — all still run per request | `ConversationScope` carries `tenant_id` + `portfolio_scope` + `snapshot_id` and is **rejected**, not merged, on any mismatch. With client-carried scope this is mandatory: the scope is untrusted input. |
| C | **Silent inheritance** | `ExecutionReceipt.render` names every applied filter | a new facet kind so an inherited predicate reads *“London (carried from your previous question)”*. Without it the receipt cannot distinguish stated from inherited. |
| D | **Silent reset** | receipt says “entire funded portfolio”; `reconcile_population` refuses a raised-but-unapplied population | `ScopeResolver` may not drop an inherited predicate without recording it in `ScopeDecision`, and the receipt must state the widening |
| E | **Ambiguous pronoun** | none. Measured: dialogue 5 answers over the whole book without clarifying. | resolver returns a **clarification**, never a choice. V1: refuse whenever the prior turn was grouped or produced >1 population. |
| F | **Numeric reference confusion** | **already prevented** — `_FILTER_COMPARATORS` / `_POSTFIX_COMPARATORS` produce no predicate from a bare number (5/5 probes) | resolver must map the money phrase to `ScopeOutput.result_id`, and must **never** synthesise a predicate from it |
| G | **Multi-output partial answer** | `measures_requested`/`executed`/`unavailable`; `_multi_measure_answer` names what did not run | `MultiResultEnvelope`: `ok` is true only when every requested output executed. **Shape C violates this today** — 2 of 3 executed, `ok:true`. |
| H | **Conflicting filters** | none — `spec.filters` is field → **one** condition, so a second bound silently replaces the first | resolver **folds** compatible bounds into one governed condition (`between` is already executable) and **refuses** an empty intersection. Never appends, never silently replaces. |
| I | **Dataset crossover** | `resolve_dataset(question)`; `metadata.dataset` and `ExecutionReceipt.dataset` name the book; `_require_column` makes an absent field `UNAVAILABLE`, never silent | on a dataset change, drop the whole inherited population unless the registry declares an equivalence. Phase 0 established `funded_correlation` is declared but **unread at bind time**, so V1 must not rely on it. |
| J | **Time scope** | `spec.comparison_basis`, `compare_periods`, `_guard_temporal_honouring`, `check_window_coverage` | `comparison_basis`/`compare_periods` are **never** inherited; a new turn is current-period unless it says otherwise |

Note the asymmetry: guards B, D, F, G(partial), I are **already in place** and
were built for a different reason. C, E, H and the strict half of G are new, and
all four live in the two new components.

---

## Phase I — state storage / API architecture

**The MI API is stateless between requests.** `MiQueryRequest` (`mi_service.py:83`)
carries no session, conversation or turn identifier; nothing in `mi_agent/` or
`mi_agent_api/` persists analytical state. The Teams bot stores a Bot Framework
*conversation reference* (`teams_bot.py:142`) for proactive sends only — it
holds no MI state.

One channel already exists and is the right one: **`MiQueryRequest.filters`**,
the drill-through channel, is a caller-supplied population merged onto the spec
by `ParsedQuestion.merge_filters` (`mi_service.py:1600`) and raised as
`drill_population_facets` (`execution_receipt.py:3366`) so it reaches the
receipt. A client-carried population is therefore an *established pattern*, not
a new one.

| Option | Reproducible | Azure scale-out | Teams | Dashboard | Concurrency | Expiry | Audit | Tenant isolation | Replay |
|---|---|---|---|---|---|---|---|---|---|
| **Client sends `ConversationScope`** | ✅ scope + question fully determine the answer | ✅ no affinity, no shared store | ✅ bot echoes it | ✅ SPA holds it | ✅ per-request | ✅ snapshot-bound | ✅ scope is on the request, so the audit line carries it | ⚠️ **untrusted input — must be re-authorised every turn** | ✅ |
| Server-side session state | ✅ | ❌ needs sticky sessions or Redis | ✅ | ✅ | ⚠️ | ⚠️ | ✅ | ✅ | ✅ |
| Durable conversation store | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅✅ | ✅ | ✅✅ |
| Signed / opaque scope token | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ built into the token | ✅ | ✅ signature binds the tenant | ✅ |

**Recommendation for V1: client sends `ConversationScope`, re-validated
server-side every turn.**

It is the only option that adds no infrastructure, it matches the stateless
capability the estate already built, it survives horizontal scale-out with no
affinity, and it makes every turn reproducible from `(scope, question)` alone —
which is the strong preference stated in the brief.

The condition is not optional: **the scope is untrusted caller input**, exactly
as `MiQueryRequest.filters` is today. Every turn must re-run
`authorise_portfolio_access` and `evaluate_source_approval`, and must reject a
scope whose `tenant_id`, `portfolio_scope` or `snapshot_id` does not match the
current request. A scope must never be able to *widen* what the caller may see;
it only narrows.

**Recommendation for V2 (not V1): a signed opaque token.** Once V1 proves the
shape, signing removes the tamper surface and gives expiry for free without
introducing a store. A durable store is worth it only when replay-an-answer-later
becomes a product requirement in its own right.

---

## Phase K — the minimum V1

### In

- one active governed population, carried by the client, re-authorised each turn;
- funded **or** pipeline (one dataset per conversation);
- 1–3 requested outputs, over one population, with `measures_executed` proving
  each ran;
- existing governed measures and filters only;
- one grouping dimension;
- **INHERIT**, **ADD** (with bound folding), **MODIFY**, **RESET**;
- table / chart presentation as a scope edit;
- every inherited predicate disclosed on the receipt as inherited;
- refuse on any ambiguous referent.

### Out — and why

I recommend a **smaller** perimeter than the brief sketched, on evidence:

| Excluded | Evidence |
|---|---|
| **Two-population comparison** | it needs the multi-result envelope *and* referent disambiguation *and* a comparison presentation contract — three new things at once. Ship it in V1.1 once Stage 1 is proven. |
| **Two grouping dimensions** | one dimension is a scope edit; two is a shape whose atoms the harness has not characterised here. |
| **Funded ↔ pipeline transition within a conversation** | `funded_correlation` is declared in `config/mi/pipeline_field_contract.yaml` and, per Phase 0, **not read at bind time**. Until an owner reads it, a dataset change must drop the population — which is a refusal, not a feature. |
| **Comparison-basis / period inheritance** | `comparison_basis` and `compare_periods` reach the compare routes; inheriting them risks governance mode J for no V1 benefit. |
| **Routed capabilities as conversation turns** (forecast, scenario, risk limits, concentration, evolution) | thirteen recognisers with route-specific scopes. V1 inherits only into the point-in-time path; a routed turn resets the scope. |
| **PPTX / deck generation from a conversation** | `deck_generation` is a separate job contract; no reason to couple it in V1. |
| **Any new economics, any new measure, any nested reasoning** | out by definition. |
| **“Of the £38m” as anything but a reference** | it must resolve to `ScopeOutput.result_id` or refuse. Never a predicate. |

---

## Readiness / stop conditions

| # | Condition | Result |
|---|---|---|
| 1 | Executed population cannot be reconstructed from governed contracts | **PASS** — reconstructed and replayed; §B |
| 2 | State would have to be natural-language text | **PASS** — `spec.filters` is canonical and already on the envelope |
| 3 | Executors cannot be reused; composition duplicates economics | **PASS** — a scope-built spec runs on the unchanged `execute_mi_query` |
| 4 | No reliable way to prove every requested output executed | **PASS for measures** (`measures_executed`); **FAIL for clause-scoped outputs** — shape C reports `ok:true` having executed 2 of 3 |
| 5 | Inheritance would bypass semantic guards | **PASS** — the resolver sits above routing and below parsing; every guard still runs on the merged spec |
| 6 | Multi-output requires changing calculation semantics | **PASS** — the measure set is shipped and registry-governed |
| 7 | Funded/pipeline transitions cannot be governed safely | **CONDITIONAL** — safe only by *refusing* the carry-over, because `funded_correlation` is unread at bind time |

One condition fails and one is conditional. Neither is a core-architecture
problem; both are bounded, and both are in the same-turn path that Stage 1
closes before any conversation exists.

---

## Verdict

# CONDITIONAL GO

The architecture is sound. The layer genuinely sits **above** the deterministic
system: a `ConversationScope` derived from `MIQuerySpec` + execution metadata
produces a spec that runs on the unchanged executor and lands on exactly the
rows the original question landed on. Deterministic execution, the calculation
primitives, the semantic registry and the execution receipt can all remain
substantially unchanged.

**The three bounded prerequisites:**

1. **Close shape C before any conversation exists.** A clause-scoped filter
   promoted to the shared population is a *silent* wrong answer today
   (`ok:true`, 2 of 3 outputs, the narrow cohort reported as the wide one). A
   conversational layer inheriting from such a turn would inherit a population
   nobody asked for. This is stop condition 4, and it is the only silent shape
   in the review.
2. **Give the population one owner in the multi-measure path.**
   `_measure_set_recognizer` must ask `_borrower_structure_filter`, as the other
   three branches do. It fails closed today, so it is not urgent for safety —
   but a follow-up turn that inherits `{}` inherits nothing, and the reader
   cannot tell an empty scope from a whole-book one.
3. **Build `ConversationScope` as a derivation, never a parse.** Its constructor
   must take `(spec, metadata, governed_result)` and must be structurally unable
   to read a question — the discipline `analytical_plan.assert_no_question_read`
   already enforces for the plan layer. A scope that can be built from prose is
   stop condition 2 arriving through the back door.

### “If the 115-bank and 100-bank atomic perimeter are high-conviction, is conversational analytical composition the next logical MI sprint?”

**Yes — but Stage 1 first, and Stage 1 is not a conversational sprint.**

The evidence:

- **The atoms genuinely compose.** 7 of 7 multi-output shapes had all atoms
  green. Not one case was blocked by a broken atom. The compositional risk is
  therefore *isolated* — it is not contaminated by atomic breadth, which is
  what the 115-bank and 100-bank were buying.
- **The failures are three defects, not a class of inability.** 4 of 7 failed;
  two share one root (`_measure_set_recognizer` not asking the borrower owner),
  one is the clause-scope contract limit, and the bare-place-name drop is
  atomic and belongs to the other sprint. This is the estate's familiar shape —
  *one concept with two owners* — not a new frontier.
- **The estate fails closed where it matters.** 3 of 4 composition failures and
  4 of 9 multi-turn follow-ups refuse rather than answer. The one silent shape
  is precisely locatable and is same-turn.
- **The dangerous half of result-reference is already solved.** 5 of 5 money-
  reference probes produce no predicate while the explicit-threshold control
  still binds. The layer must *add* meaning, not *remove* a hazard.
- **The seams already exist and are already single-owner.** `ParsedQuestion`
  exists because the question used to be parsed twice. `analytical_plan` already
  plans from a contract and is structurally forbidden to read the question.
  `MiQueryRequest.filters` already carries a caller-supplied population onto the
  spec. `artifacts[]` already carries several presentations of one result.
  `create_mi_chart` already reads only the result. The conversational layer is
  the *use* of seams the estate has already built for other reasons — which is
  the cheapest kind of sprint this codebase can run.
- **The counter-evidence is honest.** No new economics are needed, but Stage 1
  is ~300–550 production LOC of same-turn work that must land before turn 2 is
  safe, and referent ambiguity (§I-E) is a genuinely new capability with no
  existing primitive. Budget for the refusal path, not the happy path.

The conviction the two banks bought is conviction about *atoms*. This review
finds that the composition of those atoms is one contract limit and two
one-owner defects away from working — which makes it the next logical sprint,
provided Stage 1 is run as an atomic-perimeter sprint rather than as the first
week of a conversational one.

---

## Appendix — reproducing this

```bash
python migration_phase0/conversational_readiness_probe.py
python migration_phase0/conversational_readiness_probe.py --out report.json
python -m pytest mi_agent/tests/test_conversational_composition_characterisation.py -q
```

Both are read-only. The characterisation module pins the facts this verdict
rests on — including the two defects — so a change that moves one of them fails
there rather than silently invalidating this document.

**Suite note.** Three tests in `mi_agent/tests/test_p0_execution_receipt.py`
(`test_receipt_discloses_a_requested_dimension_the_dataset_lacks`,
`test_an_unavailable_dimension_is_never_replaced_by_another_field`,
`test_an_unavailable_dimension_never_simply_disappears`) fail on this branch
before any change in this sprint, with `KeyError: 'facets'`. They are unrelated
to this review and untouched by it; recorded here because they were observed
while running the suite.
