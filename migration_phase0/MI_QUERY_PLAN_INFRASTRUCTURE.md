# QueryPlan / AnalyticalScope infrastructure — deliverables

```
base    22e9646     37 failure nodes
head    cf6c253     37 failure nodes
tranches landed   A (contracts) · B (compiler) · C (provenance/reconciliation)
tranches not landed   D (integration seam) · E (cleanup)
```

## A. Architecture map

**Before**

```
question → interpretation → MIQuerySpec → executor → receipt → envelope
                              │
                              └── ONE filters dict for the whole request
```

A request about two populations cannot be represented, only mis-represented.

**After (infrastructure; not yet on the live path)**

```
question → interpretation → QueryPlan ──────────────┐
                              ├── shared AnalyticalScope
                              └── PlannedOutput[]
                                    ├── operation / measure / aggregation
                                    └── local_scope_delta
                                                     │
                            compile_query_plan ──────┘
                                    │
              ┌─────────────────────┴──────────────────────┐
       one effective scope                    several effective scopes
              │                                            │
       ONE MIQuerySpec                          SEVERAL ordinary MIQuerySpecs
       (shipped P1E measure set)                (one per population)
              │                                            │
              └──────────► existing deterministic executor ◄┘
                                    │
                          OutputResult per output
                          (requested scope, executed scope, execution_ref)
                                    │
                          reconcile_plan → MultiResultEnvelope
```

## B. Ownership

| Concept | Target owner |
|---|---|
| analytical population | `AnalyticalScope` |
| scope delta | `ScopeDelta` (narrowing only; cannot widen by construction) |
| requested output | `PlannedOutput` |
| operation / statistic | `PlannedOutput.operation`, from `OPERATIONS` — the executor's own vocabulary |
| measure | `PlannedOutput.measure` (registry key; unchanged) |
| effective scope | `effective_scope(plan, output)` |
| relationship of scope to outputs | `QueryPlan` |
| transformation into executions | `compile_query_plan` |
| calculation | **unchanged** — `mi_query_executor` |
| what actually ran | **unchanged** — `execution_receipt`, referenced not copied |
| output completeness | `reconcile_plan` |
| provenance | `OutputResult` (both scopes + execution reference) |

## C. Production changes

Three **new** modules. **No existing production file was modified.**

| File | Contract | Responsibility |
|---|---|---|
| `mi_agent/query_plan.py` | `Predicate`, `AnalyticalScope`, `ScopeDelta`, `PlannedOutput`, `QueryPlan`, `ScopeConflict`, `effective_scope`, `scope_equivalent` | what a request means between interpretation and execution |
| `mi_agent/query_plan_compiler.py` | `CompiledExecution`, `compile_query_plan` | plan → the execution contracts the executor already accepts |
| `mi_agent/query_plan_result.py` | `OutputResult`, `PlanReconciliation`, `MultiResultEnvelope`, `reconcile_plan` | per-output provenance and completeness |

**Old responsibility removed or delegated: NONE.** Tranche E is not done, and it
must not be: nothing may be retired until the new infrastructure demonstrably
owns its semantics on the live path, which is Tranche D. Recording this as zero
rather than implying otherwise is the point of the row.

## D. Compatibility

Asserted, not assumed: a single-output plan compiles to the shape the atomic
paths already emit — `metric`, `aggregation`, `filters`, with `measures` left
empty so `normalise_measures` folds it exactly as today. A count compiles to
`aggregation=count, metric=None`. A threshold reaches the spec in the executor's
own `{field: {op, value}}` shape.

**Limit of the claim:** demonstrated at the contract level. Live atomic
behaviour is unchanged because nothing calls this yet — the regression below is
what evidences that, not the compiler.

## E. Generality

Five canonical measure/filter pairs compile identically with no production
special case: `SUM(balance)` under an LTV bound; `WEIGHTED_AVERAGE(LTV)` under a
rate bound; `WEIGHTED_AVERAGE(rate)` under an age bound; `SUM(balance)` under a
categorical geography; `COUNT` under a pipeline case-age bound. Grouping is part
of the scope, so a grouped plan compiles to a grouped spec by the same path.

## F. Regression

```
mi_agent/tests + mi_agent_api/tests + question_interpretation
base 22e9646   37 failure nodes
head cf6c253   37 failure nodes
NEW    (none)
GONE   (none)
```

Identical sets. Expected: no existing production file was touched.

## G. Verdicts

| Capability | Verdict | Evidence / condition |
|---|---|---|
| AnalyticalScope infrastructure | **GO** | one population model; frozen; structural equivalence detects widening, leakage and operator change |
| QueryPlan infrastructure | **GO** | shared scope + outputs; duplicate ids rejected; output identity includes effective scope |
| Same-scope execution compilation | **GO** | compiles to the shipped P1E measure set; no second engine |
| Output-local scope execution | **CONDITIONAL GO** | compiles to separate ordinary specs and the delta provably cannot reach siblings — but **no live execution has run through it**; that is Tranche D |
| Per-output provenance / completeness | **CONDITIONAL GO** | missing, miscoped, duplicated and unrequested all detected structurally; not yet fed by real receipts |
| Existing atomic behaviour preserved | **GO** | identical failure sets; no existing production file modified |

## Remaining, in order

**Tranche D — integration seam.** Route interpretation into `QueryPlan` behind
an adapter, so the parser's existing output populates the contracts rather than
being rewritten to produce them. Until this lands, both CONDITIONAL GOs stay
conditional: the infrastructure is proven by contract tests, not by traffic.

**Tranche E — cleanup.** Retire duplicate pathways only where D proves the new
infrastructure owns their semantics.
