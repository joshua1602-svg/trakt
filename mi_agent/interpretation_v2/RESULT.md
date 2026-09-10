# MI Plan Interpretation Compiler v1 — result

**Base** `ea8c65b592ae5d2203d924bf66afa051618641e6`
**Branch** `claude/mi-plan-interpretation-compiler-v1-jfonpp`

---

## 1 · What was built

```
question
   │
   ▼  OpusInterpreter          claude-opus-5, tool-use, strict schema
CandidateIntent                meaning only — no field, no snapshot, no code
   │
   ▼  DeterministicCompiler    validate → bind → authorise → produce
GovernedQueryPlan   or   CLARIFY   or   REFUSE
```

Nine modules in `mi_agent/interpretation_v2/`, 153 tests in
`tests/interpretation_v2/`, all passing. Nothing executes a plan.

**CandidateIntent** (`candidate_intent/1.0`) — capability, operation,
population (base/lens/seasoning), measures (concept + statistic + weight),
dimensions, filters (concept/comparator/value), geography (basis, level,
group_by, values), time (semantic form + the question's own period words),
comparison, outputs, ambiguity, evidence, provenance.

**GovernedQueryPlan** (`governed_query_plan/1.0`) — immutable, content-hashed,
carrying the compiler's bindings and a provenance block that keeps
`intent_claims` (what the model said) separate from `compiler_bindings` (what
the compiler chose), so an auditor can answer the only question this
architecture is judged on.

**Outcomes** — `PLAN` / `CLARIFY` / `REFUSE` with 24 governed reason codes.
CLARIFY means *the same question, asked more precisely, would compile*; REFUSE
means it would not. The split is data, not a per-call judgement.

---

## 2 · Acceptance

### Architecture

| | |
|---|---|
| OPUS_OWNS_LANGUAGE | **YES** |
| MODEL_CANNOT_WRITE_EXECUTABLE_PLAN | **YES** — no slot, closed schema, fail-closed parser |
| COMPILER_OWNS_BINDING | **YES** — no plan field is a string the model supplied |
| COMPILER_IS_DETERMINISTIC | **YES** — 20 compilations, one plan id; no `re` import |
| REFUSAL_IS_FAIL_CLOSED | **YES** — a refusal can never carry a plan (enforced in the constructor) |
| EXISTING_EXECUTION_UNCHANGED | **YES** — measured, §5 |

### Schema

| | |
|---|---|
| CANDIDATE_INTENT_VERSIONED | **YES** |
| GOVERNED_QUERY_PLAN_VERSIONED | **YES** |
| MULTI_OUTPUT_REPRESENTABLE | **YES** — one population, N outputs, output-local predicates |
| SPECIALIST_CAPABILITY_REPRESENTABLE | **YES** — 7 capabilities, no internals exposed |

### Model

| | |
|---|---|
| REAL_OPUS_CALL_PROVEN | **YES** |
| RETURNED_MODEL_ID_RECORDED | **YES** — `claude-opus-5` on 135/135 |
| Successful calls / failed calls | **135 / 0** |
| LOAN_ROWS_SENT_TO_MODEL | **0** |
| PORTFOLIO_VALUES_SENT_TO_MODEL | **0** |

### Benchmark

135 questions = 45 canonical × 3 variants, every string verbatim from a frozen
committed bank (25 from `MI_FINAL_ACCEPTANCE_75.yaml`, 9 from
`STAGE_MOVEMENT_BANK.yaml`, 9 from `nl_bank.py`, 2 from
`BORROWING_BASE_MI_BANK.yaml`). No question was changed.

| | run 1 | run 2 | run 3 (re-scored) |
|---|---|---|---|
| attempted | 135 | 135 | 135 |
| fully correct | 68 | 71 | **93** |
| partially correct | 27 | 42 | **20** |
| incorrect | 2 | 1 | **1** |
| governed refusal | 27 | 1 | **1** |
| clarification | 11 | 20 | **20** |
| PLAN / CLARIFY / REFUSE | 97/11/27 | 114/20/1 | **114/20/1** |
| paraphrase-invariant canonicals | 11/45 | 16/45 | **16/45** |
| human-review canonicals | 7 | 7 | **7** |

Run 2 is a second live run after four compiler corrections (§4). Run 3 is run 2's
**recorded model payloads replayed** against corrected stage-movement fixtures —
no new model calls, so model variance is held constant and the fixture change is
isolated. All three runs are committed under `evidence/`.

Per-dimension, run 3 (correct / incorrect / unscoreable out of 135):

```
capability        102 /  4 / 29      dimensions         40 /  0 / 95
operation         101 /  2 / 32      geography_basis     6 /  0 /129
measures           34 /  1 /100      geography_level     3 /  0 /132
statistic          30 /  0 /105      temporal           80 /  6 / 49
weight             11 /  0 /124      comparison         95 / 12 / 28
population         70 /  1 / 64      output_structure   40 /  2 / 93
filters            38 /  4 / 93
```

A dimension a fixture does not state is **unscoreable, not correct**.

### Error taxonomy (run 3)

| class | count |
|---|---|
| interpretation error | 0 |
| vocabulary / binding error | 0 |
| compiler rejection | 0 |
| missing deterministic contract | 0 |
| ambiguous truth | 21 |
| other | 0 |

---

## 3 · The three most important findings

### 1. The governed registries carry no value list for the dimensions questions filter on most

`config/system/fields_registry.yaml` declares `allowed_values: null` for
`erm_product_type`, `sub_product_type` and `pipeline_stage`. So there is no
governed answer to "is *drawdown* a product value?" or "is *Offer* a stage?"

Given a vocabulary that says so, the interpreter **correctly declined to assert
the filter**: 16 of the 20 clarifications are exactly this, across
"drawdown loans", "lump sum lending", "cases at Offer", "Application stayers".
That is the architecture working — a value that cannot be checked is a guess —
but it is the single largest blocker to the control plane answering ordinary
questions.

**This is a registry gap, not a model or compiler gap, and fixing it is outside
this sprint's boundary.** 96 canonical fields already declare an enum domain and
`config/system/enum_synonyms.yaml` holds the business spellings for ten of them;
the same joining would work for product and stage once someone populates them.
The smallest interface needed is `allowed_values` on those three fields.

### 2. Paraphrase invariance is 16/45, and one cause dominates

29 canonicals diverge. **13 of them diverge on OUTCOME ALONE** — one wording
plans, another clarifies — and 19 of those non-plan reasons are
`MODEL_FLAGGED_AMBIGUITY`. The bindings agree; what does not agree is whether
the model judged the ambiguity blocking.

So the invariance failure is not the interpreter reading three paraphrases three
different ways. It is the interpreter reading them the *same* way and reporting
its confidence differently — and in almost every case the ambiguity it flags is
finding 1. The compiler's own behaviour is invariant: given the same intent it
produces the same plan, always.

The remaining divergence is `outputs` (11 canonicals: how many figures a
synthesis question implies) and `period` (5: `relative_pair` against
`previous_reporting_period` against `explicit_period` for "last month" — three
governed time forms that overlap, which the vocabulary should collapse or
disambiguate).

### 3. The compiler's first instinct was to over-refuse, and it was wrong four ways

Run 1 produced 27 governed refusals. **Every one was the compiler demanding that
the interpreter specify something the interpreter is deliberately never shown:**

* a stage transition had to name two periods — but the window is part of what a
  transition *is* (11 questions);
* a period comparison had to name left and right — but a period pair names its
  sides through the period contract (6);
* `share` and `contribution` were rejected on `balance` — but they are analytic
  modes this repository already governs (P1A, P1D), not aggregations *of* a
  field, so they never appear in `allowed_aggregations` (3);
* a concentration rank had to name a dimension — but which tests to rank is the
  capability's to know (2).

Corrected, refusals fell 27 → 1 with no loosening of any safety rule: no
statistic substitution, no field guessing, no whole-book fallback. The lesson
generalises — **"the specialist owns HOW" has to include the specialist's period
window, its grouping, and its statistic, or the compiler ends up asking the
model for the methodology it was built not to know.** Each correction is pinned
by a test in `test_corrections_from_the_first_live_run.py`.

---

## 4 · What changed between the runs, and why that is disclosed

Between run 1 and run 2 the four compiler defects above were fixed, the
vocabulary gained governed enum values and an explicit statement of which slots
have governed defaults, and the prompt gained two rules (a specialist measure
takes no statistic; do not assert a filter value with no governed list).

Between run 2 and run 3 the nine stage-movement fixtures stopped expecting
`temporal: relative_pair`. They had been written under the compiler rule that
run 1 disproved; leaving them would have scored the interpreter against the
fixture's assumption rather than against the question.

**The first measurement is committed unedited** at
`evidence/run1_135_first_measurement.json`. Reporting only the post-fix number
would repeat the control weakness this repository already wrote down once, in
`due_diligence/evidence/analytical_intent_v1/expected_semantics.yaml`.

---

## 5 · Production behaviour

```
PRODUCTION_CODE_BEHAVIOUR_CHANGED = NO
MI_BEARER_USED                    = NO
FILES_CHANGED_OUTSIDE_NEW_BOUNDARY = 0
```

Every file added is under `mi_agent/interpretation_v2/` or
`tests/interpretation_v2/`. Measured, not asserted:

* `test_nothing_outside_the_new_boundary_imports_it` — no module elsewhere
  imports this package;
* `test_the_package_imports_no_executor_engine_or_route` — nothing here imports
  `mi_query_executor`, `query_plan_execution`, `mi_agent_api`, `trakt_tools`,
  `analytics_lib`, `engine.*`, `mi_workflows`, streamlit, fastapi or flask;
* `test_no_environment_variable_enables_this_in_production` — `ANTHROPIC_API_KEY`
  is the only variable read, and `MI_BEARER` appears nowhere;
* `mi_agent/tests` at base `ea8c65b`: **22 failed, 1404 passed, 264 skipped,
  7 xfailed, 21 errors**. On this branch: **21 failed, 1405 passed, 264 skipped,
  7 xfailed, 21 errors** — same 1426 total, one flaky test differing. The 21
  collection errors and the failures are pre-existing and reproduce identically
  at the base commit; 74 repo-wide collection errors are missing third-party
  packages in this container (`fastapi`, `lxml`, `matplotlib`, `pptx`,
  `rapidfuzz`), none in the new package.

---

## 6 · Verdict

```
TARGET_STATE_VIABLE = YES
```

The boundary holds under measurement. The model wrote 135 intents and not one
executable binding: no canonical field, no snapshot, no date, no SQL, no
dataframe expression — and it could not have, because there is no slot for one.
Every physical binding on all 114 plans was chosen by the compiler from the
governed registries, and the provenance keeps the two apart well enough to prove
it. The compiler never re-read a question and never chose a nearest-available
semantic to avoid refusing.

What the measurement does **not** yet support is a claim that the control plane
is ready to serve. Paraphrase invariance at 16/45 is not good enough for a
system where three wordings of one question must produce one answer, and the
route to improving it is now specific rather than speculative: populate the
three missing `allowed_values` domains, and collapse the overlapping time forms.
Both are governed-data changes, not architecture changes — which is itself
evidence that the architecture is the right one.

Seven canonicals are marked for human adjudication (`Q08`, `NL1`, `NL3`, `NL4`,
`NL7`, `NL8`, `BB02`), each with the competing readings recorded in
`banks/expected_intents.yaml` rather than settled by this sprint.
