# MI Plan Interpretation Compiler v1 — result

**Base** `ea8c65b592ae5d2203d924bf66afa051618641e6`
**Branch** `claude/mi-plan-interpretation-compiler-v1-jfonpp`

---

## 1 · What was built

```
question
   │
   ▼  OpusInterpreter  ◄──►  read-only governed metadata
   │     claude-opus-5          system registry · asset config
   │     tool-use loop          portfolio config · capability catalogue
   ▼
CandidateIntent                 meaning — a PROPOSAL, never a binding
   │
   ▼  DeterministicCompiler     validate → bind → authorise → produce
GovernedQueryPlan   or   CLARIFY   or   REFUSE
```

Ten modules in `mi_agent/interpretation_v2/`, 185 tests in
`tests/interpretation_v2/`, all passing. Nothing executes a plan.

**CandidateIntent** (`candidate_intent/1.0`) — capability, operation, population
(base/lens/seasoning), measures (concept + statistic + weight), dimensions,
filters, geography (basis, level, group_by, values), time (semantic form + the
question's own period words), comparison, **target**, outputs, ambiguity,
evidence, provenance.

**GovernedQueryPlan** (`governed_query_plan/1.0`) — immutable, content-hashed,
with provenance keeping `intent_claims` (what the model said) separate from
`compiler_bindings` (what the compiler chose).

**Outcomes** — `PLAN` / `CLARIFY` / `REFUSE` with 24 governed reason codes.
CLARIFY means *the same question, asked more precisely, would compile*; REFUSE
means it would not. The split is data, not a per-call judgement.

**Governed metadata access** — seven read-only tools (`search_concepts`,
`get_concept_metadata`, `get_allowed_values`, `search_capabilities`,
`get_capability_metadata`, `get_asset_metadata`,
`get_portfolio_semantic_context`) over thirteen authoritative sources. Opus
retrieves what it needs and may name a canonical identifier it finds. The
compiler re-derives existence, ambiguity, asset applicability, portfolio
availability, permitted operation and physical binding for every one of them
before a plan can carry it — and reads the index *directly*, never through the
tool surface, because the tool surface is what the model talks to.

---

## 2 · Acceptance

### Architecture

| | |
|---|---|
| OPUS_OWNS_LANGUAGE | **YES** |
| MODEL_CANNOT_WRITE_EXECUTABLE_PLAN | **YES** — no slot, closed schema, fail-closed parser |
| COMPILER_OWNS_BINDING | **YES** — every field re-derived from the registry; a column-shaped string the registry lacks never binds |
| COMPILER_IS_DETERMINISTIC | **YES** — 20 compilations, one plan id; no `re` import |
| REFUSAL_IS_FAIL_CLOSED | **YES** — a refusal can never carry a plan (enforced in the constructor) |
| EXISTING_EXECUTION_UNCHANGED | **YES** — measured, §6 |

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
| RETURNED_MODEL_ID_RECORDED | **YES** — `claude-opus-5` on 135/135 in run 4 |
| Successful / failed calls (run 4) | **135 / 0** |
| Metadata retrievals (run 4) | **872**, all seven tools, every question retrieved |
| LOAN_ROWS_SENT_TO_MODEL | **0** |
| PORTFOLIO_VALUES_SENT_TO_MODEL | **0** |

### Benchmark

135 questions = 45 canonical × 3 variants, every string verbatim from a frozen
committed bank (25 from `MI_FINAL_ACCEPTANCE_75.yaml`, 9 from
`STAGE_MOVEMENT_BANK.yaml`, 9 from `nl_bank.py`, 2 from
`BORROWING_BASE_MI_BANK.yaml`). No question was changed.

**Run 4 is the authoritative full measurement.**

| | run 1 | run 2 | run 3 | **run 4** |
|---|---|---|---|---|
| attempted | 135 | 135 | 135 | **135** |
| fully correct | 68 | 71 | 93 | **107** |
| partially correct | 27 | 42 | 20 | **19** |
| incorrect | 2 | 1 | 1 | **1** |
| governed refusal | 27 | 1 | 1 | **2** |
| clarification | 11 | 20 | 20 | **6** |
| PLAN / CLARIFY / REFUSE | 97/11/27 | 114/20/1 | 114/20/1 | **127/6/2** |
| paraphrase-invariant | 11/45 | 16/45 | 16/45 | **20/45** |
| human-review canonicals | 7 | 7 | 7 | **7** |

Per-dimension, run 4 (correct / incorrect / unscoreable out of 135):

```
capability        114 /  1 / 20      dimensions         46 /  0 / 89
operation         106 /  5 / 24      geography_basis     6 /  0 /129
measures           42 /  0 / 93      geography_level     3 /  0 /132
statistic          36 /  0 / 99      temporal           86 /  3 / 46
weight             15 /  0 /120      comparison        109 /  9 / 17
population         85 /  0 / 50      output_structure   45 /  3 / 87
filters            46 /  3 / 86
```

A dimension a fixture does not state is **unscoreable, not correct**.

Error taxonomy (run 4): interpretation error 0, vocabulary/binding error 0,
compiler rejection 2, missing deterministic contract 0, ambiguous truth 6,
other 0.

### Run 5 — a partial re-measurement, and why it is partial

Run 5 tested the two contract fixes of §5. **The API credit balance was
exhausted at question 61**, so 75 of 135 calls returned a 400 and the run
covers 60 questions. Its whole-bank figures are therefore meaningless and are
not reported: in particular its "35/45 invariant" is an artefact of 75
questions failing identically, which makes their canonicals spuriously
agree. The file is named
`evidence/run5_PARTIAL_60of135_credit_exhausted.json` so it cannot be misread.

On the 60 questions it did serve, like for like against run 4:

| | run 4 | run 5 |
|---|---|---|
| fully correct | 45 | **53** |
| partially correct | 13 | **6** |
| governed refusal | 2 | **1** |
| `comparison` dimension misses | 5 | **0** |
| invariant (of the 20 canonicals fully served) | 10 | 10 |

The `period_pair` removal did exactly what it was meant to — every
`comparison` miss on that subset disappeared — and per-question correctness
improved. It did **not** improve paraphrase invariance on that subset, which
stayed at 10 of 20.

---

## 3 · The three most important findings

### 1. The interpreter was being starved of metadata Trakt already governed

The first build gave Opus one flat, business-name-only vocabulary from a single
registry. It clarified on ordinary questions — "how many drawdown loans", "cases
at Offer", "Application stayers" — because it could not verify that "drawdown"
is a product or "Offer" a stage. Sixteen of run 3's twenty clarifications were
that one gap.

The values were governed all along, in sources the vocabulary never read:

| | where it actually lives |
|---|---|
| `drawdown`, `lump_sum`, `rio`, `erm` | `config/asset/product_profiles.yaml` → `match.product_type` |
| `KFI`, `APPLICATION`, `OFFER`, `COMPLETED`, `WITHDRAWN` | `question_interpretation.lexical` (22 spellings → 5 stages) |
| analytical concept, temporality, asset applicability | `config/business_semantics_registry.yaml` (243 entries) |
| 12 UK regions; governed band labels | `config/mi/region_taxonomy.yaml`, `config/mi/buckets.yaml` |
| 42 governed limit metrics | `config/risk/concentration_test_library.yaml` |

Run 3 reported this as a registry gap needing `allowed_values` populated. That
was **wrong about the remedy**: nothing needed populating, the adapter needed
to read what was there. Clarifications fell 20 → 6 and fully-correct rose
93 → 107 on the strength of reading it.

The generalisable lesson is about where a boundary belongs. Withholding the
registry from the model did not make the system safer — the compiler's
independent re-derivation is what makes it safe — it only made the model guess
or decline. The safety property is *re-validation*, not *ignorance*.

### 2. Paraphrase invariance is 20/45, and the contract is now the limiting factor

Four runs moved it 11 → 16 → 16 → 20. What remains is no longer about
vocabulary; 25 canonicals still diverge, and the causes are specific:

* **`outputs` (11 canonicals)** — how many figures a synthesis question implies.
  "Summarise the portfolio" is one output or five depending on the reader, and
  nothing in the contract says which.
* **`operation` (7)** — `rank` against `summary` for "where are our largest
  concentrations", `movement` against `bridge` for "how the funded book moved".
  The capability catalogue does not draw the `period_movement` / `funded_bridge`
  line, so the model draws it differently on different wordings.
* **`period` (5)** — `relative_pair` against `previous_reporting_period` against
  `explicit_period` for "last month". Three governed time forms overlap.
* **`outcome` (8)** — one wording clarifies where another plans.

Two of these are overlapping enumerations in my own contract, which is the same
class of defect as the `period_pair` redundancy that §5 removed — and removing
that one provably worked. The route to invariance is to collapse the remaining
overlaps, not to tune the interpreter.

### 3. The compiler's instinct is to over-refuse, and it was wrong five ways

Run 1 produced 27 governed refusals. **Every one was the compiler demanding that
the interpreter specify something the interpreter is deliberately never shown:**

* a stage transition had to name two periods — but the window is part of what a
  transition *is* (11 questions);
* a period comparison had to name left and right — but a period pair names its
  sides through the period contract (6);
* `share` and `contribution` were rejected on `balance` — but they are analytic
  modes this repository already governs (P1A, P1D), not aggregations *of* a
  field (3);
* a concentration rank had to name a dimension — but which tests to rank is the
  capability's to know (2);
* and, found in run 4, a milestone had nowhere to put its threshold — "when will
  we reach £100m" is not a filter, and the interpreter was right to refuse to
  pretend otherwise (3).

Corrected, refusals fell 27 → 2 with no safety rule loosened anywhere: no
statistic substitution, no field guessing, no whole-book fallback. **"The
specialist owns HOW" has to include the specialist's period window, its
grouping, its statistic and its thresholds, or the compiler ends up asking the
model for the methodology it was built not to know.** Each correction is pinned
by a test in `test_corrections_from_the_first_live_run.py`.

---

## 4 · The data boundary

Metadata is not data, and the distinction is enforced field by field rather than
by intention. The sharpest case: `config/risk/funding_facilities.yaml` carries a
£250m commitment, an advance rate and a concentration floor two keys from the
facility's name, and the addendum permits only "presence/absence of a funding
facility". The adapter allowlists the identity keys **positively**, so a key
added upstream cannot leak by default.

No reporting date either — the client config carries one, and handing it over
would let the model pin what period an answer covers from a prompt with no data
in it. Time is stated semantically and resolved downstream.

`test_model_sees_no_data.py` walks the standing block, the tool definitions and
**every result every tool can return** — concept metadata for all 152 concepts,
allowed values for every dimension, all capabilities, both environment tools —
asserting no currency figure, no thousands-separated number, no bare number of
six digits or more, no calendar date, no loan identifier, and no
dataframe-shaped accessor.

---

## 5 · What changed between the runs, and why that is disclosed

| between | what changed |
|---|---|
| 1 → 2 | four compiler over-refusals fixed; governed enum values and declared defaults added; prompt gained two rules |
| 2 → 3 | nine stage-movement fixtures stopped expecting a period the question does not determine (they were written under the rule run 1 disproved). **Re-scored by replaying run 2's recorded payloads — no new model call separates the two numbers.** |
| 3 → 4 | the addendum: read-only metadata tools, canonical-identifier index, asset-applicability checking, orientation in place of a prompt dump |
| 4 → 5 | `period_pair` removed as redundant; `target` slot added for milestone thresholds; one fixture corrected to the governed value spelling `lump_sum` |

**The first measurement is committed unedited** at
`evidence/run1_135_first_measurement.json`, as are runs 2, 3, 4 and the partial
run 5. Reporting only a post-fix number would repeat the control weakness this
repository already wrote down once, in
`due_diligence/evidence/analytical_intent_v1/expected_semantics.yaml`.

---

## 6 · Production behaviour

```
PRODUCTION_CODE_BEHAVIOUR_CHANGED  = NO
MI_BEARER_USED                     = NO
FILES_CHANGED_OUTSIDE_NEW_BOUNDARY = 0
```

Every file added is under `mi_agent/interpretation_v2/` or
`tests/interpretation_v2/`. Measured, not asserted:

* `test_nothing_outside_the_new_boundary_imports_it` — no module elsewhere
  imports this package;
* `test_the_package_imports_no_executor_engine_or_route` — nothing here imports
  `mi_query_executor`, `query_plan_execution`, `mi_agent_api`, `trakt_tools`,
  `analytics_lib`, `engine.*`, `mi_workflows`, streamlit, fastapi or flask;
* `test_the_package_does_not_use_the_legacy_parser_as_an_oracle` — no import of
  `llm_query_parser`, `mi_agent.interpreter`, `semantic_resolver`,
  `concept_merge_arm` or the recogniser surfaces. The one thing read from
  `question_interpretation` is `lexical.pipeline_stage_vocabulary`, a governed
  value table; reading a table is not taking a second opinion on what a sentence
  means;
* `test_no_environment_variable_enables_this_in_production` —
  `ANTHROPIC_API_KEY` is the only variable read, and `MI_BEARER` appears
  nowhere;
* `mi_agent/tests` at base `ea8c65b`: **22 failed, 1404 passed, 264 skipped,
  7 xfailed, 21 errors**. On this branch: **21 failed, 1405 passed, 264 skipped,
  7 xfailed, 21 errors** — same 1426 total, one flaky test differing. The
  failures and the 21 collection errors reproduce identically at the base
  commit; 74 repo-wide collection errors are missing third-party packages in
  this container (`fastapi`, `lxml`, `matplotlib`, `pptx`, `rapidfuzz`), none in
  the new package.

---

## 7 · Verdict

```
TARGET_STATE_VIABLE = YES
```

The boundary holds under measurement, and it held through a change that made the
model far better informed. Across 135 questions the model wrote 135 intents and
not one executable binding: no snapshot, no date, no SQL, no dataframe
expression — and it could not have, because there is no slot for one. It may now
name a canonical identifier, and that bought it nothing: every field on all 127
plans was re-derived by the compiler from the governed registries, an invented
identifier still refuses, a word seven fields claim still names all seven, and
the provenance keeps the two halves apart well enough to prove it. The compiler
never re-read a question and never chose a nearest-available semantic to avoid
refusing.

What the measurement does **not** support is a claim that the control plane is
ready to serve. Paraphrase invariance at 20/45 is not good enough for a system
where three wordings of one question must produce one answer. But the diagnosis
is now specific, and it is about this sprint's own contract rather than about the
model or the registries: overlapping enumerations — `outputs` cardinality for
synthesis questions, the `period_movement`/`funded_bridge` boundary, three
time forms that all describe "last month". The one such overlap removed in run 5
eliminated every `comparison` miss on the questions it was measured over, which
is evidence that collapsing the rest is the right next move.

Seven canonicals are marked for human adjudication (`Q08`, `NL1`, `NL3`, `NL4`,
`NL7`, `NL8`, `BB02`), each with the competing readings recorded in
`banks/expected_intents.yaml` rather than settled by this sprint.

One operational note: a full live run costs roughly 1.2M input and 200k output
tokens against `claude-opus-5`, and the key supplied for this sprint ran out of
credit partway through run 5. A sixth run to re-measure the whole bank after the
§5 fixes is the obvious next step and needs nothing but budget.
