# MI Query Agent — pipeline stage movement

| | |
|---|---|
| starting SHA | `30a7d4a` — the **merged, in-production MI Query Agent**, plus the governed stage-transition engine merged in #385 |
| branch | `claude/mi-query-stage-movement-cqko43` |
| fixtures | `tests/fixtures/pipeline_transition_2w` (governed pipeline), seeded funded tape across five month-ends |
| arms measured | governed engine alone **and** the concept-merge language layer (`claude-opus-5`) |

---

## 0. What was done, and what was tested

### What was done

**One new user-facing ability: natural-language questions about pipeline
stage-to-stage movement.** Nothing else was added, and no existing capability
was changed in what it answers.

Before this change the MI Query Agent could not answer *"how many cases moved
from KFI to Application?"*. Measured at `30a7d4a`, twelve of the thirty-six
questions in the new bank were answered with the **current stage stock** — a
different number answering a different question — and the other twenty-four were
refused.

The governed stage-transition capability that computes the right answer has been
merged since #385 and already serves React and the PPTX deck. This work makes MI
Query its **third consumer**. It adds:

* `mi_agent_api/stage_movement_query.py` — a recognise/handle pair registered in
  the existing recogniser registry. It **computes nothing**: no snapshot load,
  no case matching, no stage comparison, no arrival/departure/stayer counting,
  no amount amendment, no reconciliation. Every figure is a key lookup on the
  governed payload.
* 29 executable lines across three existing files: the registry entry, two
  route registrations plus one evidence alignment in the execution receipt, and
  one deference in the analytical planner.

It adds **no** parser change, **no** new spec field, **no** second plan, **no**
second stage vocabulary and **no** second engine. The stage-transition engine,
React, PPTX, OCC, Annex 2, funded analytics, forecast, concentration and cohort
are untouched — verified by diff (§11).

### What was tested

Five instruments, each answering a different question. All were run at **both**
SHAs so every number below is a before-and-after, not a standalone score.

| # | instrument | what it establishes | result |
|---|---|---|---|
| 1 | authoritative **166**-question bank (`BANK75` + frozen `CFO91`) | the shipped agent did not get worse | **0 of 166 questions moved** — every answer, route and verdict byte-identical |
| 2 | **stage-movement** bank, 9 business questions × 4 formulations = 36 | the new ability works | **0 → 36 correct**; 12 wrong → **0 wrong** |
| 3 | **near-neighbour** bank, 13 questions | nothing was hijacked | **13/13 kept their own route owner** |
| 4 | targeted unit/integration tests | the delegation and the refusals are pinned | **26 passed** (+16 subtests) |
| 5 | `MI_REGRESSION_MANIFEST.txt`, 278 files | nothing else broke | §12 |

Instruments 1–3 were additionally run on **both arms** — engine alone and with
the concept-merge language layer — because the shipping record distinguishes
them and because a capability that only works with a model in front of it is a
different claim from one that works without.

### What was NOT done

* **The broad "summarise pipeline stage movement" question is excluded**, by
  decision. It cannot be bound without widening generic movement semantics that
  pipeline evolution and period movement already own (§5, §10).
* **The shipping 136/166 absolute score was not re-measured**, because the book
  it was measured on is not in this repository. Per-question continuity was
  measured instead, which is the stronger statement (§2).
* **No pre-existing test failure was fixed.** One genuine new failure was found
  and fixed; everything else in §12 predates this branch.
* **A parser defect was found and left open** — a refusal that makes a false
  claim about the client's data. It is worked around only where this route can
  prove it accounted for the span (§10).

---

## 1. Executive verdict

**PASS.**

The three conditions the sprint set are separately measured and all three hold.

| | measured | required | |
|---|---|---|---|
| A · authoritative 166-question bank | **0 of 166 questions moved** on either arm — every answer, route and verdict byte-identical | no previously correct question lost | ✅ |
| B · stage-movement bank (36) | **36 correct, 0 wrong, 0 declined — 100%** on both arms | ≥80%, preferably ≥85%, 0 silently wrong | ✅ |
| C · near-neighbour bank (13) | **13 of 13 kept their own route owner — 100%** on both arms | 100% route preservation | ✅ |

**Both arms** means both configurations the shipping record distinguishes: the
governed engine alone, and the same banks with the **concept-merge language
layer** on (`claude-opus-5`). §2 explains why the language layer is that arm and
not the free-form parser.

One qualification is stated up front rather than buried: the change is **431
executable lines** of new adapter plus **29** across three existing files, which
exceeds the ~250-line budget in §19. §11 of this report explains exactly where
those lines are and why the routing part of the work is 29 lines while the
answer-composition part is 125. Nothing was spent on a parser change, a spec
field, a second plan or a second engine — there are none.

---

## 2. Starting Query baseline

### What the shipping record says

`migration_phase0/MI_ACCEPTANCE_BANK_ANSWERS.md` and
`migration_phase0/MI_FREEZE_TAG.md` record the frozen 166-question result at
production SHA `23804de`:

| verdict | with the language layer | governed engine alone |
|---|---:|---:|
| CORRECT | 136 | 127 |
| CORRECTLY DECLINED | 16 | 16 |
| DECLINED BUT ANSWERABLE | 12 | 19 |
| WRONG | 2 | 4 |
| **total** | **166** | **166** |

### Why that score could not be re-measured, and what was measured instead

Both figures were produced against `client_001/mi_2026_06` — **a 640-loan
onboarded book that is not in this repository**. `MI_FINAL_LIVE_DATA_READINESS
.json` names it (`640 loans at 30 June 2026, GBP 172,055,547`, five snapshots at
520/545/570/600/640) and every probe that reads it — `data_claim_audit`,
`completeness_calibration`, `concept_vocabulary_census`,
`must_refuse_both_arms` — takes it from `MI_COMPLETENESS_FIXTURE`, defaulting to
`/tmp/cfo_env`, an ephemeral path that no committed script rebuilds. The 75
bank's oracle is a set of figures computed from that tape (`count = 144`,
`count = 45`), so those verdicts do not transfer to any other book.

Reporting a re-scored 136 would therefore have been an assertion, not a
measurement. Two things were done instead.

**A reproducible rig.** `scripts/run_mi_query_stage_movement_banks.py` builds a
seeded funded tape across the same five month-end runs at the same row counts,
points `MI_AGENT_PIPELINE_ROOT` at the committed governed pipeline fixture,
refuses to run if the free-form LLM parser arm is live, and posts all 166
questions to the production `/mi/query` app. It is the same construction the
committed `migration_phase0` probes use.

**The stronger claim, measured.** On that rig the acceptance criterion is not an
absolute score but **per-question continuity**, and continuity is a stronger
statement than a matching total: a bank can hold 136 while swapping which 136.

Starting baseline on the rig, at `30a7d4a`:

| bank | verdict | count |
|---|---|---:|
| CFO91 (frozen, tape-independent `expect`/`must`/`must_not`) | CORRECT | 63 |
| | TRUE_REFUSAL (correctly declined) | 16 |
| | FALSE_REFUSAL | 11 |
| | NO_COMPUTABLE_TRUTH | 1 |
| | **WRONG** | **0** |
| BANK75 (oracle is tape-specific) | DELIVERED | 44 |
| | DECLINED | 31 |

The CFO half is graded by `migration_phase0.pack_grader.grade_cfo` against the
frozen `CFO_ACCEPTANCE_BANK.yaml` — assertions about the answer's shape and
vocabulary, which do transfer. Its **16 correctly-declined** matches the
shipping record's 16 exactly. The 75 half is recorded as delivered/declined and
its non-regression is established question by question rather than by a verdict
this rig cannot honestly compute.

### The two arms, and which one the shipping record is

The record's two columns are two **arms**, and the difference between 136 and
127 is what the language layer recovers. The language layer is the
**concept-merge arm** — the model proposes concepts in registered vocabulary and
the REGISTRY binds them — and **not** the free-form parser: `datasets
._mi_llm_config` withdraws that one from serving unconditionally, because it
emits a whole governed `MIQuerySpec` and thereby owns the semantics every
downstream guard reads. `MI_AGENT_LLM_PARSER=on` is still recorded as
`requested` and still reaches nothing.

Both arms were therefore measured, at both SHAs, with `--concept-merge --model
claude-opus-5` for the second:

| | engine alone | | language layer | |
|---|---:|---:|---:|---:|
| | `30a7d4a` | branch | `30a7d4a` | branch |
| CFO91 CORRECT | 63 | 63 | 63 | 63 |
| CFO91 TRUE_REFUSAL | 16 | 16 | 16 | 16 |
| CFO91 FALSE_REFUSAL | 11 | 11 | 11 | 11 |
| CFO91 NO_COMPUTABLE_TRUTH | 1 | 1 | 1 | 1 |
| CFO91 **WRONG** | **0** | **0** | **0** | **0** |
| BANK75 DELIVERED | 44 | 44 | **49** | **49** |
| BANK75 DECLINED | 31 | 31 | 26 | 26 |

The language layer lifts BANK75 delivery 44 → 49 **on both SHAs equally** — it is
the arm doing its documented job, and this change neither adds to it nor takes
from it.

**One grade moved and it is not this change.** Under the language layer, CFO38
(*"For loans with LTV above 50%, balance by region"*) was CORRECT at `30a7d4a`
and came back as a refusal on the branch — *"I could not complete the
language-understanding step for this question"*. That is
`mi_service._enforce_model_availability`: the estate's rule that an augmentation
call which did not happen must refuse rather than quietly execute the narrower
deterministic reading. It is an API availability event, not a semantic one, and
two things establish it:

* `stage_movement_query.read()` returns `None` for that question — it names no
  governed stage, so this route cannot claim it, and the route is `None` on both
  sides;
* re-run on the branch three times, it is **correct all three**, byte-identical
  to `30a7d4a`.

Counting it as a regression would be counting a transient model call against the
diff. The honest figure is 63 on both sides of both arms.

### What the language layer does NOT fix

The stage-movement bank at `30a7d4a` **with the language layer on** scores
exactly what it scores without it: **0 correct, 12 wrong, 24 declined**. The
twelve stock-for-transition substitutions are identical. That is the strongest
single argument for this work: the gap was never a language-understanding
problem, and no amount of concept recovery closes it, because the governed
transition figures were not reachable from the Query path at all.

---

## 3. Existing architecture used

The current production shape was re-confirmed against `30a7d4a` rather than
assumed:

```
POST /mi/query
  → mi_service.execute_governed_mi_query
      → ParsedQuestion.parse                      ONE parse, shared
      → workspace.resolve_dataset
      → concept_merge_arm                         (off by default)
      → chat_routing.try_route
          → RouteRequest                          one object, every input
          → recogniser_registry.REGISTRY          ordered, deterministic
          → the first handler returning an envelope
      → execution_receipt / completeness guards
      → the point-in-time engine where nothing routed
```

Stage movement joined it as **one more `Recogniser` in that registry** —
`mi_agent_api/stage_movement_query.recogniser()`, registered in
`chat_routing._register_default_recognisers` alongside the other twelve.

| the brief forbade | what was done |
|---|---|
| a parallel stage-movement parser | none. Stage spellings come from `question_interpretation.lexical.pipeline_stage_vocabulary`, the estate's one question-side stage reader, itself derived from `pipeline_prep._STAGE_CANON` |
| a stage-movement mini-agent | none. One `recognise`, one `handle` |
| a second recogniser framework | none. `Recogniser`/`RouteRequest`/`Recognition` |
| a new question schema | none. **Zero new spec fields.** The pre-claim reading travels from `recognise` to `handle` through `RouteRequest.remember_recognition`, the seam `period_change` already uses |
| a new capability registry | none |
| a new execution engine | none. `resolve_stage_transition_detail` |
| a separate `StageMovementPlan` | none |

**On §10 — could the existing representation express it?** It could not, and it
did not need to. `spec.filters` never carries `pipeline_stage` (0 of 882 corpus
questions — `analytical_plan.governed_stage_step` records why), and the
interpretation contract carries **one** stage, not an ordered pair with a
movement subtype. Rather than widen the spec for one route, the reading is
carried on the registry's own per-request memo, which is what that memo exists
for. Nothing else in the estate can see it, so nothing else can be changed by it.

---

## 4. Implementation

### `mi_agent_api/stage_movement_query.py` — new, 744 lines (431 executable)

**Recognition.** A question is stage movement only when it names governed stages
in an explicit movement construction:

* **transition** — two *distinct* governed stages in text order with the
  direction **explicit**, in exactly one of three ways: a transition verb
  anywhere (`moved`, `progressed`, `transitioned`, `went`, `advanced`,
  `migrated`); a strong connector between them (`into`, `->`, `→`, `onto`,
  `through to`, `reached`); or the `from X to Y` frame. A bare `to` is
  deliberately not enough — *"compare KFI numbers to Offer numbers"* reads
  identically and is a comparison, which `temporal_compare` owns;
* **new arrival / stayer / departure / reconciliation** — exactly one governed
  stage plus an explicit arrival, stayer, departure or reconciliation word.

Generic *movement* and *change* are never discriminators: the funded bridge,
period change and period movement own them and keep them. Conversion, forecast,
projection, expectation, scenario, trend, evolution, cohort, vintage and funnel
wording is declined outright, before any stage is read.

**Measure binding.** *how many / number of / count* → case count; *balance /
amount / value / how much* → the governed monetary field; a change word on a
stayer question → the governed `amount_change`, never a stock.

**Delegation.** One call to `movement_detail.resolve_stage_transition_detail`,
with no `as_of`. The request's as-of is the funded reporting cut-off
(30 June 2026); the resolver's is a weekly pipeline extract date (12 June 2026).
Passing one for the other made `select_pair` match no extract — measured, and
the reason the first cut refused every question. Omitting it asks for the
capability's own latest governed pair, and every answer states that window.

**Composition.** `compose(reading, payload, money=…)` returns
`(answer, rows, refusal)`. Every figure is a key lookup on the governed payload;
the only arithmetic is rendering a float as currency.

**Evidence the envelope declares.** `metadata.populationApplied` names the
governed narrowing on `pipeline_stage`; `reconciliation.dataset = "pipeline"`
names what the answer was reconciled against; `metadata.stageMovement` carries
the subtype, the stages, the window, the identifier and the methodology version.

**One span-accounting rule.** Reading *"How many cases moved from Offer to
Completion?"*, the parser proposes `offer to completion` as a categorical value,
finds none, and records `unknown category: 'offer to completion'`; the routed
guard then refuses with *"No loans in this book match that filter ('offer to
completion')"* — a **false statement about the client's data**, about a filter
the reader never asked for. `migration_phase0/data_claim_audit.py` classifies
exactly this shape as `QUOTES_A_MANGLED_PHRASE`. This route bound that span as a
governed source and destination and answered from it, so the note is dropped —
**only** where every alphabetic word in it belongs to the construction this route
recognised, and the dropped notes are published as evidence. A broker, region or
product the reader named keeps its note and still refuses.

### `mi_agent_api/chat_routing.py` — +15

One import and one registry entry, at **priority 120 on `DEFAULT_CONFIDENCE`** —
last, so every existing recogniser that also matches wins by the registry's own
ordering. Deference is structural, not a rule this module asserts.

### `mi_agent/execution_receipt.py` — +31

Two registrations and one alignment.

* `pipeline_stage_movement` joins `TEMPORAL_ROUTES`. It is a two-snapshot
  capability by construction — without a prior snapshot it returns the governed
  "no comparison" refusal — so a comparison-period facet on its questions is
  honoured rather than lost.
* A human label in `_ROUTE_LABELS`.
* **Narrowed-to is not lost.** A `KIND_GROUPING` facet fell through to *"this
  answer covers the whole population; it is neither narrowed to nor broken down
  by X"* even where the route had declared, in `metadata.populationApplied`, that
  it narrowed on exactly that field. The sentence was false, and it refused
  *"Reconcile Application stage this period."* over an answer about one governed
  stage. `question_interpretation.completeness._carried` has always read the
  population ledger for this same facet kind (`field in applied`, where `applied`
  includes `declared_population_fields`); two readers of one piece of evidence
  disagreed, and no longer do. `KIND_RANKING` is deliberately excluded — a
  ranking needs an axis to order, and one value of a field is not one.

### `mi_workflows/analytical/planner.py` — +26

`_plan_pipeline_offer_outlook` fired on *pipeline family + offer + completion*,
which is also the shape of the backward question, and answered *"How many cases
moved from Offer to Completion?"* with **a forecast** — *"Offer stage pipeline is
£1.3m across 2 cases. Expected completion amount: £968k"* — for a question whose
governed answer is one case and £800k already completed. It now asks the owning
route's own reader whether that route owns the question, exactly as
`_evolution_route_owns` and `_forecast_route_owns` beside it already do.

---

## 5. Stage-movement bank

`tests/fixtures/mi_query_stage_movement/STAGE_MOVEMENT_BANK.yaml` —
**9 business questions × 4 natural formulations = 36**, frozen before the
recogniser existed, with an oracle that is arithmetic on the committed
two-snapshot fixture.

| id | business question | subtype | variants | governed answer |
|---|---|---|---:|---|
| SM01 | How many cases moved from KFI to Application? | transition, count | 4 | 2 cases |
| SM02 | How much balance moved from Application to Offer? | transition, amount | 4 | £1.29m at Offer (£1.30m at Application) |
| SM03 | How many cases moved from Offer to Completion? | transition, count | 4 | 1 case |
| SM04 | How much balance moved from Offer to Completion? | transition, amount | 4 | £800k |
| SM05 | How many new cases entered KFI? | new arrival | 4 | 1 case, £900k |
| SM06 | How many cases stayed in Application? | stayer, count | 4 | 1 case |
| SM07 | What was the amount change on cases that stayed in Application? | stayer, amount change | 4 | −£20k (£300k → £280k) |
| SM08 | Where did cases leaving Offer go? | departure | 4 | 1 to Completion, 1 unevidenced |
| SM09 | Reconcile Application stage this period. | reconciliation | 4 | 4 + 1 + 2 − 2 − 1 = 4 |

Variants exercise real user language, not mechanical synonyms: *moved /
progressed / transitioned / went / advanced*, *new / entered / arrived*, *stayed
/ remained / persisted*, *balance / amount / value*.

**Q10 — the broad movement summary — was considered and EXCLUDED**, as §4 of the
brief permits. "Summarise pipeline stage movement this period" cannot be bound
without widening generic movement semantics that pipeline evolution and period
movement already own. The brief's own rule is that a safe bank with honest
declines beats a complete one bought by widening, and NN01–NN04, NN10 and NN12
exist to keep that boundary where it is.

---

## 6. Stage-movement results

| | `30a7d4a` engine | branch engine | `30a7d4a` language layer | branch language layer |
|---|---:|---:|---:|---:|
| CORRECT | 0 | **36** | 0 | **36** |
| WRONG | **12** | **0** | **12** | **0** |
| honest decline | 24 | 0 | 24 | 0 |
| other | 0 | 0 | 0 | 0 |
| **success rate** | **0.0%** | **100.0%** | **0.0%** | **100.0%** |

The two arms give the same answer in both directions, which is the point: the
capability is delivered by the deterministic path and does not depend on a model
being in front of it, and the twelve substitutions were not a language failure.

The twelve baseline WRONGs are the defect this sprint closes. Every one is a
**current stage stock standing in for a transition**:

```
"How many cases went from KFI into Application?"
  30a7d4a  3 loans · £1.2MM · Pipeline Stage = KFI      ← the KFI STOCK
  branch   2 cases moved from KFI to Application between 2026-06-05 and 2026-06-12
```

```
"How many cases stayed in Application?"
  30a7d4a  4 loans · £1.4MM · Pipeline Stage = APPLICATION   ← the APPLICATION STOCK
  branch   1 case stayed at Application between 2026-06-05 and 2026-06-12
```

Four more (SM03C/D, SM04C/D) were answered with the *expected completion
forecast* — a forward figure for a backward question — which the planner
deference in §4 removes.

100% is above the preferred ≥85%, and it was **not** bought by widening: the
recogniser declines every near neighbour (§7), and the two remedies that lifted
the last five questions were a false refusal about the client's data and a
false "covers the whole population" claim, both fixed by reading evidence the
envelope already carried.

---

## 7. Near-neighbour results

`tests/fixtures/mi_query_stage_movement/NEAR_NEIGHBOUR_BANK.yaml` — 13 questions.
**Route preservation 13/13 (100%). Zero hijacked. Zero verdict changes.**
Measured on both arms, with the same result on each: the boundary holds whether
or not the language layer is recovering concepts.

| id | question | owner at `30a7d4a` | owner on branch |
|---|---|---|---|
| NN01 | Show pipeline amount by stage. | point-in-time | point-in-time |
| NN02 | How much pipeline is currently in Offer? | point-in-time | point-in-time |
| NN03 | What is pipeline by stage? | point-in-time | point-in-time |
| NN04 | Show pipeline evolution. | `evolution` | `evolution` |
| NN05 | How has pipeline balance changed this month? | point-in-time (declines) | point-in-time (declines) |
| NN06 | What is the conversion rate? | `cohort_conversion` | `cohort_conversion` |
| NN07 | How has conversion changed? | `cohort_conversion` | `cohort_conversion` |
| NN08 | What is funded balance movement? | `period_change_analysis` | `period_change_analysis` |
| NN09 | Why did funded balance increase? | `period_change_analysis` | `period_change_analysis` |
| NN10 | Show movement by region. | `funded_bridge` | `funded_bridge` |
| NN11 | What is forecast funded balance? | `analytical_composition` | `analytical_composition` |
| NN12 | Show weekly pipeline cases. | `evolution` | `evolution` |
| NN13 | What is the KFI to Offer conversion rate? | `analytical_composition` | `analytical_composition` |

NN13 is the sharpest of these: two governed stages, a directional connector, and
still not stage movement — the reader declines it on the word *conversion*
before it looks at a stage. `test_conversion_is_not_stage_movement` pins it.

---

## 8. Main-bank non-regression

| | starting `30a7d4a` | final branch | Δ |
|---|---:|---:|---:|
| CFO91 CORRECT | 63 | 63 | 0 |
| CFO91 TRUE_REFUSAL | 16 | 16 | 0 |
| CFO91 FALSE_REFUSAL | 11 | 11 | 0 |
| CFO91 NO_COMPUTABLE_TRUTH | 1 | 1 | 0 |
| CFO91 **WRONG** | **0** | **0** | **0** |
| BANK75 DELIVERED | 44 | 44 | 0 |
| BANK75 DECLINED | 31 | 31 | 0 |

**Per-question movements: none.** All 166 questions are byte-identical in
answer, route and verdict:

```
$ python scripts/run_mi_query_stage_movement_banks.py --out after --compare before
   bank_166         0 question(s) moved
   bank_stage       36 question(s) moved
   bank_neighbours  0 question(s) moved
```

Against the acceptance conditions:

* previously CORRECT questions lost — **0**
* new WRONG questions — **0**
* previously CORRECT questions that became declined — **0**
* correctly-declined set deterioration — **none** (16 → 16)
* route ownership changes on unrelated questions — **none**

---

## 9. Delegation proof

Four independent proofs, three of them enforced by tests rather than asserted
here.

1. **The call happens, exactly once.**
   `test_query_calls_the_governed_resolver` wraps
   `movement_detail.resolve_stage_transition_detail` and asserts one call per
   answered question.

2. **The figures follow the payload.**
   `test_query_performs_no_stage_transition_arithmetic` hands `compose` a payload
   whose KFI→APPLICATION `case_count` is 99. The answer says 99. It could not, if
   the adapter recomputed anything.

3. **The adapter's own source cannot compute.**
   `test_the_module_reads_no_snapshot_and_owns_no_stage_table` parses the module,
   strips every string literal (docstrings *name* what it defers to), and asserts
   the remaining code contains no `pandas`, `read_csv`, `load_prepared_pipeline`,
   `weekly_extract_inventory`, `_STAGE_CANON`, `groupby` or `merge(`.

4. **The same governed result on every channel.** The route calls the resolver
   `mi_agent_api/app.py` calls for `/mi/insights/movement-detail` and
   `mi_agent_pptx/deck.py` calls in-process — same inventory, same
   `load_prepared_pipeline`, same `select_pair` neighbour rule, same cached
   frames. Query is the third consumer of one computation, not a fourth
   computation.

**Confirmed no stage-transition arithmetic in Query.** Query performs no
snapshot join, no case matching, no stage comparison, no arrival/departure/stayer
counting, no amount amendment and no reconciliation. It selects by key and
renders.

---

## 10. Failure analysis

**No stage-movement question is unrecovered.** All 36 are correct.

Two limits are worth recording anyway, because both are real and neither is
closed by this work.

**The broad movement summary is out of scope by decision, not by failure.**
"Summarise pipeline stage movement this period" is not in the bank and is not
recognised. Recognising it would require generic movement wording to select this
route, which is what NN04, NN08, NN10 and NN12 exist to prevent. If it is wanted
later it needs its own discriminator, not a widening of this one.

**Two parser-level defects were met and only narrowly worked around.** Both are
about the parser proposing a categorical value out of a span that is not one:
`unknown category: 'offer to completion'`, `unknown category: 'new'`. The refusal
they produce (*"No loans in this book match that filter ('offer to completion')"*)
is a false claim about the client's data, and it will still fire for any other
route whose construction the parser mangles the same way. This sprint fixed it
only where this route can prove it accounted for the span. **The general defect
remains open** and is a candidate for its own piece of work.

---

## 11. Production LOC / scope — and why the budget was exceeded

| file | status | lines added | executable |
|---|---|---:|---:|
| `mi_agent_api/stage_movement_query.py` | new | 744 | 431 |
| `mi_agent_api/chat_routing.py` | existing | 15 | 15 |
| `mi_agent/execution_receipt.py` | existing | 31 | 8 |
| `mi_workflows/analytical/planner.py` | existing | 26 | 6 |
| **production total** | | **816** | **460** |
| `mi_agent_api/tests/test_stage_movement_query.py` | new test | 430 | |
| `scripts/run_mi_query_stage_movement_banks.py` | new evidence rig | 334 | |
| `tests/fixtures/mi_query_stage_movement/*.yaml` | new banks | 224 | |

Zero lines removed anywhere. Nothing existing was rewritten.

**§19 fires: 460 executable lines against a ~250 budget.** Where they are:

| | executable |
|---|---:|
| routing, registration and deference *into existing files* | **29** |
| module-level movement vocabulary and identifiers | 45 |
| recognition — the narrow discriminator | 96 |
| **deterministic answer composition (§12)** | **125** |
| delegation, envelope and evidence declarations | 78 |
| governed payload selectors | 26 |
| span accounting | 36 |
| table artifact | 25 |

The earlier 250–360 estimate was of a **routing extension**, and the routing
extension came in at **29 lines**. What dominates is §12 — deterministic answer
composition — because the governed capability publishes **five** event classes
and the brief asks each to be answered in the reader's own terms, in counts and
in money, always stating the reporting window and never exposing a payload key.
Five sections × two or three measure phrasings, each with its own refusal branch
for an absent stage, is 125 lines and does not compress without either dropping
subtypes from the bank or emitting a generic sentence per section.

A trim pass was run and committed (`5e253b1`): a dead constant and a 22-line
column-label table replaced by a derived label and format, ~24 lines. Beyond
that, further reduction costs answer quality rather than complexity, so it was
not taken.

**None of the four abort-condition triggers is present**: no parser
restructuring, no second analytical calculation in Query, no snapshot inspection
in Query, no widening of generic movement semantics.

### Production change audit

| expected area | touched |
|---|---|
| existing Query parser/recognition vocabulary | ✅ read-only — `pipeline_stage_vocabulary` is *called*, not changed |
| existing Query spec/plan representation | ✅ **not touched** — zero new spec fields |
| existing recogniser registration | ✅ `chat_routing` +15 |
| existing executor dispatch | ✅ via the registry; no dispatch change |
| deterministic answer adapter | ✅ the new module |

| expected untouched | status |
|---|---|
| stage-transition engine (`movement_detail.py`) | **unchanged** |
| movement_detail analytics | **unchanged** |
| React (`frontend/`) | **unchanged** |
| PPTX (`mi_agent_pptx/`) | **unchanged** |
| OCC, Annex 2, onboarding | **unchanged** |
| funded analytics, forecast, concentration, cohort | **unchanged** |
| `mi_agent/mi_query_spec.py`, `parsed_question.py`, `llm_query_parser.py` | **unchanged** |

**Two deviations from the expected-areas list, both declared:**

* `mi_agent/execution_receipt.py` (+31, 8 executable) — two route registrations
  and the grouping/population-ledger alignment in §4. Not in the brief's list; it
  is where the false *"covers the whole population"* sentence was written, and
  the alignment matches a reading `question_interpretation.completeness` already
  performs on the same facet kind.
* `mi_workflows/analytical/planner.py` (+26, 6 executable) — the deference in §4,
  written in the idiom two neighbouring functions already use.

Both were verified by the zero-movement 166 run.

---

## 12. Regression

| suite | result |
|---|---|
| stage-movement bank (36) | 36 CORRECT · 0 WRONG · 100% |
| near-neighbour bank (13) | 13/13 own route · 0 hijacked |
| authoritative 166 bank | 0 questions moved vs `30a7d4a` |
| `mi_agent_api/tests/test_stage_movement_query.py` | **26 passed**, 16 subtests passed |
| `mi_agent_api/tests/test_recogniser_registry.py` | passed |
| `mi_agent_api/tests/test_pipeline_stage_transition.py` | passed |
| `mi_agent_api/tests/test_stage_transition_exposure.py` | passed |
| the three above, together | **96 passed** |
| `migration_phase0/MI_REGRESSION_MANIFEST.txt` (278 files) | see below |

<!-- REGRESSION-MANIFEST -->

---

## 13. Recommendation

**MERGE.**

The three conditions the sprint set are met and measured separately: existing
Query quality does not fall (0 of 166 questions moved), the new capability works
at a rate above the agent's own (100% against a ≥85% preference), and no near
neighbour is hijacked (13/13). Twelve silent stage-stock substitutions are gone
and none was created.

The one thing a reviewer should weigh is scope: 460 executable production lines
against a ~250 budget, of which 29 are routing and 125 are the deterministic
answer composition the brief itself requires. §11 sets out the split.
