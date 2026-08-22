# B21 — a disclaimed view word must not choose the frame

Written before implementing.

Base: HEAD `4b30fe3`; merge-base `4e051f3`; clean tree.

---

## 1. WHAT MOVES IF THIS IS WRONG — and whether anything can see it

Carried into this work order because the surface named for B22 reported the live
defect as **FIXED**: it asserted route, verdict and facet kinds, and none of
those moves when a population silently drops to 3,909. **That was the fourth
instrument found inadequate by the change it was meant to measure.** So this
section comes before the design, not after the result.

### 1.1 The two ways B21 can be wrong, and what each would do

| failure | consequence | what moves on THIS book |
|---|---|---|
| **too aggressive** — a genuine forecast/pipeline question stops selecting its view | the question is answered against the **funded** frame | **nothing.** The forecast frame and the funded book agree to the penny here (below), so the number is identical, and the route is reached the same way |
| **wrong view** — pipeline selected where funded was meant, or the reverse | the question is answered against the wrong frame | **nothing.** This book carries no pipeline, so every pipeline-view question returns the same sentence — *"No governed pipeline data is available for the pipeline view"* — whatever the intent was |

**Both failure modes are silent on this tape.** That is not a hypothetical: it is
the same coincidence that makes B21 lower-severity here and higher-severity on a
real portfolio.

### 1.2 Nothing in the estate asserted the view

Measured, across the whole repository:

```
tests/test_analytical_capability_layer.py:557   "datasetContext": "funded"    <- request
tests/perf/measure.py:214                       "datasetContext": "funded"    <- request
tests/perf/parity.py:112                        "datasetContext": "funded"    <- request
question_interpretation/answer_diff.py:123      "datasetContext": "funded"    <- request
question_interpretation/run_robustness_deterministic.py:124                   <- request
```

**Every occurrence is a parameter being SENT. Not one is a check on what came
back.** So a wrong view was unfalsifiable by construction — not merely unmeasured
but unmeasurable.

### 1.3 What was done about it, before writing the fix

`routed_surface.observe` now carries `metadata.datasetContext`, `check` asserts
it through `expect_view`, and the self-test's can-fail probes cover it.

**`rt_030` and `rt_031` are the proof that the assertion was necessary.** They
are opposite intents —

```
rt_030  "the balance by seasoning segment EXCLUDING pipeline cases"   view=pipeline
rt_031  "How much pipeline is overdue?"                               view=pipeline
```

— and today they return **byte-identical** answers. Nothing in `ok`, the verdict,
the facets, the population, the filters or the answer text distinguishes them.
**Only the view does.**

## 2. The severity rests on a coincidence of this book, and the proof cannot come from this tape

```
no pipeline   funded = 1,964,886,258.21   forecast view = 1,964,886,258.21   diverge = 0.00
```

`build_forecast_view_frame` puts the forecast CONTRIBUTION into
`current_outstanding_balance` — the same column name, a different meaning. With
no pipeline the contribution IS the funded balance, so they coincide exactly.

Constructed, with three pipeline cases carrying weighted expected amounts:

```
with pipeline funded = 1,964,886,258.21   forecast view = 1,965,656,258.21   diverge = 770,000.00
rows 11,035 -> 11,038   state_component: funded 11,035, forecast_pipeline 3
```

**£770,000 under the same field name, disclosed nowhere.** On this tape, £0.00.

That divergence is the whole severity argument and **this book cannot show it**.
It is demonstrated on a constructed frame instead, and the evidence is
constructed — stated here rather than implied.

## 3. A correction to the diagnosis: the doctrine does NOT transfer

`mi_b21_b22_b23_diagnosis.md` said *"For B21 the doctrine transfers and the
vocabulary does not."* Measuring the pipeline vocabulary shows the first half is
wrong too.

B22's doctrine is *the word must QUALIFY a book noun* — `acquired` is an
adjective, and "the acquired book" names a book while "the borrower acquired the
property" does not. But:

```
"pipeline amount by stage"          pipeline qualifies a noun
"how many cases are in the pipeline"  pipeline IS the noun
"Which broker has the largest pipeline?"   pipeline IS the noun
"How much pipeline is overdue?"     pipeline IS the noun
"What is the forecast?"             forecast IS the noun
```

**`forecast` and `pipeline` are nouns naming the subject, not qualifiers naming a
scope.** Requiring them to qualify something would reject half the corpus's own
pipeline family. So the qualified-mention test does not apply here, and the
shared component is the **second** one: the disclaiming guard.

That makes B21 a smaller fix than the diagnosis implied, and the smaller claim is
the honest one: **B21 closes the disclaiming class, and the broader
"mentions-versus-about" distinction for a word that IS the subject is not settled
by it.**

## 4. The rule

One helper, now three callers. `_qualified_span_re` stays as it is —
parameterised by vocabulary, which is what stopped five governed phrases being
dropped in B22 — and the **disclaimer** pattern is parameterised the same way:

```
disclaimer + (up to a short gap) + a word from the caller's vocabulary
```

* `portfolio_lens`'s existing disclaimer test keeps the scope vocabulary.
* `resolve_active_view` gets the same test over `forecast` / `pipeline` /
  `funded`.

A view word every one of whose occurrences is disclaimed does not select. One
occurrence undisclaimed is enough to select, because *"the forecast excluding
pipeline"* is a forecast question.

**B24 is not opened here.** Whether `resolve_active_view` should run before
parsing at all remains recorded and unsettled.

## 5. Pre-registered prediction

### 5.1 What moves

| id | today | predicted |
|---|---|---|
| `rt_028` balance by vintage, ignoring the forecast | view `forecast`, `vintage_year: field_missing` | **view `funded`, answered, 13 groups** |
| `rt_030` seasoning segment excluding pipeline cases | view `pipeline`, "no governed pipeline data" | **view `funded`, answered** |
| `rt_029` Show forecast balance by region | view `forecast` | **unchanged** |
| `rt_031` How much pipeline is overdue? | view `pipeline` | **unchanged** |

`answer_diff`: **2 moved**, both `routed_surface`.

**Zero of the 697 corpus questions change view.** Measured against the rule
before writing it: every corpus question's view was recorded, the rule was run
over all 697, and none moved. No corpus question disclaims a view word.

### 5.2 What must not move

1. **`rt_029` and `rt_031` keep their views.** They are the can-fails, and on
   this book the view assertion is the only thing that can catch a
   too-aggressive guard.
2. **No corpus answer moves**, and **no corpus question changes view** — the
   stronger of the two, because the answers would not move even if the views did.
3. **The seasoning families stay at their by-name counts**, both books.
4. **Robustness `32/10/2`; calibration `259/259`, 0 hard fails, 0 known gaps.**
5. **No lexical decision moves.** 693 of 693.
6. **`portfolio_lens`'s disclaimer behaviour is unchanged** — B22's four
   disclaiming cases still decline, and its nine governed phrases still select.

### 5.3 Stop conditions

* `rt_029` or `rt_031` changing view;
* any corpus question changing view;
* any corpus answer moving;
* a B22 case moving;
* the divergence test failing to show a divergence on the constructed frame, or
  showing one on this tape.

### 5.4 Acceptance

* the disclaimer test has **one** implementation, parameterised by vocabulary,
  with the lens and the view resolver as its callers;
* `expect_view` is asserted by the surface and covered by its self-test;
* the £770,000 divergence is demonstrated on a constructed pipeline-bearing
  frame, and its absence on this tape is asserted rather than assumed;
* all five surfaces, deterministic arm, both books; seasoning by name;
* the constructed-coverage statement below, applied in the same form as B22's.

## 6. The constructed-coverage statement, in the form B22 established

To be reported verbatim in shape once measured:

> **N of M identical means the fix did not reach the corpora — nothing more.**
> The corpora contain no question that disclaims a view word, so they would read
> the same whether the change were correct, inert, or wrong in a direction the
> constructed cases do not probe. The claim rests on the constructed cases and
> the constructed divergence frame, and the class beyond them is argued rather
> than measured.

---

# MEASURED — appended after implementation, prediction left as written above

## 7. The prediction was right about the rule and wrong about the count

§4 said "one helper, now three callers". **There were four owners, not two, and only the site-by-site pass found the last two.**

| # | owner | what it decides | how it was found |
|---|---|---|---|
| 1 | `workspace.resolve_active_view` | which FRAME the question loads | named in the diagnosis |
| 2 | `chat_routing._dataset_for` | which DATASET a routed answer is built from | **the enumeration**, before any test failed |
| 3 | `mi_workflows.analytical.intent` (`_any`/`_hits`) | the structural REQUIREMENT that makes a question refusable | **rt_030, after 1 and 2 were fixed** |
| 4 | `execution_receipt._PROJECTION_RE` | the requested-projection FACET | **rt_028, after 1 and 2 were fixed** |

Owner 2 was found the way the standing rule intends — by enumerating where the
answer arrives rather than by waiting for a red test. Its word list is *wider*
than the one named in the diagnosis (`case`, `kfi`, `application`, `offer`), so
fixing only `resolve_active_view` would have left the excluding question
narrowed to the very dataset it excluded, by vocabulary nobody had looked at.

**Owners 3 and 4 are the finding worth keeping.** With the frame and the dataset
both correct, rt_028 computed the *right number over the right 11,035 loans* —
Balance, grouped by Vintage, 13 groups — and was then **refused**, because a
fourth reader still raised a forward-projection facet from a forecast word the
sentence had ruled out. The honour-or-clarify guard was working exactly as
designed on a request that had never been made. A fix measured only by "does
the number come out right" would have declared victory two owners early.

This is the same shape the census exists for, one level deeper than the census
recorded it: **"does this question ask for a forecast?" had four independent
readers**, each with its own vocabulary and its own way of locating a hit. Only
the *window* and the *sentence boundary* are common to them, so that is what was
consolidated — `portfolio_lens.is_disclaimed_span` is the primitive, and
`undisclaimed_mention`, the scope-phrase regex, the intent matcher and the facet
raiser all measure the same distance and stop at the same boundary.

## 8. Results against §5

| declared | measured | |
|---|---|---|
| `rt_028` → view `funded`, answered, 13 groups | view `funded`, `verdict=ok`, `grouping_dimension applied`, 13 groups over 11,035 | ✅ |
| `rt_030` → view `funded`, answered | view `funded`, `verdict=ok`, `grouping_dimension applied` | ✅ |
| `rt_029` unchanged (`forecast`) | unchanged | ✅ |
| `rt_031` unchanged (`pipeline`) | unchanged | ✅ |
| `answer_diff` 2 moved, both `routed_surface` | **729 compared, 727 identical, 2 moved — `rt_028`, `rt_030`** | ✅ |
| zero corpus questions change view | **0 of 683**; 113 mention a view word and all 113 hold | ✅ |
| robustness `32/10/2`, both books | `32/10/2` on alderbridge AND kestrelmoor | ✅ |
| seasoning families by name | Q1 4 CORRECT, Q7 4 CORRECT, Q8 12 CORRECT — both books | ✅ |
| calibration `259/259` | 259/259, 0 hard failures, 0 known gaps | ✅ |
| no lexical decision moves | 690 compared, 688 identical; the 2 moves are **pre-existing** — byte-identical before and after B21, verified by stashing | ✅ |
| B22 unchanged | its 9 governed phrases select, its 4 disclaiming cases decline, 33 tests pass | ✅ |
| routed surface | 32 passed, 0 failed, 2 declared defects reported FIXED and converted | ✅ |

One measurement the prediction did not state, added because owner 3 is a broad
change: applying the disclaiming test inside the intent classifier covers **all
fourteen** of its vocabularies, not just the pipeline one. Across 661 corpus
questions × 14 vocabularies, **exactly one question/term pair stops signalling**,
and it is `rt_030` — a constructed case.

## 9. The constructed-coverage statement, in B22's form

> **727 of 729 identical means the fix did not reach the corpora — nothing
> more.** The corpora contain no question that disclaims a view word: 0 of 683
> change view, and 1 of 661 × 14 vocabulary pairs stops signalling, both of them
> cases this work constructed. So the corpora would read the same whether the
> change were correct, inert, or wrong in a direction the constructed cases do
> not probe. The claim rests on the four constructed cases, the constructed
> divergence frame, and the twenty-two tests in
> `test_b21_disclaimed_view.py`. The class beyond them is argued, not measured.

And the severity, stated the same way: **£770,000 is a constructed number.** On
this tape the divergence is £0.00, because the forecast frame reuses
`current_outstanding_balance` for the contribution and with no pipeline the two
agree exactly. Anyone reading "B21 was closed" should understand it as *a defect
that changes no number on this book and every number on a book with a pipeline*,
closed before that book arrives — not as a correction to a figure anyone has
been shown.

## 10. What stays open

* **B24** — whether `resolve_active_view` should run before parsing at all.
  Recorded and unsettled; not opened here. B21 makes it *less* urgent (a
  disclaimed word no longer chooses the frame) and not resolved (an
  *undisclaimed* incidental mention still does — "how does this compare with the
  forecast we ran last quarter" still loads the forecast projection).
* **The mentions-versus-about distinction** for a word that IS the subject. B22's
  qualified-mention doctrine does not transfer here, as §3 measured, so nothing
  currently separates "the forecast" as subject from "the forecast" as passing
  reference. That is B24's territory.

## 11. A fifth inadequate instrument, and this one was the meta-instrument

`test_every_case_states_at_least_one_expectation` exists to catch a routed case
that asserts nothing. Its key list named five expectations. The checker makes
**eight**: `expect_population` and `expect_filters` arrived in B22,
`expect_view` in B21, and none was added to the guard — so `rt_029`, whose only
assertion is the view, read as *a case asserting nothing at all*.

The guard was doing its job correctly on a list that had fallen two releases
behind. It is now **derived from `routed_surface.check` by inspection** and
asserted equal to it, so it cannot fall behind again.

That is the fifth instrument in this programme found inadequate by the change it
was meant to measure, and the first inside a meta-instrument — a test whose whole
job is to notice unmeasured cases was itself unmeasured. The pattern is now
consistent enough to state as a rule rather than a tally: **when a surface gains
an assertion, the guards over that surface are part of the surface.**

## 12. The rest of the estate

`question_interpretation/tests/ mi_agent/tests/ mi_agent_api/tests/ mi_workflows`
— **61 failures before this work and the identical 61 after**, compared by name
in the same tree with the same data (`comm` over both sorted lists: zero new,
zero fixed). They are pre-existing and environment-dependent — risk-limit
documents, deck artefacts, pipeline snapshot folders, funded central-tape
resolution. `test_every_case_states_at_least_one_expectation` was among the 61
and is now fixed, taking the count to 60.

The worktree comparison at `61d8956` was discarded rather than quoted: it
reported 20 failures because 306 tests SKIPPED there for want of the demo data,
which would have made B21 look like it fixed forty things. Recorded because it is
the same error the pack warns about one level up — **a clean result that is
evidence about the harness before it is evidence about the product.**
