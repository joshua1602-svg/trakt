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
