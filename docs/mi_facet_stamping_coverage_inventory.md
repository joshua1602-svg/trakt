# Facet stamping coverage inventory

Inventory only. Nothing is fixed here, and no design is proposed. Where the
brief asks for a fix shape it is recorded, unapplied.

## 0. Base confirmation

```
HEAD                8a084c9  Measure three variants of the unresolved-role default
merge-base HEAD origin/claude/mi-analytical-capability-layer-vlkjfw
                    4e051f36a31b5e135854c750fb132bc15d6db861
4e051f3             IS an ancestor of HEAD
28ece25             IS an ancestor of HEAD
```

Deterministic arm, both books by name, run against the tree this inventory
describes:

```
robustness   alderbridge  32 CORRECT / 10 SAFE_REFUSAL / 2 CWDL
             kestrelmoor  32 CORRECT / 10 SAFE_REFUSAL / 2 CWDL
             seasoning    Q1 4, Q7 4, Q8 12 — 20 of 20 CORRECT, both books
             agreement    44 / 44
answer text  340 of 340 identical
```

**Both surfaces are clean, and three ordinary questions about the front and back
book refuse on the shipped tape (§3).** That is the single most important line in
this document: the measurement regime this programme has relied on for eight
commits cannot see the defect it is measuring.

---

## 1. The route inventory

`mi_agent_api/mi_service.execute_governed_mi_query` is the one entry point every
real caller uses. It tries `chat_routing.try_route` first. A route that claims
the question answers it and the envelope is adjudicated by
`_guard_routed_answer` → `reconcile_routed_facets`. A question **no route
claims** falls through to `run_mi_agent_query` → `reconcile_facets`.

So there are exactly two adjudicators, and the routing decision selects between
them:

| | point-in-time | routed |
|---|---|---|
| entry | `run_mi_agent_query` (workflow) | `chat_routing.try_route` → `_guard_routed_answer` |
| adjudicator | `reconcile_facets` | `reconcile_routed_facets` |
| runs the role split? | **yes** — `_split_named_dimension_roles` is called on its first line, and that is its only call site | no |
| population ledger? | no | yes — `mi_service` appends `population_facets(spec)` and stamps them with `reconcile_population` |
| unstamped facet | keeps `LOST` (the `RequestedFacet` default) | keeps `LOST` |
| effect of `LOST` | blocks for every kind in `NUMBER_OR_SUBJECT_FACETS`, plus population, grouping and ranking | same |

There are 16 route identities in total. All 15 routed ones share one
adjudicator, parameterised by route name, so the matrix has 16 rows but two
distinct mechanisms.

A second entry point exists and matters for the measurement, not the product:
`mi_calibration._run` calls `run_mi_agent_query` **directly**, bypassing routing
entirely. The calibration bank is therefore always point-in-time; the robustness
bank goes through `/mi/query` and is mostly routed. Neither exercises the same
mix a real caller sees.

**Nothing is silently ignored anywhere.** Every unstamped facet of a blocking
kind refuses. That is the fail-closed design working as intended, and it is also
why every gap in §2 surfaces as a wrong refusal rather than a wrong number.

---

## 2. The coverage matrix

`question_interpretation/stamp_coverage.py`, measured dynamically. A sentinel
status no reconciler writes distinguishes "left alone" from "stamped LOST for a
stated reason"; per-kind evidence bundles decide whether a branch can reach
APPLIED.

```
route                     cohor compa aggre geogr group multi row_p proje ranki relat share reque stres thres unrM unrR
(point-in-time)           .     o     .     .     .     .     HOLE  o     .     .     .     .     .     .     HOLE HOLE
analytical_composition    o     o     ?     o     .     ?     .     o     o     ?     ?     .     o     ?     HOLE n/a
period_change_analysis    o     .     ?     o     .     ?     .     o     o     ?     ?     .     o     ?     HOLE n/a
portfolio_summary         o     o     ?     o     .     ?     .     o     o     ?     ?     .     o     ?     HOLE n/a
period_movement           o     .     ?     o     .     ?     .     o     o     ?     ?     .     o     ?     HOLE n/a
temporal_compare          o     .     ?     o     .     ?     .     o     o     ?     ?     .     o     ?     HOLE n/a
evolution                 o     .     ?     o     .     ?     .     o     o     ?     ?     .     o     ?     HOLE n/a
evolution_funnel          o     .     ?     o     .     ?     .     o     o     ?     ?     .     o     ?     HOLE n/a
evolution_pipeline_stage  o     .     ?     o     .     ?     .     o     o     ?     ?     .     o     ?     HOLE n/a
forecast_extrapolation    o     .     ?     o     .     ?     .     .     o     ?     ?     .     o     ?     HOLE n/a
scenario                  o     .     ?     o     .     ?     .     .     o     ?     ?     .     .     ?     HOLE n/a
risk_limits               o     o     ?     o     .     ?     .     o     .     ?     ?     .     o     ?     HOLE n/a
funded_bridge             o     .     ?     o     .     ?     .     o     o     ?     ?     .     o     ?     HOLE n/a
cohort_progression        o     .     ?     o     .     ?     .     o     o     ?     ?     .     o     ?     HOLE n/a
cohort_conversion         o     .     ?     o     .     ?     .     o     o     ?     ?     .     o     ?     HOLE n/a
geo_exposure              o     o     ?     o     .     ?     .     o     .     ?     ?     .     o     ?     HOLE n/a

  .    stamped — a branch can confirm it on this route
  o    route-bound — a branch considered it; this route does not do that thing.
       CORRECT behaviour, not a hole: "compare with last quarter" on a risk-limit
       schedule genuinely was not compared, and blocking is right.
  HOLE no branch claims the kind: it keeps LOST whatever the route actually did
  ?    not constructed — an untested cell, NOT a hole
  n/a  cannot be raised on this route
```

18 HOLE cells. **17 are designed. One is live.**

### 2.1 `unresolved_measure` — designed, 16 cells

Raised by `detect_requested_facets` already carrying `status=LOST` and its own
reason: *"the parser never resolved these words, so no execution could have
honoured them and there is no evidence to weigh."* A reconciler branch would
have nothing to read. Blocking always is the intent — the outcome it prevents is
a silent 3-of-4. The user sees a named refusal identifying the unrecognised
slot. Designed-unreachable, permanently.

Asserted against the code rather than the table:
`test_an_unresolved_measure_is_adjudicated_at_construction` fails if the
detector ever stops setting a reason, at which point the hole becomes live.

### 2.2 `unresolved_role` — designed, 1 cell

Only raised under the clarify variant, and only on the point-in-time path.
`assess` reads it **before** the blocking test and returns a clarification; the
facet's status is never consulted, so having no reconciler branch costs nothing.
It is also the only kind in the matrix that does not block.

### 2.3 `(point-in-time)/row_population` — **LIVE**

The only live hole. `reconcile_facets` has no `KIND_POPULATION` branch at all.
A population facet arriving on the point-in-time path falls through every
branch, keeps `LOST`, and blocks.

* **What a user sees today:** *"I understood that you asked for the population
  seasoning_segment = Front Book, but that could not be applied to the
  calculation … I have not substituted a broader figure."* — for "What is the
  balance of the front book?"
* **Currently reachable:** yes, on the shipped tape through the shipped entry
  point. Demonstrated in §3.
* **Designed or incidental:** incidental, and newly so. Before `e35a01b` no
  constructor put a population facet on this path, so the absent branch cost
  nothing. The split is exactly the change that starts putting them there.

---

## 3. Reachability, by construction

`question_interpretation/hole_reachability.py`. Three ordinary questions naming a
governed population, none of them composite, so no route claims them.

| | at `43f264a` (before the split) | at `8a084c9` (HEAD) |
|---|---|---|
| "What is the balance of newly originated loans?" | **ok** — `grouping_dimension/applied` | **refuse** — `row_population/lost` |
| "What is the average LTV of the back book?" | **ok** — grouping + statistic applied | **refuse** — `row_population/lost` |
| "What is the balance of the front book?" | **ok** — `grouping_dimension/applied` | **refuse** — `row_population/lost` |

Reproduced identically through both entry points — `run_mi_agent_query` directly,
and `execute_governed_mi_query` with routing as shipped (`route=None`, no route
claims them).

**Which book reaches it: the alderbridge tape. The shipped one.** No fixture had
to be constructed, and the brief's instruction to build one if none reached it
does not apply. The `p1j1` fixture book reaches it too, which is how it first
surfaced, but that framing understated it — this is the production book and the
production entry point.

**The question shape that reaches it:** a question naming a governed population
term that (a) the parser puts in `spec.filters`, (b) the facet detector also
raises as a dimension, and (c) no route claims. Front book, back book, new
lending, newly originated — the core seasoning vocabulary.

### 3.1 Why both measurement surfaces are clean anyway

* The **calibration bank** is always point-in-time, but only three of its 252
  questions have a field that is both filtered and raised as a dimension, and
  all three are `borrower_type`, which this tape does not carry.
* The **robustness bank** contains the seasoning questions, but they are
  composite, so `analytical_composition` claims them and they never reach
  `reconcile_facets`.

The vocabulary that reaches the hole is in the robustness bank only in its
composite phrasings. Neither bank contains the simple form — "what is the
balance of the front book?" — which is the form a user asks.

---

## 4. The routing dependency

**What sends the 61 away from `reconcile_facets`:** `analytical_composition`
recognises a question when `planner.plan_for(...)` returns a plan and
`plan.is_composite` is true — more than one governed capability. It registers at
priority 5, ahead of every single-capability route, at confidence 0.8. Composite
questions are claimed; simple ones fall through.

**Is that decision load-bearing for safety?** Measured, not argued. Routing was
forced off in-process (`try_route` returns `None` for everything) and all seven
governed-window questions re-run:

| id | routing as shipped | routing forced off |
|---|---|---|
| Q1.1 | ok, `analytical_composition`, population applied | **fails at measure resolution** — *"'changed last few months' is not a governed measure in this dataset"* |
| Q7.1 | ok, cohort + grouping + `row_population` **applied** | fails at measure resolution |
| Q7.3 | ok | fails at measure resolution |
| Q7.4 | ok | fails at measure resolution |
| Q8.1 | ok | fails at measure resolution |
| Q8.3 | ok | refuse — `grouping_dimension/lost` |
| Q8.4 | ok | fails at measure resolution |

**The premise in the brief is wrong, and so was mine.** A routing change does
not deliver the 14 governed lending windows to the split. Six of the seven
questions never get that far — the point-in-time parser cannot resolve a measure
from their wording at all, so no facets are raised and the split has nothing to
reclassify. The seventh reaches the guard as a grouping and refuses, exactly as
it does today.

So the guarantee against 32c263a recurring on those 14 is not routing. It is
that **the point-in-time path cannot answer those questions in the first place**
— a much older and more robust property, and one no routing change would alter.
Routing is a second layer on top of it, not the load-bearing one.

**Does the governed comparison alone hold for the 14 if they arrived?**
Unanswerable end-to-end, and it must be reported that way rather than inferred:
they cannot be made to arrive by turning routing off, so there is no run in
which the governed comparison is the thing being tested. At unit level it does
hold — `_governed_population_predicates` resolves "front book" to
`seasoning_segment = Front Book` and "new lending" to `months_on_book le 1`, and
`test_population_keeps_the_wording_the_governed_check_reads` proves it fails when
the wording is destroyed. That is generated coverage standing in for a construct
the corpus cannot exercise, and it should be quoted as such.

**What WOULD arrive at the split if routing changed:** on the evidence above,
for this vocabulary, nothing new. The live hole is reached by simple questions
that already fall through today. The routing dependency is real but it is
guarding a door that has a second lock.

---

## 5. The `e35a01b` diagnosis

**Mechanism, exactly.** `_split_named_dimension_roles` runs on the first line of
`reconcile_facets`. For a field in `spec.filters` it rewrites a `KIND_GROUPING`
facet as `KIND_POPULATION`:

```
split in :  ('grouping_dimension', 'newly originated',  'seasoning_segment')
split out:  ('row_population', 'the population seasoning_segment = Front Book')
receipt  :  row_population / LOST      ->      REFUSE
```

The grouping branch it was moved *out of* contains precisely the clause that
handled this case:

```python
elif any(k in (getattr(spec, "filters", None) or {})
         for k in candidates if k):
    # The concept was honoured as a POPULATION rather than an axis.
    facet.status, facet.reason = APPLIED, ""
```

That clause was written for exactly the situation the split now intercepts. The
split moves the facet past it into a kind with no branch at all. Every remaining
`elif` is skipped, the loop ends, and the detection-time `LOST` stands.

**Routes affected:** the point-in-time path only, because that is the split's
only call site. Every routed answer is unaffected — `population_facets(spec)` +
`reconcile_population` + the governed comparison cover populations there, which
is why Q7.1 shows `row_population/applied` on the routed run in §4.

**Fix shape** (measured, unapplied; the diff is preserved at
`scratchpad/unapplied_fix.patch` and is 103 lines):

1. `mi_query_executor._apply_filters` gains an `applied: List[str]` out-parameter
   and records each field key after `_require_column` confirms the column and
   before the branch split, so every filter shape counts once. The executor
   publishes it as `metadata["applied_filter_fields"]` — evidence, distinct from
   `reconciliation.filters`, which echoes the spec.
2. `reconcile_facets` gains a `KIND_POPULATION` branch reading that list, at
   `reconcile_population`'s bar: APPLIED only when the executor reports having
   run a predicate against the field; UNAVAILABLE when the book lacks the
   column; LOST otherwise.

Measured with it applied: the 5 `p1j1` failures pass and the file is 53/53.

**Which grading path it moves:** the point-in-time population verdict, which is
currently "always refuse" because the branch does not exist. Anything it changes
is a case that refuses today. It does not touch any routed path, the population
ledger, or the governed comparison.

**Where the fix belongs.** In the **reconciler**, with the executor supplying the
evidence — not in the classification and not in the route.

* In the classification (make the split not produce a population where the
  grouping branch would have stamped it) would patch the instance and leave the
  hole: the next constructor that emits a population on this path repeats it.
  It also throws away the role information the split exists to record.
* In the route (wire `population_facets` / `reconcile_population` into the
  workflow, as `mi_service` does for routed answers) would close the hole, but
  by duplicating the ledger rather than by giving the reconciler an evidence
  path. The duplication is the second-order defect the first attempt at this fix
  produced: two population facets for one field, one stamped and one not.
* In the reconciler, the hole closes for **every** future producer, which is the
  test of whether this is one defect or a class.

**Can any other facet kind undergo the same transition?** Today, no: the split
is the only reclassifier, and it produces only `KIND_POPULATION` and
`KIND_UNRESOLVED_ROLE`. But the transition is a class, not an instance, and the
matrix states its precondition exactly: **any move into a kind whose target
reconciler has no branch.** There are three such kinds. Two are designed-inert.
One is `row_population`, and it is the one the split moves into. A future
reclassification into `unresolved_measure` — say, a measure slot demoted after
parsing — would reproduce it identically.

---

## 6. Sequencing recommendation

Recommendation only; none of it is authorised by this brief.

**Must close before the clarify default is applied — `(point-in-time)/row_population`.**
Not because clarify would make it worse, but because **clarify does not touch
it**. Measured: under `UNRESOLVED_ROLE_DEFAULT="clarify"`, "What is the balance
of newly originated loans?" and "What is the balance of the front book?" still
produce `row_population/lost` and still refuse, identically to the current
default. Clarify governs the *unresolved* role; these facets have a resolved
filter role and go down the population branch either way. Applying clarify on
top of the live hole would ship a clarification improvement while three ordinary
questions stay broken, and would make the eventual fix harder to attribute.

**Can wait — nothing.** There is one live hole.

**Backlog.**

* **B6 — the two measurement surfaces cannot see the point-in-time population
  path.** Both were clean throughout the regression. The calibration bank has
  three qualifying questions and this tape carries none of their field; the
  robustness bank has the vocabulary only in composite phrasings. A surface that
  exercises simple governed-population questions on the point-in-time path is
  missing, and this inventory is not it — an inventory is not a regression net.
* **B7 — `mi_calibration._run` bypasses routing.** The calibration bank measures
  a path no user reaches, and the difference was invisible until now. Whether
  that is right is a separate decision; that it is undocumented is not.
* **B8 — the `?` cells.** Six kinds have no routed evidence bundle
  (`aggregate_contribution`, `multi_measure`, `relationship`, `share`,
  `threshold`, and `geographic_scope` beyond the analytical form). They are
  untested, not holes, and should be closed out before the matrix is quoted as
  complete.

---

## 7. What this inventory contradicts in what has already been reported

Five corrections, four of them to my own reports.

1. **"The split is inert on this tape."** Reported at `e35a01b` and repeated as
   limitation L1 in the variant measurement. **False.** It is inert on the
   *corpora measured*, on the `borrower_type` cases. On the same tape it is
   live, through `seasoning_segment`, for simple front-book questions — and
   those refuse. L1 should have read "inert on the questions these two banks
   contain", which is a statement about the banks, not the tape.

2. **"e35a01b breaks 5 tests in `test_p1j1_vintage_seasoning`."** True but
   understated, and the framing was wrong. It was reported as a fixture-book
   finding. It is a shipped-surface regression on the shipped book, reachable
   through `execute_governed_mi_query` with default routing, for questions a
   user would plainly ask.

3. **"The recurrence is out of reach, not out of the set … contingent on
   routing."** Reported in the variant measurement, and repeated as the premise
   of this brief. **Wrong about the mechanism.** Turning routing off does not
   deliver those 14 to the split — six of the seven questions die at measure
   resolution first. The guarantee is not routing; routing is the second layer.

4. **"32c263a and e35a01b are the same defect."** Reported by me last turn and
   restated in the brief. **Half right, and the difference matters for the fix.**
   32c263a was a *comparison* defect: the facet reached a branch that read it
   wrongly, and the branch was fixed. e35a01b is an *absence*: the facet reaches
   no branch. A fix for the first — better comparison logic — could not have
   prevented the second. What they share is the consequence, not the cause: a
   correct classification producing a refusal.

5. **The brief's own framing, "of 64 unresolved facets only 3 reach the split".**
   Still true, and still the wrong number to hold onto. The live hole is not in
   the *unresolved* set at all — N1/N2/N3 have a **resolved** filter role. The
   unresolved-role work and the live regression are disjoint, which is the whole
   of §6's sequencing argument.

---

## 8. Instruments added

Both carry can-fail tests, per standing rule 4.

* `question_interpretation/stamp_coverage.py` — the matrix. `--self-test`
  asserts it can produce all four cell values, including the hole it exists to
  find and a correct non-application it must not confuse with one. Its first
  draft reported eleven false holes from a one-size evidence bundle; its second
  reported a false hole on the routed population cell from a malformed
  envelope. Both were caught by the self-test rather than by reading the output.
  Instances ten and eleven of the standing pattern.
* `question_interpretation/hole_reachability.py` — reachability by construction,
  one stage per process because the app binds its book and its routing at import
  time. `--stage routing-off` monkeypatches `try_route` in-process only.
* `question_interpretation/tests/test_stamp_coverage_instrument.py` — 8 tests.

---

## 9. Closed — and two corrections to §5 and to the brief

The fix in §5 was applied in the commit that carries this section. The matrix
now reads **17 holes, all designed, zero live**.

### 9.1 The regression was wider than §3 measured

§3 found three questions. The test sweep found **eleven** tests repaired by the
fix, in five modules:

```
tests/test_p1j1_vintage_seasoning.py              5
mi_agent/tests/test_mi_query_invariants.py        3   including the full generated suite
tests/test_p1n_statistic_breadth.py               3   extremes over a requested population
```

`test_full_generated_suite_holds_the_invariant` is the one that should have
caught this at `e35a01b`, and it was already failing there. Three of the eleven
were unknown until the sweep. The blast radius reported at §3 — "three ordinary
questions" — was the part reachable through the entry points this programme's
instruments use. It was a floor, not a measurement, and should have been
labelled as one.

### 9.2 "Same shape twice" is the wrong lesson, and it was mine

I wrote that 32c263a and e35a01b were "the same defect in different clothes",
and the brief adopted it. They are not, and the difference decides the fix:

* **32c263a** — the facet reached a branch that **read it wrongly**: a field name
  compared against governed predicate text. A comparison defect, fixed by
  comparing governed definitions.
* **e35a01b** — the facet reached **no branch at all**. An absence. The default
  status stood whatever execution had done.

A better comparison could not have prevented an absent branch. Had the lesson
been taken as "same shape", the fix would have been more comparison logic and the
hole would still be open. What they share is the **consequence** — a correct
classification producing a refusal — and a **precondition**: a facet
reclassified after detection onto a path that cannot confirm it. The consequence
is not actionable; the precondition is, and §9.3 closes it.

### 9.3 The class, closed structurally

`RECLASSIFICATION_TARGETS` in `execution_receipt.py` names every kind a facet
can be moved into after detection.
`question_interpretation/tests/test_reclassification_targets.py`:

* **discovers** the moves the code actually performs, by running the
  reclassifier under every variant and both spec shapes — not by reading a list,
  because a registry checked against a hand-kept list passes while the code does
  something else;
* requires every discovered move to be **registered**;
* requires every registered target to have a **receiving branch** on the path
  that can produce it, or a stated reason why none is needed;
* proves it can fail three ways — deregistering a target, registering one with
  no receiver, and a discovery that finds nothing.

Verified by construction: with the reconciler branch removed and everything else
in place, `test_every_registered_target_has_a_receiver_or_a_stated_reason` fails
with `['row_population']`. This is the test `e35a01b` would not have passed.

### 9.4 The recurrence guarantee, stated correctly

Recorded here because it is quoted elsewhere and the earlier phrasing was wrong.
The guarantee against 32c263a recurring on the 14 governed lending windows is
**not routing**. It is that the point-in-time path cannot resolve a measure from
that wording at all — six of the seven questions fail at measure resolution when
routing is forced off, before any facet is raised. **Routing is a second lock,
not the first.** Both were measured (§4); neither is an argument.
