# D8 — "was a requested population actually applied"

Written before implementing. §2 is BASELINE measurement of `7c12837`.

Base: HEAD `7c12837`; merge-base `4e051f3`; `4e051f3` and `28ece25` both
ancestors; clean tree.

---

## 1. The sequence, steps 1–4 complete before this document

| # | step | result |
|---|---|---|
| 1 | name the surface | the routed surface (which is also differ surface 5, added in D6) |
| 2 | cases in, declared failing | `rt_016` (B17, from D7), `rt_020` (new), `rt_021` (control) |
| 3 | extend the differ | **not needed** — surface 5 already carries every routed-surface question as text |
| 4 | **RE-RECORD the baseline, before the fix** | **720 answers across 5 surfaces** → `answer_baseline_d8.json` |

## 2. Baseline, measured

The census classed D8 **agree-by-maintenance**: three evidence sources, each
scoped to a path the others do not run on, so they cannot contradict one another
*today*. Measurement finds the maintenance has already lapsed, in two places, and
neither is the contradiction the census anticipated.

### 2.1 B17 — a drill-through on a routed question always refuses

```
"Show geographic exposure by ITL3 area."  + filters={"collateral_geography": "South East"}
   -> refuse: "the population collateral_geography = South East … could not be applied"
```

`material_predicates` is computed from `parsed.spec.filters` **before**
`try_route` calls `parsed.merge_filters(extra_filters)`. So the drill-through
narrowing never reaches the frame resolver, but it IS on the spec by the time
`population_facets(spec)` reads it. Raised, never applied, refused. Fail-closed
and correct in outcome; wrong in cause.

### 2.2 The new one: a narrowing that RAN, unrecorded

```
"What is the balance of the front book?"  + filters={"collateral_geography": "South East"}
   -> ok.  "Seasoning Segment = Front Book · South East · 278 loans"   (down from 1,177)
   facets: [row_population seasoning_segment APPLIED]
```

**Both narrowings ran. The receipt records one.** The geography drill-through is
applied and absent.

It is B16a's mirror. There a narrowing was **lost** and unrecorded; here one is
**applied** and unrecorded. Either way the receipt does not say what happened to
the rows, and the honour-or-clarify contract can only adjudicate what is
represented.

The cause is the D8 divergence stated in one line:

> The **routed** path raises populations from `spec.filters`, through
> `population_facets(spec)`. The **point-in-time** path raises them only from the
> seasoning owner, or by reclassifying a dimension the question NAMED.

So the same drill-through on *"balance by region"* IS recorded — region is named,
so a grouping exists for the role owner to reclassify — and on *"balance of the
front book"* is not. **Whether a narrowing appears on the receipt depends on
whether the question happens to mention the field**, which is not a property of
the narrowing at all.

### 2.3 The three sources, and why they have not collided yet

| source | read by | path |
|---|---|---|
| `metadata.populationApplied` | `reconcile_population` | routed |
| the analytical plan's `narrowedTo` | `_analytical_population_satisfies` | routed, composite only |
| the executor's `applied_filter_fields` | `reconcile_facets` `KIND_POPULATION` | point-in-time |

`applied_filter_fields` is written by the point-in-time executor, which no route
runs. So the third cannot meet the first two — **by construction of the paths,
not by design**, which is exactly what "agree-by-maintenance" names. B16a added a
fourth reader of the same three sources for `KIND_LOST_NARROWING`, and did it by
copying the branch.

## 3. The class, and the illustration

**The class:** *a population is raised wherever the spec carries it, on every
path, and stamped by one owner reading the three evidence sources in a fixed
order. Whether a narrowing reaches the receipt must not depend on whether the
question named the field, and must not depend on which path answered.*

**The illustration:** `rt_016` (raised, never applied, refuses) and `rt_020`
(applied, never raised, silent). Both constructed — no corpus question uses the
drill-through API, because no corpus question can: it is an API parameter, not
words.

## 4. The rule

1. **One raiser.** `population_facets(spec)` runs on **both** paths, so a
   narrowing on the spec is represented wherever it came from — the sentence, the
   seasoning owner, or the drill-through API. Deduped against what the detector
   already raised, on `(kind, field_key, label)`, as `7c46f81` established.
2. **One stamper.** `population_applied(facet, …)` reads the three sources in a
   fixed order — the executor's applied fields, the route's `populationApplied`
   ledger, the analytical plan's declaration — and is called by every reader
   including `KIND_LOST_NARROWING`, which stops carrying its own copy.
3. **B17's ordering.** `material_predicates` is computed **after** the
   caller-supplied filters are merged, so the frame resolver narrows on what the
   spec actually carries.

## 5. Every place the owner's answer arrives — provisional, VERIFIED in implementation

Moved into implementation as instructed; the headline check has now missed three
times.

| # | site | today |
|---|---|---|
| 1 | `reconcile_facets` `KIND_POPULATION` | reads `applied_filter_fields` |
| 2 | `reconcile_population` | reads `populationApplied` |
| 3 | `_guard_routed_answer`'s analytical top-up | reads `narrowedTo` |
| 4 | `reconcile_routed_facets` `KIND_POPULATION` | reads `narrowedTo` |
| 5 | `reconcile_facets` `KIND_LOST_NARROWING` | **a copy of 1**, added by B16a |
| 6 | `reconcile_routed_facets` `KIND_LOST_NARROWING` | **a copy of 2 and 3**, added by B16a |
| 7 | `reconcile_facets` `KIND_GEOGRAPHIC_SCOPE` | reads the narrowing ledger by VALUE |
| 8 | `reconcile_routed_facets` `KIND_GEOGRAPHIC_SCOPE` | reads `narrowedTo` |
| 9 | `mi_service` population merge | dedupes and stamps the survivors |

Sites 5 and 6 are the ones to watch: B16a consolidated a decision and then
**copied the evidence branch to a new kind**, which is the standing rule about
consolidation creating a new reader, arriving from inside this programme's own
work for the second time.

## 6. Pre-registered prediction

### 6.1 What moves

| id | today | predicted |
|---|---|---|
| `rt_016` | refuse — the drill never reached the frame | `ok`, geography population **applied** |
| `rt_020` | one population facet | **two**, both applied; the answer text unchanged |
| `rt_021` | refuse, `row_population lost` | **unchanged** — `geo_exposure` resolves its frame through the seam for geography but reports nothing for `account_status` |

`answer_diff`: **718 identical, 2 moved**, both `routed_surface`, and **`rt_020`'s
answer text must NOT move** — only its receipt. So of the two, `rt_016` moves its
answer and `rt_020` moves only `executionSummary`.

### 6.2 What must not move

1. **`rt_021` stays refusing.** It is the can-fail against widening what counts
   as evidence.
2. **No corpus answer moves at all.** No corpus question uses the drill-through
   API. If any of the 697 moves, the population raiser has changed what it raises
   from the SENTENCE, which is not this change.
3. **The seasoning families stay at their by-name counts**, both books.
4. **Robustness `32/10/2`; calibration `259/259`, 0 hard fails, 0 known gaps.**
5. **No lexical decision moves.** 693 of 693.
6. **No duplicate population on any receipt** — `7c46f81`'s ten-minute defect.
7. **The stamping matrix stays at 0 live holes.**

### 6.3 Stop conditions

* `rt_021` answering;
* any corpus answer moving;
* `rt_020`'s answer TEXT moving;
* a duplicate population facet anywhere;
* any seasoning family count moving;
* a population stamped applied without one of the three evidence sources naming
  it.

### 6.4 Acceptance

* one raiser and one stamper; §5 verified site by site and the differences
  reported;
* B16a's copied branches consume the owner rather than repeating it;
* both defects closed: raised-and-never-applied, and applied-and-never-raised;
* all five surfaces, deterministic arm, both books; seasoning by name.

---

# Result, measured against §6

## Against the prediction

| predicted | measured |
|---|---|
| `rt_016` moves to `ok`, geography population applied | **yes** |
| `rt_020` gains a second population; **answer text unchanged** | **yes — only `executionSummary` moved** |
| `answer_diff` 2 moved, both `routed_surface` | **719 compared, 717 identical, 2 MOVED, both `routed_surface`** |
| no corpus answer moves | **none** — calibration, service path and both robustness books all identical |
| seasoning families unchanged, both books | **Q1 4, Q7 4, Q8 12 all CORRECT** |
| robustness `32/10/2`; calibration `259/259`, 0 hard fails, 0 gaps | **both** |
| lexical 693 of 693 | **693 of 693** |
| stamping matrix 0 live holes | **17 holes, 17 designed, 0 live** |
| **`rt_021` unchanged** | **NO — it fired, and it was right to** |

## The stop condition fired, and the case was what was wrong

§6.3 listed *"`rt_021` answering"* as a stop condition:

> If this ever answers `ok`, a population reached the receipt applied on
> something other than execution evidence.

It answered `ok`. Work stopped and the evidence was checked before anything else
was written:

```
"Show geographic exposure by ITL3 area." + {"account_status": "Active"}
  populationApplied: {"applied": ["account_status = Active"],
                      "rowsBefore": 11035, "rowsAfter": 11000}
```

**The narrowing genuinely ran.** The answer text does not change because Active
is 11,000 of 11,035 loans and the largest ITL3 area is the same either way — a
real narrowing whose effect is invisible at that precision.

So the case's premise was wrong, not the code. It was written on the belief that
`geo_exposure` could not narrow on `account_status`. It can: the route resolves
its frame through the governed population seam, so it narrows on **any** material
predicate and reports evidence — which is what the seam's own comment says it
does.

Replaced by `rt_022` against `forecast_extrapolation`, which answers from a
replayed history model rather than a resolved frame and therefore genuinely
reports nothing. That is the route where "no evidence, no certification" can be
tested.

**Recorded rather than quietly rewritten**, because a stop condition that fires
and turns out to be the instrument's fault is the case a surface exists to
produce, and it is the second time in three commits that a declared expectation
was the thing that was wrong.

## Three defects, not two

§2 predicted two. The tests found a third, in the change itself.

`"What is the balance by region?"` with a drill to South East raised **two
identical** `row_population collateral_geography` facets — the drill ledger and
the role owner reclassifying the named `region` grouping, with byte-identical
labels. One stamped applied, one left lost, the answer refusing itself.

**That is `7c46f81`'s ten-minute defect, on the path this commit added a raiser
to.** The dedupe was in the wrong place: it ran where the drill is raised, and
the split's copy does not exist yet at that point. Moved to after the split,
where every raiser has run.

Caught by `test_no_duplicate_population_reaches_a_receipt`, which exists because
`7c46f81` recorded the defect. **The standing rule paid for itself.**

## What the site-by-site verification found

Nine sites listed, all nine confirmed. Two things the list did not say:

* **`reconcile_facets`'s `KIND_POPULATION` branch matched on `satisfied_by()` and
  canonicals; `reconcile_population` matched on `field_key` alone.** Two rules
  for one question, which is what "agree-by-maintenance" hides. The owner uses
  the richer rule and both readers now share it; no surface moved, so the
  difference had never been reachable.
* **The two analytical accessors are not interchangeable, and the caller must
  choose.** `_analytical_population_satisfies` reads a population LABEL and needs
  field and value to match; `_analytical_narrowed_to` reads a bare VALUE label
  and matches on value alone. Passing a population facet through the second
  silently fails — which it did, in a test I wrote, which is how it was found.
  Now stated in the owner's docstring rather than discovered again.

## A measurement hazard, recorded

Re-pointing `rt_021` to a different question **under the same id** made
`answer_diff` report a third movement. The differ keys on `(surface, id)`, so a
case replaced in place reads as behaviour that moved.

The replacement took a new id (`rt_022`), and the differ then reported it
correctly as `only_before: rt_021` / `only_after: rt_022` — one case gone, one
arrived. **A replaced case must take a new id**, or the instrument reports a
product change where there was a test change.

## Mutations, and what caught each

| mutation | caught by |
|---|---|
| B17's ordering restored | `test_a_drill_through_on_a_routed_question_is_applied` |
| the drill raiser removed | `test_a_drill_through_that_ran_is_recorded` |
| the dedupe moved back before the split | `test_no_duplicate_population_reaches_a_receipt` |
| the stamper accepts any ledger entry | `test_nothing_proves_nothing` |
