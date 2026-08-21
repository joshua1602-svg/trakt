# D7 (B12) — one owner for "was the requested grouping actually applied"

Written before implementing. Everything in §2 is BASELINE measurement of
`4dff358`, taken before the fix was designed.

Base: HEAD `4dff358`; merge-base `4e051f3`; `4e051f3` and `28ece25` both
ancestors; clean tree.

---

## 1. The decision, and its two owners

**D7 — was the requested grouping actually applied.** Census (`9ee0e5b`): two
owners, **disagreeing**, reachable by construction.

| path | owner | evidence it accepts |
|---|---|---|
| point-in-time | `reconcile_facets`, `KIND_GROUPING` branch | the executor's **declared group keys** (`group_field_keys`), or the canonical field appearing as a **result column** |
| routed | `reconcile_routed_facets`, `KIND_GROUPING`/`KIND_RANKING` branch | a declared ranking field; failing that, an **artifact row-key name match**; failing that, **the mere existence of any axis** |

The last rung is the defect. Its own comment names it:

> *"where the answer is cut by SOME axis this cannot identify, the facet stays
> applied, and which axis it was remains unproven."*

That is the exact inverse of the bar `reconcile_population` holds on the same
receipt — *"a facet is APPLIED only when the route reports having applied that
field. A route that reports nothing leaves every population facet LOST."*

## 2. Baseline, measured

### 2.1 Every routed grouping claim that stands today stands on the last rung

Classified across the 343-question answer corpus and the 250-question
calibration corpus, by re-running the ladder over each envelope:

```
route                     tier                     n
concentration_analysis    residual (unproven)      4
concentration_analysis    not applied              9
evolution                 not applied              8
geo_exposure              residual (unproven)     11
geo_exposure              unattributed             1
risk_limits               residual (unproven)      4
risk_limits               not applied              1
```

**Nineteen live certifications, and not one of them is proven.** No routed
grouping facet on either corpus reaches APPLIED by a name match or a
declaration; the name-match rung fires zero times, because routes label their
columns for a reader (`area`, `code`, `category`) and never by canonical field.

*(The one `geo_exposure` case my reconstruction cannot attribute is recorded as
unattributed rather than assigned to a rung. It is checked during
implementation, not assumed.)*

### 2.2 Eleven are true, eight are false, and the eight are two different wrongs

| route | n | true? | what the reader is told, against what the answer is |
|---|---|---|---|
| `geo_exposure` | 11 | **true** | *"broken down by region"* over `Funded exposure by ITL3 area`. Right claim, guessed evidence. |
| `concentration_analysis` | 4 | **false — wrong dimension** | *"broken down by vintage"* over `Amortisation Type concentration`, `Collateral Type concentration`, `Purpose concentration`… The route publishes seven concentration tables and **none of them is the one asked for**. |
| `risk_limits` | 4 | **false — no dimension at all** | *"broken down by region"* / *"by account status"* over a limit-test table whose columns are `test · actual · limit · headroom · status · movement · source`. |

The `concentration_analysis` four are the worse pair. There IS a breakdown, and
it is the wrong one — the shape `detect_substitution` exists to catch on the
point-in-time path and which nothing catches on the routed one:

```
risk_216            claims account_status   over  Purpose concentration
vintage_cohort_018  claims vintage_year     over  Amortisation Type concentration
nneg_er_005         claims age_bucket       over  Collateral Type concentration
nneg_er_006         claims vintage_year     over  Purpose concentration
```

`nneg_er_005` — *"Show NNEG exposure by borrower age bucket"* — returns seven
tables, no age bucket among them, and a receipt saying the age-bucket breakdown
was applied.

### 2.3 The evidence already exists and the reader does not read it

The sharpest fact in this diagnosis.

```
envelope["workflow"]["dimension_results"] ->
  [('amortisation_type',  'Amortisation Type'),
   ('collateral_type',    'Collateral Type'),
   ('purpose',            'Purpose'),
   ('exposure_currency_denomination', 'Exposure Currency Denomination'),
   ('collateral_geography','Collateral Geography'),
   ('origination_channel','Origination Channel'),
   ('interest_rate_type', 'Interest Rate Type')]
```

`concentration_analysis` **already declares, on the envelope, the exact canonical
field keys it grouped by.** `_route_concentration` sets `envelope["workflow"] =
result`, and the workflow computes `dimensions_selected` from the same list. The
receipt reader ignores all of it and guesses from a display column called
`category`.

`geo_exposure` and `risk_limits` carry no `workflow` block and declare nothing.

---

## 3. The class, and the illustration

**The class:** *a grouping is APPLIED only where execution NAMES the field it
grouped by — the same bar `reconcile_population` already holds — and one owner
applies that bar on both paths.* The residual rung is removed rather than
narrowed: a rung that stamps APPLIED on what it cannot disprove is the inverse of
every other bar on this receipt, and narrowing it leaves the inversion in place.

**The illustration:** the 19 claims in §2.1 and the 8 false ones in §2.2. They
are the measured extent, not the definition. Per the standing rule, movement
outside the 19 is expected and is not a stop condition; movement outside the
class is.

---

## 4. Every place the owner's answer arrives — and what was already deriving it

| # | site | derives it today? | after |
|---|---|---|---|
| 1 | `reconcile_facets` `KIND_GROUPING` branch | **yes** — owner A | calls the owner; supplies the executor's declared keys as evidence |
| 2 | `reconcile_routed_facets` `KIND_GROUPING`/`KIND_RANKING` branch | **yes** — owner B, including the residual rung | calls the owner; supplies the route's declaration as evidence |
| 3 | `detect_substitution` (point-in-time) | reads `status != APPLIED` and the executor's group keys — a CONSUMER, not an owner | unchanged; it will now see routed-shaped losses too if it is ever wired there (it is not, in this commit) |
| 4 | `_blocks` / `assess` | consume the status | unchanged — but 5 answers change verdict as a consequence, which is the point |
| 5 | `answer_axis_keys` | supplies the residual rung's input | **retained only as a fallback where a route declares nothing**, or removed; settled by measurement in §5 |
| 6 | `ranking_evidence` | a declaration reader, already the strongest rung | becomes one of the owner's evidence suppliers, unchanged |
| 7 | `check_period_grain` / `check_window_coverage` | read granularity and period facets | untouched |
| 8 | `granularity_facets` | owns the reporting GRAIN | untouched — the grain and the axis are different decisions, and `geo_exposure` needs both (`rt_007`/`rt_008`) |

**Nothing else derives "was this grouping applied".** Confirmed by reading every
`KIND_GROUPING` reference in `execution_receipt.py` and every consumer of
`executionSummary.facets` in `mi_service.py`.

---

## 5. The rule

One owner: `grouping_status(facet, …)`. A grouping or ranking facet is APPLIED
only when a candidate key the facet resolves to appears in the **declared**
fields execution grouped by. The suppliers:

* **point-in-time** — the executor's `group_field_keys`, plus the canonical field
  appearing as a result column. Unchanged; this path already held the bar.
* **routed** — `declared_group_fields(envelope, route)`, reading in order:
  1. the concentration workflow's `dimension_results` field keys (already on the
     envelope, §2.3);
  2. `rankedMovement.canonicalField` (already read);
  3. a **declaration added at the route** for capabilities whose axis is fixed
     by the capability itself — `geo_exposure` (geography at ITL3 grain) and
     `risk_limits` (the fields its executed limit tests cover).

The existing exemptions are kept, because they are about a role rather than
evidence: a candidate key in `spec.filters` was honoured as a population, not
lost; a field the book cannot express is UNAVAILABLE, not LOST.

**The residual rung goes.** Where no supplier names a field, the facet is LOST
and the honour-or-clarify contract decides what the reader is told.

---

## 6. Pre-registered prediction

### 6.1 What moves

**Five answers, all from `ok` to a refusal, all on the routed path.**

| id | question | today | predicted |
|---|---|---|---|
| `risk_216` | *(claims `account_status`)* | ok, breakdown certified | refuse — the route grouped by Purpose, not account status |
| `vintage_cohort_018` | Show 2025 vintage concentration. | ok | refuse |
| `nneg_er_005` | Show NNEG exposure by borrower age bucket. | ok | refuse |
| `nneg_er_006` | Show NNEG exposure by vintage. | ok | refuse |
| `risk_limits_013` | Show concentration limit status. | ok | refuse |

The first four are unambiguous improvements: a reader asking for a vintage
breakdown is currently shown amortisation types and told it is vintages.

**`risk_limits_013` is not.** Its ANSWER is right — *"8 passed, 1 breach"* — and
it refuses only because the word *status* in *"limit status"* resolved to the
registry field `account_status`, which the limit tests do not cover. That is a
**term-resolution** defect, not D7's, and the refusal is the fail-closed
consequence of an incorrect premise: wrong reason, safe outcome, no wrong number
— the posture already recorded for B1. **Predicted here rather than explained
afterwards**, and filed as its own backlog entry.

`rt_013` — the declared expected-to-fail added in `4dff358` — **flips to
passing**, which is what that instrument is for.

### 6.2 What must not move

1. **The eleven `geo_exposure` claims stay APPLIED**, now proven rather than
   guessed. Any of them turning LOST stops the work.
2. **`rt_007` and `rt_008` keep both grouping facets applied** — the geography
   axis and the ITL3 grain are different decisions and this commit touches one.
3. **No point-in-time answer moves at all.** That path already held the bar.
4. **No lexical decision moves.** 693 of 693.
5. **The seasoning families stay at their by-name counts**, both books.
6. **The stamping matrix stays at 0 live holes**; `RECLASSIFICATION_TARGETS`
   stays closed.
7. **Robustness stays `32/10/2` on both books.** The five moving answers are not
   in that bank; if the count moves, the change reached further than measured.

### 6.3 Stop conditions

* any `geo_exposure` grouping claim turning LOST;
* any point-in-time answer moving;
* any answer moving that is not one of the five in §6.1;
* a lexical decision moving;
* a live hole appearing in the stamping matrix;
* a facet reaching a reconciler with no branch to receive it.

### 6.4 Acceptance

* one owner; both reconcilers demonstrably consume rather than derive, and §4 is
  confirmed line by line;
* the residual rung is **removed**, not narrowed;
* `concentration_analysis`'s existing declaration is read instead of guessed at;
* `geo_exposure` and `risk_limits` declare, and a test asserts each declaration
  against what the route actually publishes — not against a comment;
* all three surfaces, deterministic arm, both books; seasoning by name;
* `rt_013` flips to passing, and new declared expected-to-fail cases are added
  for the decisions still outstanding.

---

## 7. Settled alongside — the three items raised with this work order

### 7.1 The unreachable FILTER branch: it is NOT dead code, and my D2 report was wrong

Ruled by measurement, not by reasoning.

The D2 report said the role owner's FILTER branch was *"unreachable through a
well-formed question on this arm"*. That is true of question TEXT and **I did not
check the other way in**. `MiQueryRequest.filters` — the **drill-through API** —
merges caller-supplied filters into `spec.filters` on both paths, before the
guard runs. A UI drill from a *"balance by region"* table into one region is
exactly the shape that triggers the branch:

```
"balance by region"  + filters={"collateral_geography": "South East"}
   ->  row_population collateral_geography APPLIED    (the FILTER branch)
"balance by vintage" + filters={"vintage_year": 2024}
   ->  row_population vintage_year         APPLIED
"balance by region"  + no filters
   ->  grouping_dimension collateral_geography APPLIED
```

**The branch is live, shipped, and user-reachable.** It is not carried on a test
asserting its own unreachability; it is carried on a test that exercises it
through the API that reaches it. The corpus fact stays, reframed as what it is —
a statement about the corpus, not about the code:

> No corpus question's TEXT reaches the branch on the deterministic arm. The
> drill-through API does, and the LLM arm can.

Both go in this commit: `test_no_corpus_question_slots_a_named_dimension_as_a_filter`
keeps its corpus claim and loses its "therefore unreachable" framing, and a new
test drives the branch through `MiQueryRequest.filters`.

**And the drill-through found a defect while proving it.** A drill-through on a
question that ROUTES always refuses:

```
"Show geographic exposure by ITL3 area." + filters={"collateral_geography": "South East"}
   ->  refuse: "the population collateral_geography = South East … could not be applied"
```

`material_predicates` is computed from `parsed.spec.filters` **before**
`try_route` calls `parsed.merge_filters(extra_filters)`, so the drill-through
narrowing never reaches the frame but is on the spec by the time the guard reads
it. Fail-closed and correct in outcome, wrong in cause. Filed as **B17**, which
belongs to **D8** (whether a requested population was actually applied). Not
this commit.

### 7.2 Where B16 belongs: NEXT, before D6 — and it is worse than I reported

The D2 report described B16 as a mention nobody reads. Measured on the shipped
service path, it is a **wrong number**:

```
"What is the balance where account status is active?"
   ->  ok.  "Total Balance · grouped by Account Status · 2 groups · 11,035 loans"
       facets: grouping_dimension account_status APPLIED
```

The reader asked for the balance of ACTIVE loans. They were given the whole book
split by status, and the receipt certified the breakdown as applied. **That is
B13's class** — a narrowed question answered over 11,035 loans — and B13 was
treated as the defect it is.

**Placement: immediately after D7, ahead of D6, D8, D10, D9 and D14.** Because:

* it is the only remaining item that produces a **wrong number** rather than a
  wrong receipt; every census entry left is a receipt defect;
* its landing zone is already built and tested — D2's owner takes a resolved
  filter at source 1 and needs no further change;
* it cannot be folded into any facet-layer commit, which is why it must be placed
  rather than absorbed.

**And it splits in two, which is what makes the placement affordable:**

* **B16a — the facet layer, no parser change.** `lexical.is_filter_subject`
  already owns the mark. A narrowing the sentence marked and execution did not
  apply is recorded as a request that was lost, and honour-or-clarify refuses
  instead of answering over the whole book. This needs **no predicate value**, so
  it does not require the parser and does not invent one. This is the step that
  removes the wrong number, and it should go next.
* **B16b — the parser resolves the filter**, turning the refusal into an answer.
  **B1 is a hard prerequisite**: resolving more categorical filters through a
  denylist multiplies the fabricated-binding class B1 exists to retire. B16b
  goes wherever B1 goes, and not before.

**Why not before D6:** D6 (B14) governs whether a field is judged available in
the book being asked about, and it currently answers from whichever frame a route
loaded. Resolving more filters before that is fixed would multiply B14's wrong
message across a wider set of questions. B16a does not resolve filters — it only
records the loss — so it is safe ahead of D6; **B16b is not.**

### 7.3 The `rt_013` pattern, extended

Kept, and generalised as instructed: a declared expected-to-fail stating the
CORRECT outcome, added before the fix, flipping when it lands.

`rt_013` flips in this commit. Added here, each verified to fail today for the
stated reason:

| case | decision | what it pins |
|---|---|---|
| `rt_014` | **D6 / B14** | *"What is the forecast run rate for the front book?"* refuses with *"front book — field is unavailable in this dataset"* — true of the pipeline frame the route loaded, false of the book. |
| `rt_015` | **B16a** | *"What is the balance where account status is active?"* answers over 11,035 loans with the breakdown certified. Correct outcome: it must not answer over the whole book. |
| `rt_016` | **D8 / B17** | a drill-through on a routed question refuses because the narrowing is merged after the frame is resolved. |

D9, D10 and D14 are **not** given cases here, and the reason is stated rather
than left as an omission: D9 and D14 are classed *agree-by-maintenance* — there
is no wrong outcome to state, so the right instrument is a case that fails when
the owners diverge, not an expected-to-fail. D10's probe question does not route,
so a case pinning it needs a routed question this measurement has not yet found.
Both are owed before their fixes, not before this one.

---

# Result, measured against §6

## Against the prediction

| predicted | measured |
|---|---|
| five answers move, all `ok` → refuse, all routed | **five moved, exactly the five named** |
| the eleven `geo_exposure` claims stay APPLIED | **all eleven**, now proven from the declaration |
| `rt_007`/`rt_008` keep both grouping facets | **`rt_008` yes; `rt_007` NO — and the expectation was wrong.** See below |
| no point-in-time answer moves | **none** |
| lexical 693 of 693 | **693 of 693** |
| seasoning by name, both books | **Q1 4, Q7 4, Q8 12 all CORRECT, both books** |
| robustness `32/10/2` both books | **`32/10/2` both books; 44 of 44 same verdict** |
| stamping matrix 0 live holes | **17 holes, 17 designed, 0 live** |
| `rt_013` flips to passing | **NO — and the case itself was wrong.** See below |

```
answer diff      693 compared, 689 identical, 4 MOVED   (a FOURTH surface, added here)
routed surface   18 of 18            (5 declared expected-to-fail)
robustness       32/10/2 both books
calibration      249/249 generated; 255/255 curated held, 0 hard fails, 0 gaps
lexical          693 of 693
qi tests         316 passed
```

## The five, by name

| id | before | after |
|---|---|---|
| `nneg_er_005` — Show NNEG exposure by borrower age bucket. | `age_bucket: applied`, seven concentration tables, none by age bucket | `age_bucket: lost`, refuses |
| `nneg_er_006` — Show NNEG exposure by vintage. | `vintage_year: applied` | `vintage_year: lost`, refuses |
| `vintage_cohort_018` — Show 2025 vintage concentration. | `vintage_year: applied` | `vintage_year: lost`, refuses |
| `risk_216` — concentration by account status | `account_status: applied` | `account_status: lost`, refuses |
| `risk_limits_013` — Show concentration limit status. | `account_status: applied` | `account_status: lost`, refuses |

Four of the five are unambiguous: a reader asking for a vintage breakdown was
being shown amortisation types and told it was vintages. The fifth is the
predicted fail-closed case and is filed as **B18**, below.

## Four corrections, all of them to my own work

### 1. The answer differ could not see this commit, and has been extended

It reported **343 of 343 identical** while five answers moved on the shipped
path. Not a bug — a corpus gap, and a serious one:

* its calibration half calls `run_mi_agent_query` directly and is therefore
  always **point-in-time** (B7), so a routed-only change is structurally
  invisible there;
* its robustness half does route, but is 44 sentences.

So **every one of the three standing surfaces was blind to a change that moved
five real answers.** The routed surface saw the mechanism (18 hand-picked cases)
and no instrument saw the answers.

Fixed here: `answer_diff` gains a fourth surface, `service_path` — the 350-question
`ere_mi_questions` corpus driven through `execute_governed_mi_query`, the same
entry point the routed surface uses, over every question rather than eighteen.
Recorded at 693 and demonstrated to fail: run against the pre-D7 baseline it
reports **689 identical, 4 moved**, all on the new surface.

**A residual blind spot remains and is stated rather than closed.** `risk_216`
lives in the calibration corpus, whose questions are never driven through the
service, so its movement was measured directly and is still invisible to every
standing instrument. The fifth answer is real and no surface reports it.

### 2. The prediction missed `analytical_composition`, and the surface caught it

`rt_010` — *"How does the front book compare with our older lending from a risk
perspective?"* — went from `ok` to refuse: `seasoning_segment` was certified on
the removed rung.

My tier survey covered the answer corpus and the calibration corpus. **Q7 and Q8
live in the robustness bank, which it did not cover** — so the seasoning
comparison family, the family `32c263a` broke, was outside the measurement that
scoped the change. The routed surface is what caught it.

The plan does declare, and the reader was not reading it:

```
analytical_evidence.narrowedTo
  [{field: seasoning_segment, value: Front Book, rows: 1177},
   {field: seasoning_segment, value: Back Book,  rows: 9858}]
```

**Two or more governed populations of one field IS a breakdown by that field**,
with two groups; one is a filter, which is what `row_population` records. That
threshold is `_two_or_more_populations`, and both halves of it are pinned.

### 3. `rt_013`'s declared correct outcome was itself wrong

Written in D2 against *"Are any regional limits breached?"*, believing
`risk_limits` has no dimension axis at all. Reading the limit-test rows shows
otherwise — the tests ARE per region: London 21.1% against 25.0%, South East
26.3% against 30.0%, Scotland 3.3% against 8.0%. **Certifying a geography
breakdown there is true**, and the declared correct outcome of *no facet* was
wrong.

Re-pointed at the genuinely false member, *"Show concentration limit status."*,
and the true case is now `rt_013b` as an ordinary passing one. **A declared
expected-to-fail states a correct outcome, and mine was not correct** — which is
the failure mode that instrument has to be watched for, since a wrong
expectation flips a real fix into a reported regression.

### 4. `rt_007` was pinning a false certification, and I removed one rung too many

Two separate things, found together.

`rt_007` — *"show geographic exposure by postcode"* — expected **both** grouping
facets applied, and the second is `postcode`: a dimension `geo_exposure` does not
publish and cannot break down by. It was applied only on the removed rung. **A
ninth false certification, not in my count of eight**, because the survey did not
cover the routed surface's own questions. Expectation corrected to `lost`, which
is what the granularity facet beside it has always said in words.

And: my first cut removed the **name-match** rung as well as the residual one.
Two tests in `test_routed_grouping_evidence.py` failed, correctly — a result
column named by REGISTRY FIELD is execution naming the field, and it is the same
evidence the point-in-time path accepts through `result_cols`. Restored. It fires
zero times on these corpora, which is exactly why the measurement could not
distinguish the two rungs and the tests could. **The corpus said "one rung"; the
tests said "two". The tests were right.**

## What landed

* `grouping_proven` — the one decision, both paths.
* `declared_group_fields` — four declaration sources, in order: the concentration
  workflow's `dimension_results` (already on the envelope, never read), a route's
  `metadata.groupedBy`, `rankedMovement.canonicalField`, `ROUTE_DECLARED_AXES`,
  plus the answer's own axis keys where they name a registry field.
* `_two_or_more_populations` — two governed populations of one field are a
  breakdown; one is a filter.
* `risk_limits` derives `metadata.groupedBy` from the fields its **executed**
  tests cover, so a limit reported unavailable certifies nothing.
* `ROUTE_DECLARED_AXES` for `geo_exposure`, checked against a real envelope by
  `test_declared_axes_match_what_the_route_publishes` rather than left as a
  comment.

Four mutations were applied and each was caught by the intended test: restoring
the residual rung, not reading the concentration declaration, counting one
population as a breakdown, and dropping the `geo_exposure` declaration.

## Two new backlog entries

**B17 — a drill-through on a routed question always refuses.** `material_predicates`
is computed from `parsed.spec.filters` before `try_route` calls
`parsed.merge_filters(extra_filters)`, so the narrowing never reaches the frame
but is on the spec when the guard reads it. Fail-closed, wrong cause. Belongs to
**D8**. Pinned as `rt_016`.

**B18 — "limit status" resolves to `account_status`.** A term over-match, not a
role or an evidence defect. It is why `risk_limits_013` now refuses a correct
answer: wrong reason, safe outcome, no wrong number. Pinned as `rt_013`.
