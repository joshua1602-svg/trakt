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
