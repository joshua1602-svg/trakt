# T3 now, or T3 at Phase 5 — both options scoped

**Scoping only. Nothing built, nothing chosen, Phase 0 not started.**

Follows `docs/mi_compositional_plan_scoping.md`. One new read-only instrument;
no product module changed.

```bash
python -m compositional_plan_scoping.t3_now
```

**On the phase numbering.** This study did not define phases. I have read
"Phase 5" as *the `evolution` route conversion* — the one my §6 named as
delivering T3–T5 and deliberately ordered last — and "Phase 0" as the
preparatory work before the first conversion. **Where the answer depends on what
Phase 4 specifically is, §5.3 tests it against every candidate rather than
assuming one.** Correct me if the mapping is wrong; it changes §5.3 and nothing
else.

---

## 1. The premise does not survive contact

The proposition was: *the evolution route already computes a full period-by-region
series — 516 rows measured — and discards it, so T3 may be a small carriage
change.* The 516 rows are real. **Everything else in that sentence fails, in
three independent places.**

### 1.1 The route does not know a dimension was asked for

Four questions, four specs, measured:

```
balance over time                     chart='line' dimension=None dimensions=[]
balance over time by region           chart='line' dimension=None dimensions=[]
balance over time by LTV band         chart='line' dimension=None dimensions=[]
balance over time by broker           chart='line' dimension=None dimensions=[]
```

**The four specs are identical.** `_route_evolution` receives the same object for
all of them. There is no carriage change available inside the route, because
there is nothing arriving to carry. The dimension is destroyed at parse — in the
line branch, which is the finding from the first study (§2.2 there).

So the work is not in the route. **It starts in the parser** — the layer that
study named as the largest unbounded unknown and put outside the migration's
route-conversion scope.

### 1.2 The discarded series is cut at the wrong granularity

```
geographic_region_obligor   172 categories,  516 rows / 3 periods   what the breakdown COMPUTES
collateral_geography         12 categories,   36 rows / 3 periods   what the request RESOLVES to
```

`_FUNDED_BREAKDOWN_DIMS["region"]` points at `geographic_region_obligor` — the
**172-value ITL3 code column**. The request `balance over time by region`
resolves to `collateral_geography` — the **12-value readable region**. Same
word, different level.

The 516 rows are not T3's answer at the level the rest of the product calls
"region". T3's answer is 36 rows, and **nothing computes it today.**

### 1.3 The receipt would refuse the discarded rows anyway — correctly

```
the request raises: kind='grouping_dimension' label='region' satisfied_by=('collateral_geography',)
grouping_proven(groupedBy=['geographic_region_obligor']) -> False   still LOST — the answer refuses
grouping_proven(groupedBy=['collateral_geography'])      -> True    APPLIED
```

Publishing the discarded rows produces a **still-refused answer**. The guard is
already right, and — worth saying, because it is the good news here — it does
**not** produce a false certification at the wrong level either. `grouping_proven`
is purely declaration-based, and for `KIND_GROUPING` the governance layer is
already shaped the way a compositional layer would need it: `metadata.groupedBy`
is a per-step declaration, not a route-identity allowlist.

This case is not hypothetical to the codebase. The comment in
`reconcile_routed_facets` names it:

> *"`evolution` publishes a frame whose rows carry `period` and `value` and
> nothing else, so 'balance by month BY REGION' returned the whole-book series
> with the receipt vouching for a regional breakdown never computed."*

It was a false-APPLIED, it was closed, and the closure is what refuses T3 today.

**And the shortcut is already written down as an anti-pattern.**
`granularity_facets`, the single owner of reporting grain, says:

> *"Duplicating the call into `evolution` — the obvious short path to making
> time-series questions work — would be a twelfth reader and would defeat the
> premise."*

---

## 2. What landing T3 now actually takes

Six pieces. Ordered as they would have to be satisfied.

| # | piece | measured cost | where it lands |
|---|---|---|---|
| 1 | **Parser carries the grouping dimension on the line branch** | 11 of 677 bank specs move (1.6%) | `llm_query_parser` |
| 2 | **A grouping/subject role decision** | 3 of those 11 recover the question's *subject*, not its grouping | `llm_query_parser`, no role slot exists |
| 3 | **Compute the breakdown at the requested granularity** | 36 rows, not the 516 that exist | `evolution` / route-local |
| 4 | **A top-N-with-residual policy** | region = 12 series, ITL3 = 172; `_breakdown` has no residual policy | 3rd instance in the tree |
| 5 | **Declare `metadata.groupedBy`** | one line; `grouping_proven` already accepts it | `_route_evolution` |
| 6 | **Multi-series chart + table artifact** | the mechanism exists (`evolution_pipeline_stage` builds N series) | reuse |

### 2.1 Piece 2 is the one that bites

Of the 11 specs a line-branch dimension carry would move, **3 recover the
question's subject rather than its grouping**:

```
[SUBJECT-SIDE] ['age_bucket']    <- 'Show average borrower age evolution by month.'
[SUBJECT-SIDE] ['ltv_bucket']    <- 'Show LTV bucket evolution over time.'
[SUBJECT-SIDE] ['age_bucket']    <- 'Show age bucket evolution over time.'
```

The first is an unambiguous regression: the metric is `youngest_borrower_age`
with `avg`, and carrying `age_bucket` turns *average borrower age over time* into
*average borrower age by age bucket* — a tautology, replacing an answer that
works today. The other two are genuinely ambiguous: *"LTV bucket evolution"* may
mean the weighted-average LTV trend or the bucket-mix trend, and the sentence
does not say.

**The parser cannot resolve this, structurally.** It has no role slot.
`question_interpretation.schema.DimensionClaim.role` — `grouping` / `filter` /
`unresolved` — is the owner of exactly this distinction, and the deterministic
parser does not read it. So piece 2 either ships with a known misclassification
on 3 of 11 moved specs, or it waits on the interpretation contract.

**That is the same prerequisite the migration has** (first study §5). It is not
a cost Option A avoids by going first.

---

## 3. What Option A would deliver

Measured against the standing phrasing banks, assuming all six pieces:

```
T3:  7/8 phrasings reach the evolution route AND carry a dimension   (today: 0/8)
T4:  3/7                                                              (today: 0/7)
```

and, from §1.2 / §2:

* **2 of 25** dimensions this book can be cut by — `region` and `ltv_bucket`. `broker_channel` is not on this tape and `_breakdown` returns `[]` **with no disclosure**.
* **Unfiltered only.** The four T4 phrasings that fail do so because a seasoning population (*"for the front book"*) makes the parser's two-dimension branch claim the question — `_is_evolution` is `False` and the evolution route is never reached. Worse, `_explicit_dimensions` returns `seasoning_segment` as a *grouping* for those, which is the population-as-grouping conflation and would be a fresh defect if carried.

So the honest client-visible delivery is: **`balance over time by region` and
`balance over time by LTV band`, unfiltered, at 7 of 8 phrasings.** That is a
real capability the product does not have. It is materially narrower than "T3".

---

## 4. What Option B costs

Option B is: leave T3 refused, deliver it when the `evolution` conversion lands.

* **Client-visible delivery before then: none.** That is the premise of the question and it is correct.
* **The refusal is honest.** T3 does not answer wrongly today; it names what it could not apply. The cost is a capability gap, not a correctness risk.
* **The gap is measured**: T3–T6 hold 23 of 27 capability failures, and T3 is 0/8.
* **`evolution` is the largest conversion in the table** — 14 test files, 39 references, 167 handler lines — and the first study deliberately ordered it last, because converting it first proves the mechanism and claims the capability in one commit.

Option B's real cost is not technical. It is that the migration's first four
phases are all mechanism and no capability, against a client who has not yet
supplied a tape or a question in their own words.

---

## 5. Does landing T3 first create work the migration would undo?

### 5.1 Plainly: yes, some of it.

Pieces **3 and 4** — computing a per-period breakdown at a chosen granularity,
and a top-N-with-residual policy — are a **route-local implementation of
`stack ∘ group ∘ measure ∘ rank`.** That is the composition itself, built inside
`_route_evolution`.

Concretely, it would become:

* the **5th** implementation of `group` (the first study counted 4), and consolidating those 4 into 1 is the migration's **dominant cost**;
* the **3rd** top-N-with-residual policy (`funded_bridge`'s `top_n=8` + `"Other"`, and `_apply_top_n`);
* a **second owner** of the geography-granularity decision, which `geo_exposure` already owns through `ROUTE_DECLARED_AXES` — the dual-mechanism pattern this repo has a standing document about.

Sizing it by the nearest existing analogue: `funded_bridge` (`stack ∘ select ∘
group ∘ measure ∘ compare ∘ rank ∘ residual`) is **74 lines** in `evolution.py`
plus **80** in its route handler. T3 needs less than that — no `compare` — and
more rendering. Call it 60–120 lines, and note that is an analogy, not a
measurement.

**Every one of those lines is deleted by the `evolution` conversion.** Under your
stated rule, that settles it against.

### 5.2 But two of the six pieces are not undone, and one of them is a prerequisite either way

I would be misreporting if I stopped at 5.1.

* **Piece 1 and 2 — the parser carrying a dimension, with a role decision — are not undone by any route conversion.** The parser is upstream of the router, and the first study found its conversion unestimated and out of the migration's route scope. This work has to happen for T3 to exist under *either* option.
* **The T3 acceptance tests are not undone.** They become the byte-identical bar the `evolution` conversion has to clear. The migration needs that bar and does not currently have one for T3.

So the accounting is: **pieces 3–4 are thrown away; pieces 1–2 and the tests are
brought forward.** The question is whether ~60–120 lines of discarded route-local
composition is worth the client-visible capability plus the parser incision
arriving early.

### 5.3 Tested against every candidate for Phase 4

Since I do not know what Phase 4 is, here is the answer for each candidate the
first study's §8 implies:

| candidate Phase 4 | does it undo T3-now? |
|---|---|
| the conversion immediately before `evolution` (`funded_bridge` or later) | **No.** Different route. T3-now's route-local code survives untouched until Phase 5. |
| **consolidating the `group` implementations** | **Yes, directly.** T3-now adds a 5th to the pile Phase 4 exists to clear, and it would be in the newest, least-settled code. |
| **moving the facet layer off route identity** | **No, and it helps.** §1.3 shows `KIND_GROUPING` already runs on per-step declaration; T3-now would add one more correct `groupedBy` declaration, not another route literal. |
| **generalising the arity-1 disclosure rules** | **No.** T3 is arity 1 within each period. It does not touch the arity-2 gap. |

**One candidate makes it strictly worse, two are neutral, one is mildly
positive.** If Phase 4 is the `group` consolidation, 5.1 is decisive and there is
nothing to weigh against it.

---

## 6. What this study does not say

* It does not choose. Both options are scoped and neither is recommended.
* It does not estimate the parser change beyond its blast radius (11 of 677 specs, 3 of them wrong). What it takes to make the parser read a role is not scoped here, and the first study did not scope it either.
* It did not measure the LLM arm. Every number here is deterministic.
* It assumes the phase mapping in the preamble. §5.3 is the only section that depends on it.
* **Phase 0 has not been started**, per your instruction.
