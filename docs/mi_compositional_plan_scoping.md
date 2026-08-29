# The compositional plan layer — a scoping study

**Scoping only. Nothing is built, nothing is chosen, no design is proposed.**
Two read-only instruments and this report. No product module is changed, no
route is converted, no flag is added.

Base: `42cef00` (*Root cause 1: widen the time-axis vocabulary, and put it under
one owner*), on `claude/clause-splitting-phase-1-cft1wx`.

Reproduce:

```bash
python -m compositional_plan_scoping.census                    # structural counts, from source
python -m compositional_plan_scoping.compose                   # the composition test, on a live book
python -m compositional_plan_scoping.compose --book kestrelmoor
```

**Follow-up:** `docs/mi_t3_now_versus_migration.md` scopes landing T3 under
today's architecture against waiting for the migration. It corrects one reading
of §2.1 below: the discarded per-period series is cut at ITL3 (172 categories),
not at the readable region level (12) the request resolves to, so it is not
T3's answer as it stands.

---

## 0. The finding, before the detail

**The routes factor.** Every shape in the T3–T7 family — which holds 23 of the
27 measured capability failures — composes from primitives the product **already
owns, unmodified**, and each composition reconciles to the shipped answer **to
the penny on both books**. Nothing needed inventing.

**Three routes do not factor**, and they cluster: they are the pipeline and
limits routes, not the funded-book routes.

**But the diagnosis in the brief is one layer too low, and that changes the
shape of the work.** The brief says *the spec cannot hold both limbs*. It can:
`MIQuerySpec` carries `dimensions`, `hierarchy`, `trend_grain` and
`temporal_mode` simultaneously. What is true is stronger and worse:

> **0 of 29** `MIQuerySpec` constructions in the deterministic parser bind a
> reporting-period axis and a grouping dimension together. Not because the spec
> forbids it — because the parser is a **13-branch shape cascade** in which time
> and grouping are owned by *different, mutually exclusive branches*, each of
> which early-returns a finished spec.

So there are not thirteen routes. There are **three shape cascades** —
13 parser branches, 15 router recognisers, 14 executor branches, **42 shape
decisions in total** — and the router is the middle one. Converting routes
without converting the parser converts the layer that was never the binding
constraint.

---

## 1. The decomposition

### 1.1 The primitives, derived

Seven. The brief's first guess was eight; the decomposition keeps five of them,
dissolves two, and splits one.

| primitive | does | implementations found |
|---|---|---:|
| **select population** | narrow the row set by a governed predicate — row filter, portfolio lens, seasoning segment, dimension value | 5 |
| **resolve measure** | one governed aggregate over one frame | 5 |
| **group** | partition a frame by *n* governed dimensions | 4 |
| **stack periods** | replace one frame with an ordered sequence of governed snapshot frames | 2 |
| **compare** | two values → absolute, relative, directional verdict | 3 |
| **rank** | order a grouped result by a basis; truncate to top-N with a declared residual policy | 3 |
| **project** | forward estimate from a fitted run-rate, gated and caveated | 1 |

**7 primitives, 23 implementations — 3.3 implementations per primitive.**

**Report the number before proposing anything**, as instructed: **seven**. That
is in the range the brief called "probably right", and the routes do express in
them. But the second number is the one that governs cost. Nothing has to be
*invented*; three to five existing implementations of each primitive have to be
**consolidated into one**, and consolidation is where a byte-identical migration
can fail. See §7.

### 1.2 What the decomposition corrected in the first guess

Three corrections, each earned by the code rather than by preference:

**`movement` is not a primitive. It dissolves into `compare`.** T6
(*period-over-period movement by segment*) is `stack ∘ group ∘ measure ∘ compare`
applied element-wise across two grouped results. `mi_workflows.engine.compare_values`
already computes it and is documented as the platform's only comparison formula.
The instrument builds T6 this way and it works. Removing `movement` removes a
primitive that would have duplicated an existing one.

**`partition by period` narrows to `stack periods`.** The code has two things
wearing that name and they are not the same operation:

* `evolution.funded_frames` reads **N governed snapshot frames** — the reporting-period axis. This is the primitive.
* `_execute_line` does `pd.to_datetime(col).dt.to_period("M")` on a **date column inside one frame**. That is a grouping over a date-derived bucket, and it collapses into `group`.

They are routinely confused *in the product*: for `balance over time`, the
deterministic parser emits `x="origination_date"` — a **vintage** axis — while
the `evolution` route answers with the **reporting-period** axis. Two different
questions under one word. Keeping them as one primitive would carry that
confusion into the plan layer.

**`select population` must be non-destructive.** `share` and `contribution` need
the *pre-narrowing* frame as well as the narrowed one —
`_execute_share(spec, df, work, ...)` takes both, and its own comment says so:
*"A share needs BOTH populations."* A plan is therefore **not a linear chain of
`frame → frame` steps**. It is a small dataflow graph in which the base frame
stays addressable. That is a real constraint on §3.1's model 2 and is stated
here rather than discovered during a migration.

### 1.3 Routes that do not decompose

Four entries, three genuinely distinct, and they share a property: **every one
of them reads the pipeline or limits datasets, not the funded book.**

| route | what it needs that no primitive expresses |
|---|---|
| `cohort_conversion` | an **entity timeline** joined across weekly snapshots on a stable case identifier (`mi_agent_api/pipeline_history.py`). A longitudinal reshape of the row set, not a grouping of it. Names a genuinely missing primitive: **track entities across snapshots**. |
| `scenario` | a **hypothetical parameter** read from the question (*"what if conversion improved 10%"*), substituted into a fitted model before it is re-solved. Not a population, not a measure, not a dimension. |
| `forecast_extrapolation` (`threshold_projection`) | the **inverse of `project`** — solve for the date a projection crosses a level. A forward composition cannot express a solve. |
| `risk_limits` | its *actuals* decompose cleanly (`select`/`measure`/`group`/`rank`/`share`), but the route is **driven by rows of an external governed limits schedule**, and `_headroom`/`_status_for` compare a value against a **contractual threshold with a declared direction**. `compare` relates two values from the same book. |

This is the single cleanest statement the study produces:

> **The primitives factor the funded book completely and the pipeline and limits
> datasets not at all.**

That is a scoping boundary, not a defect. It says a compositional layer can be
scoped to the funded book and leave four routes exactly where they are — and it
says that a layer scoped to "everything" is a materially larger and less
certain piece of work than one scoped to the funded book.

---

## 2. T3–T7 expressed as compositions — measured, not asserted

`python -m compositional_plan_scoping.compose`. Every callable used already
ships and is used **unmodified**. Alderbridge, 3 governed snapshots:

```
  T1  stack x measure                               3 rows
  T3  stack x group(region) x measure             516 rows
  T4  stack x select x group(region) x measure     496 rows   (9.1% of the book by balance)
  T5  stack x group(region, ltv) x measure       2882 rows
  T6  compare(T3[-1], T3[-2]) elementwise         172 categories
  T7  rank(T6)                                 top 3 by |change|
```

Then the migration discipline this programme uses — sum each composition back
over the dimensions it added, against the answer the product **ships today**:

```
  T3:  2026-04  composed=1,932,310,991.20  shipped=1,932,310,991.20  delta=0.0000  OK
       2026-05  composed=1,946,827,440.60  shipped=1,946,827,440.60  delta=-0.0000 OK
       2026-06  composed=1,964,886,258.21  shipped=1,964,886,258.21  delta=-0.0000 OK
  T5:  identical on all three periods.
```

**Kestrelmoor reconciles exactly too.** Both books, every period.

**The case the brief set is met: they express cleanly, and only the executor is
missing.** No primitive that no existing route uses was required.

### 2.1 The sharpest evidence: T3 is computed today and thrown away

`_route_evolution` calls `evolution_mod.funded_evolution(output_root, client_id,
run_id)`. That function's `breakdowns` argument defaults to
`["broker", "region", "ltv_bucket"]`, so **every `evolution` answer computes a
full `period × region × balance` series** — measured: **516 rows across 3
periods on Alderbridge**. The route then builds
`rows = [{"period", "value"}]` from `evo["periods"]` and discards it.

Measured end to end on the chat path:

```
T1  "balance over time"            ok=True   — answers, and computes T3's data in the same call
T3  "balance over time by region"  ok=False  — "I understood that you asked for region,
                                               but that could not be applied…"
```

T3 is not a missing capability. It is a **computed result with no channel to
the answer**, because the route's handler hardcodes its output shape. That is
the brief's diagnosis, evidenced at the strongest possible level.

### 2.2 The T5 mechanism, precisely

`Show me balance by month by region and LTV band` reaches the parser's
**two-dimensional grouped branch** (line 2783), which fires *before* the line
branch (line 2933) and returns `chart_type="heatmap"`, `dimensions=[region,
ltv_bucket]` — **with no time axis at all**. `balance over time by region for
the front book` does the same: the seasoning clause makes two segments, the
two-dimension branch claims it, and time is gone.

Meanwhile `balance over time by region` reaches the **line** branch, which
returns a spec that never sets `dimension`, and the executor's `_execute_line`
groups by `[period_col]` only — it even reads `spec.dimension` *as the date key*
(`date_key = spec.x or spec.dimension`), so a grouping dimension arriving there
would be misread as the time axis.

Both limbs are representable. **The branch is the shape, and one question gets
one branch.**

### 2.3 What arity costs — and this one is not good news

A composition can be arithmetically exact and still be a worse answer:

```
  T3  group(region)             516 groups        0 thin     0.0%     (Kestrelmoor:  0.0%)
  T5  group(region, ltv)       2882 groups     1152 thin    40.0%     (Kestrelmoor: 11.8%)
```

*thin* = below the product's own `LOW_GROUP_COUNT` floor of 5 loans.

And the disclosure that exists for this is guarded by `len(group_cols) == 1`
(`mi_agent/mi_query_executor.py::_execute_grouped`). The `loan_count`
denominator column and the thin-sample warning are both written for **arity 1**
and **do not fire at arity 2** — which is the first arity a compositional layer
unlocks. On Alderbridge that means 40% of T5's cells would ship thin, uncounted
and undisclosed.

Stated exactly, because overstating what exists is the failure mode here: at
arity 1 the `loan_count` column is always attached, and the thin-sample warning
is *additionally* gated on `agg in ("avg", "weighted_avg")` — so a summed
measure has never raised it. What arity 2 removes is the denominator column
outright, and with it any basis on which the warning could be generalised.

**This is the §5 property failing on the very first composition the layer would
make reachable**, and it is the most actionable finding in the study: the
disclosure rules are arity-1 rules, and generalising them is work that is
currently invisible in every estimate.

---

## 3. The three governance models, assessed

### 3.1 Model 1 — allowlist of shapes

The brief's own assessment holds and the measurement supports it: this is the
route problem renamed. Nothing to add except the size — an allowlist that
covered today's reachable shapes would need at least the 42 shape decisions
already in the tree, and T5 exists precisely because a shape was not on a list.

### 3.2 Model 2 — per-primitive contracts plus composition rules

The mechanism **already exists and is exercised**:
`mi_workflows.analytical.contract.AnalyticalCapability` declares
`required_inputs`, `optional_inputs`, `supported_scopes`, `datasets`,
`produces`, `limitations`; `validate_inputs()` runs before any data is touched;
`planner.validate()` rejects an unusable plan. Ten capabilities are declared.

Three things the measurement says about extending it downward to primitives:

* **The chain is not linear.** `share`/`contribution` need the pre-narrowing frame (§1.2), so "the chain type-checks" is a graph property, not a sequence property.
* **`stack periods` changes the frame count**, so every downstream step must be *lifted* over a sequence. That is a second kind of type, not a second input name.
* **Cardinality is undecidable before execution.** T5's 2,882 groups and 40% thin rate are not knowable from the registry — only from the data. A pre-execution validator, which is the whole safety claim, structurally cannot decide the question §2.3 raises.

### 3.3 Model 3 — governance at the plan

The brief says model 3 *follows from what already exists* and asks whether it
holds when the composition is arbitrary. **The measurement says it currently
holds because the routes are fixed, and the evidence is specific:**

> `mi_agent/execution_receipt.py` names the 15 governed routes **literally 54
> times**, and keeps **7 route allowlists**: `TEMPORAL_ROUTES`, `RANKING_ROUTES`,
> `COHORT_ROUTES`, `PROJECTION_ROUTES`, `SCENARIO_ROUTES`, `LISTING_ROUTES`,
> `SHARE_BEARING_ROUTES`.

`reconcile_routed_facets` decides *"was this facet honoured?"* partly from
**which route answered**, not only from what the step declared it applied. Two
examples from the source:

* a `KIND_STATISTIC` facet is stamped `APPLIED` unconditionally on any routed answer, with the reason given as *"Specialist routes publish no statistic evidence"*;
* a `KIND_SHARE` facet is `APPLIED` when `route in SHARE_BEARING_ROUTES` and the prose *states a proportion*.

Neither test survives a plan composed of primitives, because there is no route
to name. Model 3 is the right shape — the facet layer already tracks
applied / unavailable / lost per concept, and `RequestedFacet.status` already
has the `LOST` fail-closed state the compositional version needs — but the
honour *evidence* would have to move from **route identity** to **per-step
declaration**, on 54 sites and 7 allowlists.

That is not a reason against model 3. It is the cost of model 3, stated, and it
has not appeared in any estimate so far.

---

## 4. Who composes the plan

The safety property the brief wants preserved is **already implemented, and
already partly violated.**

**Implemented.** `planner.plan_for()` is deterministic;
`planner.plan_from_proposal()` accepts an LLM-proposed plan and puts it through
`validate()` plus `populations.assert_not_fabricated()` before anything runs; a
model may name a governed population but `_population_from_name` accepts only
names the governed resolvers already own. Exactly the `parse_with_repair` shape.

**Not in service.** `plan_from_proposal` is called by **five tests and no
production path**. The validator ships; the proposer does not. So "the model
proposes a plan" is currently an untested-in-service claim, and this study
should not be read as evidence that it works.

**Already violated.** The brief's §3.3 says *do not let the plan layer re-read
the question*. `planner.plan_for()` does:

```python
text = _norm(question)
if _any(text, _EXPORT_TERMS): return None
reading = intent_mod.classify(question, spec=spec)
```

And so do 6 of the 14 route handlers (`_route_evolution` reads the raw question
5 times) and 11 of the 15 recognisers. Three of the five `select population`
implementations — `portfolio_lens.resolve_lens(text)`,
`seasoning.resolve_population_predicate(text)`,
`portfolio_lens.resolve_comparison_lenses(text)` — take **raw question text**,
not a parse.

So the defect the programme has spent a month removing is **not yet removed at
the plan layer**; it is present in the layer that would become the plan layer.
A compositional migration either fixes that or institutionalises it.

### 4.1 What validating a plan requires that validating a spec does not

`validate_mi_query` is: enum membership, field existence in the registry,
canonical column presence in the dataset, chart-structure sanity. **All local,
all decidable against a registry and a column list, no data touched.**

A plan validator additionally needs:

1. **Step composition typing** — does step *N*'s output satisfy step *N+1*'s input? (`rank` needs a grouped result; `compare` needs two values of the same measure and unit.)
2. **Frame-count lifting** — `stack periods` turns 1 frame into *N*; every later step must be valid *under the lift*.
3. **Dataset compatibility** — a plan mixing funded and pipeline steps needs a declared join basis. `AnalyticalCapability.datasets` is the beginning of this and nothing consumes it for cross-step checks.
4. **Non-destructiveness** — a plan that narrows and then asks for a share must keep the basis frame addressable.
5. **Cardinality and sparsity** — **and this one cannot be validated before execution** (§2.3, §3.2). It is the first check in the product's history that a pre-execution validator cannot perform.

---

## 5. What the plan layer needs from the interpretation contract

`question_interpretation/schema.py` is the right shape and sits in the right
place — downstream. It carries what a planner needs for the funded-book
compositions: `dimensions` is a **list** with per-item `grouping`/`filter`
roles, so **T5's two limbs are already representable at interpretation**, and
`TimeClaim` already separates `requested_grain` (the axis) from `trend_window`
(the narrowing) from `comparison_period`. That separation is exactly what the
parser's line/heatmap branches conflate.

Five things are missing, each stated with what it blocks:

| missing | blocks | evidence |
|---|---|---|
| **the statistic** | `resolve measure` cannot pick an aggregation. `OperationClaim.type` carries the answer *type* (`count`/`amount`/`average`), not the governed statistic — and statistic identity is a live governed concern (`mi_agent/statistic.py`, `KIND_STATISTIC`, P1M). | no slot in the contract |
| **rank parameters** | `rank` has no direction, no *N*, no basis. `OPERATION_TYPES` has `RANKING` and nothing else. | `schema.py` |
| **comparison pairing** | `population` is a flat `List[PopulationClaim]`; it cannot say *"these two are the sides of a comparison"*. T8 needs it, and `portfolio_lens.resolve_comparison_lenses` supplies it today **from the raw question**. | §4 |
| **joined filter clauses** | a predicate needs field + operator + value. The contract records **halves**: on **71 of 690** questions one clause is read twice, wording half located by span **71/71**, binding half located **0/71**, `clause_id` set **0**. | `docs/mi_question_interpretation_stage2.md` |
| **spans on a quarter of claims** | precedence between two overlapping claims is undecidable without offsets. **120 of 939** claims still have none (down from 215). | same |

The `clause_id` gap is the load-bearing one: **a planner cannot build a
`select population` step from a contract that holds unjoined halves.** Stage 2
named the fix (`_parse_filters` rewrites its working string, so offsets index a
mutated buffer) and deferred it to Stage 3 as a consumer conversion. **That
conversion is a prerequisite for a plan layer, not a parallel workstream.**

Nothing in the contract needs *removing*, and the plan layer must not re-read
the question to fill these gaps — see §4.

---

## 6. Migration order

The brief asks for the route whose composition is simplest and whose blast
radius is smallest, **not the one whose capability is most wanted**. Measured
(`census.py` §4):

| route | test files | test refs | handler LOC | receipt literals |
|---|---:|---:|---:|---:|
| `cohort_conversion` | 3 | 4 | 50 | 4 |
| `concentration_analysis` | 4 | 5 | 98 | 5 |
| `portfolio_risk_comparison` | 4 | 9 | 199 | 2 |
| `temporal_compare` | 6 | 10 | 63 | 3 |
| `funded_bridge` | 7 | 8 | 80 | 5 |
| `portfolio_summary` | 7 | 21 | 94 | 0 |
| `forecast_extrapolation` | 7 | 14 | 148 | 5 |
| `cohort_progression` | 8 | 9 | 77 | 4 |
| `period_change_analysis` | 8 | 22 | 111 | 3 |
| `period_movement` | 9 | 15 | 159 | 3 |
| `scenario` | 10 | 18 | 79 | 5 |
| `risk_limits` | 13 | 22 | 119 | 4 |
| `evolution` | 14 | 39 | 167 | 3 |
| `geo_exposure` | 15 | 29 | 94 | 7 |

**The order the numbers give:**

**First — `portfolio_summary`.** Not the smallest by test files, but it is the
only route in the table with **zero receipt literals**: the facet reconciler
does not name it, so converting it does not touch the governance surface at all.
Its composition is the shortest possible (`select population ∘ resolve measure`,
no group, no stack, no compare) and it already returns `None` to defer on
failure, so the fall-through is proven. It is the conversion that tests the
*mechanism* while exercising the *fewest* of its parts — which is what a first
conversion is for.

**Second — `temporal_compare`.** `stack periods ∘ select ∘ measure ∘ compare`,
6 test files, 63 handler lines, 3 receipt literals. It is the first conversion
that exercises the frame-count lift (§4.1 item 2), which is the riskiest typing
property, on the smallest route that has it.

**Third — `funded_bridge`.** Adds `group`, `rank` and the residual-preserving
top-N in one step, 7 test files, 80 handler lines. It is where the "top-N that
still reconciles" policy gets proven, and its own reconciliation constraint
(deltas sum exactly to close − open) makes an unattributable movement impossible
to miss.

**Not first, despite being the prize — `evolution`.** 14 test files, 39
references, 167 handler lines, and it is the route whose conversion actually
delivers T3–T5. Converting it first would mean proving the mechanism and
claiming the capability in the same commit, with the largest test surface in the
table. Everything the brief says about not rewriting applies most sharply here.

**Excluded from the ordering entirely:** `cohort_conversion`, `scenario`,
`forecast_extrapolation`, `risk_limits`. They do not decompose (§1.3), and the
smallest blast radius in the table belongs to one of them —
`cohort_conversion`, at 3 test files — which is exactly the trap a
blast-radius-only ordering would walk into.

---

## 7. Testing

### 7.1 How the estate changes

* **Primitives tested exhaustively, in isolation.** Feasible, and cheaper than it looks: 7 primitives with a small parameter space each. The real work is that **23 existing implementations must first collapse to 7** (§1.1), and each collapse is a byte-identical equivalence proof against every route that used the implementation being retired.
* **Composition rules tested as rules.** Feasible for typing (§4.1 items 1–4): they are static properties over declared contracts, testable without data — this is what `AnalyticalCapability.validate_inputs` already does, one level up.
* **A representative set of compositions, end to end.** The criterion should be stated from evidence, not chosen: **the 61-phrasing bank × the 8 measured shapes × 2 books** is a standing surface that already exists (`question_interpretation/mi_recognition_diagnosis.py`, `time_series_surface.py`) and already carries the both-books discipline. It is a criterion the programme has already earned rather than a new one.
* **A property, not a list.** Discussed below.

### 7.2 Can the property in §5 of the brief be established?

> *any composition either executes correctly or refuses, and never answers
> partially without declaring it*

**Split it. Two of the three clauses can be established today. The third cannot,
and that is the honest answer.**

**"Executes correctly or refuses" — yes, and the mechanism ships.**
`AnalyticalPlan.required_kinds` + `AnalyticalResult.satisfied` already implement
exactly this: a plan that runs end to end but produces none of the finding kinds
it declared necessary *"has produced a REFUSAL … never present the half it did
compute as the answer."* `unmet_reasons()` returns the refusal in the failing
capability's own words. This is a property, tested as a property, and it
generalises to primitives without change.

**"Never answers partially" — yes for declared elements.** `RequestedFacet`
defaults to `status = LOST` and `LOST` fails closed. Defaulting to the unsafe
state is the right construction and it survives composition.

**"Without declaring it" — NO, not today, and §2.3 is the proof.** The
disclosure rules are **arity-1 rules**. At arity 2, 40% of T5's groups are below
the thin-sample floor and *nothing is declared*, because the guard is
`len(group_cols) == 1`. The property is not merely unproven at higher arity —
it is **measurably false** at the first arity a compositional layer unlocks.

So: **the property can be established, but not on the current disclosure rules.**
Generalising every arity-1 disclosure to arity *n* is a prerequisite, it is not
optional, and it does not appear in any estimate so far. **An untestable planner
is worse than thirteen testable routes** — and on today's rules, the planner
would be untestable in exactly the one clause that matters.

---

## 8. The honest estimate

### Bounded — and smaller than expected

* **The arithmetic.** T3/T4/T5/T6/T7 compose from unmodified existing code and reconcile to the penny on both books. There is no calculation to write.
* **The primitive count.** Seven, derived. Not fifteen.
* **The plan contract.** `AnalyticalCapability` / `AnalyticalPlan` / `Finding` / `validate()` / `plan_from_proposal()` already exist, are tested, and are the right shape.
* **The refusal property.** `required_kinds` + `satisfied` + `unmet_reasons` already implement two of the three clauses of §5's property.
* **The first three conversions.** `portfolio_summary`, `temporal_compare`, `funded_bridge` — 20 test files, 237 handler lines, 8 receipt literals between them, and the numbers are in §6.

### Not bounded

* **Consolidating 23 implementations into 7.** Each retirement is a byte-identical proof against every consumer of the implementation being retired. This is the dominant cost and it is invisible from the outside, because the primitives are not missing — they are duplicated.
* **Moving 54 route-name literals and 7 allowlists off route identity.** The facet layer's honour test would have to become per-step. Neither the size nor the risk of that is knowable until one facet kind has actually been moved.
* **Generalising every arity-1 disclosure to arity *n*.** §2.3 shows the first one is already wrong. How many more there are is not known; the census did not look for them, and looking for them is itself a piece of work.
* **The parser.** It is 13 branches in a 4,504-line module, and it is upstream of everything the plan layer would do. Converting routes while the parser still decides shape delivers a plan layer that receives a spec whose shape was already chosen. **No estimate of the route conversions is meaningful without a separate estimate of this**, and the study did not produce one.

### What would have to be discovered before an estimate means anything

1. **One facet kind, actually moved from route identity to per-step declaration.** Until one is done, the 54-literal number is a count, not a cost.
2. **One primitive, actually consolidated** — `group` is the obvious candidate, because its four implementations differ in *arity* (one n-ary, three unary) and the disagreement is therefore visible. The byte-identical proof for that one consolidation is the unit that every other estimate should be quoted in.
3. **A full sweep for arity-1 disclosure rules.** §2.3 found one because T5 was measured. The others will be found the same way or not at all.
4. **Whether the parser is in scope.** This is a decision, not a discovery, and it changes the size of the work by more than any other single factor in this document.

### Sequencing

Nothing here changes the brief's own reading, and the measurement supports it:
**this is the right architecture and the wrong moment.** The study's specific
contribution to that judgement is item 4 above — the work is larger than the
route count suggests, because the route count is one of three cascades — and
§2.1, which says the most-wanted capability (T3) is a route-shape problem that
is **already computed**, not a missing primitive.

**P1 does not need this.** P1 is the per-period-breakdown family under the
current architecture, and §2.1 shows its data is already produced on every
`evolution` answer. Whether P1 should be built under the current model is not
this study's call, but the study removes one argument for waiting: **P1 is not
blocked on a compositional layer.**

The migration is not authorised, should not start before the client is live,
and — on the evidence in §4 — should not start at all until the interpretation
contract can supply a joined filter clause (§5).

---

## 9. What this study did not do

* Built nothing. Chose nothing. Proposed no design. The two instruments import nothing into the serving path and change no product module.
* Did not treat the compositional model as decided. §1.3 reports four routes that do not factor, and §2.3 reports a property that is measurably false today, both of which count against.
* Did not fold P1 in. §8 states only that P1 is not *blocked* by this; its scope is untouched.
* **Did not measure the LLM arm.** Every run in this study is deterministic. Whether an LLM-proposed plan behaves is unknown, and §4 records that the proposer is not in service.
* **Did not estimate the parser conversion**, which §8 names as the largest single unknown.
