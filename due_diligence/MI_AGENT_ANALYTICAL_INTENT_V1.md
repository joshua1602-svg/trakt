# MI Agent — Analytical Intent Family Hardening V1

**Scope.** Improve analytical-intent recognition without changing the underlying
deterministic analytics. No new analytical mathematics, no replacement of the
analytical capability layer, no rewritten specialist route, no weakened safety
guard, and no test phrasing added as an exact string patch.

**Baseline.** Branch `claude/mi-analytical-capability-layer-vlkjfw`, SHA
`104c89d` — the frozen 752-run measurement in
`due_diligence/MI_AGENT_ANALYTICAL_NL_ROBUSTNESS.md`.

**Change.** Six commits, one new module and one new test file:

| | |
|---|---|
| `a6331aa` | the boundary: six families, the lending ruling, the fail-closed rule |
| `cf7e673` | make a refused envelope say one thing — guard and receipt agree with the answer |
| `235ba7b` | settle `forecast_mode` only for a PIPELINE run rate (§9) |
| `bd97c2b` | leave a single-measure series with the `evolution` route (§11.5) |
| `9b925eb` | correct the lending-role docstring to what the code does |
| `0bcf9de` | prove every governed operation is reachable |

---

## 1. Executive verdict

**The recognition surface now matches the architecture behind it.**

The same 752-run bank, unaltered — same 44 phrasings, same two books, same
repetitions, same frozen scorer, same declared expected intents:

| | Baseline `104c89d` | V1 |
|---|---|---|
| CORRECT + CORRECT_WITH_DISCLOSED_LIMITATION | 425 (56.5%) | **675 (89.8%)** |
| SAFE_REFUSAL | 140 (18.6%) | 77 (10.2%) |
| **INCORRECT_SUCCESSFUL** | **40** | **0** |
| **SILENT_SEMANTIC_ERROR** | **147** | **0** |
| **HARD_FAILURE** | 0 | **0** |
| **Unsafe total** | **187 (24.9%)** | **0 (0.0%)** |
| genuine LLM parses | 648 (86%) | 641 (85%) |

**Every one of the 187 unsafe outcomes is gone, and not one was traded for
another.** 16 of 44 variations changed verdict; every change moved toward
safety or toward a correct answer. Zero regressions into an unsafe class. Zero
capabilities lost.

Four intents that were systemically broken now work under every phrasing tested:

| | Baseline | V1 |
|---|---|---|
| Q1 new-origination profile | 0 of 80 acceptable | **74 correct, 6 safe refusals, 0 unsafe** |
| Q4 completion run rate | 24 of 48 unsafe | **28 correct, 20 safe refusals, 0 unsafe** |
| Q6 limits closest to breach | 23 of 48 unsafe | **47 correct, 1 refusal, 0 unsafe** |
| Q7 vintages vs front book | 20 of 80 unsafe | **80 correct, 0 refusals, 0 unsafe** |
| Q8 relative movement | 40 of 240 unsafe | **240 correct, 0 unsafe** |

Q8 was the brief's control, and it is the cleanest single result: the identical
sentence *"Are X and Y balances developing differently over time?"* resolves the
same way across all three governed population resolvers — provenance, seasoning
and dimension value — where before it produced three different outcomes.

**Every number the layer produced was right.** 6,856 numeric findings reconciled
independently against the fixture CSVs with pandas — **zero mismatches**. That
includes 1,020 checks on the two new lending windows, whose truth was recomputed
from `origination_date` against each snapshot date rather than taken from the
code under test.

**It generalises.** 44 of 44 variations reach the same verdict on both books.
166 of 176 variation × book × arm groups are identical across every repeat.

**And it did not disturb what worked.** 0 of 30 simple-MI answers changed; 8 of
9 canonical CFO answers are byte-identical on both books; a 79-question sweep of
ordinary MI phrasings run against both checkouts on the same data is 76/79
identical, with all three differences improvements.

## 2. The six governed analytical families

Every analytical question the platform will recognise belongs to one or more of
exactly six families, declared once in `mi_workflows/analytical/intent.py`:

| Family | What it is about |
|---|---|
| `MIX_PROFILE` | what the book is made of, and how that make-up moved |
| `PIPELINE` | the pre-funding dataset, and the flow out of it into the book |
| `LIMITS_CONCENTRATION` | governed controls, and the distance to them |
| `FORECAST_PROJECTION` | a value, a date or a condition that has not happened yet |
| `MOVEMENT_TREND` | how a measure moved across governed reporting snapshots |
| `VINTAGE_COHORT` | origination cohorts, and how they differ from one another |

`tests/test_analytical_intent_boundary.py::test_exactly_the_six_governed_families`
asserts the set is exactly these six. There is no seventh, and no "other".

## 3. The governed operations, and what they map onto

Each family declares its operations and the routes and capabilities that
**already own them**. Nothing here is new mathematics; every entry is a pointer.

| Family | Governed operations | Existing owners |
|---|---|---|
| `MIX_PROFILE` | SNAPSHOT · COMPOSITION · COMPARISON · CHANGE · DIVERGENCE · ATTRIBUTION | `analytical_composition`, `period_change_analysis`, `portfolio_summary`, `period_movement` — capabilities `population_profile`, `portfolio_snapshot`, `period_movement` |
| `PIPELINE` | STOCK · MOVEMENT · CONVERSION · RUN_RATE · EXPECTED_COMPLETION · TIMING · MIX | `analytical_composition`, `forecast_extrapolation`, `cohort_conversion`, `scenario` — capabilities `pipeline_stock`, `pipeline_completion_forecast`, `completion_run_rate` |
| `LIMITS_CONCENTRATION` | CONCENTRATION · STATUS · HEADROOM · RANKING · MOVEMENT · FORECAST_BREACH | `risk_limits`, `analytical_composition` — capabilities `concentration_limits`, `threshold_projection` |
| `FORECAST_PROJECTION` | PROJECT_VALUE · MILESTONE · HORIZON · SCENARIO | `forecast_extrapolation`, `scenario`, `analytical_composition` — capabilities `funded_balance_forecast`, `threshold_projection`, `completion_run_rate`, `pipeline_completion_forecast` |
| `MOVEMENT_TREND` | DELTA · TREND · RANKING · ACCELERATION · ATTRIBUTION | `analytical_composition`, `period_change_analysis`, `period_movement`, `evolution`, `temporal_compare`, `funded_bridge` — capabilities `period_movement`, `portfolio_snapshot` |
| `VINTAGE_COHORT` | SNAPSHOT · COMPARISON · EVOLUTION · RANKING · DIVERGENCE | `analytical_composition`, `cohort_progression` — capabilities `vintage_analysis`, `portfolio_snapshot`, `population_profile` |

SCENARIO is listed for `FORECAST_PROJECTION` because the `scenario` route and
`scenario_mod.multiplier_from_conversion_delta` already exist. It is declared
where the platform supports it and nowhere else.

Two tests enforce the mapping rather than trusting it:

* `test_every_declared_capability_already_exists` resolves every capability id
  against `mi_workflows.analytical.registry.CAPABILITIES`;
* `test_every_declared_route_already_exists` resolves every route name against
  the live `chat_routing.REGISTRY`.

If a future family names something that does not exist, the suite fails. That is
the mechanical guarantee behind "map them onto existing capabilities only".

## 4. Shared arguments — no duplicate registries

The boundary creates no argument vocabulary of its own. Every governed concept it
reads is resolved by the module that already owns it:

| Argument | Owner consulted |
|---|---|
| population — seasoning / lending window | `mi_agent.seasoning` |
| population — provenance (direct / acquired) | `mi_agent.portfolio_lens`, via the governed portfolio registry |
| population — dimension value | the governed frame, matched literally |
| measure, dimension, format, role | `mi_agent/mi_semantics_field_registry.yaml` |
| aggregation, weighting, share basis, directionality | `config/business_semantics_registry.yaml` |
| period / snapshot resolution | the retained governed reporting snapshots |
| portfolio scope | the portfolio lens and `trakt_core.tenancy` |
| thresholds and filters | `mi_agent.population.material_predicates` |

**Two vocabularies were deleted, not added.** The capability layer's planner
carried its own `_PROFILE_TERMS`, `_CHANGE_TERMS`, `_COMPARATIVE_TERMS`,
`_RISK_PROFILE_TERMS`, `_VINTAGE_TERMS` and `_PIPELINE_TERMS`. They are gone;
the plan builders now ask the boundary. That is not tidying — two comparison
vocabularies is precisely how *"are X and Y developing differently?"* came to
resolve one way and *"how has X moved relative to Y?"* another, for the same
question and the same two populations.

What remains in the planner is what shapes a **plan** rather than identifying an
**intent**: which pipeline stage to read, whether to use the whole retained
window or the last pair of snapshots, and what to decline outright.

## 5. The governed seasoning / lending ruling

Applied in `mi_agent/seasoning.py`, on the axis that already owns seasoning, and
driven by the same `seasoning:` block in `config/mi/buckets.yaml`.

| Window | Definition | Predicate |
|---|---|---|
| NEW | originated in the last **1** month | `months_on_book <= 1` |
| RECENT | originated in the last **3** months | `months_on_book <= 3` |
| FRONT BOOK | originated in the last **12** months | `seasoning_segment = Front Book` |
| BACK BOOK | older than **12** months | `seasoning_segment = Back Book` |

**They are nested, not a partition.** Every NEW loan is also RECENT and also
FRONT BOOK. That is the point of the ruling: a CFO asking about "new lending" and
a CFO asking about "the front book" are asking two different questions, and
collapsing them into one segment is the defect the ruling exists to fix.

**Front and back keep the predicate they already had.** Anything that resolved to
them before resolves to identical rows now
(`test_front_and_back_keep_the_predicate_they_already_had`).

**Configuration, never code.** `test_the_thresholds_come_from_configuration`
moves the boundary to 18 months and NEW to 2 months from a temporary config file
and asserts the windows follow.

### "Lending" is not globally mapped

`_SEGMENT_PHRASES` — the vocabulary that selects a population *everywhere in the
stack* — was **not touched**. `segments_named("our new lending")` still returns
`[]`, and a test asserts it. The lending vocabulary is a separate function,
`lending_windows_named`, and naming a window is not the same as executing one.

The role is decided by **analytical context**:

| Context | Role | Example |
|---|---|---|
| PROFILE / MIX / RISK / CHARACTERISTICS | population of loans | *"the risk profile of our new lending"* |
| RUN RATE / AMOUNT / VOLUME / FLOW / ORIGINATED IN PERIOD | origination flow | *"our new lending run rate"* |
| neither | **unresolved** | no population is created; the fail-closed rule applies |

`test_the_role_is_not_settled_by_matching_the_sentence` asserts that the same
words (`lending_windows` identical) carry different roles in the two contexts.
That is the property the ruling asks for, and it cannot be obtained from an
exact-sentence match.

One further consequence, deliberate: windows are returned **in the order the
question names them**, so the first is the subject and the second the comparand.
A pair assembled in the other order reverses the sign of every delta reported.

## 6. Canonical mapping of the nine intents

| # | Question | Family | Operation(s) | Owner |
|---|---|---|---|---|
| Q1 | profile of new originations, changed lately | MIX_PROFILE + MOVEMENT_TREND + VINTAGE_COHORT | CHANGE, ATTRIBUTION, COMPOSITION | `analytical_composition` → `period_movement` + `population_profile` |
| Q2 | when do we reach £100m | FORECAST_PROJECTION | MILESTONE, HORIZON | **`forecast_extrapolation`** (unchanged) |
| Q3 | offers at stage, how much completes, when | PIPELINE + FORECAST_PROJECTION | STOCK, EXPECTED_COMPLETION, TIMING | `analytical_composition` → `pipeline_stock` + `pipeline_completion_forecast` |
| Q4 | current completion run rate | PIPELINE | MOVEMENT, RUN_RATE | **`forecast_extrapolation`** (unchanged) |
| Q5 | forecast limit breach in 3 months | LIMITS_CONCENTRATION + FORECAST_PROJECTION | FORECAST_BREACH, STATUS | **`risk_limits`** (unchanged) |
| Q6 | limits closest to breaching | LIMITS_CONCENTRATION | HEADROOM, RANKING, STATUS | **`risk_limits`** (unchanged) |
| Q7 | older vintages vs front book, risk | MIX_PROFILE + VINTAGE_COHORT | COMPARISON, DIVERGENCE | `analytical_composition` → `portfolio_snapshot` ×2 + `vintage_analysis` |
| Q8 | balance from X vs Y, over time | MOVEMENT_TREND (+ VINTAGE_COHORT when the pair is seasoning) | DELTA, RANKING, ATTRIBUTION | `analytical_composition` → `period_movement` ×2 |
| Q9 | forecast funded balance from pipeline | PIPELINE + FORECAST_PROJECTION | PROJECT_VALUE, EXPECTED_COMPLETION | `analytical_composition` → `funded_balance_forecast` + `pipeline_completion_forecast` |

**Q2, Q4, Q5 and Q6 remain owned by their existing routes.** The family layer
does not force them through `analytical_composition`; it makes sure they *reach*
their owner. See §7.

Ownership is also actively defended in the other direction. `_executor_already_compares`
makes the analytical layer stand down when the governed parse has already
resolved a comparison — a measure grouped by the dimension that partitions the
two populations. That is the test of "unnecessarily", and it is structural
rather than a list of exceptions: it applies only when the comparison is the
whole question, never when two periods, a forecast, the pipeline extract or a
limit schedule are also needed.

## 7. The analytical intent boundary

`mi_workflows/analytical/intent.py`, 828 lines, importing no dataframe library.
It sits at two seams and calculates nothing.

**Seam 1 — `chat_routing.try_route`, before any recogniser is consulted.**
`intent.settle(question, spec)` classifies the question and settles governed
intent flags the parser left open. It may set exactly two, both of which an
existing route already recognises:

| Flag | Set when | Route it reaches |
|---|---|---|
| `risk_limit_query` | `LIMITS_CONCENTRATION` recognised and the parse left it unset | `risk_limits` |
| `forecast_mode = "extrapolation"` | `RUN_RATE` operation recognised, no count requested, parse left it unset | `forecast_extrapolation` |

Three rules keep this safe, each with a test:

* it **never overrides** a flag the parser already settled;
* it **never names a capability that does not exist** (§3);
* it **never widens a measure** — the governed run-rate capability produces a
  currency rate and nothing else, so *"how many loans are we completing?"* is not
  handed to it. That question falls to §8 instead and is refused rather than
  answered in the wrong unit.

**Seam 2 — the planner.** The plan builders ask the boundary which family and
operations were recognised instead of carrying private vocabularies (§4).

**Concept signals, not question templates.** The five governed signals are
comparison, change/trend, run-rate, limits and forecast — exactly the five the
brief names. The vocabularies are the words a book, a control or a movement is
described with. Three candidate entries were **rejected** during construction
and the rejections are recorded in the source: `" monthly "`, `" a month "` and
`" level of "`, because *"what is the monthly payment?"* and *"what level of
arrears do we have?"* are point-in-time questions and a rate signal there would
refuse an answer the product gives correctly today. `" since "` was removed from
the change vocabulary for the same reason.

`"concentration"` alone does **not** name the limits family. A concentration is a
measure; it becomes a control only when a limit, headroom, breach, covenant,
tolerance or threshold is named with it. Without that rule *"what is our
concentration by region?"* would have been refused.

## 8. The fail-closed safety rule

**The rule.** A materially analytical question that no governed family, route or
plan can confidently own must never fall through to the generic point-in-time
executor as a confident answer.

**Where it sits.** `mi_service._fail_closed_analytical`, on the point-in-time
path only, **after** the executor has run. After, not before, for the same reason
the P0 execution receipt runs after: what matters is not what an answer was meant
to be but what it demonstrably *carries*.

**How it decides.** The boundary derives structural requirements from the
recognised operations; the check asks whether the executed answer satisfies them.

| Requirement | Satisfied by | Can the funded point-in-time executor satisfy it? |
|---|---|---|
| `pipeline_dataset` | an answer computed from the governed pipeline extract | **No** — it reads the loan tape |
| `limit_evidence` | an answer evaluating the governed limit schedule | **No** |
| `forecast` | a forward-looking governed figure | **No** |
| `period_comparison` | two governed reporting snapshots | **No** — it reads one |
| `population_comparison` | two populations measured separately, **or** grouped on the governed dimension that partitions them | **Yes**, when the parse produced that grouping |

The last row is the one that keeps a capability. A front/back comparison the
executor really did make — grouped on `seasoning_segment`, both sides present,
nothing narrowed and nothing passed off as the other — **is** the comparison the
reader asked for, reached by another mechanism. It is not refused
(`test_a_comparison_the_executor_really_did_make_is_not_refused`).

**The four named failures.** Each returned `ok=True` with a green guard in the
frozen baseline. `test_the_four_measured_failures_are_refused` asserts each is
now refused:

| Question | Baseline answer |
|---|---|
| *"How many loans are we completing at the moment?"* | 11,035 loans — the whole book |
| *"What completion rate are we running at?"* | £1.96bn — the whole balance |
| *"Where are we closest to our limits?"* | weighted-average LTV by region |
| *"Which of our limits are most at risk?"* | balance by account status |

**The refusal is a refusal, not a hedge.** It names the family it understood and
what could not be established, states that no current-position figure was
substituted, and offers the governed analytic that would answer. It carries no
number. The semantic guard records `refuse` with one unavailable facet per unmet
requirement, and the execution summary is cleared — a green guard beside a
refusal reads as a spurious refusal, and an execution summary still saying
"Calculated: Count of · 11,035 loans" would leave the discarded figure on the
envelope for any channel that renders the receipt.

**It cannot over-refuse a non-analytical question.** `materially_analytical` is
false unless the question carries one of the five governed signals or belongs to
a family whose dataset the executor does not read. A seven-case test asserts that
*"balance by region"*, *"what is the total balance?"*, *"weighted average LTV by
region"*, *"show me balance by LTV band"*, *"what is the largest single loan
exposure?"*, *"how many loans are on the book?"* and *"what is the balance of the
front book?"* are all outside it. Measured directly: a 30-question sweep of
ordinary MI phrasings produced **zero** fail-closed refusals.

## 9. Route contention and ownership

The baseline had two contention clusters. Both are resolved, and **neither
required a precedence change**.

**Cluster 1 — the generic executor answers anything (107 baseline runs).** Split
in two by cause. Where a governed owner exists and the parse merely missed its
flag (Q4, Q6), §7 settles the flag and the unchanged route claims the question.
Where no owner can answer (Q4.2's count run-rate), §8 refuses.

**Cluster 2 — a neighbouring route claims it and answers narrowly (80 baseline
runs).** `period_change_analysis` took Q1; `cohort_progression` and `evolution`
took Q8.3. Both dissolve because the analytical planner now *recognises* those
questions, and `analytical_composition` already sits at priority 5 / confidence
0.8 — ahead of all of them. The arbitration the recogniser registry was built for
does the work. No priority, confidence or registration order changed.

**Q8 is the control, and it now resolves consistently.** The identical sentence
shape *"Are X and Y balances developing differently over time?"* against three
different population resolvers:

| Pair | Baseline | V1 |
|---|---|---|
| direct vs acquired (provenance) | `cohort_progression`, whole book | `analytical_composition`, both populations |
| front vs back book (seasoning) | `evolution`, whole book | `analytical_composition`, both populations |
| region vs region (dimension value) | `evolution`, refused on lost scope | `analytical_composition`, both populations |

Same words, three resolvers, one outcome. The cause was a single missing
comparison concept — *"differently"* — in a vocabulary that was a private copy of
one that had it.

### 9.1 Contention this change CREATED, and how it was found

Widening a recognition surface can move a question off a route that was already
answering it. Two instances occurred and both were caught and fixed before the
final measurement; they are recorded because a report that only lists the wins
is not a measurement.

**Q2.4 — a milestone answered as a balance.** *"Based on the current book and
pipeline, how long until we reach £100m?"* moved from `forecast_extrapolation`
to the analytical layer, which answered a **date** question with a **projected
balance**. The cause was mine: the funded-balance-outlook gate had been matching
"forecast / project / expect" literally, and once it consulted the whole forecast
signal it also saw "how long until". Both are forecast questions; only one is a
projected value. Fixed by having the gate ask the boundary which operation it
recognised — MILESTONE and HORIZON belong to the forecast route, PROJECT_VALUE
is composed here.

**This was found by the rerun itself**, in the first of four arms: 3 unsafe runs
out of 188, all the same variation. Had the bank not been rerun in full, it would
have shipped.

**The metric-evolution series** — see §11.5.

Both fixes are deference, not precedence: no priority, confidence or
registration order was changed for any route.

## 10. No change to the analytical engines

Not one line of `mi_workflows/engine.py`, the deterministic executors, the
forecast model, the concentration-limit evaluator or the vintage analysis was
changed. The full production diff:

| File | Change |
|---|---|
| `mi_workflows/analytical/intent.py` | **new** — the boundary |
| `mi_agent/seasoning.py` | lending windows + vocabulary; `_SEGMENT_PHRASES` untouched |
| `config/mi/buckets.yaml` | the two new window thresholds |
| `mi_agent/population.py` | the fabrication guard consults the window vocabulary |
| `mi_workflows/analytical/populations.py` | `lending_window_population` |
| `mi_workflows/analytical/planner.py` | gates read the boundary; six private vocabularies deleted |
| `mi_agent_api/chat_routing.py` | boundary called before routing; evidence stamped |
| `mi_agent_api/mi_service.py` | the fail-closed check |
| `docs/mi_query_agent_architecture.md` | §13 |
| `tests/…` | the new suite; one orientation assertion updated (§11) |

`mi_workflows/analytical/intent.py` imports no dataframe library. The existing
test that parses every module in the analytical package and fails on a `pandas`
or `numpy` import covers it.

## 11. Regression — the WIN-WIN rule

The rule is that improving analytical recognition must not break a capability
that already worked. Four independent checks, all run on the final tree.

### 11.1 The named baseline suites

| Suite | Result |
|---|---|
| `tests/test_analytical_capability_layer.py` | 90 passed |
| `tests/test_fabricated_population.py` | passed |
| `tests/test_p1i_scope_resolution.py` (governed scope) | passed |
| `tests/test_p1j1_vintage_seasoning.py` (vintage / seasoning) | 53 passed |
| `tests/test_p1l_population_propagation.py` | passed |
| `tests/test_p1m_statistic_identity.py` | passed |
| `tests/test_p1n_statistic_breadth.py` | passed |
| `tests/test_p1e_golden_bank.py` | passed |
| `mi_agent/tests/test_mi_calibration_bank.py` (252-question bank) | 245 passed, 13 xfailed |
| `tests/test_analytical_intent_boundary.py` (**new**) | 110 passed |
| `mi_agent_api/tests/test_forecast_extrapolation.py` | passed |
| **Combined focused run** | **796 passed, 13 xfailed** |

### 11.2 The 30-question simple-MI bank

**0 of 30 answers changed** — route, `ok` and answer text byte-identical to the
frozen baseline, including the two that legitimately refuse.

### 11.3 The nine canonical CFO questions, both books

**8 of 9 byte-identical on each book**, and all nine `ok=True`. The ninth is Q7,
and it changed on purpose.

> *"How does the risk profile of older vintages compare with the front book?"*
>
> | | subject | comparand |
> |---|---|---|
> | baseline | Front Book | Back Book |
> | V1 | **Back Book** | **Front Book** |
>
> Same two governed populations, same row counts, same figures — Alderbridge
> 43.97% vs 34.71% weighted-average LTV, Kestrelmoor 39.16% vs 29.14%. Only the
> **orientation** moved, and it moved to follow the question: *older vintages* is
> the subject of that sentence and the front book is what it is compared with.
>
> Before this work, *"older vintages"* was not recognised as a governed
> population at all, so the pair was assembled from *"front book"* alone and the
> binary partition supplied the other side — which put the two sides in the
> opposite order to the sentence and reported every delta with the opposite sign.
> `test_q7_compares_the_two_governed_sides_and_reconciles` now asserts the
> question's own orientation.
>
> This is reported as a **change, not a silent improvement**, because §10
> requires trade-offs to be surfaced. It is not a trade: nothing was lost.

### 11.4 A 79-question sweep, baseline against V1, same data

Run through the production `POST /mi/query` entrypoint on the *same* checkout,
switching only the code under test, and deliberately weighted toward the
territory this change could disturb — seasoning populations, evolution series,
concentration language, pipeline stages, provenance comparisons, plain
point-in-time MI.

**76 of 79 identical. 3 changed, all in the product's favour, none a loss.**

| Question | Baseline | V1 |
|---|---|---|
| *"How has the front book changed?"* | `period_change_analysis` **refused** — the front-book population could not be applied | **answered**, population honoured, 1,177 loans, £175.4m → £171.7m |
| *"How has the back book changed over the last few months?"* | `period_change_analysis` **refused** — same cause | **answered**, 9,858 loans, £1.75bn → £1.79bn |
| *"How many cases are at offer?"* | generic executor: **"11,035 loans"** | **controlled refusal** — a pipeline question, and the funded tape is not the pipeline |

The third is the fail-closed rule working on a question **that is not in the
44-variation bank at all** — independent evidence that it generalises rather
than fitting the test set.

### 11.5 One capability that was nearly traded, and was not

An intermediate build claimed *"Show the balance evolution for the front book"*
for the analytical layer and answered it with a two-snapshot movement. The
`evolution` route had been answering it as a **three-period series**, scoped to
the population — and a movement between two snapshots is *less* than a series.

That is exactly the trade §10 forbids, so it was not accepted. The planner now
asks the evolution route's **own recogniser** whether it owns the question, and
stands down when it does. The deference lifts when a composition is also asked
for, because a single-measure series does not carry what the book is made of.
Four tests hold the boundary in place.

## 12. The 752-run rerun, against the frozen baseline

**The bank was rerun unaltered**: the same 44 phrasings, the same two books, the
same repetition counts (5 for Q1/Q3/Q5/Q7/Q8, 3 elsewhere), the same two arms
(production and forced-LLM), and the same scorer — `nl_score.py` was not
modified, and re-running it over the frozen baseline files reproduces the
published baseline distribution exactly, which is what makes the comparison
sound.

### 12.1 Distribution

| Outcome | Baseline | V1 |
|---|---|---|
| CORRECT | 405 (53.9%) | **675 (89.8%)** |
| CORRECT_WITH_DISCLOSED_LIMITATION | 20 (2.7%) | 0 |
| HONEST_PARTIAL | 0 | 0 |
| SAFE_REFUSAL | 140 (18.6%) | 77 (10.2%) |
| INCORRECT_SUCCESSFUL | 40 (5.3%) | **0** |
| SILENT_SEMANTIC_ERROR | 147 (19.5%) | **0** |
| HARD_FAILURE | 0 | **0** |

The disclosed-limitation class emptying is itself a result: the one variation in
it (Q7.3) was being answered *adjacently* by the generic executor grouping on the
seasoning segment. It is now answered by the capability that owns it.

### 12.2 Every variation that moved

16 of 44 changed verdict. Every one is listed; every one improved.

| Variation | Baseline | V1 | Now answered by |
|---|---|---|---|
| Q1.1 *"profile of our new lending…"* | SILENT | **CORRECT** | `analytical_composition` |
| Q1.2 *"originating different types… vs a few months ago"* | SILENT | **CORRECT** | `analytical_composition` |
| Q1.3 *"recent lending vs earlier in the year"* | SILENT | **CORRECT** | `analytical_composition` |
| Q1.4 *"risk and borrower profile of new business"* | INCORRECT | **CORRECT** | `analytical_composition` |
| Q3.1 *"how much at offer and how much completes"* | SAFE_REFUSAL | **CORRECT** | `analytical_composition` |
| Q3.4 *"how much sitting at offer… and when"* | SAFE_REFUSAL | **CORRECT** | `analytical_composition` |
| Q4.1 *"what completion rate are we running at?"* | SILENT | **CORRECT** | `forecast_extrapolation` |
| Q4.2 *"how many loans are we completing?"* | SILENT | **SAFE_REFUSAL** | fail-closed |
| Q6.2 *"where are we closest to our limits?"* | SILENT | **CORRECT** | `risk_limits` |
| Q6.3 *"which of our limits are most at risk?"* | SILENT | **CORRECT** | `risk_limits` |
| Q7.1 *"front book vs older lending, risk"* | SAFE_REFUSAL | **CORRECT** | `analytical_composition` |
| Q7.2 *"are older loans riskier…?"* | SILENT | **CORRECT** | `analytical_composition` |
| Q7.3 *"recent originations versus the back book"* | DISCLOSED | **CORRECT** | `analytical_composition` |
| Q8.3 / provenance | INCORRECT | **CORRECT** | `analytical_composition` |
| Q8.3 / seasoning | SILENT | **CORRECT** | `analytical_composition` |
| Q8.3 / dimension value | SAFE_REFUSAL | **CORRECT** | `analytical_composition` |

**Regressions into an unsafe class: NONE. Capabilities lost (ok → not ok):
NONE.** Both are asserted mechanically by the comparison script, not read off
by eye.

### 12.3 Generalisation

| | |
|---|---|
| Variations reaching the same verdict on **both books** | **44 / 44 (100%)** |
| Variation × book × arm groups identical across every repeat | **166 / 176 (94%)** |
| Genuine LLM parses (`parser_used == "llm"`) | 641 / 752 (85%) |

The 10 groups that varied all move between CORRECT and SAFE_REFUSAL — never
between two different answers, and never into an unsafe class. The cause in every
case is the LLM parse emitting a population predicate the governed route did not
apply (for example `origination_date >= 2024-01-01` where the plan resolved the
RECENT window), which the pre-existing P1L population ledger refuses. That is the
guard doing its job, and it behaved identically in the baseline.

### 12.4 Numerical reconciliation

| Arm | Findings reconciled | Mismatches |
|---|---|---|
| Alderbridge production | 1,704 / 1,704 | 0 |
| Alderbridge forced-LLM | 1,704 / 1,704 | 0 |
| Kestrelmoor production | 1,724 / 1,724 | 0 |
| Kestrelmoor forced-LLM | 1,724 / 1,724 | 0 |
| **Total** | **6,856 / 6,856** | **0** |

2.6× the baseline's 2,686 findings, because far more questions now reach a
capability that produces them. Truth computed independently with pandas from the
fixture CSVs; populations verified by row count as well as by value.

**The two new lending windows are verified, not merely exercised**: 1,020 of
those checks are on `new` and `recent`, whose truth is recomputed from
`origination_date` against each snapshot date rather than read from the code
under test.

### 12.5 The gate

| §12 criterion | Required | Measured |
|---|---|---|
| INCORRECT_SUCCESSFUL | 0 | **0** |
| SILENT_SEMANTIC_ERROR | 0 | **0** |
| HARD_FAILURE | 0 | **0** |
| CORRECT / DISCLOSED | ≥ 80% commercial target | **89.8%** |
| remainder HONEST_PARTIAL or SAFE_REFUSAL | all | **77 of 77** |

The 89.8% was not reached by broadening unsafe inference. It was reached by
routing questions to capabilities that already existed, and by refusing the ones
nothing can answer — the remaining 10.2% are all stated refusals, and §13
accounts for every one of them.

## 13. Truth and safety validation of every newly successful question

§14 of the brief: success is not awarded because the prose sounds plausible. For
each of the 15 variations that moved into a correct outcome, the whole chain was
read — family, operation, population predicate and row count, period, capability
calls, findings, guard verdict and narrative — and the figures reconciled
independently.

| Variation | Plan | Population actually applied | Rows |
|---|---|---|---|
| Q1.1, Q1.4 | `origination_profile_change` | `months_on_book le 1` | 11,035 → **115** |
| Q1.2, Q1.3 | `origination_profile_change` | `months_on_book le 3` | 11,035 → **258** |
| Q3.1, Q3.4 | `pipeline_offer_outlook` | `pipeline_stage = OFFER` (pipeline dataset) | **157 cases** |
| Q4.1 | *(no plan)* `forecast_extrapolation` | whole book, as the run-rate requires | — |
| Q6.2, Q6.3 | *(no plan)* `risk_limits` | the governed limit schedule | — |
| Q7.1, Q7.3 | `vintage_risk_comparison` | `seasoning_segment` Front **and** Back | 1,177 / 9,858 |
| Q7.2 | `vintage_risk_comparison` | `seasoning_segment = Back Book` **and** `months_on_book le 3` | 9,858 / 258 |
| Q8.3 / seasoning | `population_movement_comparison` | Front **and** Back | 1,177 / 9,858 |
| Q8.3 / provenance | `population_movement_comparison` | portfolio lens Direct **and** Acquired | 7,126 / 3,909 |
| Q8.3 / dimension | `population_movement_comparison` | `collateral_geography` South East **and** London | 2,420 / 1,380 |

Three things this table is meant to let a reader check:

* **The population is the one the question named, and it was applied.** Every row
  shows a `rowsBefore → rowsAfter` narrowing recorded by the governed population
  ledger, not a claim. Q1's 115 and 258 match a pandas recomputation of
  `months_on_book ≤ 1` and `≤ 3` against the same snapshot exactly.
* **Both sides of a comparison were measured separately.** Q7.2 is the sharpest:
  *"Are older loans riskier than the loans we've originated recently?"* resolves
  to Back Book **against Recent lending (L3M)** — two different governed
  windows, in the order the sentence names them.
* **The period is real and disclosed.** Every movement finding carries
  `{start, end, method, available}`; Q1 compares 2026-04-30 against 2026-06-30
  and says so in the answer.

**Q4.1 and Q6.2/Q6.3 produce no analytical plan at all, and that is the correct
outcome.** They are answered by `forecast_extrapolation` and `risk_limits` — the
routes that own those families — reached because the boundary settled a governed
flag the parse had left unset. No new analytic was introduced for either.

### 13.1 Every remaining refusal, accounted for

77 refusals across 9 variations. None is silent; every one states a reason.

| Variation | Runs | Cause |
|---|---|---|
| Q5.4 | 20 | forward projection requested; `risk_limits` is point-in-time. **Unchanged from baseline** |
| Q2.3 | 12 | forward projection requested; no route claimed it. **Unchanged from baseline** |
| Q9.1 | 12 | parse asked for a period comparison; the plan measures one point in time. **Unchanged from baseline** |
| **Q4.2** | **12** | **the new fail-closed rule** — a pipeline question, and there is no governed COUNT run rate |
| Q4.4 | 8 | LLM parse emitted `origination_date >= LAST_4_WEEKS`; the run-rate route does not narrow. Pre-existing P1L ledger |
| Q1.3 | 6 | LLM parse emitted `origination_date >= 2024-01-01`; the plan resolved the RECENT window. Pre-existing P1L ledger |
| Q3.4 | 4 | LLM parse emitted `account_status = offer` against the funded frame; the plan reads the pipeline extract. Pre-existing P1L ledger |
| Q3.3 | 2 | as Q3.4, with `pipeline_stage = Offer` |
| Q6.3 | 1 | LLM parse emitted `arrears_balance > 0`; the limit schedule does not narrow by it. Pre-existing P1L ledger |

**Only 12 of the 77 come from the new rule.** The other 65 are the existing P1L
population ledger and P0 projection facet refusing exactly as they did before —
in several cases refusing a population the *model* invented, which is what that
guard exists for. Nothing was weakened to raise the correct-answer rate.

## 14. Scope, residual gaps and verdict

### 14.1 What was deliberately not done

Every item the brief placed out of scope stayed out:

| Out of scope | Status |
|---|---|
| new deterministic analytics | none added |
| new statistical capabilities | none added |
| new forecast methodologies | none added |
| the bubble-disclosure fix | not touched |
| the threshold-attachment fix | not touched |
| telemetry | not touched |
| a clarification UX redesign | not touched — the refusal reuses the existing controlled-refusal envelope |
| broker / product data | not touched |
| new asset classes | not touched |
| generic LLM fine-tuning | none |
| prompt self-learning | none |

### 14.2 Residual gaps, stated plainly

**A pipeline STOCK question with no forward element is refused, not answered.**
*"How many cases are at offer?"* names a governed stage and the `pipeline_stock`
capability exists, but the analytical layer's contract is that it engages only
for **composite** work — two or more capabilities — so a single-capability plan
is not built, and no other route owns a bare stage count. The refusal is honest
and is a strict improvement on the baseline (which answered "11,035 loans"), but
it is a lost answer rather than a gained one. Closing it means either a
single-capability plan shape or a pipeline-stock route, and both are new plan
surface rather than intent recognition — out of scope for V1.

**There is no governed COUNT run rate.** *"How many loans are we completing at
the moment?"* is refused because the completion run-rate capability produces a
currency rate and nothing else, and answering in pounds would be a measure
substitution. Building a count run rate is new deterministic analytics (§16).

**The lending windows are nested, and a narrow one has no governed complement.**
"New lending" (L1M) and "recent lending" (L3M) have no automatic other side, so a
comparative question naming only one of them resolves a pair only when the named
window is one half of the front/back binary. This is deliberate — synthesising
"everything that is not new lending" would be inventing a population.

