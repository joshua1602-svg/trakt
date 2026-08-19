# MI Agent — Analytical Intent Family Hardening V1

**Scope.** Improve analytical-intent recognition without changing the underlying
deterministic analytics. No new analytical mathematics, no replacement of the
analytical capability layer, no rewritten specialist route, no weakened safety
guard, and no test phrasing added as an exact string patch.

**Baseline.** Branch `claude/mi-analytical-capability-layer-vlkjfw`, SHA
`104c89d` — the frozen 752-run measurement in
`due_diligence/MI_AGENT_ANALYTICAL_NL_ROBUSTNESS.md`.

**Change.** 12 commits, one new module, one new test file, and the measurement artefacts:

| | |
|---|---|
| `a6331aa` | Decide, once, what kind of question this is |
| `cf7e673` | Make a refused answer say one thing |
| `235ba7b` | Hand a run-rate question to the run-rate route only when it is one |
| `bd97c2b` | Leave a series question with the route that answers it as a series |
| `9b925eb` | Say what the lending-role resolver actually does |
| `0bcf9de` | Prove the governed operations are reachable, not decorative |
| `0060751` | Record the change set in the report front matter |
| `d116bdb` | Leave a milestone with the route that solves for the date |
| `c108b0e` | Report what the rerun measured |
| `9125e77` | Leave the three neighbouring forward questions with their owners |
| `044d13b` | Record the two routes the full suite caught |
| `5f5d697` | Make the measurement auditable rather than narrated |

---

## 1. Executive verdict

**The recognition surface now matches the architecture behind it.**

The same 752-run bank, unaltered — same 44 phrasings, same two books, same
repetitions, same frozen scorer, same declared expected intents:

| | Baseline `104c89d` | V1 |
|---|---|---|
| CORRECT + CORRECT_WITH_DISCLOSED_LIMITATION | 425 (56.5%) | **672 (89.4%)** |
| SAFE_REFUSAL | 140 (18.6%) | 80 (10.6%) |
| **INCORRECT_SUCCESSFUL** | **40** | **0** |
| **SILENT_SEMANTIC_ERROR** | **147** | **0** |
| **HARD_FAILURE** | 0 | **0** |
| **Unsafe total** | **187 (24.9%)** | **0 (0.0%)** |
| genuine LLM parses | 648 (86%) | 645 (86%) |

**Every one of the 187 unsafe outcomes is gone, and not one was traded for
another.** 16 of 44 variations changed verdict; every change moved toward
safety or toward a correct answer. Zero regressions into an unsafe class. Zero
capabilities lost.

Four intents that were systemically broken now work under every phrasing tested:

| | Baseline | V1 |
|---|---|---|
| Q1 new-origination profile | 0 of 80 acceptable | **67 correct, 13 safe refusals, 0 unsafe** |
| Q4 completion run rate | 24 of 48 unsafe | **30 correct, 18 safe refusals, 0 unsafe** |
| Q6 limits closest to breach | 23 of 48 unsafe | **48 correct, 0 refusals, 0 unsafe** |
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

**It generalises.** 43 of 44 variations reach the same verdict on both books;
the one that differs moves between CORRECT and SAFE_REFUSAL, never into an
unsafe class. 165 of 176 variation × book × arm groups are identical across
every repeat.

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

**Q4 conversion and scenario — two routes taken over.** The full suite caught
two more, in `mi_agent_api/tests/test_chat_routing_e2e.py`:

| Question | Owner | What happened |
|---|---|---|
| *"What conversion do we need to reach £50m funded balance?"* | `forecast_extrapolation` | claimed by the analytical layer and answered with a projected balance — the question is an **inverse solve** for the conversion needed |
| *"If our completed conversion rate increased by 10%, what is the impact on the time to reach £50m?"* | `scenario` | claimed, then failed outright |

Same root cause as Q2.4 and one level deeper: the funded-balance outlook had no
way to read a **named monetary target** or a **conditional**, so it claimed
anything forward-looking that mentioned the pipeline and a balance. It now asks
the owning recognisers and the governed parse — a target value, a milestone or a
what-if all belong elsewhere — and what remains is the single forward question
this plan does answer.

**The metric-evolution series** — see §11.5.

All four fixes are deference, not precedence: no priority, confidence or
registration order was changed for any route. Three of the four were found by a
regression gate rather than by inspection, which is the argument for running all
of them rather than the ones that seemed relevant.

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
| `tests/test_analytical_intent_boundary.py` (**new**) | 113 passed |
| `mi_agent_api/tests/` (all, incl. routing e2e) | passed |
| **Combined focused run** | **1,989 passed, 13 xfailed** |
| **Full suite, shipped tree** | **9,061 passed, 0 failed**, 26 skipped, 21 xfailed (baseline 8,947; the difference is the 114 tests added here) |

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
| CORRECT | 405 (53.9%) | **672 (89.4%)** |
| CORRECT_WITH_DISCLOSED_LIMITATION | 20 (2.7%) | 0 |
| HONEST_PARTIAL | 0 | 0 |
| SAFE_REFUSAL | 140 (18.6%) | 80 (10.6%) |
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
| Q1.3 *"recent lending vs earlier in the year"* | SILENT | **SAFE_REFUSAL** | `analytical_composition`, refused by the population ledger — see §13.1 |
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
| Variations reaching the same verdict on **both books** | **43 / 44 (98%)** |
| Variation × book × arm groups identical across every repeat | **165 / 176 (94%)** |
| Genuine LLM parses (`parser_used == "llm"`) | 645 / 752 (86%) |

The one variation that differs across books is Q4.4, SAFE_REFUSAL on Alderbridge
and CORRECT on Kestrelmoor: the model parse emits an
`origination_date >= LAST_4_WEEKS` predicate on one book and not the other, and
the population ledger refuses when it does. Safe against correct, never unsafe.

The 11 groups that varied all move between CORRECT and SAFE_REFUSAL — never
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
| CORRECT / DISCLOSED | ≥ 80% commercial target | **89.4%** |
| remainder HONEST_PARTIAL or SAFE_REFUSAL | all | **80 of 80** |

The 89.4% was not reached by broadening unsafe inference. It was reached by
routing questions to capabilities that already existed, and by refusing the ones
nothing can answer — the remaining 10.6% are all stated refusals, and §13
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

Attribution is by the refusal's own wording, so it is mechanical rather than a
judgement: the fail-closed rule opens *"I understood this as a … question, but I
have not answered it"* and nothing else does.

| Variation | Runs | Source | Cause |
|---|---|---|---|
| Q5.4 | 20 | pre-existing | forward projection requested; `risk_limits` is point-in-time. **Unchanged from baseline** |
| Q1.3 | 13 | pre-existing | model parse emitted `origination_date >= 2024-01-01`; the plan resolved the RECENT window, so the population ledger refuses |
| Q2.3 | 12 | pre-existing | forward projection requested; no route claimed it. **Unchanged from baseline** |
| **Q4.2** | **12** | **the new rule** | a pipeline question, and there is no governed COUNT run rate |
| Q9.1 | 12 | pre-existing | parse asked for a period comparison; the plan measures one point in time. **Unchanged from baseline** |
| Q4.4 | 6 | pre-existing | model parse emitted `origination_date >= LAST_4_WEEKS`; the run-rate route does not narrow |
| Q3.4 | 3 | pre-existing | model parse emitted `account_status = offer` against the funded frame; the plan reads the pipeline extract |
| Q3.3 | 2 | pre-existing | as Q3.4, with `pipeline_stage = Offer` |

**12 of the 80 come from the new rule. The other 68 are the existing P1L
population ledger and P0 projection facet refusing exactly as they did before**
— in five of the eight cases refusing a population the *model* invented, which
is precisely what that guard exists for. Nothing was weakened to raise the
correct-answer rate.

Q1.3 is the honest cost of this work and is worth stating plainly: it was a
SILENT SEMANTIC ERROR in the baseline and is now a stated refusal in the
majority of runs, because the model frequently parses *"earlier in the year"*
into a date predicate the governed plan does not execute. A refusal is a large
improvement on a confident wrong answer, and it is not a correct answer.


### 13.2 Ten runs traced end to end

Ten measured runs, chosen to cover every distinct mechanism rather than
ten easy wins: five per book, two under the forced-model arm, one refusal,
two answered with no analytical plan at all, and the three-resolver control
in full. Each is the FIRST recorded run of that variation — not a
hand-picked repeat.

Provenance is stated per section, because that is what makes the trace
auditable rather than decorative:

* **RECORDED** — read verbatim out of the measured run file; not recomputed.
* **DERIVED** — the boundary's own classifier re-run on the same question.
  `intent.classify` reads nothing but the question and the parse, so this is
  what the boundary saw.
* **TRUTH** — computed in the trace from the fixture CSVs with pandas,
  referencing nothing the agent produced.

### Q1.1 — alderbridge / production
*Chosen because: the governed lending ruling: NEW = L1M, on the question that failed 100% of the time in the baseline.*

**1. Question** (RECORDED)

> How has the profile of our new lending changed over the last few months?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `llm` |
| mode detail | `llm_repaired` |
| LLM failure | `None` |
| repeats in this arm | 5, all reaching `analytical_composition` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | MIX_PROFILE, MOVEMENT_TREND, VINTAGE_COHORT |
| operations | SNAPSHOT, CHANGE, ATTRIBUTION, DELTA, EVOLUTION |
| concept signals matched | change_trend |
| lending windows named | new |
| lending role | population |
| structural requirements | period_comparison |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | months_on_book le 1 |
| predicates unavailable | none |
| rows before → after | 11035 → 115 |
| periods compared | 2026-04-30, 2026-06-30 |
| narrowed to | — |

**5. Deterministic capabilities invoked** (RECORDED)

| | |
|---|---|
| route | `analytical_composition` |
| plan intent | `origination_profile_change` |
| capabilities | `period_movement`, `population_profile` |
| required finding kinds | movement |
| composition version | 1.0.0 · plan origin deterministic |

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

| # | kind | measure | population (rows) | delivered | **independent truth** | agrees |
|---|---|---|---|---|---|---|
| 1 | movement | `current_outstanding_balance` | New lending (last 1 month) (115) | £18.3m | **£18.3m** | ✅ |
| 1p | prior | `current_outstanding_balance` @ 2026-04-30 | New lending (last 1 month) (143) | £21.4m | **£21.4m** | ✅ |
| 2 | movement | `current_loan_to_value` | New lending (last 1 month) (115) | 34.7259% | **34.7259%** | ✅ |
| 2p | prior | `current_loan_to_value` @ 2026-04-30 | New lending (last 1 month) (143) | 34.4568% | **34.4568%** | ✅ |
| 3 | movement | `current_interest_rate` | New lending (last 1 month) (115) | 6.3391% | **6.3391%** | ✅ |
| 3p | prior | `current_interest_rate` @ 2026-04-30 | New lending (last 1 month) (143) | 6.4597% | **6.4597%** | ✅ |

*(plus 15 composition / cohort / forecast / timing findings on this run, reconciled in §12.4 but not tabulated here)*

**8. Final answer** (RECORDED, verbatim)

> Across 2026-04-30 → 2026-06-30, New lending (last 1 month), 115 loans: Current Outstanding Balance £21.4m → £18.3m (−£3.2m); Current Loan To Value 34.46% → 34.73% (+0.27pp); Current Interest Rate 6.46% → 6.34% (-0.12pp). Current profile: Region — South East 43.7%, London 23.4%, South West 11.4%; LTV band — 30-40% 47.7%, 20-30% 25.0%, 40-50% 23.5%; Borrower age band — 70-75 36.7%, 65-70 30.6%, 60-65 12.7%; Ticket size — 200-300k 31.9%, 150-200k 26.5%, 100-150k 22.4%; Interest rate band — 6-7% 50.7%, 5-6% 25.7%, 7-8% 16.9%.

Calculated: Composed governed capabilities.

| | |
|---|---|
| ok | `True` |
| controlled refusal | `False` |
| semantic guard | `ok` [('comparison_period', 'applied'), ('grouping_dimension', 'applied')] |

---

### Q1.2 — alderbridge / forced_llm
*Chosen because: the same family reached by a different window (RECENT = L3M) and a different phrasing, under a forced model parse.*

**1. Question** (RECORDED)

> Are we originating different types of loans now compared with a few months ago?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `llm` |
| mode detail | `llm_repaired` |
| LLM failure | `None` |
| repeats in this arm | 5, all reaching `analytical_composition` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | MIX_PROFILE, MOVEMENT_TREND, VINTAGE_COHORT |
| operations | SNAPSHOT, COMPARISON, CHANGE, ATTRIBUTION, DIVERGENCE, DELTA, RANKING, EVOLUTION |
| concept signals matched | comparison, change_trend |
| lending windows named | recent |
| lending role | population |
| structural requirements | period_comparison |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | months_on_book le 3 |
| predicates unavailable | none |
| rows before → after | 11035 → 258 |
| periods compared | 2026-04-30, 2026-06-30 |
| narrowed to | — |

**5. Deterministic capabilities invoked** (RECORDED)

| | |
|---|---|
| route | `analytical_composition` |
| plan intent | `origination_profile_change` |
| capabilities | `period_movement`, `population_profile` |
| required finding kinds | movement |
| composition version | 1.0.0 · plan origin deterministic |

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

| # | kind | measure | population (rows) | delivered | **independent truth** | agrees |
|---|---|---|---|---|---|---|
| 1 | movement | `current_outstanding_balance` | Recent lending (last 3 months) (258) | £39.7m | **£39.7m** | ✅ |
| 1p | prior | `current_outstanding_balance` @ 2026-04-30 | Recent lending (last 3 months) (367) | £53.2m | **£53.2m** | ✅ |
| 2 | movement | `current_loan_to_value` | Recent lending (last 3 months) (258) | 34.7274% | **34.7274%** | ✅ |
| 2p | prior | `current_loan_to_value` @ 2026-04-30 | Recent lending (last 3 months) (367) | 34.1750% | **34.1750%** | ✅ |
| 3 | movement | `current_interest_rate` | Recent lending (last 3 months) (258) | 6.3961% | **6.3961%** | ✅ |
| 3p | prior | `current_interest_rate` @ 2026-04-30 | Recent lending (last 3 months) (367) | 6.4768% | **6.4768%** | ✅ |

*(plus 15 composition / cohort / forecast / timing findings on this run, reconciled in §12.4 but not tabulated here)*

**8. Final answer** (RECORDED, verbatim)

> Across 2026-04-30 → 2026-06-30, Recent lending (last 3 months), 258 loans: Current Outstanding Balance £53.2m → £39.7m (−£13.5m); Current Loan To Value 34.17% → 34.73% (+0.55pp); Current Interest Rate 6.48% → 6.40% (-0.08pp). Current profile: Region — South East 37.6%, London 22.7%, South West 12.8%; LTV band — 30-40% 49.3%, 20-30% 25.4%, 40-50% 22.5%; Borrower age band — 70-75 34.4%, 65-70 29.9%, 60-65 15.2%; Ticket size — 200-300k 30.8%, 150-200k 23.0%, 100-150k 23.0%; Interest rate band — 6-7% 51.2%, 5-6% 23.9%, 7-8% 19.5%.

Calculated: Composed governed capabilities.

| | |
|---|---|
| ok | `True` |
| controlled refusal | `False` |
| semantic guard | `None` [] |

---

### Q3.1 — alderbridge / production
*Chosen because: a DIFFERENT GOVERNED DATASET — the pipeline extract, not the loan tape.*

**1. Question** (RECORDED)

> How much do we currently have at offer and how much of it is likely to complete?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `llm` |
| mode detail | `validation_failed` |
| LLM failure | `parse_failure` |
| repeats in this arm | 5, all reaching `analytical_composition` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | PIPELINE, FORECAST_PROJECTION |
| operations | CONVERSION, RUN_RATE, EXPECTED_COMPLETION, PROJECT_VALUE |
| concept signals matched | forecast |
| lending windows named | — |
| lending role | — (none named) |
| structural requirements | pipeline_dataset, period_comparison, forecast |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | — |
| predicates unavailable | none |
| rows before → after | None → None |
| periods compared | — |
| narrowed to | [{"field": "pipeline_stage", "value": "OFFER", "rows": 157, "dataset": "pipeline"}] |

**5. Deterministic capabilities invoked** (RECORDED)

| | |
|---|---|
| route | `analytical_composition` |
| plan intent | `pipeline_offer_outlook` |
| capabilities | `pipeline_stock`, `pipeline_completion_forecast` |
| required finding kinds | measure, forecast |
| composition version | 1.0.0 · plan origin deterministic |

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

| # | kind | measure | population (rows) | delivered | **independent truth** | agrees |
|---|---|---|---|---|---|---|

*(plus 3 composition / cohort / forecast / timing findings on this run, reconciled in §12.4 but not tabulated here)*

**8. Final answer** (RECORDED, verbatim)

> Offer stage pipeline is £29.4m across 157 case(s) as at 2026-06-29. Expected completion amount from pipeline cases at Offer stage: £5.0m. Expected to land: 2026-07 £5.0m.

Calculated: Composed governed capabilities.

| | |
|---|---|
| ok | `True` |
| controlled refusal | `False` |
| semantic guard | `ok` [('row_population', 'unsupported')] |

---

### Q4.2 — alderbridge / production
*Chosen because: THE FAIL-CLOSED REFUSAL. Baseline answered '11,035 loans'.*

**1. Question** (RECORDED)

> How many loans are we completing at the moment?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `llm` |
| mode detail | `llm_repaired` |
| LLM failure | `None` |
| repeats in this arm | 3, all reaching `None` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | PIPELINE |
| operations | MOVEMENT, RUN_RATE |
| concept signals matched | — |
| lending windows named | — |
| lending role | — (none named) |
| structural requirements | pipeline_dataset, period_comparison |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | — |
| predicates unavailable | none |
| rows before → after | None → None |
| periods compared | — |
| narrowed to | — |

**5. Deterministic capabilities invoked** (RECORDED)

**None — no analytical plan was built.** Route: `(none: the point-in-time path)`. This is the correct outcome for this question; see the note below.

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

**No findings — nothing was computed.** That is the point of this trace.

**8. Final answer** (RECORDED, verbatim)

> I understood this as a pipeline question, but I have not answered it: this asks about the pipeline (applications, offers or completions), which is a different governed dataset from the funded book; and this asks how something changed, which needs two governed reporting snapshots to compare. I have NOT substituted a current-position figure, because that would answer a different question from the one you asked. Ask for the governed analytic directly — for example the concentration limit tests, the completion run-rate, the pipeline at a named stage, or a named measure compared across two reporting periods — and I will compute it.

| | |
|---|---|
| ok | `False` |
| controlled refusal | `True` |
| semantic guard | `refuse` [('pipeline_dataset', 'unavailable'), ('period_comparison', 'unavailable')] |

---

### Q6.2 — kestrelmoor / production
*Chosen because: governed FLAG SETTLING: no analytical plan at all, the existing limits route answers. Second book.*

**1. Question** (RECORDED)

> Where are we closest to our limits?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `llm` |
| mode detail | `llm` |
| LLM failure | `None` |
| repeats in this arm | 3, all reaching `risk_limits` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | LIMITS_CONCENTRATION |
| operations | STATUS, HEADROOM, RANKING, CONCENTRATION |
| concept signals matched | limits |
| lending windows named | — |
| lending role | — (none named) |
| structural requirements | limit_evidence |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | — |
| predicates unavailable | none |
| rows before → after | None → None |
| periods compared | — |
| narrowed to | — |

**5. Deterministic capabilities invoked** (RECORDED)

**None — no analytical plan was built.** Route: `risk_limits`. This is the correct outcome for this question; see the note below.

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

**No findings — nothing was computed.** That is the point of this trace.

**8. Final answer** (RECORDED, verbatim)

> Contractual risk limits are unavailable for this portfolio (No Schedule 8 limits available — extraction required.). I can show observed concentrations once limits are provided.

Calculated: Concentration limits vs the governing document.

| | |
|---|---|
| ok | `True` |
| controlled refusal | `False` |
| semantic guard | `None` [] |

---

### Q7.2 — alderbridge / production
*Chosen because: an ASYMMETRIC pair — a segment against a window (Back Book vs RECENT).*

**1. Question** (RECORDED)

> Are older loans riskier than the loans we've originated recently?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `llm` |
| mode detail | `llm_repaired` |
| LLM failure | `None` |
| repeats in this arm | 5, all reaching `analytical_composition` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | MIX_PROFILE, VINTAGE_COHORT |
| operations | SNAPSHOT, COMPARISON, DIVERGENCE |
| concept signals matched | comparison |
| lending windows named | back_book, recent |
| lending role | population |
| structural requirements | population_comparison |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | seasoning_segment = Back Book, months_on_book le 3 |
| predicates unavailable | none |
| rows before → after | 11035 → 9858 |
| periods compared | — |
| narrowed to | [{"field": "seasoning_segment", "value": "Back Book", "rows": 9858, "dataset": "funded"}] |

**5. Deterministic capabilities invoked** (RECORDED)

| | |
|---|---|
| route | `analytical_composition` |
| plan intent | `vintage_risk_comparison` |
| capabilities | `portfolio_snapshot`, `portfolio_snapshot`, `vintage_analysis` |
| required finding kinds | comparison, cohort |
| composition version | 1.0.0 · plan origin deterministic |

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

| # | kind | measure | population (rows) | delivered | **independent truth** | agrees |
|---|---|---|---|---|---|---|
| 1 | measure | `loan_count` | Back Book (13+ months) (9858) | 9,858 | **9,858** | ✅ |
| 2 | measure | `current_outstanding_balance` | Back Book (13+ months) (9858) | £1.79bn | **£1.79bn** | ✅ |
| 3 | measure | `current_loan_to_value` | Back Book (13+ months) (9858) | 43.9657% | **43.9657%** | ✅ |
| 4 | measure | `current_interest_rate` | Back Book (13+ months) (9858) | 6.5699% | **6.5699%** | ✅ |
| 5 | measure | `youngest_borrower_age` | Back Book (13+ months) (9858) | 71.7988 | **71.7988** | ✅ |
| 6 | measure | `loan_count` | Recent lending (last 3 months) (258) | 258 | **258** | ✅ |
| 7 | measure | `current_outstanding_balance` | Recent lending (last 3 months) (258) | £39.7m | **£39.7m** | ✅ |
| 8 | measure | `current_loan_to_value` | Recent lending (last 3 months) (258) | 34.7274% | **34.7274%** | ✅ |
| 9 | measure | `current_interest_rate` | Recent lending (last 3 months) (258) | 6.3961% | **6.3961%** | ✅ |
| 10 | measure | `youngest_borrower_age` | Recent lending (last 3 months) (258) | 68.3527 | **68.3527** | ✅ |
| 11 | comparison | `loan_count` | Back Book (13+ months) (9858) | 9,858 | **9,858** | ✅ |
| 11c | comparand | `loan_count` | Recent lending (last 3 months) (258) | 258 | **258** | ✅ |
| 12 | comparison | `current_outstanding_balance` | Back Book (13+ months) (9858) | £1.79bn | **£1.79bn** | ✅ |
| 12c | comparand | `current_outstanding_balance` | Recent lending (last 3 months) (258) | £39.7m | **£39.7m** | ✅ |
| 13 | comparison | `current_loan_to_value` | Back Book (13+ months) (9858) | 43.9657% | **43.9657%** | ✅ |
| 13c | comparand | `current_loan_to_value` | Recent lending (last 3 months) (258) | 34.7274% | **34.7274%** | ✅ |
| 14 | comparison | `current_interest_rate` | Back Book (13+ months) (9858) | 6.5699% | **6.5699%** | ✅ |
| 14c | comparand | `current_interest_rate` | Recent lending (last 3 months) (258) | 6.3961% | **6.3961%** | ✅ |
| 15 | comparison | `youngest_borrower_age` | Back Book (13+ months) (9858) | 71.7988 | **71.7988** | ✅ |
| 15c | comparand | `youngest_borrower_age` | Recent lending (last 3 months) (258) | 68.3527 | **68.3527** | ✅ |

*(plus 13 composition / cohort / forecast / timing findings on this run, reconciled in §12.4 but not tabulated here)*

**8. Final answer** (RECORDED, verbatim)

> Back Book (13+ months) against Recent lending (last 3 months) (9,858 vs 258 loans): Loan Count 9,858 vs 258 (+9,600); Current Outstanding Balance £1.79bn vs £39.7m (+£1.75bn); Current Loan To Value 43.97% vs 34.73% (+9.24pp); Current Interest Rate 6.57% vs 6.40% (+0.17pp); Youngest Borrower Age 71.8 vs 68.4 (+3.4). Across 13 governed origination vintage(s), 2014 holds £67.6m at 54.47% weighted-average LTV and 2026 holds £71.8m at 34.59%.

Calculated: Composed governed capabilities.

| | |
|---|---|
| ok | `True` |
| controlled refusal | `False` |
| semantic guard | `None` [] |

---

### Q8.3/provenance — kestrelmoor / production
*Chosen because: §8 control, resolver 1 of 3.*

**1. Question** (RECORDED)

> Are direct and acquired balances developing differently over time?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `llm` |
| mode detail | `llm_repaired` |
| LLM failure | `None` |
| repeats in this arm | 5, all reaching `analytical_composition` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | MOVEMENT_TREND |
| operations | DELTA, TREND, RANKING, ATTRIBUTION |
| concept signals matched | comparison, change_trend |
| lending windows named | — |
| lending role | — (none named) |
| structural requirements | period_comparison |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | — |
| predicates unavailable | none |
| rows before → after | None → None |
| periods compared | 2026-04-30, 2026-06-30 |
| narrowed to | — |

**5. Deterministic capabilities invoked** (RECORDED)

| | |
|---|---|
| route | `analytical_composition` |
| plan intent | `population_movement_comparison` |
| capabilities | `period_movement`, `period_movement` |
| required finding kinds | movement, comparison |
| composition version | 1.0.0 · plan origin deterministic |

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

| # | kind | measure | population (rows) | delivered | **independent truth** | agrees |
|---|---|---|---|---|---|---|
| 1 | movement | `current_outstanding_balance` | Direct (5612) | £558.0m | **£558.0m** | ✅ |
| 1p | prior | `current_outstanding_balance` @ 2026-04-30 | Direct (5302) | £530.4m | **£530.4m** | ✅ |
| 2 | movement | `current_outstanding_balance` | Acquired (6643) | £1.21bn | **£1.21bn** | ✅ |
| 2p | prior | `current_outstanding_balance` @ 2026-04-30 | Acquired (6425) | £1.17bn | **£1.17bn** | ✅ |
| 3 | comparison | `current_outstanding_balance` | Direct (5612) | £558.0m | **£558.0m** | ✅ |
| 3c | comparand | `current_outstanding_balance` | Acquired (6643) | £1.21bn | **£1.21bn** | ✅ |

**8. Final answer** (RECORDED, verbatim)

> Across 2026-04-30 → 2026-06-30, Direct, 5,612 loans: Current Outstanding Balance £530.4m → £558.0m (+£27.7m). Across 2026-04-30 → 2026-06-30, Acquired, 6,643 loans: Current Outstanding Balance £1.17bn → £1.21bn (+£40.0m). Direct against Acquired (5,612 vs 6,643 loans): Current Outstanding Balance £558.0m vs £1.21bn (−£656.4m).

Calculated: Composed governed capabilities.

| | |
|---|---|
| ok | `True` |
| controlled refusal | `False` |
| semantic guard | `None` [] |

---

### Q8.3/seasoning — kestrelmoor / production
*Chosen because: §8 control, resolver 2 of 3.*

**1. Question** (RECORDED)

> Are the front book and the back book balances developing differently over time?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `llm` |
| mode detail | `llm_repaired` |
| LLM failure | `None` |
| repeats in this arm | 5, all reaching `analytical_composition` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | MOVEMENT_TREND, VINTAGE_COHORT |
| operations | DELTA, TREND, RANKING, ATTRIBUTION, SNAPSHOT, COMPARISON, DIVERGENCE, EVOLUTION |
| concept signals matched | comparison, change_trend |
| lending windows named | front_book, back_book |
| lending role | population |
| structural requirements | period_comparison, population_comparison |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | seasoning_segment = Front Book, seasoning_segment = Back Book |
| predicates unavailable | none |
| rows before → after | 12255 → 3020 |
| periods compared | 2026-04-30, 2026-06-30 |
| narrowed to | [{"field": "seasoning_segment", "value": "Front Book", "rows": 3020, "dataset": "funded"}, {"field": "seasoning_segment", "value": "Back Book", "rows": 9235, "dataset": "funded"}] |

**5. Deterministic capabilities invoked** (RECORDED)

| | |
|---|---|
| route | `analytical_composition` |
| plan intent | `population_movement_comparison` |
| capabilities | `period_movement`, `period_movement` |
| required finding kinds | movement, comparison |
| composition version | 1.0.0 · plan origin deterministic |

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

| # | kind | measure | population (rows) | delivered | **independent truth** | agrees |
|---|---|---|---|---|---|---|
| 1 | movement | `current_outstanding_balance` | Front Book (0-12 months) (3020) | £299.9m | **£299.9m** | ✅ |
| 1p | prior | `current_outstanding_balance` @ 2026-04-30 | Front Book (0-12 months) (2626) | £262.4m | **£262.4m** | ✅ |
| 2 | movement | `current_outstanding_balance` | Back Book (13+ months) (9235) | £1.47bn | **£1.47bn** | ✅ |
| 2p | prior | `current_outstanding_balance` @ 2026-04-30 | Back Book (13+ months) (9101) | £1.44bn | **£1.44bn** | ✅ |
| 3 | comparison | `current_outstanding_balance` | Front Book (0-12 months) (3020) | £299.9m | **£299.9m** | ✅ |
| 3c | comparand | `current_outstanding_balance` | Back Book (13+ months) (9235) | £1.47bn | **£1.47bn** | ✅ |

**8. Final answer** (RECORDED, verbatim)

> Across 2026-04-30 → 2026-06-30, Front Book (0-12 months), 3,020 loans: Current Outstanding Balance £262.4m → £299.9m (+£37.6m). Across 2026-04-30 → 2026-06-30, Back Book (13+ months), 9,235 loans: Current Outstanding Balance £1.44bn → £1.47bn (+£30.1m). Front Book (0-12 months) against Back Book (13+ months) (3,020 vs 9,235 loans): Current Outstanding Balance £299.9m vs £1.47bn (−£1.17bn).

Calculated: Composed governed capabilities.

| | |
|---|---|
| ok | `True` |
| controlled refusal | `False` |
| semantic guard | `ok` [('grouping_dimension', 'applied')] |

---

### Q8.3/dimension_value — kestrelmoor / production
*Chosen because: §8 control, resolver 3 of 3.*

**1. Question** (RECORDED)

> Are North West and Scotland balances developing differently over time?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `deterministic_fallback_after_llm_failure` |
| mode detail | `deterministic_fallback` |
| LLM failure | `parse_failure` |
| repeats in this arm | 5, all reaching `analytical_composition` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | MOVEMENT_TREND |
| operations | DELTA, TREND, RANKING, ATTRIBUTION |
| concept signals matched | comparison, change_trend |
| lending windows named | — |
| lending role | — (none named) |
| structural requirements | period_comparison |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | collateral_geography = North West, collateral_geography = Scotland |
| predicates unavailable | none |
| rows before → after | 12255 → 2935 |
| periods compared | 2026-04-30, 2026-06-30 |
| narrowed to | [{"field": "collateral_geography", "value": "North West", "rows": 2935, "dataset": "funded"}, {"field": "collateral_geography", "value": "Scotland", "rows": 1987, "dataset": "funded"}] |

**5. Deterministic capabilities invoked** (RECORDED)

| | |
|---|---|
| route | `analytical_composition` |
| plan intent | `population_movement_comparison` |
| capabilities | `period_movement`, `period_movement` |
| required finding kinds | movement, comparison |
| composition version | 1.0.0 · plan origin deterministic |

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

| # | kind | measure | population (rows) | delivered | **independent truth** | agrees |
|---|---|---|---|---|---|---|
| 1 | movement | `current_outstanding_balance` | Region North West (2935) | £416.7m | **£416.7m** | ✅ |
| 1p | prior | `current_outstanding_balance` @ 2026-04-30 | Region North West (2815) | £403.0m | **£403.0m** | ✅ |
| 2 | movement | `current_outstanding_balance` | Region Scotland (1987) | £291.7m | **£291.7m** | ✅ |
| 2p | prior | `current_outstanding_balance` @ 2026-04-30 | Region Scotland (1897) | £279.7m | **£279.7m** | ✅ |
| 3 | comparison | `current_outstanding_balance` | Region North West (2935) | £416.7m | **£416.7m** | ✅ |
| 3c | comparand | `current_outstanding_balance` | Region Scotland (1987) | £291.7m | **£291.7m** | ✅ |

**8. Final answer** (RECORDED, verbatim)

> Across 2026-04-30 → 2026-06-30, Region North West, 2,935 loans: Current Outstanding Balance £403.0m → £416.7m (+£13.8m). Across 2026-04-30 → 2026-06-30, Region Scotland, 1,987 loans: Current Outstanding Balance £279.7m → £291.7m (+£12.1m). Region North West against Region Scotland (2,935 vs 1,987 loans): Current Outstanding Balance £416.7m vs £291.7m (+£125.0m).

Calculated: Composed governed capabilities.

| | |
|---|---|
| ok | `True` |
| controlled refusal | `False` |
| semantic guard | `ok` [('geographic_scope', 'applied'), ('geographic_scope', 'applied')] |

---

### Q9.3 — kestrelmoor / forced_llm
*Chosen because: a composed FORECAST on the second book, under a forced model parse.*

**1. Question** (RECORDED)

> If the current pipeline converts as expected, what will our funded balance be?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `deterministic_fallback_after_llm_failure` |
| mode detail | `deterministic_fallback` |
| LLM failure | `parse_failure` |
| repeats in this arm | 3, all reaching `analytical_composition` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | PIPELINE, FORECAST_PROJECTION |
| operations | CONVERSION, EXPECTED_COMPLETION, PROJECT_VALUE |
| concept signals matched | forecast |
| lending windows named | — |
| lending role | — (none named) |
| structural requirements | pipeline_dataset, forecast |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | — |
| predicates unavailable | none |
| rows before → after | None → None |
| periods compared | — |
| narrowed to | — |

**5. Deterministic capabilities invoked** (RECORDED)

| | |
|---|---|
| route | `analytical_composition` |
| plan intent | `funded_balance_outlook` |
| capabilities | `funded_balance_forecast`, `pipeline_completion_forecast` |
| required finding kinds | forecast |
| composition version | 1.0.0 · plan origin deterministic |

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

| # | kind | measure | population (rows) | delivered | **independent truth** | agrees |
|---|---|---|---|---|---|---|
| 1 | measure | `current_outstanding_balance` | the whole funded book (12255) | £1.77bn | **£1.77bn** | ✅ |

*(plus 7 composition / cohort / forecast / timing findings on this run, reconciled in §12.4 but not tabulated here)*

**8. Final answer** (RECORDED, verbatim)

> Current funded balance is £1.77bn as at 2026-06-30. Gross pipeline in the governed extract is £76.7m as at 2026-06-29. Expected completions from the pipeline: £12.5m. Forecast funded balance: £1.78bn. Expected completion amount from the open pipeline: £7.7m. Expected to land: 2026-07 £3.9m; 2026-08 £1.9m; 2026-09 £1.8m.

Calculated: Composed governed capabilities.

| | |
|---|---|
| ok | `True` |
| controlled refusal | `False` |
| semantic guard | `ok` [('projection', 'applied')] |



### 13.3 The raw evidence, and how to recompute it

Every number in this report is recomputable by someone who does not trust the
narration. The measurement artefacts are committed at
`due_diligence/evidence/analytical_intent_v1/` — the four run files hold all 752
responses verbatim as the production endpoint returned them, including the ones
that refused.

| File (as committed) | Bytes | SHA-256 of the **uncompressed** JSON |
|---|---|---|
| `v1_nl_alderbridge_production.json.gz` | 94,841 (from 2,478,770) | `50f4b38e85f121b98a9f6e6f4cfeba84053c1b932527199c0c48204956905514` |
| `v1_nl_alderbridge_forced_llm.json.gz` | 95,083 (from 2,482,727) | `e08fdeb250226b644be1db7423fd60bef6e6a89d56f54a29ac4f1547abaea286` |
| `v1_nl_kestrelmoor_production.json.gz` | 96,416 (from 2,495,500) | `2d2511babd0139274eece83acc25e283c79dbbda68d6348851bb147b55038e3b` |
| `v1_nl_kestrelmoor_forced_llm.json.gz` | 96,287 (from 2,498,863) | `eb89bdf72b48c8544b9ab79ba6fe373d8e22880df31f1e4603ab8e423877b3a5` |
| `nl_bank.py` | 6,877 | `f37729113df3a6734d661ac63f862f9963b592d13bbffe9e7c38566020bd8874` |
| `nl_harness.py` | 6,285 | `de059ef3fd07357092d8109747d0c46405f4329413cd17e4e9a01ec9477e8a9c` |
| `nl_score.py` | 10,490 | `aade14173ab3623196434e833b90a1d112d4106c526381cade6c0cbc4c7164f7` |
| `nl_reconcile.py` | 5,455 | `4802c50be93ff55ec923ab473250e6e30c4a5233456362dae063be769f4b3d39` |
| `v1_final.py` | 5,466 | `786ca261bf33702ceac57f0985e7621ba2643a3fd8b65f3ea7f68a774cd48ea5` |
| `v1_score_run.py` | 3,397 | `3c625636ef0da0ea48e58f2284c85fa08c417a854808f1e619cbcd4c7a4f2b86` |
| `v1_trace.py` | 10,653 | `3306224571fec5b9a4b9c922eb8155a28af8a5bb23bb19476ae7c3b46577dfb8` |

```bash
cd due_diligence/evidence/analytical_intent_v1 && gunzip -k v1_nl_*.json.gz

python v1_final.py       # the distribution, the baseline comparison, the gate
python nl_reconcile.py v1_nl_*.json   # recompute every figure from the CSVs
python v1_trace.py       # regenerate the ten traces in §13.2
python nl_harness.py alderbridge production out.json   # reproduce from scratch
```

**The control that makes the comparison honest:** `nl_score.py` is the scorer
used for *both* the frozen baseline and this run, unmodified. `v1_final.py`
recomputes the baseline column of every table in §12 from the baseline's own run
files, and reproduces the previously published figures exactly — 405 correct,
147 silent semantic errors, 40 incorrect-successful, 187 unsafe. A scorer bent
to flatter this change would no longer reproduce them.

**What the run files contain that a summary cannot:** the parser provenance per
run (`llm`, `deterministic`, or `deterministic_fallback_after_llm_failure`), the
governed population ledger with row counts before and after, every structured
finding with its period and population, the semantic-guard verdict and facets,
and the answer text. §13.2 traces ten of them; the other 742 are in the files.


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

### 14.3 The gates, all measured on the shipped tree

| Gate | Required | Measured |
|---|---|---|
| INCORRECT_SUCCESSFUL over 752 runs | 0 | **0** |
| SILENT_SEMANTIC_ERROR over 752 runs | 0 | **0** |
| HARD_FAILURE over 752 runs | 0 | **0** |
| CORRECT / DISCLOSED | ≥ 80% | **89.4%** (672 / 752) |
| remainder HONEST_PARTIAL or SAFE_REFUSAL | all | **80 of 80** |
| numeric findings reconciled | all | **6,856 / 6,856**, 0 mismatches |
| nine canonical CFO questions, both books | all green | **18 / 18 `ok=True`**; 16 byte-identical, Q7 changed orientation to follow the question (§11.3) |
| 30-question simple-MI bank | no change | **0 of 30 changed** |
| 252-question calibration bank | green | **245 passed, 13 xfailed** |
| named baseline suites (P1I, P1J-1, P1L, P1M, P1N, P1E, fabricated-population, analytical layer) | green | **green** |
| full test suite | green | **9,061 passed, 0 failed** |
| regressions into an unsafe class | none | **none** |
| capabilities lost | none | **none** |

### 14.4 Verdict

The brief asked for better analytical-intent recognition without touching the
deterministic analytics beneath it, and set the primary gate on safety rather
than on coverage. Both were met, and the second was met by the first: the way
the unsafe count reached zero was by routing questions to capabilities that
already existed and refusing the ones nothing can answer — not by inferring
more freely.

What was actually built is small. One module that classifies and routes and
computes nothing; four governed lending windows on an axis that already existed;
two governed flags settled before routing; and one structural check that refuses
an answer which does not carry what its question needs. No engine changed, no
route was rewritten, no precedence was altered, and no test phrasing was added
as a string patch.

Three qualifications belong in the same breath as the result.

**Three of the four route contentions this change created were caught by
regression gates, not by inspection** (§9.1, §11.5). The 752-run bank found one;
the full suite found two more, and both were live defects — a question answered
in the wrong unit and a route that failed outright. That is the strongest
argument in this report for running every gate rather than the ones that looked
relevant, and it is an argument about my own reliability, not the product's.

**Q1.3 is a real cost.** *"How does recent lending compare with what we were
originating earlier in the year?"* was a silent wrong answer before and is a
stated refusal in 13 of 20 runs now. Safer, and not a correct answer.

**Two capability gaps are named rather than closed** (§14.2): a pipeline stock
question with no forward element, and a count-based run rate. Both refuse
honestly; both would need new plan surface or new analytics, which the brief put
out of scope.

ANALYTICAL INTENT V1: PASS
