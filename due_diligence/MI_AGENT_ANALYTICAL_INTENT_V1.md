# MI Agent — Analytical Intent Family Hardening V1

**Scope.** Improve analytical-intent recognition without changing the underlying
deterministic analytics. No new analytical mathematics, no replacement of the
analytical capability layer, no rewritten specialist route, no weakened safety
guard, and no test phrasing added as an exact string patch.

**Baseline.** Branch `claude/mi-analytical-capability-layer-vlkjfw`, SHA
`104c89d` — the frozen 752-run measurement in
`due_diligence/MI_AGENT_ANALYTICAL_NL_ROBUSTNESS.md`.

**Change.** Two production commits (`a6331aa`, `cf7e673`), one new module and
one new test file. Full diff: 11 files, +1,780 / −70 (of which 303 lines are the new
test suite and 108 the architecture note).

---

## 1. Executive verdict

*(filled at the end of this document)*

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
| `tests/test_analytical_intent_boundary.py` (**new**) | 81 passed |
| **Combined focused run** | **757 passed, 13 xfailed** |

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

