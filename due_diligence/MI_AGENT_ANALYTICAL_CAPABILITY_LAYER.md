# MI Agent — Analytical Capability Layer

**Purpose:** let a CFO ask nine analytical questions in normal language and have
Trakt compose *existing* governed deterministic capabilities into a trustworthy
answer, rather than relying on one giant query specification or an LLM-generated
calculation.
**Baseline:** `7dffa6c` — the frozen second-book commercial acceptance, with the
fabricated-population safety fix (`48d9d44`) merged and authoritative.
**Mode:** inventory → thin layer → two-book acceptance → regression.

---

## 1. Current architecture, before

```
question
  └─► ParsedQuestion.parse (ONE parse)
        └─► RecogniserRegistry           14 recognisers, ordered, capability-gated
              └─► ONE deterministic capability
                    └─► governed response envelope + P0 execution receipt
        └─► (unmatched) point-in-time executor
```

The registry is deliberate and explicit about its ceiling — its own module
docstring says it *"routes to ONE deterministic capability and stops"*, and
`docs/mi_query_agent_architecture.md` §11 reserves the layer above it for
multi-capability workflows that had not been built.

That ceiling is exactly what the nine questions hit. Measured on the
demonstration book at the frozen baseline:

| Q | Baseline behaviour | Class |
|---|---|---|
| Q1 profile of new originations | **refused** — the front-book population reached a route that calculates whole-book, so P1L refused | orchestration |
| Q2 when do we reach £100m | **answered** — `forecast_extrapolation` | already correct |
| Q3 offers, expected completions, timing | **refused** — nothing composes pipeline stock with a completion forecast | orchestration + data |
| Q4 completion run rate | **answered** — `forecast_extrapolation` | already correct |
| Q5 likely to breach in three months | **answered** (current position only) | data / configuration |
| Q6 limits closest to breaching | **answered** — nearest-to-limit named | already correct |
| Q7 older vintages vs the front book | **"Here is the result for your query, covering 2 group(s)"** — `ok=True`, a count of the front book by vintage, no comparison, no figures | **wrong question answered successfully** |
| Q8 balance from X relative to Y | **refused** — one governed capability returns one measure | orchestration |
| Q9 forecast funded balance | **refused** — *"Forecast Funded Balance is not available in this dataset"* | orchestration + wrong reason |

Q7 is the one that mattered most: a plausible-looking success to a question the
product had not answered.

## 2. Existing deterministic capability inventory

Almost all of the mathematics already existed. Nothing below was written for
this sprint.

| Analytical operation | Owner (unchanged) | What it already produces |
|---|---|---|
| governed aggregation (sum / average / weighted average / share / count) | `mi_workflows.engine.aggregate` | value + unit + population + valid/excluded rows + denominator |
| governed distribution & concentration ranking | `mi_workflows.engine.distribution` / `ranked_distribution` | count & exposure shares, cumulative share, explicit unknown bucket |
| comparison arithmetic and directionality | `mi_workflows.engine.compare_values` / `directionality_verdict` | absolute + relative difference, one definition platform-wide |
| period change across two governed snapshots | `mi_agent.period_change.run_period_change_analysis` | per-metric movement, composition shift, units, interpretation, evidence |
| governed dated snapshot supply | `mi_agent_api.period_change_route.build_snapshots` | lens-narrowed frame per reporting date |
| origination vintage / static pool | `mi_agent_api.cohorts.cohort_analysis` | per-vintage balance, count, share, weighted LTV / rate / months-on-book |
| pipeline stock by stage | `mi_agent_api.pipeline_contract.compute_pipeline_snapshot` | case counts, gross amounts, stage breakdown |
| expected completion amount and month | `pipeline_contract._expected_completion_breakdown` + `pipeline_prep` | probability-weighted amount per expected completion month |
| funded + pipeline forecast bridge | `mi_agent_api.forecast_bridge.compute_forecast_bridge` | funded balance + probability-weighted completions → forecast balance |
| completion run-rate & milestone dates | `mi_agent_api.forecast_extrapolation` | base/downside/upside monthly run-rate, threshold dates |
| limits, actuals, headroom, forward states | `mi_agent_api.concentration_tests_api.compute_concentration_tests`, `mi_agent.concentration_tests.forward` | per-test actual vs limit, headroom, expected/full-pipeline states, emerging risks |
| governed populations and their evidence | `mi_agent.population`, `mi_agent.seasoning`, `mi_agent.portfolio_lens` | predicates, rows-before/after, front/back partition, direct/acquired lens |

**Per question, what was actually missing:**

| Q | Missing piece |
|---|---|
| Q1 | **orchestration** — no capability applies a governed population to dated snapshots |
| Q2 | nothing |
| Q3 | **orchestration** (stock + forecast in one answer) and **data** on the demonstration book |
| Q4 | nothing |
| Q5 | **data / configuration** — forward limit states need an approved concentration configuration *and* a governed pipeline |
| Q6 | nothing |
| Q7 | **orchestration** — two governed populations measured and compared, plus the vintage detail |
| Q8 | **orchestration** — two period movements and the difference between them |
| Q9 | **orchestration** — funded position + expected completions + timing in one answer |

No calculation was genuinely absent. Nothing was duplicated.

## 3. Canonical analytical capability set

Ten capabilities, derived backwards from the nine questions. Each declares an
engine that already existed.

| # | Capability | Answers | Engine |
|---|---|---|---|
| 1 | `portfolio_snapshot` | Q7, Q9 | `mi_workflows.engine.aggregate` (BSR-driven) |
| 2 | `population_profile` | Q1 | `mi_workflows.engine.ranked_distribution` |
| 3 | `period_movement` | Q1, Q8 | `mi_agent.period_change.run_period_change_analysis` |
| 4 | `vintage_analysis` | Q7 | `mi_agent_api.cohorts.cohort_analysis` |
| 5 | `pipeline_stock` | Q3 | `pipeline_contract.compute_pipeline_snapshot` |
| 6 | `pipeline_completion_forecast` | Q3, Q9 | `pipeline_contract` governed completion probabilities |
| 7 | `funded_balance_forecast` | Q9 | `forecast_bridge.compute_forecast_bridge` |
| 8 | `completion_run_rate` | Q4 *(route-owned)* | `forecast_extrapolation.run_rate_model` |
| 9 | `threshold_projection` | Q2 *(route-owned)* | `forecast_extrapolation` milestone solver |
| 10 | `concentration_limits` | Q5, Q6 *(route-owned)* | `concentration_tests_api.compute_concentration_tests` |

Capabilities the brief's illustrative list names but which were **not** created,
because the repository and the nine questions do not justify them: *ranking* is
an output option on the distribution and limit capabilities, not an operation
(`ranked_distribution` and headroom ordering already produce it); *contribution
/ attribution* is the funded-bridge route's, untouched; *portfolio/cohort
comparison* is a composition of two `portfolio_snapshot` calls plus the engine's
one comparison definition, not a capability of its own.

## 4. The capability contract

`mi_workflows/analytical/contract.py`, one dataclass:

```
AnalyticalCapability
  id                 stable id
  intent             the question shape it serves
  required_inputs    validated BEFORE any data is read
  optional_inputs
  supported_scopes   total | seasoning | provenance | dimension_value
  datasets           funded | funded_history | pipeline | limits
  engine             the deterministic owner, named for the record
  produces           the finding kinds it emits
  limitations        declared, not folklore
  route_owner        the existing recogniser that already owns this shape
  executor           the thin adapter
```

`route_owner` is the **deference rule**. Capabilities 8–10 are fully implemented
and unit-tested, and are composable, but the planner never claims a question
whose whole answer is one of them — those questions keep the route that already
answers them correctly.

`registry.py` refuses a duplicate id and refuses a capability with no executor:
*a described capability that cannot run is documentation, not a capability.*

Deliberately **not** created: a second field registry, a second business
semantics registry, a second chart engine, a second parser. Measure identity,
statistic identity, aggregation, weighting, units and directionality all stay in
`mi_semantics_field_registry.yaml` and `business_semantics_registry.yaml`.

## 5. Orchestration

```
question
  └─► planner            deterministic plan (or an LLM proposal, VALIDATED)
        └─► registry     required inputs checked before any data is read
              └─► executors   the existing deterministic engines
                    └─► findings     structured, evidenced, unit-carrying
                          └─► engine.compare_values   derived comparisons
                                └─► narrative   composed from findings only
```

**The LLM may choose capabilities; it may not calculate.** `plan_from_proposal`
accepts `{intent, rationale, calls:[{capability, inputs, because}]}` and rejects
the whole proposal on an unknown capability, a missing required input, an
undeclared input, a free-text population, or a population the question never
named. This is the discipline the MI Query parser already applies to an
LLM-proposed `MIQuerySpec`: the model proposes, the deterministic stack
disposes. The deterministic planner is the default and the only path used in
every result below.

**The layer engages only for composition.** `planner.plan_for` returns `None` —
defer — unless it composes two or more capabilities, and never for a question a
`route_owner` covers. `handle` additionally returns `None` when a plan produces
nothing computable, so a book missing the data a plan needs falls through to
exactly the behaviour it had before.

**No adapter computes a financial result.** Enforced structurally: a test parses
every module in `mi_workflows/analytical/` and fails if any imports `pandas` or
`numpy`. What the layer does do — add up subtotals a governed function returned
— happens only at call sites that name that function in the finding's evidence.

## 6. Structured findings

```
Finding(capability, kind, label, metric, population, period, unit, aggregation,
        value, prior_value, change, relative_change,
        comparand, comparand_value, rank, rank_of, share, prior_share,
        limit, headroom, limit_status,
        forecast_value, forecast_date, probability_basis,
        status, note, evidence)
```

Rules the dataclass enforces: an unknown kind is refused at construction; a
finding whose status is not `ok` **must** carry a note saying why. `PopulationRef`
carries rows-before / rows-after and the governed predicate, so a population is
execution evidence rather than a claim. Every `ok` finding names the engine that
produced it and its calculation version.

The narrative reads findings and nothing else, and a test proves it: every
count rendered in an answer must be a population some finding declares.

## 7. Nine-question mapping and results

Demonstration book (Alderbridge, 11,035 loans, £1,964,886,258.21, three
reporting snapshots) with a governed twelve-week pipeline pack.

| Q | Plan | Deterministic capabilities called | Findings | Outcome |
|---|---|---|---|---|
| **Q1** | `origination_profile_change` | `period_movement` → `population_profile` | 18 | **fully answered** |
| **Q2** | *deferred* | `forecast_extrapolation` route | — | **fully answered** (unchanged) |
| **Q3** | `pipeline_offer_outlook` | `pipeline_stock` → `pipeline_completion_forecast` | 3 | **fully answered** |
| **Q4** | *deferred* | `forecast_extrapolation` route | — | **fully answered** (unchanged) |
| **Q5** | *deferred* | `risk_limits` route | — | **partially answered** — current position only |
| **Q6** | *deferred* | `risk_limits` route | — | **fully answered** (unchanged) |
| **Q7** | `vintage_risk_comparison` | `portfolio_snapshot` ×2 → `vintage_analysis` | 28 | **fully answered** |
| **Q8** | `population_movement_comparison` | `period_movement` ×2 | 3 | **fully answered** |
| **Q9** | `funded_balance_outlook` | `funded_balance_forecast` → `pipeline_completion_forecast` | 8 | **fully answered** |

**Delivered answers (Alderbridge).**

* **Q1** — *"Across 2026-04-30 → 2026-06-30, Front Book (0-12 months), 1,177
  loans: balance £182.3m → £171.7m (−£10.6m); LTV 34.40% → 34.71% (+0.31pp);
  rate 6.45% → 6.45%. Current profile: Region — South East 28.3%, London 21.6%,
  South West 12.5%; LTV band — 30-40% 52.6% …"* Disclosed: *"Front Book is a
  rolling population — loans move out of it as they season, so this is the
  movement in the segment, not of one fixed set of loans."*
* **Q3** — *"Offer stage pipeline is £29.4m across 157 case(s) as at 2026-06-29.
  Expected completion amount from pipeline cases at Offer stage: £5.0m. Expected
  to land: 2026-07 £5.0m."* Source-noted: the completion-probability basis, and
  that an empirical rate reads as a floor while the observation window
  right-censors recent cases.
* **Q7** — *"Front Book (0-12 months) against Back Book (13+ months) (1,177 vs
  9,858 loans): balance £171.7m vs £1.79bn; LTV 34.71% vs 43.97% (-9.26pp); rate
  6.45% vs 6.57%; borrower age 68.0 vs 71.8. Across 13 governed origination
  vintage(s), 2014 holds £67.6m at 54.47% weighted-average LTV and 2026 holds
  £71.8m at 34.59%."*
* **Q8** — *"Direct, 7,126 loans: £1.36bn → £1.39bn (+£21.5m). Acquired, 3,909
  loans: £568.3m → £579.4m (+£11.1m). Direct against Acquired: £1.39bn vs
  £579.4m (+£806.1m)."*
* **Q9** — *"Current funded balance is £1.96bn as at 2026-06-30. Gross pipeline
  in the governed extract is £94.4m as at 2026-06-29. Expected completions from
  the pipeline: £15.2m. Forecast funded balance: £1.98bn. Expected to land:
  2026-07 £5.0m; 2026-08 £2.3m; 2026-09 £2.3m."*

**Q8 is a generic pattern, not a fixed pair.** The same plan answers *"how has
the balance from the South East changed relative to London?"* — 2,420 vs 1,380
loans, £516.2m vs £413.8m — and would answer front vs back, resolving each side
through the governed resolver that owns the concept.

**Remaining gap on Q5.** *"…in the next three months"* is answered with the
current limit position. The forward-looking capability already exists —
`concentration_tests.forward.evaluate_forward_states` computes funded /
expected-forecast / full-pipeline states, `expected_breach_horizon` solves the
crossing month, and `identify_emerging_risks` ranks the result — but it runs
only from an **approved concentration configuration**, and the demonstration
book has none, so `compute_concentration_tests` falls back to the Schedule 8
extracted monitor, which has no forward state. This is a configuration and data
gap, not an architectural one; no bespoke forecast-breach calculation was
written, per the brief.

## 8. Two-book results

The second book, **Kestrelmoor** (12,255 loans, £1,772,471,338.39, four
portfolios, three snapshots), is deliberately not the first book: acquired is
68.5% of AuM against 29.5%; the direct/acquired LTV gap is ~14 points against
~0.7; the front book is 24.6% of loans against 10.7%; the geography is North
West / Scotland led rather than South East; tickets are smaller and more skewed;
and it carries realistic missing valuations and regions. It has its own weekly
pipeline pack with a different funnel and ticket profile.

| Q | Alderbridge | Kestrelmoor |
|---|---|---|
| Q1 | ✅ front book 1,177 loans, £182.3m → £171.7m | ✅ front book 3,020 loans, £262.4m → £299.9m |
| Q2 | ✅ already reached | ✅ already reached |
| Q3 | ✅ £29.4m at offer / 157 cases | ✅ £23.6m at offer / 229 cases |
| Q4 | ✅ £16.3m/month | ✅ £33.8m/month |
| Q5 | ⚠️ current position only | ⚠️ **no Schedule 8 limits configured — stated plainly** |
| Q6 | ✅ nearest to limit named | ⚠️ same limits gap, stated plainly |
| Q7 | ✅ LTV 34.71% vs 43.97%, 13 vintages | ✅ LTV 29.14% vs 39.16%, 14 vintages |
| Q8 | ✅ direct £1.39bn vs acquired £579.4m | ✅ direct £558.0m vs **acquired £1.21bn** |
| Q9 | ✅ £1.96bn → £1.98bn | ✅ £1.77bn → £1.78bn |

**The direction of Q8's answer inverts between the books** — direct dominates on
the first, acquired on the second — which is the clearest available evidence
that the capability reads the book rather than a tuned assumption. No
Alderbridge-specific constant, threshold, region, portfolio id or column exists
anywhere in the layer.

**Behaviour with no pipeline at all** (the demonstration book as published):
Q3 defers to the existing pipeline-view refusal, and Q9 refuses with the real
reason — *"No governed pipeline source is available for this book, so a forecast
funded balance cannot be produced. The current funded balance above is not a
forecast."* — replacing the baseline's misleading *"Forecast Funded Balance is
not available in this dataset."*

## 9. Numerical reconciliation

Every delivered figure was recomputed with pandas straight from the fixture
CSVs. The MI read path was never its own oracle, and populations were verified
by row count as well as by value.

| Book | Funded figures reconciled | Pipeline figures reconciled | Variance |
|---|---|---|---|
| Alderbridge | **97 / 97** | **7 / 7** | **zero** |
| Kestrelmoor | **100 / 100** | **7 / 7** | **zero** |

Representative (Alderbridge):

| Figure | Delivered | Independently computed |
|---|---|---|
| front-book balance, closing | 171,736,116.72 | 171,736,116.72 |
| front-book waLTV, closing | 34.706480854528884 | 34.706480854528884 |
| back-book balance | 1,793,150,141.49 | 1,793,150,141.49 |
| back-book waLTV | 43.965661450355995 | 43.96566145035599 |
| direct balance / rows | 1,385,508,582.98 / 7,126 | exact / exact |
| acquired balance / rows | 579,377,675.23 / 3,909 | exact / exact |
| 13 vintage balances + loan counts | exact | exact |
| offer-stage stock / cases | 29,407,505.06 / 157 | exact / exact |
| forecast = funded + expected | 1,980,061,662.36 | 1,980,061,662.36 |

Three defects were caught — two by reconciliation, one by adversarial
self-review — and all three were fixed before acceptance:

* **A mislabel.** The forecast bridge's gross amount is the *whole weekly
  extract*, not the open pipeline; the first draft labelled it "Gross open
  pipeline", which would have overstated what is still to come by £5.6m. It now
  reads "Gross pipeline in the governed extract", the open-stage subtotal is
  carried in evidence, and the settled component is disclosed in business
  language.
* **A pipeline predicate published into the funded row-population ledger.**
  `metadata.populationApplied` is the P1L ledger for the FUNDED book, and an
  offer-stage plan was writing `pipeline_stage = OFFER` into it. A
  `PopulationRef` now declares which governed dataset its rows come from, and
  only funded populations reach the ledger.
* **A population that could not be applied was measured anyway.**
  `apply_population` leaves the frame alone when the book does not carry the
  predicate's column, so a capability could have measured the whole book and
  labelled it "Front Book" — the exact P1L failure mode, at this layer's own
  seam. Every population-taking capability now refuses instead, in the
  population's own words, and the loss reaches the receipt as `unavailable`.
  Covered by a test that builds a book with no seasoning column.

One pre-existing precision limitation was observed and left alone:

* **`cohorts._weighted_avg` rounds to 4 decimal places on the column's stored
  scale**, so on a fraction-stored LTV column the vintage weighted-average LTV
  has 0.01-percentage-point granularity. Pre-existing, immaterial for display,
  reconciled to that granularity and reported here rather than silently
  tolerated. Not introduced by this work and not changed by it.

**Safety counters, both books:**

```
INCORRECT_SUCCESSFUL  = 0
SILENT_SEMANTIC_ERROR = 0
HARD_FAILURE          = 0
```

## 10. Regression results

| Asset | Result |
|---|---|
| 30-question simple-MI regression bank, before vs after | **0 changed answers of 30** — route, ok flag and answer text byte-identical |
| P-gates (P0, P1C–P1N), fabricated population, `mi_agent`, `mi_agent_api`, workflows | **3,113 passed, 1 skipped, 21 xfailed, 0 failed** |
| New analytical layer suite | **82 passed** |
| Full repository suite | *see §12* |

The regression bank covers exactly what the brief named as must-not-regress:
total AuM, WA LTV, borrower age, balance by region, filters, tables, heatmaps,
min/max/median, front/back, direct/acquired, sponsored book, risk limits,
concentration, evolution, temporal compare and portfolio summary.

**Two existing tests were updated, and neither was weakened.** Both are route
*inventories* rather than behaviour assertions:
`test_existing_route_order_is_preserved` excludes workflow-layer routes and
asserts the twelve migrated routes keep their relative order — the new route
joins the excluded set and the assertion is unchanged;
`test_lens_aware_routes_are_declared_on_the_recogniser` enumerates the lens-aware
set, which the new route genuinely joins. No expectation was relaxed, no test
deleted, and no assertion about an existing answer changed.

## 11. What already existed vs what was added

**Already existed (unchanged, and now reachable in composition):** every
calculation in §2 — governed aggregation, distribution, ranking, comparison,
period change, snapshot supply, vintage analysis, pipeline preparation and
probabilities, expected completion timing, the forecast bridge, run-rate
extrapolation, limits and forward limit states, population predicates, the
seasoning partition, the portfolio lens and the P0 execution receipt.

**Added:**

| Component | Lines | What it is |
|---|---|---|
| `mi_workflows/analytical/contract.py` | 367 | capability, finding and plan dataclasses |
| `.../registry.py` | 268 | the ten declarations |
| `.../executors.py` | 1,081 | thin adapters — no arithmetic |
| `.../route.py` | 506 | recogniser registration and the governed envelope |
| `.../planner.py` | 490 | deterministic planning, deference, LLM proposal validation |
| `.../narrative.py` | 278 | findings → prose |
| `.../orchestrator.py` | 257 | plan execution, derived comparisons, evidence |
| `.../populations.py` | 215 | governed population construction + the fabrication guard applied to this layer |
| `.../context.py` | 204 | lazily resolved, memoised governed inputs |
| `.../__init__.py` | 55 | package contract |
| `mi_agent/execution_receipt.py` | +107 | `analytical_evidence` and four evidence-based facet branches |
| `mi_agent_api/*` | +56 | one recogniser registration, one un-narrowed frame resolver, two route-inventory tests updated |
| tests + fixtures | 1,368 | 82 tests, a second book, a weekly pipeline pack |

**No new mathematics was written.** The only arithmetic in the layer is adding
up subtotals a governed function returned, at call sites that say so.

## 12. Governance preserved

Every control the brief lists remains authoritative, and two were **strengthened**:

* **P1L population propagation.** Populations are applied through
  `mi_agent.population.apply_population`, and the layer publishes
  `metadata.populationApplied` from execution evidence. A predicate no finding
  proves is absent, and the receipt then refuses — verified on Q1 and Q7.
* **Fabricated populations.** Every plan's populations are passed through
  `mi_agent.population.fabricated_concepts` *before any data is read*. The layer
  is held to the safety fix's own rule rather than trusted to respect it: a
  plan that would execute a governed population concept the question never
  requested raises and the layer declines. Tested in both directions.
* **The execution receipt was made stricter, not looser.** Four facets
  (comparison period, forward projection, cohort comparison, geographic scope)
  previously reconciled on *route membership*. A composite plan compares periods,
  projects forward or splits a book by cohort only for the questions that ask it
  to, so route identity could not tell those apart; adding the route to those
  sets would have claimed all of them for every analytical answer. Instead the
  layer publishes what it actually computed and the receipt checks that. Every
  new branch is gated on that block being present, so **no other route's
  reconciliation changes at all**.
* Measure, statistic and scope identity, provenance, seasoning, filter
  preservation, denominator correctness and execution receipts are untouched.

No guard was weakened to make a question answerable.

## 13. Remaining genuine gaps

| # | Gap | Class | Evidence |
|---|---|---|---|
| G1 | **Q5's three-month horizon.** Forward limit states need an approved concentration configuration *and* a governed pipeline. Neither test book has an approved configuration, so both fall back to the funded position. | configuration / data — the capability exists | `compute_concentration_tests` returns `SOURCE_APPROVED` only with an active configuration |
| G2 | **Kestrelmoor has no Schedule 8 document**, so Q5/Q6 report limits unavailable. Correct behaviour; a second-book limits gap, not a capability gap. | data | *"Contractual risk limits are unavailable for this portfolio"* |
| G3 | **The demonstration book carries no pipeline.** Q3 and Q9 were exercised against a governed weekly pipeline pack supplied by the harness; without one both refuse honestly. | data | §8 |
| G4 | **A pre-existing parser defect refuses one Q8 phrasing.** *"How has the balance from the front book changed relative to the back book?"* parses to a fabricated `collateral_geography = 'Changed Relative To The'` filter, which P1L then correctly refuses. **Reproduced identically at the frozen baseline**, so it is neither caused nor fixed here. The analytical plan itself resolves the front/back pair correctly. | pre-existing parser | baseline run recorded during this review |
| G5 | **`fabricated_concepts` governs seasoning and provenance only.** G4 shows the same class of fault on a *geography* predicate. Extending the guard is a new safety fix touching the population-planning seam, and is deliberately out of this brief's scope. | recommended follow-up | — |
| G6 | **No governed materiality threshold** is configured, so movements are ranked by observed size within a unit and nothing is called material. Disclosed on every answer. | pre-existing, disclosed | Q1/Q8 warnings |
| G7 | **Empirical completion rates read as a floor** while the observation window right-censors recent cases. The basis and the bias are stated in a source note on every forecast answer. | methodology, disclosed | *"a case that has not yet had time to complete within that window counts against the rate"* |

## 14. Commercial-readiness verdict

A CFO can now ask, in normal language: how the profile of new originations has
changed; what is at offer, how much completes and when; how older vintages
compare with the front book; how one part of the book has moved relative to
another; and what the book is forecast to reach given the pipeline. Each answer
is composed from governed deterministic capabilities, carries the population and
period it was measured over, names the engine that produced every figure, and
states what it could not compute. Two of the five were previously refused, one
was answered wrongly with `ok=True`, and one refused for the wrong reason.

The four questions the product already answered correctly are answered by
exactly the same routes, with byte-identical text.

Every figure on both books reconciles to independently computed truth at zero
variance. The layer is asset-agnostic and fixture-agnostic: the second book
inverts the first book's economics and the answers follow the data. The
remaining gaps are configuration and data, not capability — with one
pre-existing parser defect that this work reproduces at baseline and does not
touch.

**Not merged. Not pushed for release.** Full-suite evidence is recorded below
for review.

| Dimension | Score | Basis |
|---|---|---|
| Composite answerability | **A** | 5 of 5 composite questions answered; 4 of 4 existing answers unchanged |
| Numerical correctness | **A** | 211 figures, two books, zero variance |
| Governance | **A** | P1L and the fabrication guard hold; the receipt was made stricter |
| Generalisation | **A** | inverted economics, same behaviour, no fixture-specific code |
| No regression | **A** | 0 of 30 changed answers; 3,112 accumulated tests green |
| Forward limit analysis (Q5) | **C** | capability exists; blocked on an approved configuration |
| Pipeline availability | **C** | demonstration book carries none; honest refusal without it |

---

ANALYTICAL CAPABILITY LAYER: PASS
