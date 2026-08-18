# MI Query Agent — P1K: Cross-Gate Semantic Consistency

**Status:** analysis complete. **A genuine architectural contradiction was found.**
Per the brief I stopped and reported rather than attempting a redesign. **No
production code was changed in P1K** — the tree is byte-identical to the pushed,
verified P1J-1 HEAD.

---

## 1. Executive verdict

The statement under test was:

> "The semantic constraints and capabilities introduced through P0–P1J-1 compose
> correctly when multiple governed concepts occur in the same question."

**This is DISPROVED.**

It holds — impressively — on the **executor** path. Scope, provenance, seasoning,
row filters, multi-measure sets, dimensions, aggregation and contribution all
compose correctly there, with populations reconciling to independent truth with
**zero variance** across 11 checked compositions.

It **fails on the routed path**, and the failure is systemic rather than
incidental:

> **`spec.filters` is read by exactly ONE of thirteen specialist routes
> (`evolution`). On the other twelve it is silently ignored — while the response
> still displays a spec asserting the filter.**

So a governed population expressed as a filter — which is exactly how P1J-1
represents "the back book" — is **discarded** by concentration, geographic
exposure, period movement, ranked movement, risk limits, comparison, bridge,
cohort progression, forecast and scenario. The answer is computed over the whole
book, returned `ok=True`, and accompanied by a spec that claims it was narrowed.

There is no facet kind for a lost filter/population, so **P0's fail-closed guard
cannot see it**. This is the contradiction: P1J-1 governs a population through a
channel that the routing layer and the P0 ledger do not observe.

**Five silent semantic errors are demonstrated below on real questions.** The
acceptance gate therefore fails.

```
INCORRECT_SUCCESSFUL   = 5   (must be 0)
SILENT_SEMANTIC_ERROR  = 5   (must be 0)
HARD_FAILURE           = 0
```

Two further, narrower cross-gate defects were also found (§7), and three
compositions are simply unsupported and refuse safely (§8).

**P1K CROSS-GATE CONSISTENCY: FAIL** — see the verdict line at the end.

Nothing here is a regression introduced by P1J-1. The seasoning axis merely made
an existing structural gap *reachable*: any row-filter population — not just
seasoning — has always been dropped by these routes.

---

## 2. Current semantic architecture map

### 2.1 Where each concept lives

| Concept | Parser | Governed spec | Execution | Guard | Receipt |
|---|---|---|---|---|---|
| **A. Portfolio scope** | `portfolio_lens.resolve_lens*` — resolved **after** parse | `spec.portfolio_lens` + injected `filters["source_portfolio_id"]` | executor: `mi_agent_workflow` L628-630 → `apply_scope`; routed: `chat_routing._resolve_lens` → `_apply_lens_filter` (dataframe-level, **never the spec**) | `PortfolioScope.fell_back_to_total` warning; `metadata.lensApplied` | `portfolio_scope` / `portfolioCoverage`; `_stamp_routed_scope` |
| **B. Provenance** | `_DIRECT_TERMS` / `_ACQUIRED_TERMS` → `_type_lens` | `PortfolioScope.filters` always emits **ids, never a type string** | `portfolio_scope.apply_scope` | `reject_scope_role_filters` (**LLM path only**) | `coverage_for_frame` |
| **C. Vintage / seasoning** | `seasoning.resolve_segment_population`; `vintage`→`vintage_year` | `filters["seasoning_segment"]`; dims `vintage_year`/`seasoning_bucket` | derived in `funded_prep` → applied as an **ordinary row filter** in `_apply_filters` | `resolve_seasoning_role` (both paths); `_is_seasoning_population` makes a lost seasoning facet REFUSE | `_applied_filter_phrases` |
| **D. Row filters** | `_parse_filters` and friends | `spec.filters`, `spec.unavailable_filters` | `_apply_filters` — **executor only** | unavailable/rejected/zero-row warnings | `describe_filter` |
| **E. Measure identity** | `_detect_metric` / `_resolve_metric` | `spec.metric`, `aggregation`, `weight_field` | `resolve_measures` | `named_measure_concepts` vs `executed_measure_concepts` → `detect_measure_substitution` | `build_receipt(measure=)` |
| **F. Multi-measure** | `detect_measure_set`, `unresolved_measure_slots` | `spec.measures` | `_execute_measure_set` | `KIND_MULTI_MEASURE`, `KIND_UNRESOLVED_MEASURE` | `_measure_set_phrase` |
| **G. Dimension identity** | `_explicit_dimensions` | `spec.dimension(s)`, `hierarchy` | `_resolve_group_column`; `metadata.group_field_keys` | `KIND_GROUPING`, `dimension_substituted` | `build_receipt(dimensions=)` |
| **H. Aggregation** | `_aggregation_intent` | `spec.aggregation`, `weight_field` | `aggregate_series` | `reconcile_measure_aggregations` | `_AGGREGATION_LABELS` |
| **I. Period comparison** | `_detect_periods`, period-change recognition | `temporal_mode`, `compare_periods` | **routes only** | `KIND_COMPARISON_PERIOD`; `check_period_grain` | `_ranked_period` |
| **J. Ranking** | `_detect_ranking`, `_detect_top_n` | `top_n`, `ranking_mode`, `sort_*` | `_apply_top_n`; `period_change.ranking` | `KIND_RANKING`; `ranking_evidence` | `_ranking_phrase` |
| **K. Contribution** | `_contribution_request` | `aggregation == "contribution"` | `_execute_contribution` — **executor only**; every route defers | `KIND_CONTRIBUTION` ∈ NUMBER_OR_SUBJECT → refuse | aggregation label |
| **L. Exposure / EAD** | bare "exposure" → balance; explicit EAD → `exposure_at_default` | `spec.metric` | ordinary measure path | exposure↔balance deliberately **not** a substitution | concentration basis |
| **M. Receipt** | — | — | — | — | `build_receipt` (executor) / `build_routed_receipt` (routed) |
| **N. P0 guard** | — | — | — | `detect_requested_facets` → `reconcile_facets` / `reconcile_routed_facets` → `assess` | `semanticGuard` |

### 2.2 The eight mutations of `spec.filters`, in order

| # | Site | Winner on conflict |
|---|---|---|
| 0 | `normalise_filters` (spec construction) | shape only |
| 1 | `_parse_filters` & friends (deterministic parse) | builds the dict |
| 2 | `_grouped_value_filters` pops a key that is also a grouping dim | removal |
| 3 | `resolve_seasoning_role` (deterministic spec) | sets/canonicalises `seasoning_segment` |
| 4 | `reject_scope_role_filters` — **LLM path only** | pops `source_portfolio_type` |
| 5 | `resolve_seasoning_role` (LLM spec) | as #3 |
| 6 | `ParsedQuestion.merge_filters` in `try_route` | **caller `extra_filters` wins** |
| 7 | `apply_scope` (executor path only) | **governed scope wins** |
| 8 | `extra_filters` merge in `mi_agent_workflow` | **caller wins again, over the governed scope** |

Two structural observations fall straight out of this table:

* **#4 runs on the LLM path only.** A *deterministic* spec is never scope-role
  checked, so a deterministic `source_portfolio_type` predicate can survive
  alongside the lens's `source_portfolio_id` — two concurrent scopes. Not
  observed in the bank, but the asymmetry is real.
* **#8 runs after #7**, so a caller-supplied `source_portfolio_id` overrides the
  governed lens — a precedence inversion against `apply_scope`'s own docstring
  ("lens wins on conflict").

### 2.3 The contradiction, stated precisely

`spec.filters` is read in the entire routing layer at exactly one site
(`chat_routing._filtered_funded_evo`, the `evolution` route, funded dataset
only). Every other `.filters` reference in `chat_routing.py`,
`period_change_route.py` and `mi_workflows/` is `lens.filters` — the portfolio
scope, a different channel.

| Route | Applies portfolio **lens**? | Applies `spec.filters`? |
|---|---|---|
| `evolution` | no | **YES** (the only one) |
| `period_change_analysis`, `period_movement`, `concentration_analysis`, `geo_exposure`, `funded_bridge`, `cohort_progression`, `portfolio_risk_comparison` | yes | **no — silently ignored** |
| `risk_limits`, `forecast_extrapolation`, `temporal_compare`, `scenario`, `cohort_conversion` | **no** (`lens_aware=False`) | **no — but the un-narrowed scope IS disclosed** |

The last row matters: a route that does not apply the lens **tells the user so**
("Scope not narrowed: this risk-limit answer is computed across the whole
platform book"). A **lens-aware** route that drops a `spec.filters` population
says nothing at all — because there is no `KIND_FILTER` / population facet in
`execution_receipt.py` for the guard to raise. Verified by search: **NOT FOUND**.

That asymmetry is the defect. The honest behaviour already exists; it is simply
not wired to the filter channel.

---

## 3. The cross-gate bank

25 questions, at `scratchpad/p1k_bank.py`, each carrying its **expected semantic
plan** (provenance, seasoning, measures, dimension, denominator), not just an
expected number. The harness records REQUESTED / RESOLVED / EXECUTED / RECEIPT
for every run: route, metric, aggregation, measures, dimensions, filters,
seasoning filter, scope ids, executed record count, guard facets, receipt text.

Three further adversarial probes (§4.3) were added once the architecture map
showed where to aim.

---

## 4. Deterministic results

### 4.1 Compositions that hold (15)

| ID | Composition | Executed population | Verdict |
|---|---|---|---|
| X01 | acquired + back book + WA LTV | 3,659 | ✅ |
| X02 | direct + front book + WA LTV | 927 | ✅ |
| X03 | direct + back + balance/count/LTV | 6,199 | ✅ |
| X04 | acquired + front + balance/LTV | 250 | ✅ |
| X08 | direct + age>85 + avg age | 55 | ✅ |
| X10 | selection + largest loan + **share of selected** | acquired | ✅ 0.12% |
| X11 | entire AuM + largest loan + **share of AuM** | whole | ✅ 0.043% |
| X12 | sponsored book = ENTIRE_AUM + 3 measures | 11,035 | ✅ |
| X13 | acquired + back + region + 2 measures | 3,659 | ✅ |
| X14 | direct + front + region + 2 measures | 927 | ✅ |
| X15 | contribution within **back book** | 9,858 | ✅ portfolio WA LTV 43.97 = back-book truth |
| X16 | contribution within **acquired** | 3,909 | ✅ 42.68 = acquired truth |
| X17 | period movement + acquired **lens** | acquired | ✅ |
| X18 | period movement + direct lens, **rate** ranking | direct | ✅ distinguishes rate from absolute |
| X20 | exposure (=balance, not EAD) + direct + back | 6,199 | ✅ |

X10 vs X11 is the denominator contrast the brief asked for, and it holds: the
same question against a selection uses the **selected** book as denominator
(0.12%) and against the whole book uses AuM (0.043%).

### 4.2 Defects found in the bank (4)

| ID | Question | Classification | What happened |
|---|---|---|---|
| **X09** | "What is the balance below 75% LTV in the acquired book?" | **PARSER_ROLE_ERROR → MEASURE_LOSS** (silent) | Answered **WA LTV**, not balance. `ok=True`. The requested measure never appeared in `measures` at all. Isolation shows "total balance **for loans with** LTV below 75%…" is answered correctly — so it is the phrasing "balance below 75% LTV", which P1E deliberately classifies as a *filter subject*, colliding with measure identity. |
| **X21** | "What is EAD and average LTV for the direct back book?" | **CROSS_GATE_CONTRADICTION → MEASURE_LOSS** (silent) | Answered LTV only, `ok=True`, **EAD silently dropped**. Isolation: "What is the EAD of the direct back book?" **refuses correctly** (P1F holds). "EAD and total balance" also drops EAD. **P1F's EAD refusal is defeated by compounding**: the single-measure path sets `metric=exposure_at_default` and fails validation → refusal; the multi-measure path drops the unresolvable leg and proceeds. |
| **X22 / X23** | "How many acquired loans are in the front book?" / "…directly originated loans are in the back book?" | **PARSER_ROLE_ERROR** (safe refusal) | Parser created `collateral_geography: 'Front'` / `'Back'` — a *place* called "Front" — alongside the correct seasoning filter. Zero rows → refusal. This is exactly the P1I-A defect class: **P1I-A masks governed *scope* phrases from the geography resolver, but P1J-1's *seasoning* phrases are not masked.** Populations the ruling says must exist (250 and 6,199 loans) are unreachable by this phrasing. |
| **X06 / X07** | "How does the direct front book compare with the direct back book on LTV?" | **CROSS_GATE_CONTRADICTION** (false refusal, wrong reason) | The plan executed was *correct* — receipt: "grouped by Seasoning Segment · 2 groups · 7,126 loans" over the direct book. But the cohort detector raised a **sourcing** comparison facet because "direct" appears twice, marked it LOST, and refused with *"you asked for a comparison by how the loans were sourced"* — which the user did not ask. **P1G's cohort detection does not know that a provenance word already consumed as SCOPE is not a requested comparison axis.** |

### 4.3 The systemic finding — routed capabilities discard the population

Aimed by the architecture map. Every one returns `ok=True`, displays
`spec.filters` **claiming** `seasoning_segment: Back Book`, and computes over the
**whole book**. No warning, no facet, no receipt line.

| Question | Route | Returned | Truth (back book) |
|---|---|---|---|
| "Where is the back book most concentrated?" | concentration | whole-book concentrations (100% of exposure) | back book only |
| "Where is the back book most concentrated, and how much headroom is left?" | risk limits | whole-book limits | back book only |
| "How much of the back book is concentrated in the top 10 postcodes?" | geo exposure | **0.3825%** | **0.4191%** |
| "Which region grew the most last month in the back book?" | period movement | South East **£516,214,137** | South East **£467,663,554** (**£48.5m** overstated) |
| "What is the largest single-loan exposure in the back book?" | concentration | £841,638.96, share **0.0428%** | share **0.0469%** |

The back book is 89.3% of loans and £1.79bn of £1.96bn, so the errors are
*plausible* — which is what makes them dangerous. A reader has nothing in the
answer, the warnings or the receipt that would reveal the population was dropped.

**Classification: CROSS_GATE_CONTRADICTION + ROUTE_DROPS_SEMANTICS + SILENT
SEMANTIC ERROR ×5.**

---

## 5. Genuine-LLM repeated results

The mandated high-risk subset, 5 runs each, `zero_cost_first` forced off, parser
provenance recorded per run.

| ID | correct | safe refusal | bad | parser provenance |
|---|---|---|---|---|
| X01 | 5 | 0 | 0 | `llm` ×5 |
| X03 | 5 | 0 | 0 | `llm` ×5 |
| X05 | 0 | 5 | 0 | routed (no parse) |
| X09 | 5* | 0 | 0 | `llm` ×5 |
| X10 | 5 | 0 | 0 | routed |
| X13 | 5 | 0 | 0 | `llm` ×5 |
| X15 | 0 | 5 | 0 | `llm` ×5 |
| X17 | 5 | 0 | 0 | routed |
| X20 | 5 | 0 | 0 | `llm` ×5 |
| X23 | 5 | 0 | 0 | `llm_repaired` ×5 |
| X25 | 0 | 5 | 0 | `llm` ×4, `llm_repaired` ×1 |

**46 genuine model calls.** No provenance loss, no seasoning loss, no hard
failure on the LLM path.

Two notes, stated rather than buried:

* **X09 scores "correct" only because the scorer checks scope and seasoning, not
  measure identity.** The LLM path has the same measure-role defect as the
  deterministic path. The scorer is not evidence that X09 is right; §4.2 is
  evidence that it is wrong.
* **X23 answers on the LLM path** (`llm_repaired`) where the deterministic parser
  refuses — the LLM does not emit the bogus `collateral_geography: 'Back'`. The
  defect in §4.2 is deterministic-path-specific.

---

## 6. Independent truth reconciliation

Recomputed with pandas directly from the fixture; the production calculation was
never its own oracle.

| Population | Executed n | Truth n | Truth WA LTV | Truth balance |
|---|---|---|---|---|
| acquired + back (X01, X13) | 3,659 | 3,659 | 43.274184 | £555,602,755.20 |
| direct + front (X02, X14) | 927 | 927 | 35.637866 | £147,961,196.69 |
| direct + back (X03, X20, X21) | 6,199 | 6,199 | 44.276076 | £1,237,547,386.29 |
| acquired + front (X04) | 250 | 250 | 28.910082 | £23,774,920.03 |
| direct + age>85 (X08) | 55 | 55 | avg age 86.9273 | — |
| back book (X15 contribution) | 9,858 | 9,858 | 43.9657 | — |
| acquired (X16 contribution) | 3,909 | 3,909 | 42.6846 | — |

Denominators: X10 largest £684,845.61 = **0.1182%** of the acquired book;
X11 largest £841,638.96 = **0.0428%** of entire AuM. Both as returned.

**Population mismatches: 0. Unexplained variance: 0.** Where the semantic plan is
right, the arithmetic is right. Every defect in this report is a *routing or
role* defect, not a calculation defect.

---

## 7. Contradictions discovered

1. **ROUTED POPULATION LOSS (systemic).** P1J-1 governs a population via
   `spec.filters`; twelve of thirteen routes never read it; P0 has no facet kind
   for it. Lens-aware routes therefore drop the population **silently**, while
   `lens_aware=False` routes correctly disclose an un-narrowed scope. 5 demonstrated
   silent semantic errors.
2. **P1F vs P1E — EAD survives alone, vanishes in company.** An unavailable
   governed measure refuses as a single measure and is silently dropped as one of
   several.
3. **P1G vs P1I-A — scope words read as comparison cohorts.** A provenance word
   already consumed as *scope* still raises a cohort-comparison facet, producing a
   false refusal with a wrong stated reason on a correctly executed plan.
4. **P1I-A masking not extended to P1J-1 vocabulary.** Seasoning phrases are
   consumed by the geography resolver, creating a filter on a place called
   "Front"/"Back". Fails safe, but blocks two governed populations.

Items 2–4 are narrow and self-contained. **Item 1 is architectural** and is why
this phase reports FAIL rather than a list of small fixes.

---

## 8. Unsupported compositions (safe refusals — acceptable)

| ID | Composition | Outcome |
|---|---|---|
| X05 | two compound populations compared (direct-back vs acquired-front) | refuses; does **not** collapse to direct-vs-acquired or front-vs-back ✅ |
| X19 | period movement + seasoning | refuses ✅ — the brief's required behaviour |
| X24 | acquired back book by sourcing channel | refuses: `broker_channel` not in this dataset (DATA_UNAVAILABLE) ✅ |
| X25 | "Compare the back book with the acquired book" | refuses; does not present two axes as one dimension ✅ |

X19 refusing is the correct outcome — but note it is refused by the *period*
facet, not by any population guard. It is safe **by luck of a different guard**,
not by design, which is precisely the gap in §7.1.

---

## 9. Regression results

**No production code was changed in P1K.** `git diff HEAD` is empty; the working
tree is byte-identical to the pushed, verified P1J-1 HEAD. All P-gate suites
therefore stand as verified at P1J-1:

| Suite | Result |
|---|---|
| P0 fail-closed / cohort identity | green |
| P1E multi-measure | green |
| P1F exposure / B21 | green |
| P1G measure + cohort identity | green |
| P1I governed scope + sponsored ENTIRE_AUM | green |
| P1J-1 vintage & seasoning | green (53/53) |

---

## 10. 40-bank before → after

```
P1J-1  14/40   ->   P1K  14/40      changed: NONE
```

Re-run deterministically on the unchanged tree. As expected: P1K adds no
capability and changes no answer.

---

## 11. Full-suite result

The working tree is byte-identical to the pushed P1J-1 HEAD (`git diff HEAD`
empty), so the verified P1J-1 figure stands:

```
8645 passed, 30 skipped, 21 xfailed, 6 subtests passed  (0 failed)
```

A confirmatory re-run on the unchanged tree was started for the record and was
still in progress when this report was written. It is not claimed as a result
here — the figure above is the P1J-1 measurement, and the tree it measured is the
tree analysed in this report.

---

## 12. Remaining architectural risks

1. **Population loss on routed capabilities** — §7.1. The highest-severity open
   item. Any row-filter population (seasoning today; borrower age, LTV band,
   region tomorrow) is dropped by twelve routes with no disclosure.
2. **No population/filter facet.** P0's ledger covers grouping, measures,
   thresholds, ranking, contribution, period, share — but not "the population was
   narrowed and the route ignored it". Until that facet exists the guard is
   structurally blind to §7.1.
3. **Deterministic specs are never scope-role checked** (`reject_scope_role_filters`
   is LLM-path only), so a deterministic type predicate could coexist with the
   governed id scope.
4. **Caller filters override the governed lens** (`extra_filters` merge runs after
   `apply_scope`), inverting that function's documented precedence.
5. **Measure-set completeness does not enforce availability** — §7.2.
6. **Cohort detection is not scope-aware** — §7.3.

---

## 13. Recommended next phase

**Not P1J-2.** Adding projection would put a further capability behind the same
routing layer that currently drops populations, widening the exposure.

Recommended instead: **P1L — Governed population propagation**, narrowly scoped to
close §7.1 by making the population a first-class facet rather than an
unobserved dict entry. Two sub-decisions belong to you, not to me:

* whether a route that cannot honour a population should **refuse** or should
  **apply the filter** where the frame permits it;
* whether the population facet should cover every `spec.filters` key or only
  governed *population* concepts (scope, provenance, seasoning).

Items §7.2–§7.4 are small and generic and could ride along; §12.3 and §12.4 are
separate hygiene items worth their own decision.

I did **not** attempt any of this, per the instruction to stop and report on
discovering a genuine contradiction.

---

## Appendix — scope note

The 40-question bank is an adversarial evaluation set, not a requirements
document. Nothing in this report recommends building HPI stress, HHI or
correlation; the earlier P1J report's capability families are superseded on that
point by this instruction.

P1K CROSS-GATE CONSISTENCY: FAIL
