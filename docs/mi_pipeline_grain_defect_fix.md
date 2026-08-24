# Live Pipeline Evolution Grain Defect — Owner Fix

Bounded product-hardening task. Not a conversion. The C6 conversion is **not**
executed here.

## 1. The defect, reproduced

The five-week pipeline fixture (committed at `44d3f59`) publishes five distinct
weekly observations. The single-metric evolution route collapses all five onto
one x-axis point:

```
=== 1. THE UNDERLYING PIPELINE SERIES (from the fixture, via pipeline_evolution) ===
   week=2026-05-01  period=2026-05  count=6  amount=2300000.0
   week=2026-05-08  period=2026-05  count=7  amount=2800000.0
   week=2026-05-15  period=2026-05  count=8  amount=3600000.0
   week=2026-05-22  period=2026-05  count=8  amount=3600000.0
   week=2026-05-29  period=2026-05  count=8  amount=3600000.0
   source observations : 5
   distinct week values: 5
   distinct period vals: 1
```

Five governed weekly extracts, five distinct measured values, **one** period
label. A user asking for the weekly pipeline trend gets a chart titled
"by week" with a single point, and the route's own single-period warning
("Only one reporting period is available") fires on a five-week series.

## 2. The owner

One line owns it.

```
chat_routing.py:1025   period_field = "period"          # set unconditionally
chat_routing.py:1043   rows = [{"period": p.get(period_field), ...} for p in periods]
```

`period_field` is written once and read once. Every grain decision in the
single-metric branch passes through that one name, so the fix has exactly one
site. The defect is **not** in the producers: `pipeline_evolution` already
publishes the correct weekly identity, and the route already *labels* the axis
weekly (`f"{label} by {'week' if dataset == 'pipeline' else 'month'}"`). Only
the values were monthly. The label and the data disagreed, and the label was
right.

## 3. The rule, proven structurally — not from wording

Three producers can fill `evo["periods"]` for that reader. Their published key
sets decide the grain with no wording, no new parser and no dataset-name
allowlist:

| producer | keys published on a period | publishes `week`? |
|---|---|---|
| `evolution.funded_evolution` | `run_id`, `reporting_date`, `period`, `metrics`, `reconciliation`, `source_file` | **no** |
| `chat_routing._filtered_funded_evo` | `period`, `reporting_date`, `metrics`, `filteredRows` | **no** |
| `evolution.pipeline_evolution` | `extract_date`, `period`, `week`, `metrics`, `reconciliation`, `source_file` | **yes** |

The rule that falls out is the one the brief expected — FUNDED → `period`,
PIPELINE → `week` — but it is derived from the data the producer already
returns, not from the dataset name:

> **The observation identity is whatever the series publishes as its own grain.**
> A series that carries `week` is keyed on `week`; one that does not is keyed on
> `period`.

Structural, so a future producer that publishes weekly observations is keyed
correctly without being added to any list, and one that stops publishing `week`
degrades to the monthly key rather than to `None`.

The corroborating evidence that weekly is the *intended* pipeline grain, all
pre-existing: `pipeline_evolution`'s docstring ("across the governed UNIQUE
weekly extracts"), its `week` field, the route's own weekly axis title, and the
sibling funnel branch in the same function, which already keys its rows
`{"week": p.get("week"), ...}`. The route held two grains and only the
single-metric branch used the wrong one.

## 4. Pre-registered blast radius

**Authorised to move**

- Pipeline single-metric evolution questions may move from a one-point series
  (and its spurious single-period warning) to a five-point weekly series.
- The x-axis *values* on those charts/tables become `YYYY-MM-DD` extract dates.
- The route's trend sentence and `len(rows)` period count change accordingly,
  because they now count real observations.
- Answer state may move refusal/degenerate → delivery for weekly pipeline
  trend questions.

**Must not move**

- Funded evolution, filtered or unfiltered, at any grain.
- Dataset ownership (`workspace.resolve_dataset`) — untouched.
- Measures, measure resolution, and every value in `metrics`.
- Filtering and population semantics, including the `populationApplied` ledger.
- Funnel and stage selection, and the conversion mathematics.
- Route selection, route names, and every other route.
- The interpretation contract and the compositional plan layer.

**Not authorised**

- No new grain parser, no wording inspection, no route-name or dataset-name
  allowlist.
- No change to any producer.
- No teaching of the compositional path the same defect (this fix lands at the
  legacy owner, so C6 inherits the corrected rule rather than reproducing it).

## 5. The fix

One decision, moved and made conditional. `mi_agent_api/chat_routing.py`:

```python
-    period_field = "period"                      # before the producer is known
     ...
     periods = evo.get("periods", [])
     if not periods: ...
+    # The observation identity is whatever the series publishes as its own grain.
+    # ...
+    period_field = "week" if any("week" in p for p in periods) else "period"
```

The assignment moves below the producer branch, because the rule reads what the
producer returned. Nothing else changed: same reader at the next line, same row
key `"period"`, same `x_key`, same artifact shapes, same producers.

## 6. Measured blast radius

Fourteen probes through the live `/mi/query` path against the five-week fixture
and two governed month-end funded runs, captured before and after on identical
deterministic data, compared on every field of the envelope (`ok`, `route`,
`answer`, `warnings`, row count, x-values, values, `reconciliation`).

| | probe | before → after |
|---|---|---|
| **MOVED** | P2 `How has the pipeline changed over time?` | `ok=False`, 0 rows → `ok=True`, 5 weekly rows |
| **MOVED** | P4 `Show pipeline case count over time.` | `ok=False`, 0 rows → `ok=True`, 5 weekly rows |
| **MOVED** | P3 `Show the pipeline trend.` | 5 rows all `2026-05` → 5 distinct weeks |
| unchanged | P1, P5 (weekly-worded pipeline) | refused, both sides — see §7 |
| unchanged | F1–F4 funded evolution | `['2026-04','2026-05']`, identical values |
| unchanged | FF1–FF2 filtered funded | identical, including the disclosure text |
| unchanged | S1 pipeline-by-stage | identical |
| unchanged | S2–S3 funnel | identical |

Exactly the pre-registered set moved, and nothing else — no funded value, no
reconciliation block, no route selection, no disclosure sentence.

**The answer-state movement is the finding.** P2 and P4 did not merely mislabel
their x-axis; they were **refused outright**:

> I understood that you asked for a series over time, but that could not be
> applied to the calculation (a series over time — the answer that was produced
> carries no time axis; it reports a single position and cannot show movement).

That guard was right. A series whose five x-values are all `2026-05` genuinely
carries no time axis. The grain defect was upstream of it, and the correct
refusal downstream concealed the cause: the product looked like it lacked a
pipeline trend capability when it had one and was mislabelling it.

## 7. A SECOND owner of the same claim — found, measured, NOT absorbed

Three probes still refuse, before and after identically, with a different
sentence:

> I understood that you asked for week, but that could not be applied to the
> calculation (**week — this answer is reported at month level, not by week**).

That claim comes from a different module and a different owner:

```
mi_agent/execution_receipt.py:3530   _ROUTE_TIME_GRAIN = {"evolution": "month",
                                       "evolution_funnel": "month",
                                       "evolution_pipeline_stage": "month", ...}
```

a static route → grain map, on the stated premise that *"every one of these
reads the governed month-end funded snapshots, so every one publishes MONTHS"*.
`time_axis_disclosure` stamps `(asked, reported)` from it, and
`reconcile_facets` refuses on the mismatch.

The premise is false, and was false **before this task**:

- `evolution_funnel` has always keyed its rows `{"week": p.get("week"), ...}` —
  verified at `44d3f59`, before any change here.
- `evolution_pipeline_stage` keys on the day-level extract date.
- `evolution` on pipeline now joins them.

So this fix did not create the second owner's staleness; it made a third route
join two that were already misdescribed. The executable proof is probe **S2**
(`Show the KFI trend by week.`), refused identically before and after — a
weekly funnel answer told it is monthly, with no change of mine involved.

`migration_phase0/time_grain_claim_census.py` sizes it over the governed
corpora:

```
2. QUESTIONS NAMING A REPORTING UNIT: 53 of 882
     month 37   week 7   year 7   quarter 2
3. OF THOSE, NAMING A SUB-MONTH UNIT: 7
   ...on the PIPELINE dataset: 3
      applications over the last four weeks
      Show pipeline amount evolution by week.
      Show pipeline case count evolution by week.
   ...on a FUNDED dataset (correctly told months): 4
4. The static map is wrong for 3 of the 10 routes it covers.
```

**Three corpus questions**, and they include the exact phrasing the shipped e2e
test `test_pipeline_amount_evolution_by_week_e2e` is named after.

**This is not fixed here, deliberately.** It is a second owner, in a second
module, and closing it means threading a per-answer grain declaration through
the receipt plumbing — `time_axis_disclosure(unit, route)` takes no envelope, and
a route → grain map cannot express a route whose grain depends on its producer.
That is an unplanned dependency, and this programme's rule is not to absorb one
into a bounded fix and report the combined diff as the fix's cost. It is named,
sized and evidenced here instead.

**Recommended shape**, for whoever takes it: the route declares the grain it
published (it knows `period_field` exactly), the receipt reads that declaration
when present and falls back to `_ROUTE_TIME_GRAIN` — the same
execution-evidence-over-assertion pattern `declared_series_periods`,
`population_ledger` and `concentration_evidence` already use in that file. Not a
wider map, and not reading grain back out of the prose.

## 8. Cost

Canonical unit: raw added + raw deleted production diff lines.

| | raw lines |
|---|---|
| `mi_agent_api/chat_routing.py` | **10** (9 added, 1 deleted) |
| of which comment | 8 |
| of which executable | 2 (one moved, one made conditional) |

Classification: **product hardening**. Not shared plan-layer code, not
route-specific migration work, and it is excluded from every conversion cost
figure and from the C6 estimate.

Test and instrument code, reported separately and never counted as production
cost: `tests/test_pipeline_evolution_grain.py` (10 assertions),
`migration_phase0/time_grain_claim_census.py` (read-only).

---

# Part II — the pre-registration, amended by measurement

## 9. Why §7's decision was wrong

§7 named the second owner, sized it at three corpus questions, and left it
closed as an unplanned dependency. Re-measuring C6's owned surface then showed
something §6's fourteen probes had not covered:

```
=== owned questions whose GRADE changed ===
  REFUSED -> DELIVERED [evolution] Show pipeline amount evolution by month.
  REFUSED -> DELIVERED [evolution] Show pipeline case count evolution by month.
```

Those questions ask for **months**. Executed after the owner-1 fix:

```
ok        : True
answer    : Pipeline amount over 5 period(s): latest £3.6m (up over the window).
warnings  : []                                   <-- nothing disclosed
chart ttl : Pipeline amount by week
x values  : ['2026-05-01','2026-05-08','2026-05-15','2026-05-22','2026-05-29']
```

A question that asked for months receives weeks, and the governed disclosure
layer says nothing — because the receipt still believes the answer is monthly,
stamps `asked=month, reported=month`, and marks the facet APPLIED.

One stale claim, two opposite failures:

| question | stale claim | reality | before owner-1 | after owner-1 |
|---|---|---|---|---|
| pipeline `by week` | month | week | false **refusal** | false refusal (unchanged) |
| pipeline `by month` | month | week | refused (no time axis) | **undisclosed substitution** |

The first was pre-existing. **The second was introduced by the owner-1 fix.**
An undisclosed grain substitution is precisely the defect class this receipt
layer exists to prevent, so leaving owner 2 open is not a deferral — it is
shipping a new one. §7's reasoning about not absorbing unplanned dependencies
holds for work that is merely *adjacent*; it does not license leaving a
regression my own change created.

**Pre-registration amended, and the reason recorded rather than smoothed
over:** owner 2 is closed in this task.

## 10. Owner 2 — pre-registered blast

**Authorised to move**

- Pipeline `by week` questions: refusal → delivery (`evolution`, `evolution_funnel`).
- Pipeline `by month` questions: the undisclosed substitution introduced in §9 is
  removed — they return to the baseline's refusal, now for a stated reason
  ("reported at week level, not by month") instead of a wrong one.

**Must not move**

- Any funded question, at any grain — the funded routes still declare months.
- Any question naming no reporting unit: `time_axis_disclosure` returns None
  when `unit` is falsy, before any grain is read.
- The seven other routes in `_ROUTE_TIME_GRAIN`, which declare nothing and keep
  the static fallback.
- Measures, values, populations, datasets, route selection.

**Not authorised**

- No widening of `_ROUTE_TIME_GRAIN` into a bigger map of assertions.
- No reading grain back out of the answer prose.
- No new lexical vocabulary: `period_request.requested_unit` stays the one
  reader of the question.

## 11. Owner 2 — the fix

The route declares the grain it published; the receipt reads that declaration
and keeps the static map as the fallback.

```python
# mi_agent_api/chat_routing.py
def _declare_grain(envelope, grain):           # new, 3 executable lines
    envelope.setdefault("metadata", {})["seriesGrain"] = grain
    return envelope
#   funnel branch      -> _declare_grain(out, "week")
#   by-stage branch    -> _declare_grain(out, "week")
#   single-metric      -> _declare_grain(out, "week" if period_field == "week" else "month")

# mi_agent/execution_receipt.py
def declared_series_grain(envelope): ...       # new
    grain = declared_series_grain(envelope) or route_time_grain(route)

# mi_agent_api/mi_service.py
granularity = receipt_mod.granularity_facets(question, route, routed)
```

`routed` was already in scope at that call site — the population ledger reads
its metadata two lines below — so nothing new is threaded through the request.

All four quadrants, executed:

| question | before | after |
|---|---|---|
| pipeline `by week` | refused: *"reported at month level"* | **delivered**, 5 weekly points |
| pipeline `by month` | delivered weekly, **nothing disclosed** | refused: *"reported at week level, not by month"* |
| funded `by month` | delivered, 2 monthly points | **unchanged** |
| funded `by week` | refused: *"reported at month level"* | **unchanged** |

## 12. A pre-registration that was wrong, and how

§10 said *"any funded question, at any grain"* must not move. Two did:

```
DELIVERED -> REFUSED  [evolution_funnel|funded] completions by month
DELIVERED -> REFUSED  [evolution_funnel|funded] Show expected completions by month.
```

That clause was **mis-specified**, and the movement is correct. `resolve_dataset`
labels those questions FUNDED because their wording carries no pipeline term —
but the route that answers them is the pipeline **funnel**, and its series
carries only a `week` key:

```
funnel COMPLETED series keys: ['count', 'value', 'week']
funnel x-values            : ['2026-05-01' ... '2026-05-29']
```

The dataset LABEL and the producer's GRAIN are independent facts, and §10 wrote
a rule about one while meaning the other. Before this fix those two questions
delivered weekly numbers to a monthly question and disclosed nothing — the same
undisclosed substitution as the pipeline `by month` case, which is why they move
the same way. Pinned by
`test_a_monthly_question_on_the_weekly_funnel_is_told_so`.

Recorded as a defect in the pre-registration rather than reclassified after the
fact: the correct clause was *"no route whose producer publishes months may
change"*, and under that clause nothing moved that should not have.

## 13. C6 owned surface — re-measured on the five-week fixture

882 corpus questions executed through the live path. Coverage graded
DELIVERED / EMPTY / REFUSED, because `ok=True` is not "exercised".

| partition | at `44d3f59` | after both owner fixes |
|---|---|---|
| all owned | 34 owned, **18** delivered | 34 owned, **18** delivered |
| route=`evolution` | 30, 14 | 30, **16** |
| route=`evolution_funnel` | 2, 2 *(weekly under a monthly claim)* | 2, **0** *(refused honestly)* |
| route=`evolution_pipeline_stage` | 2, 2 | 2, 2 |
| **dataset=pipeline** | 7, **2** | 7, **4** |
| funnel-stage vocabulary | 2, 2 *(same false pass)* | 2, 0 |
| by-stage vocabulary | 3, 3 | 3, 3 |

The headline total is unchanged at 18, and saying only that would misrepresent
it: **two questions gained real delivery and two lost a false one.** The
pipeline partition — the one that blocked C6 — doubled from 2 to 4.

Still not delivered on the pipeline side, with reasons:

| question | why |
|---|---|
| `Show pipeline amount evolution by month.` | correct refusal — the pipeline is weekly |
| `Show pipeline case count evolution by month.` | as above |
| `Show pipeline by broker over time.` | `_route_evolution` returns `None` for a **filtered non-funded** trend and defers to the point-in-time path. Unrelated to grain; a genuine capability gap, unchanged by this task. |

## 14. Cost — both owners, canonical unit

Raw added + raw deleted production diff lines, attributed per file.

| file | added | deleted | raw |
|---|---|---|---|
| `mi_agent_api/chat_routing.py` | 43 | 9 | **52** |
| `mi_agent/execution_receipt.py` | 35 | 11 | **46** |
| `mi_agent_api/mi_service.py` | 3 | 1 | **4** |
| **total** | **81** | **21** | **102** |

| split | lines |
|---|---|
| executable | 58 (44 added, 14 deleted) |
| comment / blank | 44 |

| owner | raw |
|---|---|
| owner 1 — the series grain | **10** |
| owner 2 — the receipt's grain claim | **92** |

Owner 2 is nine times owner 1 because the re-indentation of three
`return _envelope(...)` calls into `out = _envelope(...); return
_declare_grain(out, …)` counts at both ends, as the canonical unit requires, and
because both stale comment blocks were rewritten rather than left asserting
something false.

**Classification: product hardening.** Not shared plan-layer code, not
route-specific migration work. **Excluded from every conversion cost figure and
from the C6 estimate** — it is not evidence for or against the cost-regime model.

Test and instrument code, reported apart and never counted as production cost:
`tests/test_pipeline_evolution_grain.py` (20 assertions),
`migration_phase0/route_ownership_evolution.py` and
`migration_phase0/time_grain_claim_census.py` (both read-only).

## 15. Mutation results — six, all non-vacuous

| mutation | fails |
|---|---|
| `period_field = "period"` (defect restored) | 5 — every pipeline assertion |
| `period_field = "week"` (over-reach) | 5 — every funded assertion |
| `period_field = "week" if dataset == "pipeline"` (allowlist) | 1 — **only** the structural test |
| receipt ignores the declaration | 4 — both weekly quadrants + the funnel |
| route declares `"week"` unconditionally | 5 — every funded assertion |
| route declares nothing | 2 — both pipeline quadrants |

The allowlist mutation is the important one: it passes every behavioural
assertion and is caught by exactly one test. Without that test the suite would
have accepted a dataset-name rule.

---

# Part III — C6 prerequisites, resumed

**C6 is not executed here.** This re-runs the dependency analysis on the
post-fix estate and stops.

## 16. Prerequisite 2 — funnel / stage selection, measured

`migration_phase0/funnel_stage_representation.py`, read-only, from code:

**What selects the sub-routes today**

```
funnel branch   : raw-question membership test against
                  {'kfi':'KFI','application':'APPLICATION','offer':'OFFER',
                   'completion':'COMPLETED','completed':'COMPLETED'}
by-stage branch : raw-question substring test against
                  ['by stage','stage migration','stage over time']
```

Both read the raw question **inside the handler**. Neither consults the
interpretation contract, and no plan primitive expresses either.

**A governed vocabulary already exists — and the route cannot reach all of it**

```
pipeline_prep._STAGE_CANON -> 24 spellings -> ['APPLICATION','COMPLETED','KFI','OFFER','WITHDRAWN']
_FUNNEL_KEYWORDS reaches    ->                ['APPLICATION','COMPLETED','KFI','OFFER']
governed stages the route's vocabulary CANNOT name -> ['WITHDRAWN']
```

A second, smaller defect, found by comparing the two vocabularies rather than by
reading either alone: the canonical stage set has five members and the route's
funnel vocabulary reaches four. **No question can ask for a withdrawn-case
trend**, though the prep layer classifies withdrawals and the five-week fixture
deliberately contains one (case 2004). Not fixed here — it is stage *vocabulary*,
which is exactly the dependency C6 must represent properly rather than extend in
place.

**Contract representation: still none**

```
DatasetClaim  DimensionClaim  FilterClaim  OperationClaim  PopulationClaim
SourceScopeClaim  SubjectClaim  TargetClaim  TimeClaim
claims that can carry a stage TODAY: NONE
```

Nine claims, none with a stage-bearing field. The C6 matrix's "no contract
representation at all" is confirmed independently of the C6 report.

**Corpus demand**: 24 of 882 questions name a funnel stage word; 13 name a
by-stage phrase.

## 17. Existing conversion capability — three owners, none to be rebuilt

| | what it is |
|---|---|
| `evolution._conversion_pct` | stage-to-stage percentage, used by the funnel |
| `forecast_extrapolation.kfi_conversion_model` | empirical 5-week completion-vs-KFI rate, shared with the forecast bridge |
| `chat_routing._route_conversion` | the shipped conversion route |

C6 **consumes** these. Adding a fourth conversion rate would repeat the exact
two-owners-that-agree-until-measured failure this programme has now closed three
times.

## 18. Four-part dependency matrix, re-run post-fix

| dependency | represented? | owner agreement? | consumed by plan? | delivered coverage? | status |
|---|---|---|---|---|---|
| `dataset` | yes | 0 disagreements / 34 | `dataset_of` exists | funded ✓ · **pipeline 4 of 7** (was 2) | **READY** |
| `subject` / measure | yes | 0 / 34 | `measure_request` exists | ✓ | READY |
| `time` span | yes | n/a — takes the whole series | `span_periods` exists | ✓ | READY |
| **`time.grain`** | yes | **now one owner** — route declares, receipt reads | not needed for dispatch | ✓ all four quadrants | **CLOSED THIS TASK** |
| `population` / scope | yes | route does not narrow | `_whole_dataset_step` provable | n/a | READY |
| `dimensions` | yes | n/a | `grouping_concepts` exists | n/a | not required |
| `filters` | yes | not measured | **no** — per-period filtering is route machinery | 1 of 2 | **UNPROVEN** |
| `operation` | yes | reads `spec.chart_type` | dispatch, as C1–C5 | ✓ | acknowledged exception |
| **funnel-stage selection** | **NO** | vocabulary reaches 4 of 5 governed stages | no | ✓ 2 of 2 *(now refused honestly, 0 delivered)* | **PREREQUISITE** |
| **by-stage selection** | **NO** | none | no | ✓ 3 of 3 | **PREREQUISITE** |

Two dependencies moved: `time.grain` from a silent two-owner disagreement to a
single declared owner, and `dataset` from blocking to ready on coverage.

**Two prerequisites remain, both the same one**: no claim on the contract can
carry a pipeline stage, so neither sub-route's selection can be expressed in a
plan. That is a contract task, not a conversion task.

## 19. Status

# C6 STILL PREREQUISITE-BLOCKED — one prerequisite, not four

Prerequisite 1 (pipeline data) is closed. The grain defect it exposed is closed
at both owners. `time.grain` and `dataset` now pass all four parts.

What remains is a single contract question: **how a pipeline stage is
represented as a governed claim** — covering all five canonical stages, not the
four the current vocabulary reaches. Until it is answered, funnel-stage and
by-stage selection stay raw-question reads inside the handler, and C6 would have
to migrate them by copying that read into the new path.

**Do not solve C6 by teaching the compositional path the existing defect.**

**Recommended next task:** design the governed stage claim as its own bounded
contract task — vocabulary from `pipeline_prep._STAGE_CANON` (the existing
single owner), all five stages, consuming the three existing conversion
capabilities rather than adding a fourth. Then re-run this matrix and
pre-register C6 with thresholds.

---

## 20. Regression by exact name

Two runs are reported, because the first method was invalid and saying only the
second would hide why.

**Method 1 — worktree baseline: INVALID.** A `git worktree` at `44d3f59`
skipped 739 tests where the working tree skips 35, so ~700 tests could not fail
in the baseline and any failure in the after-run looked introduced. It flagged
75. Re-running those 75 names in the working tree, same isolation, both sides:

```
PRE-FIX  failures: 3      POST-FIX failures: 3      INTRODUCED: (none)
   test_evidence_manifest.py::test_every_evidence_input_matches_the_manifest
   test_evidence_manifest.py::test_production_code_moving_on_is_reported_but_not_fatal
   test_p1e_measure_safety.py::test_a_routed_capability_may_satisfy_a_share_by_stating_one
```

All 75 were artefacts of the baseline environment. Recorded rather than deleted:
a baseline that skips what it should run manufactures regressions, and the
comparison looked rigorous while measuring nothing.

**Method 2 — same working tree, production files checked out at `44d3f59`, then
restored.** Same data, same config, **35 skipped on both sides**.

| | before | after |
|---|---|---|
| failed | 185 | **183** |
| passed | 10,293 | 10,313 *(+20 new assertions)* |
| errors | 28 | 28 |
| skipped | 35 | 35 |

```
=== INTRODUCED (fail after, not before) ===
   (none)
=== FIXED (fail before, not after) ===
   mi_agent_api/tests/test_chat_routing_e2e.py::test_pipeline_amount_evolution_by_week_e2e
   mi_agent_api/tests/test_chat_routing_e2e.py::test_kfi_trend_by_week_e2e
=== unchanged failures: 183 ===
```

**Zero introduced.** The two that now pass are the estate's own end-to-end tests
for exactly this behaviour — both were red on the branch before this task, which
is how long the defect had been shipping under a green-looking suite.
