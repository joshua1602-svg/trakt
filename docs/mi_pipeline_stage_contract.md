# Closing the last C6 prerequisite: a governed Pipeline Stage claim

Base `9ae4535`. **C6 not executed.** No route switched, no conversion
mathematics touched.

---

## 1. What was missing, and what was not

The prerequisite was stated as *"the interpretation contract cannot structurally
express Pipeline Stage semantics"*. Measured, the gap was narrower and more
awkward than that: **the concept was already governed everywhere except in a
question.**

| where | what it already said |
|---|---|
| `config/mi/pipeline_field_contract.yaml` | `pipeline_stage: role: dimension`, `semantic_registry_field: pipeline_stage`, and the five canonical stages named in its own description |
| `config/mi/stratification_catalogue.yaml` | `pipeline_stage: bucket: categorical`, `applies_to_states: [total_pipeline, total_forecast_funded]` |
| `mi_agent_api/pipeline_prep.py` | `_STAGE_CANON`, 24 spellings → 5 canonical stages; `_STAGE_BUCKET`, the funnel order |

So there was no vocabulary to invent and no dimension to register. What did not
exist was any path from a **sentence** to that governed field — measured over 22
targeted probes:

```
CONTRACT CAN NAME A STAGE AT ALL: NO
contract carries a stage DIMENSION on any probe: 0
contract carries a stage FILTER VALUE on any probe: 0
```

which is exactly why `_route_evolution` re-read the raw question.

## 2. The three duplicate owners, executed

| raw condition | dataset | function called | stage behaviour |
|---|---|---|---|
| `any(kw in q for kw in _FUNNEL_KEYWORDS)` | **not checked** | `evolution.pipeline_funnel_evolution` | one stage's weekly flow + stock |
| `"by stage" / "stage over time" / "stage migration"` in `q`, and dataset is pipeline | pipeline | `evolution.pipeline_evolution().byStage` | amount per stage per week |
| neither | either | `pipeline_evolution` / `funded_evolution` | ordinary series |

Executed rather than read, the shipped selector is **both over- and
under-inclusive**:

- `"Show the offer price distribution."` → **funnel, stage OFFER**. Substring
  matching; an offer *price* is not a pipeline stage.
- `"Show the completion trend by week."` → **funnel**, on a **funded** dataset.
  The funnel branch runs before any dataset check, so a Pipeline-only analysis
  answers a funded question.
- `"Show pipeline stage balances over time."`, `"What is the pipeline stage
  distribution?"`, `"Which stage has the largest pipeline?"` → **ordinary**.
  The three-phrase list misses them.
- Nothing reaches **WITHDRAWN** at all.

## 3. Canonical vocabulary, and the five stages

`canonical_pipeline_stages()` reads `_STAGE_BUCKET` and returns, in funnel order:

```
KFI  →  APPLICATION  →  OFFER  →  COMPLETED
                                  WITHDRAWN
```

`COMPLETED` and `WITHDRAWN` are terminal — `pipeline_prep._OPEN_PIPELINE_STAGES`
is `{KFI, APPLICATION, OFFER}`. Governed aliases (24 spellings) normalise to
those five. The route's `_FUNNEL_KEYWORDS` reaches **four**; the missing one is
`WITHDRAWN`, and the fix for that is not another keyword.

## 4. The representation — composition, not a new claim

Both sub-routes turn out to be compositions of things the contract already has,
so **no new claim type and no funnel-shaped field were added**:

| shipped sub-route | compositional meaning |
|---|---|
| `evolution_funnel` | `dataset=PIPELINE` + `dimension=pipeline_stage (role=filter)` + that stage's value + `operation=EVOLUTION` |
| `evolution_pipeline_stage` | `dataset=PIPELINE` + `dimension=pipeline_stage (role=grouping)` + `operation=EVOLUTION` |

**FUNNEL is not a separate analysis type.** The shipped funnel branch returns one
stage's weekly flow and stock — stage-filtered evolution. It surfaces no
conversion figure at all (the branch reads `label`, `latestFlowValue`,
`fiveWeekAvgFlowValue`, `trend`, `latestStockValue` and nothing else), so C6
needs no funnel concept and no conversion arithmetic to reproduce it.

One reader, in the module that already owns every question vocabulary:

```python
question_interpretation/lexical.py
    PIPELINE_STAGE_FIELD
    pipeline_stage_vocabulary()      # derived from _STAGE_CANON
    canonical_pipeline_stages()      # derived from _STAGE_BUCKET
    pipeline_stage_request(question) -> (canonical_stage | None, names_axis)
```

consumed once by `projection.py` into the **existing** `DimensionClaim` and
`FilterClaim`.

### Two rules that keep a data map from becoming a question vocabulary

Both are derived, not hand-listed, so they cannot drift from the map:

1. **View-name collision.** `_STAGE_CANON` maps `"funded" → COMPLETED` — correct
   for a tape cell, catastrophic for a sentence, where *funded* names the
   governed dataset. Without this, `"Show funded balance evolution by month"` —
   the most ordinary question in the corpus — acquires a COMPLETED stage.
2. **Fragment rule.** A spelling that is a prefix of a longer spelling for the
   same stage, and is not the canonical token, is a word fragment: `complete`
   (prefix of `completed`/`completion`) and `app` (prefix of `application`).
   `complete` matched five corpus questions about **data completeness** —
   *"How complete is interest rate?"*. The canonical token itself always
   survives, so `offer` is kept despite prefixing `offer issued`.

And one asymmetry, stated rather than hidden: a canonical stage **name** is its
own evidence, but the bare word *stage* is ordinary English, so it names the axis
only under an axis marker (`by stage`) or where the governed dataset owner has
already decided the question is pipeline. That is what keeps *"What stage is the
securitisation at?"* out.

## 5. Blast radius — executed, not inferred

All 882 corpus questions run end to end through `/mi/query`, before and after,
compared on `ok`, `route`, `dataset`, answer text, warnings and row count:

```
=== EXECUTED BLAST: 0 of 882 corpus questions moved ===
```

**Zero.** The change is purely additive: it gives the contract something to say,
and no consumer acts on it yet. That is the correct shape for a prerequisite —
the behaviour change belongs to C6, under C6's own equivalence proof.

## 6. Owner agreement — contract vs shipped

904 questions (882 corpus + 22 probes):

```
agree: 894    disagree: 10
```

Every one of the ten is the contract reaching something shipped **cannot**:

| | questions | why |
|---|---|---|
| WITHDRAWN | 3 | the stage `_FUNNEL_KEYWORDS` has no word for |
| stage axis | 7 | `stage balances`, `stage distribution`, `Which stage has the largest pipeline?`, `forecast balance by stage`, `expected funded by stage` |

**There is no case where shipped names a stage the contract misses.** That is the
direction that matters: a replacement may over-reach into a governed concept, it
may not under-reach.

## 7. Five-stage coverage — semantic and execution, kept apart

The fixture carries observations of every canonical stage (KFI 12, APPLICATION 9,
OFFER 10, COMPLETED 4, WITHDRAWN 2). Coverage is **not** the same thing:

| stage | semantically representable | stage-scoped delivered answer |
|---|---|---|
| KFI | yes | yes — `evolution_funnel`, 5 rows |
| APPLICATION | yes | yes — `evolution_funnel`, 5 rows |
| OFFER | yes | yes — `evolution_funnel`, 5 rows |
| COMPLETED | yes | yes — `evolution_funnel`, 5 rows |
| **WITHDRAWN** | **yes** | **no** — falls through to ordinary whole-book evolution |

WITHDRAWN is representation-only, and deliberately so. The shipped calculation
does not support it:

```
funnel series keys: ['APPLICATION', 'COMPLETED', 'KFI', 'OFFER']
WITHDRAWN points  : 0
```

`pipeline_funnel_evolution` builds no WITHDRAWN series. Wiring the claim through
would therefore **expand user-facing capability beyond the shipped calculation**,
which this task is explicitly not permitted to do. The gap is now named and
structural rather than invisible.

## 8. No second conversion calculator

The existing deterministic capability, unchanged:

| owner | what it computes |
|---|---|
| `evolution._conversion_pct` | forward conversion vs the KFI denominator, with `_lagged_value` shifting the denominator by the KFI→completion lag |
| `forecast_extrapolation.kfi_conversion_model` | empirical 5-week completion-vs-KFI rate, shared with the forecast bridge |
| `chat_routing._route_conversion` | the shipped cumulative cohort conversion route |

The diff introduces **no arithmetic of any kind** — no division, no percentage,
no maturity or cohort calculation. Grepped over every added line; the only
matches are the words *conversion* and *stage* inside comments.

## 9. Duplicate-owner removal — NOT done, and why

`_FUNNEL_KEYWORDS` and the three by-stage phrases **remain in place.** Of the four
conditions required before deleting them:

1. every shipped Stage/Funnel path receives the structural claim — **holds** (zero under-reach)
2. the claim agrees with shipped behaviour — **holds on the shipped-owned surface**
3. no consumer relies on the old disagreement — **fails**: the route's sub-route
   selection *is* that consumer, and nothing else selects it
4. all five canonical stages remain representable — **holds**

Condition 3 cannot be satisfied by this task by construction: the raw read can
only be removed at the moment the route starts consuming the claim, and that
moment is the C6 switch. Removing it now would leave nothing selecting the
sub-routes; wiring it now would move the ten disagreements — including
WITHDRAWN, which the calculation cannot serve — and that is a behaviour change
requiring C6's equivalence proof.

Per the stated rule, the old read stays.

## 10. Four-part C6 dependency matrix

| dependency | represented | owner agreement | plan consumable | delivered coverage | status |
|---|---|---|---|---|---|
| dataset | yes | 0 disagreements | `dataset_of` | funded ✓ · pipeline ✓ | **READY** |
| measure | yes | 0 disagreements | `measure_request` | ✓ | **READY** |
| historical series | yes | n/a | `span_periods` | ✓ 5 weekly / 2 monthly | **READY** |
| time / grain | yes | declaration now wins | n/a | ✓ | **READY** |
| population | yes | route does not narrow | `_whole_dataset_step` | n/a | **READY** |
| ordinary evolution | yes | ✓ | dispatch | ✓ | **READY** |
| **Pipeline Stage** | **yes** (new) | **894/904, superset only** | **no** | 4 of 5 stage-scoped | **REPRESENTED, NOT CONSUMED** |
| **Stage evolution** | **yes** (new) | ✓ | **no** | ✓ 2 of 2 | **REPRESENTED, NOT CONSUMED** |
| **Funnel** | **yes** — not a concept, a composition | ✓ | **no** | ✓ 4 stages | **REPRESENTED, NOT CONSUMED** |
| **filters** | yes (`FilterClaim`) | **not proven** | **NO** | **no delivered case** | **PREREQUISITE** |

## 11. Filters — the blocker this task did not create

`_route_evolution` interprets filters itself:

```
filtered = bool(getattr(spec, "filters", None))
evo = _filtered_funded_evo(output_root, client_id, run_id, spec, semantics, metric_key)
      -> _apply_filters(df, spec, semantics, [])
```

`analytical_plan.lens_filters` exists and is what the four converted routes use.
`evolution` does not call it. And no filtered evolution question delivers:

```
Show funded balance evolution by month for London.
   ok=False  rows=0
   populationApplied={'applied': ['geographic_region_obligor (applied within each period)'], 'rowsAfter': 22}
```

The route applies the filter, declares it, and the answer is still refused — by
the geographic-scope owner, not the population ledger. Identical before and after
this change, and identical on the corpus (`Show monthly balance evolution by
region.` is `ok=False` on both sides). This is the same `filters` row the
post-C5 matrix already marked UNPROVEN; nothing here improved or worsened it.

## 12. Cost

Canonical unit: raw added + raw deleted production lines.

| prerequisite | raw production lines |
|---|---|
| Pipeline Stage contract — `lexical.py` reader + vocabulary rules | 139 |
| projection wiring — two composed claims | 32 |
| Funnel structural representation | **0** (it is a composition, not a concept) |
| duplicate-owner removal | **0** (not performed — §9) |
| other shared work | 0 |
| **total prerequisite** | **171** (171 added, 0 deleted) |

Of the 171: 56 comment/docstring, 23 blank, 92 executable. Reported in the raw
unit throughout; net-executable counting is not used.

## 13. C6 re-estimate

`_route_evolution` is **178 lines** — 11 more than when the C5 regime model
ranked it, because of the grain fix. The estimate does not improve with time:

| item | basis | raw lines |
|---|---|---|
| plan builder | three sub-paths rather than C5's two | 60–75 |
| executor | C5's 27, plus stage grouping and stage filtering | 35–50 |
| plan readers | stage value + stage axis, on top of C5's accessors | 12–18 |
| switch + registration | C5's 42 | 40–50 |
| **route-specific** | | **≈ 150–195** |
| shared accessors | `dataset_of`, `measure_request` exist; stage claim now exists | **0** |
| **prerequisite already spent** | this task | **171** |
| **filter prerequisite** | no design exists | **unknown** |

**Normalised burden remains not computable**, for the same reason as before: one
term has no design. Producing a number would repeat the C5 error, which missed by
89% on work that at least had a known shape.

## 14. Thresholds — not pre-registered

Part I permits threshold derivation *only if the dependency matrix is complete*.
It is not: `filters` fails on two of four columns, and three stage rows are
represented but not yet consumable by any plan. `docs/mi_conversion6_stop_conditions.md`
is **deliberately not created** — thresholds derived against an incomplete model
are guesses wearing the costume of a pre-registration.

## 15. Regression by exact name

Baseline and after run **in the same tree** (the change stashed, not a worktree —
a worktree baseline showed 687 skips against 7 and was not like-for-like), over
the 57 suites touching projection, lexical, dimensions, filters, stage or funnel:

```
baseline: 116 failed, 6607 passed, 7 skipped, 16 xfailed, 17 errors
after   : 117 failed, 6635 passed, 7 skipped, 16 xfailed, 17 errors

INTRODUCED: tests/test_dataset_ownership.py::test_no_production_module_re_decides_the_dataset_from_raw_text
FIXED     : none
UNCHANGED : 116 pre-existing failures
```

### The one introduced failure, and what was done about it

That guard asserts the disclaim-aware reading lives in exactly one production
module, **keyed on `undisclaimed_mention` call sites**. `pipeline_stage_request`
calls it — to read a STAGE, not a dataset.

Its own docstring predicted this: *"An earlier cut keyed on the artefact tuple
instead and collided with pipeline STAGE tuples that name the same words for an
unrelated purpose — the words are shared, the disclaim-aware reading of them is
not."* The invariant is about the DATASET reading; the key had stopped isolating
it, because a general helper acquired a second legitimate user.

The guard was re-keyed to a **named** allowance (`workspace.py` the dataset
owner, `lexical.py` the stage owner) rather than a loosened pattern, and a
companion test now proves why the second name is safe: the stage reader returns a
canonical stage or None — never a view — and CONSUMES `resolve_dataset` where it
needs the dataset rather than deriving one.

Mutation-checked, because re-keying a guard I wrote is exactly the move that can
quietly turn it into a pass-anything test: inserting a third `undisclaimed_mention`
caller into `chat_routing.py` that returns a dataset still fails it.

Not a silent accommodation — the invariant is unchanged and still enforced; only
what identifies a dataset reading was corrected.

## 16. Status

# STOP — C6 FILTER PREREQUISITE

The prerequisite this task was set — a governed, structural Pipeline Stage
representation — **is closed**: five canonical stages, derived vocabulary, no new
claim type, no funnel field, no second conversion calculator, 894/904 owner
agreement with superset-only disagreement, and zero executed blast across 882
questions.

C6 is still not ready, for a reason this task did not introduce and was asked to
check: **`evolution` interprets `spec.filters` inside the route, no plan primitive
consumes a filter claim, and no filtered evolution question delivers an answer.**

**Recommended next task:** close the filter prerequisite — make the governed
`FilterClaim` the authoritative representation for evolution, prove per-period
application through `analytical_plan.lens_filters` rather than route machinery,
and establish at least one delivered filtered case. That is the last row of the
matrix, and it is the same row the post-C5 correction already flagged.
