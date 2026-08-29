# C5 — cost-regime re-baseline and candidate selection

**Inspection, modelling and threshold design only. No production code was
changed and Conversion 5 was not started.**

The question this document answers is not *"what should C5 cost based on
previous route numbers?"* but:

> **Exactly which generic semantic capabilities does C5 still need that the
> shared layer does not already provide, and what has each comparable
> capability actually cost before?**

---

## 1. Verified C1–C4 costs

Every total re-derived from `git diff --numstat -M`, production only, rather
than read from the reports:

```
C1  a56b7eb..f56bd35  = 383   ✓
C2  f56bd35..97557ae  = 282   ✓
C3  67981d5..5daf451  = 151   ✓
C4  39544bb..af6df52  = 219   ✓
```

| Conversion | Shared | Route-specific | Total |
|---|---|---|---|
| C1 `portfolio_summary` | 200 | 176 | **383** |
| C2 `period_movement` | 138 | 144 | **282** |
| C3 `geo_exposure` | 21 | 129 | **151** |
| C4 `funded_bridge` | 65 | 154 | **219** |

**Nothing has been reclassified.** These are the figures as published.

## 2. Field-level inventory of the three bridged axes

The C4 lesson made concrete: an axis being "bridged" does not mean every field
on it is consumed. "Read" below means read by the **shared layer** — either
`analytical_plan.py` or the shared time bridge `period_request.span_from_claim`.

### `source_scope` — 10 fields, 7 read

| field | read? | reader | routes needing it |
|---|---|---|---|
| `state` | ✔ | `_population_step` | all converted |
| `base_population` | ✔ | `_population_step` | all converted |
| `portfolio_ids` | ✔ | `_population_step`, `lens_filters` | all converted |
| `portfolio_label` | ✔ | `_label_for` | all converted |
| `provenance` | ✔ | `_population_step` | all converted |
| `raw_text` | ✔ | `_label_for` | all converted |
| `reason` | ✔ | blocked-step message | all converted |
| `scope` | ✘ | — | **none** — legacy string, superseded by `base_population` + `portfolio_ids` |
| `span` | ✘ | — | **none** — telemetry |
| `source` | ✘ | — | **none** — provenance of the decision, not the decision |

**Complete for execution.** The three unread fields are legacy or telemetry.

### `time` — 6 fields, 4 read

| field | read? | reader | routes needing it |
|---|---|---|---|
| `window_periods` | ✔ | `span_from_claim` (C2), `span_periods` | `period_movement` |
| `window_governed` | ✔ | `span_from_claim`, movement plan | `period_movement` |
| `trend_window` | ✔ | `span_from_claim` (the wording/label) | `period_movement` |
| `comparison_period` | ✔ | `analytical_plan.comparison_period` (C4) | `funded_bridge`, **`temporal_compare`** |
| `grain` | ✘ | — | grain-aware series (`evolution`, `temporal_compare`) |
| `requested_grain` | ✘ | — | same |

**`grain` and `requested_grain` are populated by `projection` and consumed by
NOTHING in production** — not by the plan layer and not by any route. They are
the clearest remaining instance of the C4 trap.

### `dimensions` — 7 fields, 5 read

| field | read? | reader | routes needing it |
|---|---|---|---|
| `candidate_concept` | ✔ | `grouping_concepts` (C4) | `funded_bridge` |
| `role` | ✔ | `grouping_concepts` | `funded_bridge` |
| `state`, `raw_text`, `reason` | ✔ | claim construction / refusal wording | — |
| `span`, `source` | ✘ | — | **none** — telemetry |

**Complete for execution.**

### The inventory's headline

> Of the three bridged axes, **only `time` carries genuinely unread fields that a
> route could need** (`grain`). `source_scope` and `dimensions` are complete —
> their unread fields are legacy or telemetry, and no route requires them.

## 3. The six unbridged axes

| axis | contract representation | semantic owner | plan representation needed | likely bridge shape | known consumers |
|---|---|---|---|---|---|
| **`dataset`** | `DatasetClaim(dataset, provenance)` | `chat_routing._dataset_for` | which governed dataset the series is built from | **like `comparison_period`** — a guarded field read | `evolution`, `temporal_compare` |
| **`subject`** | `SubjectClaim(candidate_concept)` ← `parser.metric` | `spec.metric` | *which measure* feeds `RESOLVE_MEASURE` | **like `grouping_concepts`** — governed concept, guarded | `evolution`, `temporal_compare` |
| **`operation`** | `OperationClaim(type, modifiers)` | `spec.aggregation` / ranking / compare | *how it is aggregated*; also the ranking intent | like `grouping_concepts`; **coarser than the raw `spec.aggregation`** the routes use today | `evolution`, `temporal_compare`, `period_change_analysis` |
| **`filters`** | `FilterClaim(operator, value, …)` | parser filters | — | **probably none** | already honoured **upstream**: `mi_service._population_frame` applies material predicates (P1L) so routes receive an already-narrowed frame |
| **`population`** | `PopulationClaim(concept)` | population parser | — | **probably none** | same P1L seam |
| **`target`** | `TargetClaim(value, target_source)` | stated/configured threshold | — | not analogous — forward-looking | **specialist only** (`forecast`, `scenario`) — outside the compositional core |

**Two of the six may need no bridge at all** (`filters`, `population` — handled
by the P1L propagation seam before a route sees its frame), and one is
specialist-only (`target`). The genuine remaining generic work for the
compositional core is **`dataset`, `subject`, `operation`** — plus `time.grain`.

## 4. Remaining compositional-core routes

The core is 7 (closure report §8, ranks 1–7). **4 are converted.** The specialist
six (`scenario`, `cohort_conversion`, `forecast_extrapolation`,
`cohort_progression`, `risk_limits`, `concentration_tests`) are deliberately
outside the contract and are **not** candidates.

| route | handler | new axes required | unread fields on bridged axes | other shared work | route-specific complexity |
|---|---|---|---|---|---|
| **`temporal_compare`** (`_route_compare`) | **63 lines** | `dataset`, `subject`, `operation` | — (`comparison_period` **already bridged in C4**) | none identified | small: reads `spec.metric`, `spec.aggregation`, `spec.compare_periods`, `_dataset_for`; delegates to `compare_mod` |
| `evolution` (`_route_evolution`) | 167 lines | `dataset`, `subject`, `operation` | possibly `grain` | shared with 3 routes — converting it entangles the measurement | medium: `resolve_metric_key(dataset, metric, aggregation)`, `is_count` |
| `period_change_analysis` (`period_change_route.py`) | **1,112-line module** | `operation` (**ranking not represented at all**) + others | `grain` likely | own workflow, own envelope, ranking subject + direction | **largest** — 4 local semantics recorded at closure |

Derived from AST inspection of every handler, not from route names:

```
_route_compare      63 lines   reads: dataset | spec.metric, spec.aggregation, spec.compare_periods
_route_evolution   167 lines   reads: dataset | spec.metric, spec.aggregation, resolve_metric_key
period_change_route 1112 lines  reads: lens, dataset, ranking, span, dimensions
```

**There is no reuse-only candidate left in the compositional core.** All three
remaining routes need at least one new axis, and the two cheapest need three.
That is the central finding of this inspection and it was not visible at axis
level.

## 5. The cost-regime model

Built from measured components, with each regime's evidence stated — and its
thinness stated with it.

### Regime A — reuse only
*No new axis, no unread field to connect.*

| anchor | measured |
|---|---|
| C3 `geo_exposure` | **21 shared** |

**One observation, and not a pure one.** C3's 21 lines were `scope_frame`, a
small *new* generic helper (the frame-input entry point). A conversion needing
literally nothing generic has never been measured, so **21 is a floor for "one
small helper", not for "nothing".**

### Regime B — existing-axis field extension
*No new axis; one or more unread contract fields connected.*

| anchor | measured |
|---|---|
| C4 `comparison_period` | **20 shared** |

**One observation.** Not a rate.

### Regime C — one new axis
*One generic axis bridge.*

| anchor | measured |
|---|---|
| C2 `span_from_claim` (`time`) | **24 shared** |
| C4 `grouping_concepts` + `ROLE_GROUPING` (`dimensions`) | **31 shared** |

**Two observations, differing by 29%.** Enough to say "tens of lines", not
enough to call a rate.

### Regime D — compound
*A new axis plus field extensions, or several generic additions at once.*

| anchor | measured |
|---|---|
| C4 `funded_bridge` | **65 shared** |

C4's 65 decomposes cleanly, which is what makes the model usable:

```
  31   dimensions bridge          (Regime C)
+ 20   comparison_period access   (Regime B)
+ 14   section documentation
= 65
```

**Regime D behaved additively in the one case measured.** That is the model's
central assumption and it rests on a single observation.

### What the model replaces

Budgeting from conversion number (C1 → C2 → C3 → …) predicted decay and was
wrong twice. Budgeting from **which generic capabilities the candidate still
needs** explains all four observations:

```
C1  200   built the plan layer itself
C2  138   generalised it + first axis bridge (24)
C3   21   one small helper, no axis
C4   65   one axis (31) + one field (20) + docs (14)
```

## 6. C5 candidate ranking

Preference order per the brief: reuse-only → single field extension → one new
axis → compound. **No candidate exists in the first three tiers.** All three
remaining core routes are Regime D, so the ranking is by *cleanliness of the
measurement*, not by tier.

| rank | route | why |
|---|---|---|
| **1** | **`temporal_compare`** | smallest handler by a wide margin (63 lines); needs 3 new axes but **already gets `comparison_period` free from C4**; delegates execution to an existing module; no unread-field surprises identified |
| 2 | `evolution` | same three axes but a 167-line handler, plus it is the module **shared with three routes** — converting it entangles shared-machinery cost with route cost, which is exactly the confound C3 was chosen to avoid |
| 3 | `period_change_analysis` | 1,112-line module; **ranking is not represented in the contract at all**, so it is a contract-extension task before it is a conversion |

### Recommended C5: `temporal_compare`

**What it would test:** whether a compound conversion costs **the sum of its
parts** (the additive assumption C4 produced from one observation) — with three
accessors instead of one axis plus one field. If additivity holds, the regime
model becomes predictive and C5 is the first conversion budgeted from field-level
evidence rather than optimism. If C5 costs materially less, the parts share more
than the model assumes; materially more, and compound work has a super-additive
cost the model must carry.

**Next best:** `evolution`.

## 7. Derived C5 thresholds — for pre-registration before C5 begins

**Not committed here** — C5 is not authorised in this task. These are the numbers
that should be committed as stop conditions before any C5 production change.

### Shared: **≤ 75**

Decomposed from measured analogues, not chosen to fit:

| component | analogue | measured | estimate |
|---|---|---|---|
| `dataset` accessor | `comparison_period` (guarded field read) | 20 | **20** |
| measure accessor (`subject` + `operation`) | `grouping_concepts` (governed concept, guarded) | 31 for one axis | **28** for both as one accessor |
| section documentation | every prior conversion's banner | 14 | **14** |
| | | | **62 predicted** |
| justified margin | one unforeseen guard or state case | | **+13** |
| | | | **≤ 75 threshold** |

**This is higher than C4's 65, and deliberately so.** The model says shared cost
tracks *how many generic capabilities the candidate needs*, and `temporal_compare`
needs three where `funded_bridge` needed two. A threshold below 65 would be
budgeting from conversion number again — the exact error this re-baseline exists
to correct.

### Route-specific: **90–150**

| basis | |
|---|---|
| observed | C1 176, C2 144, C3 129, C4 154 — range 129–176 |
| handler size | `_route_compare` **63 lines** — the smallest yet converted (geo 94 → 129; bridge 111 → 154) |
| plan complexity | a period pair + one measure; no waterfall residual, no ranking |
| duplicate-owner removal | none identified — this route does not call `_resolve_lens` or `resolve_lens_with_default` |

A handler a third smaller than `geo_exposure`'s should not cost more than
`geo_exposure`'s 129, but the floor of 90 matters as much: **far below 90 would
suggest a shallower conversion than C1–C4 received**, making the comparison
meaningless in the other direction.

### Total: **≤ 225**

Derived (75 + 150). Predicted landing zone: **~62 shared + ~115 route-specific
≈ 180**.

## 8. Pre-defined C5 verdict rules

Written before any C5 code exists.

**REGIME MODEL SUPPORTED** — shared ≤ 75; route-specific 90–150; total ≤ 225;
exactly the three identified accessors added (`dataset`, measure) and no other
generic abstraction; no new primitive; economics, payload and receipt equivalent;
regressions clean.
*Means:* cost is predictable from field-level dependency evidence, and the
additive assumption for compound work holds on a second observation.

**REGIME MODEL INCOMPLETE** — technically clean, but the conversion requires
generic work **not identified in §4** (an unread field, an axis, or an executor
or receipt change this inspection missed).
*Means:* the field-level inventory is still not deep enough. Record what was
missed and extend the inventory before C6. **Do not proceed automatically.**

**MIGRATION ECONOMICS FALSIFIED** — shared materially exceeds 75 with no
newly-discovered capability to explain it; or total materially exceeds 225; or
one conversion forces structural redesign of the plan layer, the executor
contract or the receipt.
*Means:* even a candidate characterised at field level cannot be budgeted, and
the remaining migration should be re-scoped rather than continued.

## 9. Migration progress — beyond "4 of 7"

**Routes converted: 4 of 7 compositional-core (57%)** — but that number alone is
the least informative of the three measures below.

### Semantic infrastructure coverage

```
contract axes bridged            : 3 of 9   (source_scope, time, dimensions)
axes needing no bridge           : 2 of 9   (filters, population — honoured upstream by P1L)
axes that are specialist-only    : 1 of 9   (target)
axes still to bridge for the core: 3 of 9   (dataset, subject, operation)
```

**On required-for-the-core axes, coverage is 3 of 6 (50%)** — and the three
remaining are all needed by the same two routes.

### Unread fields on bridged axes

**One** field pair remains that any core route could need: `time.grain` /
`requested_grain`, currently consumed by nothing in production. Every other
unread field is legacy or telemetry.

### Reuse-heavy versus bridge-heavy

| | routes |
|---|---|
| reuse-heavy (Regime A/B) | **none remaining in the core** |
| bridge-heavy (Regime C/D) | `temporal_compare`, `evolution`, `period_change_analysis` — all three |

### Is the remaining migration bounded?

**Yes, and now countably so** — which is a stronger statement than the programme
could make before this inspection:

* **3 routes** remain in the core;
* **3 axes** remain to bridge, and `evolution` and `temporal_compare` need *the
  same three*, so the second of the two should be close to Regime A;
* **1 unread field pair** (`grain`) is the only known field-level surprise left;
* `period_change_analysis` is the one genuine unknown — ranking is absent from
  the contract, so it is a contract-extension task before it is a conversion.

The bound is therefore: **three accessors plus one field, shared across three
routes** — not an open-ended sequence.

## 10. A1–A5

| | condition | status |
|---|---|---|
| **A1** | cost explosion | **NOT FIRED**, and now **supplemented** by the regime model. A1's own trigger is `2 × m` over the first three conversions: median total *m* = 282 across C1–C4 (151, 219, 282, 383), so the trigger is **564** and no conversion has approached it. A1 answers "has cost exploded"; the regime model answers "what should this conversion cost, and why" — they are complementary and the model does not replace the condition. |
| **A2** | failure to reconcile | **NOT FIRED.** C4: 27 delivering renders, 0 economic differences; 36 envelope pairs, 6,269 leaf fields, 0 differences. The £0.005 tolerance has never been approached. |
| **A3** | governance cannot generalise | **NOT FIRED — strengthened.** C4's `grouping_concepts` obeys the contract's role, and the generic `grouping_proven` began certifying `funded_bridge` from execution evidence with no route allowlist. |
| **A4** | interpretation ownership cannot be made singular | **NOT FIRED.** No plan builder takes a question; enforced over the AST in all four conversions' tests. |
| **A5** | unattributable regression | **NOT FIRED.** C4: introduced failing names 0, attributed against `39544bb` in the same environment, set difference empty in both directions; five A5 surfaces byte-identical; silent drops 0. |

**The historical threshold breaches stand and are not erased:** C1 `STOP — COST
ASSUMPTION BREACHED` (383 vs 240), C2 `STOP — COST ASSUMPTION BREACHED` (282 vs
240), C4 `CONVERGENCE NOT PROVEN` (65 shared vs 45; 219 total vs 215). Those were
breaches of **per-conversion pre-registered caps**, which are a different and
also-real mechanism from A1. Three of four conversions missed their registered
budget — that is the evidence this re-baseline exists to act on, and the regime
model is the response to it rather than a way of retiring it.

## 11. Recommended next task

> **Pre-register Conversion 5 on `temporal_compare` with the §7 thresholds
> (shared ≤ 75, route-specific 90–150, total ≤ 225) and the §8 verdict rules,
> committed before any production change — then run it.**

Three things to carry in, all from this inspection:

1. **Confirm the three accessors are still exactly three** before implementing.
   C4's lesson was that an axis label hides field-level work; this inspection
   went to field level, but `_route_compare` delegates to `compare_mod`, and
   that module's own inputs were **not** inspected here. Enumerate them first.
2. **`time.grain` is unread by anything.** If `temporal_compare` turns out to
   need it, that is a fourth accessor and the budget must be restated **before**
   implementation, not after.
3. **Do not convert `evolution` in the same task.** It shares the execution
   module, and the whole value of C5 as a measurement is that its cost is
   attributable to one route.
