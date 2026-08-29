# Post-C5 methodology correction, and C6 pre-registration assessment

Base `ee55f26` → `3501bec`. **C6 conversion not started.**

C5 stands as recorded: **C5 CONVERSION WITHIN THRESHOLD**, independently
assured as **C5 SUBSTANTIALLY CORRECT — QUALIFICATIONS**. No C1–C5 production
diff was altered and no historical cost was reclassified. The
`REGIME MODEL PARTIALLY SUPPORTED` wording must now be read conservatively —
§2 below shows why its supporting arithmetic did not hold.

---

## 1. The canonical accounting unit — verified, not asserted

Recomputed from git, production files only:

```
C3  raw production diff = 151    C3 report claims 151  ✓
C4  raw production diff = 219    C4 report claims 219  ✓
C5  raw production diff = 198    C5 report claims 198  ✓
```

Three of three land exactly. The programme has always used raw lines; the unit
was never the problem, only its consistency.

> **MIGRATION COST UNIT — raw added + raw deleted production diff lines,
> attributed hunk by hunk.**
>
> * **Classification** is the re-baseline's rule, unchanged: *is this code
>   reachable from more than one route at the end of the conversion?* →
>   shared / route-specific / product hardening / cleanup.
> * **Comments, docstrings and banners count.** A banner is classified by the
>   section it heads — an axis banner is shared, a route banner route-specific
>   (C4's precedent: shared banner 14, route banner 4).
> * **Relocated code counts at both ends** — added in the new site, deleted in
>   the old. A move is not free, and calling it "net zero" is what hid the
>   dataset-ownership work from the C5 model.
> * **Prerequisite work is normalised in this same unit.** A prerequisite is
>   counted when the candidate would have had to build it itself.
> * **Net-executable counting is prohibited** for any threshold or normalised
>   figure. It may appear only as commentary, labelled.
> * Tests, instruments and documents outside `docs/` code are never production
>   cost.

## 2. C5 normalised burden, corrected

The reported `50 + 19 = 69` added a raw figure to an executable one. In one
unit:

| | |
|---|---|
| C5 measured shared (independent recount) | **52** |
| `comparison_period` prerequisite (`ad67ee9`), raw | **65** |
| **normalised C5 shared burden** | **117** |
| pre-registered prediction | 62 |
| **prediction miss** | **+55 (+89%)** |
| headroom on the conversion figure | +23 |
| headroom on the normalised figure | **−42** |

The prerequisite is genuinely C5's: with the structural field removed,
`build_temporal_compare_plan` **blocks** — *"time.comparison_periods names 0
period(s), not 2"* — so C5 would have had to build it.

Reported separately and **not** in the normalised burden: the dataset-ownership
remediation, **303 raw lines**, classified *product hardening* because it fixed
a live defect (the same sentence meaning different things on different tabs)
that was worth fixing without any migration. Recorded so the reader can put it
back if they disagree with that classification.

## 3. C5 assurance surface — gap closed, and a real hole found

| | before | after |
|---|---|---|
| cases | 28 | **32** |
| owned | 21 | **24** |
| renders | 42 | **48** |
| delivered | 16 | **18** |
| envelope leaves | 4,464 | **5,126** |
| differences | 0 | **0** |

Added: `O1` and `O2`, the two runtime-owned questions the audit found missing;
`V2 "Compare this month and last month funded balance."`, **delivered**.

`V2` matters more than the count suggests. The relative-period branch of
`_match_period` — `latest` and the `_RELATIVE_PRIOR` vocabulary — was exercised
by **no delivered case at all**. A whole resolution path sat behind a green
equivalence. `V2` runs it: `latest` → 2026-06, `last month` → 2026-05, with
real balances.

One candidate was declared owned and is **not**: *"Compare the latest funded
balance with the prior period."* is claimed by `period_change_analysis` at
priority 85. The instrument caught the declaration, which is what it is for.

## 4. The population invariant, and why the first version was too weak

The audit planted a builder that wrote the literal `"whole_dataset"`, read
nothing, and **passed**. A magic string is not a governed fact.

There are now exactly two governed constructors, and every plan builder must
reach one:

```
_population_step(scope)          the route narrows; the scope states decide
_whole_dataset_step(route, ds)   the route does not narrow, AND PROVES IT
                                 against Recogniser.lens_aware
```

`lens_aware` is not a new declaration — it is the single place the platform
already records which routes narrow, and the product's own *"Scope not
narrowed"* disclosure already depends on it. A lens-aware route asking for the
whole-dataset step gets a **BLOCKED** step naming the contradiction; so does a
route the registry does not know, because being unable to prove a claim is not
the same as the claim being false.

### Mutation results

The first mutation run was **invalid and is recorded as such**: `git checkout
HEAD --` reverted the uncommitted implementation, so three cases failed for the
wrong reason. Re-run against a committed baseline, with an assertion that each
mutation actually applied:

| case | expected | result |
|---|---|---|
| valid scoped builder | pass | **pass** |
| valid whole-dataset builder | pass | **pass** |
| magic-string claim, no governed constructor | fail | **fail** |
| whole-dataset claim that reads a scope field | fail | **fail** |
| duplicate local population decision | fail | **fail** |
| lens-aware route claiming the constructor | plan blocks | **blocks** |

## 5. Two C5 assumptions about C6, falsified by code

**`time.grain` is not a dependency of `evolution`.** `_route_evolution` calls
`funded_evolution(root, client_id, run_id)` and `pipeline_evolution(...)` with
**no grain argument**, and `funded_evolution` has no grain parameter. The grain
is intrinsic to the tape: funded evolution is monthly governed runs, pipeline
evolution weekly extracts. There is no second owner to disagree with, so
**`C6 PREREQUISITE — TIME GRAIN OWNER DISAGREEMENT` does not fire.**

**Ranking is represented.** `OperationClaim.type` includes `RANKING`,
projection sets it from the ranking facet or `spec.ranking_mode`, and **94 of
882** corpus questions carry it. C5's "not represented at all" was wrong.
`_route_evolution` uses ranking **0 times**, so it is irrelevant to C6 either
way — but the claim mattered for how `period_change_analysis` was ranked, and
it should not be carried forward.

## 6. C6 candidate — `evolution`, reconfirmed on code

| | `evolution` | `period_change_analysis` |
|---|---|---|
| handler | `_route_evolution`, **167 lines** | `period_change_route.py`, **1,112 lines**, 29 functions |
| spec reads | `spec.metric`, `spec.aggregation` | none at module level — semantics live in `mi_agent.period_change` |
| raw-question reads | `_FUNNEL_KEYWORDS`, `"by stage"`, `"stage over time"`, `"stage migration"` | 0 at module level |
| accessors already built | `dataset_of`, `measure_request` (both from C5) | same, plus ranking parameters |

`evolution` remains the candidate — but on the evidence below it is **not
ready**.

## 7. Four-part dependency matrix

Runtime-owned surface derived from all 882 corpus questions, not from a
declared fixture: **32 owned questions**, routes `evolution` 28,
`evolution_funnel` 2, `evolution_pipeline_stage` 2. **14 delivered, 20 ok.**
Distinct semantic cases 32; renders 32 (no tab axis — the tab is inert since
the dataset remediation).

| dependency | represented? | owner agreement? | consumed by plan? | delivered coverage? | status |
|---|---|---|---|---|---|
| `dataset` | yes | **0 disagreements / 32** | `dataset_of` exists | funded ✓ 14 · **pipeline ✗ 0 of 7** | **PARTIAL** |
| `subject` / measure | yes | **0 disagreements / 32** | `measure_request` exists | ✓ 14 numeric | **READY** |
| `time` span | yes (`window_periods`) | n/a — evolution takes the whole series | `span_periods` exists | ✓ | READY |
| `time.grain` | yes | **no owner to disagree with** | not needed | n/a | **NOT REQUIRED** |
| `time.requested_grain` | yes | as above | not needed | n/a | **NOT REQUIRED** |
| `population` / scope | yes | route does not narrow (`lens_aware=False`) | `_whole_dataset_step("evolution")` provable | n/a | READY |
| `dimensions` | yes | n/a for this route | `grouping_concepts` exists | n/a | not required |
| `filters` | yes | not measured — see below | **no** — per-period filtering is route machinery | ✓ 1 of 2 | **UNPROVEN** |
| `operation` / temporal mode | yes | recogniser reads `spec.chart_type` | dispatch, as C1–C5 | ✓ | acknowledged exception |
| ranking | yes (94/882) | n/a | n/a | n/a | **NOT REQUIRED** |
| **funnel-stage selection** | **NO** | **no contract representation at all** | no | **✗ 0 of 2** | **PREREQUISITE** |
| **by-stage selection** | **NO** | **no contract representation at all** | no | ✓ 1 of 3 | **PREREQUISITE** |

## 8. Delivered coverage — the blocking finding

```
dependency / partition            owned  delivered
dataset=funded                       25         14   COVERED
dataset=pipeline                      7          0   NO DELIVERED CASE
route=evolution                      28         14   COVERED
route=evolution_funnel                2          0   NO DELIVERED CASE
route=evolution_pipeline_stage        2          0   NO DELIVERED CASE
filters                               2          1   covered
funnel-stage vocabulary               2          0   NO DELIVERED CASE
by-stage vocabulary                   3          1   covered
```

Root cause, measured directly rather than inferred:

```
funded_evolution            3 periods
pipeline_evolution          0 periods     <-- the fixture has no weekly extracts
pipeline_funnel_evolution   4 series entries, none delivering
```

**Seven of 32 owned questions, and two of three route identities, cannot be
exercised at all.** An equivalence measured on this surface would be
green — and vacuous for exactly the paths that differ most, the sub-routes
selected by raw-question reads with no contract representation. That is the
failure mode C5 hit and repaired by adding delivered cases; here it cannot be
repaired that way, because the data does not exist.

## 9. C6 cost estimate — and why it is uncertain

**Conversion cost (inside C6)**

| item | basis | raw lines |
|---|---|---|
| plan builder | C5's `build_temporal_compare_plan` 50, more branches | **60–75** |
| executor | C5's 27, three sub-paths | **35–45** |
| plan readers | C5's `compare_period_pair`+`compare_dataset` 11 | **12–18** |
| switch + registration | C5's `_route_compare` 37 + 5 | **40–50** |
| **route-specific** | | **≈ 150–190** |
| shared accessors needed | `dataset_of`, `measure_request` **already exist** | **0** |

**Prerequisite hardening (before C6 can fairly start)**

| item | raw lines |
|---|---|
| governed representation of funnel/stage selection | **unknown — no design exists** |
| pipeline-extract fixture, or an equivalent governed source | not production code, but blocking |

**Normalised burden: not computable.** The largest term has no design and the
coverage to validate it does not exist. Producing a number here would repeat
the C5 error in a worse form — C5's miss was 89% on work that at least had a
known shape.

## 10. Proposed C6 thresholds — withheld, with reasons

Not proposed. §14 permits derivation *"if and only if the dependency model is
complete enough"*, and it is not: two dependencies have no contract
representation, one has no delivered coverage on a third of the surface, and
`filters` has no proven plan consumption. Thresholds derived now would be
guesses dressed as pre-registration.

## 11. C6 verdict rules — defined now, for when it runs

Three independent claims. **Do not collapse them.**

**CONVERSION WITHIN THRESHOLD** — measured cost inside committed limits;
economics, payload and receipt equivalent on a **non-vacuous** surface;
regressions clean by name.

**DEPENDENCY MODEL SUPPORTED** — every required semantic dependency passed all
four parts *before* conversion, and no new generic prerequisite appeared during
it.

**COST MODEL SUPPORTED** — normalised like-for-like burden inside the
pre-registered band, in the single canonical unit, with prerequisite work
included.

C5 would now be scored: conversion **passed**, dependency model **failed**
(two prerequisites surfaced), cost model **failed** (117 against 62).

## 12. Status

# C6 PREREQUISITE REQUIRED

Cost units are standardised, C5's accounting is corrected, C5's surface gap is
closed and the population guard is strengthened. But C6's dependencies do not
pass: funnel-stage and by-stage selection have **no contract representation**,
and the pipeline half of the surface has **no delivered coverage**.

**Recommended next task:** *build the pipeline-extract fixture coverage first —
without it, no C6 equivalence evidence can be non-vacuous for 7 of 32 owned
questions and 2 of 3 route identities.* Then design the governed representation
of funnel/stage selection as its own contract task, and only then re-run this
four-part matrix and pre-register C6.
