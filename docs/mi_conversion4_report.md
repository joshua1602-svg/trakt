# Conversion 4 — `funded_bridge` — report

## Verdict

# CONVERGENCE NOT PROVEN

The conversion is **technically clean** — one new axis bridged, no new primitive,
behaviour identical, regressions clean — but **shared cost came in at 65 against
a pre-registered ≤ 45**, and total at 219 against ≤ 215.

| | pre-registered | measured | |
|---|---|---|---|
| **shared** production lines | ≤ 45 | **65** | **BREACH** |
| **route-specific** production lines | 90–170 | **154** | pass |
| **total** production lines | ≤ 215 | **219** | **BREACH** |
| new semantic axes bridged | exactly 1 (`dimensions`) | **1** | pass |
| new primitives | 0 | **0** | pass |
| economics (delivering surface) | equivalent | **0 differences** | pass |
| payload / receipt | equivalent | **36 pairs, 6,269 fields, 0 differences** | pass |
| behaviour movement | none | **0 across 36 renders / 216 fields** | pass |
| silent drops | 0 | **0** | pass |

**The thresholds were not revised.** They stand exactly as committed at
`de0edd0`, before the first C4 attempt.

Per the pre-registered definition, `MIGRATION ECONOMICS FALSIFIED` does **not**
apply: shared did not return materially toward Conversion 2's 138, exactly one
new generic abstraction was required, total is 1.9% over rather than materially
over, and no architecture redesign was forced. **Do not proceed automatically to
C5.**

---

## 1. Base and prerequisite re-verification

| | |
|---|---|
| base | `39544bb`, working tree clean |
| Defect B | `a126e45` present |
| contract-role fix | `a290f30` present |
| Defect A | `39544bb` present |
| original C4 thresholds | committed `de0edd0`, **unchanged** |

The five cases that previously stopped C4, re-run:

```
[1] valid grouped bridge calculates truthfully
    available=True  dimensionCol='collateral_geography'  net=32,575,267.01  contribs=9
[2] grouping role present in the contract
    {'collateral_geography': ('grouping', 'parser.bridge_dimension')}
[3] execution publishes the actual grouping
    metadata.groupedBy=['collateral_geography']
[4] grouping_proven=True
    declared_group_fields contains executed dim: True   facet ('region','applied')   ok=True
[5] missing dimensions remain unavailable
    available=False   end-to-end ok=False   groupedBy=None   '£0' in answer=False

ALL PREREQUISITES HOLD
```

C1/C2/C3 remain live through composition (27/27, 21/21, 12/12 plans built, 0
deferrals), and `_route_bridge` had no production conversion before this task.

## 2. Candidate assumption

```
contract axes the plan layer reads today : ['source_scope', 'time']
UNBRIDGED axes required                  : ['dimensions']  -> count=1
NEW primitives needed                    : none
time.comparison_period read by the plan layer today: False
```

**Holds: exactly one unbridged axis, no new primitive.** The already-known extra
shared requirement — `time.comparison_period`, a field on the already-bridged
`time` axis that the plan layer had never read — was recorded in advance and is
counted as shared cost below.

## 3. The `dimensions` bridge

`analytical_plan.grouping_concepts(interpretation)` — the axis bridge:

* consumes the authoritative contract, never the question;
* **obeys the role** — a `filter` or `unresolved` dimension is not returned, so a
  caller cannot silently promote a selector into an axis;
* governed field identity only (`candidate_concept`), never user wording;
* preserves contract order and de-duplicates;
* knows nothing about `funded_bridge` (asserted by test), so it is reusable.

`analytical_plan.comparison_period(interpretation)` — the named start period,
read from `time.comparison_period`. **Not a new axis**: `time` was bridged by
Conversion 2 through `window_periods`; this reads another field on the same
governed claim. Only a FILLED slot answers.

`_bridge_dimension` changed from reading `spec.bridge_dimension` itself to
**taking the governed concept as an argument**, so the decision of which
dimension is the axis moved to the contract while the helper kept its real job:
turning a concept into the column(s) and label this tape spells it with.

## 4. Route surface

12 owned cases × 3 scopes = **36 owned renders**, from executable ownership:
27 delivering, 9 refusing (the post-Defect-A baseline). 12 renders carry a stated
bridge dimension; 3 carry a stated comparison period.

## 5. Plan and economic equivalence

`migration_phase0/plan_equivalence_funded_bridge.py` — the shipped calculation
path reconstructed and run against the compositional one:

```
comparisons expected     : 36
comparisons made         : 36
economic fields compared : 1,788
A2 tolerance             : 0.005 (not widened)
```

Split by what the client actually sees:

| | renders | with differences |
|---|---|---|
| **delivering** | **27** | **0** |
| refusing | 9 | 6 |

**Every economic difference is on a render that refuses, before and after**, and
all six are the same two cases:

| case | question | shipped path resolved | contract path resolves |
|---|---|---|---|
| S3 | …for the **ALP Origination Book** | `total` → **whole book** | `cohort` → `['alp_origination']` |
| R1 | …for the **Highgate Mortgages Book** | `total` → whole book | `unresolved` → no narrowing |

This is the conversion **removing a known silent-widening path**, not
introducing a divergence: `resolve_lens_with_default` was registry-blind, so a
governed portfolio name fell through to the caller default — exactly the Phase 1E
defect the C4 STOP report documented as still live in this route (4 of 5 probes).
The route-independent guard refuses both cases either way, so **nothing
client-visible moves** (§7).

## 6. Payload and receipt equivalence

```
pairs compared            : 36  (expected 36 — OK)
envelope leaf fields      : 6,269
DIFFERENCES               : 0
```

Denominator asserted, not assumed: duplicate keys rejected, both sides required
to cover the same render set, pair count checked against `--expect`, list length
compared as its own leaf, and the populated-field census printed —
`answer` 36/36, `payloadKeys` 36/36, `portfolioScope` 36/36, `reconciliation`
33/36, `executionSummary` 33/36, `artifacts` 27/36, `facets` 15/36,
`verdict` 18/36. **No bespoke `funded_bridge` exception was required anywhere.**

## 7. Behaviour-delta control

The post-Defect-A behaviour was frozen as C4's baseline and re-measured after the
switch:

```
renders compared      : 36
fields compared       : 216   (ok, groupedBy, answer, facets, payloadKeys, reconciliation)
BEHAVIOUR DIFFERENCES : 0
delivering before=27 after=27    refusing before=9 after=9
```

**No new refusal→delivery, no delivery→refusal, no economic movement, no route
movement.** C4 changed the execution mechanism and not the behaviour, which is
exactly what it was asked to do.

The five A5 surfaces are **byte-identical** to the post-Defect-A capture, and
**silent drops remain 0**.

## 8. `resolve_lens_with_default` and duplicate narrowing

Checked over the AST:

```
_route_bridge  calls resolve_lens_with_default = False
               calls lens_from_selection       = False
               calls _apply_lens_filter        = False
```

The contract supplies the population semantics the helper previously supplied —
and supplies them **better**, through the governed registry (§5). The old owner
is unreachable from the converted route, pinned by a test that also asserts the
helper still has other owners, so this conversion cannot have over-reached and
retired it globally: `_route_cohort_progression` and `mi_agent_workflow` still
depend on it.

Narrowing: the converted route reaches `evolution._scope_frame_lens` through the
plan's own `lens_filters` only. `_apply_lens_filter` is untouched and still used
by `period_change_route`. **One narrowing mechanism active for this route.**

## 9. Cost, hunk by hunk

`git diff --numstat -M 39544bb`, production only. The already-completed Defect
B / contract-role / Defect A fixes are **not** charged to C4 — the diff base is
the commit that closed the last of them.

```
162    0   mi_agent_api/analytical_plan.py
 39   18   mi_agent_api/chat_routing.py
                                  TOTAL  219
```

| bucket | lines |
|---|---|
| **shared** | **65** |
| **route-specific** | **154** |
| product hardening | 0 |
| cleanup | 0 |
| **total** | **219** |

Shared, itemised — this is the number that breached, so it is broken out in full:

| lines | item |
|---|---|
| 24 | `grouping_concepts` — **the `dimensions` axis bridge** |
| 7 | `ROLE_GROUPING` constant |
| 20 | `comparison_period` — the `time.comparison_period` access |
| 14 | the axis section banner (documentation) |
| **65** | |

Route-specific: `build_funded_bridge_plan` 44, `funded_bridge` executor 38,
the switch and deferral in `_route_bridge` 32, `_bridge_dimension`'s change of
input 18, `bridge_start_period` 6, `BRIDGE_TOP_N` 5, banner 4, prose 4,
registration 3, signature 3 — **154**.

| | |
|---|---|
| modules changed | 2 |
| test files touched | 1 new (`test_conversion4_funded_bridge.py`, 21 tests) |
| instruments | 2 (equivalence harness; envelope snapshot gains the surface) |
| commits to equivalence | 1 production commit, 0 corrections |
| bridges added | **1** (`dimensions`) |
| new primitives | **0** |
| duplicate owners removed | 1 (`resolve_lens_with_default`, from this route) |

### Why shared breached, stated without arguing the threshold down

The ≤ 45 budget was built as *"21 (the C3 shared floor) + 24 (one measured
bridge, `span_from_claim`)"*. Measured against that:

* the **bridge itself** cost **31** (`grouping_concepts` 24 + `ROLE_GROUPING` 7)
  against the 24 predicted — a 29% overshoot on the item the budget was actually
  sized for;
* **`comparison_period` cost a further 20**, and it was **not in the budget** —
  it was identified only in the C4 STOP report, after the threshold was set, and
  this task's own brief required it to be counted as shared;
* the **14-line section banner** is documentation, counted the same way every
  prior conversion counted its comments.

That is an explanation of the composition, not a case for moving the line. The
threshold said ≤ 45 and the measurement says 65. **The verdict follows the
measurement.**

## 10. Regression, by name

Full run over `mi_agent/tests/`, `mi_agent_api/`, `question_interpretation/tests/`,
the C1/C2/C3 guards, the Defect A tests and the new C4 tests:

```
18 failed, 3134 passed, 2 skipped, 7 xfailed in 391.04s
```

Attributed against the correct base — `39544bb`, the commit C4 branched from,
run in the same environment:

```
base (39544bb) failures : 18
C4 retry failures       : 18
INTRODUCED BY C4        : 0     (set difference empty)
FIXED BY C4             : 0     (set difference empty — nothing silently fixed)
```

**Identical by name in both directions.** The 18 are the pre-existing baseline
set (live-fixture cases, the time-axis wording case, the `mi_agent_api` pipeline
and receipt cases carried since C3).

One test differs from the *older* C3 baseline —
`test_funded_balance_bridge_returns_reconciling_waterfall`, which now passes.
That was **fixed by Defect A** (`39544bb`), not by this conversion: it is absent
from the `39544bb` failure list, so it was already passing before C4 began. It
is recorded here so the improvement is not mis-credited to C4.

* `tests/test_conversion4_funded_bridge.py` — **21 passed** (new).
* Five A5 surfaces **byte-identical**; **silent drops 0**.

**Introduced failing names: 0. Silent drops: 0. Route movement: 0. Answer /
refusal movement: 0. Economic movement on the delivering surface: 0.**

## 11. The four-conversion sequence

Consistent re-baseline method throughout; C1–C3 figures unchanged from
publication.

| | shared | route-specific | hardening | cleanup | total | production commits |
|---|---|---|---|---|---|---|
| C1 `portfolio_summary` | 200 | 176 | 0 | 7 | **383** | 2 |
| C2 `period_movement` | 138 | 144 | 0 | 0 | **282** | 1 |
| C3 `geo_exposure` | 21 | 129 | 0 | 1 | **151** | 1 |
| **C4 `funded_bridge`** | **65** | **154** | 0 | 0 | **219** | **1** |

```
shared           200 -> 138 ->  21 ->  65
route-specific   176 -> 144 -> 129 -> 154
total            383 -> 282 -> 151 -> 219
```

**What C4 proves, and only this:** adding the `dimensions` axis to the
compositional core cost **65 shared lines** — far below C2's 138, well above C3's
21, and above the 45 budgeted. Route-specific stayed in range at 154, consistent
with the 129–176 band the first three established.

**What C4 does not prove:** that every future bridge costs ~65. This is **one**
bridge, and the one measurement before it (`span_from_claim`, 24) differed by
more than a factor of two. Two observations are not a rate.

## 12. Contract axes

```
bridged BEFORE C4 : 2 of 9   source_scope, time (window_periods)
bridged AFTER  C4 : 3 of 9   source_scope, time (window_periods + comparison_period),
                             dimensions
remaining unbridged: operation, subject, filters, target, population, dataset
```

## 13. Should C5 proceed?

**Not automatically** — that is what `CONVERGENCE NOT PROVEN` means, and it is
the pre-registered consequence.

What should happen first is a **re-baseline of the shared-cost model on four
observations**, because the sequence 200 → 138 → 21 → 65 no longer fits a simple
decay: it fits "a conversion that needs no new generic work costs ~21 shared, and
one that adds an axis costs ~45–65". If that two-regime reading is right it is
directly testable, and it makes the next threshold predictable instead of
arbitrary.

## 14. Recommended next step

> **Re-baseline the shared-cost model against the two regimes the four
> conversions now show — "no new generic work" (~21) versus "one new axis"
> (~65) — and pre-register C5's threshold from whichever regime C5 falls into,
> before choosing the route.**

Two things to carry in:

1. The remaining candidates split cleanly by regime: `evolution` and
   `temporal_compare` both need the **`dataset`** axis (a new-axis conversion),
   while any route drawing only on the three bridged axes is a reuse
   conversion. Choose the regime deliberately rather than the route first.
2. `comparison_period` showed that an already-bridged axis can still carry
   unread fields. Before pre-registering C5, enumerate which fields of the three
   bridged axes the plan layer actually reads — so the next budget is not
   surprised the same way this one was.
