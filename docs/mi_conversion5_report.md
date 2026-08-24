# Conversion 5 — `temporal_compare` — report

Base `5cf09f9` → `cc91339`. Thresholds as committed at `7d9f4c6`, untouched.

---

## 1. Base and preconditions

HEAD `5cf09f9`, clean tree. All prerequisite commits present (`7d9f4c6`,
`83da6a7`, `cb9b64c`, `ad67ee9`, `4c62a0f`, `5cf09f9`). C1–C4 live through the
compositional path at `chat_routing` lines 473, 622, 1698/1704, 2031.

Dependency verifier before any C5 change:

```
readings                                   : 26
dataset disagreements                      :  0
measure disagreements                      :  0
readings whose periods are STRUCTURAL      : 26 of an EXPECTED 26
```

## 2. The owned surface — and why it had to be extended first

The enumeration inherited from the C5 STOP report asked for October, November
and September. **This book carries 2026-04, 2026-05 and 2026-06.** Every one of
its 13 owned cases therefore *refused*, and an equivalence measured over them
would have been vacuous: both sides decline for the same reason and no
calculation is exercised. Conversion 1 already reported one pass of exactly
that shape.

Eight delivered cases were added **before any equivalence was claimed**, on
periods the fixture holds, one per governed measure family:

```
A1 April vs May funded balance      A5 May vs June WA current LTV
A2 May vs June funded balance       A6 May vs June average interest rate
A3 April vs June funded balance     A7 May vs June borrower age
A4 May vs June loan count           A8 May vs June case count
```

`A8` is deliberate: it pins the bare-`case` house rule end to end on the
delivered path, not only at the resolver.

**Asserted denominator: 28 cases × 2 tabs = 56 renders; 21 owned → 42.**
Executed routing, 0 disagreements with the declaration. **16 of the 42 carry
real numbers.**

The second axis is the workspace **tab**, not the source lens, because this is
the first converted route whose dataset decision was ever tab-sensitive. It is
not any more, and snapshotting across tabs is how that stays true.

## 3. The old route's semantic inventory

| fact | shipped source | after C5 |
|---|---|---|
| period pair | `spec.compare_periods[0:2]` | `time.comparison_periods` |
| dataset | `workspace.resolve_dataset(question)` | `interpretation.dataset` |
| measure | `spec.metric`, `spec.aggregation` | `subject.candidate_concept` |
| population | **never passed** — `lensApplied` False on every owned case | plan declares `whole_dataset` |
| temporal mode | `spec.temporal_mode` (recogniser) | unchanged — recognisers stayed on the spec through C1–C4 |

## 4. The plan

```
STACK_PERIODS      dataset, take=named_pair, periods=[a, b]   (BLOCKS on < 2)
SELECT_POPULATION  kind=whole_dataset
RESOLVE_MEASURE    metric, aggregation
COMPARE            absolute and percentage delta, b relative to a
```

Two shared accessors added: `dataset_of` and `measure_request`.

**The population step is deliberately not `_population_step`.** This route does
not narrow by source portfolio. Borrowing the shared step would declare a
narrowing the route does not apply — a false narrowing on the receipt — and
would BLOCK on an empty scope, refusing questions that answer today. A question
that *names* a scope is already refused by the facet layer as a lost narrowing;
that owner is left alone.

**A known lossy edge, stated not hidden.** `measure_request` expands the
contract's single governed concept back into the `(metric, aggregation)` pair
the existing resolvers take. A parser output of `metric="loan_count"` with a
non-count aggregation projects to the same concept and would read as a count
request. Measured unreachable across all 42 readings; closing it means the claim
carrying the aggregation, which is a contract change and not a route's.

## 5. Economics

`migration_phase0.plan_equivalence_temporal_compare` — shipped path vs
compositional path, same engine.

```
comparisons made          : 42  (expected 42)
fields compared per pair  : 20
DIFFERING COMPARISONS     :  0        (A2 tolerance, £0.005)
comparisons carrying a delivered result : 16
```

## 6. Payload and receipt

`migration_phase0.envelope_snapshot`, before captured from the **actual
pre-switch route**.

```
pairs compared        : 42  (expected 42 — OK)
envelope leaf fields  : 4464
DIFFERENCES           :  0

answer            non-empty in 42/42      artifacts       16/42
executionSummary  non-empty in 42/42      reconciliation  16/42
metadataKeys      non-empty in 42/42      facets           8/42
payloadKeys       non-empty in 42/42      verdict          8/42
portfolioScope    non-empty in 42/42      route           42/42
```

## 7. The switch, and the duplicate ownership removed

`_route_compare` calls `_plan.temporal_compare`. **`spec` is gone from its
signature.** That absence is the conversion's real result: there is nothing left
for the route to read from the parse, and a route that cannot reach the parse
cannot quietly re-decide any of it. `spec_dict` stays — echoed into the envelope
for the receipt layer, consulted for no semantic fact.

Proved before removal: the contract owns each fact; the plan consumes it; the
single call site passes the contract; and the 42-pair envelope diff shows no
consumer relied on a disagreement.

## 8. One C1–C4 guard updated — the invariant, not the assertion

`test_every_route_plan_reads_the_one_population_step` required every plan
builder to call `_population_step`. That encodes *"every route narrows by
lens"*, which is false for this one. It now asserts what it was always meant to:
a builder must not keep a **second copy** of the population decision — it either
reaches the one definition, or it declares `whole_dataset` **and reads no scope
field at all**, with the exemption unclaimable by a builder that touches one
anyway.

## 9. Regression, by name

Full estate at `cc91339` against the 214-name baseline:

```
185 failed, 10279 passed, 35 skipped, 16 xfailed, 28 errors  =  213 names
introduced : 0
gone       : 1   (the rename from the dataset remediation)
```

Eleven further failures appear only when this route's tests share a process with
the conversion guards. Attributed rather than assumed — C5 stashed, same
combination re-run: **14 before, 14 after, 0 introduced, 0 fixed.** Pre-existing
pollution in that combination.

C1–C4 guards in isolation: **69 passed.**

## 10. Cost

Hunk-by-hunk, same classification method as C1–C4; module-level lines attributed
to the definition they introduce.

| definition | + | − | bucket |
|---|---|---|---|
| `build_temporal_compare_plan` | 60 | 0 | route-specific |
| `_route_compare` | 28 | 9 | route-specific |
| `measure_request` | 32 | 0 | **shared** |
| `temporal_compare` (executor) | 29 | 0 | route-specific |
| `dataset_of` | 18 | 0 | **shared** |
| `compare_period_pair` | 9 | 0 | route-specific |
| `compare_dataset` | 6 | 0 | route-specific |
| `_register_default_recognisers` | 3 | 2 | route-specific |
| section separators | 2 | 0 | route-specific |

| bucket | lines |
|---|---|
| **Shared** | **50** |
| **Route-specific** | **148** |
| Product hardening | 0 |
| Cleanup | 0 |
| **Total** | **198** |

### Normalised architectural burden — a comparability disclosure

```
C5 measured shared              = 50
comparison_period prerequisite  = 19
normalised C5 shared burden     = 69
```

The 19 is **not** added to the threshold measurement. It is disclosed because
C5 is what exposed that work, and a threshold verdict that quietly benefited
from its removal would compare C5 against C1–C4 on different terms.

## 11. Conversion threshold verdict

| condition | threshold | measured | |
|---|---|---|---|
| shared | ≤ 75 | **50** | ✓ |
| route-specific | 90–150 | **148** | ✓ *(by 2 lines)* |
| total | ≤ 225 | **198** | ✓ |
| economics equivalent | 0 | 0 of 42 | ✓ |
| payload/receipt equivalent | 0 | 0 of 4464 leaves | ✓ |
| regression clean | 0 introduced | 0 | ✓ |
| no unplanned semantic dependency *during C5* | — | none | ✓ |

# C5 CONVERSION WITHIN THRESHOLD

## 12. Cost-regime model verdict

Judged on the normalised burden and on whether the work was **predicted**.

| | predicted | actual |
|---|---|---|
| `dataset` accessor | 20 | **18** ✓ |
| measure accessor (`subject`+`operation`) | 28 | **32** ✓ |
| `operation` accessor | required | **not needed at all** ✗ |
| `comparison_period` | *"already bridged, no unread fields"* | **19 lines of structural closure** ✗ |
| dataset axis | *"a guarded field read"* | a guarded field read **plus two disagreeing owners** ✗ |
| | **62** | **69 normalised** |

The line prediction was good — **69 against 62, +11%**. The *completeness*
prediction was wrong three times, twice in the expensive direction:

* `comparison_period` was declared already bridged. It was not, at field level,
  and closing it took its own task.
* The `dataset` axis was described as one guarded field read. In fact two
  owners disagreed about 27% of this route's readings, and reconciling them
  required a product decision about workspace-tab semantics with blast radius
  across all 882 corpus questions. That is a **semantic layer the model never
  mentioned**, and its line cost — net executable **+0**, a pure relocation —
  is precisely why a line-based model could not see it coming.
* `operation` was predicted as required and was not needed at all.

The definition of REGIME MODEL SUPPORTED requires that *no new generic semantic
work appeared*. Two pieces did. They appeared during prerequisite closure rather
than inside C5 — which is why the conversion itself was clean — but they were
discovered **because C5 went looking**, and pretending otherwise would make the
model look more predictive than it was.

# REGIME MODEL PARTIALLY SUPPORTED

*Meaning: field-level dependency inspection now predicts line cost well. It does
not yet predict whether the fields it inspects agree with production. Those are
different questions, and the second one is where both surprises came from.*

## 13. What C5 proves

1. **Can `temporal_compare` run entirely through the compositional
   architecture?** Yes. `spec` is not in its signature.
2. **Did it require a new semantic owner?** No. Two accessors, both reading
   claims that existing owners populate.
3. **Did the field-level inventory correctly predict the required generic
   work?** Partly. It predicted the *cost* well and the *inventory* imperfectly
   — it missed one unread field and one disagreeing owner, and over-predicted
   one axis.
4. **Is the remaining migration now wiring rather than semantic discovery?**
   Closer, but not yet. The two accessors C5 built are reusable, and the one
   remaining known unread field pair is `time.grain` / `time.requested_grain`.
   Whether *that* axis has a second owner is unmeasured — and that is exactly
   the question C5 says to ask before budgeting C6.
5. **C6:** see below.

## 14. Remaining generic semantic infrastructure

| | |
|---|---|
| bridged and consumed | `source_scope`, `time` (window + comparison pair), `dimensions`, `dataset`, `subject` |
| known unread | `time.grain` / `time.requested_grain` — populated by projection, consumed by nothing |
| not represented at all | ranking / `operation` beyond `movement` |
| accessors now available to any route | `_population_step`, `span_from_claim`, `grouping_concepts`, `comparison_period`, `comparison_periods`, `dataset_of`, `measure_request` |

## 15. Recommended C6

**`evolution`.** Measured, not assumed:

```
_route_evolution   167 lines
  spec reads       spec.metric, spec.aggregation   -> measure_request EXISTS
  dataset          workspace.resolve_dataset        -> dataset_of EXISTS
  raw question     `question.lower()`, `in q`       -> the work that remains
  grain            week / month handling            -> the one unread field pair
```

Both generic accessors it needs were built by C5. Its remaining generic
dependency is the `grain` pair, and its route-specific work is the raw-question
reads. `period_change_analysis` stays last: at 1,112 lines it needs `operation`
/ ranking, which the contract does not represent **at all**, making it a
contract-extension task before it is a conversion.

**Before budgeting C6, run the disagreement check that C5's two surprises both
came from:** does `time.grain` have a second owner, and does the contract's
answer equal production's? That check is now an instrument
(`dependency_verification_*`) and it is cheap.

## 16. Recommended next task

**Run the C6 dependency verification on `evolution` — including the
owner-agreement check on `time.grain` — and pre-register C6's thresholds from
what it finds, before touching any production code.**
