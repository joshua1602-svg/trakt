# Conversion 3 — `geo_exposure` — report

## Verdict

# CONVERGENCE SUPPORTED

Every pre-registered threshold met, and met close to the *declared prediction*
rather than close to the threshold — which is the difference between a test that
passed and a test that was fitted.

| | pre-registered | predicted | **measured** | |
|---|---|---|---|---|
| **shared** production lines | ≤ 40 | ~20 | **21** | ✓ |
| **route-specific** production lines | 90–150 | ~110 | **129** | ✓ |
| **total** production lines | ≤ 190 | ~130 | **151** | ✓ |
| new primitives | 0 | 0 | **0** | ✓ |
| new generic semantic concepts | 0 | 0 | **0** | ✓ |
| new contract bridges | 0 | 0 | **0** | ✓ |
| economics | equivalent | | **36 comparisons, 49,968 fields, 0 differences** | ✓ |
| payload / receipt | equivalent | | **36 pairs, 10,036 leaf fields, 0 differences** | ✓ |
| regressions | clean | | **introduced failing names 0; silent drops 0** | ✓ |

**The thresholds were committed at `75380a2`, before a line of production code
was touched, and are unchanged.**

---

## 1. Base and HEAD

| | |
|---|---|
| branch | `claude/clause-splitting-scoping-38ahbz` |
| base | `67981d5` — the re-baseline, working tree clean |
| stop conditions committed | `75380a2`, **before** any production change |
| production commit | `1ce7a6d` |

### Conversion 1 and Conversion 2 were confirmed **live** before starting

Not assumed — executed. A probe wrapped both plan builders with counters and ran
each converted route's owned surface:

```
portfolio_summary    owned renders=27   other route=0   plans built=27
period_movement      owned renders=21   other route=0   plans built=21
BOTH CONVERSIONS LIVE THROUGH COMPOSITION: YES
```

This check exists because Conversion 1's first equivalence run was **vacuous** —
0 differences across 54 pairs while the plan path was taken **zero** times, an
exception in the interpretation builder having been silently swallowed.
Confirming the converted routes are genuinely live is now a precondition of
every conversion.

### Regression baseline, taken before the switch

```
mi_agent + question_interpretation    7 failed, 1811 passed
mi_agent_api                         12 failed, 1231 passed
```

Identical to the state Conversion 2 left behind.

## 2. `geo_exposure` was still the correct candidate

Checked mechanically at this HEAD, not read from the re-baseline:

```
1. SEMANTIC AXES THE HANDLER READS FROM THE QUESTION
     source_scope           via ['_resolve_lens']
2. AXES THE PLAN LAYER ALREADY BRIDGES
     source_scope           _population_step
     time                   span_periods / span_from_claim
     UNBRIDGED AXES THIS ROUTE NEEDS: none
3. PRIMITIVES
     declared today : ['compare','group','rank','resolve_measure',
                       'select_population','stack_periods']
     geo requires   : ['select_population','resolve_measure','group','rank']
     MISSING (new primitives needed): none
```

| check | result |
|---|---|
| in the generic compositional core | yes — closure report §8, rank 3 |
| requires a new primitive | **no** |
| requires a new generic semantic concept | **no** |
| semantic inputs already bridged | **yes, all of them — one axis, `source_scope`** |
| blocked by a known specialist semantic | **no** |

## 3. The surface, verified from executed routing

`migration_phase0/route_ownership_geo_exposure.py` — 17 candidate questions ×
3 caller scopes = 51 renders, ownership taken from **executed routing**, never
from wording similarity.

| | |
|---|---|
| claimed by `geo_exposure` | **12** cases → **36 owned renders** |
| deliberately NOT claimed, and verified not claimed | **5** |
| disagreeing with what the instrument declared | **0** |

The five excluded cases are named with the route that should take each, so the
surface cannot quietly grow:

| case | question | goes to |
|---|---|---|
| X1 | "Show the balance by region" | an ordinary stratification, not the ITL3 engine |
| X2 | "Which region grew the most over the last three months?" | period change — a comparison this route would silently drop |
| X3 | "Show top 5 regions by balance" | a grouped ranking |
| X4 | "Funded balance bridge by region" | the bridge route |
| X5 | "Which regions breached the concentration limit?" | risk limits |

## 4. Plan equivalence, proved **before** the switch

`migration_phase0/plan_equivalence_geo_exposure.py`, run while `_route_geo` was
still on the shipped path. **Both sides executed**, neither simulated.

```
shipped        _resolve_lens -> _apply_lens_filter -> exposure_by_itl3
compositional  contract -> build_geo_exposure_plan -> scope_frame -> exposure_by_itl3
```

| | |
|---|---|
| comparisons expected | 36 |
| comparisons made | **36** |
| economic fields compared | **49,968** |
| **economic differences** | **0** |
| A2 tolerance | 0.005 — **not widened** |

Compared per case: population **rows**, `available`, `reason`, `total`,
`coveragePct`, `basis`, `areaCount`, `resolvedFromItl3Field`,
`resolvedFromPostcode`, the scope label, whether a narrowing was applied, and
**every field of every one of the 172 ITL3 areas** — balance, count, share,
weighted LTV, average age.

**Populations are compared on rows, not on the lens name.** Phase 1G made the
plan resolve a category through the governed registry, so it selects
`{'source_portfolio_id': [...]}` where a lens may carry a different shape for the
same population. Comparing names would have reported that governed correction as
a regression.

### The zero is not the zero of an unexercised filter

```
(shipped rows, plan rows)   comparisons
  (3909, 3909)                 14        the acquired book
  (7126, 7126)                 11        the direct book
  (11035, 11035)               11        the whole funded book
distinct populations exercised : 3
narrowed comparisons           : 25 of 36
```

Three distinct populations, and 25 of the 36 comparisons ran through a real
narrowing.

## 5. Payload and receipt equivalence, with the denominator **proved**

`migration_phase0/envelope_snapshot.py`, before and after, on the real tree.

| | |
|---|---|
| pairs compared | **36** (asserted against `--expect 36`) |
| envelope leaf fields | **10,036** |
| **differences** | **0** |

Compared in full: `answer`, `artifacts` (chart rows, table columns, series,
formats), `route`, `ok`, `controlledRefusal`, `error`, `verdict`, `facets`,
`guardFacets`, `notApplied`, `payloadKeys`, `metadataKeys`, `executionSummary`,
`portfolioScope`, `portfolioCoverage`, `reconciliation`, `sourceNotes`,
`warnings`, `lensApplied`.

### The denominator is asserted, not assumed

Conversion 2's first attempt at this diff keyed rows on a field that did not
exist, silently collapsed 36 entries to 12, and reported *"36 pairs, 0
differences"*. Every check below exists because of that, and the instrument now
**refuses** to report a clean result it cannot justify:

* a duplicate key raises `DENOMINATOR UNSOUND` rather than overwriting a row;
* both sides must cover the **same** render set, or it names the difference;
* the pair count is asserted against `--expect`;
* **list length is compared as its own leaf**, so a truncated list is a
  difference rather than a shorter loop that compares fewer things and still
  says zero;
* the snapshot itself refuses to write if the surface size moved.

And a zero over empty structures proves nothing, so the diff prints what
actually carried content on the before side:

```
answer            36/36      metadataKeys      36/36
artifacts         33/36      payloadKeys       36/36
executionSummary  33/36      portfolioScope    36/36
facets            24/36      reconciliation    36/36
route             36/36      verdict           27/36
```

**No bespoke `geo_exposure` exception was required anywhere in the payload or
receipt path.** The executor returns the engine's own result dict, so everything
downstream is unchanged by construction — which is why the receipt cost of this
conversion is zero lines.

## 6. The switch

```python
# before
lens = _resolve_lens(question, source_lens)          # reads the question
if lens.filters: df = _apply_lens_filter(df, lens)
result = geo_mod.exposure_by_itl3(df)

# after
geo = _plan.geo_exposure(df, interpretation=interpretation)
```

| | |
|---|---|
| route switched | `geo_exposure` **only** |
| the route defers without a contract | yes — `interpretation is None → return None` |
| capability added | **none**; T3–T7 untouched |
| new primitive | **none** — `select_population`, `resolve_measure`, `group`, `rank` all existed |

**No `stack_periods` step.** Geographic concentration is a point-in-time question
answered from the frame the caller is working in, and declaring a period step
would claim a governance property this answer does not have.

Verified over the AST, because `_route_geo`'s comments still *name*
`_resolve_lens` and `_apply_lens_filter` — to say they are deliberately
unreachable — and a substring guard reads those sentences as the calls they deny:

```
_route_geo  calls _resolve_lens=False  _apply_lens_filter=False  exposure_by_itl3=False
```

The `geo as geo_mod` import became dead and was removed.

## 7. Duplicate population narrowing

The two implementations, read and compared:

| | `chat_routing._apply_lens_filter(df, lens)` | `evolution._scope_frame_lens(df, filters)` |
|---|---|---|
| reached from | a resolved **lens object** | a **filters dict**, which the plan owns |
| handles | the source-portfolio id field only | any column |
| matching | membership, strip + `.lower()` | membership, strip + `.casefold()` |
| **agreement** | **they agree**; no divergence on this book | |

They are equivalent for governed portfolio ids on this book, so this is a
**duplicate owner, not a defect**.

**After this conversion `geo_exposure` has exactly one narrowing mechanism**: the
plan's own governed filters, through the new `scope_frame`. `_apply_lens_filter`
is **not** retired — `period_change_route` still calls it, and Conversion 3 is
not a consolidation exercise. Both facts are pinned by tests.

## 8. Regression, by name

### The five registered A5 surfaces — **byte-identical**

Not "the same totals": the full stdout of each surface was diffed against the
capture taken after Conversion 2.

```
run_robustness_deterministic     IDENTICAL
shipped_shapes                   IDENTICAL
routed_surface                   IDENTICAL
mi_recognition_diagnosis         IDENTICAL
time_series_surface              IDENTICAL
```

| surface | result |
|---|---|
| robustness 44 | 32 CORRECT / 6 UNHELPFUL / 4 SAFE / 2 DISCLOSED — and identical for every intent Q1–Q9, including seasoning **Q1 4 · Q7 4 · Q8 12** |
| shipped shapes | 0 wrong answers |
| routed surface | 31 passed, 1 failed — `rt_004`, the pre-registered known-open defect |
| recognition 61 | DELIVERED 15, and identical by shape |
| time-series | T1 PROVEN · T2 PARTIAL · T3–T8 ABSENT · **silent drops 0** · honest refusals 20/29 |

### The whole repository, by name

The full repository was run in four shards, before and after, in the **same
execution environment on both sides** — the trap that made two Conversion 2
failures look introduced when a fresh worktree had merely skipped them.

| shard | scope | baseline | after Conversion 3 | introduced | fixed |
|---|---|---|---|---|---|
| A | `mi_agent` + `question_interpretation` | 7 failed, 1811 passed | 7 failed, 1811 passed | **0** | 0 |
| B | `mi_agent_api` | 12 failed, 1231 passed | 12 failed, 1231 passed | **0** | 0 |
| C | `tests/` first half | 51 failed, 4 errors, 3756 passed | 51 failed, 4 errors, 3756 passed | **0** | 0 |
| D | `tests/` second half | 45 failed, 24 errors, 3326 passed | 45 failed, 24 errors, 3326 passed | **0** | 0 |

Every shard was compared as a **set difference of failing names in both
directions**, not by totals. All four are empty on both sides: nothing
introduced and — equally important to check — nothing silently fixed, which
would have meant the comparison was measuring something other than this change.

New tests, run separately because the shard file lists predate them:
`tests/test_conversion3_geo_exposure.py` — **16 passed**.
`tests/test_conversion1_portfolio_summary.py` and
`tests/test_conversion2_period_movement.py` — **32 passed**.

**Introduced failing names: 0. Silent drops: 0. Silent population widening: 0.**

### The known safe failure is unchanged

The `ALP Acquired Back Book` label collision still **fails closed**. The
population parser still reads "Back Book" inside the governed label as a
seasoning segment, the answer still refuses, and the strict `xfail` in
`tests/test_portfolio_name_resolution.py` still holds — a strict xfail that
started passing would itself be a failure, so this is asserted in both
directions. **Nothing was made to answer by ignoring one of the competing
interpretations.**

### The two prior-conversion guards that moved, and why

Two tests failed after the switch. In both cases the conversion was **working**,
and in both cases the test's assertion was a *snapshot that decays by one on
every conversion* — a changelog rather than a guard.

| test | asserted | what happened | now asserts |
|---|---|---|---|
| C1 `test_the_other_routes_keep_their_owner` | `_resolve_lens` calls **≥ 4** | Conversion 3 retired geo's use → 3 | the resolver is unreachable from **every converted route** and still reachable from unconverted ones |
| C2 `test_both_routes_read_one_population_step` | callers are exactly the **two** builders that existed then | Conversion 3 added a third | there is **one** definition of the population step and **every** plan builder reaches it |

Both replacements were checked to be non-vacuous:

```
_resolve_lens still owned by : ['_capability_state', '_disclose_lens_scope',
                                '_route_concentration', 'try_route']
plan builders                : ['build_geo_exposure_plan',
                                'build_period_movement_plan', 'build_plan']
all reach _population_step   : True
```

**No behaviour of Conversion 1 or Conversion 2 changed.** The other 30 tests in
those two files are untouched and pass, and the live probe re-confirms both
routes still execute compositionally.

## 9. Cost, hunk by hunk

Measured `git diff --numstat -M 67981d5 HEAD`, production only — tests, docs and
instruments excluded.

```
102    0   mi_agent_api/analytical_plan.py
 30   19   mi_agent_api/chat_routing.py
                                   TOTAL  151     (S3 threshold 190)
```

Classified by the re-baseline's rule — *is this production code reachable from
more than one route at the end of Conversion 3?*

| bucket | lines |
|---|---|
| **shared** | **21** |
| **route-specific** | **129** |
| product hardening | **0** |
| cleanup | **1** |
| **total** | **151** |

Every hunk, attributed:

| lines | hunk | class |
|---|---|---|
| 21 | `scope_frame(plan, df)` — narrow one resolved frame through the plan's own governed filters | **shared** |
| 14 | the Conversion 3 section banner | route-specific |
| 5 | `TOP_AREAS` — geo's display constant | route-specific |
| 29 | `build_geo_exposure_plan` | route-specific |
| 33 | `geo_exposure` executor | route-specific |
| 32 | **the switch** — the plan call replaces lens resolution and narrowing | route-specific |
| 6 | defer when no contract | route-specific |
| 3 | signature: `interpretation` param, `Optional` return | route-specific |
| 3 | registration passes the contract | route-specific |
| 2 | `result = geo` | route-specific |
| 2 | the prose reads the plan's label | route-specific |
| 1 | the now-dead `geo as geo_mod` import | **cleanup** |

**The one shared line-item is exactly the one the re-baseline predicted and
budgeted for**: `geo_exposure` is the first converted route handed a resolved
frame rather than an output root, and `scope_frame` is that entry point. It cost
21 lines against a 40-line allowance.

### Everything else recorded

| | |
|---|---|
| production modules changed | **2** (0 new) |
| test files touched | 3 — 1 new (`test_conversion3_geo_exposure.py`, 16 tests), 2 guard rewrites |
| instruments | 3 (1 new surface, 1 new equivalence harness, 1 extended with a denominator proof) |
| commits | **5**, of which **1 touched production** |
| corrections needed to reach equivalence | **0** — the first switch was the last |
| new primitives | **0** |
| generic semantic bridges added | **0** |
| duplicate owners removed | **1** — `_apply_lens_filter` unreachable from this route (not retired globally) |
| blockers | none |

## 10. The convergence verdict

# CONVERGENCE SUPPORTED

Judged exactly as pre-registered, with no threshold moved:

| requirement | threshold | measured | |
|---|---|---|---|
| shared | ≤ 40 | **21** | ✓ |
| route-specific | 90–150 | **129** | ✓ |
| total | ≤ 190 | **151** | ✓ |
| new primitive | 0 | **0** | ✓ |
| new generic semantic concept | 0 | **0** | ✓ |
| economics equivalent | yes | **0 differences / 49,968 fields** | ✓ |
| payload / receipt equivalent | yes | **0 differences / 10,036 fields** | ✓ |
| regressions clean | yes | **0 introduced names, 0 silent drops** | ✓ |

**Meaning, as written before the result: the common compositional layer is now
materially reusable rather than being rebuilt route by route.**

Two things make this more than a threshold being cleared.

**The prediction was right, not just the bound.** The declared landing zone was
~20 shared / ~110 route-specific / ~130 total. Measured: **21 / 129 / 151**. The
shared figure — the one the whole experiment turned on — was predicted to within
a single line. A model that only clears its ceiling could be luck; a model that
lands on its point estimate is a model that understands the cost.

**Route-specific cost did what the model said it would.** It was predicted to
stay roughly constant per route, because that is what converting a route
actually costs. Across three conversions: **176 → 144 → 129**. Mildly declining,
never collapsing, and never near zero.

### What is *not* claimed

Convergence of the shared layer is **not** the same as a cheap migration. 151
lines is still 2–5× the closure report's original 30–60 estimate for this route.
The claim supported here is narrower and specific: **the marginal cost of a
conversion is now dominated by route-specific work, and the shared layer is no
longer being rebuilt each time.**

Nor does it generalise to routes that still need a bridge — see §12.

## 11. A1, with three conversions measured

The consistent re-baseline method, applied to all three. **No C1 or C2
attribution was altered to make this cleaner** — the C1/C2 figures are exactly
those published in the re-baseline before Conversion 3 was run.

| conversion | shared | route-specific | hardening | cleanup | total | production commits |
|---|---|---|---|---|---|---|
| C1 `portfolio_summary` | 200 | 176 | 0 | 7 | **383** | 2 |
| C2 `period_movement` | 138 | 144 | 0 | 0 | **282** | 1 |
| C3 `geo_exposure` | **21** | **129** | 0 | 1 | **151** | 1 |
| **median** | **138** | **144** | 0 | 1 | **282** | **1** |

### The medians, and why they mislead here

A1 as written stops the migration when a later route needs more than `2 × m`
lines or `2 × c` commits. With three conversions those are finally computable:

| | median *m* | A1 trigger `2 × m` |
|---|---|---|
| total production lines | 282 | **564** |
| production commits | 1 | **2** |

**But the median of a decaying series is a poor summary, and saying so is more
useful than quoting it.** Shared cost went 200 → 138 → **21**: the median, 138,
describes no conversion and predicts none. The series has a shape, and the shape
is the finding:

```
shared           200  →  138  →   21      collapsing
route-specific   176  →  144  →  129      roughly constant
total            383  →  282  →  151      falling, driven by the shared term
```

### Status of the original A1 cost thesis

**Falsified as a level. Replaced by a supported convergence model as a shape.**

* **Falsified**: the closure report predicted 40–80 lines for `portfolio_summary`
  and 30–60 for the other two. Actuals were 383, 282 and 151 — 2–5× over even on
  the best conversion. That prediction is wrong and is not being re-written.
* **Replaced by a supported model**: cost = **shared + route-specific**, where
  shared decays to near zero once a route's semantic axes are bridged, and
  route-specific sits at **129–176 lines per route** across the three. Three observations fit
  it, and the third was a genuine out-of-sample prediction that landed.
* **Still unsupported**: any claim about a route that needs a **new contract
  bridge**. All three conversions so far used only axes that were already
  carried. §12 is the reason that matters.

## 12. Should Conversion 4 proceed?

**Yes — but the prediction has to change, and the honest reading is that
Conversion 4 tests something the first three did not.**

Every remaining compositional-core candidate needs an axis the plan layer does
**not** bridge:

| candidate | axis needed | bridged? |
|---|---|---|
| `funded_bridge` | `source_scope` + **`dimensions`** | scope yes, **dimensions no** |
| `evolution` | **`dataset`** (+ scope) | **no** |
| `temporal_compare` | **`dataset`** | **no** |
| `period_change_analysis` | **ranking subject and direction** | **not in the contract at all** |

The plan layer bridges **2 of the contract's 9 claim axes**. Conversions 1–3 all
drew on those two. So this result — however clean — says nothing yet about the
cost of a conversion that must build a bridge, and **Conversion 4 will be one**.

There is already one measurement of that cost: `span_from_claim`, the `time`
bridge Conversion 2 built, was **24 lines**. That is the number to test against.

**Recommended Conversion 4: `funded_bridge`.** It needs exactly one new bridge
(`dimensions` → the `GROUP` step), which makes it the cleanest available
measurement of bridge cost — the same reasoning that made `geo_exposure` the
cleanest measurement of reuse. `evolution` and `temporal_compare` both need the
`dataset` bridge *and* are shared with other routes, confounding the measurement.

### The pre-registration Conversion 4 needs

Written now so it cannot be fitted later:

| | proposed threshold | basis |
|---|---|---|
| shared | **≤ 45** | one bridge at ~24 (measured: `span_from_claim`) plus margin. **Not ≤ 40** — that budget assumed no new bridge, and pretending otherwise would make the test unfalsifiable |
| route-specific | **90–170** | `_route_bridge` is 80 lines, but it carries a rank residual and a second semantic read |
| total | **≤ 215** | derived |

**If shared cost for one bridge materially exceeds ~45, the recurring term is
larger than `span_from_claim` suggested, and the migration's remaining cost is
seven bridges rather than one.** That is the next real question, and it is not
the question this conversion answered.

## 13. Position after three conversions

```
compositional   3 of 15   portfolio_summary, period_movement, geo_exposure
specialist     12 of 15
```

All three share one plan module, one population step, one refusal path, one
narrowing owner, and take every semantic decision from the interpretation
contract. None of them reads the raw question.

**The architecture is working, the shared layer has converged, and the remaining
open question is the cost of the seven unbridged contract axes.**

## 14. Recommended next step

> **Pre-register Conversion 4 on `funded_bridge` with the §12 thresholds
> (shared ≤ 45, route-specific 90–170, total ≤ 215), committed before any
> production change, and run it as a measurement of BRIDGE cost.**

Carry three things into it:

1. The threshold **must** be raised from 40 to 45 and the reason stated in the
   pre-registration — this conversion needs a bridge and the previous three did
   not. Reusing 40 would be a test rigged to fail; ignoring the change would be
   a test rigged to pass.
2. `funded_bridge` reads its scope through `resolve_lens_with_default`, a
   **third** lens-resolution entry point that neither of the first three
   conversions touched. Confirm before switching that it resolves identically to
   `_resolve_lens`, or the population equivalence proof will be comparing two
   different populations.
3. The `dimensions` axis carries a `role` (`grouping` / `filter` /
   `unresolved`). The bridge must not collapse those — an unresolved dimension
   read as a grouping is exactly the silent-widening class of defect this
   programme exists to remove.
