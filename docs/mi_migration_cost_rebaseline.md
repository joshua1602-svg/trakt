# Migration cost — re-baseline before Conversion 3

**Measurement and pre-registration only. Nothing was implemented, and Conversion
3 was not started.**

The question this document exists to answer is not "can Conversion 3 be made to
fit". It is:

> What result from Conversion 3 would actually prove that the shared
> compositional layer is **converging**, rather than being **rebuilt route by
> route**?

That test is written below, before the result is known.

---

## 1. Evidence read

| source | what was taken from it |
|---|---|
| `docs/mi_conversion1_report.md` | C1's own figures: 383 lines, "plan layer ~150 raw / ~95 code", "switch 94", 4 commits |
| `docs/mi_conversion2_report.md` | C2's figures: 282 lines, 138 shared / 144 route-specific, 5 commits |
| `docs/mi_migration_abort_conditions.md` | A1–A5 as written, including what A1 actually says |
| `docs/mi_target_state_contract_closure.md` | the ranked migration order and the **40–80 / 30–60 / 30–60** line predictions |
| `docs/mi_conversion1_stop_conditions.md`, `docs/mi_conversion2_stop_conditions.md` | the 240 cap, registered twice |
| the actual commits and diffs | every figure below is re-derived from `git diff`, not copied from a report |

**The headline line counts were not taken on trust.** Both were re-measured, and
both reproduce:

```
C1  a56b7eb..f56bd35   production only   = 383   ✓ matches the C1 report
C2  f56bd35..97557ae   production only   = 282   ✓ matches the C2 report
```

C1's base is `a56b7eb` (the target-state closure report), not `44bc90c` — the
three commits between them are the closure task's own contract work (`schema.py`
+77, `projection.py` +56), which belong to the closure phase and not to any
conversion. Measured from `44bc90c` the figure would be 548, and attributing 137
lines of pre-conversion contract work to Conversion 1 would flatter neither the
history nor the trend.

---

## 2. One classification method, applied to both conversions

Every changed production hunk in both conversions was classified by a single
rule, applied mechanically:

| | class | rule |
|---|---|---|
| **A** | shared compositional infrastructure | the code's consumers are **not tied to one route** — plan/step types, primitive ids, the population step, contract→request bridges, the interpretation provider, plan-reading helpers |
| **B** | route-specific conversion | the plan builder and executor **for one route**, and that route's switch and registration |
| **C** | product hardening | correctness work that would have been valuable **without** composition |
| **D** | cleanup / deletion | removal of a duplicate semantic owner or of dead route-specific logic |

The rule is deliberately checkable after the fact: *is this code reachable from
more than one route at the end of Conversion 2?* That is why, for example,
`build_plan`'s population block is classified **A** even though Conversion 1
wrote it inline for one route — Conversion 2 extracted it verbatim into
`_population_step` and both routes now reach it. The classification was fixed
before the totals were computed and was not adjusted afterwards.

### Result

| conversion | A shared | B route-specific | C hardening | D cleanup | total |
|---|---|---|---|---|---|
| **C1 `portfolio_summary`** | **200** | **176** | 0 | 7 | **383** |
| **C2 `period_movement`** | **138** | **144** | 0 | 0 | **282** |

Per file:

```
CONVERSION 1                                          A     B     C     D
  portfolio_summary_plan.py       (280 new)          173   107     -     -
  chat_routing.py                  (80)               11    69     -     -
  movement_summary.py              (23)               16     -     -     7
                                                     200   176     0     7  = 383

CONVERSION 2                                          A     B     C     D
  analytical_plan.py              (212)              110   102     -     -
  chat_routing.py                  (46)                4    42     -     -
  period_request.py                (24)               24     -     -     -
                                                     138   144     0     0  = 282
```

### Where this disagrees with Conversion 1's own report, and why

C1's report recorded "the switch itself: **94**". Under the uniform rule C1's
route-specific cost is **176**. Both are correct measurements of different
things: C1 counted the `chat_routing` switch, and excluded
`portfolio_summary`'s own plan builder and executor because at the time they
were simply "the plan layer". The uniform rule puts a route's plan builder and
executor in **B**, exactly as it does for `period_movement` — otherwise C2's 102
lines of `build_period_movement_plan` would be compared against a C1 figure that
excluded the equivalent code, and the trend would be an artefact of the
accounting.

The 383 total is unchanged either way. **Only the split moves, and it moves
because it is now measured the same way on both sides.**

### Uncertainty, stated

* **`C` is 0 in both conversions, and that is a real finding rather than a
  definitional convenience.** Neither conversion contained correctness work that
  stood on its own. The one candidate — C1's fix to the interpretation
  provider (`Index or []` raising, 10 lines) — is classified **A** because the
  code it repairs exists only for composition and had no product value before it.
* **`D` is near-invisible in a net diff, by construction.** Both totals are
  base→head diffs, so logic added and removed inside one conversion nets to
  zero. C1's removal of the `_resolve_lens` fall-through (commit `6b5c7db`) does
  not appear as a deletion in the net diff because the fall-through never
  existed at `a56b7eb`. The 7 lines of D that do appear are the inline cohort
  calculation `movement_summary.portfolio_summary` stopped owning. **Treat D as
  a floor, not a measurement.**
* Hunk-level attribution inside `chat_routing.py` and the plan modules is a
  judgement at the boundary of a handful of docstring lines. Moving every
  contestable line to the other column changes each total by well under 10.

### What the consistent split actually shows

```
                 C1     C2     change
shared          200    138      −31%
route-specific  176    144      −18%
total           383    282      −26%
```

**Both are falling. Neither is collapsing.** Shared cost did not behave like a
one-off — a one-off would have gone to roughly zero — but it also did not stay
flat. That is the fact the rest of this document has to explain and then turn
into a test.

---

## 3. What Conversion 2's 138 shared lines actually did

| # | change | lines | what it generalised | foreseeable from closure? | C1 deferred it? | C3 reuses unchanged? | analogous work still visible? | class |
|---|---|---|---|---|---|---|---|---|
| 1 | module renamed `portfolio_summary_plan` → `analytical_plan`, docstring rewritten | 26 | the module's **identity**: one route → many | yes, trivially | **yes, deliberately** — it named the module after one route | yes | no | **ONE-OFF** |
| 2 | `_population_step` extracted out of `build_plan` | 83 | the EMPTY / UNRESOLVABLE / FILLED **population decision**, into one owner | yes — closure named source scope as a carried concept | **yes** — written inline for one route | yes, and it is frame-agnostic | **yes** — `resolve_measure`, `group` and `rank` are still built inline inside each plan builder | **ONE-OFF item, RECURRING pattern** |
| 3 | `period_request.span_from_claim` | 24 | contract `TimeClaim` → route `SpanRequest`: the **first bridge for a second contract axis** | the *field* yes (closure added `window_periods`); the *bridge* no | n/a — C1 needed no window | only if C3 needs a window; **`geo_exposure` does not** | **yes, and it is measurable — see below** | **RECURRING PATTERN** |
| 4 | `COMPARE` primitive id declared | 1 | the sixth of the seven ids | yes | yes | yes | one left (`project`) | **ONE-OFF** |
| 5 | `chat_routing` module-rename churn | 4 | — | yes | yes | yes | no | **ONE-OFF** |

**ONE-OFF: 114 lines. RECURRING PATTERN: 24 lines.**

### The recurring pattern, measured rather than asserted

The plan layer reads the interpretation contract in exactly two places:

```
analytical_plan.py:127,158,218,340   source_scope   → _population_step
analytical_plan.py:331,332,336       time           → span_periods
```

The contract carries **nine** claim axes:

```
operation   subject   dimensions   filters   time
target      population   source_scope   dataset
```

**2 of 9 are bridged. Seven are not.** Every future conversion whose route reads
`dimensions` (a grouping), `filters`, `operation` (a ranking) or `dataset` will
need a bridge of the same kind `span_from_claim` was — because the contract
field existing is not the same as the plan layer being able to consume it.
Conversion 2 discovered that distinction the expensive way.

So the honest classification of the 138 is neither "one-off" nor "recurring":

* **114 lines were one-off** and are genuinely spent. The module is now
  route-neutral, the population step has one owner, six of seven primitive ids
  exist. None of that recurs.
* **24 lines were the first instance of a pattern that has seven more
  opportunities to recur** — one per unbridged contract axis.

**That is what makes Conversion 3 a real experiment rather than a formality.**
`geo_exposure` needs **no unbridged axis at all**, so if shared cost does not
fall sharply on this conversion, the recurring cost is not the bridges — it is
something structural that will not decay, and the migration method is wrong.

---

## 4. Inspection of the remaining candidates

Read from production code. **Nothing was implemented.**

What each remaining compositional-core route reads from the raw question, taken
from the AST rather than from the reports:

| route | semantic reads at the handler | contract axes required | bridged today? |
|---|---|---|---|
| **`geo_exposure`** | `_resolve_lens` only | `source_scope` | **yes — all of it** |
| `funded_bridge` | `resolve_lens_with_default`, `_bridge_dimension(spec, semantics)` | `source_scope` + **`dimensions`** | scope yes, **dimensions no** |
| `evolution` | `_dataset_for` | **`dataset`** (+ scope) | **no** |
| `temporal_compare` | `_dataset_for` | **`dataset`** | **no** |
| `period_change_analysis` | `resolve_rank_intent` | + **ranking subject and direction** | **not in the contract at all** |

### `geo_exposure`, in detail

| | |
|---|---|
| handler | `chat_routing._route_geo`, **94 lines** — against `_route_period_movement`'s 170 and `_route_portfolio_summary`'s 116 |
| semantic reads from the question | **one** — `_resolve_lens(question, source_lens)` |
| everything else | fixed by the route's identity: ITL3 is the grouping, top-15 is a display constant, the ordering comes from `geo.exposure_by_itl3` |
| primitives required | `select_population`, `resolve_measure` (balance, count), `group` (ITL3), `rank` (desc) — **all four exist, none new** |
| plan/result types required | `Plan`, `Step`, `lens_filters`, `lens_label`, `_population_step` — **all exist and none touch a frame shape** |
| contract fields required | `source_scope` — **already bridged** |
| payload/receipt | `_envelope(..., reconciliation, source_notes, lens_applied=True)`, the same call shape the converted routes use; **no route literal to generalise** |
| **generic infrastructure visibly missing** | **one, and it is small**: every existing plan executor takes `(output_root, client_id, to_run_id)` and calls `evolution.funded_frames`. `_route_geo` receives a `frame_resolver` and works on a single DataFrame, so the executor needs a frame-input entry point. The shared pieces it would call — `_population_step`, `lens_filters`, `lens_label` — are already frame-agnostic. |
| **duplicate semantic owner it will expose** | **one, named**: `chat_routing._apply_lens_filter(df, lens)` and `evolution._scope_frame_lens(df, filters)` are two owners of "narrow a frame to a portfolio scope". Both were read and compared: they agree (membership match, case- and whitespace-insensitive) and **there is no divergence on this book**. `_apply_lens_filter` has a second caller in `period_change_route`, so retiring it entirely is not Conversion 3's job — removing geo's use of it is. |
| route-specific wiring expected | a plan builder, a frame-input executor, and the switch in `_route_geo` plus its registration |

### Live check of the candidate

`geo_exposure` was exercised against the governed book. It routes, it honours a
scope, and it refuses an unheld name:

```
"What is the geographic exposure?"                 geo_exposure  ok  Westminster £83.4m (4.2% of the book)
"Geographic exposure for the acquired book"        geo_exposure  ok  Westminster £26.3m (4.5% of Acquired)
"Geographic exposure for the direct book"          geo_exposure  ok  Westminster £57.1m (4.1% of Direct)
"...for the Highgate Mortgages Book"               geo_exposure  REFUSED — not a governed portfolio
```

**No live wrong-number defect was found.** The Phase 1E unresolved-scope guard
is route-independent (it lives in `mi_service`, wrapping both the routed answer
and the fallback), so `geo_exposure` is already covered by it without ever
having been converted — which is itself evidence that the governance layer
generalises (A3).

One known defect was re-observed and is **not** new: the governed display label
"ALP Acquired Back Book" refuses on **every** route, because the population
parser reads "Back Book" inside the label as a seasoning segment. It is recorded
as a strict `xfail` in `tests/test_portfolio_name_resolution.py` with the fix
named (registry-aware span masking). It **fails closed** — a refusal, never a
wrong number — and it is not geo-specific.

**It should not pre-empt Conversion 3.** It is a parser-precedence defect in the
interpretation layer, orthogonal to whether the plan layer converges, and
converting a route neither fixes nor worsens it.

---

## 5. Recommended Conversion 3

# `geo_exposure`

**Next best: `funded_bridge`.**

The closure report also ranked `geo_exposure` third, but for a weaker reason
("complete contract, low governance"). The reason now is sharper and is the
reason it should be chosen:

> `geo_exposure` is the **only** remaining candidate whose entire semantic input
> is already bridged into the plan layer.

That is exactly what makes it the cleanest test of reuse. Every other candidate
would need a new contract bridge, and a conversion that builds a bridge cannot
distinguish "the layer is converging" from "the layer is still being built" —
which is precisely the ambiguity Conversion 2 ended in. Choosing
`funded_bridge` or `evolution` now would guarantee an uninterpretable result.

It is also **not** the largest capability prize. `evolution` is shared by three
routes and would retire more duplication; `period_change_analysis` would unlock
ranking. Both were rejected for this slot on purpose: this conversion is being
spent on a measurement, not on capability.

---

## 6. A1, re-baselined

### The historical record, unchanged

**Original conversion-cost prediction: FALSIFIED by Conversions 1 and 2.**

| route | predicted (closure report §9) | actual | ratio |
|---|---|---|---|
| `portfolio_summary` | 40–80 lines, 2–3 commits | **383 lines, 4 commits** | **4.8× – 9.6×** |
| `period_movement` | 30–60 lines, 1–2 commits | **282 lines, 5 commits** | **4.7× – 9.4×** |
| `geo_exposure` | 30–60 lines, 1–2 commits | not yet run | — |

The two ratios are strikingly close. Whatever the estimate was missing, it was
missing it by a consistent factor rather than at random — which is a further
reason to think the cost is structural and therefore measurable, rather than
noise.

**Nothing above is rewritten.** The two `STOP — COST ASSUMPTION BREACHED`
verdicts stand, both against a 240 cap that was registered before the work and
is not being retro-fitted now.

### One thing that is often conflated, stated precisely

**A1 has not fired.** A1 stops the migration "when a subsequent conversion
exceeds `2 × m`, where *m* is the per-route median of the first three
conversions". Only two conversions exist, so *m* is undefined and A1 is **not
yet measurable**. The two breaches were against the **per-conversion 240 cap**,
which is a conversion-level stop condition, not A1. Both mechanisms are real and
they are not the same mechanism.

### The revised measurement framework

A1 tracked one number. It should track four, because they behave differently and
only some of them scale with the number of routes:

| tracked per conversion | why separately |
|---|---|
| **shared production lines** | should **decay** if the layer converges; flat means it is being rebuilt |
| **route-specific production lines** | should be roughly **constant per route** — this is what a conversion actually costs |
| **total production lines** | the affordability figure, and the only one comparable to the historical record |
| **commits to equivalence** | independent evidence of how many attempts equivalence took |

Recorded but **not** counted as production-line cost: test files, instruments,
and documents.

**No three-route median is computed here.** Only two conversions exist; A1's
threshold stays undefined until Conversion 3 supplies the third observation.
That is the point of running it.

---

## 7. Pre-registered Conversion 3 hypothesis

**Committed before Conversion 3 begins. These numbers are not to be revised
afterwards.**

### S1 — shared cost: **≤ 40 production lines**

Justified from §3 and §4, item by item, rather than chosen as "lower":

| Conversion 2 shared item | lines | does it recur for `geo_exposure`? |
|---|---|---|
| module rename + docstring | 26 | **no** — `analytical_plan` is already route-neutral |
| `_population_step` extraction | 83 | **no** — it exists, and it is frame-agnostic |
| `span_from_claim` | 24 | **no** — `geo_exposure` has no time window |
| `COMPARE` id | 1 | **no** — not used by a point-in-time route |
| rename churn | 4 | **no** |

On the evidence, **every one of C2's 138 shared lines is non-recurring for this
specific candidate.** The only shared work this conversion can honestly need is
a frame-input entry point for the executor, because `geo_exposure` is the first
converted route to receive a DataFrame rather than an output root.

**40 lines buys exactly one such generalisation and no more.** It is ~29% of
138. The *prediction* is that the actual figure lands **under 20**; 40 is the
threshold, set with room for one genuine surprise, and deliberately too tight to
absorb a third module-scale reshaping.

### S2 — route-specific cost: **90–150 production lines**

| basis | |
|---|---|
| observed | C1 **176**, C2 **144** — declining |
| handler size, measured | `_route_geo` **94** lines · `_route_period_movement` **170** · `_route_portfolio_summary` **116** |
| plan complexity | **4** primitives, against C2's 5 — no period pair, no `compare`, no window |
| semantic reads to move | **1**, against C2's 2 |
| payload/receipt adapters | already exist; the envelope call shape is unchanged |

Building the estimate from C2's own two parts rather than from the total: C2
spent **102** on its plan builder and executor and **42** on the switch. Geo's
plan is strictly simpler (one frame, no period pair, no `compare`), so 70–85 is
the like-for-like expectation; its handler is 45% smaller than
`_route_period_movement`'s, so 25–35 for the switch. That gives **95–120**, and
the registered range of **90–150** is that estimate with a margin on the
upside.

A simpler route than C2 with one fewer semantic input should not cost more than
C2's 144. The range floor of 90 matters as much as the ceiling: **a figure far
below 90 would suggest the route was converted more shallowly than C1 and C2
were**, which would make the comparison meaningless in the other direction.

### S3 — total: **≤ 190 production lines**

Derived (40 + 150), **not inherited from 240**. C2 was 282, so this requires a
33% reduction. It is reachable if convergence is real — the expected landing
zone is ~20 shared + ~110 route-specific = **~130** — and it is missed if the
shared layer is still being rebuilt. It is not set where Conversion 3 passes
automatically.

### S4–S8 — carried forward unchanged from Conversion 2

new primitive required · A2 economic tolerance (£0.005) · a bespoke
`geo_exposure` exception in payload/receipt · any silent drop · any silent
population widening · any unexplained regression · a generic semantic concept
the contract does not carry.

---

## 8. What each Conversion 3 outcome will mean

Written now, so the result cannot be interpreted to taste.

### CONVERGENCE SUPPORTED

**All** of:

* shared ≤ **40**;
* route-specific within **90–150**;
* total ≤ **190**;
* **0** new primitives;
* **0** new generic semantic concepts;
* equivalence clean — economics, payload, receipt, silent drops 0, introduced
  failing names 0.

**Means:** the compositional layer is becoming reusable. Three observations
would then exist, A1's median *m* becomes computable for the first time, and the
migration can be costed rather than hoped at. It would **not** mean the
migration is cheap in absolute terms — 190 is still far above the closure
report's 30–60 — only that the *marginal* cost is falling and is dominated by
route-specific work, which is the shape a converging migration has.

### CONVERGENCE NOT PROVEN

Total falls below 190 **but shared cost exceeds 40**.

**Means:** migration is getting cheaper, but generic infrastructure is still
being extended route by route. **Do not proceed automatically to Conversion 4.**
The required next step is to identify what the shared lines bought and whether
*that* is now finished — the same diagnostic as §3, which will by then have three
data points and can distinguish decay from a plateau.

### MIGRATION ECONOMICS FALSIFIED

**Any** of:

* shared cost is near or above **138**;
* another substantial generic abstraction is required — a new contract bridge,
  a new primitive, or a second module-scale restructuring;
* total exceeds **190**.

**Means:** the compositional architecture may still be technically sound — A2–A5
would say so — but **the current migration method is not converging
economically**. On `geo_exposure` specifically this verdict is close to
decisive, because it is the candidate that needs no new bridge: if the cheapest
possible conversion still costs shared lines, the recurring cost is structural
and will not decay with more conversions. Stop and re-evaluate the migration
strategy — including the possibility that the remaining routes should stay
specialist.

---

## 9. A1–A5 status

| | condition | status | evidence |
|---|---|---|---|
| **A1** | cost explosion | **NOT YET MEASURABLE**, and **SUPERSEDED BY RE-BASELINE** for how it is measured | A1's threshold is `2 × m` over the first **three** conversions; only two exist, so *m* is undefined and A1 cannot have fired. Its *prediction* (40–80 / 30–60) is **FALSIFIED**. The two `STOP` verdicts were against the per-conversion **240 cap**, a different and also-real mechanism, and both stand on the record. From Conversion 3, A1 is measured on the four-part split in §6. |
| **A2** | failure to reconcile | **NOT FIRED** | C1: 54 rendered pairs, 0 differences. C2: 36 pairs, 7,633 envelope leaf fields, 0 differences. The £0.005 tolerance was not approached on any field in either conversion. |
| **A3** | governance cannot generalise | **NOT FIRED — strengthened** | The unresolved-scope guard is route-independent and already protects `geo_exposure`, an **unconverted** route (§4). Duplicate owners fell: `_resolve_lens` callers 5 → 4, `requested_span` production callers 1 → 0. |
| **A4** | interpretation ownership cannot be made singular | **NOT FIRED** | Neither plan builder takes a question parameter, enforced over the AST in both conversions' tests. `build_period_movement_plan` params: `interpretation, region_column, has_portfolio_column`. |
| **A5** | unattributable regression | **NOT FIRED** | All five registered surfaces identical **by name** after both conversions; silent drops **0**; C2 additionally ran the full 10,373-test repository and attributed every failing name. |

A2–A5 are unchanged. Nothing in this re-baseline required altering them.

---

## 10. Recommended next task

> **Run Conversion 3 on `geo_exposure` against the thresholds pre-registered in
> §7, as a measurement.**

Commit §7's thresholds to a `docs/mi_conversion3_stop_conditions.md` before any
production change, in the same form as Conversions 1 and 2 — including the
declared expected landing zone (~20 shared, ~110 route-specific, ~130 total), so
that a result near the threshold can still be told apart from a result near the
prediction.

Two things to carry into it, both from §4 and neither of them optional:

1. `geo_exposure` will be the first converted route that receives a **DataFrame**
   rather than an output root. Whatever frame-input entry point that needs is
   **shared** cost and must be counted as such, not quietly booked as route
   wiring.
2. It will expose `_apply_lens_filter` and `_scope_frame_lens` as **two owners
   of one narrowing**. They currently agree, so this is a cleanup and not a
   defect. Removing geo's use of the duplicate belongs in Conversion 3, in its
   own commit and only with the same four-part proof Conversion 1 used;
   retiring the function outright does **not**, because `period_change_route`
   still calls it.

**Do not begin Conversion 3 until §7 is committed.** A threshold written after
the code is a description, not a test.
