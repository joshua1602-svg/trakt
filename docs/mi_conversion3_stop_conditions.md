# Conversion 3 — `geo_exposure` — pre-registered stop conditions

**Committed before any production change.** These thresholds come from the
re-baseline (`docs/mi_migration_cost_rebaseline.md` §7) and are **not revised
after seeing the implementation**.

Base: `67981d5`. Working tree clean.

---

## Why this conversion exists

It is a **measurement**, not a capability delivery. It answers one question:

> Has the shared compositional layer become reusable enough that a route whose
> semantics are **already bridged** can migrate mostly through route-specific
> wiring?

`geo_exposure` was selected because it is the **only** remaining candidate whose
entire semantic input is already carried and bridged — one semantic read,
`source_scope`. Every other candidate would have to build a new contract bridge,
and a conversion that builds one cannot distinguish "the layer is converging"
from "the layer is still being built" — which is the ambiguity Conversion 2
ended in.

It is deliberately **not** the largest capability prize. `evolution` would retire
more duplication and `period_change_analysis` would unlock ranking; both were
declined for this slot on purpose.

## The measured history this tests

| | shared | route-specific | hardening | cleanup | total |
|---|---|---|---|---|---|
| C1 `portfolio_summary` | 200 | 176 | 0 | 7 | **383** |
| C2 `period_movement` | 138 | 144 | 0 | 0 | **282** |
| | −31% | −18% | | | −26% |

Both fell. Neither collapsed. A one-off shared cost would have gone to roughly
zero by C2; it did not.

---

## Thresholds

| # | condition | threshold |
|---|---|---|
| **S1** | **shared** production lines | **≤ 40** |
| **S2** | **route-specific** production lines | **90–150** |
| **S3** | **total** production lines | **≤ 190** |

**Declared expected landing zone, so a result near the threshold can still be
told apart from a result near the prediction: ~20 shared, ~110 route-specific,
~130 total.**

### Why 40 for shared

Every one of Conversion 2's 138 shared lines is **non-recurring for this
candidate**, item by item:

| C2 shared item | lines | recurs for `geo_exposure`? |
|---|---|---|
| module rename + docstring | 26 | no — `analytical_plan` is already route-neutral |
| `_population_step` extraction | 83 | no — it exists, and it is frame-agnostic |
| `span_from_claim` | 24 | no — `geo_exposure` has no time window |
| `COMPARE` id | 1 | no — not used by a point-in-time route |
| rename churn | 4 | no |

The only shared work this conversion can honestly need is a **frame-input entry
point** for the executor: `geo_exposure` is the first converted route that
receives a DataFrame rather than an output root. 40 lines buys exactly one such
generalisation and no more. It is ~29% of 138.

### Why 90–150 for route-specific

Built from Conversion 2's own two parts rather than its total — C2 spent **102**
on its plan builder and executor and **42** on the switch:

| | |
|---|---|
| observed | C1 **176**, C2 **144** — declining |
| handler sizes, measured | `_route_geo` **94** · `_route_period_movement` **170** · `_route_portfolio_summary` **116** |
| plan complexity | **4** primitives against C2's 5 — no period pair, no `compare`, no window |
| semantic reads to move | **1** against C2's 2 |

A simpler plan gives 70–85; a handler 45% smaller than `_route_period_movement`'s
gives 25–35 for the switch. That is **95–120**, and 90–150 is that estimate with
margin on both sides.

**The floor matters as much as the ceiling.** A figure far below 90 would suggest
the route was converted more shallowly than C1 and C2 were, which makes the
comparison meaningless in the other direction.

### Why ≤ 190 total

Derived (40 + 150), **not inherited from 240**. C2 was 282, so this needs a 33%
reduction. Reachable if convergence is real; missed if the shared layer is still
being rebuilt. It is not set where Conversion 3 passes automatically.

### S4–S8, carried forward unchanged

| # | condition |
|---|---|
| S4 | a **new primitive** is required |
| S5 | economics breach the **A2** tolerance (£0.005, or one unit of the measure) |
| S6 | payload/receipt equivalence needs a **bespoke `geo_exposure` exception** |
| S7 | any **silent drop**, or any **silent population widening** |
| S8 | any **unexplained regression**, by exact case/test name |
| S9 | a **generic semantic concept the contract does not carry** is required |

---

## Measurement method

The re-baseline's rule, applied hunk by hunk:

> **Is this production code reachable from more than one route at the end of
> Conversion 3?**

| class | |
|---|---|
| **shared** | generic compositional infrastructure reusable by more than one route |
| **route-specific** | required only for `geo_exposure` |
| **product hardening** | correctness work valuable regardless of composition |
| **cleanup** | removal of duplicate or dead semantic logic |

Tests, instruments and docs are **not** production-line cost and are recorded
separately.

**A net base→head diff cannot see logic added and then removed inside one
conversion, so `cleanup` is a floor rather than a measurement.** Recorded that
way in Conversion 2 and recorded that way again here.

---

## Known conditions carried in

**The `ALP Acquired Back Book` label collision is out of scope and must keep
failing closed.** The population parser reads "Back Book" inside the governed
label as a seasoning segment, so the answer refuses. It is recorded as a strict
`xfail` in `tests/test_portfolio_name_resolution.py`, it is route-independent,
and it refuses rather than returning a wrong number. **Do not make that case
answer by ignoring one of the competing interpretations.** If its behaviour
changes at all, stop and attribute it.

**Two owners of population narrowing are expected to surface.**
`chat_routing._apply_lens_filter(df, lens)` and
`evolution._scope_frame_lens(df, filters)` both narrow a frame to a portfolio
scope. They were read and compared during the re-baseline and they agree. After
this conversion **exactly one narrowing mechanism may be active for
`geo_exposure`**. `_apply_lens_filter` is **not** retired globally —
`period_change_route` still calls it, and this is not a consolidation exercise.

---

## Verdicts, defined now

**CONVERGENCE SUPPORTED** — all of: shared ≤ 40; route-specific within 90–150;
total ≤ 190; 0 new primitives; 0 new generic semantic concepts; economics,
payload and receipt equivalent; regressions clean.
*Means:* the common compositional layer is materially reusable rather than being
rebuilt route by route.

**CONVERGENCE NOT PROVEN** — total falls but **shared exceeds 40**, while the
conversion is otherwise technically clean.
*Means:* migration may still be getting cheaper, but the shared architecture has
not demonstrated convergence. **Do not proceed automatically to Conversion 4.**

**MIGRATION ECONOMICS FALSIFIED** — any of: shared near C2's 138; another
substantial generic abstraction required; total materially over 190; or the
supposedly cheapest reuse candidate still needs major shared-engine development.
*Means:* the architecture may work technically, but the migration method is not
converging economically. On this candidate that verdict is close to decisive,
because it is the one that needs no new bridge.

---

**Do not optimise the implementation to pass the threshold. Let the measured
result answer the question.**
