# Conversion 4 — `funded_bridge` — pre-registered stop conditions

**Committed before any production change. No production code was changed.**

> **Conversion 4 did not proceed.** It stopped at §4 —
> `docs/mi_conversion4_stop_contract_prerequisite.md`. These thresholds stand
> unchanged for when the named prerequisite is closed, and must not be revised
> in the meantime.

Base: `5daf451`. Working tree clean.

---

## The experiment

Conversions 1–3 all drew on the **two** contract axes that were already bridged
(`source_scope`, `time`). Conversion 3 measured what a conversion costs when
nothing new must be built: **21 shared / 129 route-specific / 151 total**.

Conversion 4 is a different experiment. It asks:

> Can one new governed semantic axis be connected to the compositional engine at
> a small, predictable incremental cost, without reopening the architecture?

## Thresholds

| # | condition | threshold |
|---|---|---|
| **S1** | **shared** production lines | **≤ 45** |
| **S2** | **route-specific** production lines | **90–170** |
| **S3** | **total** production lines | **≤ 215** |

### Why 45 and not 40

Conversion 3's threshold was ≤ 40 *because it needed no new bridge*, and it came
in at 21. Conversion 4 needs one. The only measured bridge cost in this
programme is `period_request.span_from_claim`, the `time` bridge Conversion 2
built: **24 lines**.

`21 (the C3 shared floor) + 24 (one measured bridge) = 45.`

Reusing 40 would be a test rigged to fail. Ignoring the change and setting it
high would be a test rigged to pass. **45 is the C3 floor plus exactly one
bridge at its measured cost, with no margin beyond that.**

### Why 90–170 route-specific

`_route_bridge` is 80 lines — smaller than `_route_period_movement` (170) and
`_route_geo` (94) — but it carries **two** semantic reads plus a rank residual,
where `geo_exposure` carried one. Observed: C1 176, C2 144, C3 129.

### Why ≤ 215

Derived (45 + 170).

### S4–S9, carried forward

| # | condition |
|---|---|
| S4 | a **new primitive** is required |
| S5 | economics breach the **A2** tolerance (£0.005) |
| S6 | payload/receipt equivalence needs a **bespoke `funded_bridge` exception** |
| S7 | any **silent drop** or **silent population widening** |
| S8 | any **unexplained regression**, by exact case/test name |
| S9 | **more than one** unbridged contract axis is required, or a generic semantic concept the contract does not carry |

## Measurement method

Unchanged from the re-baseline: *is this production code reachable from more
than one route at the end of Conversion 4?* → **shared** / **route-specific** /
**product hardening** / **cleanup**. Tests, instruments and docs are not
production-line cost.

`cleanup` remains a **floor**, not a measurement: a net base→head diff cannot see
logic added and then removed inside one conversion.

## Verdicts, defined now

**BRIDGE CONVERGENCE SUPPORTED** — shared ≤ 45; route-specific 90–170; total
≤ 215; exactly one generic bridge added; no new primitive; no new generic
semantic concept beyond the expected axis; **dimensions preserve their role**;
economics, payload and receipt equivalent; regressions clean.
*Means:* the architecture can absorb a new semantic axis at controlled
incremental cost.

**CONVERGENCE NOT PROVEN** — technically clean, but shared > 45, or more generic
infrastructure than the one expected bridge. **Do not proceed automatically to
C5.**

**MIGRATION ECONOMICS FALSIFIED** — shared returns materially toward C2's 138;
more than one substantial generic abstraction unexpectedly required; total
materially > 215; or one axis forces significant architecture redesign.

---

## Conditions carried in

* The `ALP Acquired Back Book` label collision stays **out of scope** and must
  keep failing closed.
* `resolve_lens_with_default` may be a third population entry point. If safe,
  make it unreachable from `funded_bridge` and pin it — but **do not** retire it
  globally; `mi_agent_workflow` and `_route_cohort_progression` still call it.
* **A requested dimension must keep its grouping role** through
  interpretation → contract → bridge → plan → execution. If the contract cannot
  preserve it, that is a **contract prerequisite** and must not be patched
  inside route-specific logic.

**Do not optimise the implementation to pass the threshold.**
