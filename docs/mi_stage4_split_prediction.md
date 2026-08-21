# Stage 4 role split — pre-registered prediction

**Written and committed BEFORE the split is implemented and before anything is
measured.** Recorded so the result can be judged against a statement made in
advance rather than explained afterwards.

| | |
|---|---|
| Base | merge-base `4e051f3`; `4e051f3` ✓ and `28ece25` ✓ ancestors of HEAD |
| B5 before the split | **unreachable** — 83 population facets built, 83 carry a field key, **0** labels omit it |

---

## What the split does

A `KIND_GROUPING` facet is reclassified to `KIND_POPULATION` **only where the
parser put that field in `spec.filters`** — that is, only where the object's
`dimensions[].role` is `filter`.

Reclassification happens at **reconciliation**, not at detection.
`detect_requested_facets` runs before any parsing decision and must keep doing
so; the spec is available at reconcile, and the spec is what carries the slot
assignment.

## Where this deliberately differs from `32c263a`

`32c263a` assigned a dimension term to `GROUPING` only where the question's
words justified it — a grouping clause, a superlative, a coverage question —
and let **everything else fall to POPULATION**, "the side that blocks". That
over-assignment is what cost 160 runs.

**Here, only a positively-identified filter moves. A dimension with no role from
any source stays exactly as it is.** On the 939-question corpus the object
records 663 `grouping`, 15 `filter`, 55 `unresolved`; the 55 do not move.

Conservatism in the opposite direction from `32c263a`: it fell to the blocking
side when unsure, this stays put when unsure.

---

## The prediction

### Must move

* Facets for a dimension the parser resolved into `spec.filters` — the object's
  `role == filter` — change kind from `grouping_dimension` to `row_population`,
  **and their labels are rebuilt in the population form** so the field name is
  present.

### Must not move

* Facets for a dimension the parser resolved into `spec.dimension` /
  `spec.dimensions` — `role == grouping`. These stay `grouping_dimension`.
* Facets for a dimension no source assigned a role to — `role == unresolved`.
  **These stay `grouping_dimension`.** Moving them is the `32c263a` failure.
* **Every other facet kind.** `threshold`, `geographic_scope`,
  `comparison_period`, `ranking`, `projection`, `requested_statistic`, `share`,
  `multi_measure`, `unresolved_measure`, `relationship`,
  `aggregate_contribution`, `stress_scenario`, `cohort_comparison` — unchanged
  in kind, status and count.

### Must hold

* **The seasoning families are unmoved**, measured by name, per book:
  Q1 4/4, Q7 4/4, Q8 12/12 — **20/20 `CORRECT`** on both books.
* **Answer text byte-identical**, 340/340.
* Calibration bank 260 passed; lexical decisions 690/690.
* **B5 stays unreachable**: 0 population facets whose label omits its field.
* **`POPULATION` is not over-assigned.** The count of `row_population` facets
  rises by exactly the number of `role == filter` dimensions and by no more.

---

## What stops the work

Any of these, per the standing conditions:

1. A facet **outside the pre-registered set** moves — stop and report, do not
   absorb.
2. The split would produce a population label that **omits its field name** —
   stop and fix B5 first. A guard that accepts a front-book facet against a
   declared `Back Book` must not be latent while labels are moving.
3. The seasoning families move at all.
4. Answer text moves.

---

## Test shape, carried over

The pattern that proved Q7's coincidence was gone is kept: **three tests
establishing the property, and a fourth proving the check can still fail.** For
the split that means proving a filter-role dimension moves, a grouping-role one
does not, an unresolved one does not — and a fourth showing the reclassifier
does not simply move everything.
