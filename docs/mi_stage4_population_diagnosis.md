# Stage 4 — the 160-run regression, located

**Diagnosis only. No change made. The role split is not started.**

| | |
|---|---|
| Base | merge-base `4e051f3`; `4e051f3` ✓ and `28ece25` ✓ ancestors of HEAD |
| Method | ran the seasoning questions through `/mi/query`, deterministic arm, and called the acceptance check with the exact facet `32c263a` would have produced |

## The recorded hypothesis

> Those populations are resolved by intent rather than by an applied facet, so
> the facet reads as unapplied.

**Nearly right, and not precise enough to fix.** The population *is* applied and
the plan *does* declare it. The check simply cannot recognise the declaration.

## What actually happens

`32c263a` reclassified a dimension term with no grouping warrant from
`KIND_GROUPING` to `KIND_POPULATION`. Populations block unless
`_analytical_population_satisfies` accepts them, and that check matches the
facet's **field name** against the **literal text** of the predicates the plan
declares.

Measured on the alderbridge book:

| Question | Plan declares | Facet field | Check |
|---|---|---|---|
| Q7 *"…older vintages compare with the front book"* | `seasoning_segment = Front Book` | `seasoning_segment` | **accepted** |
| Q1 *"…the profile of our new lending changed…"* | **`months_on_book le 1`** | `seasoning_segment` | **rejected** |

Both populations were applied — Q1 narrowed 11,035 rows to 115. Q7 passes
because its predicate happens to name the same field the term resolves to. Q1
fails because **the governed definition of "new lending" is a months-on-book
bound, not a seasoning-segment value**, and the check has no way to know those
are the same population.

That is the 160-run regression, in one line of reasoning.

## Why this matters for the fix

The hypothesis as recorded would send the fix to the wrong place — towards
making intent-resolved populations exempt, which would weaken the guard for
every population, not only these.

The precise defect is narrower and safer to close: **the acceptance check
compares literal field names where it should compare governed populations.**
"New lending" and `months_on_book le 1` are the same population by
configuration — `lending_windows.recent_max_months` — and the check should
resolve the concept before matching, exactly as the registry resolves a synonym
before matching a field.

## What this predicts, and how to test it

If the diagnosis is right, extending `_analytical_population_satisfies` to
resolve a seasoning concept to its governed predicate should make the
`32c263a`-shaped facet accepted for Q1 **without touching any other facet
kind**, and the role split can then proceed with the seasoning families unmoved.

The test is already available and named: the deterministic robustness arm
reports Q1, Q7 and Q8 separately, by name, on both books. The baseline is
**20 / 20 `CORRECT`**. Any attempt that moves it has reproduced the regression,
and any attempt that holds it has not.

## Not done

* The acceptance check is unchanged.
* `KIND_GROUPING` is unchanged; no role split has been made.
* Nothing in this document has been applied.
