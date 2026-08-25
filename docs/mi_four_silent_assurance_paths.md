# Closing four silent `except` → empty-result assurance paths

Base `b7fc518`. **Zero production files changed.** C6 not resumed.

The P0 loader audit closed one shape of silent assurance failure. It named four
instruments carrying a second: a broad `except` that converts a crashed
**measurement** into an empty result, so the run continues and reports a number.

---

## 1. Before-state inventory

| instrument | exception caught | empty fallback | legitimate empty possible? | risk |
|---|---|---|---|---|
| `contract_role_census.py:99` | `Exception` on parse + projection | row `{question, error}` with **no `dimensions` key** | **yes** — 275 of 645 questions genuinely name no dimension | `_diff` reads `.get("dimensions", [])`, so an un-measured question is indistinguishable from one with no dimensions |
| `equivalence_portfolio_summary.py:123` | `Exception` on `build_registry` | `registry = None` | `build_registry` *returning* None is legitimate; **raising is not** | falls back to "the pre-1G reading" that the same comment calls meaningless, and still prints 0 differences |
| `filter_ownership_trace.py:146` | `Exception` on corpus parse | `continue` — question vanishes | a question with no filters is legitimate; a **skip** is not | corpus denominator shrinks silently |
| `route_ownership_evolution.py:124` | `Exception` on `/mi/query` | row with `route=None` | a REFUSED answer is legitimate; an exception is not | `route=None` ⇒ `owned` False ⇒ dropped from the owned-surface denominator |

Measured in normal operation: **zero** errors at site 1 (645 rows) and **zero**
parse failures at site 3 (882 questions). So an exception at these sites is not
data — the comment at site 1 claiming "a parse fault is data, not a stop" is
contradicted by the corpus it runs on.

## 2. Reproduced silent degradation

Each fault injected at runtime; every instrument exited **0** and looked healthy.

**Site 1** — every parse faulted:
```
645 questions projected
rows written: 645     rows that are ERRORS: 645
error row yields dimensions -> []
questions compared          : 645
questions with any delta    : 0
ILLEGAL deltas (blast)      : 0
"Every delta is a bridge question ... Nothing else moved."
```
A census where **100% of measurements failed** printed a full denominator and a
clean blast result.

**Site 2** — `build_registry` faulted:
```
cases on the surface     : 9
economic differences     : 0
cases the plan BLOCKS    : 0 -> []
```
No mention that the registry was missing.

**Site 3** — every corpus parse faulted:
```
corpus questions carrying spec.filters: 0
of these, expressible by lens_filters (source_portfolio_id only): 0
```
The second line is **identical** to the real C6 finding. Total measurement
failure was indistinguishable from the conclusion it was meant to establish.

**Site 4** — every query faulted:
```
OWNED BY THE EVOLUTION FAMILY: 0
all owned   0   0   0   0
```
Indistinguishable from a route family that owns nothing.

## 3. The distinction enforced

Added to `assurance_semantics.py`, the existing assurance-control home:

```python
class AssuranceError(RuntimeError)            # this run cannot be trusted
class AssuranceSemanticsError(AssuranceError) # (P0, unchanged)
class AssuranceMeasurementError(AssuranceError)
def measurement_failed(instrument, case, exc) -> AssuranceMeasurementError
```

> measurement ran, found nothing → an empty result, which is evidence
> measurement could not run → an exception, which is not

Every site raises `... from exc`, so the root cause survives; a test asserts
that at all four (`node.cause is not None`).

## 4. Remediation per site

| site | change | legitimate empty preserved |
|---|---|---|
| `contract_role_census` | parse/projection fault → `measurement_failed`; `_diff` refuses any census carrying error rows, and refuses an empty census | `dimensions: []` and `filters: []` still recorded — 275 and 538 rows |
| `equivalence_portfolio_summary` | `build_registry` fault → `measurement_failed`; the import moves out of the `try` so only the call is guarded | `build_registry` *returning* None still flows through |
| `filter_ownership_trace` | corpus parse fault → `measurement_failed`; prints and asserts the examined denominator | a question with `spec.filters == {}` still counts as examined |
| `route_ownership_evolution` | query fault → `measurement_failed`; asserts one reading per corpus question | REFUSED / EMPTY grades still count |

Exception handling is narrowed by **scope**, not by type: the guarded region is
now only the call that can fail, and anything it raises becomes an explicit
assurance failure rather than an empty value. No site had a *specific* expected
exception to catch — normal operation produces none.

## 5. Mutation tests

| instrument | mutation | expected | actual |
|---|---|---|---|
| `contract_role_census` | `parse_with_repair` raises | assurance failure | `AssuranceMeasurementError` naming the question |
| `equivalence_portfolio_summary` | `build_registry` raises | assurance failure | `AssuranceMeasurementError` naming case `A1` |
| `filter_ownership_trace` | corpus `ParsedQuestion.parse` raises | assurance failure | `AssuranceMeasurementError` naming the question |
| `route_ownership_evolution` | every `/mi/query` raises | assurance failure | `AssuranceMeasurementError` naming the question |

All four restored; tree clean.

## 6. Normal runs — before vs after

| instrument | before | after |
|---|---|---|
| `contract_role_census` | 645 rows, `_diff` 645 compared, 0 illegal | **identical**, plus 0 error rows asserted |
| `equivalence_portfolio_summary` | 9 cases, 2 unclaimed, 0 economic differences, 0 blocked | **identical** |
| `filter_ownership_trace` | "119 carrying spec.filters", 0 expressible | **identical**, plus "corpus questions examined: 882" |
| `route_ownership_evolution` | 34 owned, 18 delivered, 16 refused | **identical** |

## 7. Historical claims, from actual references

| instrument | prior claim(s) — verified by doc reference | recomputed | conclusion changed? |
|---|---|---|---|
| `contract_role_census` | `mi_contract_role_fix_report.md` — contract role/claim evidence | 645 compared, 0 illegal deltas | **UNCHANGED** |
| `equivalence_portfolio_summary` | `mi_phase0_report.md`, `mi_phase1g_report.md` — C1 portfolio-summary equivalence | 9 cases, 0 economic differences | **UNCHANGED** |
| `filter_ownership_trace` | `mi_c6_filter_prerequisite.md` — C6 filter ownership | 119 of 882, 0 expressible | **UNCHANGED** |
| `route_ownership_evolution` | `mi_pipeline_grain_defect_fix.md` — C6 evolution owned surface | 34 owned, 18 delivered | **UNCHANGED** |

All four **UNCHANGED**. No `STOP — MIGRATION ASSURANCE BASELINE INVALID`.

## 8. Denominator assertions added

Minimum needed to stop the known shape resurfacing:

- `filter_ownership_trace` — corpus non-empty, and the examined count printed
  alongside the filtered count.
- `route_ownership_evolution` — one reading per corpus question, else fail.
- `contract_role_census._diff` — no error rows, census non-empty.
- `equivalence_portfolio_summary` — no new assertion; its surface count and
  unclaimed list were already printed, and the registry fault is now fatal.

No repository-wide "no broad except" rule was added. The previous audit's guard
took three iterations because textual rules fire on things that merely resemble
the anti-pattern; this task touched exactly the four measured paths.

## 9. Cost

| bucket | raw lines |
|---|---|
| `assurance_semantics.py` (shared vocabulary) | +39 −1 = 40 |
| `contract_role_census.py` | +26 −3 = 29 |
| `equivalence_portfolio_summary.py` | +15 −5 = 20 |
| `filter_ownership_trace.py` | +20 −3 = 23 |
| `route_ownership_evolution.py` | +19 −4 = 23 |
| **assurance tooling total** | **135** |
| tests (`test_assurance_measurement_failure.py`) | 205 |
| docs (this file) | separate |
| **production** | **0** |

Assurance methodology work. **Not** part of C6 conversion or normalised migration
cost.

## 10. Regression

```
297 passed, 1 xfailed, 0 failed
production files changed: 0
```

Covering the two assurance suites, the dataset-ownership guard, contract target
state, portfolio-summary prerequisites, scope resolution, the source-scope claim,
the Pipeline Stage contract and the evolution grain tests. A full estate run was
not required: zero production files changed, so no product answer or refusal can
move.

## 11. Status

# FOUR SILENT ASSURANCE PATHS CLOSED

All four remediated; unexpected measurement failures now raise an explicit
`AssuranceMeasurementError` naming instrument, case and root cause; genuine empty
results remain distinguishable and are asserted to still work; mutation tests
fire at all four; all four historical claims re-run **UNCHANGED**; zero
production files changed.

**Recommended next task:** return to the C6 blocker on a now-trustworthy
instrument base — close the unconditional `KIND_THRESHOLD` refusal in
`reconcile_routed_facets`, which stamps every routed threshold facet LOST without
consulting the `populationApplied` ledger the route already publishes. It is the
reason no filtered evolution question can deliver, it is a single owner with the
fix pattern ten lines away in the same function, and `filter_ownership_trace` can
now be trusted to measure the result.
