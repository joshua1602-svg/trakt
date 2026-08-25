# Predicate execution parity — one governed meaning for one Predicate

Base `34ccbba`. Steps 1–2 of the C6 filter binding are complete and pushed.
**C6 is not executed in this task**, and `SELECT_POPULATION(kind=row_predicates)`
is not wired.

---

## 1. The invariant this task exists to establish

> A governed `Predicate(field, op, value)` has one deterministic execution
> meaning everywhere in Trakt. For any frame and predicate, the shipped filter
> executor and the reusable population executor must either select the same
> rows, or fail in the same governed way. There must be no case where one
> narrows while another silently widens.

---

## 2. The two paths, and who owns each decision

Traced from the code, then **verified by execution** (§3) rather than from
docstrings.

| decision | `_apply_filters` (shipped) | `apply_population` (reusable) | same? |
|---|---|---|---|
| field lookup | `resolve_semantic_field`; falls back to the raw key if it is already a column; **raises** for an unknown key that is neither | `_canonical()` — a plain dict lookup defaulting to the key itself; never raises | **NO** |
| missing column | `_require_column` → **raises** `MIQueryExecutionError` | records `unavailable`, **leaves the frame alone** | **NO** |
| operator alias | `_OP_ALIASES` (`>`, `above`, `gte`, `greater_than_or_equal`, …), defaulting to `eq` | none — the raw op goes to `_apply_numeric_op`, which `KeyError`s → `unavailable` | **NO** |
| percent normalisation | `format == "percent"` and `percent_storage_scale(col) == PERCENT_FRACTION` → `/100`, for a scalar and for both bounds of a `between` | **none** | **NO** |
| numeric coercion | `_apply_numeric_op` → `coerce_numeric` | same, delegated | yes |
| date comparison | `_apply_numeric_op` | same, delegated | yes |
| categorical compare | `astype(str).strip().casefold()` | `astype("string").strip().lower()` | yes |
| **value-domain resolution** | `_resolve_domain_value(entry["value_domain"], …)` — "the South East" → the ITL values the column actually holds | **none** | **NO** |
| membership (`in`/`not_in`) | casefold `isin` | lower `isin` | yes |
| null handling | numeric branch `mask.fillna(False)` | plain mask | yes (verified) |
| `eq` dispatch | on the **spec value shape**: a dict goes to `_apply_numeric_op`, a bare string to the categorical branch | on the Predicate only — the shape is already erased | **NO** |

`apply_population`'s docstring claims it "[r]euses the executor's own comparison
semantics … so a route and the point-in-time path cannot disagree". It reuses
**the comparator**. It reuses neither the resolution nor the normalisation, and
the two paths disagree in five distinct ways.

---

## 3. Five divergence classes, each reproduced

A direct probe — the identical predicate handed to both executors on the same
frame — rather than inference from reading:

```
1  PERCENT           LTV gt 50            shipped 3 rows      reusable 0 rows
                     LTV between 40,60    shipped 1 row       reusable 0 rows
2  OPERATOR ALIAS    age op '>'           shipped 2 rows      reusable UNAVAILABLE, 5 rows (whole frame)
                     age op 'above'       shipped 2 rows      reusable UNAVAILABLE, 5 rows
                     age op 'gte'         shipped 2 rows      reusable UNAVAILABLE, 5 rows
3  UNAPPLIABLE       missing column       shipped RAISES      reusable UNAVAILABLE, 5 rows
                     unknown field key    shipped RAISES      reusable UNAVAILABLE, 5 rows
                     value 'abc' for gt   shipped RAISES      reusable UNAVAILABLE, 5 rows
4  VALUE DOMAIN      geo eq 'the South East'    shipped 2,420 rows   reusable 0 rows
                     geo eq 'Greater London'    shipped 1,380 rows   reusable 0 rows
5  eq SHAPE          {'op':'eq','value':'Joint'}  shipped RAISES    reusable 2 rows
                     'Joint'                      shipped 2 rows    reusable 2 rows
```

Classes 2, 4 and 5 were **not** visible in the 119-question census. Class 2 and
class 4 are the same silent-widening / silent-emptying shape as the two the
census did find, and they would have shipped inside C6 unnoticed.

Class 5 was the awkward one: shipped's meaning depends on the **spec value
shape**, which `material_predicates` erases. So shipped `_apply_filters` is not
a well-defined function of the Predicate at all — one Predicate, two shipped
answers, and the invariant in §1 was literally unstatable.

**Product semantics have since been ruled** (§6a), which settles it: the shape
carries no intended business meaning and is normalised away.

---

## 4. Percent field inventory

The rescale is not LTV-specific and must not be fixed LTV-specifically. From the
governed registry, on the real funded frame (11,035 rows):

```
current_interest_rate     percent_points      (no rescale — and it AGREES today, 2/2)
current_loan_to_value     percent_fraction    (rescale — 56 disagreements)
original_loan_to_value    percent_fraction    (rescale)
```

Three further `format: percent` fields exist in the registry but are not columns
on this book. The scale is decided **per column at execution time** by
`percent_storage_scale`, never from the field name, so a book that stores LTV in
points is handled by the same rule with no rescale.

---

## 5. `apply_population` consumer inventory

Required before touching shared failure semantics.

| consumer | row predicates? | source lens? | expects on `unavailable` | fail-closed change safe? |
|---|---|---|---|---|
| `mi_workflows.analytical.populations.apply` | yes | yes (caller-supplied `lens_filter`) | copies into `evidence["unavailable"]`; capabilities call `_population_unavailable` → **refuse** | yes — already guards `work is None` |
| `analytical.executors` portfolio_snapshot / population_profile / vintage_analysis | via `populations.apply` | yes | `_population_unavailable` → `STATUS_UNAVAILABLE` Finding | yes |
| `analytical.executors.period_movement` | yes, per snapshot | yes | collects across snapshots → **refuses** | yes, with a small edit |
| `mi_agent_api.mi_service._population_frame` | yes | no (lens is separate) | writes `metadata.populationApplied`; the frame is returned to the route regardless | needs care — see §7 |
| `execution_receipt.population_facets` | **no** — `material_predicates` only | no | n/a | unaffected |
| `execution_receipt.drill_population_facets` | no | no | n/a | unaffected |
| `execution_receipt.threshold_execution_proven` | no | no | any `unavailable` ⇒ not proven | unaffected |
| `execution_receipt.reconcile_population` | no | no | `UNAVAILABLE`, or `UNSUPPORTED` when the book cannot express the field | unaffected |
| `question_interpretation.projection._row_predicates` | no | no | n/a | unaffected |

Only **four** call sites execute predicates. Every one of them already stops on
`unavailable`; the exposure is that nothing *forces* them to.

---

## 6. Pre-registered authorised blast

**Authorised**

1. `apply_population` gains the shipped percent normalisation, operator aliases,
   field resolution and value-domain resolution — by **calling the shipped
   rule**, not by copying it.
2. A predicate the reusable path cannot execute stops being a silently widened
   frame.
3. `eq`/`ne` dispatch becomes a function of the **value type** (string →
   categorical + domain; numeric/date → `_apply_numeric_op`) rather than of the
   spec value shape — see §6a, where the product semantics are ruled.
   Pre-registered movement: a dict-shaped `eq` with a string value stops raising
   and starts matching. **Corpus exercise of that shape: 0 of 121 predicates**
   (76 `dict gt`, 7 `dict between`, 6 `dict ge`, 4 `dict lt`, 4 `dict le`, 24
   bare `str`), so no shipped question exercises it today.

**Not authorised**

new filter vocabulary · new field binding · changed Boolean semantics ·
multi-filter AND/OR redesign · Pipeline filter capability expansion · changing
what bare "over 50" binds to · changing duplicate same-field predicate collapse ·
changed dataset or population interpretation · C6 conversion or plan wiring.

**Preserved deliberately** (§13 of the brief)

- "over 50" continues to bind to `current_outstanding_balance`.
- "LTV above 50 and LTV below 80" continues to collapse to the last predicate,
  because `spec.filters` is a dict. `Predicate[]` could express both; that is
  not authorisation to change the shipped product mid-migration.

**Stop conditions**

- **STOP — PARITY BASELINE STALE** — 48/119, 56 percent, 15 absent, 0
  unexplained cannot be reproduced.
- **STOP — APPLY_POPULATION FAILURE CONTRACT PREREQUISITE** — fail-closed
  parity cannot be had without redesigning a shared consumer.
- **STOP — PREDICATE PARITY BLAST** — unrelated shipped behaviour moves.
- **STOP — PREDICATE SEMANTIC OWNER AMBIGUOUS** — no safe single owner exists.

---

## 6a. The Class 5 ruling, and its three boundaries

> Bare categorical borrower-type values and explicit equality predicates are
> semantically identical. `Joint` and `Single` in governed borrower context
> canonicalise to `borrower_type eq <value>`. The current input-shape
> distinction is not to be preserved in `Predicate`; it has no intended business
> meaning.

This is a **product decision**, not a compatibility workaround, and it is what
makes the §1 invariant statable at all: a Predicate can now have one meaning
because the two shapes were never two meanings.

The ruling came with three boundaries, and the implementation observes all
three:

1. **No shape or provenance state on `Predicate`.** It stays `field`, `op`,
   `value`. The two shapes produce an *equal* object, not merely an equivalent
   one — asserted by
   `test_the_predicate_carries_no_shape_or_provenance_state`.

2. **No joint/single recognition in the predicate executor.** The dispatch tests
   the OPERAND'S TYPE — `isinstance(value, str)` — and knows nothing about
   borrower types, or about any other product vocabulary.
   `test_the_executor_recognises_no_borrower_type_vocabulary` proves it by
   running the same predicate against a field named `anything_at_all` and
   getting the identical categorical execution.

3. **Field binding stays upstream.** Which field a phrase binds to is settled
   once by `llm_query_parser._filter_field_of`, before any executor sees it.
   `governed_predicate_mask` takes a `field_key` it never derives.

The normalisation therefore lives in exactly one place —
`population.predicate_of`, the one function that turns a spec filter entry into
a governed `Predicate` — and both executors read the result.

