# Loan-type abstraction and categorical population resolution

Base `3fd85d2`. Work: `9682da1`, `2594bd1`.

---

## 1. Inventory — the eight questions, answered before editing

1. **Does a canonical `loan_type` field already exist?** Yes. `type_of_loan` in
   `config/system/fields_registry.yaml`, `portfolio_type: common`,
   `layer: core` — already generic and cross-asset.
2. **Why was MI not using it?** It is not in the Business Semantics Registry,
   and it did not need to be: `erm_product_type` already carries the synonym
   **"loan type"**, so loan-type questions already reached the right field as a
   DIMENSION. What was missing was never a concept.
3. **What acts as its equivalent?** `erm_product_type`
   (`analytical_concept: product_mix`, `analytical_role: dimension`).
4. **Is `erm_product_type` data or semantics?** Both — a canonical field
   (`portfolio_type: equity_release`, `layer: product`) and a BSR concept.
5. **Where were `lump_sum` / `drawdown` recognised?** Nowhere as *values*. The
   only categorical value resolver was geography-shaped.
6. **Why did `drawdown` collide with Pipeline?** `mi_workflows/analytical/
   intent.py::_COMPLETION_TERMS` lists `" drawdown "` as a pipeline→funded FLOW
   event. In equity release it is also a loan type.
7. **Why did `lump_sum` disappear?** `_CATEGORICAL_FILTER_RE` requires a
   preposition. "how many lump sum loans do we have?" has none, so no filter was
   produced and nothing recorded that a category had been named.
8. **Why did the balance form resolve against geography?**
   `_parse_categorical_filter` ended `field = _preferred_region(...) or
   "geographic_region_obligor"` — **every** categorical value was bound to
   region, whatever had been named.

**Case A**: the generic concept exists. No `loan_type` concept was added, and no
`is_drawdown`, `erm_loan_type` or equity-release filter.

## 2. What changed

**The field comes from the value.** `execution_receipt.book_values(frame,
semantics)` publishes `{governed dimension field: {value: the book's spelling}}`
from the data itself, threaded to the parse exactly as `available_columns`
already was. Nothing holds a vocabulary; a value two fields both claim resolves
to neither; a value no field claims resolves to nothing rather than to
geography. With no catalogue (the routing parse) behaviour is unchanged.

**The attributive form.** A category used as an adjective before a population
noun now resolves — safe only because the value must be one the book carries.

**The qualifier rule.** A completion term standing in front of a population noun
is a qualifier, not an event. Generic, names no term: *"how many loans are we
completing at the moment?"* has no population noun after the term and is
unchanged.

## 3. The original blocker — closed

| question | expected | actual | receipt predicate |
|---|---|---|---|
| How many lump_sum loans do we have? | 396 | **396** | `Product Type = lump_sum` |
| How many drawdown loans do we have? | 244 | **244** | `Product Type = drawdown` |
| What is the balance for lump sum loans? | 396 | **396** | `Product Type = lump_sum` |

Generalises with no per-value code: `performing` 619, `owner occupied` 585,
`direct` 441, `London` 83, unfiltered 640.

## 4. Categorical sweep and supplement

```
SWEEP       69 questions, 8 governed fields
            47 CORRECT NARROWING · 14 HONEST REFUSAL · 4 UNCLEAR · 4 SILENT DROP

SUPPLEMENT  24 questions (frozen before execution)
            17 CORRECT · 5 SAFE REFUSAL · 2 WRONG/SILENT
```

**All six silent drops are one defect class.** A phrase that is both a governed
VALUE and a DIMENSION name, used attributively, is consumed as a grouping and
the population silently widens:

```
"What is the balance for owner occupied loans?"
  → balance grouped by Occupancy Type over 640 loans, no warning   (truth 585)
"How many owner occupied loans do we have?"
  → Occupancy Type = owner_occupied · 585 loans                    correct
```

Same constraint, different shape, opposite outcome — exactly the generalisation
gap the supplement exists to catch.

It is **reported, not fixed**. The fix belongs in role resolution — deciding
whether an attributive phrase is a grouping or a filter — and I no longer had
budget to change that layer *and* measure its blast across 882 questions.
The bounded fix is to route this case through the ambiguity guard that already
exists and already handles it correctly one shape along: *"How many broker loans
do we have?"* → **"I could not tell how you meant broker… I have not answered
over the whole book in the meantime."** That converts a silent widening into an
honest clarification, which the gate accepts.

## 5. Frozen bank and architecture

```
EXACT 66 (was 65)   DISCLOSED 2   TRUE REFUSAL 13   FALSE REFUSAL 10 (was 11)
WRONG / SILENT 0
```

All eleven protected commercial fixes verified live. Post-claim census **0**,
substitution detector **0 of 2**, canary intact, 68 guard tests green.

## 6. MI regression — one introduced failure, attributable

```
modules 278 → 278    passed 5957 → 5956    failed 81 → 82
failing names 85 → 86

INTRODUCED: mi_agent_api/tests/test_chat_routing_e2e.py::
            test_geographic_exposure_degrades_honestly_without_itl3_or_postcode
FIXED/REMOVED: none
```

**This is a real design conflict, and the guard is not wrongly formulated.** It
asserts, in its own comment, that *"the route still owns the answer and explains
why, rather than silently falling back"* — written to prevent a specialist
capability's failure being papered over by a weaker answer. The earlier
`geo_exposure` deferral does exactly what it forbids.

The two intents are both legitimate:

* the guard's — a specialist failure must not be hidden;
* the deferral's — a specialist failure must not block a correct generic answer.

The deferral's outcome is not silent: the reader receives *"Total Balance ·
grouped by Obligor Region (NUTS3) · 7 groups · 640 loans"*, a complete and
disclosed answer, where before they received a refusal on a book that carries
seven regions.

**The test was not edited and the change was not reverted.** Both would be
decisions taken to make a report look clean. This is an owner decision:
either accept the deferral and restate the invariant, or revert it and accept
two false refusals back.
