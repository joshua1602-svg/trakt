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

## 4. Categorical sweep and supplement — both gates met

```
SWEEP       69 questions, 8 governed categorical fields
            49 CORRECT NARROWING · 16 HONEST REFUSAL · 4 UNCLEAR · 0 SILENT DROP

SUPPLEMENT  24 questions (frozen before execution)
            19 CORRECT · 5 SAFE REFUSAL · 0 WRONG/SILENT
```

### The defect the first pass left open, and how it closed

Dimension terms come from registry SYNONYMS, and a field's synonyms routinely
spell its own values — "owner occupied" is both the wording of `occupancy_type`
and one of its two values. Matched as a dimension, *"what is the balance for
owner occupied loans?"* was answered as a breakdown over all 640 loans for a
question about 585, while *"how many owner occupied loans do we have?"* narrowed
correctly. **The same constraint, two shapes, opposite outcomes** — which is the
generalisation gap the supplement exists to catch.

Two rules closed it, neither naming a field or a value:

* a term that names a value the book carries, and does **not** stand after a
  grouping marker, is a qualifier — so the categorical resolver claims it as a
  predicate rather than the parser claiming it as an axis;
* a group segment is cut at its own qualifier, so *"by region for owner occupied
  loans"* is one axis narrowed to a value rather than two axes. The same cut is
  what makes *"by region in London"* a filtered regional breakdown.

```
What is the balance for owner occupied loans?  → Occupancy Type = owner_occupied · 585
Balance by region for owner occupied loans.    → owner_occupied · grouped by region, 7 groups
Balance by region for drawdown loans.          → Product Type = drawdown · grouped by region
Show balance by occupancy type.                → grouped by Occupancy Type      (unchanged)
Balance by region and broker channel.          → heatmap, 28 groups             (unchanged)
What is the balance for broker loans?          → "I could not tell how you meant broker…"
```

### The four UNCLEAR, examined

Not silent drops — every one narrows and says so. One is a real defect:

* **`"How many Gamma Direct loans do we have?"` → 104**, applying *both*
  `Broker = Gamma Direct` and `Source Portfolio in direct_001`, where the broker
  alone is 147. A governed SCOPE term matched inside a categorical VALUE. It is
  disclosed in the receipt, so not silent, but it is a wrong number for a plainly
  worded question. **P1** — fixing it means touching portfolio-lens scope
  resolution, which reaches well beyond this change.
* `"How many / what is the balance for / balance by region for **direct** loans"`
  resolve to the direct BOOK (441) rather than `origination_channel = direct`
  (146). Genuinely ambiguous, governed, and disclosed.

## 5. Frozen bank and architecture

```
EXACT 64   DISCLOSED 2   TRUE REFUSAL 13   FALSE REFUSAL 12   WRONG / SILENT 0
```

All eleven protected commercial fixes verified live and green. Post-claim census
**0**, substitution detector **0 of 2**, canary intact, 68 guard tests passing.

EXACT is 64 rather than the 66 an intermediate pass reached, because the
`geo_exposure` deferral was reverted — see §6.

## 6. MI regression — clean

```
modules 278 → 278    passed 5957 → 5957    failed 81 → 81
skipped 711 → 711    xfailed 15 → 15       errors 4 → 4    timeouts 1 → 1
failing names 85 → 85

INTRODUCED: 0        FIXED/REMOVED: 0
```

### The geo_exposure deferral, reverted

An intermediate pass had `geo_exposure` defer when it could not build an ITL3
view, so the generic path could answer *"which region has the largest balance?"*
— which it does, completely and with disclosure. That introduced the run's only
MI failure, against
`test_geographic_exposure_degrades_honestly_without_itl3_or_postcode`, whose own
comment states the invariant: *"the route still owns the answer and explains why,
rather than silently falling back."*

I tried to keep both intents by deferring only where the contract carries a
resolvable measure. It does not separate them — *"Where is the book concentrated
geographically?"* carries a measure too. So the deferral was **reverted**: the
guard is not wrongly formulated, and §17 rules out opportunistically fixing a
false refusal this change did not cause. The cost is two false refusals back,
recorded rather than traded away.

A correction to an earlier report in this series:
`test_cumulative_cohort_conversion_routes` was flagged as a possible second
regression. It is not — it appears in the frozen baseline's 85 failing names.

## 7. UX

Loan-type answers surface the scalar cleanly in the KPI (`Loan 396`,
`£105.4MM`) and the receipt reads `Product Type = lump_sum`. No `field op value`
expression leaks. It uses the tape's own spelling (`lump_sum`, not "Lump sum"),
which is cosmetic — nothing misstates the answer.
