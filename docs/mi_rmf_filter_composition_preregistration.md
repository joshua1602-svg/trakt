# Ranked movement + filter — pre-registration

**Written and committed BEFORE the production edit.** Base `8400a8c`. Nothing
here is adjusted afterwards; a breach is reported as a breach.

---

## 1. The defect, reproduced from execution

Traced through parse → binding → projection. The before-state, measured:

| | question | `spec.metric` | `spec.filters` |
|---|---|---|---|
| works | `balance by region for loans with LTV above 50%` | `current_outstanding_balance` | `{current_loan_to_value: gt 50.0}` |
| **broken** | `Which region added the most balance since last month for loans with LTV above 50%?` | **`current_loan_to_value`** | **`{}`** |
| works | `Which region added the most balance for loans with LTV above 50%?` | `current_outstanding_balance` | `{current_loan_to_value: gt 50.0}` |
| **broken** | `How did balance change since last month for loans with LTV above 50%?` | **`current_loan_to_value`** | **`{}`** |
| **broken** | `…since last month for loans with borrower age above 70?` | **`youngest_borrower_age`** | **`{}`** |

Two facts the prior report did not establish, both from execution:

1. **Ranking is irrelevant.** Removing "since last month" fixes it; removing the
   ranking does not. The trigger is a **temporal clause**.
2. **The field family is irrelevant.** Borrower age fails identically, so this
   is not an LTV quirk.

## 2. The single composition fault

**Classification: parser ordering, compounded by subject assembly — one site.**

`llm_query_parser._compare_recognizer` short-circuits the deterministic parse.
It builds a `MIQuerySpec` directly and:

* never calls `_parse_filters`, so **no filter reaches the spec at all**;
* calls `_detect_metric(q, …)` over the **whole question**, filter clause
  included, so "LTV above 50%" makes LTV the measure.

Downstream is innocent. `projection` faithfully carries what the parse gave it,
which is why the contract shows `subject=current_loan_to_value` and
`row_predicates=[]` — a correct projection of a wrong parse.

## 3. The fix, and why it is not a second resolver

`_parse_filters` already accepts a `spans` argument that returns
`{field_key: (start, end)}` — the offsets of the clause each filter came from.
The recogniser will:

1. call **the existing** `_parse_filters` and attach its result to the spec;
2. mask the returned clause spans out of the text before **the existing**
   `_detect_metric` runs.

No new resolver, no new vocabulary, no field-specific handling, and **no change
to global parse order** — one recogniser is made to do the two steps the main
path already does, with the same helpers.

## 4. Pre-registered blast

### Expected to change

Only questions that are **all three** of: temporally recognised
(`_compare_recognizer` fires), carrying a governed row-predicate clause, and
currently losing it. Measured over the 882-question corpus **before** the edit:
questions where `_compare_recognizer` fires **and** a filter clause is present.

### Expected NOT to change — byte-identical

* unfiltered ranked movement, unfiltered unranked movement;
* ranked level, filtered level (the working path);
* every C1–C6 converted surface, delivered economics included;
* pipeline / stage / funnel, source-portfolio scope, seasoning, geography;
* honest missing-measure and missing-period refusals;
* D1, D2, D4 fixes; the LEVEL/MOVEMENT owner; ordering direction/basis/limit;
* every canary case unrelated to filtered movement.

### Prohibited

* touching `period_change_route.py` — the measurement shows the fault is **not**
  there, so editing it would exceed the target defect;
* changing `_parse_filters`, `_detect_metric`, or global parse order;
* expanding `spec.filters` from dict-collapse to multi-predicate semantics;
* any field-specific branch.

## 5. Negative control, registered

A question where LTV is genuinely the measure must keep it:

```
"How did LTV change since last month?"          -> metric stays current_loan_to_value
"Which region has the highest average LTV?"     -> metric stays current_loan_to_value
```

The fix must not mechanically demote every LTV mention to a predicate.

## 6. Mutation controls, registered

The RM3 proof must FAIL if any of these is applied, and each must change an
absolute expected outcome rather than an old/new agreement:

drop the RowPredicateClaim · replace the predicate field with the grouping
field · replace the measure with the predicate field · drop the grouping
dimension · flip MOVEMENT to LEVEL · reverse the comparison periods · reverse
the ranking direction.

## 7. Stop conditions

* **STOP — BLAST EXCEEDS TARGET DEFECT** if any unrelated semantic owner needs
  changing.
* **STOP — PARSE-ORDER BLAST RISK** if global parse order must change.
* **STOP — FIX IS NOT LOCAL** if production cost implies a new subsystem.
* **STOP — ZERO-BLAST CONDITION BREACHED** on any unexplained movement.

## 8. Recorded, out of scope

`"For loans with LTV above 50%, balance by region"` parses to
`metric=current_loan_to_value` with `filters={current_outstanding_balance: gt
50.0}` — the predicate binds to the **wrong field** when the filter clause
LEADS. A separate defect in the leading-clause path, not this one, and not fixed
here.
