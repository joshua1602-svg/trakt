# C7 targeted fix — ranked movement + filter contract composition

Base `8400a8c`. Pre-registration `fe1580c`, written and committed **before** the
production edit; nothing in it was adjusted afterwards. This report records what
was measured, including where a measurement first came back wrong.

---

## 1. The fault, located to one site

`llm_query_parser._compare_recognizer` short-circuited the deterministic parse:
it built its `MIQuerySpec` directly, never called `_parse_filters`, and ran
`_detect_metric` over the **whole** question. So a comparison carrying a
governed predicate lost the predicate *and* took the predicate's field as the
measure.

Two properties, both established from execution rather than from reading:

* **the trigger is the temporal clause, not the ranking** — removing "since
  last month" fixed it; removing the ranking did not;
* **the field family is irrelevant** — borrower age failed identically to LTV.

Downstream is innocent. `projection` carried exactly what the parse produced, so
the contract's `subject=current_loan_to_value, row_predicates=[]` was a correct
projection of a wrong parse. `period_change_route.py` was **not** modified.

## 2. The fix

Inside `_compare_recognizer` only, using the two helpers the main deterministic
path already uses, in the order it already uses them:

1. call **the existing** `_parse_filters`, asking for the `spans` it already
   returns, and attach its result to the spec;
2. mask those clause spans out of the text before **the existing**
   `_detect_metric` runs.

No second resolver, no new vocabulary, no field-specific branch, no change to
global parse order.

### The one correction the zero-blast sweep forced

The first sweep after the edit reported **one** changed corpus contract, and it
was a regression, not a target:

```
Q: Show funded balance evolution from October to November.
     metric  'current_outstanding_balance' -> None
```

`_parse_filters`, run on the lowercased text, resolves the period phrase as a
categorical geography filter — `{collateral_geography: "October To November"}` —
whose clause span covers the word *balance*, so masking it deleted the measure.

The correction is generic and names no field: **where a filter was resolved from
text the period detection has already claimed, the period detection wins.** The
recogniser has already resolved its periods at that point; `_parse_filters` has
not. One corpus question was affected, and it is back to identical.

## 3. Blast

### Zero-blast, 882 corpus questions, both trees swept

`migration_phase0.contract_plan_delta` run in the working tree and in a clean
worktree at `8400a8c`, diffing every spec and contract field per question:

```
CONTRACTS_CHANGED 0        FIELDS {}
```

Denominator: 882 distinct Stage 1 + Stage 2 questions, all swept in both trees.

### Intended blast, the target set

The defect questions are not in the corpus, so a zero there proves no collateral
movement but nothing about the fix working. Measured separately, same method:

| question | changed | what moved |
|---|---|---|
| `…added the most balance since last month for loans with LTV above 50%?` | **yes** | measure `LTV → balance`; predicate `{} → LTV gt 50` |
| `…since last month for loans with borrower age above 70?` | **yes** | measure `age → balance`; predicate `{} → age gt 70` |
| `How did balance change since last month for loans with LTV above 50%?` | **yes** | measure `LTV → balance`; predicate `{} → LTV gt 50` |
| `Which region added the most balance since last month?` | no | unfiltered ranked movement untouched |
| `How did LTV change since last month?` | no | negative control — LTV stays the measure |
| `Which region has the highest average LTV?` | no | negative control |
| `balance by region for loans over 50` | no | load-bearing bare threshold preserved |
| `Show funded balance evolution from October to November.` | no | the regression above, corrected |

**3 of 8 changed, and they are exactly the three defect targets.**

## 4. Contract agreement tests

`tests/test_ranked_movement_filter_composition.py` — 7 tests, all passing. Each
asserts **every channel at once** (measure, dimension + alternates,
`ordering_of == movement`, comparison periods, direction, basis, predicate),
because the defect was one channel stealing another's content while both looked
individually plausible. RM-F2 uses a different predicate family and RM-F3 drops
the ranking, so neither the field nor the ranking is load-bearing in the fix.

## 5. RM3 executed non-vacuously

`Which region added the most balance since last month for loans with borrower
age above 70?`, executed through the plan primitives with
`period_change_route` not imported:

* population genuinely narrowed: `76 → 49` and `80 → 47` rows;
* 3 populated groups; grouping bound from the alternate
  (`region → geographic_region_obligor`);
* movement `South East +1,453,508.40`, `London +555,175.86`,
  `Scotland −1,917,494.65`;
* ranked `[South East, London]`, Scotland recorded in `excludedGroups`;
* receipt complete and chronological, per-group reconciliation true.

### Mutation controls

All **seven registered in section 6 of the pre-registration** discriminate, and
the plan restores to the base result afterwards:

```
DIFFERS  1 drop the RowPredicateClaim
DIFFERS  2 replace the predicate field with the grouping field
DIFFERS  3 replace the measure with the predicate field
DIFFERS  4 drop the grouping dimension
DIFFERS  5 flip MOVEMENT to LEVEL
DIFFERS  6 reverse the comparison periods
DIFFERS  7 reverse the ranking direction
DISCRIMINATING 7/7      RESTORED: True
```

Three supplementary controls (flip the operator, move the threshold, change the
ordering basis) also discriminate.

## 6. Preservation

19 modules run **module by module in both trees** — the working tree and a clean
worktree at `8400a8c` — with a per-test timeout.

```
baseline failing test names: 9      after failing test names: 9
INTRODUCED:      (none)
FIXED/REMOVED:   (none)
```

Pass counts are identical module by module. The 9 pre-existing failures are
carried in the denominator, not removed from it:

* `tests/test_conversion2_period_movement.py` — 5 (population-from-governed-ids
  and four movement figures)
* `mi_agent/tests/test_p0_execution_receipt.py` — 3 (unavailable-dimension
  disclosure)
* `mi_agent/tests/test_mi_predicate_extraction.py` — 1
  (`test_complex_query_executes_all_filters`)

`tests/test_assurance_measurement_failure.py` times out **identically in both
trees** and is reported as a timeout, not as a pass.
`tests/test_ranked_movement_filter_composition.py` (7 passed) exists only in the
working tree and is therefore not in the baseline denominator.

## 7. Canary and audit

* `tests/test_compound_canary_bank.py` — 11 passed in **both** trees; grades and
  breach count have not moved and no frozen observation was edited.
* Executed canary: **0 invariant breaches**, no `DROPPED` element. 21 UNEVIDENCED
  elements across 9 cases and 5 unexercised families — unchanged, and recorded as
  evidence gaps rather than as breaches.
* `migration_phase0.c7_independent_audit` — **10 of 10** checks pass, including
  "the route does not read the raw question for meaning" and "no implicit period
  or measure".

## 8. Cost

| | added | deleted |
|---|---|---|
| production (`mi_agent/llm_query_parser.py`, one function) | 51 | 2 |
| — of which executable code | 22 | 2 |
| — of which commentary | 29 | 0 |
| tests (`tests/test_ranked_movement_filter_composition.py`) | 140 | 0 |
| instrument hardening (`migration_phase0/c7_target_plan_proof.py`) | 16 | 0 |
| documentation (pre-registration + this report) | 118 + this | 0 |

No contract or projection change was needed: the schema already carried the
predicate slot and the ordering fields. The whole fix is **shared composition**
inside one recogniser — 22 lines of code in one function, no new subsystem.
