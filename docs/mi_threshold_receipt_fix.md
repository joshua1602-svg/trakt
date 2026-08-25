# KIND_THRESHOLD receipt defect — fixed from execution evidence

Base `470a92a` → pre-registration `d6e4a63` → fix `2ce9bbc`.
**Production diff: 1 file, 66 added / 3 deleted.** C6 not executed.

---

## 1. The defect, reproduced on the live book

`alderbridge`, three governed periods. The route narrows correctly inside each
period, publishes the narrowing, and the answer refuses.

| case | `spec.filters` | ledger `applied` | rowsAfter | outcome |
|---|---|---|---|---|
| `…for loans above 50% LTV` | `current_loan_to_value gt 50.0` | `current_loan_to_value (applied within each period)` | 1889 | **ok=False** |
| `…for borrowers over 75` | `youngest_borrower_age gt 75.0` | `youngest_borrower_age (…)` | 2722 | **ok=False** |
| `…for loans above 200000` | `current_outstanding_balance gt 200000.0` | `current_outstanding_balance (…)` | 3666 | **ok=False** |
| `…by month` (control) | `{}` | — | — | ok=True |

```
I understood that you asked for LTV over 50, but that could not be applied to
the calculation (LTV over 50 — this governed capability does not apply a value
threshold, so the figure is not restricted to it).
```

filter executed correctly **+** execution evidence exists **+** receipt ignores
it **=** false refusal.

## 2. Why the two objects cannot be matched

| | field | operator | value |
|---|---|---|---|
| `KIND_THRESHOLD` facet (comparator form) | **no** — `field_key=None` | only inside `label` | only inside `label` |
| `populationApplied.applied` | yes | **no** | **no** |
| `spec.filters` → `material_predicates` | yes | yes | yes |

The threshold facet is detected from the question text; the ledger reports the
field alone. Matching them would mean parsing `"LTV over 50"` — the receipt
reading the question back to itself, which is exactly what `_ROUTE_GRANULARITY`
and the P1K silent errors warn against.

The codebase already recorded that the threshold facet and the population facet
are twins for one predicate: *"…all of them numeric bounds — 'LTV above 50%' —
which `KIND_THRESHOLD` already represents. Two facets for one predicate is the
duplicate-claim defect."* Measured 1:1 on every probe, including a
two-threshold question.

## 3. The evidence rule

`threshold_execution_proven(envelope, semantics, threshold_count)`, derived from
an **executor invariant** rather than from a match:

> `_apply_filters` applies **every** `spec.filters` entry or raises
> `_require_column`, and appends each field it narrowed on **after** the column
> is confirmed and the mask is built.

So the threshold is proven when, and only when:

1. `metadata.populationApplied` exists and is a mapping;
2. `unavailable` is empty;
3. `applied` is non-empty;
4. the spec carries at least one governed material predicate;
5. `len(predicates) >= threshold_count`;
6. **every** predicate's field appears in `applied`.

Anything else → LOST, with the original reason unchanged. The function is never
given the facet — its parameters are `(envelope, semantics, threshold_count)` —
so it *cannot* read wording, and a test asserts that structurally.

## 4. Delivered filtered evolution, pinned per period

Non-vacuous: every period differs, in both row count and value.

| question | 2026-04 | 2026-05 | 2026-06 | rows/period |
|---|---|---|---|---|
| balance, LTV > 50% | £432,425,355.79 | £450,969,362.11 | £472,527,483.38 | 1721 · 1799 · 1889 |
| balance, borrower age > 75 | £565,452,027.47 | £575,304,529.17 | £588,411,793.07 | 2648 · 2682 · 2722 |
| balance, loan > £200k | £1,031,317,551.54 | £1,047,465,870.63 | £1,064,930,912.22 | 3555 · 3610 · 3666 |
| **loan count**, LTV > 50% | 1721 | 1799 | 1889 | 1721 · 1799 · 1889 |
| unfiltered (control) | £1,932,310,991.20 | £1,946,827,440.60 | £1,964,886,258.21 | — |

**Economics unchanged.** The count series equals the per-period row counts the
ledger already declared *before* the fix (1721/1799/1889), and the ledger's
`rowsAfter` is identical on both sides. No calculation file is in the diff —
`_apply_filters`, `_filtered_funded_evo` and the route are untouched.

## 5. Negative controls — fail-closed

| control | result |
|---|---|
| threshold requested, no `populationApplied` | LOST |
| evidence proves a **different** field | LOST |
| `unavailable` non-empty | LOST |
| spec carries no material predicate | LOST |
| two thresholds, one predicate | **LOST for both** |
| two predicates, only one applied | LOST |
| no threshold requested | untouched |
| geographic scope (`…for London`) | still refuses — a different facet and owner |

## 6. Mutation tests

| | mutation | result |
|---|---|---|
| A | real ledger replaced by a fabricated non-empty one | **8 failed** |
| B | stop matching the predicate field | **3 failed** |
| C | drop the `unavailable` / count guards | **4 failed** |
| D | restore unconditional threshold **success** | **5 failed** |

**Mutation D initially failed to fire — and that is the most useful thing this
task produced.** With sixteen tests green, replacing the branch condition with
`if True` broke nothing, because every negative control exercised the helper
directly and none went through `reconcile_routed_facets`. Six branch-level
controls were added; D now fails five of them. A control that cannot see the
branch it guards is not a control.

Mutation A also refused to apply at first: its anchor matched **two** lines
(the pre-existing `population_applied` and the new helper). The assertion caught
the ambiguity rather than editing the wrong one.

## 7. Blast radius

**Owned evolution surface** (34 owned):

| | before | after |
|---|---|---|
| DELIVERED | 18 | **19** |
| REFUSED | 16 | **15** |

Exactly **one** movement:

```
REFUSED -> DELIVERED  [evolution|funded]  "balance trend where LTV above 50%"
```

Requested threshold `current_loan_to_value gt 50.0`; evidence
`current_loan_to_value` applied within each period, `unavailable` empty; old
facet status LOST (unconditional), new status APPLIED (evidence-backed).

**882-question corpus census:**

```
dataset changes         0
route changes           0
answer/refusal changes  1   <- the same question
row-count changes       1   <- the same question
answer-text changes     1   <- the same question
```

UNEXPLAINED movements = **0**. Interpretation and measure are untouched by
construction: the diff contains one file and no interpretation, parser or
measure code.

## 8. Cost

| bucket | raw lines |
|---|---|
| **production defect fix** (`mi_agent/execution_receipt.py`) | 66 added, 3 deleted = **69** |
| tests (`test_threshold_receipt_evidence.py`) | 262 |
| docs (pre-registration + this report) | separate |

Classified **product hardening / live defect remediation**. **Not** C6 conversion
cost and must not enter C6's formal threshold.

## 9. Regression by exact name

Both sides in the **same tree** — only `mi_agent/execution_receipt.py` reverted
for the baseline. A worktree baseline was tried first and discarded: it came back
739 skips against 36, exactly the not-like-for-like failure the P0 audit had
already recorded, and repeating that method cost a wasted 45-minute run.

```
baseline: 203 failed, 10289 passed, 36 skipped, 16 xfailed, 28 errors
after   : 203 failed, 10311 passed, 36 skipped, 16 xfailed, 28 errors

INTRODUCED failing names : 0
FIXED failing names      : 0
UNCHANGED pre-existing   : 203
silent drops             : 0  (identical skip and error counts)
```

The +22 passes are the new suite. No pre-existing guard or test was amended —
nothing asserted the old unconditional LOST, so nothing had to be re-blessed.

## 10. Refreshed four-part filter status

```
representation     : FAIL  - FilterClaim carries operator and value but no FIELD;
                             the field↔bound binding is still unresolved
owner agreement    : NOT MEASURABLE - cannot compare a binding that does not exist
plan consumable    : FAIL  - evolution still calls _apply_filters directly;
                             analytical_plan.lens_filters expresses only
                             source_portfolio_id, 0 of the 119 filtered questions
delivered coverage : PASS  - four delivered filtered evolution cases with real
                             per-period movement, pinned above
```

One leg moved from FAIL to PASS. Three remain.
