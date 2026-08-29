# C7 — the live ranked-movement evidence path

Base `1c49e61`. This closes the evidence path only. No interpretation, contract
semantic, ranking rule, population rule or C1–C6 behaviour is changed, and no
answer moves.

---

## 1. The finding, before any change

**The live path did not use the governed receipt.** Two facts, both measured:

* **Structural.** An AST sweep of every non-test, non-instrument module in the
  repository for an import of `mi_agent_api.movement_receipt` returned
  `NONE`. The only importers were `migration_phase0/c7_target_plan_proof.py`
  and `migration_phase0/c7_independent_audit.py`.
* **Executed.** A delivered ranked movement —
  *"Which two geographic region obligors added the most balance since last
  month?"*, route `period_change_analysis`, `ok=true` — published its evidence
  as `metadata.rankedMovement`, a dict assembled inline in `_render`. The
  response carried no receipt.

Traced end to end, each of the six facts had **three independent derivations**:

| fact | prose (`build_rank_answer`) | table (`_rank_rows` / artifact) | `metadata.rankedMovement` |
|---|---|---|---|
| comparison periods | `result.summary["period"]` | `summary['period']` in the description | `summary["period"]` |
| grouping dimension | `ranking.distribution.display_name` | same, in the title | `ranking.distribution.field` |
| start/end/movement | `ranking.movement.rows[i]` | `ranking.movement.rows[i]` | `ranking.movement.rows[i]` |
| direction / basis | `movement.direction`, `movement.basis_label` | `movement.basis` | `movement.direction`, `movement.basis` |
| **rank position** | `movement.rows[0]` | **recomputed** by `enumerate(...)` | not published at all |
| **population / filter** | — | — | **absent entirely** |

They agreed only because the same objects were iterated three times.

**Verdict at this point: LIVE EVIDENCE PATH STILL DUPLICATED.**

## 2. The wiring

One `MovementReceipt` is built at the single point where both the intent (which
governed fields the reader's term could bind to) and the outcome (which one it
did) are in scope, and everything downstream reads it.

* `movement_receipt_for(result, intent, ranking)` — new, in the route. Carries
  the ranking engine's figures **verbatim**: opening, closing, absolute and
  percentage movement, the value sorted on, presence and the engine's note. It
  recomputes nothing, and reads no question, chart column, artifact title or
  route identity.
* `build_rank_answer(receipt)` — the result, the ranking outcome and the
  question are no longer parameters.
* `_rank_rows(receipt)` — the rank position is **read** from
  `element.rank` instead of re-counted with `enumerate`.
* `_render(..., receipt)` — the `ranking` parameter is **removed**, so the
  renderer has no second source to read.
* `metadata.rankedMovement` — same keys, same values, now a projection of the
  receipt.
* `metadata.movementReceipt` — additive, the full governed record including the
  population block.

No second receipt type: `build_movement_receipt` remains the one builder and
`MovementReceipt` the one type. The builder gained an `elements` parameter so a
caller that already has ranked rows hands them over instead of having them
re-derived — which is what stops two calculations of one percentage existing.

### Population evidence, stated rather than implied

This route selects a population by scope, not by row predicate. The receipt
therefore publishes an empty predicate tuple with equal filtered and unfiltered
per-period row counts, so `narrowed` is `false` **as a fact**. An absent
population block would have left it unknown.

### Two corrections made along the way, both recorded

1. `MovementReceipt.missing_facts()` raised `TypeError` on an element with a
   missing endpoint: it appended the gap and then ran the reconciliation
   arithmetic anyway. A `continue` was added after the gap is recorded. Nothing
   is removed from `gaps`; the audit now fails such a receipt instead of
   crashing on it.
2. `DistributionChange` has no `aggregation` attribute — it is a constant in its
   `to_dict`. The receipt reads the aggregation from the governed result payload
   for that dimension rather than naming one here.

## 3. Blast — the live surface

`/mi/query` executed in a worktree at `1c49e61` and in the working tree against
**one shared fixture**, comparing answer, warnings, every artifact (title,
description, rows) and `metadata.rankedMovement`, field by field:

```
TOTAL  values changed = 0    keys removed = 0    keys added = additive only
```

Nine questions spanning delivered ranked movement (four variants including
top-N and percent basis), the alternate-term binding (`region`), unranked
movement, and three governed refusals. The only key that appears is
`metadata.movementReceipt`. **0 unexplained answer movement, 0 economic
movement, 0 new silent drops, refusals unchanged.**

The 882-question contract sweep is not re-run for this change: nothing on the
contract path was touched — the diff is confined to `mi_agent_api/`.

## 4. Mutation controls — on the LIVE response

`tests/test_live_movement_receipt_evidence.py`, 20 tests. Each perturbs exactly
one fact in the receipt the route built and requires the live response to move.
If any part of the response were still deriving that fact independently, the
mutation would leave it unchanged and the test would fail.

| mutation | live response |
|---|---|
| period order reversed | prose dates, table description and `openingPeriod`/`closingPeriod` all move |
| movement value negated | published figures move |
| ranking position swapped | table order and the prose lead move |
| ranking direction flipped | "increased" → "decreased" |
| **grouping field substituted** | the answer is **refused**: the estate's disclosure guard reads the published evidence, sees a dimension the reader did not ask for and returns *"I have not substituted a broader figure"* with no ranked artifact |
| predicate evidence added | `population.predicates` and `narrowed` move |

The grouping-field control is the strongest of the six and was not what the
test first expected: a substituted dimension is not retitled, it is caught. That
only works because the guard and the renderers now read one record.

A restore test asserts the route is unmutated afterwards, and three structural
tests assert that `build_rank_answer`, `_rank_rows` and `_render` do not take a
ranking outcome, that the two ranked renderers name neither the question nor the
result, and that every value in `rankedMovement` is read off `receipt`.

## 5. Blast — the whole corpus, and what it does not cover

All **882** Stage 1 + Stage 2 corpus questions executed through the live
`/mi/query` path in a worktree at `1c49e61` and in the working tree, against one
shared fixture, comparing route, `ok`, answer, warnings and every artifact
(title, description, rows) field by field:

```
QUESTIONS SWEPT   882      EXCEPTIONS 0
QUESTIONS CHANGED 0        ROUTES CHANGED 0
KEYS REMOVED      0        KEYS ADDED 0
ok=True  before/after  381 / 381
```

**This denominator does not exercise the changed path, and saying so is the
point.** `ranking applied` is **0 of 882 in both trees** — the corpus carries
ranking language on 97 questions but none of them reaches a delivered ranked
movement, which the C7 ranking census already established. The sweep is
therefore evidence of *no collateral movement across the estate's live
surface*, not evidence that ranked movement still works.

The changed path is exercised by, and only by:

* the targeted live trace — **4 delivered ranked-movement responses** (absolute,
  percent-basis, top-N, and the `region` alternate-term binding), 0 values
  changed, additive key only;
* the executed compound canary — **6 delivered ranked cases** (F9.c, F11.a–e),
  grades unmoved;
* `tests/test_live_movement_receipt_evidence.py` — 20 tests over a delivered
  case, including the six mutation controls.

## 6. Preservation

19 modules run **module by module in both trees**, serially, working tree
against a worktree at `1c49e61`:

```
failing test names:  before 9   after 9
INTRODUCED:      (none)
FIXED/REMOVED:   (none)
summary lines:   identical, module by module
```

The 9 pre-existing failures (`test_conversion2_period_movement` 5,
`test_p0_execution_receipt` 3, `test_mi_predicate_extraction` 1) stay in the
denominator. `tests/test_live_movement_receipt_evidence.py` — 20 passed —
exists only in the working tree and is not in the baseline denominator.

### A method caveat, found the hard way

The previous commit's whole-repository run reported one test as "fixed", and it
was not. `test_registry_governance.py::test_checked_in_registry_matches_generator`
compares the **absolute path** recorded in the checked-in registry against the
path of the tree regenerating it, so it fails in any worktree and passes only at
`/home/user/trakt`. Reproduced serially in both trees. Any worktree-baseline
diff needs its survivors checked rather than trusted; only this one test was
affected, but the class of error is not specific to it.

Canary and audit, measured on this tree:

* executed compound canary — **0 invariant breaches**, nothing `DROPPED`;
  21 UNEVIDENCED elements across 9 cases and 5 unexercised families, all
  unchanged. The canary reads `metadata.rankedMovement` only; it was **not**
  taught to read the new receipt channel, because doing so would move frozen
  baselines.
* `c7_independent_audit` — **10 of 10**.
* `c7_target_plan_proof` — 10 cases still build and execute; the seven
  registered RM3 mutation controls still discriminate 7/7 and restore.


## 7. Verdict

**LIVE RANKED-MOVEMENT EVIDENCE PATH CLOSED.**

Every one of the six facts is read from the receipt, and the renderers have no
second source to read because `_render` no longer receives one. The remaining
honest gaps, stated rather than closed:

* the compound canary reads `metadata.rankedMovement` only. It was deliberately
  not taught the new channel, so its 21 UNEVIDENCED elements are unchanged;
* `period_change` selects a population by scope and not by row predicate, so the
  receipt's predicate tuple is empty **as a published fact**. The C7 filter
  composition fixed in `1c49e61` reaches the contract, not this route;
* a hazard found while debugging and NOT changed here: when the receipt builder
  raised, `chat_routing` logged `route period_change_analysis failed` at WARNING
  and silently fell through to `temporal_compare`, which answered with a
  plausible refusal. A faulting route is answered by a different route rather
  than by an error. Out of scope for this task; worth its own look.
