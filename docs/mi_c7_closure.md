# C7 closure — governed pieces finished, route reduced

Base `92ef6ae`. This task closed the genuine gaps, reduced the route, and
re-measured. **`period_change_route.py` was changed** — that is the point of it.

---

## Product rulings applied

**No implicit measure. No implicit comparison period.** A required element the
question does not supply is a **governed clarification**, not a capability gap.
The matrix below distinguishes *the architecture cannot represent this* from
*the user did not say*.

Checked by running the system, not by reading code:

```
"Which region grew the most?"          ok=False — "…names no period to compare
"Which region added the most balance?" ok=False    over, and I have not chosen
                                                   one for you."
```

**No STOP — IMPLICIT SEMANTIC DEFAULT REMAINS.** And one was *introduced and
then removed inside this task*: fixing D1 unmasked the recogniser's
latest-versus-previous default, which had been hidden behind the false
dimension refusal. It is now guarded from the contract.

---

## Phase 2 — ranking limit, closed

The vocabulary moved to `question_interpretation.lexical.ordering_request`,
joining LEVEL/MOVEMENT under one owner. `rank_request` delegates; no route-local
number parsing remains.

```
which two  -> 2      top 3      -> 3      bottom five -> 5
biggest 20 -> 20     which five -> 5
```

| | before | after |
|---|---|---|
| ranking questions carrying a limit | 26 / 97 | **30 / 97** |
| carrying a direction | 80 | **95** (81 increase, 14 decrease) |
| **generic plans blocked** | **15 / 97** | **0 / 97** |

The 15 unblocked because the owner no longer requires a resolved dimension
before it will report a direction and a basis — the old reader returned nothing
without one, losing two facts to explain a third.

**Two direction bugs were found and fixed in my own owner**: `last` as a
bottom-rank word made *"since last month"* read as a descending ranking, and
superlatives in the increase set made *"largest fall"* resolve to
increase-and-decrease-at-once. Direction now comes from the verb; `most`,
`largest` and `top` say only that an ordering was asked for.

---

## Phase 3 — the ranked-movement receipt

`mi_agent_api/movement_receipt.py`. Independent of `period_change` — audited by
AST, its only imports are `__future__`, `dataclasses` and `typing`.

Built from execution facts **before any prose exists**; nothing in it reads a
chart column, an artifact title or an answer string. Per ranked element it
carries group, measure, both periods, both values, absolute and percentage
movement, basis, direction, limit and rank position, plus population evidence
with per-period row counts filtered and unfiltered.

Two properties are checkable rather than asserted: `missing_facts()` fails a
receipt whose delta contradicts its own endpoints, and `chronological` makes D4
structural.

---

## Phase 4 — population, proven non-vacuously

```
predicate on the contract   youngest_borrower_age gt 70.0
row counts    filtered [49, 47]      unfiltered [76, 80]
groups        London, Scotland, South East          (3 populated)
movement      South East +1,453,508.40 · London +555,175.86 · Scotland −1,917,494.65
ranked        South East, London                     receipt complete: True
control       predicate removed -> counts [76, 80], winner becomes London   DISCRIMINATES
```

The filter **changes the winner**, which is the strongest available evidence
that it ran.

**Two findings recorded rather than smoothed over.** The LTV predicate the task
named cannot exercise this: the frame normalises `current_loan_to_value` to a
fraction (0.20–0.549) while the contract predicate carries `50.0`, so it matches
nothing — a **unit mismatch between predicate representation and executed
frame**. And separately, every ranked-movement phrasing of a filtered question
**loses the predicate entirely and hijacks the subject**:

```
"balance by region for loans with LTV above 50%"
    subject=current_outstanding_balance   predicates=[(current_loan_to_value, gt, 50.0)]
"Which region added the most balance since last month for loans with LTV above 50%?"
    subject=current_loan_to_value         predicates=[]
```

So the **population primitive is proven**; **ranked movement + filter in one
question is not representable**. RM3 as specified is **RED at representation**.

---

## Phase 5 — new-capability set

**NEW-CAPABILITY EXECUTION PROOF — NOT LEGACY EQUIVALENCE.** The 882-question
corpus contains zero genuine ranked historical movements and none is invented.

| | case | result |
|---|---|---|
| RM1 | added the most balance since last month | London +£1.98m, South East +£1.82m, Scotland −£1.57m |
| RM2 | largest fall in balance since last month | Scotland −£1.57m, direction=decrease |
| RM3 | …for loans with LTV above 50% | **BLOCKED — predicate not represented** (above) |
| RM4 | which **two** added the most balance | `topN=2` honoured |
| RM5 | grew **fastest** in balance | `basis=percent` |

**Mutation controls 5 of 5 discriminate, all restored.** The period-order
control **failed first** — the executor took the last two frames regardless of
the plan's stated order, D4 reproduced inside the harness built to detect it —
and was fixed by binding each stated token to its frame.

---

## Phase 6 — the route, reduced

| | before | after |
|---|---|---|
| total lines | 1,112 | 1,132 |
| **INTERPRETS** | **16** | **0** |
| **VOCABULARY** | **23** | **8** |
| **semantic ownership** | **39 (4.0%)** | **8 (0.8%)** |
| RENDERS | 190 | 190 |
| ADAPTS | 320 | 377 |
| **raw-question readers** | 1 (`_rank_subject`) | **0** |
| route-local decision sites | 7 | 7 (K1, K7 defects closed) |

Deleted: `_rank_subject`, `_NARRATIVE_RANK_SUBJECTS`, `_RANK_SUBJECT_LEAD_RE`,
`_RANK_SUBJECT_SKIP`. `resolve_rank_intent` no longer takes a question — it
takes the interpretation and reads `ordering_direction`, `ordering_basis`,
`ordering_limit` and the dimension claim's `candidate_concepts`.

**The line count went UP by 20 and the semantics went DOWN by 31.** That is the
honest shape of this change and it is not hidden: deleting 60 lines of
vocabulary while adding a contract reader, a candidate-carrying refusal message
and the no-implicit-period guard nets positive. Reduction here means *semantic
ownership*, not bytes. Seven composition decisions remain route-local.

**The narrative guard went with the vocabulary.** It asked whether the noun
after "which" was one of 36 words; the contract answers structurally — a ranking
with no dimension claim is the narrative. That reading does not depend on which
interrogative opened the sentence, which is why *"show me the drivers that grew
the most"* used to miss the guard entirely.

---

## Phase 7 — route identity

`grew the most` vs `added the most`: contract **identical**, plan **identical**,
and both now reach the same execution semantics. D1's alternate binds from the
natural term:

```
"Which region grew the most since last month?"  ->  geographic_region_obligor
```

**D3 is narrowed but NOT closed.** No route decides measure, period, movement,
basis, dimension or direction any more — all come from the contract. What
remains is that two routes can still both be eligible and **precedence** picks
one. Per instruction, not fixed here.

---

## Phase 9 — the matrix

| dependency | represented | owner agreement | plan consumable | non-vacuous execution |
|---|---|---|---|---|
| Dataset | GREEN | GREEN | GREEN | GREEN |
| Measure | GREEN | GREEN | GREEN | GREEN — absent ⇒ governed clarification, proven |
| Comparison period | GREEN | GREEN | GREEN | GREEN — absent ⇒ governed clarification, proven |
| LEVEL vs MOVEMENT | GREEN | GREEN | GREEN | GREEN — F11 ×5 |
| Grouping dimension | GREEN | **THIN** | GREEN | GREEN — `role` unresolved on RM cases |
| Alternate dimension | GREEN | GREEN | GREEN | GREEN — binds, mutation-controlled |
| Ranking requested | GREEN | **THIN** | GREEN | GREEN — contract 95 / route 97, 2 disagree |
| Direction / basis | GREEN | GREEN | GREEN | GREEN — 95/97, plans blocked 0 |
| Ranking limit | GREEN | GREEN | GREEN | GREEN — 30/97, `topN=2` executed |
| Population | **THIN** | GREEN | GREEN | GREEN — proven on age; **RED for ranked-movement + filter** |
| Receipt | GREEN | GREEN | GREEN | **THIN** — structure complete and audited, but the estate's live receipt is still route-published |

---

## Phase 10 — canary and regression

```
breaches 7 -> 0      grade movements 70      route movements 20      answer movements 23
new breaches 0       cleared 7
```

**The zero is not a clean bill, and is not reported as one.** Most of the bank's
ranked cases name no period, so under the ruling they now refuse: F3, F5, F6 and
F7 joined F4 as **UNEXERCISED**. Zero breaches measured over refusals is exactly
the vacuity I6 exists to catch.

Family **F11 was ADDED** — never edited — with five period-carrying ranked
movement cases. All five honour **every element they declare**, including
`TOP_N` and `BASIS`. Recorded as ledger entry **M3** with the evidence loss
stated.

**No canary moved into WRONG or SILENTLY INCOMPLETE.**

Focused estate: **14 modules, 382 tests — 374 passed, 8 failed.** The 8 are the
same 8 that fail at `174d14d`, verified in a C6-close worktree. **0 introduced,
0 fixed.** The full estate does not complete in this environment; no broader
claim is made.

## Phase 11 — independent audit

`migration_phase0/c7_independent_audit.py` reads the source and the running
system, no report and no prior JSON. AST-based, so a comment explaining a
deletion cannot satisfy a check.

**10 of 10 pass**: `_rank_subject` gone · route ranking vocabulary gone · zero
functions read the raw question for meaning · no implicit period or measure ·
ranked movement reconciles · alternate binds · limit honoured · receipt
independent of `period_change` · receipt carries the required facts · owner
still singular.

---

# Verdicts

### 1. C7 CONTRACT VERDICT — **C7 CONTRACT STILL INCOMPLETE**

Closed: ranking limit, direction, basis, alternates, level-vs-movement, receipt
structure. **Open, with one exact reason:** a filter combined with
ranked-movement wording is not representable — the predicate is dropped and the
subject is overwritten by the filter's field. Plus the predicate/frame unit
mismatch on LTV.

### 2. C7 EXECUTION VERDICT — **RANKED MOVEMENT COMPOSES GENERICALLY**

Measure, period, movement, grouping, ranking and receipt compose from contract
values through one generic builder and execute without `period_change_route`,
with 5 of 5 mutation controls discriminating. **No parser, recogniser or
executor shape logic was added.** Population composes; population *expressed
inside a ranked-movement question* does not.

### 3. C7 ROUTE VERDICT — **C7 RADICALLY REDUCED**

```
                       before      after
total lines             1,112      1,132
semantic ownership     39 (4.0%)   8 (0.8%)
  INTERPRETS              16          0
  VOCABULARY              23          8
adapter / rendering       510        567
raw-question readers        1          0
semantic decision sites     7          7   (K1, K7 defects closed)
```

Not RETIRED: seven composition decisions remain route-local, and the live
receipt is still route-published.

### 4. EVIDENCE VERDICT — reported separately, not combined

| | |
|---|---|
| legacy equivalence | **1** |
| refusal preservation | **2 proven** + 7 owned refusals preserved |
| new capability | **4 of 5** RM cases execute; RM3 blocked at representation |
| canary | 7 → 0 breaches, **5 families unexercised**, F11 ×5 fully honoured, 0 into WRONG |
| regression | 14 modules / 382 tests / 374 passed / 8 failed / **0 introduced** |

### 5. THESIS VERDICT

> **Can a ranked historical movement question be answered by composing governed
> measure + time + movement + grouping + population + ranking + receipt, without
> adding parser/recogniser/executor shape logic?**

**Yes, with one exception, and no shape logic was added.** Every element is
carried by the contract and consumed by one generic builder; the exception is a
**filter expressed inside a ranked-movement question**, which the parser drops.

> **After C7 reduction, how many route-specific semantic decision owners remain
> across the seven migrated core routes?**

**Seven, all in `period_change_route`** — K1–K7. Every other converted route
(C1–C6) reads its semantics from the contract. Two of the seven (K1, K7) had
their defects closed here but remain route-local decisions.

**The 7/7 architectural assessment is not yet warranted.** Legacy equivalence is
still 1, seven route-local decisions remain, and the receipt is structurally
complete but not yet the estate's live evidence channel.
