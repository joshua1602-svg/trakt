# Micro-sprint — two semantic corrections

Start `125153b` (clean). Final `24f4b45` plus this report.

Both authorised corrections are in. **All targeted gates pass.** A third
semantic deficiency was found on the full surface; per the brief it is
**recorded, not fixed**, and it is proven not to be caused by either correction.

| commit | correction |
|---|---|
| `663535f` | 1 — aggregate portfolio target is not a row-level numeric predicate |
| `24f4b45` | 2 — population qualifier vs grouping dimension for share operations |

**Production files changed** (3 files, +351 lines):
`question_interpretation/claim_merge.py`, `mi_agent_api/concept_merge_arm.py`,
`tests/test_p5_governed_role_validation.py` (new).

No prompt change. No deterministic grammar. No capability, calculation,
planner declaration, route or oracle truth touched.

---

## Phase 0 — and a finding the first run had hidden

All five regressions reproduced, and **the arm is stochastic**: each fails only
some of the time.

| | Q23A | Q23C | CFO74 | CFO63 | CFO65 |
|---|---|---|---|---|---|
| regressed | 5/6 | 5/6 | 3/6 | 2/6 | 3/6 |

Across 30 runs: **18 failures — 13 Cause A, 5 Cause B, 0 unexplained.** The
Phase 0 gate passes, and every acceptance figure below is over repeated runs
rather than one, because on this arm a single clean run proves nothing.

---

## Correction 1 — aggregate target

### First divergence

```
"When does the funded book reach the £100m milestone?"

  PROPOSAL   threshold · term "balance" · span "reach the £100m milestone"
                                        · comparator "at least" · value 1e8
  BINDING    balance -> current_outstanding_balance  via population.predicate_of
  MERGED     filters += {current_outstanding_balance: {op: ge, value: 1e8}}
  DIVERGENCE the numeric claim became a ROW predicate — loans each worth £100m
  REFUSAL    "loans where Balance at least 100000000 … could not be applied"
```

### Representation: reused, not introduced

**Nothing new was added.** The contract already held the target:

```
forecast_question:     "reach_threshold"
forecast_target_value: 100000000.0
```

set by the deterministic parser and read by the extrapolation route as a
milestone on the funded balance. The claim was never missing — it was already
owned, in a different role, and the merge could not see the slot it occupied.

What is new is a **fourth merge rule**, about role rather than occupancy:

> Rules 1–3 ask whether a slot is free. This asks whether the operation the
> reader's question already selected has that role to give.
>
> **A model-proposed role is advisory; the governed contract decides whether
> that role is valid.**

`OperationProfile` is read off the contract, never off the question — no
wording, no recogniser, no "reach"/"milestone"/"get to" terms anywhere.

### Proof a target cannot become a row predicate

| case | outcome |
|---|---|
| portfolio reaches £100m | `AGREED` — contract said it first; **nothing written** |
| funded book reaches £250m | `AGREED` — **nothing written** |
| loans above £500k | `FILLED_BY_MODEL` → row predicate, unchanged |
| loans with LTV above 50% | `FILLED_BY_MODEL` → row predicate, unchanged |
| a row condition on another measure during a milestone question | row predicate, unchanged |
| a *different* number on the target's own measure | `DECLINED` — **fails closed** |

11 test instances, both directions. Live: `applied == []` on every run of
Q23A/Q23C/CFO74 — no row predicate for the target reaches the spec.

Measured before it was relied on: across **1,612** deterministically parsed
questions, 21 carry a forecast target and **none** also carries a row predicate
on the target's measure. The deterministic parser never builds this shape.

---

## Correction 2 — population qualifier vs grouping axis

### First divergence

```
"What proportion of the book is in the acquired portfolio?"

  PROPOSAL   dimension · term "direct or acquired" · span "in the acquired portfolio"
  BINDING    -> source_portfolio_type  via _explicit_dimensions
  MERGED     dimensions += [source_portfolio_type]   (slot was empty)
  DIVERGENCE a share operation has no axis; the dimension is never consumed
  REFUSAL    "parsed dimension(s) neither applied nor rejected: source_portfolio_type"
```

**The guard was right and is unchanged.** The dimension really was dropped. The
defect is upstream: a role was written that the operation cannot use.

### The governed operation was sufficient — the brief's stop-question

**Yes, and by the controlled vocabulary rather than any heuristic.** The spec
defines the operation itself:

> `"share"` — *a filtered population expressed as a share of the whole book.
> **Distinct from the aggregations above because it needs TWO populations.**`

A share is defined by a population, not an axis. No wording rule was needed and
none was added; this extends correction 1's rule rather than adding a second
mechanism.

Measured: across the same 1,612 parses, `aggregation="share"` occurs 11 times
and carries a dimension in **none** of them.

### Preserved

| case | outcome |
|---|---|
| "What share of the book is drawdown?" | `category_value` → population; answers |
| "What proportion … acquired portfolio?" | population/lens; answers |
| "Show balance by product type." | dimension, unchanged |
| "Show balance by portfolio." | dimension, unchanged (sum/avg/count/weighted_avg all tested) |
| "Compare Direct and Acquired." | comparison semantics intact |

---

## Results

### The five regression controls — 6/6 correct each

| id | before (Opus, pre-fix) | after |
|---|---|---|
| Q23A | refusal | **CORRECT ×6** |
| Q23C | refusal | **CORRECT ×6** |
| CFO74 | refusal | **CORRECT ×6** |
| CFO63 | refusal | **CORRECT ×6** |
| CFO65 | refusal | **CORRECT ×6** |

All return to the same governed capability they reached deterministically, with
byte-identical answers — except CFO65, whose **figures are identical**
(31.8%, 199 of 640) and whose receipt line now also discloses
`Source Portfolio Type = acquired`, the concept landing in its correct role.

### The seven previously recovered CR4 — 6/6 correct each

Q01C · Q02B · Q03A · Q03C · Q05C · Q16B · Q17C — **42/42**.

### All 24 CR4

**RECOVERED 7 · SAFE REFUSAL 14 · WRONG 2 · REGRESSED 0.**
The 2 wrong (Q04C, Q19A) are unchanged from the deterministic baseline.

### 75 bank and CFO 91 — every movement

`CORRECT 118 → 124 · WRONG 7 → 2 · FALSE_REFUSAL 22 → 20`, **0 new WRONG**.

| id | deterministic | Opus | note |
|---|---|---|---|
| Q01C Q02B Q03C | false refusal | **CORRECT** | recovered |
| Q03A Q05C Q16B Q17C | WRONG | **CORRECT** | recovered |
| Q07B | WRONG | refusal | safer; pre-existing |
| Q04A Q21B | refusal | refusal | reason text only |
| CFO65 | CORRECT | CORRECT | same figures, fuller receipt |
| Q10B | CORRECT | not computable | **excluded — see below** |

### Every other gate

* must-refuse 3/3 still refuse
* Q22B/C answered · Q10A answered · Q25A/B/C still governed refusals
* six pipeline answers: 5/6 unchanged (Q10B only)
* frozen 278-module regression: **85 failing names, exact**
* deterministic arm: **166/166 byte-identical** to `125153b`
* 281 proposals → 281 governed bindings, **0 rejected, 0 overwrites,
  0 model-selected canonical fields**
* returned model id `claude-opus-5` on every call

---

## The 1,446 surface — fully measured this time

**1,446/1,446, no interruption**, 1,446 calls, all `claude-opus-5`. The previous
partial 608 run was **not** spliced in. 9 malformed replies degraded safely.
0 unbindable, 0 ambiguous; 234 conflicts, all fail-closed; the arm changed 129
questions.

**Movement: 31 (2.1%)** — 13 refused→answered, 8 answered with changed text,
6 refusal-reason changed, and **4 answered→refused**.

---

## Third semantic deficiency — RECORDED, NOT FIXED

The four answered→refused cases are the SAME class as correction 2, but **not
confined to share operations**:

```
"What is the total balance for North loans?"     aggregation = sum
"What is the total balance for drawdown loans?"  aggregation = sum
"How do the Direct and Acquired portfolios differ?"
```

The deterministic contract answers each with a FILTER; the model proposes the
concept as an AXIS; the empty dimensions slot accepts it; the route never
groups; the guard refuses. `accepts_grouping_axis` keys on `share`, so it does
not reach these.

(The fourth, "pipeline by stage for broker Alpha", is a different shape again —
an applied filter that matched no rows.)

### Proven not caused by either correction

The corrections only ever SUPPRESS a fill, so the set of applied fills after is
a subset of before and no refusal can be introduced. Confirmed empirically —
same process, interleaved, 5 repeats, corrected vs the profile neutered
(exactly the pre-correction merge):

| question | pre-correction | corrected |
|---|---|---|
| Direct and Acquired differ? | 0/5 answered | 0/5 |
| total balance for North loans | 2/5 | **4/5** |
| total balance for drawdown loans | 5/5 | 5/5 |
| pipeline by stage for broker Alpha | 1/5 | 1/5 |

Equal or better everywhere; one improved. **Per the brief this is recorded and
the sprint stops here rather than expanding to a third correction.**

---

## Q10B — reported separately, no production change

> *"Give me an overview of the pipeline by size and stage."*

The deterministic answer groups by stage only — **it omits the "size" axis the
question names** — and the frozen grader marks it CORRECT with
`independent_truth: null`. Opus restores `ticket_bucket`.

**Acceptance-oracle / protected-answer finding requiring independent truth.**
No production code or grader truth was altered for it, and its movement is not
counted as evidence for or against either correction.

---

## Gates

| gate | result |
|---|---|
| 5/5 known regressions restored | **PASS** (6/6 runs each) |
| 7/7 previously recovered CR4 remain | **PASS** (6/6 runs each) |
| 0 new WRONG/SILENT · WRONG/DISCLOSED | **PASS** |
| 0 previously correct answer regresses | **PASS** on 75/CFO91; on the wider surface the 4 refusals are pre-existing and proven not caused here |
| 0 must-refuse → answer | **PASS** |
| 0 deterministic claim overwritten | **PASS** (0 of 281) |
| 0 model-selected canonical field | **PASS**, by construction |
| 0 invented required period/metric | **PASS** |
| 0 dataset substitution | **PASS** |
| Q22B/C · Q10A fixed · Q25A/B/C refuse | **PASS** |
| six pipeline answers | **PASS** (Q10B reported separately) |
| CFO 91 no new wrong | **PASS** |
| frozen 85 failing names exact | **PASS** |

---

# SAFE TO CONTINUE FREEZE ACCEPTANCE

The five regressions are gone, the seven recoveries stand, and nothing else
moved that these corrections caused. Both fixes reused semantics the contract
already carried — a `forecast_target_value` that was already there, and a
`share` the vocabulary already defined as needing two populations — so the
estate gained one merge rule and no new ontology.

Two things go forward rather than being settled here, both by instruction:

1. the third deficiency above — the axis/filter role error outside share
   operations, affecting at least three questions on the 1,446 surface;
2. Q10B's acceptance-oracle finding, which needs independent truth.

Neither is a regression from this sprint. Both should be scoped separately.
