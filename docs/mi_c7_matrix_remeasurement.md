# C7 matrix re-measurement — evidence gate

**C7 remains not authorised. `period_change_route.py` is untouched and reads no
`ordering_of`** (grep count 0; zero diff against `174d14d`). Nothing in this task
changed C7 execution.

---

## 1. Starting state — **REPRODUCED**

```
HEAD 4a3e5aa · branch claude/mi-query-agent-c7-2tlhr6 · tree clean · local == remote
period_change_route.py         0 lines changed since 174d14d, 0 references to ordering_of
canary bank                    33 cases · 4 frozen defects · ledger M1, M2
owner live and singular        temporal_aspect("How did the balance change…") = movement
                               CHANGE_MARKERS / _COMPARE_TRIGGER_RE absent from the estate
```

Blast recomputed in two trees from execution, not read from the report:

| | reported | re-measured |
|---|---|---|
| contracts changed | 113 / 882 | **113** |
| plans changed | 2 | **2** |
| routes changed at plan level | 1 | **1** (`temporal_compare`) |
| answer/refusal movements | 4 / 140 | **4 / 140** |
| delivered-economic movements | 0 | **0** |

---

## 2. The matrix

Gates: **A** represented · **B** owner agreement · **C** plan consumable from the
contract alone · **D** non-vacuous execution.

Evidence for C and D is `migration_phase0/c7_target_plan_proof.py`, whose plan
builder takes a `QuestionInterpretation` **and nothing else** — it cannot see the
question text, the route, `_rank_subject` or any C7 vocabulary, because they are
not arguments.

| # | dependency | A | B | C | D | evidence / exact reason |
|---|---|---|---|---|---|---|
| 1 | Dataset | GREEN | GREEN | GREEN | GREEN | `DatasetClaim`; owner `workspace.resolve_dataset`; `dataset_of` consumed in all 7 plans; executed on all 4 |
| 2 | Measure | **THIN** | GREEN | **RED** | **THIN** | subject **empty** on 3 of 7 proof cases. "Which region declined the most since last month?" and "How did the book change since last month?" carry **no subject at all** → `measure_request` = `(None,"sum")` → plan cannot say what to compare. `period_change` answers these from its own governed overview policy (`config/period_change_selection.yaml`) — a route-local default the contract has no equivalent of |
| 3 | Comparison period / span | GREEN | GREEN | **THIN** | GREEN | carried where the question names one (`comparison_periods` or `window_periods`). **Blocked where it does not**: "Which region grew the most?" → *"no contract field: neither comparison_periods nor time.window_periods names a period"*. The contract has **no governed default pair**; the route supplies one |
| 4 | LEVEL vs MOVEMENT | GREEN | GREEN | GREEN | **THIN** | `ordering_of` on 95 of 97 (2 unstated); one owner, singularity enforced structurally; the plan branches on it (`compare` step present iff movement). Execution proven only on **new-capability** RM cases — the corpus has none |
| 5 | Grouping dimension | GREEN | **THIN** | GREEN | GREEN | `DimensionClaim` on 84 of 97. `role` is `unresolved` on the RM cases, so the plan carries the dimension but not what it is *for*; and "declined" raised a spurious second claim (`pipeline_stage`) |
| 6 | Alternate valid dimension | GREEN | GREEN | GREEN | GREEN | D1 closed: resolver → contract (`alternate_concepts`) → plan (`group.by=['collateral_geography','geographic_region_obligor']`) → execution **binds `geographic_region_obligor`**, the field the book carries. Mutation control: dropping the alternate fails the execution |
| 7 | Ranking requested | GREEN | **THIN** | GREEN | GREEN | `operation.type == "ranking"`. Contract 95, route 97, disagree **2** — both the contract being *more* conservative ("regions with the most loans", "which limits are most at risk" → `amount`) |
| 8 | Ranking direction | **THIN** | GREEN | **THIN** | GREEN | carried on **80 of 97**; 15 plans blocked on *"no contract field: operation.ordering_direction / ordering_basis"* |
| 9 | Ranking basis | **THIN** | GREEN | **THIN** | GREEN | same 80 of 97, same 15 blocked |
| 10 | Ranking limit | **RED** | GREEN | **RED** | **RED** | carried on **26 of 97**, and the vocabulary is incomplete: *"Which **two** regions grew the most"* yields `ordering_limit=None` because `_TOP_N_RE` knows `three|four|five|ten` and not `two`. The plan then ranks every riser. Silently wrong, not refused |
| 11 | Population / filters | GREEN | GREEN | GREEN | **RED** | `SourceScopeClaim` + `RowPredicateClaim` carried and consumed into `select_population`, but **no proof case narrowed anything** — every case ran whole-book. Unexercised, so not counted |
| 12 | Receipt / evidence | **RED** | **RED** | **RED** | **RED** | there is no receipt primitive in the plan vocabulary at all (`STACK_PERIODS, SELECT_POPULATION, RESOLVE_MEASURE, GROUP, RANK, COMPARE`). Every receipt in the estate is built by a route; `metadata.rankedMovement` is published by exactly one. A generic plan produces numbers and no evidence trail |

**Two structural findings behind the RANK rows.** The `RANK` primitive exists but
both production uses hard-code `basis="funded_balance"`, `direction="desc"` and a
route constant for `top_n`, and rank a **level**. No plan builder reads
`ordering_direction`, `ordering_basis`, `ordering_limit` or `alternate_concepts`
— representation is real, consumption is zero.

---

## 3. Class evidence

### Class L — ranked level · **2 of 2 execute**
```
Which region has the largest balance?   -> South East 7,669,070.94 · London 6,943,256.79 · Scotland 6,490,175.47
which Broker has the largest balance    -> Alpha 6,305,213.35 · Delta 5,335,267.82 · Gamma 5,237,163.86 · Beta 4,224,858.17
```

### Class M — unranked movement · **1 of 2 execute**
```
How did the balance change since last month?  2026-05-31 -> 2026-06-30   +£2,229,701.57
How did the book change since last month?     BLOCKED — contract carries no subject
```

### Class RM — ranked movement · **NEW-CAPABILITY EXECUTION PROOF — NOT LEGACY EQUIVALENCE** · **1 of 3 execute**

The corpus contains **zero** genuine ranked historical movements; these are
pre-registered new-capability cases and are never counted as equivalence. Each
exercises a different part of the plan — direction, basis, limit — not three
paraphrases of one path.

```
Which region added the most balance since last month?
  periods  2026-05-31 -> 2026-06-30      grouping bound  geographic_region_obligor
  movement London +1,981,868.42 · South East +1,815,302.51 · Scotland −1,567,469.36
  ranked   London, South East            (three populated groups; one decliner)

Which region declined the most since last month?   BLOCKED — no subject on the contract
Which two regions grew the most since last month?  BLOCKED — no subject on the contract
```

### Mutation controls — **5 of 5 discriminate**, all restored
```
reverse comparison periods   -> ranked [Scotland +1,567,469.36]        discriminates
MOVEMENT -> LEVEL            -> ranked by level, not movement          discriminates
change grouping dimension    -> ranked by broker                       discriminates
reverse ranking direction    -> ranked [Scotland −1,567,469.36]        discriminates
drop the alternate dimension -> error: collateral_geography not carried discriminates
```

The first of these **failed on the first attempt**: the executor took the last
two frames regardless of the plan's stated order, so reversing the periods
changed nothing — the D4 defect reproduced inside the harness built to detect
it. Fixed by binding each stated period token to its frame, and the control now
discriminates.

---

## 4. D1 – D3 retested

| | verdict | evidence |
|---|---|---|
| **D1** valid alternate dimension | **CLOSED** | survives resolver → contract → plan → execution; binds the field the book carries; mutation-controlled |
| **D2** movement answered as a level | **CLOSED at contract and plan** | `ordering_of=movement` → the plan contains `compare`; a single-period level cannot satisfy it. Over the 97, *"MOVEMENT ASKED, LEVEL DELIVERED"* = **0** |
| **D3** route identity owns meaning | **RED — ROUTE IDENTITY STILL OWNS SEMANTICS** | contract identical and plan identical for "grew the most balance" vs "added the most balance". But route **precedence** still decides which of two eligible routes answers: three canary questions moved `temporal_compare` → `period_change_analysis` on the owner commit, with identical economics. The *contract* no longer differs; the *answer* still can. Not fixed here, per instruction |

---

## 5. Ranking census — re-run, 97 questions

```
ranked_level 81 · ranked_level_no_dimension 13 · ranking_underspecified 3 · ranked MOVEMENT 0
ordering_of over the 97:  level 95 · unstated 2
contract says ranking 95 · route says ranking 97 · disagree 2 (contract more conservative)
route recognised as period_change: 0 of 97
DELIVERED 58 · ranking actually applied 0
executed by: (no route) 78 · concentration_analysis 9 · geo_exposure 7 · risk_limits 3
```

**The previous finding is confirmed, not corrected: 97 ranking-language
questions, 0 genuine ranked historical movements.**

**Representation is not sufficiency.** Building the generic plan for all 97:
**82 unblocked, 15 blocked**, every one on *"operation.ordering_direction /
ordering_basis"*.

---

## 6. Honest denominators — kept apart

| denominator | count | note |
|---|---|---|
| **legacy equivalence** | **1** | the single delivering C7-owned corpus question (six-snapshot control); **0** on a production-shaped book |
| **new capability** | **1 of 3** pre-registered RM cases execute | labelled NEW-CAPABILITY EXECUTION PROOF; never counted as equivalence |
| **refusal preservation** | **7** | C7-owned refusals that must stay refusals; all 7 are honest disclosures today |

Not combined. Legacy equivalence of **1** cannot support route migration.

---

## 7. Canary — no movement of any kind

```
breaches 7 (unchanged) · grade movements 0 · route movements 0 · answer-text movements 0
none new · none cleared · F4 still unexercised
```

Nothing moved into WRONG or SILENTLY INCOMPLETE.

## 8. Test denominator — stated, not inflated

**14 modules, 382 tests: 374 passed, 8 failed.** All 8 failures are **pre-existing
at `174d14d`** — verified by running the same modules in a C6-close worktree,
where the identical 8 fail. **0 introduced.**

The full estate was **not** run: it does not complete in this environment. That
is the measured denominator and no broader claim is made.

---

# Verdicts

### 1. C7 CONTRACT VERDICT — **C7 CONTRACT STILL INCOMPLETE**

Four named gaps, each with its exact reason: **no measure** when the question
names none (row 2); **no default comparison period** when the question names none
(row 3); **ranking limit** carried on 26 of 97 with an incomplete number
vocabulary (row 10); **no receipt representation at all** (row 12).

### 2. C7 PLAN VERDICT — **GENERIC PLAN STILL REQUIRES ROUTE SEMANTICS**

A ranked movement that names its measure and its period plans and executes
generically, mutation-controlled. One that names neither does not: the defaults
live in `period_change`'s own governed selection policy. 15 of 97 ranking
questions cannot be planned at all.

### 3. C7 EVIDENCE VERDICT — **EQUIVALENCE DENOMINATOR INSUFFICIENT**

Legacy equivalence = **1**. New-capability evidence = 1 of 3, kept separate and
not offered as equivalence.

### 4. C7 ROUTE RECOMMENDATION — **RADICALLY REDUCE** *(not implemented)*

Unchanged from the previous report and now better evidenced: ranking is applied
on 0 of 882, and the ranked-level capability the corpus actually asks for (97
questions, 58 delivering) is served by other routes already.

---

## Can ranked historical movement now be expressed and executed as ordinary composition?

**Partly — and the boundary is exact.**

**Yes** when the question names its measure and its period: one generic builder
produced periods, population, measure, grouping-with-alternates, compare and
rank from contract values alone, and executing those primitives without touching
`period_change_route` reproduced the movement per group, the direction filter and
the order — with five of five mutation controls discriminating.

**No** otherwise, and for three reasons that are not ranking's:
1. **no measure** on the contract when the question names none;
2. **no default comparison period** when the question names none;
3. **no receipt** — the plan vocabulary has no evidence primitive, so a composed
   answer today carries numbers and no audit trail.

None needs *parser or recogniser shape logic* — which is the encouraging half.
All three need **contract fields and one executor concept** (a governed default
measure set, a governed default period pair, and a receipt step).

## If C7 were reduced next, what must still leave `period_change_route.py`?

Beyond the ranking path already identified (`_rank_subject`, the three
vocabularies, `resolve_rank_intent`, `apply_ranking`, `_rank_refusal_envelope`,
`_rank_rows`, `build_rank_answer`):

1. **K2** — the span honour-or-clarify rule, which rewrites `period_request` in
   place. No shared layer owns it.
2. **The default measure set** — the overview selection policy that lets the
   route answer "how did the book change" when the contract carries no subject.
   This is the row-2 gap, and it is real product semantics, not a route quirk.
3. **The default period pair** — latest versus previous, when the question names
   none. The row-3 gap.
4. **K1** — "a ranked dimension *is* the requested metric", which overwrites
   `requested_fields` and drops the alternates the contract now carries.
5. **K7** — the false refusal that says a book lacks a dimension it has. Deleted,
   not moved.
6. **K3–K6** — the ordering of the rank guard against the span guard, the
   concept suppression, the bridge inclusion and the composition-focus flag.
7. **The receipt** — `metadata.rankedMovement`, the estate's only element-level
   evidence channel, is published by this route alone. Reducing the route without
   moving it removes the only place honouring is verifiable.
