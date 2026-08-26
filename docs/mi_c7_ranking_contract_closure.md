# C7 — ranking contract closure, and what `period_change` is actually for

**Branch** `claude/mi-query-agent-c7-2tlhr6`. **The route was not converted, and
must not be** — see §6 and §9. What was done: D4 fixed in isolation, the
ranking/ordering contract closed, and the product question answered with
evidence.

---

## Phase 0 — evidence gate: **PASSED**

Every C7 figure was recomputed from the repository and from execution, not read
from the previous report. All reproduced exactly:

```
canary        9 breaches, 33 grades byte-identical, unexercised ['F4'],
              negative control still fails all three executed guards
inventory     INTERPRETS 16 (1.6%) · VOCABULARY 23 (2.3%) = 39 lines / 4.0%
              7 of 7 composition decisions route-local
census d2     8 owned · 0 delivered · 8 refused · 0 with rank language
census d6     8 owned · 1 delivered · 7 refused
matrix        RED on 7 of 11; 4 delivered cells THIN on one shared question
ranking       route 97 · contract 111 · both 95 · disagree 18
              OperationClaim.modifiers empty · ranking applied on 0 of 882
```

C6 remains closed: nothing under `mi_agent/`, `mi_agent_api/`,
`question_interpretation/`, `engine/` or `trakt_core/` differed from `174d14d`.

---

## Phase 1 — D4, fixed in isolation

Full detail in `docs/mi_d4_comparison_direction_preregistration.md`. Summary:

**Root cause, one expression.** `_compare_recognizer` built the pair as
`["latest", <relative>]` when a question named a relative period but no explicit
month. `latest` **is** the closing period, so the comparison opened at the close.
Two consequences: the sign inverted, and `pct = abs/va` then divided by the
closing value, so the magnitude was wrong too.

**Single owner, and the narration was NOT patched.** The plan declares
"b relative to a", the executor computes `vb - va`, the receipt reports what
executed and the prose reports the receipt. Everything below the parser is
faithful — they agreed on a reversed pair. Compensating in the sentence would
have left `absoluteDelta`, `percentageDelta` and `direction` inverted on the
receipt.

**Exact economics.**

| | before | after |
|---|---|---|
| book, since last month | £21.1m (06) → £18.9m (05), **-10.57% down** | £18.9m (05) → £21.1m (06), **+11.81% up** |
| LTV, since last month | 38.6% (06) → 38.1% (05), **-1.35% down** | 38.1% (05) → 38.6% (06), **+1.37% up** |

Both denominators are stated so the change of base is visible: `2229701.57 /
18872801.63 = +11.81%`, not `/21102503.20 = 10.57%`.

**Scope.** Only the branch that *invents* an order. The 8 `explicit_pair`
questions are byte-identical — "compare November and October" states its own
order, and reordering it would substitute our chronology for the reader's
question. `explicit_plus_relative` is untouched because whether "last month"
precedes a named October is not decidable from the question.

**Canary movement: exactly as pre-registered.** 9 → 7, the two I7 clearances,
**0 grade movements**, 0 new breaches.

**The registered blast was WRONG and is reported as wrong.** §6 registered two
assertions; it was three, across three modules. The blast search grepped for
`compare_periods`; the missed module names the field `comparison_periods`. A
search narrower than the thing it searches for is not a measurement. Acceptance
condition 5 is **BREACHED**; the other five hold.

**Regression scope actually achieved, stated as such.** The full estate could not
be run here — `tests/` reaches ~3% in fifteen minutes and contains modules that
hang past a 120s per-test timeout. Two earlier attempts each returned a
confident "introduced 0" from a run that had aborted before executing a single
test. The honest denominator is **12 modules** that own or exercise the changed
recogniser, the projection, the comparison plan and the canary, run
module-by-module in two worktrees:

```
test_mi_analytical_intents           36 passed  ->  36 passed
test_comparison_periods_structural   14 passed  ->   2 failed   INTRODUCED, fixed
test_conversion2_period_movement      5 failed  ->   5 failed   pre-existing
```

Everything outside those 12 modules is **UNMEASURED** and reported as unmeasured.

---

## Phase 2/3 — what C7 is for, and where D1–D3 actually break

### The distinction the estate could not make

| | example | contract before | contract now |
|---|---|---|---|
| **C. level** | "which region has the **largest** balance" | `operation=ranking` | `ranking`, `ordering_of=None` |
| **A+B. movement** | "which region **grew the most**" | `operation=ranking` | `ranking`, `ordering_of=movement` **when an owner marks it temporal** |

Before the closure these were **indistinguishable**: same `operation.type`, same
subject, same dimension role. That is the missing generic concept, and it is why
a question asking which region GREW could be answered with which region IS.

### D1–D3 decomposed, per the requested layers

| defect | recognition | binding | **contract representation** | planning | execution | rendering |
|---|---|---|---|---|---|---|
| **D1** wrong refusal | ok | resolver found BOTH fields | **THE BREAK** — `DimensionClaim` held one `candidate_concept`; the alternates were discarded as `_alt` | — | K1 also passed only the primary | K7 stated a falsehood |
| **D2** movement→level | **THE BREAK** — `recognise()` returns `no_change_language` | — | contributory: nothing said the ranking was over a movement | — | — | — |
| **D3** "grew" vs "added" | **THE BREAK** — same as D2 | — | — | — | — | — |

**D3's cause is not the contract.** Measured: for "grew the most balance" and
"added the most balance" the contract is **byte-identical** — `operation=ranking`,
`subject=current_outstanding_balance`, dimension role `grouping`. Only
`period_change.recognition.recognise()` differs (`matched=True` vs
`no_change_language`). The contract already contained enough to make them
identical; **route selection ignores it and reads raw English instead.** This
corrects the earlier C7 report, which attributed D3 to the contract.

---

## Phase 4 — the design, and the alternative rejected

The contract has an established pattern, applied twice:

```
trend_window      (wording) -> window_periods      (the magnitude)
comparison_period (wording) -> comparison_periods  (the values)
```

Each time, the defect closed was identical to C7's: *a consumer had to ask the
owner again because the contract carried the wording and not the values.*

**Design 1 — extend `OperationClaim` (CHOSEN).** `type == RANKING` is the
wording-level fact; add the values as typed fields. This is the third
application of an existing pattern, not a new primitive.

**Design 2 — a new top-level `OrderingClaim` (REJECTED).** Cleaner in isolation,
but ranking *is* an operation, so a separate claim splits one fact across two
homes and adds a top-level primitive where an extension suffices. The brief's
own preference ordering settles it.

**Stringly-typed `modifiers` was rejected outright.** Encoding
`"basis:absolute"` would make every consumer split a serialisation back into
structure — precisely the defect `comparison_periods` exists to close, and the
brief's "do not overload unrelated fields" rule.

```python
OperationClaim:  ordering_direction  increase | decrease | either
                 ordering_basis      absolute | percent | share | count
                 ordering_limit      int >= 1
                 ordering_of         level | movement          <- the new concept
                 orders_a_movement   property; False when unstated

DimensionClaim:  alternate_concepts  the other governed fields the term resolves to
                 candidate_concepts  property; every field it could bind to
```

`ordering_of` never defaults. **"Does not say" must not read as "ranks a
level"** — a consumer defaulting the unknown reintroduces the substitution the
field exists to prevent.

### The required property, tested

> After projection, `"Which region added the most balance since last month?"`
> must be fully meaningful without re-reading English, knowing the route, or
> consulting any C7-local vocabulary.

**Partly met, and the gap is pinned rather than papered over.** Direction, basis,
limit and the dimension (with alternates) are all carried. `ordering_of` is
**not**, because no owner tells the contract the question is temporal:
`_compare_recognizer`'s trigger vocabulary does not fire on "grew … since".
Deriving it here by reading the question would put a **fourth** change-language
reader in the estate — the defect class this programme exists to remove — so
`test_the_known_gap_is_recorded_rather_than_papered_over` asserts the `None` and
says what closing it would take.

---

## Phase 5 — ranking recognition census, all 97

`python -m migration_phase0.ranking_recognition_census`

### The finding that decides the product question

```
INTENDED CLASSIFICATION (derived from the question, not hand-labelled)
  ranked_level                  81
  ranked_level_no_dimension     13
  ranking_underspecified         3
  ranked MOVEMENT                0     <-- ZERO
```

Cross-checked three independent ways over the same 97:

```
questions containing ANY change verb            : 0
questions containing ANY temporal phrase        : 0
questions with a span on the contract           : 0
questions with comparison_periods on the contract: 0
```

**The shipped corpus contains no ranked historical movement questions at all.**

The classifier originally reported one. It was a false positive of my own
making: *"What is the largest geographic concentration **versus** limit?"* — a
ranking of a level against a **threshold**. `versus` was in the change
vocabulary for "October versus November". Counting it would have been this
census committing the exact level/movement conflation it exists to measure. The
instrument is corrected and the reason is written into it.

### Contract versus legacy route

```
contract says ranking : 95        route says ranking : 97      disagree : 2
route RECOGNISED as period_change : 0 of 97
DELIVERED : 58 of 97      ranking actually applied : 0 of 97
executed by: (no route) 78 · concentration_analysis 9 · geo_exposure 7 · risk_limits 3
"MOVEMENT ASKED, LEVEL DELIVERED" over all 97 : 0
```

The 18-question disagreement reported earlier was measured against `contract
says ranking = 111` over the **whole 882**, i.e. it included questions with no
ranking language at all. Restricted to the 97 that carry ranking language it is
**2**, both `contract='amount'` where the route says ranking ("regions with the
most loans", "Which of our limits are currently most at risk?"). Both are the
contract being *more* conservative than the route, not less.

**The D2 class does not occur in the shipped corpus.** It is real — the canary
reproduces it — but no shipped question triggers it.

---

## Phase 6 — product necessity: H1 / H2 / H3

| | hypothesis | verdict |
|---|---|---|
| **H1** | convert — a genuine specialist capability | **NOT SUPPORTED** |
| **H2** | reduce — generic composition plus a small adapter | **SUPPORTED for the ranking half** |
| **H3** | retire | **SUPPORTED for the ranking half only** |

Evidence:

* **Ranked historical movement has zero corpus demand.** 0 of 882. The ~200
  lines of P1C machinery (`_rank_subject`, three vocabularies,
  `resolve_rank_intent`, `apply_ranking`, `rank_request`, `ranking`) serve **no
  shipped question**. It is reachable only by spelling the canonical registry
  field name.
* **All 97 ranking questions are LEVEL rankings**, and 58 already deliver
  through ordinary composition — `geo_exposure`, `concentration_analysis`,
  `risk_limits`, the grouped-chart route. None needs `period_change`.
* **The narrative period-change capability is real but tiny**: 8 owned
  questions, 1 delivering on a six-snapshot book, 0 on a production-shaped one.
  All 7 refusals are honest disclosures in the estate's controlled
  non-substitution wording — **no silent drop occurs on the owned surface**.

| | disposition |
|---|---|
| semantics that must remain | the governed period-change **workflow** (`mi_agent/period_change/*`): snapshot pairing, eligibility, distribution, bridge, units. Its tests pass and its economics are governed. |
| calculations that must remain | all of them — they live in the workflow, not the route |
| rendering that can remain | `RENDERS` 190 + `ADAPTS` 320 = 510 lines, unchanged |
| route interpretation that must disappear | `_rank_subject` (16) + 3 vocabularies (23) + K1–K7 |
| dead / duplicated | the P1C ranking path end-to-end, on corpus evidence |

**Caveat, stated rather than buried.** Zero corpus demand is evidence about the
*measured shipped surface*, not proof of zero product value.
`mi_phrasing_bank_widened.md` already records that natural lender phrasing does
**worse** than the declared bank, so the corpus may under-represent what a lender
would ask. The honest statement is that the capability is **unvalidated**, not
worthless — and that four cycles of measurement have never found a question
demanding it.

---

## Phase 7 — contract closure delivered

Committed at `06a19d8`. Corpus-wide structural evidence, parser and projection
only:

```
ranking-language questions            97
  direction carried            0 ->  80
  basis carried                0 ->  80
  limit carried                0 ->  26
questions raising a dimension        588
  carrying alternates          0 ->  94   (16.0%)
```

Against the brief's prohibitions:

| | |
|---|---|
| no raw-question reads downstream | **held** — every value comes from `detect_rank_request` and the parser's own temporal facts; one test pins that the contract and the owner cannot drift |
| no route identity in the contract | **held** — `period_change.recognition` was deliberately NOT used as the owner of level-vs-movement, precisely because it decides which route claims a question |
| no `_rank_subject` equivalent hiding elsewhere | **held** — no new question-reading code |
| no C7-specific dimension fallback | **held** — alternates come from the existing resolver, for all 588 dimension questions, not for ranking |
| no bespoke `period_change` schema | **held** |
| no unrelated capability expansion | **held** — canary shows zero movement |

**This is not a conversion.** No route reads the new fields; the canary confirms
**0 grade movements, 0 cleared, 0 new**. The contract carries facts nothing acts
on yet, which is what a closure before a conversion should look like.

---

## Phase 8 — four-part matrix, rebuilt

Structural columns re-measured over all 97 after the closure. The **delivered**
column is carried forward from the executed census, and that is sound rather
than assumed: the closure adds fields no route reads, and the frozen canary
shows zero behavioural movement, so delivered coverage **cannot** have changed.

```
dependency                   repr        owner agreement   plan     delivered
ranking: requested           GREEN       GREEN  95/97      GREEN    RED
ranking: dimension           GREEN 84/97 GREEN             GREEN    RED
ranking: direction           GREEN 80/97 GREEN  80/97      GREEN    RED
ranking: basis               GREEN 80/97 GREEN  80/97      GREEN    RED
ranking: top N               GREEN 26/97 GREEN  97/97      GREEN    RED
ranking: level vs movement   RED    0/97 RED               RED      RED
span honour-or-clarify (K2)  RED         RED               RED      RED
```

Four of the five ranking rows moved **RED → GREEN on representation, owner
agreement and plan consumability**. That is the closure working.

**Every delivered cell is still RED, and that is the point.** Ranking is applied
on **0 of 882** corpus questions, so no ranking dependency has any executed
evidence at all.

> ### STOP — C7 EQUIVALENCE DENOMINATOR INSUFFICIENT
>
> Owned surface 8. Delivered 1 (six-snapshot control), 0 (production-shaped).
> Ranked deliveries in the corpus: **0**. C6 required 8 delivered plus a
> penny-exact filtered case plus a 5×5 governed grid.
>
> No fixture questions were invented to inflate it, per the brief.

---

## Phase 9 — canary impact

| after | breaches | grade movements | classification |
|---|---|---|---|
| D4 fix | 9 → **7** | **0** | **authorised defect correction** (M1) |
| contract closure | 7 → **7** | **0** | **unchanged** — nothing consumes the new fields |

**No question moved into WRONG or SILENTLY INCOMPLETE.**

The freeze observations are **not** rewritten — D1–D4 stand as the record of what
was true on 2026-08-25. What advances is the detector's baseline, and only
alongside an `authorised_movements` ledger entry whose arithmetic closes from the
file alone. Two structural guards now enforce both halves: a defect may not be
edited out of history, and the baseline may not sit ahead of the freeze without
an entry explaining it.

---

## Phase 10 — recommendation

### A. D4 verdict

**Root cause** `_compare_recognizer` built `["latest", rel]`; `latest` is the
close, so the comparison opened at the close. **Fix** one expression, at the
single owner. **Blast** 4 questions in the affected branch (1 in corpus); 3
pinned-defect assertions retired. **Regression** 12 modules, 2 introduced (the
missed pinned record, fixed), 5 pre-existing unchanged; everything else
unmeasured and reported so. **Canary** exactly the two registered I7 clearances,
0 grade movements.

### B. C7 contract verdict

> **C7 CONTRACT STILL INCOMPLETE — but materially closer, and the remaining gap
> is named with its owner.**

Direction, basis, limit and dimension alternates are closed and evidenced on the
corpus. **`ordering_of` is representable but unpopulated**, because no owner
tells the contract that "grew … since last month" is temporal. That is a
**parser vocabulary gap with a named owner** (`_compare_recognizer`'s trigger
regex), not a schema gap — and closing it is capability expansion needing its own
pre-registration.

### C. C7 route recommendation

> **RADICALLY REDUCE — and retire the ranking half outright.**

| | |
|---|---|
| **retire** | the entire P1C ranking path in the route: `_rank_subject`, the three vocabularies, `resolve_rank_intent`, `apply_ranking`, `_rank_refusal_envelope`, `_rank_rows`, `build_rank_answer`, K1/K3–K7. **Zero shipped questions reach it.** |
| **keep** | the governed workflow (`mi_agent/period_change/*`) untouched, and the adapter/renderer that turns its result into an envelope |
| **move** | K2 (span honour-or-clarify) to a shared owner — it is a real product rule that no shared layer owns |
| **do not convert** | there is nothing left to convert once the ranking path is gone: the remainder is adapter and rendering, which the compositional plan does not claim |

CONVERT is rejected because the equivalence denominator is 1 and the distinctive
capability has zero demand. RETIRE-entirely is rejected because the narrative
period-change workflow is governed, tested, and does answer 8 corpus questions
honestly.

### D. Re-authorisation package — proposed, NOT executed

| | |
|---|---|
| owned surface | 8 corpus questions, both route labels (`period_change_analysis`, `period_change`) |
| equivalence denominator | **insufficient today.** Required before any reduction: ≥8 delivering owned questions. Achievable only by loading a book with ≥2 governed snapshots into the production discovery root — **not** by inventing fixture questions |
| semantic owners to delete | `_rank_subject` (16 lines), `_NARRATIVE_RANK_SUBJECTS`, `_RANK_SUBJECT_LEAD_RE`, `_RANK_SUBJECT_SKIP` (23), K1, K3, K4, K5, K6, K7 |
| expected plan | `STACK_PERIODS(pair)` → `SELECT_POPULATION(lens)` → `RESOLVE_MEASURE` → `COMPARE`; no ranking step, because no shipped question asks for one |
| rendering retained | `RENDERS` 190 + `ADAPTS` 320 = 510 lines, unchanged |
| cost thresholds | deletion-dominated: **removal 250–400**, shared **≤ 40**, added route-specific **≤ 60**, net **negative**. A positive net total means the reduction became a conversion and stops |
| authorised movements | the 8 owned questions must answer identically or refuse identically; any REFUSED→DELIVERED is capability expansion and stops |
| STOP conditions | C6's, unchanged, plus: **STOP if deletion changes any of the 8 owned answers**; **STOP if the workflow needs any change**; **STOP if the denominator is still < 8** |

### E. Thesis impact

> **Does C7 show the compositional architecture is missing an important generic
> semantic concept, or that a legacy route has been carrying semantics that
> belong in the common contract?**

**Both, in a specific proportion, and the proportion is the finding.**

* **Carriage, not concept — the larger part.** Direction, basis, limit and
  dimension alternates were all *already resolved* by owners the estate had.
  Nothing was missing but a place to put them. That is a legacy route holding
  values the contract should have carried, and closing it cost one schema
  extension and one projection function.
* **A genuinely missing concept — the smaller but decisive part.**
  **LEVEL versus MOVEMENT** was not representable anywhere, and **no owner owns
  it.** The estate has three narrow, disagreeing readers of change language
  (`_COMPARE_TRIGGER_RE`, `period_change.recognition`, and the ad-hoc patterns).
  That absence is what lets "which region grew the most" be answered with "which
  region is largest" — and it is an architectural hole, not a route defect.

> **After closure, can a new ranked historical movement question be supported
> primarily through composition?**

**Almost — and the residue names the next task exactly.** Measure, dimension
(with alternates), scope, filters, direction, basis and limit all compose from
the contract with no parser, recogniser or executor shape logic. What still
requires new work is **one** thing: an owner that decides whether a question is
about a level or a change, and populates `ordering_of`. Give the estate that
single generic owner and ranked historical movement becomes composition. Until
then it needs a recogniser, which is exactly why `period_change` still has one.

**The architecture is not falsified by C7. It is one concept short of complete,
and C7 identified which one.**
