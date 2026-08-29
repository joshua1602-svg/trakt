# One owner for LEVEL versus MOVEMENT

**No C7 execution was wired.** `period_change_route.py` is untouched; no route
reads `ordering_of`. What changed is who decides whether a question asks for a
position or for a change — and everything that decision moved was measured
first.

---

## 1. The hole, measured

Six components inferred the distinction. Five claimed to answer "is this a
two-point change"; the sixth answers a different question and is excluded.

```
reader                                              says MOVEMENT (of 882)
A  period_change.recognition.has_change_language                     17
B  llm_query_parser._COMPARE_TRIGGER_RE                              21
C  spec.temporal_mode == "compare"                                    5
D  interpreter.deterministic's compare branch                        20
E  concentration_query's compare gate                                 0
F  period_change.recognition.TREND_MARKERS  (a SERIES — excluded)     38

union of A–E                                                         30
all five agree                                              852 of 882
```

**No reader was a superset of any other.** Each missed part of the union: A 13,
B 9, C 25, D 10, E 30.

The sharpest instance, found while writing the census: **reader A missed "How
did the balance change since last month?"** — the most canonical movement
question in the estate — because its vocabulary carried "changed", "change in"
and "has changed" but not the bare verb "change".

---

## 2. The owner

`question_interpretation.lexical.temporal_aspect` → `LEVEL | MOVEMENT`, plus the
evidence that decided it. It lives in `lexical` because it reads the question's
vocabulary, which is what that module owns, and because it must not live in a
route.

**MOVEMENT requires positive evidence.** Absence is LEVEL — a question naming no
change asks what the position is, which every route already assumed. Saying it
explicitly is what lets consumers stop guessing.

Three things it deliberately refuses to call movements, each a class reader B
got wrong:

| | example | why not a movement |
|---|---|---|
| against a **forecast** | "compare current funded balance to expected funded" | the second operand is a plan, not an earlier date |
| against a **threshold** | "the largest geographic concentration versus limit" | a limit is not a date |
| two **populations** | "how does the front book compare with our older lending" | cohorts of one snapshot — the `seasoning` owner's axis |

It also refuses a bare `and` between two periods: *"show pipeline by stage for
October and November"* asks for two **levels** side by side. Only an explicit
comparison verb makes `and` a comparison.

A **series is not a two-point movement** either: "balance by month" is a
sequence of levels, already a separate decline reason in the estate.

Owner verdict: **24 of 882.** Against each retired reader:

```
                              reader-only   owner-only
A_has_change_language                   0            7
B_compare_trigger                       6            9
C_temporal_mode_compare                 0           19
D_deterministic                         4            8
E_concentration                         0           24
```

A, C and E are strict subsets — delegating them only widens. B's and D's
reader-only firings are exactly the six rejections above.

---

## 3. Singularity, enforced

Four modules were delegated: `period_change.recognition`, `llm_query_parser`,
`interpreter.deterministic`, `concentration_query`. The retired vocabularies are
**deleted**, not left dormant — `CHANGE_MARKERS`, `COMPARISON_PERIOD_MARKERS`
and `_COMPARE_TRIGGER_RE` are gone, because dead vocabulary is a second owner
waiting to be re-used.

`tests/test_temporal_aspect_owner.py` enforces it structurally: a module-level
collection carrying three or more distinct change words IS an aspect vocabulary,
whatever it is called. Three vocabularies that contain such words without
deciding the aspect (`OVERVIEW_MARKERS`, `BRIDGE_MARKERS`, `TREND_MARKERS`) are
exempt **by name with a written reason**, and a further test fails if an
exemption outlives the thing it excused.

The guard caught its own first draft: counting change words in all string
literals flagged `OVERVIEW_MARKERS` and `BRIDGE_MARKERS`, which decide *mode
within* period_change after the aspect gate has already run. A blunt guard that
cries wolf gets disabled, so it was made precise instead of lenient.

---

## 4. Every question whose contract changed — **113 of 882**

Full contract, field by field, swept in two trees.

```
ordering_of        111     the field was unpopulated before; now says LEVEL or MOVEMENT
temporal_mode        2
compare_periods      2
comparison_periods   2
operation.type       2
intent / aggregation / candidate_concept / state / source / raw_text
```

111 changed **only** in `ordering_of` — the contract now states an aspect where
it stated nothing. Two changed structurally, and both are the D2 class closing:

| question | before | after |
|---|---|---|
| Show funded balance evolution from October to November. | `type=amount`, no periods | `type=movement`, `compare_periods=[October, November]` |
| Show pipeline growth from October to November. | `type=count`, `subject=loan_count`, `aggregation=count` | `type=movement`, `aggregation=sum`, periods carried |

**One loss is recorded rather than glossed:** on the second, `subject` goes from
`loan_count` (filled, and wrong for "growth") to `None` (empty). Not asserting a
wrong measure is better than asserting one, but a filled slot became empty and
that is a movement, not an improvement by definition.

---

## 5. Every route whose plan changed — **2 plans, 1 route**

Plans were built for all 113 changed contracts in both trees, across five route
builders. A plan is a function of the contract, so a plan can only change where
the contract did.

```
ROUTES WHOSE PLAN CHANGED: 2      distinct routes affected: ['temporal_compare']
```

Both go from **BLOCKED** — *"a comparison needs two governed reporting periods …
time.comparison_periods names 0 period(s), not 2"* — to a real plan carrying the
pair. The 111 `ordering_of` changes moved **no plan at all**, because nothing
reads it yet. That was expected; it is now measured.

---

## 6. Every answer and refusal that moved — **4 of 140**

The executable set is bounded exactly: a question can only move if a reader's
verdict changed (30) or its contract changed (113) — union **140**. Executed in
both trees against the production-shaped two-snapshot book.

```
REFUSED -> EMPTY     2
REFUSED -> REFUSED   2
DELIVERED -> *       0
* -> DELIVERED       0
```

| question | before | after |
|---|---|---|
| Show funded balance evolution from October to November. | `period_change_analysis` — *"153 days from the nearest usable snapshot, beyond the 45-day limit"* | `temporal_compare` — *"I can't compare October and November: requested period(s) unavailable"* |
| Show pipeline growth from October to November. | no route — *"No governed pipeline data is available"* | `temporal_compare` — names the periods it cannot find |
| Are we originating different types of loans now compared with a few months ago? | no route — *"I couldn't map this question to a governed analytic"* | `period_change` — *"You asked about the last 3 months … this book carries 2"* |
| How does recent lending compare with what we were originating earlier in the year? | same | same |

**No delivered economics changed. No question moved into a wrong answer.** Two
non-answers became better-specified non-answers; two "I don't understand"
refusals became governed clarifications.

`REFUSED -> EMPTY` is recorded as an authorised movement, not as equivalence:
`ok=False` became `ok=True` with zero rows, which is a different envelope even
though neither carries a number.

---

## 7. Canary: 0 grade movements, 3 route movements

7 breaches before and after, no grade moved, none cleared, none new. But **three
canary questions changed route** — F8.a, F10.a, F10.b from `temporal_compare` to
`period_change_analysis`, because the retired reader missed the bare verb in
"how did X change" and the owner does not.

Economics identical on the shared metric: `+£3.9m, £12.2m → £16.1m`; LTV
`37.4% → 36.5%`, falling, chronological in both. The LTV answer now reports
**percentage points** (−0.85 pp) rather than a relative change of a ratio, which
is the better unit for an LTV movement.

Recorded as **M2** in the bank's `authorised_movements` ledger, precisely
because a route change that leaves every grade identical is the movement a
grade-level detector cannot see.

---

## 8. What the corpus did not cover

One estate test moved that the corpus measurement could not have caught:
`"Which balances increased between January and April?"` is **not in the 882**.

It named one metric and two periods, and now defers to `temporal_compare` under
the estate's own `DECLINE_INCUMBENT_SINGLE_METRIC_COMPARE` rule — a rule that
was unreachable for this question until the parser began recognising its period
pair. Identical economics either way: `£11.9m (2026-01) → £17.2m (2026-04),
+£5.3m, +44.9%`.

The test asserted `matched`; it is named *"the wider change vocabulary is
recognised"*, so it now asserts the vocabulary and accepts the governed
deferral. **The corpus is not the whole reachable surface**, and this is the
evidence for that.

---

## 9. What is closed, and what is not

**Closed.** One owner. Five readers retired, their vocabularies deleted, the
singularity enforced structurally. The contract can state LEVEL versus MOVEMENT
for the first time, and `orders_a_movement` is False when unstated so no
consumer can default the unknown to level.

**Not closed.** Route **precedence** still decides which of two eligible routes
answers a question both could. That is what moved the three canary questions and
the one test question, and it is a different decision from the one this owner
took over. D3 — "the verb, not the meaning, selects the route" — is narrowed,
not eliminated: the verb no longer selects through five disagreeing
vocabularies, but two routes can still both be eligible and the order settles
it.

**Still not wired.** No route reads `ordering_of`. C7 execution is untouched, and
the recommendation of the previous report stands unchanged: radically reduce
`period_change`, retiring the ranking half that no shipped question reaches.
