# Conversion 7 — pre-registration and STOP assessment

**Candidate:** `period_change` (`mi_agent_api/period_change_route.py`, 1,112 lines).
**Base:** `174d14d` (C6 close). **No C7 production code has been changed.**

Every gate below was measured before any conversion decision, and the permanent
adversarial compound-question canary bank was frozen and committed (`bf395d8`)
before any of them ran.

> **VERDICT: STOP — C7 NOT AUTHORISED.**
> Five of the seven pre-registered STOP conditions fire before conversion
> begins. §7 states each one with its evidence.

---

## 0. The canary bank, frozen first

`question_interpretation/compound_canary_bank.yaml` — 33 adversarial compound
cases across 10 hazard families, 7 invariants, 3 composition lattices.
Instrument: `migration_phase0/compound_canary.py`. Guard:
`tests/test_compound_canary_bank.py`. Baseline:
`migration_phase0/COMPOUND_CANARY_FREEZE.json`.

A compound question is one declaring two or more elements owned by two or more
semantic owners. The bank pins **invariants**, never answers; the freeze
observations record what the estate did on the day, defects included, as a
baseline rather than a target. The executed guard asserts **movement**: a grade
improving fails as loudly as one regressing, because unexplained capability
movement is what stops the programme.

At freeze: **9 invariant breaches, 20 unevidenced elements across 8 cases, 1
family (F4, Top-N) unexercisable.** Four breaches are attributed to named
defects, all pre-existing and none introduced by this work:

| | breach | statement |
|---|---|---|
| **D1** | I5 refusal reason is true | "which region grew the most" is refused as *"region … is not a governed period-change dimension for this book"*. The book carries `geographic_region_obligor`, which **is** a governed cross-asset period-change dimension, and `requested_dimension_terms` already returned it as an alt. The route passes only the primary (`collateral_geography`, absent) into the analysis. **The refusal reason is false.** |
| **D2** | I2 no silent substitution | "added the most balance" leaves the route entirely and is answered by a single-period grouped bar — a declared **movement** answered with a **level**, disclosed nowhere. |
| **D3** | I4 phrasing is not meaning | "grew the most balance" and "added the most balance" declare the same thing. The verb, not the meaning, selects the route. |
| **D4** | I7 period order | "How did LTV change since last month?" is answered *"moved from 36.5% in 2026-05 to 37.4% in 2026-04 … (+2.33%, up)"*. The pair runs backwards; a fall is reported as a rise. Route is `temporal_compare` (C5), not the C7 candidate. |

Non-vacuity proven by negative control: a planted grade change, a planted breach
and a planted unexercised family each fail their guard.

**UNEVIDENCED is a third grade, and it is a finding.** `metadata.rankedMovement`
is the estate's only element-level evidence channel and exactly **one route
publishes it**. Off that route, a declared element cannot be shown to have been
honoured either way. Collapsing that into DROPPED would have scored every
non-`period_change` answer as a defect and baked *"route identity determines
meaning"* into the instrument — a condition this programme stops on, not one its
own tooling may assume.

---

## 1. C7 owned-surface census

`python -m migration_phase0.route_ownership_period_change [--depth 2|6]`

882 distinct corpus questions, executed through the live `/mi/query` path.
Routing read from execution, never from wording. The C6 non-vacuity rule is
carried forward and is permanent: `ok=True` with zero published rows is **not**
delivered.

**Two route labels, one route.** The module publishes `period_change_analysis`
from every ordinary path and the bare `period_change` from the
span-clarification envelope. A census counting only the first undercounts the
surface and misses the entire clarification partition.

```
                        depth 2 (production-shaped)   depth 6 (control)
owned                                 8                      8
  DELIVERED                           0                      1
  REFUSED                             8                      7
```

**The fixture is ruled out.** C6's recorded correction was that its measurement
had been right and its *denominator* wrong, so "the book is thin" had to be
eliminated before "the route cannot answer" could be asserted. Deepening the
book from two governed snapshots to six moves **exactly one question**. The
owned surface is 8 either way.

The 8, in full:

```
REFUSED  Show funded balance evolution from October to November.
REFUSED  How has the profile of our new lending changed over the last few months?
DELIVERED Has the risk and borrower profile of new business changed recently?
REFUSED  How has lending to the front book changed compared with the back book?
REFUSED  How has the balance of the front book loans moved relative to the back book loans?
REFUSED  Compare how the the front book and the back book books have changed over the last few months.
REFUSED  How has lending to North West changed compared with Scotland?
REFUSED  How has the balance of North West loans moved relative to Scotland loans?
```

All seven refusals are honest disclosures in the estate's controlled
non-substitution wording (*"I understood that you asked for X, but that could not
be applied … I have not substituted"*). **No silent drop occurs on the owned
surface.** That is a real positive result and it is recorded as one.

### The finding that decides C7

```
corpus questions carrying rank language        : 97
of those, owned by period_change               :  0
questions on which ranking was APPLIED (of 882):  0
```

**Not one corpus question that period_change owns carries rank language, and
ranking is applied on none of the 882.** The 97 that do carry it route
elsewhere:

```
(no route)                78
concentration_analysis     9
geo_exposure               7
risk_limits                3
```

The route's distinctive semantic content — P1C ranked period-over-period
movement — is **unreachable from the shipped corpus**. It is reachable only from
canary questions that spell the canonical registry field name
("geographic region obligor"), which is not language a reader uses.

---

## 2. Semantic-owner inventory

`python -m migration_phase0.semantic_owner_inventory_c7`

Attribution by evidence, not by name. The criterion is mechanical and cannot be
argued with:

> **A function that reads the raw question string is interpreting it.**
> Passing the question to *another owner* is delegation. Slicing it, comparing
> it, or matching it against *this module's own* vocabulary is interpretation.

The criterion was corrected once during construction, and the correction
matters: the first version treated a call to a module-local regex as delegation,
which laundered `_rank_subject` — the module's clearest interpreter — into the
delegating column. Any module could have hidden interpretation behind a regex.

```
INTERPRETS      16 lines    1.6%   _rank_subject
VOCABULARY      23 lines    2.3%   _NARRATIVE_RANK_SUBJECTS, _RANK_SUBJECT_LEAD_RE,
                                   _RANK_SUBJECT_SKIP, _BASIS_UNITS, _PROSE_RUNNERS_UP
DELEGATES      345 lines   35.1%
RENDERS        190 lines   19.3%
ADAPTS         320 lines   32.6%
STRUCTURE       88 lines    9.0%
```

**The static hypothesis holds at line level.** Semantic ownership is
`_rank_subject` plus the vocabularies: **39 lines, 4.0% of an 1,112-line
module**. The module is large and almost none of its size is meaning it owns.
The remaining 96% is adapter, rendering and structure, exactly as the inventory
predicted.

### But the line count cannot see the decisions that matter

`route_period_change` delegates *every* reading of the question — so it scores
DELEGATES — and then takes **seven decisions of its own from what the owners
returned**. A decision taken from a delegated result is still a decision. All
seven are route-local. Each is anchored to a literal source substring and the
anchors are re-verified on every run, so a stale inventory fails loudly rather
than describing a module that no longer exists.

| | decision | why it matters |
|---|---|---|
| **K1** | a ranked dimension **is** the requested metric — overwrites `mode` and `requested_fields` | drops `rank_intent.alt_fields`, which the resolver returns *precisely so an availability difference is never read as a substitution*. The mechanism behind **D1**. |
| **K2** | honour the stated span, or clarify — rewrites `period_request.requested_start` in place | a genuine product rule owned by no shared layer |
| **K3** | ranking is resolved **before** the analysis, so a rank refusal returns before the span guard runs | a question with both an unrankable dimension and an unhonourable span is told only about the dimension |
| **K4** | suppress `requested_concepts` when ranking | changes which governed concepts the analysis covers |
| **K5** | when to reconcile the book (`include_bridge`) | |
| **K6** | ranking implies composition focus | |
| **K7** | reinterpret `FAIL_NO_ELIGIBLE_FIELDS` as a dimension refusal | the statement it substitutes is the **false** one recorded as D1 |

**Under the C6 cost correction — cost scales with the decisions still owned, not
with route size — C7's driver is 1 interpreter + 3 vocabularies + 7 composition
decisions = 11 semantic decisions, not the 4 the static inventory predicted.**

---

## 3. Four-part dependency matrix

`python -m migration_phase0.c7_dependency_matrix --depth 6`

The C6 standard, unchanged: **REPRESENTED** (the contract carries the fact),
**OWNER AGREEMENT** (the contract's value equals what the shipped route
decides), **PLAN CONSUMABLE** (a plan step is buildable without re-reading
English), **DELIVERED** (a real delivered, non-empty case exercises it).

```
dependency                    repr    owner   plan    delivered
--------------------------------------------------------------------
dataset                       GREEN   GREEN   GREEN   GREEN n=1 THIN
measure / subject             GREEN   GREEN   GREEN   GREEN n=1 THIN
comparison periods            RED     GREEN   GREEN   RED
source scope                  GREEN   GREEN   GREEN   GREEN n=1 THIN
row predicates                GREEN   GREEN   GREEN   GREEN n=1 THIN
ranking: requested            GREEN   RED     GREEN   RED
ranking: dimension            RED     RED     RED     RED
ranking: direction            RED     RED     RED     RED
ranking: basis                RED     RED     RED     RED
ranking: top N                RED     RED     RED     RED
span honour-or-clarify (K2)   RED     RED     RED     RED

MATRIX: RED on 7 of 11
```

**`OperationClaim.modifiers` is empty on all 97 ranking questions.** The
contract carries *that a ranking was asked for* and **nothing** about what to
rank, which way, on what basis, or how many. All four are resolved by the route
from raw English through `rank_request.detect_rank_request`:

```
dimension : 14 distinct values resolved      contract carries NOTHING
direction :  1 distinct value  resolved      contract carries NOTHING
basis     :  3 distinct values resolved      contract carries NOTHING
top N     :  3 distinct values resolved      contract carries NOTHING
```

Two further readings from the same measurement:

* **`ranking: requested` fails OWNER AGREEMENT.** The route reads 97 questions
  as rankings, the contract reads 111, and they agree on 95 — **they disagree on
  18**. A conversion that took the contract's word would change the answer to
  eighteen questions.
* **Direction resolves to exactly one value.** `increase` on 81, `None` on 16;
  `decrease` is resolved on **no corpus question at all**, although
  `_DECREASE_RE` exists. The decrease vocabulary is dead against the shipped
  corpus.

`comparison periods` is RED on representation for this route specifically:
`TimeClaim.comparison_periods` is populated on 5 of 882 corpus questions and on
**none** this route owns. C4 bridged the field; this route does not receive its
period pair through it — it takes it from the recogniser's `period_request`.

### The four green cells are THIN

All four rest on **the same single delivered question**. C6 pre-registered
delivered *minimums* — 8 ordinary funded series, 1 penny-exact filtered case, 5
weekly frames × 5 governed stages — precisely because a cell can go green on one
case and read like proof. One case is a case. It is not a denominator.

---

## 4. Equivalence denominator

The number C7 would have to measure equivalence over.

```
owned surface                                            8
delivered, production-shaped 2-snapshot book             0
delivered, 6-snapshot control book                       1
owned questions carrying rank language                   0
questions in the corpus on which ranking was applied     0
ranked deliveries reachable at all (canary bank only)   12
```

The 12 exist only because the canary spells the canonical registry field name.
They are **synthetic-phrasing, corpus-unreachable** evidence and must be
reported as such, exactly as C6's fixture evidence was reported as
*fixture-proven, production-data-unexercised*.

Against C6's registered minimums:

| | C6 required | C7 has |
|---|---|---|
| ordinary delivered series | 8 | **1** |
| penny-exact filtered case | 1 | **0** |
| governed grid | 5 × 5 | **0** |
| questions whose delivered economics may move | 0 | 0 *(of 1)* |

**A C7 equivalence measured over this surface would compare one delivery and
seven refusals against one delivery and seven refusals.** C6's own ruling —
*"an equivalence measured over refusals alone is rejected"* — disposes of it.

---

## 5. Pre-registered cost thresholds

Registered **now**, before any conversion decision, so they cannot be fitted to
an outcome. Unit: raw production diff lines added + deleted, hunk-classified,
never net-executable. Split three ways per the C6 correction.

Verified history, as published, nothing reclassified:

| | route | shared | route-specific | total |
|---|---|---|---|---|
| C1 | `portfolio_summary` | 200 | 176 | 383 |
| C2 | `period_movement` | 138 | 144 | 282 |
| C3 | `geo_exposure` | 21 | 129 | 151 |
| C4 | `funded_bridge` | 65 | 154 | 219 |
| C5 | `temporal_compare` | 50 | 148 | 198 |
| C6 | `evolution` | 78 | 25 | 103 |

### Shared — contract extension: **≤ 145**

| component | analogue | measured | estimate |
|---|---|---|---|
| four new `OperationClaim` fields + validation + `as_dict` | C6 registering `row_predicates` in the contract guard | — | **35** |
| projection populating them from the existing resolver | `grouping_concepts` (governed concept, guarded) | 31 | **30** |
| plan-layer rank accessor | `span_from_claim` 24 / `grouping_concepts` 31 | 24–31 | **30** |
| contract pre-registration guard | C6 | 10 | **10** |
| section documentation | every prior conversion | 14 | **14** |
| | | | **119 predicted** |
| justified margin | one unforeseen guard or state case | | **+26** |
| | | | **≤ 145** |

**Higher than every prior shared cost but C1 and C2, deliberately.** C7 is the
only conversion in the programme that must **extend** the contract rather than
read it. Budgeting below C6's 78 would be budgeting from conversion number —
the exact error the C5 re-baseline exists to correct. The weakness is stated
rather than hidden: the contract-extension component has **one weak anchor and
is not a rate**.

### Semantic — route-local meaning removed: **60 – 120**

Removing `_rank_subject` and the three vocabularies is −39. Of the seven
composition decisions, K1 and K3–K6 become plan inputs, K2 needs a shared home,
and **K7 must be deleted rather than moved** — it substitutes a false statement.
A figure far below 60 would mean the decisions were carried across rather than
converted, making the conversion shallower than C1–C6 received.

### Adapter / rendering: **≤ 40, and every line justified individually**

510 lines are ADAPTS + RENDERS. They consume a result object the conversion does
not change, so movement here should be near zero. Anything above 40 means the
conversion is reshaping presentation, which is not migration.

### Total: **≤ 305.** Predicted landing zone ≈ 130 shared + 90 semantic + 15 adapter ≈ **235**.

---

## 6. Pre-defined C7 verdict rules

**COMPOSITIONAL ARCHITECTURE SUPPORTED** — shared ≤ 145; semantic 60–120;
adapter ≤ 40; total ≤ 305; the ranking axis added as *one* governed axis and no
other primitive; K1–K7 removed or given a shared owner; canary grades move only
where each movement is attributed; economics, payload and receipt equivalent.

**CONTRACT EXTENSION UNDERESTIMATED** — technically clean, but the extension
requires generic work not identified in §3. Record what was missed and extend
the inventory before any further conversion. Do not proceed automatically.

**COMPOSITIONAL ARCHITECTURE FALSIFIED** — the ranking facts cannot be expressed
as one governed axis without route-shaped special cases; or the conversion
forces structural redesign of the plan layer, the executor contract or the
receipt.

---

## 7. STOP assessment

The standing STOP rules, each answered from the evidence above.

### FIRED — a new semantic primitive

`OperationClaim.modifiers` is **empty on all 97** ranking questions. Ranking
dimension, direction, basis and Top-N are not representable in the contract at
any strength. C7 cannot begin without adding four new semantic facts. This is a
**contract-extension task before it is a conversion**, which is exactly what the
C5 re-baseline predicted when it ranked `period_change_analysis` third and
wrote *"ranking is not represented in the contract at all"*, and what
`mi_target_state_contract_closure.md` §2 recorded independently as this route's
**blocker: ranking**.

### FIRED — downstream raw-question interpretation

The route resolves four ranking facts from raw English
(`rank_request.detect_rank_request(question, term)`), plus `_rank_subject` over
three vocabularies. Removing that is not conversion work; it is moving a
resolver upstream and inventing the contract fields to carry its output.
`mi_target_state_contract_closure.md` §2 records this route as re-reading raw
text **×4** — the highest count of any route in the estate.

### FIRED — route identity determines meaning

Canary **D3**: "grew the most balance" is answered by the governed period-change
route with a ranked movement; "added the most balance" is answered by a
different route with a **level**. Same declared elements, different meaning,
selected by the verb. Canary **D2** shows the reader is told nothing. The
programme cannot convert a route whose meaning is decided by which route catches
the question.

### FIRED — bespoke exceptions to the compositional plan

**K1** ("a ranked dimension *is* the requested metric") and **K7** ("a
controlled no-eligible-fields failure *is* a statement that the dimension is not
governed") exist nowhere else in the estate, and K7's statement is **false** —
it asserts the book does not carry a dimension the book does carry. A conversion
would have to carry both as exceptions or fix them, and fixing them is
capability movement requiring its own pre-registration.

### FIRED — vacuous / empty evidence

Owned surface 8. Delivered **0** on a production-shaped book, **1** on a
six-snapshot control. All four green matrix cells rest on that same single
question. Zero owned questions exercise ranking; ranking is applied on **0 of
882**. C6's ruling stands: *an equivalence measured over refusals alone is
rejected*.

### NOT FIRED — new route-shape branching

C7 does not *require* new route-shape branching. K3's ordering is pre-existing.

### NOT FIRED — unexplained capability movement

No conversion has run. The canary bank is now in place to detect it, and it
fails on movement in either direction.

---

## 8. Verdict, and what should happen instead

> **STOP — C7 NOT AUTHORISED. 5 of 7 conditions fired.**

C7 was never a conversion. It is a **contract-extension task wearing a
conversion's clothes**, and the measurement says so three independent ways: the
matrix (7 of 11 red, all four ranking facts unrepresented), the census (0 of 882
questions exercise the capability), and the closure record written before this
cycle (*blocker: ranking*).

The recommended next task is **not** to convert `period_change`. It is:

1. **Extend the contract with the ranking axis** — dimension, direction, basis,
   Top-N as four governed facts on `OperationClaim`, pre-registered and measured
   as a contract change with its own thresholds, not folded into a conversion.
2. **Resolve the 18-question disagreement** between what the route calls a
   ranking (97) and what the contract calls one (111) — before either is
   trusted, because a conversion taking the contract's word changes eighteen
   answers.
3. **Fix D1 and D2 on their own merits**, each with pre-registered blast
   conditions. They are live wrong-refusal and silent-substitution defects
   reaching readers today, and neither needs C7 to be fixed.
4. **Decide whether the capability should survive at all.** A 1,112-line module
   whose distinctive semantics **no shipped question reaches** is a product
   question before it is an architecture question, and this assessment does not
   answer it.

Reported before, and not proceeding to, the 7/7 architectural thesis assessment.
