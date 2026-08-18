# TRAKT MI Query Agent — P1C: Ranked Period-over-Period Movement

**Date:** 17 August 2026
**Branch:** `claude/mi-query-agent-review-n8d33r`
**Scope:** ranking of the existing governed period-change calculation
**Data:** synthetic `demo_platform` (11,035 loans, £1.96bn, snapshots 2026-04-30 / 05-31 / 06-30)
**Entrypoint under test:** `mi_agent_api.mi_service.execute_governed_mi_query` — the same
capability POST `/mi/query` (React MI Agent) and the Copilot `askTraktMi` action call.

---

## 1. What P1C delivers

The agent can now answer, correctly and from the governed calculation:

| Question | Answer produced |
| --- | --- |
| Which region grew the most last month? | South East, £508.4m → £516.2m (+£7,840,963.14, +1.54%) |
| Which region grew fastest in percentage terms? | North East, +1.8194% (+£544,710.69) |
| Which region added the most loans month-on-month? | South East, 2,402 → 2,420 (+18) |
| Which region increased its share of the book the most? | South East, 26.11% → 26.27% (+0.1591 pp) |
| What were the top 3 regions by growth since last month? | South East, East of England, London |
| Which region declined the most last month? | *"No region decreased… I have not answered with the movements in the other direction instead."* |

Every figure above was reconciled against pandas computed independently from the two
governed snapshots. All four ranking bases agree to 1e-9.

**The single sentence that governs this work is unchanged:** the agent may answer
correctly, answer partially with an explicit limitation, or refuse. It must not silently
answer a materially different question. P1C expands what it can answer; it does not move
that boundary.

---

## 2. The ranking basis convention

"Which region grew the most?" has four defensible readings, and they give four different
winners on this book. The measure the reader names decides the basis, and the basis that
ran is printed on every answer:

| Phrasing | Basis | Winner on this book |
| --- | --- | --- |
| "grew the most", "added the most balance" | absolute balance movement | South East |
| "grew fastest", "in percentage terms" | percentage balance movement | **North East** |
| "added the most loans", "most cases" | loan-count movement | South East |
| "increased its share the most" | share-of-balance movement | South East |

A bare "grew the most" with no measure named is **absolute balance movement**. That is a
documented product convention, not a silent default: the book is measured in balance, and
a bare "grew" about a lending book means money rather than proportion. It is stated in the
execution receipt on every answer, so a reader who meant the growth *rate* can see
immediately that they were given the growth *amount* — and North East vs South East is
exactly the case where that matters.

---

## 3. Architecture — where every number comes from

```
question
   → period-change intent            (mi_agent.period_change.recognition)
   → ranking intent: dimension, measure, direction, Top N
                                     (mi_agent.period_change.rank_request)
   → resolve current and comparison periods
   → EXISTING governed period-change calculation
                                     (mi_agent.period_change.distribution)
   → deterministic ranking of those results
                                     (mi_agent.period_change.ranking)   ← new, pure
   → P0 semantic validation          (mi_agent.execution_receipt)
   → execution receipt
   → answer
```

No figure is recalculated in the ranking layer. It receives `CategoryShift` rows the
governed engine already reconciled and orders them. No model is asked which result looks
largest, and no model is asked to compute a difference. `ranking.py` is pure: no I/O, no
dataset access, no registry access.

Two things the governed engine does **not** provide — absolute movement and percentage
movement — are differences of two figures it *does* provide. They are computed in the
ranking module rather than assumed, so the basis is explicit and auditable.

### Tie-breaking
Inherited from the convention `distribution_change` already uses for its own top-shift
lists: **rank value first, then category name ascending**. Two equal movements always
order the same way, and pandas row order can never decide business semantics.

---

## 4. Route precedence — a defect this work exposed

Before P1C, "Which region grew the most last month?" was captured by `geo_exposure`
(priority 60), a **point-in-time** route, before the governed period-change capability
(priority 85) was ever consulted. The comparison silently disappeared. Since P0 that
produced a refusal rather than a wrong answer, but the capability was unreachable.

Two changes, both narrow:

1. **`geo_exposure` now stands down** for exactly the questions the governed
   period-change recogniser positively claims — it asks that recogniser rather than
   duplicating its vocabulary, so the two cannot drift apart. Anything period-change
   declines still lands in `geo_exposure`, which keeps every point-in-time geography
   question it already answered (verified: "Which region has the largest exposure?",
   "What is the geographic concentration of the book?" both still route there).

2. **The period-change vocabulary gained the missing conjugations of verbs it already
   carried.** It recognised "grown", "rose" and "fell" but not "grew", "growth",
   "declined" or "dropped" — so "which region grew the most last month?" was not
   recognised as change language at all.

A third change is structural rather than vocabulary: the ranking dimension is now resolved
**before** the analysis runs and passed to it as a requested field. Resolving it afterwards
meant a question like "top 3 regions by growth" could produce an analysis that never looked
at geography, and then have nothing to rank.

---

## 5. Truth reconciliation

`tests/test_p1c_ranked_movement_e2e.py` recomputes every expected figure with pandas,
directly from the two governed snapshots, importing no `mi_agent.period_change` code and
no ranking module. The categorisation rule (trim; bucket every blank/placeholder as
`Unknown`) is **restated** in the test rather than imported, so a change to the governed
normaliser shows up as a disagreement rather than being silently mirrored.

Independent truth, 2026-05-31 → 2026-06-30:

```
                          balance_start     balance_end       abs_move   pct_move
South East               508,373,173.44  516,214,136.58  +7,840,963.14     1.5424
East of England          211,823,275.66  213,736,192.37  +1,912,916.71     0.9031
London                   411,957,208.61  413,804,467.49  +1,847,258.88     0.4484
South West               235,629,074.60  237,410,495.59  +1,781,420.99     0.7560
East Midlands            103,123,239.10  104,436,198.85  +1,312,959.75     1.2732
Yorkshire and The Humber  95,926,816.34   96,800,530.68    +873,714.34     0.9108
North East                29,939,670.65   30,484,381.34    +544,710.69     1.8194
Unknown                      385,166.29      323,599.57     -61,566.72   -15.9845
```

The agent's ranking reproduces this ordering and these values exactly, on all four bases.
London's closing count of 1,380 also reconciles to the known ground truth for this book.

---

## 6. The golden bank

27 questions, each with the outcome **class** it must produce. A question landing in a
different class is a silent semantic error whether or not its numbers happen to be right.

| Class | Count | Result |
| --- | --- | --- |
| RANKED — answered by ranking the governed output | 13 | 13 correct, truth-reconciled |
| REFUSED — declined with nothing substituted | 10 | 10 refused |
| NARRATIVE — kept the governed answer that already owned it | 4 | 4 unchanged |
| **Silent semantic errors** | — | **0** |

The bank covers all four bases, both directions, Top N (3 and 5), and every refusal class.
A meta-test asserts the bank keeps that coverage, so it cannot be quietly narrowed.

---

## 7. The negative bank

| Case | Behaviour |
| --- | --- |
| Nothing moved in the requested direction | *"Between 31 May 2026 and 30 June 2026, no region decreased on absolute balance movement. I have not answered with the movements in the other direction instead."* |
| Dimension not governed for period change (LTV band, product type, arrears bucket) | Named refusal: *"…it is not a governed period-change dimension for this book. I have not ranked a different dimension instead."* |
| No governed dimension resolves ("which five segments…") | *"I could not identify a governed dimension to rank from 'segments'."* |
| Forward-looking ranking ("which region will grow the most next month?") | Refused — no projection was run |
| Category present in one period only | Ranked with `presence` stated; excluded from a percentage ranking with the reason "no opening balance, so percentage movement is undefined" |
| Zero opening base | Excluded with its reason, never ranked as zero — ranking it as zero would bury an unknown mid-table and present it as a fact |
| Invalid Top N (0, −1) | Refused |
| Ties | Broken deterministically on category name ascending |
| The `Unknown` bucket | Excluded from a ranking of categories **and disclosed**: it is missing data, not a region. A caller who genuinely wants it ranked passes `exclude_categories=()` |
| Categories that moved the other way | **Counted and stated** ("6 further categories did not increase on this basis and are not listed"), never silently dropped |

---

## 8. How P0 proves the ranking ran

P0 remains an independent control, not a narrator of the route's own claims.

* The route **declares** what it ranked: canonical field, display name, basis, direction,
  Top N, both reporting dates, ranked rows, exclusions.
* The guard **verifies** that declaration against the dimension the question asked for,
  resolved with the *same* resolver the route used (`requested_dimension_terms`). Sharing
  one resolver is deliberate — two dimension vocabularies would be two things to keep in
  step, and a disagreement would refuse every ranked answer.
* A ranking of a **different** dimension is `LOST`, not accepted. A declaration with no
  canonical field is not evidence and is not accepted. A route that ranked nothing gets the
  pre-P1C treatment unchanged.

Pinned by unit tests that construct the declaration directly, so the check is tested
independently of whether the route happens to be correct today.

---

## 9. The execution receipt

```
Calculated: Governed period change · ranked by Collateral Geography ·
absolute balance movement, largest increases first, top 3 of 12 ·
2026-05-31 → 2026-06-30.
```

It states the measure, the dimension, the ranking basis, the direction, any Top N and the
size of the universe it came from, and **both** reporting dates — so a ranked answer can
never imply a period it did not use. `comparisonPeriod` carries the two dates in the
structured summary for callers that render it themselves.

---

## 10. Defects found while building this, and fixed

Four, all outside the ranking feature itself and all found by running the real thing:

1. **`period_movement` was missing from `TEMPORAL_ROUTES`.** "What changed since last
   month?" is answered by a genuinely two-period governed capability (it reads current and
   prior reporting periods and reports the delta), but P0 classified it as point-in-time
   and refused it. A false refusal disables working governed analytics; it does not prevent
   a substitution. Now recognised as temporal, and the question answers again.

2. **A refusal still shipped the result it had just refused.** "Where did balance fall most
   since last month?" returned `ok:false` with a 12-row table titled "Result"; "If
   origination continues at the current rate, what will the balance be at year end?"
   returned `ok:false` alongside a KPI reading **£1.96BN** — the whole-book balance,
   presented next to a refusal, with the answer text "Here is the result for your query".
   A declined answer now ships no data artifacts and states why it was declined.

3. **The age threshold convention did not hold on the LLM path.** Found by the genuine LLM
   run: on *"What is my exposure to borrowers over 85?"* the deterministic parser applied
   `> 85` (**86 loans, £19.4m**) and the model applied `>= 85` (**136 loans, £31.1m**) —
   two materially different answers to one question, decided by which parser happened to
   run. The house convention is a reading of the question's language, so it is now applied
   to the model's own predicate: **only the operator moves**, and only when both parsers
   picked the same field and the same number. A different value, an opposite direction, or
   an equality predicate is a genuine disagreement and is left to the P0 guard. Both paths
   now report 86 loans.

4. **A test scanned generated output as if it were configuration.**
   `test_concentration_dimension_fallback` walks every YAML/JSON file in the repository
   looking for configurations that use a newly-resolvable dimension. It also walked `out/`
   — gitignored run output — so it failed on an MI harness transcript that merely *mentions*
   a dimension. `out/` is now skipped, which restores the test's actual intent.

---

## 11. The 40-question regression bank

Re-run on both parser paths. The bank is deliberately unchanged from P0/P1A/P1B so the
runs stay comparable; the P1C questions live in the golden bank of §6.

**Deterministic path — 6 of 40 questions moved against the P1B baseline, all improvements:**

| id | Question | P1B | P1C |
| --- | --- | --- | --- |
| A2a | Which region has grown the most in the last month? | safe refusal (geo_exposure, comparison lost) | **correct** — South East +£7.84m |
| A2b | Which broker has grown the most in the last quarter? | generic facet refusal | named refusal: broker is not a governed period-change dimension |
| A2c | Which product has grown the most in the last quarter? | generic facet refusal | named refusal |
| A2d | Which borrower type has grown the most in the last month? | generic facet refusal | named refusal |
| B02 | Which segments are driving balance growth this quarter? | unrouted refusal | routed to period_change, refused there |
| B24 | Which part of the book is growing fastest by loan count rather than balance? | unrouted refusal | *"could not identify a governed dimension to rank from 'part'"* |

No question regressed. Nothing moved from correct to incorrect, and nothing moved from
answered to refused.

**Genuine LLM path** (live model, 32 of 40 parsed by the model, 8 short-circuited by
`zero_cost_first`, 0 silent fallbacks):

* **Route agreement between the two parser paths: 40/40.** The specialist-intent carry
  forward built in P1B, plus the question-based period-change recognition, mean the model's
  spec can no longer shadow a governed capability.
* A2a is answered correctly on the LLM path **even though the model returned a generic
  balance-by-region bar with an unrelated `acquisition_date >= 2024-11-01` filter** — the
  period-change route recognises the question, not the spec.
* Three questions differ in outcome between the paths, none of them a silent error:
  * **A4** ("balance by region by borrower type") — deterministic answers by region and
    discloses *"Not applied: borrower type — field is unavailable in this dataset"*; the LLM
    path refuses outright. Both honest; the LLM's is stricter.
  * **B21** ("largest single-loan exposure and its share") — deterministic returns a ranked
    top-10 loan listing; the LLM's spec carried no ranking and is refused by the unranked-
    superlative guard. Neither answers the *share* half of the question.
  * **B04** ("credit quality of new origination vs the back book") — the LLM answers with
    average LTV by portfolio cohort as a proxy. The receipt states exactly that
    (*"Average Current LTV · grouped by Portfolio Cohort"*), so the substitution is
    disclosed, but it is not narrated in the answer text. **Flagged for judgement**, see §12.
* Parser-failure reporting was itself misleading and is fixed: 15 questions where the model
  returned a spec the governed validator rejected were reported as failure category
  `unknown`; they now report `parse_failure`, which is the most common LLM failure mode and
  was hidden behind a placeholder.

---

## 12. Known limitations — what P1C does **not** do

Stated plainly rather than discovered later. None of these produce a wrong answer; each
produces a refusal.

1. **Only dimensions the Business Semantics Registry carries for period change can be
   ranked.** On this book that is collateral geography and two ITL3 region fields. LTV
   band, product type, arrears bucket and origination vintage are refused by name. Widening
   this is a governed **registry/policy** change, not a code change, and was deliberately
   not attempted here.

2. **Adjectival dimension forms are not recognised.** "Top 3 **regional** increases" refuses
   ("could not identify a governed dimension to rank from 'regional'") while "top 3
   **regions** by growth" ranks correctly. The fix belongs in the shared dimension
   vocabulary used by both the parser and the P0 guard, and changing it affects grouping
   behaviour well beyond ranking — so it was left alone rather than patched locally.

3. **One dimension, one measure, one direction, one comparison.** Explicitly out of scope
   and refused: multi-dimensional rankings, multi-measure rankings, rankings across more
   than two periods, forecast rankings, cross-portfolio rankings, and cohort/vintage
   rankings.

4. **"Which regions are we most exposed to relative to last month?" is still refused, and
   should be.** It asks for an *exposure* ranking with a temporal reference, not a *growth*
   ranking. Making the period-change route claim it would answer a different question —
   which is precisely the failure P0 exists to prevent.

5. **B04's measure proxy is disclosed in the receipt but not narrated.** When a question
   names a concept that is not a governed measure ("credit quality") and the model answers
   with a proxy, the receipt states the proxy but the answer text does not. This is a
   judgement call for the product: the current behaviour satisfies the P0 boundary (nothing
   is silent) but a reader skimming the sentence could miss it.

6. **The demo book has no declining region.** The all-declining and mixed-direction paths
   are covered by unit tests against constructed shifts, not end-to-end against real data.

---

## 13. Test evidence

| Suite | Result |
| --- | --- |
| `mi_agent/tests/test_p1c_ranked_movement.py` | 57 passed — pure ranking, reconciled to pandas; P0 evidence verification; receipt rendering |
| `tests/test_p1c_ranked_movement_e2e.py` | 52 passed — end to end through the production entrypoint, truth-reconciled, 27-question golden bank |
| `mi_agent/tests/test_p1b_route_precedence.py` | 65 passed — route precedence across phrasings, age convention on both parser paths |
| **Full repository suite** | **8,196 passed, 30 skipped, 20 xfailed, 0 failed** |

### Reproducing

```bash
python -m demo_platform.run_demo --generate --orchestrate     # once, ~90s

pytest mi_agent/tests/test_p1c_ranked_movement.py tests/test_p1c_ranked_movement_e2e.py -q

python scripts/run_mi_capability_review.py --out out/p1c_det
ANTHROPIC_API_KEY=… python scripts/run_mi_capability_review.py --llm \
    --out out/p1c_llm --compare out/p1c_det/transcript.json
```

The harness prints which parser actually answered each question and warns loudly if no
question was answered from a live LLM parse — so a deterministic fallback can never again
be reported as an LLM evaluation.

---

## 14. Summary

P1C moves ranked period-over-period movement from *unreachable* to *correct and
truth-reconciled*, by ranking a calculation that already existed rather than adding a
second way to compute movement. The trusted answerable surface grew; the P0 safety boundary
did not move. Three pre-existing defects were found by running the real thing and fixed,
one of which — the age convention diverging between parser paths — was producing a
materially different number for the same question depending on which parser ran.

---

## 15. Full 40-question classification, both parser paths

Every question in the regression bank, classified by outcome rather than by `ok` flag.
Deterministic run: `out/p1c_det`. Genuine LLM run: `out/p1c_llm_final` (32 of 40 parsed by
the live model, 8 short-circuited by `zero_cost_first`, **0 silent fallbacks**).

| | Deterministic | LLM |
| --- | --- | --- |
| **Correct** | 5 | 5 |
| **Partial, limitation disclosed** | 5 | 4 |
| **Safe refusal** | 29 | 30 |
| **Silent semantic error** | **1** | **1** |

**Correct (both paths):** A2a (ranked movement — new in P1C), A6, B01 (concentration limits
vs the governing document), B06 (86 loans over 85 — now identical on both paths), B08
(run-rate extrapolation).

**Partial with an explicit limitation:** A4 *(deterministic only — "Not applied: borrower
type — field is unavailable in this dataset"; the LLM path refuses outright)*, A8 *(states
the direction of the LTV difference but not the two averages)*, B21 *(deterministic only —
ranked top-10 loan listing; the "share of the book" half is not answered; the LLM path
refuses)*, B22 *("Not applied: postcode — this answer is reported at ITL3 area level")*,
B25 *("no governed directional differences were observed" — honest, but gives no age
figures)*, B04 *(LLM only — answers "credit quality" with average LTV by portfolio cohort;
the receipt names the proxy, the answer text does not)*.

**Safe refusal:** the remaining 29/30. Every one names what it could not do and states that
nothing was substituted. The five P1C-shaped refusals (A2b, A2c, A2d, B02, B24) now name the
dimension or the period grain that blocked them instead of giving a generic facet message.

### The one silent semantic error — B11, and it is not a P1C regression

> **B11 — "Which region contributes most to the weighted average LTV?"**

The agent returns a chart **titled with the user's own question**, ordered by each region's
weighted-average LTV descending. The first bar is **West Midlands** (43.90%). A reader takes
that as the answer.

It is not. "Contributes most to the weighted average" is a balance-weighted contribution,
and the two rankings are almost inverted:

| Region | Region WA LTV | Share of book | Contribution to the book's 43.15% |
| --- | --- | --- | --- |
| **South East** | 43.14% | 26.28% | **11.34 pp** |
| London | 42.75% | 21.06% | 9.00 pp |
| South West | 43.31% | 12.08% | 5.23 pp |
| **West Midlands** *(shown first)* | 43.90% | 6.20% | **2.72 pp** |
| North East *(shown second)* | 43.92% | 1.55% | 0.68 pp |

The correct answer is **South East**; the agent presents **West Midlands**. The artifact does
carry `concentration_pct`, so the evidence to judge it is on screen, and the receipt states
truthfully that it calculated "Weighted-average Current LTV · grouped by Region" — but
nothing tells the reader that "contributes most" was not the calculation performed, and the
chart title asserts that it was.

**Status:** pre-existing, on both parser paths, and **not introduced by P1C** — it was
present before this work and is not caused by the ranking layer. It is reported here because
the launch gate is *0 silent semantic errors* and this is one.

**Shape of the fix (not attempted):** P0 already refuses a superlative answered over an
unranked population (`detect_unranked_superlative`, built in P0 for B21). "Contributes most
to / drives / accounts for most of ⟨an aggregate⟩" is the same failure in a different
costume: a **contribution** question answered with a per-group value ranking. It needs its
own detector, and then either the governed contribution calculation (share × value) or a
refusal. It is a self-contained P1-sized item; it was not folded into P1C because the brief
scoped this phase to ranked period-over-period movement, and quietly widening scope is how
regressions get in.

**Everything else in the bank is either correct, disclosed, or refused.**

---

# Addendum — P1D: Aggregate Contribution (B11 closed)

Scope: aggregate-contribution semantics only. Nothing else was widened.

## A1. The defect

B11 — *"Which region contributes most to the weighted average LTV?"* — returned a chart
**titled with the question** and ordered by each region's own weighted-average LTV, so
**West Midlands** appeared first. The correct answer is **South East**. It was the one
silent semantic error left in the 40-question bank.

## A2. The calculation, and why it is a governed aggregation of its own

A portfolio weighted average is `sum(w*v) / sum(w)`, which decomposes exactly across any
partition of the book:

```
contribution_g  =  sum over g of (w * v)  /  sum over the BOOK of w
                =  weight_share_g  x  value_g
```

The contributions **sum back to the portfolio figure** — checked on every execution and
disclosed if it does not hold. On the demonstration book:

| Region | Region WA LTV | Share of book | Contribution |
| --- | --- | --- | --- |
| **South East** | 43.1412 | 26.2765% | **11.3360** |
| London | 42.7453 | 21.0636% | 9.0037 |
| South West | 43.3106 | 12.0847% | 5.2340 |
| East of England | 43.0398 | 10.8696% | 4.6782 |
| North West | 43.1382 | 6.5525% | 2.8266 |
| **West Midlands** *(previously shown first)* | 43.9477 | 6.1953% | **2.7227** |
| … | | | |
| **Total** | | **100%** | **43.1562** = the portfolio WA LTV |

This is `aggregation: "contribution"` — a governed aggregation, not a sort order, because a
sort order cannot be verified by the P0 guard and a distinct aggregation can.

## A3. The two intents stay distinct

| Question | Aggregation | Answer |
| --- | --- | --- |
| "Which region has the highest LTV?" | `weighted_avg` | per-region LTV ranking — unchanged |
| "Which region contributes most to the weighted average LTV?" | `contribution` | contribution ranking |

Detection requires **both** halves: contribution language (*contributes most to, largest
contributor to, drives most of, accounts for most of, contribution to*) **and** a weighted
aggregate as its object — established from the governed registry (`default_aggregation:
weighted_avg` plus a `weight_field`), not from wording alone. "Which region contributes most
to the balance?" is not claimed: a balance is a sum, and a contribution to a sum is just its
share. Pinned by 6 negative phrasings and 8 positive ones.

## A4. The answer, the title and the receipt

```
South East contributes the most to the portfolio Current LTV: 11.34 of the 43.16
total — 26.3% of the book at 43.14. The highest Current LTV is West Midlands at
43.95, but it is 6.2% of the book and contributes 2.72.

Calculated: Contribution to portfolio weighted-average Current LTV · grouped by
Region · 11,035 loans · as at 30 June 2026.
```

The answer names the highest-**value** group as well as the largest **contributor**, because
that difference is the entire reason the calculation exists. The chart is titled
**"Contribution to Portfolio Weighted Average Current LTV by Region"** — the executed
calculation, not the question it was asked. Both figures and the weight share are in every
row, so a reader can check the arithmetic on screen.

## A5. P0 protection

A new facet kind, `aggregate_contribution`, in the number-or-subject class (refuse, never
disclose-and-continue):

* **APPLIED** only when the governed contribution aggregation actually ran.
* **LOST → refuse** when a contribution question reaches a plain per-group ranking. Verified
  with the spec the live model actually returned for B11: a valid weighted-average bar, for
  a different question, correctly refused rather than presented.
* **LOST → refuse** for every routed capability — `geo_exposure`, `concentration_analysis`,
  `risk_limits`, `period_change_analysis`. None decomposes a weighted average across groups.

The routes also stand down at recognition, using the **same detector the guard uses**, so a
route cannot defer on one set of questions while the guard refuses a different set.

A contribution question that names **no dimension** ("what drives most of the weighted
average LTV?") is refused, not answered: choosing a grouping for the reader would answer a
question they did not ask.

## A6. Results

| | Deterministic | Genuine LLM |
| --- | --- | --- |
| Correct | **6** (was 5) | **6** (was 5) |
| Partial, limitation disclosed | 5 | 3 |
| Safe refusal | 29 | 31 |
| **Silent semantic error** | **0** | **0** |
| **Incorrect successful answer** | **0** | **0** |

**Launch gate met on both paths.** Route agreement between the two parser paths: 40/40.

Against the P1C deterministic baseline, **exactly one question changed — B11**. Nothing else
moved: no route changed, no answer changed, no spec changed. That is the evidence that the
fix is confined to aggregate-contribution semantics.

## A7. Tests

| Suite | Result |
| --- | --- |
| `mi_agent/tests/test_p1d_aggregate_contribution.py` | 31 passed — the calculation against its definition, intent distinctness (8 positive / 6 negative phrasings), the P0 refusal, the chart title, the LLM-spec case |
| `tests/test_p1d_aggregate_contribution_e2e.py` | 16 passed — B11 and 5 paraphrases end to end, every figure reconciled to pandas from the loan book |

The e2e truth fixture restates the executor's own rules — drop rows with no value or no
weight; bucket a missing grouping value rather than drop it — so a change to either shows up
as a disagreement instead of being mirrored.
