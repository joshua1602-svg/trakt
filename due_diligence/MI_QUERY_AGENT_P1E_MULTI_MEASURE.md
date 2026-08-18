# TRAKT MI Query Agent — P1E Multi-Measure Composition

**Branch** `claude/mi-query-agent-review-n8d33r` · **Base commit** `cf619bb`
**Fixture** `demo_platform` / alderbridge — 11,035 loans, £1,964,886,258.21,
reporting date 30 June 2026
**Question bank** `config/mi/golden_questions/business_semantic_questions.yaml`,
sha256 `e0fc0b61…3194` — **unmodified**, byte-identical to the pre-launch
baseline manifest.

---

## 1. What P1E set out to do

Let one MI question carry **two to four governed measures** over **one
population, one filter set, one optional grouping, one reporting period** —
and answer it as a single management statement rather than as several
unrelated questions or, as before, as one number with the rest discarded.

The standing product principle governs the whole phase:

> Expand the trusted answerable surface incrementally. Never increase breadth
> by weakening the P0 safety boundary.

Everything below is measured against that second sentence, not just the first.

---

## 2. The semantic contract

A spec now carries a `measures` list; the singular `metric` / `aggregation`
slots remain and are kept consistent with `measures[0]`.

```python
measures: List[Dict[str, Any]]        # [{"field", "aggregation", "weight_field"}]
MAX_MEASURES = 4
normalise_measures(raw, *, metric=None, aggregation=None, weight_field=None)
```

**One contract downstream, not two engines.** A single-metric spec normalises
to a one-element measure set, so every consumer reads one shape. That is the
backward-compatibility guarantee, and it is asserted directly:

```python
spec = MIQuerySpec.from_dict({"metric": BALANCE, "aggregation": "sum"})
assert spec.measures == [{"field": BALANCE, "aggregation": "sum"}]
assert spec.metric == BALANCE and spec.aggregation == "sum"
```

`normalise_measures` folds every recognised shape (canonical mappings, bare
field-name strings, `metric`/`agg` aliases, a single mapping) and **reports
rather than drops** anything it does not recognise — the discipline
`normalise_filters` already followed. Duplicates fold: "balance and total
balance" is one governed measure, reported once.

`referenced_fields()` includes every measure, so validation sees the whole
request instead of only its first measure.

---

## 3. Governed aggregation — the model never chooses the arithmetic

`resolve_measures()` resolves each measure against the semantic registry:

* the aggregation is the registry's `default_aggregation` unless the request
  names one the registry also **allows**;
* an aggregation the registry forbids is **refused for that measure with a
  reason**, never silently downgraded — a simple mean where a weighted average
  is governed is a different number, not a rounding difference;
* `weighted_avg` resolves its weight from the registry
  (`current_outstanding_balance`), not from the question;
* percent measures stay in **percentage points**, per the house convention.

The LLM's role is bounded and tested: it may name governed concepts, and
`reconcile_measure_aggregations()` replaces the model's aggregation with the
house convention wherever both parsers named the same field. It may not add,
drop or re-point a measure — asserted. A field the model invents is reported as
unavailable, never calculated.

**Parser consistency.** `carry_measure_set()` closes the last version of the
original defect: if the model still answers a four-measure question with a
single metric, the deterministic parser's set is carried onto its spec — the
same precedence rule `carry_specialist_intent` already applies, namely that
something the deterministic parser positively recognised may not be shadowed by
a more generic LLM spec. Without it the LLM path would *refuse* (P0 would see a
multi-measure request and no measure set) a question the deterministic path
answers — two parsers disagreeing about one sentence. It is deliberately
narrow: the set is carried only when the model expressed no measure of its own,
or expressed one the deterministic set already contains, so it can never
re-point the question at a measure the model did not name. Both the carry and
the two refusals-to-carry are asserted.

---

## 4. The five curated CFO questions

Deterministic parser. Every figure reconciled against **pandas computed
independently** from the canonical fixture — the agent's own aggregation
functions are never the oracle.

| | Question | Answer | Independent truth |
|---|---|---|---|
| **CFO-01** | balance, loan count, WA LTV, WA rate | `Balance: £1.96bn · Loans: 11,035 · Weighted-average Current LTV: 43.16% · Weighted-average Interest Rate: 6.56%` | 1,964,886,258.21 · 11,035 · 43.156246 · 6.559723 ✅ |
| **CFO-02** | direct vs acquired on 4 measures | 3 compared: `Direct has higher observed Current Outstanding Balance… Current Loan To Value… Youngest Borrower Age than Acquired.` **`Not compared: Loan count (…)`** — see §6.4 | balance £1,385,508,582.98 vs £579,377,675.23; LTV 43.3535/42.6846; age 72.1878/69.9570 ✅ |
| **CFO-03** | London: balance, loans, WA LTV, avg age | `Balance: £413.80m · Loans: 1,380 · Weighted-average Current LTV: 42.75% · Average Borrower Age: 71.4` | 413,804,467.49 · 1,380 · 42.745340 · 71.383333 ✅ |
| **CFO-04** | borrowers over 85: 4 measures | `Balance: £19.43m · Loans: 86 · Weighted-average Current LTV: 51.97% · Weighted-average Interest Rate: 6.65%` | 19,428,730.79 · 86 · 51.973389 · 6.648052 ✅ |
| **CFO-05** | balance, loans, WA LTV **by region** | one table, 12 groups, three measure columns | every group reconciled ✅ |

Receipts name every executed measure and the one population, e.g.

```
Calculated: Balance · Loans · Weighted-average Current LTV ·
Weighted-average Interest Rate · Borrower Age > 85 · 86 loans ·
as at 30 June 2026.
```

---

## 5. One population, not one per measure

The composition rule is asserted structurally, not by inspection: the loan
count is one of the measures, so if any measure had been calculated over a
different frame the count could not agree with all of them. Every bank question
asserts the receipt names the population that pandas independently computes.

For grouped requests, the result is **one table** with one row per group and
one column per measure — asserted against the group count pandas derives,
including the executor's explicit `Unknown / Missing` bucket (2 loans have no
region; they are bucketed, not dropped, so group totals still reconcile to the
portfolio).

---

## 6. Three defects found and fixed during the phase

These were found by running the bank, not by reading the code.

### 6.1 A silently dropped measure (the serious one)

> "What are the total balance, number of loans, weighted average LTV and
> **weighted average rate**?"
> → `Balance: £1.96bn · Loans: 11,035 · Weighted-average Current LTV: 43.16%`

Three of four measures returned, with nothing in the answer, the warnings or
the receipt indicating the fourth was missing. Bare "rate" was not in the
measure vocabulary — and neither was it in P0's, so both sides shared the same
blind spot and P0 saw a complete request.

**The fix is structural, not lexical.** `unresolved_measure_slots()` reads the
question's coordinated measure list and reports any slot that resolved to *no*
governed measure. It does not need to know what the unrecognised words mean,
only that a slot of the same list came back empty — which is why it also covers
vocabulary the parser has not learnt yet, instead of needing a new pattern per
missing synonym. The slot becomes a `KIND_UNRESOLVED_MEASURE` facet, which is a
`NUMBER_OR_SUBJECT` facet, so it **refuses**:

> I understood that you asked for **weighted average rate**, but that could not
> be applied to the calculation… I have not substituted a broader figure.

Bare "rate" and bare "loans" were deliberately **not** added to the vocabulary.
"Rate" could be the interest rate, the default rate, the redemption rate or the
prepayment rate; guessing would be exactly the inference the brief forbids. An
honest refusal naming the ambiguous term is the correct outcome.

The refusal is also the *consistent* one: a measure that is unavailable in the
dataset ("credit score") already refuses the whole question. Treating a measure
the agent failed to *parse* more leniently than one it understands and cannot
calculate would invert the safety ordering. Asserted as a test.

### 6.2 A filter subject read as a second measure

> "how much balance is below 75% LTV" → threshold lost, question refused.

The measure detector counted "LTV" — the *subject of a predicate* — as a second
measure, so the multi-measure path claimed the question and masked the LTV span
before filter parsing, destroying the threshold clause. This broke ~100 existing
tests across the P1A, ranking, trust-hardening and calibration suites.

P0 already had exactly this discipline (`_is_filter_subject`) and the same
exclusion for grouping axes. The parser now **reuses P0's own helpers** rather
than re-deriving them, which is what keeps the two sides from disagreeing about
the same sentence. Two further axis cases were closed the same way: a measure
word anywhere inside a grouping clause ("balance **by region and age bucket**"),
and a measure word carrying a dimension suffix ("which **age bucket** has the
largest balance").

### 6.3 A sourcing cohort resolved as a geography

> "For the acquired book, what are balance, loan count and weighted-average
> LTV?" → *"No loans in this book match that filter"*

The leading-scope resolver read "the acquired book" as a place and invented a
`collateral_geography` value the column does not contain, on top of the correct
`alp_acquired` portfolio scope — emptying the population. Sourcing-cohort words
were added to the existing `_NON_PLACE_TERMS` guard. Now:

```
Calculated: Balance · Loans · Weighted-average Current LTV ·
Source Portfolio in alp_acquired · 3,909 loans · as at 30 June 2026.
```

### 6.4 A requested loan count vanishing from a comparison

CFO-02 asks for four measures across two books. Only three came back, and
nothing in the answer, the warnings or the receipt indicated the fourth was
missing — the same defect class as §6.1, on a different route.

The cause: `loan_count` was filtered out of the comparison's requested set
before anything could account for it. The reason it was filtered out is sound —
the comparison workflow compares **Business Semantics Registry** measures, and
a loan count carries no BSR directionality or comparability declaration, so
there is nothing governed to compare it *by*. What was wrong was that the
exclusion was silent.

It is now carried as `uncomparable_measures`, reported through the route's
explicit-partial line:

> Direct has higher observed Current Outstanding Balance than Acquired. …
> **Not compared: Loan count (a loan count is not a Business Semantics Registry
> measure for portfolio comparison).**

**Why this discloses rather than refuses, and why that is not a P0 weakening.**
`requestedMetricsNotCompared` means "a BSR measure I undertook to compare and
did not" — that list feeds P0's measure facet and still refuses. A loan count
was never in that set, so it is reported through a separate key. Injecting it
into the ledger instead would have turned CFO-02 into a total refusal, losing
three correct governed comparisons over a figure the capability does not
express at all; and it would have made the ledger claim the workflow had
undertaken something it never had. No facet that blocked before is non-blocking
now. Both halves are pinned by tests.

**Open governance question for you, not for me to decide:** a loan count per
book is trivially computable and would be genuinely useful in a comparison. Two
resolutions exist — add a governed loan-count measure to the Business Semantics
Registry (with an explicit directionality declaration, most likely *neutral*),
or leave counts outside the comparison surface and keep the disclosure above.
The first is a registry-ownership decision of the kind reserved to you in the
asset-class phase, so it is flagged rather than taken.

### 6.5 "The funded book" read as a place called *Funded*

Found by the full suite, not by the MI bank: the landing-page demo-pack build
refused *"How many loans are in the funded book?"* with

> No loans in this book match that filter (collateral_geography), so there is
> nothing to calculate.

Same shape as §6.3 — a scope word resolved as a geography value the column does
not contain, emptying the population — but §6.3's fix did not catch it, because
the extracted value is the bare word `Funded` rather than "funded book". Funding
state words joined the same `_NON_PLACE_TERMS` guard, and the question now
answers over all 11,035 loans.

**This failure pre-dates P1E.** It was verified in a clean worktree at the base
commit `cf619bb`, where it reproduces identically. It is fixed here rather than
merely reported because it is the same defect class already being fixed and the
change is one line of an existing list. A scope-control test asserts the guard
never shadows a word that appears in a real region name in the book.

---

## 7. Negative and safety tests — 37 assertions

`tests/test_p1e_measure_safety.py`. Each is a way P1E could produce a confident
wrong answer.

| Class | Behaviour asserted |
|---|---|
| Unsupported measure among supported | whole question refuses, names the measure, offers no figure |
| Consistency | an unparsed measure is treated no more leniently than an unavailable one |
| Measure dropped by parser | named in the refusal, never silently omitted (both known phrasings) |
| Guard is structural | reports `("the flurble index",)` — vocabulary it has never seen |
| Guard scope control | 7 good questions still answer; a guard that refuses good questions is an outage |
| Different scope per measure | "balance in London and the loan count in the South East" refuses |
| One scope, many measures | applied to all four, stated **once** in the receipt |
| Registry-forbidden aggregation | measure dropped with a reason, not recalculated another way |
| House weighting | applied without being asked; weight field is the governed one |
| Explicit simple average | honoured **and labelled** — "Average" can never read as "Weighted-average" |
| Duplicate measures | folded to one |
| Over `MAX_MEASURES` | refuses; no truncation to "the first four" |
| At `MAX_MEASURES` | answers — the boundary is inclusive |
| P0 boundary | `KIND_UNRESOLVED_MEASURE` is a NUMBER facet (refuses), not a SHAPE facet (partial) |
| Relationship questions | "ltv vs interest rate" stays a scatter, not a measure set |
| Model role | prompt carries the array and both prohibitions; reconciliation cannot move a measure; an invented field cannot reach a number |
| Routed capability gap | a comparison names the loan count it cannot compare; the ledger keeps BSR measures separate from measures the route does not express |
| Parser consistency | a set the model returned as one metric is carried; a measure it read differently is never overwritten; a set it expressed itself is left alone |
| Scope words | "the acquired book" and "the funded book" are never resolved as places; the guard never shadows a real region word |

---

## 8. The P1E golden bank — 26 questions, 74 assertions

`tests/test_p1e_golden_bank.py`. Twenty ungrouped and six grouped questions:
the same four measures phrased four ways; two-, three- and four-measure sets;
scoped by geography, by threshold and by sourcing cohort; grouped by region and
by originator; scoped and grouped at once.

Three properties are asserted for every question:

1. **every named measure was calculated** (read from the delivered KPI
   artifact — if it is not there, the user did not receive it);
2. **every figure reconciles to pandas** at `rel=1e-9`;
3. **one population served every measure**, named in the receipt.

Grouped questions additionally assert one table (not one per measure) and
reconcile **every group** of every measure independently.

**Result: 74 passed.**

---

## 9. Regression — the 40-question bank and the full suite

The bank and its expected answers were not touched (sha256 unchanged). Re-run
on the deterministic path and diffed against the immutable pre-launch baseline:

**Baseline 11/40 answered · now 11/40 answered.** Three answers changed, all
improvements; no question regressed.

| | Before | After |
|---|---|---|
| **A1** *"average LTV, average borrower age, average borrower type in London"* | refused: *"more than one measure (ltv and age)… only one measure can be calculated per question"* | refused: *"the average borrower type… is not a governed measure in this dataset"* — names the thing that is actually unavailable |
| **A8** *"average LTV of the direct book vs the acquired book"* | `Calculated: Portfolio comparison.` | `Calculated: Weighted-average Current Loan To Value · Direct vs Acquired.` |
| **B25** *"direct vs acquired on borrower age"* | *"no governed directional differences were observed"* | `Direct has higher observed Youngest Borrower Age than Acquired.` |

A8 and B25 come from the asset-class work already committed at `cf619bb`; A1 is
P1E's better refusal.

**One regression was caught and fixed by this diff.** B21 — *"What is the
largest single-loan exposure and what share of the book is it?"* — briefly
refused, because the new slot guard read the trailing clause *"what share of
the book is it"* as an unresolved measure. A measure slot is a noun phrase; a
slot carrying a question word or a finite verb is a second clause, not a measure
name. Fixed, B21 restored to its baseline answer, and the true positives in §6.1
still fire.

Targeted suites — all of `mi_agent/tests/` and `landing-page/tests/` plus the
P1C, P1D/B11, B25, asset-class-lifecycle and three P1E suites — **1,433 passed,
1 skipped, 21 xfailed**.

**Full suite: 8,476 passed, 30 skipped, 21 xfailed, 6 subtests passed.**

The first full-suite run surfaced one failure —
`landing-page/tests/demo_pack_reproducible_test.py::test_committed_pack_matches_a_fresh_build`.
It was verified in a clean worktree at the base commit `cf619bb` and reproduces
there identically, so it **pre-dates P1E**. It was fixed anyway (§6.5) because
it is the same defect class as §6.3 and the one-word fix was already in hand.

---

## 10. Two tests reworked, none deleted

Two tests pinned the pre-P1E behaviour that a multi-measure question is
*refused*. That behaviour is exactly what P1E replaces — but the guarantee each
protected still holds and must still be tested, so both were reworked to assert
the guarantee rather than the old mechanism:

* `test_multiple_measures_are_refused_rather_than_silently_reduced_to_one` →
  `test_multiple_measures_are_never_silently_reduced_to_one` — now asserts both
  measures are **delivered**, plus a new sibling asserting that a set which
  *cannot* be delivered whole still refuses.
* `test_a_multi_measure_query_is_still_refused_after_p1a` →
  `…_shares_one_filtered_population_after_p1e` — asserts the London filter
  applies to **both** measures and the receipt states the scope once.

---

## 11. Known limitations, stated plainly

1. **`median` is not in the aggregation-qualifier vocabulary.** "Give me the
   median balance" resolves to the registry default (`sum`) and answers
   `Balance: £1.96bn`. The receipt truthfully says `Total Balance`, so the
   substitution is disclosed rather than hidden — but the request is not
   honoured. **This pre-dates P1E and affects the single-measure path
   identically** ("What is the median balance?" → `Calculated: Total Balance`),
   so it is a general aggregation-vocabulary defect rather than a
   measure-composition one. Fixing it would change single-measure behaviour
   across the whole product, which is outside this phase's scope. It is pinned
   as a `strict=True` xfail so it cannot be forgotten, and is recommended as the
   next phase's first item.

2. **Beyond `MAX_MEASURES`, the refusal message varies.** Five measures produce
   the clear *"more than one measure (…)"* refusal; six can instead surface a
   validation message about an aggregation. Both refuse and neither returns a
   figure, so the safety outcome is identical — but the second message is less
   useful to a reader. Caused by validation ordering, not by measure counting.

3. **The genuine-LLM path was not exercised.** No `ANTHROPIC_API_KEY` was
   supplied to this session, so every result above is the deterministic parser
   (`parserMode: "deterministic"`, asserted in the bank). The LLM-side contract
   *is* tested where it can be without a key — the prompt carries the `measures`
   array and both prohibitions; `carry_measure_set` and
   `reconcile_measure_aggregations` are asserted on constructed specs standing
   in for model output, including the cases where they must decline to act.
   That is the contract, not the model's behaviour against it. **A genuine-LLM
   re-run of the five CFO questions, the P1E bank and the 40-question bank
   remains outstanding and should be completed before launch.**

4. **A loan count is not comparable across books** — §6.4. Disclosed, not
   silent; the governance decision is yours.

**API key handling.** None was supplied. A repository-wide scan for `sk-ant-`
across tracked and untracked files returns nothing. No key was written to disk,
committed or printed at any point in this phase.

---

## 12. Files changed

| File | Change |
|---|---|
| `mi_agent/mi_query_spec.py` | `measures`, `MAX_MEASURES`, `normalise_measures`, measure-aware `referenced_fields()` |
| `mi_agent/mi_query_executor.py` | `ResolvedMeasure`, `resolve_measures`, `_execute_measure_set`, `_execute_grouped_measure_set`, measure-set dispatch |
| `mi_agent/llm_query_parser.py` | `_measure_hits`, `detect_measure_set`, `unresolved_measure_slots`, `_mask_spans`, `_measure_set_recognizer`, `carry_measure_set`, `reconcile_measure_aggregations`, prompt rule 10 |
| `mi_agent/execution_receipt.py` | `KIND_UNRESOLVED_MEASURE`, measure-set reconciliation, `executed_measure_concepts`, receipt measure phrases |
| `mi_agent/mi_agent_workflow.py` | `_multi_measure_answer`, `_format_measure_value` |
| `mi_agent/mi_query_contract.py` | requested / executed / unavailable measures on the query trace |
| `mi_agent_api/chat_routing.py` | requested-measure invariant + explicit partial on the comparison route |
| `mi_workflows/portfolio_risk_comparison.py` | `requested_metrics`, measure-set comparison |
| `mi_agent/tests/test_p0_execution_receipt.py` | two tests reworked (§10) |
| `mi_agent/tests/test_p1a_single_filter.py` | one test reworked (§10) |
| `tests/test_p1e_multi_measure.py` | **new** — contract + the five CFO questions (18) |
| `tests/test_p1e_golden_bank.py` | **new** — 26-question bank (74) |
| `tests/test_p1e_measure_safety.py` | **new** — negative and safety tests (37 + 1 xfail) |

**Not merged. Not pushed.**
