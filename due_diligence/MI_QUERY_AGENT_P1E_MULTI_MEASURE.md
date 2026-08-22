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
| **CFO-02** | direct vs acquired on 4 measures | all 4 compared: `Direct has higher observed Current Outstanding Balance… Loan Count… Current Loan To Value… Youngest Borrower Age than Acquired.` | balance £1,385,508,582.98 vs £579,377,675.23; **loans 7,126 vs 3,909**; LTV 43.3535/42.6846; age 72.1878/69.9570 ✅ |
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

**Resolved — you chose the registry route (§6.6).** The disclosure above was
the interim state; a loan count is now compared like any other measure.

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

### 6.6 Loan cardinality is now a governed measure

Following your decision, loan count is declared in the Business Semantics
Registry rather than special-cased out of the comparison.

The registry is auto-generated and refuses hand edits, so the change was made
where it belongs — the **curation in the build script** — and regenerated:

* `count` joins the controlled `default_aggregations` taxonomy. It is a
  *cardinality*, not an arithmetic aggregate: the measure is declared on the
  canonical `loan_identifier`, and the engine primitive counts **presence**
  without coercing the column to numeric, so nothing about an identifier's
  value is ever interpreted.
* `loan_identifier` is curated as an `exposure` **measure**, `count`,
  **`directionality: neutral`** — more loans is neither better nor worse, it is
  scale, which is what stops a comparison reading a verdict into a difference
  in book size — with `display_name: "Loan Count"`.
* Registry content version **0.2.0 → 0.3.0** (243 entries). The controlled
  vocabulary changed, so consumers see it.

**Scoped to `portfolio_comparison` only.** The first attempt tagged it for all
four workflows and immediately tripped an existing guard —
`test_no_loan_level_data_appears_in_the_audit_or_evidence` — because the
period-change audit began naming the identifier field. The guard was right:
period-change over an identifier is not a concept (the movement a reader wants
there is net loan growth, a different measure), and ranking or monitoring an
identifier is meaningless. The curation was narrowed rather than the guard
relaxed.

CFO-02 now compares all four measures, reconciled to the book:

| Metric | Direct | Acquired | Difference |
|---|---|---|---|
| Current Outstanding Balance | £1.39bn | £579.4m | £806.1m |
| **Loan Count** | **7,126** | **3,909** | **3,217** |
| Current Loan To Value (wt) | 43.35 | 42.68 | 0.6688 |
| Youngest Borrower Age (avg) | 72.2 | 70.0 | 2.2 |

One rendering fix went with it: an integer-unit cell was formatted `,.1f`, so
the count read "7,126.0". Counts now render as whole numbers, keyed off the
**aggregation** rather than the unit — the average of an integer-unit field (a
term in months) is legitimately fractional and keeps its decimals.

### 6.7 A proportion answered as an amount

Found by the genuine-LLM acceptance pass (§9a) — invisible to every
deterministic run and to the whole test suite.

> **What proportion of the book is eligible for a 75% LTV securitisation?**
> → `Balance: £1.96bn · Loans: 11,007`

Two absolute figures, no proportion anywhere, and P0's verdict was `ok`. The
reader is left to divide one number by a denominator the answer never states.
This is an **incorrect successful answer** — the class the launch gate forbids.

P0 carried facets for geographic scope, thresholds, ranking, grouping,
contribution, relationships and measure sets. It had **none for a share**.
Before the measure set existed the substitution detector caught this question by
accident — it noticed the answer reported balance where the question said LTV. A
measure set of balance and loan count satisfies that check, so the accident
stopped happening and nothing was left watching.

`KIND_SHARE` is detected with the parser's own `_SHARE_RE`, applied only when
the governed share aggregation actually ran, and is a NUMBER facet — so a lost
one refuses.

Two narrowings, both from questions it wrongly caught:

* **A ranked share is not a share request.** "Which region increased its share
  of the portfolio the most" asks *which one*; the share is the metric being
  ranked and the ranking facet already guards it. The facet is raised only when
  no ranking was detected. (This was breaking P1C.)
* **A routed capability may satisfy a share by stating one.** The concentration
  answer reads "£83.4m (**4.2% of the book**)" — a proportion, from a capability
  that never builds a spec. Accepted on *evidence*: its answer must actually
  contain a percentage. Accepting by route name alone would let any listing
  silently discharge a share request.

**One baseline verdict changes, and it is a deliberate breadth loss.** B21 —
"What is the largest single-loan exposure **and what share of the book is it**?"
— gave the exposure and never mentioned the share. It has no dimension to rank,
so no ranking facet covers it. That was a silent omission and now refuses,
taking the deterministic bank from 11/40 to 10/40. Flagged for ratification
rather than absorbed quietly.

---

## 7. Negative and safety tests — 39 assertions

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
| Governed loan count | the comparison compares all four measures; the count reconciles to the book and renders as a whole number; declared neutral; scoped to comparison, not period-change |
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

## 9a. The genuine-LLM acceptance pass

Run against the real Anthropic API with an operator-supplied key. Two
configurations, 71 questions each — the five CFO questions, the 26-question
P1E bank (imported from the test module so the two cannot drift), and the
untouched 40-question bank.

| Configuration | What it proves | Result | LLM calls | Cost |
|---|---|---|---|---|
| **production** — `MI_AGENT_LLM_PARSER=on`, free path intact | what real users get | 40/71 answered | 43 | $0.23 |
| **forced** — every question sent to the model | the P1E contract on the LLM path, unmasked by the free path | 40/71 answered | 53 | $0.28 |

Provenance is read per question from `metadata.llm.calls` and
`parserProvenance`, not asserted.

**Correction to an earlier claim in this phase.** The first "forced" run forced
nothing: `MI_AGENT_ZERO_COST_FIRST` never reaches the governed entrypoint,
because `mi_service` calls `run_mi_agent_query` without the kwarg and it keeps
its default. Identical call counts in both runs is what gave it away. Forcing is
now done in the harness by wrapping `parse_with_repair` — no production default
was bent to suit a test.

Forced results by suite:

* **CFO 5/5.** Four went to the model; CFO-02 makes no call because portfolio
  comparison is a specialist route resolved before parsing.
* **P1E bank 25/26**, all 26 genuinely through the model. The one failure is
  P1E-02 (§11.3) — a safe refusal, no wrong number.
* **40-question bank 10/40**, identical to the deterministic path. Twenty-one
  questions never reach the model at all: they are answered by specialist
  routes (period movement, concentration, risk limits, comparison, forecast).

**What the pass found.** One incorrect successful answer — B15, §6.7 — which
was invisible to every deterministic run and to the entire test suite. That is
the whole return on running it.

Against the immutable **LLM** baseline the bank went 9 → 11 answered before the
share fix and 9 → 10 after it, so P1E introduced no LLM-path regression.

### API key handling

Environment-only, in-process. Never written to disk, a config file or a commit.
The harness asserts the key and any `sk-ant-` material is absent from its own
JSON output before writing it, and a repository-wide scan for `sk-ant-` across
tracked and untracked files returns nothing.

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

3. **"Total exposure" reads as EAD on the LLM path.** P1E-02 — "Show me
   total exposure, loan count, weighted-average loan to value and
   weighted-average interest rate" — the model resolves *total exposure* to
   `exposure_at_default`, which this book does not carry, so the question
   refuses naming the field. The deterministic parser resolves the same phrase
   to `current_outstanding_balance`, which the registry itself declares the
   "primary current-exposure metric".

   Not a catalogue defect: the model is given the available column list, and
   prompt rule 4 already tells it to prefer fields whose column is present. It
   over-read the generic word "exposure" as an explicit request for EAD. The
   catalogue is deliberately unfiltered — it is the stable, cacheable system
   prefix, and filtering it per dataset would defeat prompt caching — so the
   fix belongs in the instruction wording, which changes **every** LLM parse
   and needs its own re-baseline of both banks. Out of scope here, recommended
   next, and safe meanwhile: the outcome is a refusal that names the field, not
   a number.

4. ~~A loan count is not comparable across books~~ — **resolved**, §6.6.

**API key handling.** A temporary key was supplied mid-phase and used for the
acceptance pass in §9a. It was held in the environment of the harness process
only — never written to disk, a config file, a source file or a commit, and
never echoed. The harness asserts the key is absent from its own JSON output
before writing it. A repository-wide scan for `sk-ant-` across tracked and
untracked files matches only this report's prose describing the scan; no
key-shaped token exists anywhere in the tree. **The key is safe to revoke.**

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
| `mi_workflows/portfolio_risk_comparison.py` | `requested_metrics`, measure-set comparison, `COUNT_MEASURE_FIELD` |
| `scripts/build_business_semantics_registry.py` | `count` aggregation taxonomy, curated loan-cardinality entry, version 0.3.0 |
| `config/business_semantics_registry.yaml` | regenerated (243 entries) |
| `mi_workflows/engine.py` | `AGG_COUNT` primitive |
| `mi_agent/business_semantics.py` | `AGG_COUNT` constant |
| `mi_agent/tests/test_p0_execution_receipt.py` | two tests reworked (§10) |
| `mi_agent/tests/test_p1a_single_filter.py` | one test reworked (§10) |
| `tests/test_p1e_multi_measure.py` | **new** — contract + the five CFO questions (18) |
| `tests/test_p1e_golden_bank.py` | **new** — 26-question bank (74) |
| `tests/test_p1e_measure_safety.py` | **new** — negative and safety tests (39 + 1 xfail) |

**Not merged. Not pushed.**
