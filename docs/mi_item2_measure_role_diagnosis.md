# Item 2 — B3 and C4, diagnosed together

Diagnosis and pre-registration. No code touched. Base: HEAD `a5be307`.

---

## 1. The headline correction: they are NOT the same defect

I proposed B3 and C4 as "the exact mirror image" of one another — a field word
outside the measure position taken as the measure, in both directions. **The
diagnosis does not support that.** They share a symptom and nothing else.

| | B3 | C4 |
|---|---|---|
| question | *"Show me the LTV for loans with a balance above £150,000"* | *"Give me a breakdown of balance across LTV and ticket size"* |
| measure resolved | Balance (should be LTV) | Current LTV (should be Balance) |
| **cause** | the `is_show_loans` **drill-through branch**, which hard-codes `metric = _balance_metric(...)` | `_grouping_segments` splits on **`by` only**, so `across` never cuts the sentence |
| trigger | the words `show` + `loans` + no `by` | the axis marker not being in one consumer's list |
| symptom | the substitution guard refuses | the substitution guard refuses |

Measured:

```
"show me the ltv for loans with a balance above 150000"     -> metric=current_outstanding_balance  agg=loan_level
"what is the ltv for loans with a balance above 150000"     -> metric=current_loan_to_value        agg=weighted_avg
"show me the ltv for accounts with a balance above 150000"  -> metric=current_loan_to_value        agg=weighted_avg
```

**B3 has nothing to do with the measure position.** Change "show me" to "what
is", or "loans" to "accounts", and the measure resolves correctly. The trigger is
a branch that fires on the WORDS `show`/`list`/`display`/`drill` plus `loans`,
and then reports balance whatever the question asked for.

Diagnosing them together is what showed this. Fixing them as one would have
produced a change that closed neither cleanly.

## 2. The axis vocabulary: one decision, four consumers, two of them blind

```
DECLARED OWNER  lexical.AXIS_MARKERS: by, per, across, split by, broken down by, grouped by

marker             _grouping_segments  _GROUPING_CLAUSE_RE  receipt:901  lexical._BY_RE
--------------------------------------------------------------------------------------
by                 True                True                 True         True
per                FALSE               True                 True         FALSE
across             FALSE               True                 True         FALSE
split by           True                True                 True         True
broken down by     True                True                 True         True
grouped by         True                True                 True         True
```

`split by`, `broken down by` and `grouped by` pass the `by`-only consumers
**incidentally, because they contain the word `by`.** Only `per` and `across`
expose the gap.

`_grouping_segments` is `re.split(r"\bby\b", q)`. `lexical._BY_RE` is `\bby\b`.
**The declared owner already holds the correct list and neither reads it.**

### One decision or two?

**One.** Every one of the four consumers is asking the same question — *where
does the grouping clause start?* — and each needs the answer in a different
shape: `_grouping_segments` needs to SPLIT, `_GROUPING_CLAUSE_RE` needs REGIONS,
`execution_receipt:901` needs a SUFFIX match, `lexical.grouping_cut` needs an
OFFSET.

By the corrected rule — **share what is one fact, separate what is two** — the
vocabulary is shared and the four implementations stay distinct. The same shape
as item 1, and for the same reason: "is `across` a grouping marker?" is one fact
about English, not four.

## 3. Why C4's measure goes wrong, precisely

Two measure resolvers with **different contracts**:

```
                                                _measure_hits   _detect_metric
"show me the ltv for loans with a balance..."   ltv             ltv
"show me balance across ltv and ticket size"    balance         ltv
"show me balance by ltv and ticket size"        balance         ltv
```

`_measure_hits` masks grouping regions and filter subjects **itself** and is
correct in all three. `_detect_metric` masks **nothing** — it answers "which
measure word is in this text" and depends entirely on the caller handing it a
pre-cut string.

For `by`, `_grouping_segments` cuts and `_detect_metric(metric_part)` sees only
`"show me balance"`. For `across` the cut never happens, `_detect_metric` sees
the whole sentence, and it reads `ltv` from the axis clause.

So C4 is **caused by** the axis-vocabulary gap, not by an independent measure
defect. Fixing decision A should close C4 without touching the measure
resolvers, and the prediction below says so in a falsifiable form.

## 4. Scope

* **2A — one axis-marker vocabulary.** `lexical.AXIS_MARKERS` becomes the read
  owner for all four consumers, each keeping its own implementation.
* **2B — the drill-through branch must not override a measure the question
  named.** Separate fix, separate mechanism, same commit only because both are
  "the measure the reader asked for is not the measure reported".

**Not in scope, recorded:** unifying `_detect_metric` and `_measure_hits` into
one owner. That is a real multi-owner decision, and after 2A it has **no known
failing case** — the B24 precedent says record it rather than open it on
suspicion.

## 5. The item 1 lesson, applied before it can repeat

Item 1's vocabulary consolidation left a hard-coded twelve-character window that
the old, shorter vocabulary never reached. So for 2A, the consumers will be
exercised across the **full range** of the unified list — every marker, longest
(`broken down by`, 14 chars) and shortest (`by`, 2), in each of the four
consumers — rather than only on the two markers that currently fail.

## 6. Pre-registered prediction

### 6.1 What moves

| id | today | predicted |
|---|---|---|
| C4 `breakdown of balance across LTV and ticket size` | measure `Current LTV`, dims `['Ticket Size']`, refuses | **measure Balance, both dims, answered, 50 cells** |
| `balance per LTV and ticket size` | same failure | **same fix** |
| B3 `Show me the LTV for loans with a balance above £150,000` | measure `Balance`, refuses | **measure Current LTV, answered over 5,857** |
| C1/C2/C3/C5 (`by`, `split by`, `broken down by`) | correct | **unchanged** |
| B1, B4 (`ticket` unresolved role) | unhelpful refusal | **unchanged** — that is item 3 |
| A5 | unhelpful refusal | **unchanged** — item 4 |

`shipped_shapes` predicted: **12 correct · 0 wrong · 0 honest · 3 unhelpful.**

### 6.2 What must not move

1. `answer_diff` 729 of 729 identical.
2. The 44, both books: `CORRECT 32 · UNHELPFUL 6 · SAFE 4 · DISCLOSED 2`.
3. Calibration 259/259, 0 hard failures, 0 known gaps.
4. Routed surface 32/32.
5. Seasoning families by name, both books: Q1 4, Q7 4, Q8 12.
6. Item 1's 48 tests.

### 6.3 Stop conditions

* any `by`-phrasing changing its dimensions or measure;
* the drill-through branch ceasing to fire for a genuine loan-level drill
  (*"show me loans with LTV above 50%"* must stay a loan-level table);
* a corpus answer moving.

### 6.4 Constructed coverage, declared before the number

The corpus contains **no** question using `across` or `per` as a grouping marker
with a measure-capable field in the axis clause — the C4 shape. So a clean
`answer_diff` will mean **the fix did not reach the corpora and nothing more**,
and the claim will rest on the constructed C-series and the new tests.

## 7. The 44, verified against the book

Carried in: *"32/6/4/2 has not yet been verified against the book."* Correct, and
precisely so. The extended grader's figure check is the POPULATION check, and it
fires only when the question states a threshold. **None of the 44 does.** So the
32 CORRECT still rest on route and capability structure, exactly as the
constructed-coverage statement said.

This item verifies what is honestly verifiable and reports the rest as
unverified: the point-in-time seasoning family (Q1, Q7, Q8 — 20 of 44) has
populations and balances computable from the book. **Q2, Q3, Q4, Q5, Q6 and Q9
ask for a forward figure, a run rate or a limit headroom and stay unverified** —
holding the line ruled in item 1, that a grader claiming to check them would be
this defect wearing a better label.

---

# MEASURED — appended after implementation; prediction above left as written

## 8. Results against §6

| declared | measured | |
|---|---|---|
| C4 → measure Balance, both dims, 50 cells | **correct, 50 cells, reconciles to £1,964,886,258.21** | ✅ |
| `per` fixed by the same change | yes — all six markers now give Balance + both dims | ✅ |
| B3 → measure Current LTV over 5,857 | **loan-level over 5,857, sorted by LTV, LTV column present** | ✅ |
| C1/C2/C3/C5 unchanged | unchanged | ✅ |
| B1, B4, A5 unchanged | unchanged — items 3 and 4 | ✅ |
| shipped shapes **12 / 0 / 0 / 3** | **12 / 0 / 0 / 3** | ✅ |
| `answer_diff` 729 identical | **729 of 729, 0 moved** | ✅ |
| the 44, both books 32/6/4/2 | identical on both books | ✅ |
| calibration 259/259 | 259/259, 0 hard failures, 0 known gaps | ✅ |
| routed surface 32/32 | 32 passed, 0 failed | ✅ |
| seasoning by name, both books | Q1 4, Q7 4, Q8 12 | ✅ |
| **no lexical decision moves** | **4 moved** — see §9 | ❌ |

**The prediction that fixing 2A would close C4 without touching either measure
resolver held.** `_measure_hits` and `_detect_metric` are unchanged.

## 9. A stop condition fired, and the movement is a correction

§6.2 said no lexical decision moves. Four did:

```
  balance split by region          answer_type.subject_side  'balance split' -> 'balance'
  balance split by borrower type   answer_type.subject_side  'balance split' -> 'balance'
  balance split by product type    answer_type.subject_side  'balance split' -> 'balance'
  balance split by broker          answer_type.subject_side  'balance split' -> 'balance'
```

`grouping_cut` previously found only `by`, at offset 14, so the subject span ran
to `"balance split"` — with the word `split` inside the span that may name the
measure. It now finds `split by` at offset 8 and the subject is `"balance"`.

**All four are corrections and none changes an answer** — `answer_diff` is
729 of 729 identical. The prediction was wrong because it assumed the
consolidation could only affect `per` and `across`; `split by` was being read
as `by` with `split` left in the subject, which is the same defect in a phrasing
that appeared to work. Baseline re-recorded.

## 10. Two grader errors, both mine, both the same class

* **`grade_B` could not see a loan-level answer.** B3 returns a table of the
  right 5,857 loans, sorted by LTV, carrying an LTV column — a correct answer to
  *"show me the LTV for loans over £150k"*, which asks to SEE them. The grader
  wanted an aggregate and reported the product wrong. **Fourth instance**, and
  the fourth in the direction of marking the product wrong for being right. Now
  accepts a loan-level table over the correct population, with a can-fail
  proving a loan-level table over the WHOLE BOOK is still wrong.
* **The ownership test measured prose, twice.** It read the module source and
  matched first the COMMENT and then the DOCSTRING describing the by-only split
  it had just replaced. Replaced with a behavioural proof: add a marker to
  `lexical.AXIS_MARKERS` and assert the consumer picks it up.

## 11. The 44 verified against the book — and the first attempt was worthless

Carried in: *"32/6/4/2 has not yet been verified against the book."*

**First attempt, and it does not survive its own can-fail.** Collecting every
low-cardinality dimension's count and balance gave a 322-value truth set, then
asking whether each answer quotes *at least one* matching figure:

```
  20 of 20 answered Q1/Q7/Q8 rows quote at least one figure that matches the book
  CAN-FAIL: 20 of 20 TREBLED answers still match -> the check is WEAK
```

With 322 values and a 2% tolerance almost any number finds a neighbour. **That
check verifies nothing and is not reported as a result.** It was caught by the
same trebling mutation that exposed `nl_score`.

**Second attempt: bind each figure to the claim it labels.** Q8's answers are
structured — *"Direct, 7,126 loans: Current Outstanding Balance £1.36bn →
£1.39bn"* — so the label, the count and the closing balance can be checked
against that specific population:

```
  8 of 8 labelled claims match the book exactly
  CAN-FAIL: 0 of 8 trebled claims match -> DISCRIMINATING
```

### What that establishes, stated narrowly

* **8 labelled claims across the Q8 family are verified against the book**, by a
  check a trebled answer fails.
* **The population check is vacuous on all 44.** It fires only when a question
  states a threshold, and none does — and separately, the analytical route emits
  no `executionSummary.population` at all, so there was nothing to compare.
  *(Recorded: the analytical path does not publish a population. A receipt gap,
  not opened here.)*
* **Q1 and Q7 remain unverified.** Their answers quote figures over rolling
  origination windows and profile percentages that this check cannot bind to a
  labelled population.
* **Q2, Q3, Q4, Q5, Q6, Q9 — 24 of 44 — remain unverified by construction.**
  Forward figures, run rates and limit headroom, exactly as ruled in item 1.

So `CORRECT 32` is **not** a statement that 32 answers carry correct figures. It
is a statement that 32 answers were claimed by the expected route with the
expected capabilities, **plus** 8 labelled claims now checked against the book.
The qualifier in the pack stands and narrows by that much.
