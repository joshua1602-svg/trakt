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
