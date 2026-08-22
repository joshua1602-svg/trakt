# Extending the robustness grader — pre-registration

Written before implementing. Base: HEAD `1c414c5`, clean tree.

The instruction: *"Two things to fix in the grader before item 1, not after. It
cannot compare a figure to the book, and it grades every reasoned refusal as
safe. Extend it first, demonstrated failing on B5 today."*

---

## 1. Both flaws demonstrated on the shipped path, today

### 1.1 It cannot compare a figure to the book — proved by mutation

B5 alone does not prove it. B5 grades `INCORRECT_SUCCESSFUL` under all nine
declared intents today — but for the **route**, not the number: the generic
executor answered where an owning route was expected. Had a route owned it, the
wrong number would have passed. Saying "B5 is invisible to nl_score" would have
been the easy claim and it is not true.

The honest demonstration is a mutation. Take a bank question the grader calls
CORRECT and treble every figure in its answer:

```
Q1.1  How has the profile of our new lending changed over the last few months?

  original -> CORRECT
     New lending (last 1 month): 143 loans at 2026-04-30 against 115 loans at
     2026-06-30. Current Outstanding Balance £21.4m against £18.3m; ...

  EVERY FIGURE TREBLED -> CORRECT
     New lending (last 3.00 month): 429.00 loans at 6,078.00-12.00-90.00 against
     345.00 loans at 6,078.00-18.00-90.00. Current Outstanding Balance £64.20m ...
```

Three of three mutated answers graded **CORRECT, unchanged** — with the dates
mangled into `6,078.00-12.00-90.00` and the loan counts trebled.

**The 91.0% never measured whether any figure was right.**

### 1.2 The record is starved before the grader is reached

`run_robustness_deterministic._capture` gives the grader ten keys:

```
analyticalIntent, answer, artifacts, capabilities, controlledRefusal,
intent, ok, route, spec, warnings
```

`executionSummary` is not among them. For B5 the service knew
`population=11035, filtersApplied=[]`; **the record drops both before grading.**
So this is not a grader that declines to check the number — it is a grader that
is not given one. Fixing `nl_score` alone would leave it blind.

### 1.3 "Safe" means the refusal sentence was longer than forty characters

The whole test:

```python
if not ok:
    if len(answer) > 40:
        return SAFE_REFUSAL, [], "refused with a stated reason"
```

Measured on the three refusals from the shipped-shapes surface — every one of
which declines something the book can express and the sentence supplies:

```
  What is the LTV for loan tickets above £150k?              -> SAFE_REFUSAL
  Give me a breakdown of balance across LTV and ticket size  -> SAFE_REFUSAL
  Tell me the basics about this book                         -> SAFE_REFUSAL
```

B1 refused **while holding the answer** — filter applied, 5,857 loans, correct
measure. It grades identically to a refusal that was right to decline.

## 2. What is being extended, and what is honestly out of reach

**In reach — the population.** For any question, "how many loans does this answer
cover?" is computable from the book, and it is exactly the B5 defect: a threshold
in the sentence, the whole book in the answer. This is the check that makes item
1's success criterion assessable.

**Out of reach — the forecast.** Six of the nine intents ask for a forward
figure, a run rate or a limit headroom. There is no pandas expression for the
right answer to *"when will we reach £100m?"*. **This extension does not claim to
verify those**, and saying so is the point: a grader that claimed to would be the
same defect again.

So the figure check is scoped and stated: **the population an answer covers, and
whether a narrowing the sentence states reached it.** Nothing else is asserted to
be verified, and the pack note will say which figures remain unverified.

## 3. The owners

"What may the grader see?" and "what does the grader conclude?" are one decision
in two files, which is why `nl_score` could not be fixed alone.

| owner | change |
|---|---|
| `run_robustness_deterministic._capture` | carry `executionSummary` and the KPI raw values |
| `nl_score.grade` | a population check, and the refusal split |

## 4. The refusal split, as a rule rather than a judgement

* **HONEST** — the refusal names something genuinely unavailable, and no other
  phrasing of the same intent answers.
* **UNHELPFUL** — either the record shows execution **held the answer** (a
  measure and a population resolved, then declined), **or another variation of
  the same intent answered**. The second is mechanical and strong: four
  phrasings of one question answering while the fifth refuses is a phrasing gap,
  not a capability limit.

## 5. Pre-registered prediction

### 5.1 The 44, both books, under the extended grader

Today: `CORRECT 32 · SAFE_REFUSAL 10 · CORRECT_WITH_DISCLOSED_LIMITATION 2`.

The ten refusals by intent: Q2 1, Q3 2, Q4 1, Q5 2, Q9 4.

| intent | refusals | do other phrasings answer? | predicted |
|---|---|---|---|
| Q2 | 1 of 4 | yes, 3 CORRECT | **1 unhelpful** |
| Q3 | 2 of 4 | yes, 2 disclosed | **2 unhelpful** |
| Q4 | 1 of 4 | yes, 3 CORRECT | **1 unhelpful** |
| Q5 | 2 of 4 | yes, 2 CORRECT | **2 unhelpful** |
| Q9 | 4 of 4 | **no — all four refuse** | **4 honest** |

**Predicted: 6 unhelpful, 4 honest.** Evidence: the by-intent counts in the
deterministic run at `57a560c`, quoted above.

**Predicted WRONG_FIGURE on the 44: 0.** The bank states no row-level threshold —
Q2's "£100m" is a target to forecast toward, not a narrowing. If this predicts
wrong, the extension found a defect the bank was already carrying and could not
report, which is a better result than the prediction.

**Predicted CORRECT: 32 → 32, and the 2 disclosed unchanged.** No answer moves;
only refusals are reclassified and a new check is added.

### 5.2 What must not move

1. No product behaviour changes. This commit touches a grader and a capture.
2. `answer_diff` 729 identical — the differ does not read `nl_score`.
3. The routed surface, calibration bank and lexical surface unmoved.
4. The shipped-shapes counts unmoved: 9 / 1 / 0 / 5.

### 5.3 Stop conditions

* the extended grader reporting a WRONG_FIGURE it cannot substantiate against
  the book;
* any of the 32 CORRECT becoming a refusal, or vice versa;
* the population check firing on a question that states no narrowing.

## 6. The constructed-coverage position, stated up front

The refusal split rests on the bank's own by-intent counts, and the population
check rests on questions that state a threshold — **of which the 44 contain
none.** So the expected result on the 44 is that the figure check finds nothing,
and that will mean **the check did not reach this bank, not that this bank is
clean.** Its evidence is the shipped-shapes surface, where the same check
separates B5 from B2. That is the same form as B21's statement and it is being
declared before the number is known, not after.
