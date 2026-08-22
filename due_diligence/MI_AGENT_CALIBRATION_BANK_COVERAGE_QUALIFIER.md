# Calibration bank — what "green" covers, and what it does not

**A qualifier that must travel with the figure wherever it is quoted.**

Measured on `28ece25` against the real alderbridge tape (11,035 loans,
£1,964.89m). Reproduce:
`python -m question_interpretation.check_answer_type_expectations`.

---

## The headline figure, stated correctly

> The calibration bank runs **260 passed, 0 skipped, 0 xfailed**, and every one
> of its 252 curated cases holds its declared expectation.

That is true, and it is routinely read as meaning all 252 declared expectations
were *evaluated*. For the **answer-type** expectation specifically, they were
not.

> **"The bank is green" means 131 answer-type expectations held, not 252.**

Every case carries an `expected_answer_type`. The check that reads it sits at
the end of `mi_calibration.evaluate_case`, behind four early returns. On this
book it is reached for 131 of 252:

| Answer-type expectation | Cases | |
|---|---:|---|
| **evaluated** | **131** | of which 127 pass strictly, 4 only via `_SATISFIES` permissiveness |
| never evaluated — required field absent on this book | 79 | returns at the missing-field gate |
| never evaluated — `execution: parse_only` | 21 | returns before the type check |
| never evaluated — `expected_status: refuse` / `clarify` | 21 | returns before the type check |
| **total** | **252** | |

## What is and is not a defect here

**Not a defect.** The 79 missing-field and 21 refuse/clarify cases are returning
early *correctly*. A refusal has no answer type, and a book that does not report
broker cannot be graded on the type of a broker answer. Those cases are still
graded — on refusing correctly and naming the missing field — just not on this
axis.

**A real limit.** The 21 `parse_only` cases declare an answer type that nothing
ever checks. Two of them are known to be wrong (below).

**A real blind spot.** Four cases pass on this axis only because the grader
cannot distinguish one measure from several — see B3 in
`docs/mi_question_interpretation_programme.md`.

## Two known-wrong expectations, recorded and deliberately not corrected

Both sit in the `parse_only` group, so neither is evaluated and neither affects
the green result:

| Case | Question | Declares | Production produces |
|---|---|---|---|
| `pipe_183` | *pipeline amount by stage* | `expected_answer_type: currency` | `aggregation=count, metric=None` → `count` |
| `pipe_194` | *pipeline amount by broker* | `expected_answer_type: currency` | `aggregation=count, metric=None` → `count` |

`answer_type.asked()` reads *"amount"* as `currency`; the parser produces a
count. The disagreement is real, it is inside the bank, and the bank is
structurally unable to see it because `parse_only` returns before the check.

**These are recorded as known-wrong, and are NOT corrected during the
question-interpretation programme.** Editing a bank expectation while changing
the code the bank grades removes the control. They are listed here so the
figure is not read as covering them.

Two further `parse_only` cases in the same disagreement set — `pipe_184`
*expected funded by stage* and `pipe_188` *pipeline conversion by stage* — carry
the same structural exposure without being independently confirmed wrong.

## Working as designed — recorded so it is not mistaken for decay

56 of 252 stored expectations differ from what `answer_type.asked()` returns
today. **None is drift.** 33 of the 35 `currency`-versus-`any` cases carry an
`expected_metric` that justifies `currency` — 27 `current_outstanding_balance`,
5 `current_valuation_amount`, 1 `original_principal_balance` — and the 21 `none`
cases are authored from `expected_status`, not from the wording.

That is `derive_answer_type.py`'s documented cross-check working:
*"the question's own wording decides, cross-checked against the declared
expected_metric"*. A control is only known to be sound once someone has checked
it and said so, which is why this is recorded rather than dropped.

## How to quote the bank

* **Correct:** "260 passed; 131 of 252 answer-type expectations evaluated."
* **Correct:** "Every curated case holds its declared expectation. On the
  answer-type axis specifically, 131 of 252 are evaluated on this book."
* **Incomplete:** "252/252" — true of the cases, misleading about this axis.

## Standing rule

**Do not regenerate the calibration bank during the question-interpretation
programme.** All 252 `expected_answer_type` values were derived from
`answer_type.asked()`, which that programme changes. A bank that moves with the
code it grades has stopped being a control.
