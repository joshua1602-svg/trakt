# Where the 61 phrasings fail — recognition, or capability?

**Diagnosis only. Nothing proposed, nothing fixed.**

61 phrasings × 2 books, deterministic arm. **Both books identical, phrasing for
phrasing.** Instrument:
`question_interpretation/mi_recognition_diagnosis.py`.

---

## The headline

```
  DELIVERED                                            14  (23%)
  WORDING    — a sibling delivers this request          8  (13%)
  UNPARSED   — nothing understood, no sibling          12  (20%)
  CAPABILITY — understood, no wording reaches it       27  (44%)

  RECOGNITION total (WORDING + UNPARSED)               20  (33%)
  reached NO route at all                              16  (26%)
```

**Of the 47 failures, 20 are recognition and 27 are capability** — 43% against
57%. The plural finding was a fair signal: a material share of the 23% delivery
rate is words, not missing product.

But the number that matters is not the total. It is **where each kind sits**.

---

## By shape — the two kinds do not mix

| shape | n | delivered | wording | unparsed | **capability** | no route |
|---|---|---|---|---|---|---|
| T1 metric × time | 8 | 5 | 3 | 0 | **0** | 3 |
| T2 × filter | 8 | 1 | 1 | 2 | **4** | 3 |
| T3 × dimension | 8 | 0 | 0 | 1 | **7** | 1 |
| T4 × dimension × filter | 7 | 0 | 0 | 2 | **5** | 2 |
| T5 × two dimensions | 7 | 0 | 0 | 3 | **4** | 6 |
| T6 period-over-period by segment | 8 | 0 | 0 | 1 | **7** | 1 |
| T7 ranked historical movement | 8 | 3 | 2 | 3 | **0** | 0 |
| T8 comparison of two segments | 7 | 5 | 2 | 0 | **0** | 0 |

**T1, T7 and T8 have zero capability failures.** Every failure on those three
shapes is a wording failure. The capability is complete; a lender cannot reliably
reach it.

**T3–T6 hold 23 of the 27 capability failures.** Those are the per-period
breakdown family, and they are real. **P1's scope is untouched by this
diagnosis.**

T1 is the sharpest case, and it is the shape a client uses most: **5 of 8
delivered, 3 failed on wording, 0 on capability.** `outstanding balances by
period` and `how is the loan book tracking month to month` reach no route at all,
while `balance over time` answers.

---

## The causes are COMMON, not distinct

Three roots account for every recognition failure. Each was isolated with a
control that holds the rest of the sentence constant.

### Cause 1 — the time-axis vocabulary is narrow (9 instances)

```
balance by month              time axis: YES
balance over time             time axis: YES
balance by period             time axis: no
balance each month            time axis: no
balance per period            time axis: no
balance over the periods      time axis: no
balance between periods       time axis: no
```

Two accepted forms; five ordinary paraphrases of the same request rejected. The
measure is identical in every line, so the time phrase is the only variable.

### Cause 2 — the measure must be named explicitly (10 instances)

There is no default measure. A lender who says *"the book"* or asks *"how much
did each region move"* has named no metric, and the parse returns none.

```
how much did each region move last month              metric: no
how much did each region's balance move last month    metric: YES
what has the book done over the last few periods      metric: no
what has the balance done over the last few periods   metric: YES
```

Adding an explicit measure noun flips it in both cases. This is the same root as
the `balances` plural fixed before P1 — that was one missing surface form of one
word; this is the general case, where the word is absent entirely and nothing
supplies a default.

### Cause 3 — a second coordinated dimension destroys a resolved time axis (a distinct root, 4 instances)

```
balance over time by region                        time axis: YES
balance over time by region and ticket size        time axis: no
balance over time by region and LTV band           time axis: no
```

The time axis is resolved, and then adding `and <second dimension>` removes it.
This is T5's signature and explains why T5 reaches no route on 6 of 7 phrasings —
the worst no-route rate of any shape.

---

## A correction to my earlier reading

I previously reported that the plural `regions` was a dimension-vocabulary miss,
on the strength of `how are the regions trending` resolving no dimension. **That
was wrong, and the control says so:**

```
balance by region     dimension: YES
balance by regions    dimension: YES
balance by LTV band   dimension: YES
balance by LTV bands  dimension: YES
```

Dimension plurals resolve. `how are the regions trending` drops the dimension
because it resolves a **time axis**, and the spec cannot hold both limbs — the
same representation limit that makes T3–T6 absent, not a vocabulary gap. The
guard confirms it by naming the dimension back: *"I understood that you asked for
region, but that could not be applied."*

This distinction is why the verdicts below are not taken from the parse alone. An
empty `spec.dimension` beside a resolved time axis is the **capability** limit
wearing a recognition failure's clothes, and counting it as recognition would
have inflated the cheap category with the expensive one — reporting ~70%
recognition instead of the 33% that survives the control.

---

## How each verdict was decided

* **DELIVERED** — the artifact carries every limb the shape asks for.
* **WORDING** — not delivered, but a **sibling** delivers: same shape *and* same
  target, per the standing rule in `docs/mi_sibling_rule.md`. The request is
  reachable; this wording is the only barrier.
* **UNPARSED** — not delivered, no sibling delivers, an element the sentence
  named was not resolved, **and the system did not name it back**. The words
  themselves are unknown.
* **CAPABILITY** — understood (resolved, or quoted back in the refusal) and not
  deliverable by any wording in the bank.

The named-back test is what separates *"did not understand the word"* from
*"understood it and could not carry it"*. The honour-or-clarify refusal quotes
what it could not apply, which makes it usable as evidence.

---

## What this does and does not say

**Says.** Recognition is a third of the failures and is concentrated entirely in
shapes that already work — T1, T7 and T8 have no capability failures at all. The
causes are three common roots, not a long tail: two narrow vocabularies and one
interaction defect. All three were isolated by controlled probes, not inferred
from the failures.

**Says.** P1's scope is confirmed again and is not reduced. T3–T6 hold 23 of 27
capability failures, and their phrasings fail for reasons no vocabulary change
reaches.

**Does not say** how often a real lender picks a working phrasing. 61 phrasings
is a wider bank than 29; it is not the language, and a delivery rate across a
bank is not a user success rate.

**Does not say** what to do about any of it. No fix is proposed here.

---

## Reproducing

```
TRAKT_RUNTIME_MODE=development \
  python -m question_interpretation.mi_recognition_diagnosis \
    --book alderbridge --bank combined
```

The preflight from `docs/mi_measurement_environment_traps.md` applies: the run
refuses rather than producing an all-ABSENT table.
