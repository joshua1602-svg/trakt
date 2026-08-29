# CFO acceptance bank — pre-registered thresholds

**Written and committed BEFORE the bank is run.** Base: frozen 7/7 baseline
`0af2d9f` plus Phases 1–3 (`4ca4320`, `1601332`, `b06c17f`). Nothing here is
adjusted afterwards; a breach is reported as a breach.

---

## 1. Classification, decided independently of the answer

Every question carries an expected classification written from the question and
the governed data, before execution:

* **CORRECT DELIVERED** — the answer addresses the requested analysis with the
  right population, measure, dimension, time, filters, ranking, economics and
  evidence.
* **HONEST GOVERNED REFUSAL** — the system lacks the information or capability
  and says so, without answering a different question.
* **WRONG / SILENTLY INCOMPLETE** — wrong population, measure or period; a level
  substituted for a movement; a dropped filter or dimension; a route
  substitution; a plausible answer to a different question; or an incomplete
  answer with no disclosure.

`ok=True` is **not** correctness. An answer that executed successfully and
answered a different question is WRONG, and an answer that quietly dropped a
clause is SILENTLY INCOMPLETE.

## 2. Thresholds

| gate | threshold |
|---|---|
| **Silent correctness** | WRONG / SILENTLY INCOMPLETE = **0**. Hard requirement. One unexplained silent wrong answer ⇒ NOT COMMERCIAL GO-LIVE READY. |
| **Correct delivery** | ≥ **80%** CORRECT DELIVERED across the applicable bank. |
| **Remainder** | predominantly HONEST GOVERNED REFUSAL. |
| **75–79% with zero silent errors** | reported separately, **not** an automatic READY. |
| **Critical families** | portfolio size/composition, historical trends, comparisons, filters, ranking, concentration must each be materially usable, and each must have a silent-error rate of **0**. A core family that mostly refuses is a commercial blocker even at a high overall percentage. |

## 3. Rules that bind this run

* The bank is frozen before it is run and is not edited to improve the result.
* Delivery is not forced to reach 80%. A question that should refuse and does
  refuse is a pass, not a miss.
* No capability is added to raise coverage.
* Only capabilities that exist today are tested. Arrears, cure rates, roll
  rates, NNEG and cohort/vintage are out of scope and are not in the bank.
* A newly discovered P0 is fixed only if clearly bounded, evidence-backed, low
  blast and necessary for launch. Anything else is reported, not fixed.

## 4. Environment

A client-shaped equity-release book: governed funded snapshots across several
reporting dates, a governed portfolio registry supplying `asset_class:
equity_release` and two source portfolios (direct and acquired), and the
governed Business Semantics Registry. Pipeline questions are included only where
the environment actually carries pipeline data; the standing caveat that
Pipeline Stage temporal evidence is fixture-proven and production-data-
unexercised is retained.
