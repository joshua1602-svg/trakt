# Conversion 2 — `period_movement` — pre-registered stop conditions

**Committed before any production change.** The cap is **unchanged at 240**, and
that is the entire point of this conversion.

Base: `f56bd35`. Conversion 1 commits present; working tree clean; Conversion 1
verified **live**, not silently falling back — 54 of 54 owned renders
contract-supplied, 54 compositional plans built, 0 deferrals.

---

## The experiment

Conversion 1 measured **383** production lines against this same 240 cap. The
hypothesis offered for the overrun was that it carried a **one-off cost**:
promoting the plan layer from a shadow instrument into production. On that
reading the *switch itself* was 94 lines and the rest was infrastructure the
next conversions inherit.

**Conversion 2 tests that hypothesis, and the cap stays where it is so the
evidence can answer.** Moving the threshold now would convert a falsifiable
claim into an assumption.

Two outcomes, both informative:

* **≤ 240** — the overrun is consistent with one-off infrastructure cost. Not
  proof that the migration is cheap; A1 still needs three conversions.
* **> 240** — the thesis is wrong or incomplete, and the migration economics
  must be re-baselined before Conversion 3. **No automatic progression.**

## Stop conditions

| # | condition | instrument |
|---|---|---|
| S1 | production lines changed exceed **240** | `git diff --numstat`, excluding tests and docs; renames counted at their real changed lines, not as churn |
| S2 | a **new primitive** is required | the plan's step list must stay within the existing seven, reusing existing implementations rather than adding a fifth `group` |
| S3 | economics breach the **A2** tolerance | ≥ £0.005 or one unit of the measure, on any owned case |
| S4 | payload/receipt equivalence needs a **bespoke `period_movement` exception** | a branch naming the route, a period, a book or a dimension |
| S5 | any **silent drop** | `time_series_surface` |
| S6 | any **silent population widening** | the precedence and sufficiency matrices |
| S7 | any **unexplained regression** | every registered gate, by exact case/test name |
| S8 | the conversion needs a **generic semantic concept the contract does not carry** | the target-state coverage matrix |

**S8 is the prerequisite condition.** The closure task recorded that
`period_movement` re-decides exactly two concepts downstream — source scope and
the time window — and that both are now carried. If a third appears, the closure
finding was incomplete and that matters more than this conversion.

## Cost is recorded in two parts

Because the hypothesis is about *which* cost is one-off:

* **shared infrastructure** — production lines in the generic plan layer;
* **route-specific** — production lines that exist only because this route was
  converted.

Tests and docs are **not** production-line cost and are recorded separately.

**Do not rationalise an overrun afterwards.** Same rule as Conversion 1, and it
held there: 383 was reported as a breach rather than argued down.
