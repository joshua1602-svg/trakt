# Conversion 1 — `portfolio_summary` — pre-registered stop conditions

**Written and committed BEFORE any production change**, so an overrun cannot be
rationalised after the fact. That has been this programme's discipline since the
abort conditions were pre-registered in Phase 0, and it is the reason those
conditions have held.

Base: `44bc90c` (Phase 1G). **Converted from `a56b7eb`**, which is `44bc90c`
plus the three target-state-closure commits — converting from `44bc90c` itself
would discard the `dataset` and `window_periods` contract fields that closure
added, and the readiness finding depends on them.

---

## The conversion stops if any of these is true

| # | condition | instrument |
|---|---|---|
| S1 | production lines changed exceed **240** | `git diff --stat` excluding tests and docs |
| S2 | a **new primitive** is required | the plan's step list; must stay within the existing seven |
| S3 | payload/receipt equivalence requires a **bespoke `portfolio_summary` exception** | a branch naming the route, a period, a book or a dimension |
| S4 | economics breach the existing **A2** tolerance | ≥ £0.005 or one unit of the measure, on any owned case |
| S5 | any **unexplained regression** appears | every registered gate, compared by exact case/test name |
| S6 | any **silent drop** or **silent population widening** appears | `time_series_surface` silent drops; the precedence and sufficiency matrices |

**One attributable change per commit.** A commit that moves two things at once
cannot be attributed, and attribution is what every gate here depends on.

## A1

A1's threshold needs three conversions before a median exists. **Conversion 1's
cost is recorded, and no median is inferred from it.** Recording a single
observation and calling it an estimate is the error A1 was written to prevent.

## What "equivalence" means here

Economics are already proven over the measured surface — 9 cases, 3 scopes, 0
differences, no bespoke exception, no new primitive, and since Phase 1G on the
governed population path. **That is not what this conversion is testing.**

The unproven boundary is everything after the numbers:

    compositional result -> production payload -> receipt -> answer

Phase 0 could not measure it, and said so: a shadow produces a plan and a
result, not an envelope. So payload and receipt equivalence is proved **before**
the switch, not after it.

## Order of work, fixed in advance

1. check the live wrong-number risk (dataset conflation);
2. prove payload and receipt equivalence;
3. make the bounded switch;
4. reconfirm economics;
5. remove the duplicate semantic owner **only with proof**.

Not the other way round. A conversion that switches first and measures
afterwards has no way to tell a conversion defect from a pre-existing one.
