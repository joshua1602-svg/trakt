# C6 prerequisite work — pipeline fixture built, and a live defect it exposed

Base `11a1181`. **C6 not executed.** No production code changed.

---

## 1. The five-week fixture

`tests/fixtures/pipeline_history_5w/` — eight cases across five consecutive
Fridays, written in the **canonical M2L/KFI weekly-extract schema**, discovered
by the ordinary governed globs, prepared by the shipped
`prepare_pipeline_mi_dataset`. **No production branch, alias or validation
change exists for it**, and a test asserts that nothing in `mi_agent_api/` names
it.

```
case     w1 05-01      w2 05-08      w3 05-15      w4 05-22      w5 05-29
2001     KFI           APPLICATION   OFFER         COMPLETED     COMPLETED
2002     KFI           KFI           APPLICATION   APPLICATION   OFFER
2003     OFFER         OFFER         OFFER         OFFER         OFFER
2004     APPLICATION   APPLICATION   APPLICATION   WITHDRAWN     WITHDRAWN
2005     -             KFI           KFI           APPLICATION   APPLICATION
2006     KFI           KFI           KFI           KFI           KFI
2007     APPLICATION   OFFER         OFFER         COMPLETED     COMPLETED
2008     -             -             KFI           KFI           OFFER
```

Entry (2005, 2008), progression (2001, 2002, 2007, 2008), stasis (2003, 2006),
withdrawal (2004) and completion (2001, 2007) are each present. Loan amounts are
100k–800k by case, so a wrong subtotal names the case that caused it.

**Measured, and matching the table computed by hand before the run:**

```
2026-05-01  6 cases  £2,300,000   KFI 3  APPLICATION 2  OFFER 1
2026-05-08  7 cases  £2,800,000   KFI 3  APPLICATION 2  OFFER 2
2026-05-15  8 cases  £3,600,000   KFI 3  APPLICATION 2  OFFER 3
2026-05-22  8 cases  £3,600,000   KFI 2  APPLICATION 2  OFFER 1  COMPLETED 2  WITHDRAWN 1
2026-05-29  8 cases  £3,600,000   KFI 1  APPLICATION 1  OFFER 3  COMPLETED 2  WITHDRAWN 1
```

Pinned by `tests/test_pipeline_history_fixture.py` — **14 tests**, written from
the movement table rather than copied back from a run, including one that
asserts the *movements* rather than five independent snapshots (a fixture whose
weeks are all the same shape would satisfy every count test and exercise
nothing).

## 2. Delivered coverage — before and after

| partition | before | after |
|---|---|---|
| `evolution_funnel` | 2 owned, **0 delivered** | 2 owned, **2 delivered** |
| `evolution_pipeline_stage` | 2 owned, **0 delivered** | 2 owned, **2 delivered** |
| `dataset=pipeline` | 7 owned, **0 delivered** | 7 owned, **2 delivered** |
| total | 32 owned, 14 delivered | 32 owned, **18 delivered** |

Two of the three route identities that could not be exercised at all now
execute with real numbers. **The plain pipeline single-metric series still
cannot**, and the reason is not the fixture.

## 3. STOP — LIVE PIPELINE EVOLUTION GRAIN DEFECT

`chat_routing._route_evolution` sets `period_field = "period"` unconditionally
(line 1025) and then plots `p.get(period_field)`. For the **funded** tape that
is right — `period` is the monthly run. For the **pipeline** tape,
`pipeline_evolution` emits `period = extract_date[:7]`, the year-month, while
the real weekly grain sits alongside it in `week`.

So a five-week pipeline series is plotted on **one x value**:

```
what the route plots (period_field="period")   what the series carries (week)
  x=2026-05   y=2,300,000                        x=2026-05-01  y=2,300,000
  x=2026-05   y=2,800,000                        x=2026-05-08  y=2,800,000
  x=2026-05   y=3,600,000                        x=2026-05-15  y=3,600,000
  x=2026-05   y=3,600,000                        x=2026-05-22  y=3,600,000
  x=2026-05   y=3,600,000                        x=2026-05-29  y=3,600,000

  distinct x values: 1 for 5 weekly points
```

The facet layer catches it and fails closed —

> *"I understood that you asked for week, but that could not be applied to the
> calculation (week — this answer is reported at month level, not by week)."*

— which is why the question refuses rather than showing a flat line. **The
guard is working; the route is wrong.** With real extracts spanning several
months every month's weeks would collapse onto a single point.

This is **pre-existing**, not caused by the fixture: these questions delivered
nothing before, when there were no extracts at all. Zero coverage hid it. It is
exactly the class of finding the four-part proof's *delivered coverage* leg
exists to surface.

Per the programme's standing rule — established by Defects A and B — a live
product defect is fixed **at its owner, in its own task**, never inside a
conversion. C6 must not proceed over it: converting `evolution` while its
pipeline series is mis-keyed would bake the defect into the compositional plan
and make the equivalence evidence certify it.

### The fix, named but not applied

`period_field` should follow the tape: `"week"` for pipeline, `"period"` for
funded. One line, at the route. **Not applied here** — this task's authority is
prerequisite closure, and that change alters shipped answers (four questions
move from refusal to delivery), which needs its own before/after blast proof.

## 4. Status

# STOP — LIVE PIPELINE EVOLUTION GRAIN DEFECT

Prerequisite 1 (pipeline fixture) is **built and non-vacuous**, and it did its
job on the first run. Prerequisite 2 (funnel/stage contract representation) was
not reached.

**Recommended next task:** fix the pipeline grain defect at
`_route_evolution` as its own bounded task — pre-registered blast radius,
before/after on the four affected questions plus the 882-question corpus,
regression by name — then resume C6 prerequisite 2.

## 5. What is committed

| | |
|---|---|
| `tests/fixtures/pipeline_history_5w/build_fixture.py` | deterministic generator |
| `tests/fixtures/pipeline_history_5w/2026-05-*/…csv` | 5 weekly extracts |
| `tests/test_pipeline_history_fixture.py` | 14 assertions |
| production code | **unchanged** |

Fixture cost is test infrastructure, **not** production migration cost, and is
reported separately from any future C6 figure.
