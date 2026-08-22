# Stage 5 reachability — does a time-series question reach the carriage?

Established before building anything, for the reason the brief gives: the last
three changes each measured as something other than what they were, and the
common cause was reach.

Base: merge-base `4e051f3`; `4e051f3` and `28ece25` both ancestors of HEAD
(`1f8078d`).

---

## Verdict

**It reaches — but two of the five changes are scoped at sites the question
never passes through.**

A time-series question is claimed by the `evolution` route, which reaches four
of the sites Stage 5 would alter and none of the three that carry the grain
reading. Changes 2 and 5 land on a reached path and would be visible. Change 1's
only live consumer sits inside the `forecast_extrapolation` branch, which these
questions never enter. Change 4 would be inert until the route is taught to read
what it sets.

There is also a **live disclosure defect** the trace found, independent of Stage
5's scope, and it is the strongest argument for doing the work.

---

## 1. The trace, on the shipped service path

`question_interpretation/stage5_reachability.py` runs each question through
`execute_governed_mi_query` with routing exactly as shipped, wrapping every site
the five changes would alter so the trace records what ran rather than what
should have.

| | question | route | outcome |
|---|---|---|---|
| T1 | balance by month over the last 12 months | `evolution` | ok — "over 3 period(s)" |
| T2 | funded balance by month | `evolution` | ok — "over 3 period(s)" |
| T3 | funded balance by quarter | *(point-in-time)* | **fails** — "bar chart requires a dimension (or x)" |
| T4 | how many loans each month | *(point-in-time)* | refuse — `period_comparison/unavailable` |
| T5 | balance by month by region | `evolution` | ok — "over 3 period(s)" |
| T6 | balance by week over the last 6 months | `evolution` | ok — "over 3 period(s)" |
| T7 | balance by region *(control)* | *(point-in-time)* | ok — bar over 12 groups |

### Sites reached, for the briefed question T1

```
  -   period_request.requested_unit                 change 1
  -   period_request.finer_than                     change 1
  -   period_request.granularity_clarification      change 1 — its only live consumer
  1x  execution_receipt.detect_requested_facets     changes 2, 5
  1x  execution_receipt.granularity_disclosure      changes 2, 5
  -   execution_receipt.reconcile_facets            change 5 — point-in-time only
  1x  execution_receipt.reconcile_routed_facets     change 5 — the routed adjudicator
  1x  execution_receipt.assess                      change 5
  -   execution_receipt._split_named_dimension_roles change 3 — done, and not on this path
```

Identical for T2, T5 and T6. The point-in-time questions (T4, T7) reach
`requested_unit`, `reconcile_facets` and the split, and do not reach
`reconcile_routed_facets`.

---

## 2. What it currently returns, and the defect in it

Every routed time-series question returns the same sentence:

> Funded balance over 3 period(s): latest £1.96bn (up over the window).

The book holds exactly three governed monthly snapshots — 2026-04-30, 05-31,
06-30 — and `_filtered_funded_evo` enumerates every frame it can find. **Neither
the requested grain nor the requested window is read by anything.**

* T1 asks for **12 months** and gets 3. No warning, no note beyond *"3 governed
  period(s); source: …"*, which states what was used and is silent on what was
  asked.
* T6 asks for **weeks** and gets months. Same sentence, same silence, and the
  receipt stamps `comparison_period: applied`.

T6 is the sharp one. Honour-or-clarify was settled for periods in Tranche D and
extended to populations and groupings in this programme; the time axis is the
one place it is not applied. A question asking for a weekly series receives a
monthly one, presented as the answer, with the guard reporting the period facet
as applied.

**The reading needed to catch it already exists and is correct.** Measured
directly:

```
balance by week over the last 6 months     unit=week     finer_than_month=True
balance by month over the last 12 months   unit=month    finer_than_month=False
funded balance by quarter                  unit=quarter  finer_than_month=False
```

and `granularity_clarification("week", "month", "month-end funded snapshots")`
already produces:

> You asked about the last few weeks. This figure is measured from month-end
> funded snapshots, so the finest window it can express is one month. I have not
> answered over a monthly window in its place — ask for a monthly view, or for
> this to be measured from a series that carries weeks.

That is the answer T6 should get. It is never reached, because the one call site
is inside `chat_routing.py`'s `forecast_extrapolation` branch. This is the
inventory's finding — *"the reading already exists and is correct; it is the
carriage that is missing"* — now demonstrated end to end with a named
consequence on the shipped path.

---

## 3. The five changes, against what the trace found

| | change | reaches? |
|---|---|---|
| 1 | `requested_unit`'s reading reaching a facet | **No, as scoped.** Its only consumer is the `forecast_extrapolation` branch. For an evolution question the reading is never taken. |
| 2 | a time axis distinguishable from a dimension axis | **Yes.** `detect_requested_facets` runs on every routed question. |
| 3 | the grouping/filter conflation resolved first | Done in Stage 4, and not on this path — `_split_named_dimension_roles` is point-in-time only. |
| 4 | `trend_grain` set from the question | **Reaches nothing that reads it.** `trend_grain` is `None` on all seven, and `evolution` enumerates every available frame without consulting it. Setting it would be inert until the route reads it. |
| 5 | a facet raisable for a correct request | **Yes, but in the routed adjudicator.** `reconcile_routed_facets` and `assess` both run; `reconcile_facets` does not. |

So Stage 5 as briefed is **three-fifths reachable**: 2 and 5 land where the
questions actually go, 3 is done, and 1 and 4 are scoped against the
point-in-time path while every genuine time-series question is routed.

---

## 4. What would have to change for 1 and 4 to arrive

Not a design, and not built:

* **Change 1** needs its reading taken where the answer is produced, not only in
  the forecast branch — either in `evolution` alongside the existing check, or
  once in the facet layer so every route inherits it. The second is what "one
  owner for lexical interpretation" implies, and it is why change 5 exists.
* **Change 4** needs a reader before it needs a writer. `trend_grain` on the
  spec is written in one place (`interpreter/deterministic.py`, hard-coded
  `"monthly"`, and that interpreter is not on the serving path — backlog B4) and
  read by nothing in `evolution`. Setting it from the question first would
  produce a correct field nobody consults.

There is a sequencing consequence: **change 5 is the one that makes 1 and 4
matter.** A facet raised for a correct time-axis request, adjudicated in
`reconcile_routed_facets`, is what gives the grain reading somewhere to land and
the route something to answer to. Doing 1 or 4 first repeats the pattern this
check exists to prevent.

---

## 5. T3 — an unrelated live failure, recorded not fixed

*"funded balance by quarter"* is not claimed by any route and fails validation on
the point-in-time path:

> I could not build a governed query for this question: bar chart requires a
> dimension (or x).

`requested_unit` reads `quarter` correctly. The parser sets `chart_type: bar`
with no dimension, because a time grain is not a registry field and there is
nowhere for it to go — inventory finding 2, with a user-visible outcome. Logged
here; it is inside Stage 5's territory but is not one of the five changes.
