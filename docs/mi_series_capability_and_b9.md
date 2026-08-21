# Does anything return a series? — and B9, measured

Base: merge-base `4e051f3`; `4e051f3` and `28ece25` both ancestors of HEAD
(`e0c8e77`).

---

## 1. The question: does anything return a series?

**Yes.** The prose is a summary sentence; the payload is not.

`question_interpretation/series_emission.py`, on the shipped service path:

```
"funded balance by month"  ->  route=evolution, ok
   answer:   "Funded balance over 3 period(s): latest £1.96bn (up over the window)."
   artifact  type=chart  chartType=line  rows=3  series_rows=3
   artifact  type=table                  rows=3  series_rows=3
      {'period': '2026-04', 'value': 1932310991.2}
      {'period': '2026-05', 'value': 1946827440.6}
      {'period': '2026-06', 'value': 1964886258.21}
```

The run of values and a line chart of them are both emitted, in the response
payload every caller receives. **Trends as a capability exists**, and Stage 5's
changes were correctly scoped as carriage rather than construction. The refusal
improvements are not standing in for a missing capability.

Five of eight probe questions carry the run; the three that do not are a
period-change comparison (two dated snapshots, by design), a refusal, and a
pipeline question whose extracts are absent.

### But it is one-dimensional, and the receipt says otherwise

This is the finding that matters more than the answer above.

| question | table rows | facets |
|---|---|---|
| funded balance by month | `{period, value}` × 3 | `granularity: applied` |
| balance by month **by region** | `{period, value}` × 3 — **identical values, no region column** | `granularity: applied`, **`grouping_dimension: applied`** |
| how have the **front book and back book** changed over time | `{period, value}` × 3 — **identical values** | **`grouping_dimension: applied`** |

The second axis is silently dropped and the receipt **stamps it applied**. Three
different questions return byte-identical numbers — 1932310991.2 / 1946827440.6
/ 1964886258.21 — and the guard affirms the breakdown was honoured in two of
them.

The cause is in `reconcile_routed_facets`, in the code's own words:

```python
else:
    # The route grouped by something it declared; without a result
    # frame we cannot disprove it, and refusing on an unprovable
    # facet would disable working governed analytics.
    facet.status, facet.reason = APPLIED, ""
```

**A grouping on a routed answer is stamped APPLIED when it cannot be
disproved.** That is the exact inverse of the bar every other branch holds —
`reconcile_population`: *"a facet is APPLIED only when the route reports having
applied that field. A route that reports nothing leaves every population facet
LOST."*

And the premise is now false for this route. `evolution` publishes a result
frame — the artifact rows above — and those rows carry no region column. The
claim **can** be disproved, from evidence the route already emits.

This is the worst instance found in this programme: a confident number for a
different question, with the guard affirming the axis was honoured. It is not a
missing branch, so the stamping matrix reads it as "stamped" — correctly, in
that it *can* reach APPLIED. The matrix does not distinguish **stamped from
evidence** from **stamped by default**, and that is a limitation of the
instrument worth recording alongside the defect.

### So what is the remaining work?

Not the respecified change 4, and not "build trends". It is:

1. **A segmented series.** `evolution` answers every question with the
   whole-book metric by period; any second axis is dropped.
2. **The false APPLIED.** Provable from the artifacts today, and independent of
   (1) — the receipt should say `lost` for a dropped axis whether or not the
   route ever learns to segment.

(2) is smaller, is a correctness defect rather than a capability gap, and should
not wait for (1).

---

## 2. B9, measured before ruling

`question_interpretation/b9_series_substitution.py`, every question on every
available surface, through the shipped service path.

```
asked for a series (a time grain, or a series word)          90
   what the payload carried:
      series (3+ periods)                                    18
      two dated points (a comparison, not a series)          32
      refused                                                26
      one point                                              14

   of the 14:  stated the data is absent                      7
               forward-looking, answered in the present       6
               IN THE CLASS                                   1
```

**One question.** `funded_evolution_019`, *"Show regional concentration
evolution over time"*, answered by `geo_exposure` with a current concentration
snapshot and nothing said.

### Why the raw count is not the number

The first pass reported **51**. Every exclusion below is principled and stated
rather than argued away after the fact — a number that needs a caveat to be true
is the wrong number.

* **10 — the unit word names a dimension.** *"Show balance by origination
  year"* is a bar chart by vintage cohort and is correctly answered as one. Same
  shape as *"days in arrears"*, which is why `day` matches only grain
  constructions.
* **32 — two dated points.** *"How has lending to direct changed compared with
  acquired"* returns `£1.36bn → £1.39bn` across two governed snapshots. Not a
  series, but not a point either, and it did answer the question. Counted
  separately.
* **7 — the data is absent and the answer says so.** *"No reporting periods are
  available to build a pipeline amount trend."* Nothing was substituted.
* **6 — forward-looking, answered in the present.** *"Are any concentration
  limits likely to breach over the next quarter"* answered with today's limit
  status. **A real substitution**, and a serious one, but of the PROJECTION
  class, which has its own facet kind and its own rule. B9 must not be decided
  on it. Logged separately as **B10**.

### A finding from the exclusions: `year` has the defect `day` had

Two questions entered the raw class as `unit=year` because of *"how many loans
are over 80 **years old**"*. That is an age, not a reporting grain — the same
bare-noun failure `day` was caught on, except **`year` is already live** and
`day` was caught before it shipped.

Not fixed here. The owner's `year` pattern is untouched by this instrument, and
changing it is a reading change with a 693-question stability guarantee, which
belongs in its own commit with its own lexical gate. Logged as **B11**.

### The recommendation on B9 itself

One question, on one route, and it is `geo_exposure` — which publishes no series
and never will, being a concentration capability. The general rule ("a
point-in-time answer to a series question is a substitution") would fire on one
corpus question today. It is still the right rule, and it costs almost nothing
to apply — but it is not urgent, and the two findings it surfaced (B10's six
forward substitutions, B11's live `year` defect) are both larger than it is.

**Your ruling, on those numbers.** My reading: B10 before B9.

---

## 3. The regression test — the calibration bank cannot host it

Asked for as `pop_253/254/255` was. It cannot be done that way, and adding it
anyway would be worse than not adding it.

**The calibration bank runs `run_mi_agent_query` directly, bypassing routing
entirely** — backlog B7, logged in the inventory. All six trace questions behave
differently there from the path Stage 5 changed:

| id | question | in the bank harness | on the shipped path |
|---|---|---|---|
| T1 | balance by month over the last 12 months | refuse, generic `comparison_period` | refuse, **coverage sentence** |
| T2 | funded balance by month | ok, **no facets** | ok, `granularity: applied` |
| T3 | funded balance by quarter | validation failure | validation failure |
| T4 | how many loans each month | **ok** | refuse |
| T5 | balance by month by region | **clarify** | **ok (falsely — §1)** |
| T6 | balance by week over the last 6 months | refuse, generic | refuse, **grain + coverage** |

Four of six differ. Cases written against the bank would pin the point-in-time
behaviour and the estate would believe it had a Stage 5 regression test when it
had a test of a different path. Two of them would pin behaviour that is
*known wrong*: T4's `ok` is the B9 class, and T5's shipped `ok` is the false
APPLIED.

The robustness bank goes through `/mi/query` and therefore the real routing, but
its runner imports `nl_bank` as a frozen artefact and states that it is never
modified.

**So the honest answer is that the surface B6 has been asking for — one that
drives `execute_governed_mi_query` — is now a prerequisite rather than a
backlog item.** Three stages have ended with "both standing surfaces are
unmoved"; this is the fourth thing that cannot be regression-tested without it.
I have not built it unasked: it is a new measurement surface, and this
programme's rule is that a surface is agreed before it is trusted.

What I would build, for your approval: a routed bank in the shape of the
calibration bank — `id`, `question`, expected route, expected verdict, expected
facet kinds — driven through `execute_governed_mi_query`, seeded with the six
trace questions plus the geo granularity pair, and run as an instrument
alongside the other three.
