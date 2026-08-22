# Time-series capability — eight shapes, measured

Report only. No fix proposed and none made.

**58 runs — 29 phrasings × 2 books**, through `execute_governed_mi_query` with
routing as shipped, LLM off. Surface:
`question_interpretation/time_series_surface.py`.

**The rating rule, applied without exception:** a whole-book series returned for
a segmented request is **ABSENT**, not partial, whatever the receipt says. Every
rating below reads the ARTIFACT — the actual rows and their distinct values —
never `dimensionsApplied`.

---

## The table

| # | capability | rating | measurement behind it |
|---|---|---|---|
| 1 | metric × time | **PROVEN** | 4 of 4 phrasings, both books. Chart carries `period` with **3 distinct points**; series values differ per period. |
| 2 | metric × time × filter | **PARTIAL** | 1 of 4, both books. Only a **seasoning-population** scope works (*"for the front book"* → 3 periods scoped to `seasoning_segment`). All three **numeric/threshold** filters are refused: *"balance over 150 — this governed capability does not apply a row filter"*. |
| 3 | metric × time × dimension | **ABSENT** | 0 of 4, both books. Every phrasing **refused**, naming the loss: *"I understood that you asked for region, but that could not be applied… this answer covers the whole population"*. |
| 4 | metric × time × dimension × filter | **ABSENT** | 0 of 3, both books. All refused; the message names both the filter and the dimension. |
| 5 | metric × time × two dimensions | **ABSENT** | 0 of 3, both books. **2 of 3 return `ok=True` with the time axis silently gone** — see below. The third refuses. |
| 6 | period-over-period movement by segment | **ABSENT** | 0 of 4, both books. Routes to `period_change_analysis`; the segment is not carried. |
| 7 | ranked historical movement | **ABSENT** | 0 of 4, both books. No artifact combines a ranking with a time axis. |
| 8 | comparison of two historical segments | **ABSENT** | 0 of 3, both books. **1 of 3 returns a whole-book series with `ok=True` and nothing disclosed** — see below. |

**Both books are identical on every one of the 58 runs**: 11 answered,
18 refused, 3 silent drops each. So these are properties of the product, not of
one tape.

---

## The finding that matters: three silent drops

18 of the 29 runs per book **refuse and name what was lost**. That is the
honour-or-clarify contract working, and it is why most of the ABSENT ratings
above are safe absences rather than dangerous ones.

**Three are not.** They return `ok=True`, with `verdict: None`, `facets: []`,
`notApplied: []` — the guard did not run and nothing was disclosed.

### T5 — "Show me balance **by month** by region and LTV band"

```
ok: True   verdict: None   facets: []   notApplied: []
dimensionsApplied: ['Region', 'LTV Bucket']
period: 30 June 2026        comparisonPeriod: None
answer: "Here is the heatmap… covering 88 group(s)… 11,035 loans"
artifact: Region(12) × LTV Bucket(10), NO period column
```

The receipt's claim is *true as far as it goes* — both dimensions were applied.
**The words "by month" vanished.** A reader asking for a monthly series receives
a point-in-time cross-tab labelled as a complete answer. Same on both books
(Kestrelmoor: Region(11) × LTV Bucket(10)).

### T8 — "How have **direct and acquired** balances moved over the periods?"

```
ok: True   route: cohort_progression   verdict: None   facets: []
dimensionsApplied: []   notApplied: []
answer: "Funded balance for Total: tracked across 3 reporting period(s)…"
artifact: columns [period, funded_balance] — ONE series, 3 rows
```

**This is precisely the case the rule was written for.** Direct and acquired
never appear. The only trace is the word *"Total"* in the sentence, which a
reader takes as a heading rather than as notice that their question was
discarded.

---

## What each route does with these shapes

```
evolution                 15 runs   the main time-series route
period_change_analysis     5        period-over-period
analytical_composition     3
(no route / generic)       3        <- where the T5 heatmap comes from
funded_bridge              1
period_change              1
cohort_progression         1        <- where the T8 whole-book series comes from
```

The two silent-drop mechanisms sit on **different routes**: T5 falls through to
the generic executor, which has no notion of a time axis to lose; T8 is claimed
by `cohort_progression`, which tracks the whole book and does not carry a
segment. **They are two defects, not one**, and neither should be scoped as the
other.

---

## What this does and does not establish

**Does:** all eight shapes are now measured, on both books, against the artifact
rather than the receipt. Shape 1 works. Shape 2 works only for a governed
population scope. Shapes 3–8 do not work, and for 3, 4, 6 and 7 the product
says so.

**Does not:** these are 29 phrasings, not a corpus. A phrasing that works and
that I did not write would change a PARTIAL to a PROVEN — as it did for shape 2,
where three phrasings refuse and one answers. The ratings are floors supported
by evidence, not ceilings.

**Not measured here:** any of these shapes with the LLM parser on. Every run was
deterministic. The LLM-arm comparison covered fifteen point-in-time shapes and
none of these eight.
