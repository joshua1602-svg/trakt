# Root cause 1 — widening the time-axis vocabulary: PRE-REGISTRATION

**Written before the change was made and before anything was re-measured.**
Outcomes are recorded against these predictions in the same document, and no
prediction was edited after seeing a result.

Scope: **root cause 1 only.** Roots 2 (no implicit measure) and 3 (a coordinated
second dimension destroys a resolved time axis) are explicitly out of scope and
must be unchanged by this work.

---

## What the diagnosis found, and what it turns out to be

Root 1 was reported as "the time-axis vocabulary is narrow". Reading the
surfaces first changes the fix:

**`question_interpretation.lexical` already OWNS the time-axis vocabulary**, and
`mi_agent/period_request.py::requested_unit` already delegates to it, with a
docstring that states the gap outright:

> *"this reading … is already correct for every time-series probe, including
> 'by quarter' and 'each month', which the deterministic parser does not
> recognise as a time axis at all — and what has been missing is the
> **carriage, not the comprehension**."*

Measured against the owner directly:

| phrase | `lexical.time_axis_request` | parser `is_line` |
|---|---|---|
| `by month` | `by month` | yes |
| `over time` | `over time` | yes |
| `by period` | **`by period`** | **no** |
| `per period` | **`per period`** | **no** |
| `over the periods` | **`over the periods`** | **no** |
| `by quarter` | **`by quarter`** | **no** |
| `each month` | none | no |
| `between periods` | none | no |
| `month on month` | none | no |
| `by region` | none *(correct)* | no |
| `for the front book` | none *(correct)* | no |

So this is **two defects, not one**:

* **A — carriage.** Four forms the owner already reads are invisible to the
  parser, because `llm_query_parser`'s `is_line` keeps its own inline list. This
  is the dual-mechanism pattern again (`docs/mi_dual_mechanism_pattern.md`), in
  the time domain, and the fix is to consult the owner — not to copy its words
  into the second list.
* **B — a bounded gap in the owner.** Three lender forms the owner genuinely
  does not hold.

**Seasoning by name:** the owner already returns `None` for
`for the front book` and for `by region`. Seasoning windows and dimension
groupings are resolved by their own readers and must stay that way; nothing
below may make a seasoning phrase into a time axis. That is asserted, not
assumed — see prediction P6.

---

## The additions to the owner, each with evidence

No addition without a phrasing from one of the two banks that needs it.

| addition | evidence phrasing | bank |
|---|---|---|
| `month to month` | `how is the loan book tracking month to month` | widened T1 |
| `month on month`, `month-on-month` | `month-on-month change in balance by region` | declared T6 |
| `period on period`, `period-on-period` | `period on period movement by LTV band` | declared T6 |
| `each month`, `every month` | `give me balances per region each month` | widened T3 |
| | `balances by region and LTV band each month` | widened T5 |
| `between periods`, `between the periods` | `Which LTV band moved most between periods?` | declared T7 |
| `each period`, `every period` | sibling of `each month`, period form |
| `quarter on quarter`, `year on year` | siblings of `month on month`; `year-on-year` already appears in a separate parser list |

---

## Predictions

### P1 — carriage alone flips these to a resolved time axis
`balance by period`, `balance per period`, `balance over the periods`,
`balance by quarter`. **No owner change needed for any of them.**

### P2 — the owner's additions flip these
`balance each month`, `balance between periods`, `balance month on month`.

### P3 — `outstanding balances by period` DELIVERS
It is a T1 request whose only unresolved element was the time axis, and T1 has
no capability failures. This is the headline: a client-facing request for the
most-used shape, currently reaching no route.

### P4 — these do NOT change, because their blocker is root 2
`how is the loan book tracking month to month`, `what has the book done over the
last few periods`, `how have the big loans moved over time`, `how much did each
region move last month`. Each resolves **no metric**. A time-axis change cannot
reach them, and if any of them starts delivering, this change did something it
was not scoped to do.

### P5 — these do NOT deliver, because their blocker is root 3 or capability
`give me balances per region each month` (T3),
`balances per region each month for loans above 150k` (T4),
`balances by region and LTV band each month` (T5). Their time axis may now
resolve; the per-period breakdown still does not exist. **Predicted to move from
UNPARSED to CAPABILITY, not to DELIVERED** — a reclassification, not a fix, and
one that makes the capability count go UP.

### P6 — seasoning and dimension phrases gain no time axis
`balance for the front book`, `balance by region`, `balance by LTV band`,
`loans under 12 months old` must all still report **no** time axis. A widened
time vocabulary that swallows a seasoning window would be a silent substitution.

### P7 — the two surviving P0 refusals still refuse
`Show me balance by month by region and LTV band` and
`balance by month broken down by LTV band and region`. Both already resolve a
time axis, so this change should not touch them at all.

### P8 — aggregate
DELIVERED rises from 14/61. Recognition failures attributed to
`time axis not resolved` fall from 9. Capability count rises, because
reclassification moves questions out of UNPARSED into CAPABILITY. **A fall in
the capability count would mean this change built something, which it must
not.**

---

## Outcomes

Measured on both books, **identical phrasing for phrasing**.

| prediction | outcome |
|---|---|
| **P1** carriage flips `by period`, `per period`, `over the periods`, `by quarter` | ✅ all four |
| **P2** owner additions flip `each month`, `between periods`, `month on month` | ✅ all three |
| **P3** `outstanding balances by period` DELIVERS | ✅ |
| **P4** no-measure questions unchanged | ✅ — **after a correction, see below** |
| **P5** T3/T4/T5 reclassify, do not deliver | ✅ two moved UNPARSED → CAPABILITY; none delivered |
| **P6** seasoning and dimension phrases gain no axis | ✅ |
| **P7** the two surviving P0 refusals still refuse | ✅ |
| **P8** delivered rises, capability rises, time-axis cause falls | ✅ 14→15, 27→29, **9→5** |

```
                        before   after
  DELIVERED               14       15
  WORDING                  8        7
  UNPARSED                12       10
  CAPABILITY              27       29     <- rose, as predicted
  reached NO route        16       13
  T1 delivered           5/8      6/8
```

**Capability rose by two.** That is the correct direction: questions whose time
axis now resolves are reclassified out of "not understood" into "understood and
absent". Nothing was built. A FALL would have meant this change did P1's job by
accident.

---

## Two failures caught before shipping, both recorded

### 1. P4 was violated on the first attempt

With the axis alone deciding, `how is the loan book tracking month to month` and
`how is the front book tracking over the periods` — **neither of which names a
measure** — began answering **"Total Balance"**. The line path defaults a missing
metric (`elif metric is None: metric = _balance_metric(...)`), and the widened
axis pulled them into it.

That is root 2's substitution, arriving through a root 1 change. A guard now
applies the default only to the pre-existing vocabulary: an axis carried by the
widening does not, on its own, make a metric-less question answerable. Both
refuse again, exactly as before.

**Root 1 and root 2 are coupled through the line path's default.** That coupling
is a finding in its own right and belongs to whatever separate argument root 2
gets — it is not settled here.

### 2. The widening nearly shipped a WRONG ANSWER

Three readers ask "did this sentence request a time axis?" —
`lexical.time_axis_request` (the owner), `llm_query_parser.is_line` (sets the
chart type) and `chat_routing._EVOLUTION_MARKERS` (selects the route). With only
the first two widened:

```
balance over time     -> route=evolution   3 rows   [period, value]        correct
balance by period     -> route=None       13 rows   [vintage_year, sum]    WRONG
```

`by period` became a line, missed the evolution route, and was answered by the
generic executor as **13 origination vintages** — and the content rater scored it
PROVEN, because `vintage_year` is in its own `_TIME_HINTS` list. A cohort
distribution presented as a reporting-period series.

The parser's own note at that path had already warned: *"a VINTAGE is a cohort
label (2014, 2015, …), not a point on a time axis."*

**This is instance 5 of the dual-mechanism pattern**
(`docs/mi_dual_mechanism_pattern.md`), found by attempting to fix root 1 — and it
is the first instance where the weaker reader would have produced a confident
WRONG ANSWER rather than a refusal. `_is_evolution` now consults the owner, so
the chart type and the route are decided by one reading.

### 3. And the guard itself over-reached, once

The first draft of the no-measure guard moved `"over time"` out of the parser's
inline list and behind the guard, which stopped `Show pipeline by stage over
time` — no resolvable metric — from being a line at all. Caught by
`test_pipeline_by_stage_over_time_e2e`. The original vocabulary is now retained
verbatim and unguarded; the guard applies only to the increment.

---

## Verification

* `tests/test_time_axis_vocabulary.py` — 30 tests: every added term with its
  evidence, everything it must not swallow (seasoning by name, dimensions,
  spans, a unit inside a filter), the three readers agreeing, the vintage-axis
  refusal, and the no-measure guard with its control.
* Failure sets diffed before/after on two suites: analytical/parser/time
  selection **4 = 4**, `mi_agent/tests` + `mi_agent_api/tests` **12 = 12**. No
  new failures.
* Both books identical, phrasing for phrasing.
