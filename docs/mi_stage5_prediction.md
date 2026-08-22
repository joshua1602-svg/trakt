# Stage 5 — justification, the two defects, and the pre-registered prediction

Written before any of it is implemented, and before anything is measured beyond
the baseline in §2.

Base: merge-base `4e051f3`; `4e051f3` and `28ece25` both ancestors of HEAD
(`fdeb6be`).

---

## 1. Why this is worth doing, stated correctly

**The value is not that trends appear.** They already do — every routed
time-series question returns a series today.

The value is that **the time axis is the last place honour-or-clarify is not
enforced.** That rule was settled for periods in Tranche D, extended to
populations and groupings in this programme, and closed structurally for
reclassification in `b79f400`. On the time axis it is absent:

* a **weekly** request returns a **monthly** series, presented as the answer,
  with `comparison_period` stamped **applied**;
* a **twelve-month** request returns **three periods**, silently.

That is the same silent-substitution class this programme has closed everywhere
else. Anything said about better trend charts is beside the point.

---

## 2. The two defects are different, and are measured apart

They look alike — both are "the answer does not match the window in the
question" — and one rule for both would be wrong in one direction or the other.

| | COVERAGE LIMIT | GRAIN SUBSTITUTION |
|---|---|---|
| example | "balance by month over the last 12 months" against 3 snapshots | "balance by week over the last 6 months" against month-end snapshots |
| what happened | the whole available series was returned — all there is | a **different** series was measured and presented as the answer |
| was something substituted? | no | **yes** |
| what is owed | a **disclosure**: the window was shorter than asked | **honour-or-clarify**: refuse or ask, never a note |
| reader already exists | `requested_span` + `clarification` | `requested_unit`/`finer_than` + `granularity_clarification` |
| reached by `evolution`? | no — consumed by `period_movement` and the period-change route | no — consumed by `forecast_extrapolation` alone |

Refusing a coverage limit would reject an answer that is the best the book can
give, and honestly so. Disclosing a substitution is precisely what Tranche D
rejected for periods and this programme rejected for populations. The rules must
stay separate.

### Baseline, measured (`question_interpretation/time_axis_defects.py`)

```
book grain = month; governed periods available = 3

COVERAGE LIMITS      3, disclosed 0     (a disclosure is owed)
GRAIN SUBSTITUTIONS  2, disclosed 0     (honour-or-clarify is owed)
```

| id | question | coverage | grain | route | returned |
|---|---|---|---|---|---|
| C1 | balance by month over the last 12 months | short | — | `evolution` | "over 3 period(s)", verdict ok |
| C2 | funded balance over the last 6 months | short | — | point-in-time | refuse, but with the generic `comparison_period` message, not the coverage sentence |
| S1 | balance by week over the last 6 months | short | **substituted** | `evolution` | "over 3 period(s)", `comparison_period` **applied** |
| S2 | funded balance by week | — | **substituted** | `evolution` | "over 3 period(s)", no facet at all |

Neither owed sentence is produced anywhere, although both already exist and both
are correct.

---

## 3. Two findings the baseline turned up, recorded before building

**`requested_unit` does not recognise `day`.** `UNIT_PATTERNS` covers week,
month, quarter and year. "show me daily funded balance" reads as naming no time
unit at all and is answered as a single whole-book KPI — a third silent
substitution, and the most complete one, since no time axis survives at all. A
one-line addition to the single lexical owner, gated on `lexical_decisions`
because it changes a reading with a 693-question stability guarantee. Folded
into change 1, where the reading is the subject.

**Instrument corrections, instances 13 and 14 of the standing pattern**, both in
this baseline's own disclosure detector and both caught by reading rows rather
than totals:

1. it looked for the substring `"period(s)"`, which *"Funded balance over 3
   period(s)"* satisfies — so a twelve-month request answered over three periods
   read as **disclosed**. Stating what was USED is not disclosing what was ASKED.
2. it then looked for the requested period count as a substring. `"the last 6
   months"` matched the `6` in `"latest £1.96bn"`.

Same mistake twice: a detector matching something the output happens to contain
rather than the thing being claimed. It now requires the span label in words, or
a phrase only a deliberate disclosure produces.

---

## 4. Sequence, and why

**5, then 2, then 1.** Change 4 is deferred and respecified (§6).

* **5 — a facet raisable for a correct request, not only a problem.**
  `granularity_disclosure` today returns a facet *only* on a mismatch and
  hard-codes `status=UNAVAILABLE` at detection, so it never reaches a reconciler
  and nothing can adjudicate it. This lands the contract: the facet is raised
  either way, both adjudicators can stamp it from evidence, and `assess` carries
  the two distinct rules of §2. Demonstrated on the granularity path that
  already exists and is live (`geo_exposure`).
* **2 — a time axis distinguishable from a dimension axis.** A kind whose axis
  is a grain rather than a registry field.
* **1 — the reading, once, in the facet layer.** Every route inherits it. Not
  duplicated into `evolution`: a copy is a twelfth reader and defeats the
  premise of the whole programme.

Change 5 is first because it is what gives the grain reading somewhere to land.
Doing 1 first would wire a reading to nothing.

---

## 5. Pre-registered prediction

What may move, stated before measuring:

1. **The two owed sentences appear** where §2's baseline shows they do not:
   coverage disclosed on C1, C2 and S1; grain clarified on S1 and S2.
2. **S1 refuses or clarifies rather than answering.** It carries both defects,
   and a substitution outranks a disclosure — an answer with a note is exactly
   what honour-or-clarify forbids.
3. **The seasoning families do not move.** 20 of 20 by name, both books.
4. **No facet kind other than the time axis changes kind, status or count.**
5. **B5 stays unreachable** and the stamping matrix stays at **0 live holes** —
   the new kind must have a receiver on *both* adjudicators, which
   `test_reclassification_targets.py` will enforce rather than trust.

Four conditions stop the work and get reported rather than absorbed:

* a facet outside the pre-registered set moving;
* any movement in the seasoning families;
* a live hole appearing in the stamping matrix;
* answer text moving on a question that names no time axis.

Answer text on time-axis questions IS expected to move; that is the deliverable.
The 343-answer baseline will be re-recorded with the moves enumerated, never
silently replaced.

---

## 6. Change 4, respecified rather than deferred

The inventory scoped change 4 as *"`trend_grain` is never set from the
question"*. That is the wrong statement of the problem. `evolution` does not
consult `trend_grain` at all, and its only writer is
`interpreter/deterministic.py`, which is not on the serving path (backlog B4).
Setting a field nobody reads would repeat, in a new place, exactly the mistake
this stage exists to correct.

**The real change is that the route honours a requested grain.** A reader before
a writer. Respecified once change 5 gives the grain somewhere to land, and not
before.

---

## 7. T3, checked as a side effect

*"funded balance by quarter"* is claimed by no route and fails validation —
*"bar chart requires a dimension (or x)"*. `requested_unit` reads `quarter`
correctly; the parser sets a bar with no dimension because a time grain is not a
registry field and has nowhere to go. That is the same root as change 2.

Whether change 2 resolves it is an open question to be answered by measurement,
not assumed: giving the FACET layer a time axis does not by itself give the
PARSER's `dimension` slot one, and the validation failure is on the parser side.
Recorded here as a prediction to test: **change 2 alone will not resolve T3**,
because the facet layer and the spec are different objects and only the spec is
validated. If it does resolve, the reason will be stated; if it does not, what
would is logged.

---

# Stage 5 results — measured against §5

Changes 5, 2 and 1 applied in `bf287ea`, `502b831` and this commit. Change 4
stays respecified and unbuilt.

## Against the pre-registered prediction

| # | predicted | measured |
|---|---|---|
| 1 | the two owed sentences appear | **partly.** Coverage 0 → 2 of 3; grain 0 → 2 of 3. The two that remain are both on the point-in-time path (§below). |
| 2 | S1 refuses rather than answering | **yes.** It carries both defects and now refuses, naming both. |
| 3 | seasoning families do not move | **yes.** 20 of 20 by name, both books, at every step. |
| 4 | no other facet kind changes | **yes.** Zero new failures across the MI sweep. |
| 5 | B5 unreachable, matrix 0 live holes | **yes.** `granularity` stamped on both adjudicators; 17 holes, all designed. |

None of the four stop conditions fired.

### The measured effect

```
                     before   after
COVERAGE LIMITS       0 / 3    2 / 3
GRAIN SUBSTITUTIONS   0 / 3    2 / 3
```

| id | question | before | after |
|---|---|---|---|
| C1 | balance by month over the last 12 months | ok, "over 3 period(s)" | **refuse** — *"you asked about the last 12 months, which spans 12 reporting period(s); this answer covers 3, which is all this book carries"*, and `granularity: applied` for the month grain it did honour |
| S1 | balance by week over the last 6 months | ok, `comparison_period` **applied** | **refuse**, naming both defects separately |
| S2 | funded balance by week | ok, no facet at all | **refuse** — *"week — this answer is reported at month level, not by week"* |

## A correction to §2, from the measurement

§2 predicted a coverage limit is owed a **disclosure** and a grain substitution
**honour-or-clarify**. The first half is wrong, and the tree had already settled
it: `period_movement` returns `ok=False` with the clarification, and
`check_period_grain`'s older half stamps UNSUPPORTED. The reasoning is
`clarification`'s own — *offering the narrower window as the answer is the
substitution this guard exists to prevent*. A shorter window is a different
window.

So the two are **apart in cause, apart in message, together in verdict.** The
distinction the brief asked to preserve is real and load-bearing, and it governs
what the reader is told rather than whether they are told. Both refuse; one names
the window and what would be needed, the other names the level actually
delivered. Telling a reader their weekly request was answered monthly does not
tell them their twelve-month window was three.

## What is not closed, and why not

**C2 — "funded balance over the last 6 months"** takes the point-in-time path,
which publishes no series, so `check_window_coverage` correctly declines to act
on an unverifiable basis. It already refuses; only the message is the generic
`comparison_period` one rather than the coverage sentence.

**S3 — "show me daily funded balance"** takes the point-in-time path too, and
still returns a whole-book KPI. `time_axis_disclosure` returns nothing when the
route publishes no series, which is right in general — a KPI route must not be
told it failed to honour a grain nobody could expect of it — and wrong here,
because the question asked for a series and received a point.

Closing both means deciding that **a point-in-time answer to a series question
is itself a substitution.** That is defensible and probably correct, and it
would refuse a class of questions this stage has not measured — every "by month"
question that falls through to the point-in-time path. Not done on the strength
of two cases. Logged as **B9**.

## T3, checked as asked

**Change 2 does not resolve it, exactly as pre-registered.** "funded balance by
quarter" reaches not one of the nine instrumented sites: it fails at parser
validation — *"bar chart requires a dimension (or x)"* — before any facet is
built. The facet layer and the spec are different objects, only the spec is
validated, and validation runs first. Giving the facet layer a time axis cannot
help a spec that has nowhere to put one.

What would resolve it is a parser change: a time grain needs somewhere to live
on the spec, which is the respecified change 4 — the route honouring a requested
grain, with the spec able to carry it. Logged against that.

## Both standing surfaces are unmoved, again

343 of 343 answers identical; robustness 44/44 on both books. **Every question
this stage changes is in neither bank.** That is B6 for the third time, and the
reason the reachability check was run first: without it this would read as a
change that did nothing, when in fact it converts three silently-wrong answers
into three refusals that name what went wrong.

## Instrument corrections — instances 13 to 16

Four in one instrument, all the same shape: a detector matching something the
output happens to contain rather than the thing being claimed.

1. looked for `"period(s)"`, which *"over 3 period(s)"* satisfies. Stating what
   was USED is not disclosing what was ASKED.
2. looked for the period count as a substring — `"the last 6 months"` matched
   the 6 in `"latest £1.96bn"`.
3. read a 200-character truncation of the answer, and the disclosure sentence
   lands past it — so a correctly disclosed limit read as undisclosed.
4. the one-reader test matched its own docstring, the same defect a Stage 3
   source check had when it matched a comment. Rewritten to parse call sites
   with `ast`: a test that reads prose is not reading the program.

And a fifth of a different kind: `test_the_finest_unit_wins` pinned the literal
`"week"` as `UNIT_PATTERNS[0]` and went stale the moment `day` was added ahead
of it. Rewritten to assert the ordering property, which cannot go stale.
