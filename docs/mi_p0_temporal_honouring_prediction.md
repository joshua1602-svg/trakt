# P0 — the temporal honouring property

**Pre-registered before any implementation.** Base commit `99ea9ca`, branch
`claude/clause-splitting-phase-1-cft1wx`, working tree clean at the time this
was written. Every number below was measured at that commit, not estimated.

P0's bar, in the brief's own words: *no request is silently discarded.* It is
**not** that any new request is answered. Nothing in this stage adds capability;
the only outcomes it can produce are refusals in place of silent substitutions.

---

## 1. The property

> **TEMPORAL HONOURING.** When the sentence asks the answer to vary over time,
> the rendered object that ships must prove that variation from its own rows;
> and where the sentence names the segments the variation was asked for, the
> rendered object must prove that cut too. Where the rows do not prove it, the
> answer does not ship: refuse, naming what was lost.

One property, two limbs, one facet kind. It is deliberately not two route fixes.
T5 falls through to the generic point-in-time executor and never had a time axis
to lose; T8 is claimed by `cohort_progression`, which tracks the whole book and
carries no segment. Those are different causes. The contract is one, and it is
written to close anything else of the same shape — including instances nobody
has found.

### Proof comes from the artifact, never the receipt

This is the constraint that decides whether P0 works. T5's receipt was truthful:
`dimensionsApplied: ['Region','LTV Bucket']` is exactly what the answer did. It
was simply silent about time. A guard reading `dimensionsApplied` would pass T5
unchanged. So the check opens the artifact and counts distinct values in the
rows.

**A CORRECTION TO THE RULE, ruled on and accepted.** The brief said *"A
`period` column with three distinct points proves a time axis; nothing else
does."* This was pre-registered as a deviation; the ruling was that the WORDING
was wrong and this section records the correction against the rule rather than a
departure from it:

> *"'A period column with three distinct points and nothing else' would have
> refused three correct answers. You kept the binding principle — proof from the
> artifact, not the receipt — and admitted the column-pair form. Record the
> correction against my rule rather than as a deviation from it."*

The rule as corrected: **a time axis is proven by the rendered rows, in whatever
form the answer renders it.** Applied in its original wording it refuses three
answers that are correct today:

| question | route | artifact columns | why the literal rule is wrong |
|---|---|---|---|
| Which region has grown fastest? | `period_change_analysis` | `rank, category, start_value, end_value, movement, percent_movement, presence` | `start_value`/`end_value` **are** two points in time, one per column |
| How has the front book moved over time compared with the back book? | `analytical_composition` | `measure, population, period, prior, current, change` | `prior`/`current` are two points; `period` holds one span cell |
| Compare balance over time for direct and acquired | `analytical_composition` | same | same |

Refusing those would breach the other acceptance criterion — *no successful
query changes its calculation.* The binding principle is **proof from the
artifact, not from the receipt**, and a column pair naming two ends of a
movement is part of the rendered object the reader sees, not a claim about it.
So the time-axis proof admits two forms, both read from the rows:

* **Form A** — a time-named column with more than one distinct value.
* **Form B** — a pair of columns naming the two ends of a movement
  (`prior`/`current`, `start`/`end`, `opening`/`closing`, `previous`/`latest`,
  `before`/`after`).

`dimensionsApplied`, `filtersApplied`, `notApplied` and the guard verdict are
read by nothing in this check.

---

## 2. What is class, what is illustration

**Class.** Any answer about to ship, for a question that asked the answer to
vary over time, whose rendered rows do not carry that variation — or, where the
sentence named segments, do not carry that cut. Route-independent by
construction: the check runs after execution, on rows, and asks the same
question of every route.

**Illustration.** T5 and T8 are two members. They are named because they were
measured, not because the rule is built around them. A rule shaped to two cases
would be a pair of route fixes wearing a property's clothes.

**Evidence that the class is wider than the two illustrations.** Probing the
trigger across the standing banks before writing any product code found a
**fourth instance nobody had connected to this class**: `rt_005b`, *"Show
regional concentration evolution over time."* — answered by `geo_exposure` with
a current concentration table (`area, code, balance, count, share`, 15 rows) and
no series at all. Its own bank entry already records the correct outcome as
`refuse`, marked `expected_to_fail` pending **B9**. See §6.

---

## 3. Who owns what

The contract is *one owner per lexical decision; no twelfth interpreter*. Every
reading below is either an existing owner or a composition of existing owners.

| decision | owner | new? |
|---|---|---|
| what an axis marker is (`by`, `per`, `across`, `split by`, …) | `question_interpretation.lexical.AXIS_MARKER_RE` | existing |
| what a time unit is (`month`, `week`, `quarter`, `year`, `day`) | `question_interpretation.lexical.requested_unit` | existing |
| does the sentence ask the answer to vary over time | `question_interpretation.lexical.time_axis_request` | **new, composed of the two above plus one small series-phrase vocabulary** |
| what governed values this book carries | `execution_receipt.dimension_values` | existing |
| does the artifact prove a time axis | `execution_receipt` (new reader over rows) | **new** |
| does the artifact prove a cut | `execution_receipt` (new reader over rows) | **new** |
| the refusal sentence | `execution_receipt.assess` | existing — **not re-authored** |

The refusal wording is produced by handing `assess` a receipt carrying the lost
facet, so the sentence is the same one the eighteen existing refusals use:

> I understood that you asked for *X*, but that could not be applied to the
> calculation (*X — why*). I have not substituted a broader figure.

`time_axis_request` adds **no new unit words**. It finds an axis marker with the
existing regex, then asks the existing `requested_unit` about the single word
that follows it. That is why `"What is the balance by vintage…"` (rt_028) and
`"forecast run rate by vintage"` (rt_017) do **not** trigger — `vintage` is not
a time unit in the one vocabulary that defines them, and a vintage is a loan
attribute the grouping owner already handles. The only genuinely new words are
the grain-agnostic series phrases (`over time`, `over the periods`, `across
periods`, `month by month`, `time series`), which no existing owner holds.

**Limb 2's trigger needs no new vocabulary at all.** "Direct and acquired" is
not raised by any existing facet detector — `selector_mark` correctly says it is
a subject, not a selector, and `_COMPARISON_MARKERS` correctly finds no
comparison verb in *"How have direct and acquired balances moved over the
periods?"*. Rather than add a reading, limb 2 asks the **book**: two or more
distinct values from `dimension_values` named in the sentence is the segment
signal. On the 29 time-series phrasings that fires on exactly three, and they
are exactly the three segment comparisons.

---

## 4. The prediction

### 4a. Time-series surface (the instrument, 29 phrasings × 2 books)

| | before (`99ea9ca`) | predicted after |
|---|---|---|
| silent drops | **3** | **0** |
| honest refusals | 18 of 29 | **21 of 29** |
| answered | 11 | 8 |
| T1 metric × time | PROVEN | **PROVEN** (unchanged) |
| T2 × filter | PARTIAL | **PARTIAL** (unchanged) |
| T3–T8 | ABSENT | **ABSENT** (unchanged) |

**No rating improves.** P0 adds no capability; it converts three silent
substitutions into three named refusals. A rating that moves up is a stop
condition, not a success.

The three that move, and the sentence each is owed:

1. `Show me balance by month by region and LTV band` — 88-group heatmap over one
   period. **Limb 1**: no time column, no end-pair. Refuses.
2. `balance by month broken down by LTV band and region` — same. Refuses.
3. `How have direct and acquired balances moved over the periods?` — a
   three-point whole-book series. **Limb 1 passes** (`period` × 3). **Limb 2**
   fires: `direct` and `acquired` are two governed values, and the artifact
   carries `[period, funded_balance]` with no cut. Refuses.

### 4b. The eight answers that stand today

Predicted unmoved, with the proof each relies on:

| question | proof | limb 2 |
|---|---|---|
| Show me balance by month | Form A, `period`×3 | not triggered |
| balance over time | Form A, `period`×3 | not triggered |
| How has the funded balance moved over time? | Form A, `period`×3 | not triggered |
| total balance by reporting period | Form A, `period`×3 | not triggered |
| How has balance moved over time for the front book? | Form A, `period`×3 | one value named, not a comparison |
| Which region has grown fastest? | Form B, `start_value`/`end_value` | not triggered |
| How has the front book moved over time compared with the back book? | Form B, `prior`/`current` | `population`×2 — cut proven |
| Compare balance over time for direct and acquired | Form B, `prior`/`current` | `population`×2 — cut proven |

### 4c. Standing surfaces

The trigger was run across every bank at `99ea9ca` before writing product code.
It touches **11 questions in total**:

* **Calibration bank (251 distinct):** 4 trigger — `pipe_187`, `pipe_192`,
  `fcast_195`, `fcast_196`. All four are `execution: parse_only` with
  `expected_artifact_type: none`, so the bank grades them at parse level and
  never reaches this guard. **Predicted movement: zero.**
* **Robustness bank (44 × 2 books):** the `Q8` shape *"Are {a} and {b} balances
  developing differently over time?"* triggers both limbs on all three pairs.
  All twelve Q8 alderbridge variations were dry-run: every one stands, because
  `analytical_composition` returns `prior`/`current` (Form B) with `population`
  carrying two distinct values (cut proven). **Predicted movement: zero. The
  seasoning families are unmoved by name.**
* **Routed surface (30 distinct):** 6 trigger. Four already refuse. `rt_002`
  *"funded balance by month"* stands on Form A. **`rt_005b` moves — see §6.**
* **Shipped shapes (15):** 0 trigger. **Predicted 15/15, unmoved.**
* **Answer differ:** predicted to move on exactly the answers named above and
  nowhere else.

### 4d. One false positive found and excluded before implementation

`completions by month` (calibration `pipe_187`) answers, on the shipped path,
`ok=True` with **no artifact at all** and the text *"No weekly Completed
extracts are available yet."* A naive rule refusing every triggered answer that
cannot prove an axis would refuse it. That would be wrong: it ships no figure
and it already tells the reader plainly that the data does not exist. Nothing
was discarded.

So the property fires **only on an answer that ships a rendered artifact**. If
you ship an object, the object must prove the axis; if you ship nothing, you are
not asserting a series.

---

## 5. Stated limits — what P0 does not prove

Named now rather than discovered later. A vocabulary gap is not bounded by the
cases that fail.

1. **Limb 2 proves the answer was cut, not that it was cut by the right thing.**
   An answer split by the wrong segment passes. "The right thing" belongs to the
   grouping-evidence owner (D7/B12); where that owner cannot see a route's axis,
   the residue is the segmented-series backlog, which P1/P2 own and P0 does not.
2. **A prose answer that states a figure with no artifact escapes.** The rule
   requires a shipped object. The one instance in evidence (`pipe_187`) ships no
   figure, so the gap is currently theoretical — but it is a gap, and it is not
   closed here.
3. **Form B accepts any pair of end-named columns.** A table naming
   `prior`/`current` proves two points exist; it does not prove they are the two
   points the sentence named. Window and grain are different defects with
   different owners (`check_window_coverage`, `check_period_grain`) and are not
   re-adjudicated here.
4. **The instrument stays independent of the product.** `time_series_surface`
   keeps its own artifact reader. Sharing one would make the surface unable to
   catch the guard being wrong, and seven instruments in this programme have
   already failed to see the change they were built to measure.

---

## 6. B9 — declared, not smuggled

`rt_005b` is on the routed surface as `expected_to_fail`, and its recorded
reason is **B9**: *"the one question of 90 that asks for a series and SHIPS a
point. `geo_exposure` answers with a current concentration snapshot and says
nothing about the series that was asked for. The expectation below is the
CORRECT outcome, so this case fails until B9 is ruled on and closed, and passes
the moment it is."* Its declared expectation is `expect_verdict: refuse`.

B9 is on the brief's list of items **not authorised**. It is being reported, not
opened: B9 is a member of the class this property defines, so closing the class
closes it as a consequence, and `rt_005b` moves from failing to passing against
its own recorded expectation. The alternative would be to carve B9 out of a rule
that otherwise catches it, which would make the property a set of exceptions
rather than a contract.

**RULED: let B9 close.** In the ruling's own words:

> *"It is a member of the class the contract now covers, reached by a third
> route, and its bank entry already recorded refuse as correct. A carve-out to
> keep it artificially open would be worse than closing it. Record it as closed
> by the contract rather than by a fix."*

So the backlog entry for B9 is **closed by the contract, not by a fix**. Nothing
was written for `geo_exposure`, nothing was written for a series, and no line of
code names B9. What changed is that an answer which could not prove a requested
time axis from its rows stopped shipping — and B9 was one such answer, found by
the property rather than by anybody looking for it. That distinction is the
whole argument for stating this as a property: a fix closes the case you aimed
at, a contract closes the ones you did not know about.

---

## 7. Stop conditions

Implementation stops and reports, rather than continuing, at the first of:

* any answer moving that is not in §4a or §6;
* any of the eighteen existing refusals changing its wording or its route;
* any time-series **rating** improving (P0 must not add capability);
* any Q8 robustness variation moving on either book;
* the shipped shapes falling below 15/15;
* the estate failing a test that did not fail at the base commit, BY NAME —
  see §13 on why the count of 60 is not the comparison;
* a movement in any surface I cannot attribute to a named cause.

---

## 8. What is measured before the prediction can be called right

Both books, all 29 phrasings; the four standing surfaces; the estate; the
shipped shapes; the twelve Q8 variations on both books. Then the frozen baseline
is re-recorded, and the commit it was taken at is stated.

---
---

# P0 — the result, measured

Implemented across three commits from the base, each with its own before/after:

| commit | change |
|---|---|
| `924eaa8` | the sentence-side owner, inert — nothing read it |
| `4c2cf9b` | limb 1: a requested time axis must be proven by the shipped rows |
| `1b90fe4` | rt_005b re-declared (B9 closed as a consequence) |
| `1f79f19` | limb 2: the segments the sentence named must survive the series |

## 9. Prediction against outcome

| prediction (§4) | outcome | |
|---|---|---|
| silent drops 3 → 0, both books | **3 → 0 on alderbridge, 3 → 0 on kestrelmoor** | ✅ |
| honest refusals 18 → 21, both books | **21 of 29 on both** | ✅ |
| exactly three runs move per book | **exactly three, and the same three, on both** | ✅ |
| no shape rating changes | **every rating identical, both books** | ✅ |
| the eight standing answers unmoved | **unmoved** | ✅ |
| calibration bank unmoved | **267 passed** | ✅ |
| robustness bank unmoved, both books | **CORRECT 32 / UNHELPFUL 6 / SAFE 4 / DISCLOSED 2 on each** | ✅ |
| the three seasoning families unmoved by name | **Q1 4/4, Q7 4/4, Q8 12/12 CORRECT on both books** | ✅ |
| shipped shapes 15/15 | **15 correct, 0 wrong, 0 refusals of either kind** | ✅ |
| routed surface: only rt_005b moves | **32 passed / 0 failed / 0 declared defects**, from 31/0/1 | ✅ |

## 10. The three runs that moved, in full

**T5 — the time axis, two phrasings, both books.** An 88-group heatmap over a
single reporting period, `ok=True`, no verdict, no facet, no note. Now:

> I understood that you asked for a series by month, but that could not be
> applied to the calculation (a series by month — the answer that was produced
> carries no time axis; it reports a single position and cannot show movement).
> I have not substituted a broader figure.

**T8 — the segments, one phrasing, both books.** A three-point whole-book series
answering a question about two segments, labelled *"Funded balance for **Total**:
tracked across 3 reporting period(s)"*. Now:

> I understood that you asked for Direct and Acquired tracked separately, but
> that could not be applied to the calculation (Direct and Acquired tracked
> separately — the answer that was produced is a single whole-book series; it is
> not split by the segments you named). I have not substituted a broader figure.

Both sentences are `assess`'s, not this stage's.

## 11. Two corrections earned during implementation

Recorded because the discipline is that a fix measured only by "does the right
thing happen" stops at the first case that works.

1. **The unit owner was asked the wrong question.** The first composition asked
   `requested_unit` about the noun alone, and `"balance by day"` read as no time
   axis at all — `UNIT_PATTERNS` holds `day` only as phrases (`by day`, `per
   day`, `daily`), never as the bare word. Caught by the test that derives its
   cases from `UNIT_PATTERNS` rather than listing them. The owner is now asked
   about the marker and the noun together.

2. **A blacklist reported a cut on the answer it exists to catch.** Limb 2 first
   asked whether any column that was neither a time nor a measure varied,
   against a list of measure words. `cohort_progression` ships a second table
   carrying `wa_ltv` and `wa_interest_rate` beside the whole-book series, and
   neither word was on the list, so the whole-book series "proved" a cut.
   Replaced by a structural rule with no vocabulary at all: a row set is cut
   when it carries more rows than an uncut answer of its shape would. That
   count differs by form — a series carries one row per point, a movement table
   carries one row — and reading both as "one row per point" then refused the
   two `analytical_composition` answers that correctly track front book against
   back book. Both errors were found by measurement, not by review.

## 12. What P0 did not do

No capability was added. Every time-series shape carries the rating it carried
before: T1 proven, T2 partial, T3–T8 absent. Three answers that were confident
and wrong are now refusals that say what was lost. The eight shapes are no
closer to working than they were at `99ea9ca`; they are only no longer silent.


---

## 13. The estate is compared by NAME, and why 60 is not the number

**Ruled:** *"Estate by name, not by count — agreed, and state why in the pack.
60 was never portable; it came from an environment carrying lxml, python-pptx
and pyarrow, which this box lacks. Any figure quoted from that number needs the
environment named alongside it."*

The full suite on this box reports **189 failed, 9560 passed, 36 skipped,
8 xfailed, 43 errors** in 39 minutes. The 43 are collection errors, and they are
imports rather than defects:

| missing module | tests it takes out |
|---|---|
| `lxml` | the Annex 2 XML builder and XSD suites |
| `python-pptx` | every `tests/mi_agent_pptx/` module |
| `pyarrow` / `fastparquet` | parquet-backed serving paths |

So **189 and 60 are not two measurements of the same thing**, and neither number
is portable on its own. A failure count is only meaningful beside the
environment that produced it, and the standing 60 was never recorded with one.
Any future quotation of either figure must name its environment.

What IS portable is the SET of failing test names, compared between two trees in
one environment. That is the comparison P0 is held to: every failure at the P0
head must also fail at `99ea9ca`, with the same name.

### Two P0-adjacent names, attributed before the long run

Five failures carried names close enough to this work to check immediately
rather than wait:

    tests/test_p0_cohort_identity.py::test_a_sourcing_cohort_still_answers
        [Compare the direct and acquired books on balance, loan count,
         weighted-average LTV and average borrower age.]
        [How does the direct book compare with the acquired book on borrower age?]
    mi_agent/tests/test_p0_execution_receipt.py
        ::test_receipt_discloses_a_requested_dimension_the_dataset_lacks
        ::test_an_unavailable_dimension_is_never_replaced_by_another_field
        ::test_an_unavailable_dimension_never_simply_disappears

The first two name `direct` and `acquired` — the exact governed values limb 2
reads — so they were the ones to rule out first. **All five reproduce identically
at `99ea9ca`.** The cohort pair never reaches the property at all: neither
sentence asks the answer to vary over time, so `time_axis_request` returns
``None`` and neither limb fires. They fail on `KeyError: 'facets'`, which is
older than this stage.

### The set difference

Both trees run in this environment, one at a time (three concurrent suites, one
of them stale, were killed first — a contended run is not a measurement).

| | base `99ea9ca` | P0 head |
|---|---|---|
| failed | **189** | **189** |
| passed | 9495 | **9560** |
| skipped / xfailed / errors | 36 / 8 / 43 | 36 / 8 / 43 |
| wall clock | 38:41 | 39:12 |

The 65 extra passes are this stage's own tests: 29 on the sentence-side owner,
36 on the property. Nothing else changed count.

By NAME the two sets differ in exactly two places, and neither is a regression.

**NEW at the P0 head — one, real, and acted on.**

    question_interpretation/tests/test_stamp_coverage_instrument.py::test_no_hole_is_live

The facet-stamping coverage instrument reported `series_axis` as a LIVE HOLE in
sixteen (route, kind) cells: *"a facet kind can now be raised somewhere that
cannot confirm it."* The instrument is right that no reconciler branch receives
it, and that is the design — the facet is raised AFTER execution from the
rendered rows, arrives LOST with its reason already set, and goes straight to
`assess`. A reconciler reading declared evidence is precisely what T5 defeats.

That matches the instrument's own criterion for a DESIGNED hole — *"a kind
constructed with a status and a reason already set, which the detector has
finished adjudicating"* — so it is declared in `DESIGNED_HOLES`. Declared **with
a proof, not an assertion**: three tests require that the real raiser's output is
already adjudicated, that exactly two construction sites exist and both are
inside that one function, and that the declaration cannot outlive the kind. If a
second raiser ever appears — especially one running before execution — the hole
stops being designed and the instrument is right again.

**"FIXED" at the P0 head — one, and it is an artefact of the measurement, not a
change in the product.**

    mi_agent/tests/test_registry_governance.py::test_checked_in_registry_matches_generator

It fails at base and passes at the P0 head, which is the wrong direction for
anything P0 did — and P0 touches neither file. Cause: the checked-in registry
records its source as an ABSOLUTE path,

    source_registry: /home/user/trakt/config/system/fields_registry.yaml

and the generator regenerates it from wherever the tree is checked out. The base
run was a git worktree under `/tmp`, so the paths differ and the metadata
assertion fails. Both input files are byte-identical between the two trees
(md5-checked), and the `fields` block matches entry for entry; only
`source_registry` differs. **The test fails in any checkout not located at
`/home/user/trakt`.** That is a real brittleness in that test, unrelated to this
stage and not authorised here — recorded, not fixed.

**So: zero new failures by name that P0 caused and did not close, and zero
fixes.** The estate criterion is met on that reading, and the reading is stated
rather than reduced to a count.
