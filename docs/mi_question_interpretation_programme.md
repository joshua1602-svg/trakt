# Question interpretation — programme brief (living)

The standing conditions and backlog for the question-interpretation contract,
kept in the repository so an amendment has somewhere to live. Base `4e051f3`;
release candidate `28ece25`, unchanged and shippable throughout.

## Standing conditions

1. **Every measurement runs both surfaces, deterministic arm, at every stage.**
   **AMENDED — see below.**
2. One change per commit, each with its own before/after. Stop at the first
   unattributable movement.
3. Confirm the base commit before reporting anything.
4. No instrument ships without a test proving it can fail.
5. Report a flat result as flat. Do not re-author an instrument after seeing
   its result.
6. Do not weaken a rule to make a change fit.
7. **Every stage diffs answer TEXT, not only grades.** See below.

---

## Amendment 1 — standing condition 1, the measurement arm

**Adopted.** Standing condition 1 previously required the calibration bank and
the 44-variation robustness bank without naming an arm. It now requires the
**deterministic arm of both surfaces at every stage.**

### The reason

The LLM arm **self-disagrees on 6–10% of cells** — larger than any change this
programme will make. An instrument whose own noise floor exceeds the effect
size cannot attribute a movement to a cause, which is the entire purpose of
running it at each stage. The deterministic arm has always been the
attribution-grade instrument.

This is not a workaround for the missing API key. It is the correct instrument
for per-stage attribution, and it happens also to be the one that runs here.

### What this changes in practice

| | Before | After |
|---|---|---|
| Per-stage gate | ambiguous | deterministic arm, both surfaces, both books |
| Robustness bank | 752 runs, LLM on | 88 runs, LLM off, reproducible run-for-run |
| Repeat variance | measured | not applicable — a deterministic arm cannot vary |

### What is reserved, not discarded

**One LLM-arm run is reserved for the final merge decision**, if a key becomes
available. Merging is a separate decision taken on the full result, and that is
the point at which the noisier, more realistic arm is the right instrument.

### What this amendment does not claim

A regression that manifests **only** through the LLM parser will not be caught
by per-stage measurement. That is accepted, deliberately, in exchange for
attribution. The reserved merge-decision run is the control for it.

### Numbers that must not be compared across arms

The recorded **91.0% correct/disclosed** and the **160-run regression** are
LLM-arm measurements. The deterministic arm's **34 of 44 correct/disclosed** is
a different arm on a different denominator. Neither figure should ever be
quoted against the other.

### Deterministic baseline, as at `1863b1b`

| | alderbridge | kestrelmoor |
|---|---:|---:|
| `CORRECT` | 32 | 32 |
| `SAFE_REFUSAL` | 10 | 10 |
| `CORRECT_WITH_DISCLOSED_LIMITATION` | 2 | 2 |
| unsafe outcomes | 0 | 0 |

Same verdict on **44 / 44** variations across both books. Seasoning family
(Q1 4, Q7 4, Q8 12) — **20 / 20 `CORRECT`**, per book, by name.

Reproduce: `python -m question_interpretation.run_robustness_deterministic --all-books`

---

## Standing condition 7 — diff the answer text, not only the grades

**Adopted, and it is what covers Finding 1 for the duration of the programme.**

Every stage compares the **answer text byte for byte**, on both surfaces,
deterministic arm — not merely whether each case still grades as passing.

### Why this is the safe form

The bounded pre-Stage-2 check found four cases (`kpi_028`–`kpi_031`) whose
`expected_answer_type: mixed` is satisfied by an observed type of `count`, so a
portfolio summary that silently dropped its balance measure would **pass the
bank**. The grader cannot see that regression.

The answer diff can. A summary that lost a measure produces different text.

> **Byte-identical answer text is strictly stricter than the grader.**
> A change that passes the bank and fails the diff is a real regression the bank
> could not see. A change that fails the diff and passes the bank is stopped.

That is why option (a) — proceed, record, and fix nothing in the grading path —
is safe rather than merely convenient. It does not rely on the grader being
right about `mixed`; it relies on the text not moving.

### Why the grading path is not touched instead

`of_measure` and `_SATISFIES` are **graders**. Changing a grader mid-programme
invalidates every measurement taken before the change, because the before and
the after are no longer scored by the same instrument. A moved grader is worse
than a weak one: a weak grader has a known blind spot, a moved grader has no
comparable history.

---

## Amendment 2 — join sequencing, Stage 2 vs Stage 3

**Adopted.** The filter join is built in halves.

**Stage 2 emits the facet half.** `_detect_thresholds` and
`_detect_geographic_scope` already compute `match.start()` / `match.end()` and
discard them; recording those into the object is **additive** and cannot move an
answer. Parser-side claims are recorded as **spanless**, and the join is reported
as **half-built, with the missing half named**.

**Stage 3 supplies the parser half**, when `_parse_filters` is converted as a
consumer. `_parse_filters` rewrites the question as it consumes clauses
(`work_q = work_q[:bm.start()] + " " + work_q[bm.end():]`), so sound spans need
either removing the rewrite or maintaining an offset map through it. **That
choice is made then, on measurement rather than preference.**

**Why not in Stage 2:** Stage 2's guarantee is byte-identical answers. Changing
how `_parse_filters` rewrites the question forfeits that guarantee, and a stage
whose acceptance is "nothing moved" cannot also be the stage that moves
something.

---

## Amendment 3 — the principle behind removing `coverage` and keeping `CONFIGURED`

**Adopted.** *Remove things that can be wrong; keep empty things that can only be
unused.*

An unused **operation type** can still misclassify later — it is a value
something may be assigned. An unfilled **slot state** is inert: it describes a
condition nothing currently reports. That distinction, not the presence of a
rationale, is what justified removing `coverage` and keeping `CONFIGURED` marked
unsupplied.

**The contract's rationale for the configured-target sense is not evidence.**
The wording (*on target*, *versus plan*, *versus budget*) appears in **0 of 690**
real-surface questions, which contradicts it. The slot is retained because it is
inert, not because the corpus supports it.

**Review rule:** if the configured-target wording does not appear in the
client's real questions **within the first month**, the slot is removed.

---

## Note 1 — a governed-config dependency here is correct

**For the record, to prevent a false conflict later.**

`_analytical_population_satisfies` now reads the governed seasoning
configuration to resolve a lending window before comparing. That does **not**
contradict the reason conversion 4 was left partial.

| | |
|---|---|
| `question_interpretation.lexical` | the LEXICAL owner. Reads question text and nothing else — no registry, no frame, no spec, no config. `requested_span` stayed out of it because resolving vague recency against `lending_windows.recent_max_months` is a config decision. |
| `execution_receipt` | the SEMANTIC layer, downstream. Resolving a governed concept is exactly its job. |

The principle protects the lexical owner's domain-blindness. It is not a general
prohibition on configuration, and applying it downstream would be a
misreading — a semantic layer that cannot consult the governed model cannot do
its job at all.

## Note 2 — generate the coverage where the corpus cannot exercise a construct

**Adopted as the standing pattern.**

Removing the `_parse_filters` rewrite could not be proved by the corpus: exactly
**one** of 690 real-surface questions contains `between`, the only construct the
rewrite existed for. So the old algorithm was reproduced verbatim against the
same helpers and compared across a generated set built to hit every shape the
construct appears in — **11,474 questions, 22,948 comparisons, 0 mismatches**.

The rule this sets:

> Where the corpus cannot exercise a construct, **generate the coverage rather
> than declare it untested.**

"The corpus does not cover this" is a statement about the corpus, not evidence
about the change. A construct rare in the corpus is not rare in the field, and
the one case that exists cannot distinguish a correct rewrite from a lucky one.
This applies to any conversion touching a path the banks barely reach.

---

## Standing rule — an instrument tends to carry the defect it was built to find

Not an incident. A pattern, recorded because it has now happened often enough
that it must be designed against rather than caught by luck. Every instance was
found by chance or by a late cross-check, and each one would have shipped a
false clean result.

1. The Phase 1 measurements were taken 136 commits off the intended base. Every
   score was internally consistent and every one was void.
2. The calibration bank graded against `build_fixture` rather than the real tape
   — the surface built to prove the tape's answers, not reading the tape.
3. `answer_diff` keyed on `(intent, variation)` and silently dropped 16 of 88
   robustness answers, all of them the seasoning family: the differ built to
   catch seasoning movement could not see the seasoning questions.
4. Two of the 14 mutations did not reproduce the defect they named, so the
   mutation suite that proves the instruments can fail contained instruments
   that could not fail.
5. My own role-split test asserted the Stage-1 role value rather than the value
   production gives.
6. The source-check test for the removed rewrite matched a comment describing
   the rewrite rather than executable code.
7. The B5 scanner watched detection-time facets while the split it guards
   happens at reconcile — it would have missed precisely the facets it exists to
   guard.
8. `run_robustness_deterministic --all-books` re-invokes itself per book and
   forwarded only `--book`, so a variant run measured the default twice and
   reported it as the variant.
9. `answer_diff` had the same defect on `--only-book`, so a variant run moved
   the 252 in-process calibration records and left the 88 subprocess robustness
   records on the default — which reads as "the variant only affects the
   calibration bank", a conclusion about the product drawn from a defect in the
   instrument.

Three properties separate the instances that were caught from the ones that
nearly were not, and they are the rule:

* **An instrument must be able to produce the failure it rules out.** Every
  instrument ships with a case proving it fails. Where the corpus cannot
  exercise the construct, generate it (Note 2).
* **An instrument must be read at the point the code it measures runs.** Both
  the B5 scanner and the two forwarding defects were instruments reading a
  different moment, or a different process, from the one under test.
* **Every argument that changes what is measured must reach every process that
  measures.** A runner that fans out to subprocesses and forwards a subset
  cannot report that it did not measure what it was asked to.

Corollary, from instance 9: when a measurement splits cleanly along the seam of
the instrument's own plumbing — one surface moves, the other does not, and the
seam is a process boundary or a call site rather than anything in the product —
suspect the instrument first.

Instances 10 and 11, both from the stamping coverage inventory and both caught
by its own `--self-test` rather than by reading its output: a one-size evidence
bundle reported eleven false holes, and a malformed analytical envelope reported
a false hole on the very cell that distinguishes the two reconcilers. The rule
they add: **an instrument that classifies must be able to produce every class it
reports.** A hole-finder that can only produce holes has not found any.

**Companion rule, from instances 10 and 11: an instrument that CLASSIFIES must
be able to produce every class it reports.** A hole-finder that can only produce
holes has found none. The coverage matrix reports four cell values and its
`--self-test` asserts each is producible, including the one it must not confuse
with a defect: a route that correctly did not do the thing asked of it.

**And its converse, from closing the hole: an instrument must not be anchored to
the defect it was built to find.** The first self-test asserted
`(point-in-time)/row_population` reads as a hole. That was true when written and
started failing the moment the hole was closed — the right outcome from the
wrong assertion, because a test tied to a bug stops asserting anything once the
bug is fixed and silently becomes a test of nothing. Re-anchored on a DESIGNED
hole, which will still be there, plus a separate assertion that the fixed one
stays fixed.

Instance 12 is of a different and worse kind, from the same inventory: the two
standing measurement surfaces were BOTH clean throughout a live shipped
regression — three ordinary questions about the front and back book refusing on
the shipped tape. Neither surface was defective; between them they simply do not
exercise the point-in-time population path. Recorded as backlog B6. The rule:
a clean surface is evidence about the surface's coverage before it is evidence
about the product.

## Recorded as implemented and inert — the unresolved-role clarification

Stated plainly so nobody later quotes the three-variant measurement as evidence
of a behavioural gain, because it is not.

**Implemented** (`1f8078d`). A dimension no source gave a role to becomes a
question rather than an answer over a set the reader may not have asked for. The
principle is right: a refusal and a clarification both decline to answer, and
only one hands the reader the next move.

**Currently inert on these corpora.** 343 of 343 answers identical; robustness
44/44 on both books; zero new test failures. It classifies 10 facets across 9
questions on the real tape and the clarification wins on five of them through
the point-in-time workflow — but on the shipped service path all five are
claimed by `risk_limits` or an evolution route and never reach
`reconcile_facets`.

**The measurement that chose it no longer applies.** §2.2 of
`docs/mi_stage4_unresolved_role_variants.md` recorded clarify converting three
refusals into three questions. Two rules added while applying it — a
clarification is only worth asking when answering it changes something, and a
field the book cannot express has no role worth settling — remove all three,
because all three are `borrower_type` on a tape that does not carry it.

Anyone citing this work should cite it as: correct, shipped, and not yet
observable on any available book. Not as an improvement measured on the
surfaces.

## The three measurement surfaces, and what each is blind to

Recorded together because the recurring failure of this programme has not been a
wrong surface. It has been an undocumented blindness, discovered by a defect
walking through it.

| surface | entry point | sees | blind to |
|---|---|---|---|
| calibration bank | `run_mi_agent_query` | numeric correctness, the parse, 255 questions | **routing — it is always point-in-time** (B7). Four of six Stage 5 questions behave differently in production. |
| robustness bank | `/mi/query` | routing, two books, a frozen grader | *which* route answered, and what the receipt claimed. Grades one outcome label. |
| routed surface | `execute_governed_mi_query` | the routing decision, the verdict, the facet kinds and statuses | numbers, the LLM arm, kestrelmoor, answer text, and anything a route declines to declare |
| answer diff | both banks | every answer, byte for byte | anything not in either bank — which is where three stages' worth of change has landed |

The rule that follows: **a clean surface is evidence about that surface's
coverage before it is evidence about the product.** Quote none of them without
its blind column.

## Standing rule — a comment stating an invariant is not evidence it holds

Three instances, so it is a pattern rather than three accidents. In each the
comment was accurate about intent and wrong about behaviour, and in each the
gap survived review because the comment read as a description.

1. The region resolver's docstring claimed a `None` return made validation fail
   clearly. It did not.
2. A P1A test's assertion documented the very defect it was taken to guard
   against.
3. The comment delegating the seasoning ROLE decision to the analytical intent
   layer — "the only place that context exists". That layer runs only when a
   plan is composite, so for every simple question the decision it claims to own
   is never taken. B13 is the whole of that gap.

The rule: **where a comment asserts an invariant, the assertion belongs in a
test.** Until it is in one it is a hypothesis about the code, not a description
of it — and it is worse than no comment, because it stops the next reader
looking.

## Standing rule — a scoping query built around one reader cannot bound a change that replaces the arrangement of readers

From the seasoning consolidation (`7c46f81`). The pre-registered prediction
defined group A correctly — *a single lending window named, no segment phrase* —
and then listed **Q1.1** as its member. Q1.2, Q1.3 and Q1.4 are also group A and
were not listed.

The cause is the rule. That list came from a scan of **where reader 2 raises a
seasoning dimension**, and those three questions do not raise one. The change
was owner-shaped; the scoping query was reader-2-shaped, so it enumerated the
wrong population of questions.

Had the LIST been treated as the prediction rather than the DEFINITION, three
correct movements would have read as stop-condition violations and the work
would have halted on a false alarm.

**When a change replaces how readers are arranged, scope it from the DECISION,
not from any one reader's view of it.** And when a prediction states a class and
then enumerates members, the class is the prediction and the enumeration is
illustration — say which is which before measuring.

## Standing rule — consolidating a decision can create a new reader of it

The same commit, and the sharper of the two. Making one owner decide the
seasoning role meant raising the population facet from the owner's answer. On
the routed path `mi_service` already raised that population from `spec.filters`
— which the owner had just written. **The same decision arrived twice, from two
places, and the receipt stamped one applied and left the other lost, so
`geo_exposure` and the movement path refused themselves.**

Live for about ten minutes, and it is exactly the failure the consolidation
exists to prevent, occurring **inside** the consolidation. The diagnosis had even
recorded it as constraint 5 — *do not duplicate what already resolves correctly,
or it becomes a fourth reader* — and the implementation created one anyway.

**Before a consolidation lands, enumerate every place the owner's answer now
arrives, and check none of them was already deriving it.** A consolidation adds
a producer; the readers it was meant to replace do not stop reading merely
because a new one is better.

## Standing rule — a reader that defaults is a reader

Earned in D2. The census counted three owners of "is this named dimension an
axis or a filter" and there were four. The one it missed was
`requested_dimension_terms`, which **decides nothing**: it raises every named
dimension as a `grouping_dimension` and moves on. That is not an abstention —
on the routed path, where the reader that DOES decide never runs, it is the
operative decision, and it asserted "axis" for every dimension any question
named.

A census that enumerates deciders will miss it every time, because it looks like
plumbing. **When counting owners of a decision, count the places that produce an
answer, not the places that deliberate.** A default is an answer.

## Standing rule — a consolidation that moves nothing is a result, but only with the measurement that says why

Also D2. The commit gave the decision one owner, both paths consumed it, and
**nothing moved on any of the four surfaces** — because the branch it
consolidates is unreachable through any well-formed question the deterministic
arm can ask. That is not a null result and it is not a clean bill of health; it
is the removal of a way for two readers to drift apart, plus a measurement of
which part of the decision the corpora cannot see.

The rule follows from the older one — *a clean surface is evidence about
coverage before it is evidence about the product* — and sharpens it for
consolidations specifically: **a consolidation whose surfaces do not move must
say which construct they could not reach, and carry constructed coverage for
it.** Otherwise "nothing moved" reads later as "it was already right".

## Standing rule — a branch that fires zero times is unmeasured, not unused

Earned twice in one commit, in opposite directions.

**The FILTER branch.** D2 reported the role owner's population branch as
*"unreachable through a well-formed question on this arm"* and carried it on a
test asserting that unreachability. It is reachable: `MiQueryRequest.filters` —
the drill-through API a UI uses to drill from a breakdown into one of its groups
— merges caller-supplied filters into `spec.filters` on both paths before the
guard runs. `"balance by region"` with `{"collateral_geography": "South East"}`
stamps `row_population applied`. The measurement had looked only at question
TEXT.

**The name-match rung.** D7 measured that the routed grouping ladder's
name-match rung fired **zero times across 593 questions**, and the first cut
removed it along with the residual rung below it. Two tests failed, correctly: a
result column named by registry field is execution naming the field, and it is
the same evidence the point-in-time path has always accepted. The corpus said one
rung; the tests said two; the tests were right.

**So: a zero count is a fact about the corpus, not about the code.** Before
removing a branch because nothing reaches it, find the caller that does — another
API, another arm, another entry point — and either exercise it or state plainly
which paths were searched. And never carry a branch on a test asserting its own
unreachability: that is the position that produced `e35a01b`, where a facet
reached a reconciler with no case to receive it and nobody could say whether that
mattered.

## Standing rule — a declared expected-to-fail is only as good as its stated correct outcome

`rt_013` was added in D2 as a declared expected-to-fail against *"Are any
regional limits breached?"*, on the belief that `risk_limits` has no dimension
axis at all. Reading the limit-test rows showed the tests ARE per region, so the
geography certification was TRUE and the declared correct outcome — no facet —
was wrong.

Had D7 not re-read those rows, the fix would have been reported as leaving a
known defect open, or worse, the case would have been "satisfied" by breaking a
correct claim. **A wrong expectation turns a real fix into a reported regression,
and a real regression into a reported fix.** Before declaring one, read what the
route actually publishes; state the correct outcome from evidence, not from the
diagnosis that prompted the case.

## Standing rule — the corpora are built from registry names and miss plain phrasings

Four instances, and the fourth arrived unprompted while measuring D6.

`balance where account status is active` is about as ordinary as a management
question gets. Across 693 questions there are **fifteen** mentions of a named
dimension behind a selector preposition and **not one** is that construction; all
four cases proving the defect had to be constructed. The seasoning whole-book
answer and the front-book regression were hidden the same way — the family
enumerates the SEGMENT names, so the months-on-book windows were unreachable.

D6 is the clearest statement of the mechanism. The 27 forecast questions are
*forecast balance by region · by broker · by LTV bucket · by completion month*
— built from the twelve fields the forecast projection carries. **A family
enumerated from a projection cannot exercise that projection's gap.** The corpus
and the defect share an author's assumption.

> **The corpora are an instrument for REGRESSION, not for DISCOVERY.** A count of
> "N of 693 affected" is a lower bound, and where the construct is a plain
> phrasing rather than a registry term it may be a lower bound of zero against a
> live defect.

Recorded in the due diligence pack as a limitation on every corpus-derived figure
there, not only here.

## Standing rule — before adding a producer, check whether the answer is already published and merely unread

Five instances now, and the last two were found in the same commit:

1. **Stage 5** — the time grain was read from the question and dropped before
   anything could act on it.
2. **Stage 2** — the parser's filter offsets were computed and discarded, never
   reaching the spec. Still open as B0.
3. **D7** — `concentration_analysis` published `workflow.dimension_results`, the
   exact canonical field keys it grouped by, on every envelope. The receipt
   reader ignored it and guessed from a display column called `category`.
4. **B16a** — the whole lost-narrowing mechanism existed, restricted to geography
   by one `if` in `geographic_values`: a value allowlist built from the book, a
   detector, a blocking facet kind, branches in both reconcilers. A variant the
   rule did not name — not *produced and unread*, but **built for one field when
   it generalises to all**.
5. **B16a again** — `portfolio_lens` publishes `mask_scope_phrases` and states
   the doctrine ("claim the span first"), and the filter and dimension parsers
   already call it. The new owner did not, and refused eleven scope questions
   until it did.

6. **B21/B22** — the sharpest of the six. `portfolio_lens` states the doctrine
   ("Only QUALIFIED phrases count. A bare 'current' or 'entire' is ordinary
   English") and implements it as `scope_phrase_spans`, over a vocabulary that
   **already contains `direct`, `acquired`, `purchased` and `funded`** — the
   exact words of both defective decisions. The function is called by the filter
   and dimension parsers, to protect them FROM the scope vocabulary. It was never
   turned on the two decisions that own that vocabulary.

In every one the fix was **a reader, not a producer**, and in every one the
instinct was to add a producer. The sixth is the variant to remember: not
produced-and-unread, not built-for-one-field, but **written as a doctrine,
implemented as a function, and applied to everyone except the decision it was
written for.**

> **Before adding a producer, check whether the answer is already being published
> and merely unread — and record the result of the check, negative or positive.**

Recording the negative matters as much as the positive. B16a's pre-registration
performed this check for the categorical selector MARK and reported back that
nothing produces it, which is why that part of the commit adds an owner. Two
paragraphs later the same commit found two things that were already published.
**The check is only worth anything if it is made against each piece of the design
rather than against the design's headline.**

## Standing rule — recording a defect with a test is worth more than fixing it

The clearest evidence for this practice so far, and it is not an argument — it is
a sequence of events in D8.

`7c46f81` hit a duplicate-population defect that was live for about ten minutes:
two readers raised the same governed population, one was stamped applied and the
other left lost, and the answer refused itself. It was fixed **and recorded**,
with a test asserting no receipt carries a duplicate claim.

D8 added a new raiser for drill-through populations, and deduped it — **at the
point the drill is raised**. That looked right and was wrong: the role owner
creates its copy later, inside the split, so at dedupe time the second facet did
not exist yet. `"balance by region"` with a drill to South East produced two
identical population facets and the answer refused itself.

**The only thing that caught it was the test written for `7c46f81`.** No surface
moved — the defect is unreachable from question text, because the drill-through
is an API parameter. No review would have found it: the dedupe was present,
looked correct, and was in the wrong place by one function call.

> **A fix protects the instance. A recorded defect protects the class — including
> against the person who recorded it.** The cost of writing the test is paid once;
> it was collected here by a different commit, months of work later, against a
> mistake made by someone who knew the original defect intimately.

The corollary is about where to put the test: it asserted the PROPERTY (no
receipt carries a duplicate claim) rather than the mechanism (the dedupe runs
here). A test written against the mechanism would have passed.

## Standing rule — a declared expectation states the evidence it rests on

Twice in three commits a case was wrong because it assumed a route could not do
something it could.

* **`rt_013`** was declared expected-to-fail on the belief that `risk_limits` has
  no dimension axis. Its tests ARE per region — London 21.1% against 25.0% — so
  the certification it called false was true.
* **`rt_021`** was declared on the belief that `geo_exposure` could not narrow on
  `account_status`. It resolves its frame through the governed population seam
  and narrows on **any** material predicate, reporting `rowsBefore 11,035,
  rowsAfter 11,000`.

Both were assumptions about a route's capability, stated as expectations, with no
record of what they rested on. Both fired as stop conditions and cost a
diagnosis each to unwind.

> **A declared expectation states the evidence it rests on, not only the outcome
> it expects.** "This refuses" is an expectation. "This refuses because the route
> publishes no narrowing ledger — verified in its envelope, which carries
> `populationApplied: null`" is an expectation that can be checked before it
> costs a stop condition.

The discipline is cheap: the evidence is in hand when the case is written,
because that is when the behaviour was measured. Writing it down is what makes
the difference between a case that is wrong and a case that is **visibly** wrong.

## Standing rule — a replaced test case takes a new id

D8. `rt_021`'s premise turned out to be wrong, and it was re-pointed at a
different question **under the same id**. `answer_diff` keys on
`(surface, id)` and duly reported a third movement — a product change where
there had only been a test change.

Given a new id, the differ reported it correctly: one case gone, one arrived.

> **A replaced case takes a new id.** An instrument that keys on identity cannot
> tell a rewritten case from moved behaviour, and the reading it produces is the
> more alarming of the two.

## STANDING RULES — two corrections earned by items 1 and 2

### The parameterisation rule, replaced

The `_qualified_span_re` ruling generalised wrongly. It was read as *"parameterise
the implementation and keep the vocabularies distinct"*, and item 1 needed the
exact opposite. The rule is:

> **Share what is one fact. Separate what is two.**

| | what was ONE fact | what was TWO |
|---|---|---|
| **B22** `_qualified_span_re` | the qualified-mention TEST | the vocabularies — scope nouns are genuinely not lens nouns |
| **item 1** comparators | the VOCABULARY — "is 'bigger than' a comparator" is one fact about English | the implementations — an operator is not a receipt word |

**Hard-coding was the error in both, in opposite directions.** In B22 hard-coding
one noun list dropped five governed phrases; in item 1 keeping two comparator
lists dropped five comparators. Neither case is evidence for a fixed shape; both
are evidence for asking which part is genuinely singular.

### A vocabulary consolidation is not complete when the lists agree

**Fixing a multi-owner decision can expose a hard-coded constraint the old
vocabulary was too small to reach.** Item 1 unified the comparator lists, the
lists agreed on 29 of 30 — and the three target phrasings still failed, because
`_filter_field_of` probed a **fixed twelve characters** for a currency sign and
`"bigger than "` is exactly twelve. Every phrase the old vocabulary held was
short enough for the window. Enlarging the vocabulary made the window the
binding constraint.

That is the **third** instance of a fix revealing something the layer above was
masking. So:

> A vocabulary consolidation is complete not when the lists agree, but when the
> **consumers of the unified list have been exercised across its full range** —
> longest and shortest phrase, every direction, every consumer.

### A phrasing that appears to work while carrying the defect masks the mechanism

Item 2's axis vocabulary had four consumers; two knew only `by`. But `split by`,
`broken down by` and `grouped by` all **passed those two consumers** — not
because they were handled, but because each phrase CONTAINS the word `by`. Only
`per` and `across` failed visibly.

`"balance split by region"` looked correct and was not: the cut landed at offset
14 instead of 8, and `answer_type.subject_side` read the measure as
`"balance split"`. It answered, it answered with the right number, and the word
`split` sat inside the span that may name the measure the whole time. Four
corpus questions were in exactly that state and no surface reported them.

> **A vocabulary gap is not bounded by the cases that fail.** The phrasings that
> appear to work are where it hides, and they hide it best when they overlap the
> vocabulary that is present. Enumerate the whole vocabulary against every
> consumer; do not infer the gap's size from the failures you can see.

This is why item 2's prediction of "no lexical decision moves" was wrong: it
assumed the consolidation could only touch `per` and `across`, the two markers
that visibly failed. Four decisions moved, all corrections, all in phrasings that
had been passing.

### `test_the_detectors_stay_separate` guards a property nobody would notice losing

Kept prominent deliberately. `execution_receipt._detect_thresholds` reads the
SENTENCE and never the applied filters. Deriving its facet from the filters is
the obvious simplification, it would pass every existing test, and it would
**silently delete** the protection that makes `exceeding`, `in excess of`,
`minimum of`, `beneath`, `up to`, `maximum of` and `capped at` refuse rather than
answer over the whole book. Two independent detectors is the design: a threshold
the parser misses must still be RAISED, so the guard has something it cannot
honour. The test asserts `_detect_thresholds` takes the question and nothing
else, which is the only mechanical form that claim has.

## Backlog

### The analytical path publishes no population — scope stated, not opened

The extended grader's figure check compares the population an answer covers
against the narrowing the sentence states. It is **vacuous across all 44**, and
for two reasons rather than one:

* none of the 44 states a row-level threshold, so the check never fires; and
* **the analytical route emits no `executionSummary.population` at all**, so on
  that path there is nothing to compare against even when a question does state
  one. The check is unexerciseable there, not merely unexercised.

**What closing it would take:** the analytical composition layer emitting, per
composed capability, the population it measured and the whole-book total it
measured against — the same two numbers the point-in-time path already
publishes. A receipt change on the analytical path, not a grader change.

**Not now.** Recorded so the size is known before anyone decides.

### "compare" read as a measure — the same shape, a seventh time

*"How does the front book compare with the back book?"* is **refused**, while
*"Compare direct and acquired balances"* answers correctly. Found while writing
the capability summary.

**The mechanism, named:** the word `compare` is resolved as a MEASURE. The
refusal is the measure-substitution guard doing its job — *"'compare' is not a
governed measure in this dataset; no substitute was used"* — on a word that is
not naming a measure at all. It is the verb of the sentence.

That is the same shape as the four closed this week: **a word outside the measure
position taken as the measure.** Item 2 fixed two instances of it (a field word
in an axis clause, and the drill-through branch overriding a named measure) and
recorded that `_detect_metric` masks nothing and depends on its caller
pre-cutting. This is a third instance, with the additional wrinkle that the word
is not a field word at all.

Note the corpus does not catch it: the robustness bank's front-versus-back family
scores 4 of 4 correct on both books, because none of its four phrasings uses the
bare verb "compare" against two named populations. **A phrasing that appears to
work while another carries the defect** — the `split by` rule, again.

**Recorded, not fixed.**

### Item 1 follow-ons — recorded during the fix, deliberately not opened inside it

* **The bare `N+` threshold is applied but never disclosed.** *"how many
  borrowers are 70+"* narrows correctly (`Borrower Age >= 70`, 6,862 loans) and
  raises no threshold facet, because the receipt's postfix pattern ends
  `(?:\+|...)\b` and a `\b` after `\+` can never match. **A branch that cannot
  fire** — the same class as the dead guard found in B16a. Belongs to B20's
  mutation pass over the guard set. A disclosure gap, not a wrong number.
* **Subject binding without a currency marker.** *"loans no more than 150000"*
  binds the threshold to `current_loan_to_value`, not the balance, because with
  no `£` the nearest-subject rule picks the measure named earlier in the
  sentence. With a currency sign it is correct. `_filter_field_of`'s precedence
  is a genuinely separate decision from the comparator vocabulary.


### B24 — should `resolve_active_view` run before parsing at all?

**Open, recorded as its own item, and deliberately not settled inside B21.**

The view is chosen by a substring test on the raw question before anything has
parsed it. B21 fixes the test; it does not ask whether the decision belongs
there. B21 is now closed and this remains open, narrowed: a DISCLAIMED
word no longer chooses the frame, but an undisclaimed incidental mention still
does — *"how does this compare with the forecast we ran last quarter"* still
loads the forecast projection. B22's qualified-mention doctrine does not transfer
(`forecast` and `pipeline` are nouns naming the subject, not qualifiers naming a
scope), so nothing currently separates the word as SUBJECT from the word as
passing reference. That distinction is this item's territory. That is a larger question — it touches when the frame is loaded, what a
route may assume about it, and whether a view is a property of the question or of
the workspace — and settling it inside a fix to the test would decide it by
accident.

### B21 — the view resolver  ·  **CLOSED**, before real client data arrives

Higher severity than B22 despite changing no number on this book, and the reason
is a property of the book rather than the code: `build_forecast_view_frame` puts
the forecast CONTRIBUTION into `current_outstanding_balance` — same column name,
different meaning — and with no pipeline data the two coincide exactly at
£1,964,886,258.21. **On any book carrying a pipeline they diverge, and it becomes
a wrong figure under the same field name, disclosed nowhere.** A defect waiting
for a live portfolio rather than one this book exposes.

**Closed. It took FOUR owners, not the one the diagnosis named.** Full record in
`mi_b21_disclaimed_view_prediction.md`.

| # | owner | decides | found by |
|---|---|---|---|
| 1 | `workspace.resolve_active_view` | the FRAME | the diagnosis |
| 2 | `chat_routing._dataset_for` | the DATASET | the site-by-site enumeration, before any test failed |
| 3 | `mi_workflows.analytical.intent` | the structural REQUIREMENT | rt_030, after 1 and 2 |
| 4 | `execution_receipt._PROJECTION_RE` | the requested-projection FACET | rt_028, after 1 and 2 |

With owners 1 and 2 fixed, *"the balance by vintage, ignoring the forecast"*
computed **the right number over the right 11,035 loans** — Balance by Vintage,
13 groups — and was then **refused**, because owner 4 still raised a
forward-projection facet from a forecast word the sentence had ruled out. The
honour-or-clarify guard was working exactly as designed on a request never made.
A fix measured by "does the number come out right" would have stopped two owners
early.

What was consolidated is the WINDOW, not the vocabulary: `is_disclaimed_span`
measures one distance and stops at one sentence boundary, and all four owners
plus B22's scope resolver read it. Each keeps its own vocabulary, because they
are genuinely different vocabularies — owner 2's includes `case`, `kfi`,
`application`, `offer`, none of which owner 1 knows about.

**The severity is constructed and should be read that way.** On this tape the
divergence is £0.00; on a book with three constructed pipeline cases it is
£770,000, arriving under `current_outstanding_balance` — the funded balance's own
field name — and disclosed nowhere. 727 of 729 answers identical means the fix
did not reach the corpora and nothing more: 0 of 683 corpus questions change
view, and 1 of 661 × 14 vocabulary pairs stops signalling, both of them cases
this work constructed.

### B22 — the lens resolver  ·  **CLOSED**

Closed by the qualified-mention test and the disclaiming guard. Full record in
`mi_b22_qualified_mention_prediction.md`.

**Diagnosed together in `mi_b21_b22_b23_diagnosis.md`; reported in
`mi_view_selection_report.md`. B23 collapses into B21 — it is that decision's
missing disclosure half, not a third defect.**

**Ranked ahead of D10, D9, D14 and the segmented series, and ahead of B9 and
B10.** They change the number in the shipped product; the census entries left are
receipt defects and agree-by-maintenance entries where nothing is currently
wrong. Within the pair, measurement puts **B22 first**: it is the only one
changing a number on this book — 3,909 of 11,035 rows for "loans purchased at
auction" — and its fix is a call to `scope_phrase_spans`, a function in the same
module that discriminates every case correctly and that `resolve_lens` does not
call.

Asked alongside D8: is `resolve_active_view` one instance or one of several?
**One of several — there are two, and both are substring tests.**

* **B21** — `workspace.resolve_active_view` is a bare substring test that loads
  the forecast frame whenever the word "forecast" appears anywhere, dropping 60
  of the book's 71 columns. *"What is the balance by vintage, ignoring the
  forecast?"* answers `vintage_year: field_missing`: **the clause saying to
  ignore the forecast is what causes the forecast frame to be loaded.** D6 did
  not fix this — D6 made the RECEIPT honest, and this fails earlier, at
  prepared-data validation.
* **B22** — `portfolio_lens.resolve_lens` has the same shape on ROWS, with bare
  single-word terms (`direct`, `acquired`, `purchased`, `organic`, `inorganic`).
  *"the balance for loans purchased at auction"* silently answers over the
  acquired cohort.
* **B23** — neither decision is disclosed where a reader would see it.

The row decision is **more dangerous by construction** (it changes the number and
leaves the answer looking complete) and **less harmful by accident** (it is
declared on the receipt and has a comparison guard). That is accidental safety
for the fourth time.

All three are changes to what data is put in front of the question — upstream of
every decision this programme consolidates — so none was folded into D8.

### B20 — a mutation pass over the guard set

**Scoped, not scheduled. Do not run it as part of another commit.**

B16a's `selector_mark` was written with a runtime axis check so that "balance by
broker" could not read as a narrowing. Mutating it away **changed no outcome and
failed no test**: the selector and axis vocabularies are disjoint, so the branch
could never fire. It was removed and the invariant asserted in its place.

**The point is not that one guard was dead. It is that a dead guard is a
DISTINCT class of instrument defect, and this programme has been recording a
different one.**

* *An instrument carries the defect it was built to find* — 16 instances — is
  about an instrument being WRONG.
* A guard whose removal changes nothing is about an instrument that **looks like
  coverage and is not**. It was providing false assurance for its whole life, and
  every review that read it as protection was misled. Nothing fails when it is
  deleted, so nothing ever surfaces it.

The two are found by different means. The first is found by measuring the
instrument against reality. The second is found ONLY by mutation: remove the
guard, run everything, and see whether anything notices.

**The work:** enumerate the guard set — every conditional whose stated purpose is
to prevent a wrong outcome rather than to compute one — remove each in turn, run
the four banks and the test suite, and list every guard whose removal is silent.
For each: either find the path that reaches it, or delete it and assert the
invariant that makes it unnecessary.

Related to, and distinct from, the D7 standing rule *a branch that fires zero
times is unmeasured, not unused*: that rule is about branches found by
measurement, this is about finding them systematically.

### B18 — "limit status" resolves to the registry field `account_status`

**A TERM RESOLUTION defect. Not a role defect and not an evidence defect.**

*"Show concentration limit status."* names no dimension. `requested_dimension_terms`
matches the word *status* to `account_status`, no source gives it a role, and
`risk_limits` runs no test covering it — so after D7 the unproven grouping blocks
and a correct answer (*"8 passed, 0 warnings, 1 breach"*) refuses.

Fail-closed, wrong reason, no wrong number — the posture recorded for B1. Pinned
as `rt_013`, a declared expected-to-fail stating the correct outcome: the answer
standing, with no breakdown claim attached.

### B17 — a drill-through on a routed question always refuses

**Belongs to D8.** `material_predicates` is computed from `parsed.spec.filters`
**before** `try_route` calls `parsed.merge_filters(extra_filters)`, so a
caller-supplied drill-through narrowing never reaches the frame but IS on the
spec by the time the guard reads it. The population is stamped LOST and the
answer refuses.

Fail-closed and correct in outcome — no whole-book figure is passed off as the
narrow one — and wrong in cause. The same API works on the point-in-time path,
which is how it was found: proving D2's FILTER branch reachable. Pinned as
`rt_016`.

### B16b — resolve the categorical filter the sentence marks

**B16a is CLOSED.** A narrowing the sentence marked and execution did not apply
is now recorded and refuses rather than answering over the whole book. What
remains is resolving the value so the answer NARROWS instead of refusing.

**B1 and D6 are both hard prerequisites**, in that order: resolving more
categorical filters through a denylist multiplies the fabricated-binding class
B1 exists to retire, and D6 governs whether the resolved field is judged
available in the book being asked about.

Original entry follows, kept because it is the diagnosis.

### B16 — the sentence marks a selector and nobody reads the mark

**Found while working D2. A parser change, not a facet change.**

`question_interpretation.lexical.is_filter_subject` is the declared lexical
owner of *"this mention is a selector rather than an axis"*. **Nothing reads
it.** `dimension_role` consults `grouping_cut` (the axis half) and deliberately
does not consult this one, because a facet reclassified to a population must
carry a **resolved predicate** — `_analytical_population_satisfies` recovers the
value by splitting the label on the field name, and a valueless label accepts
any predicate naming that field (B5). The parser is the only source of a
resolved value, so the facet layer cannot act on the mark without inventing one,
which would be a twelfth interpreter.

Measured: across 693 questions the intersection of "field the parser slotted as
a filter" and "dimension the detector named" is **empty**. Four of the five
fields that ever appear in `spec.filters` are numeric bounds on measures; the
fifth, `collateral_geography`, binds only where the dimension word is absent.

So *"balance by region where account status is active"* has its narrowing
asserted as a **breakdown** on the routed path and asked about — *"did you want
the book split by it, or narrowed to one value of it?"* — on the point-in-time
one, for a sentence that marked the role unambiguously.

**Measured on the shipped service path in D7, it is a WRONG NUMBER, not a wrong
receipt:**

```
"What is the balance where account status is active?"
  -> ok.  "Total Balance · grouped by Account Status · 2 groups · 11,035 loans"
          facets: grouping_dimension account_status APPLIED
```

The reader asked for the balance of ACTIVE loans and was given the whole book
split by status, certified. That is **B13's class**, and B13 was treated as the
defect it is.

**Order: immediately after D7, ahead of D6, D8, D10, D9 and D14** — it is the
only remaining item producing a wrong number rather than a wrong receipt, and
its landing zone is already built and tested. It splits:

* **B16a — the facet layer, no parser change.** A narrowing the sentence marked
  and execution did not apply is recorded as a request that was lost;
  honour-or-clarify then refuses instead of answering over the whole book. Needs
  no predicate value, so it invents nothing. **This is the step that removes the
  wrong number, and it is safe ahead of D6.** Pinned as `rt_015`.
* **B16b — the parser resolves the filter**, turning the refusal into an answer.
  **B1 is a hard prerequisite**, and **D6 is too**: D6 governs whether a field is
  judged available in the book being asked about, and resolving more filters
  before that is fixed would multiply B14's wrong message.

**The fix is in the parser**: resolve the categorical filter the sentence marks,
and the role owner's existing source 1 answers correctly on both paths with no
further change. B1 is the neighbouring half — the one categorical binding the
parser does perform is validated against a denylist and fabricates values.

### B1 — Route the categorical filter regex through the profiled allowlist

**Scoped, not scheduled. Not part of this programme.**

`llm_query_parser._CATEGORICAL_FILTER_RE` (`llm_query_parser.py:1820`) validates
an extracted geography value against two **denylists**,
`_CATEGORICAL_STOPWORDS` and `_NON_PLACE_TERMS`, accepting anything not listed.
`execution_receipt.geographic_values` already holds the **allowlist** — the
values the loaded book actually contains, 11 tokens on the alderbridge tape.

**The change:** route the regex's operand through the profiled allowlist instead
of the denylists.

**Why it is worth doing once rather than patching again:** the denylist has
already been patched twice, each time after a defect — *"when is it expected to
complete"* binding a geography called **Complete**, and *"for joint borrowers"*
binding a borrower predicate to the geography field. A denylist cannot be
completed. Routing through the allowlist retires the fabricated-geography class
permanently rather than removing its third instance.

**Why it is not urgent:** the surviving cases **fail closed**. *how much is in
the good book* binds `geographic_region_obligor='Good'`, matches zero rows, and
returns *"No loans in this book match that filter … I have not returned a
whole-book figure in its place."* Wrong reason, correct refusal, no wrong
number.

Fuller analysis, including what tape normalisation would additionally require:
`docs/mi_value_domain_prerequisite.md`.

### B0 — wire the parser's filter spans onto the spec, completing the join

**Ready, not started.** `_parse_filters` now returns `{field_key: (start, end)}`
through an optional `spans` sink, but nothing carries it onto `MIQuerySpec`, so
the object still reports the join as half-built.

The obstacle I expected is not there. `_mask_spans` — the other place the parser
appears to rewrite the question — **blanks characters in place rather than
deleting them**, and its docstring says so: *"Blanking rather than deleting keeps
every other offset valid"*. So offsets taken from the masked remainder are valid
offsets into the original question, and all four `_parse_filters` call sites can
supply sound spans.

What remains is mechanical rather than risky: an additive `filter_spans` field
on `MIQuerySpec`, excluded from `referenced_fields()` and validation as the
other non-semantic fields are, and a sink threaded at four call sites. It is its
own commit with its own before/after.

### B4 — `mi_agent/interpreter/deterministic.interpret` duplicates a serving concern

**Recorded, not now.** The package is a development smoke tool — imported only
by `scripts/mi_nlq_dev_smoke.py`, `scripts/phase8e_live_anthropic_smoke.py` and
its own modules — but it carries a second whole parser for a concern the serving
path also implements.

Dev-only code that duplicates a serving-path concern will drift, and the drift
is invisible because nothing measures it. A future reader finding two parsers
has no way to know which one ships. Either it consumes the same owner as the
serving path, or it is deleted, or it carries a header saying plainly that it is
not the parser and must not be read as one.

### B5 — the literal population comparison is permissive when a label omits its field

**Found during Stage 4, pre-existing, recorded rather than fixed.**

`_analytical_population_satisfies` derives the value it wants by splitting the
facet's label on its field name. Where the label does not contain the field
name, the value comes out empty and the check accepts **any** predicate naming
that field — including the wrong population. A facet for *front book* is
accepted against a declared `seasoning_segment = Back Book`.

Verified present before the Stage 4 change by stashing it. Not reachable with
the labels the receipt builds today, which embed the field and the value, and
the governed comparison added in Stage 4 is stricter rather than looser.

Not fixed here because it would change acceptance for populations outside the
seasoning family, and the pre-registered prediction for Stage 4 says to report
such a movement rather than absorb it.

### B2 — `answer_type.asked` disagrees with the parser on 46 questions

**Recorded, not to be fixed in Stage 2.** No user-visible difference: `asked()`
is on no production path. Analysis:
`docs/mi_question_interpretation_stage2_readiness.md`.

### B3 — `of_measure` cannot distinguish one measure from several

**Open, blocking nothing, and it weakens `mixed` as an acceptance type.**
`of_measure` types an answer from a single `metric` + `aggregation`. A portfolio
summary carries `metric=None, aggregation='count'` and types as `count`, which
`_SATISFIES[MIXED]` accepts — so four calibration cases declaring `mixed` would
pass identically if the answer lost every measure but the count.

Found by the bounded pre-Stage-2 check:
`docs/mi_answer_type_expectation_check.md`.

**The right shape of the fix, recorded while it is fresh.**

The defect is *four cases asserting a property nothing verifies*. So the fix is
to **verify the property**: assert the required measures on `kpi_028`,
`kpi_029`, `kpi_030` and `kpi_031` directly — that the result carries both
`loan_count` and a balance measure — rather than inferring it from a type.

That is a **test-side addition**. No production change, no effect on the
baseline, and it makes the four cases detect the regression they describe.

**Two fixes that look adjacent and are wrong:**

* **Changing `of_measure`** so it reads the result's measure set rather than a
  single spec slot. It is a grader. Changing it mid-programme means before and
  after are scored by different instruments, and every measurement taken
  earlier stops being comparable.
* **Narrowing `_SATISFIES[MIXED]`** so `count` no longer satisfies `mixed`. Same
  objection, and it would likely fail those four cases immediately — which is
  Finding 1 becoming visible rather than being fixed. The property would still
  be unverified; only the symptom would move.

The distinction is that asserting the measures **adds a check**, while both
alternatives **move an existing one**.

### Recorded as working — the derivation cross-check

56 of 252 stored `expected_answer_type` values differ from `answer_type.asked()`
today, and **none of them is drift**. 33 of the 35 `currency`-versus-`any` cases
carry an `expected_metric` that justifies `currency` — 27
`current_outstanding_balance`, 5 `current_valuation_amount`, 1
`original_principal_balance` — and the 21 `none` cases are authored from
`expected_status` rather than from the wording.

That is `derive_answer_type.py`'s documented cross-check behaving exactly as
designed: *"the question's own wording decides, cross-checked against the
declared expected_metric"*. Recorded as a control that works, not dropped
quietly — a mechanism only known to be sound if someone has checked it and said
so.

### Standing rule — do not regenerate the calibration bank

All 252 `expected_answer_type` values were derived from `answer_type.asked()`.
Regenerating the bank during this programme would rewrite those expectations
from a classifier the programme is changing, and a bank that moves with the code
it grades has stopped being a control.
