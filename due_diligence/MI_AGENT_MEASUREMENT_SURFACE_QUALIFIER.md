# Measurement surfaces — what each one is blind to

For the due diligence pack. Anyone quoting a clean measurement from this
programme should read this first.

## The rule

**A clean surface is evidence about that surface's coverage before it is
evidence about the product.**

It was earned, not assumed. Three separate stages of work ended with "both
standing surfaces are unmoved", and in each case the reason was the same: the
surfaces did not exercise the path the change was on. Twice a live defect sat in
front of ordinary questions while every measurement read clean —

* `e35a01b` made *"What is the balance of the front book?"* refuse, on the
  shipped tape through the shipped entry point, for a full commit. Both surfaces
  were clean throughout. Eleven tests were repaired when it was fixed.
* A routed grouping was stamped APPLIED whenever it could not be DISPROVED, so
  *"balance by month by region"* returned the whole-book series with the receipt
  vouching for a regional breakdown that was never computed. Both surfaces clean.

Neither surface was wrong. Both were blind, and the blindness was undocumented
until a defect walked through it.

## The four surfaces, and their blind spots

| surface | entry point | sees | blind to |
|---|---|---|---|
| calibration bank (255) | `run_mi_agent_query` | numeric correctness, the parse, artifact shape | **routing — it is always point-in-time.** Four of six time-axis questions behave differently in production. |
| robustness bank (44 x 2 books) | `/mi/query` | routing, two books, a frozen grader | *which* route answered, and what the receipt claimed. Grades one outcome label. |
| routed surface (18) | `execute_governed_mi_query` | the routing decision, the verdict, the facet kinds and statuses; the drill-through API, via a case's `filters` | numbers, the LLM arm, kestrelmoor, answer text, and anything a route declines to declare |
| answer diff (693) | both banks **and the governed service path** | every answer, byte for byte | the LLM arm; and the calibration corpus is recorded POINT-IN-TIME, so a routed-only change to one of its questions is still invisible — one such movement is known and unreported by any instrument |

## What follows for a reader of these numbers

1. **Quote no surface without its blind column.** "343 of 343 identical" is a
   true statement about 343 answers, not about the product.
2. **A change measuring as inert is a claim to check, not a result to accept.**
   Establish that the questions the change affects are ON the surface before
   concluding the change did nothing.
3. **"No surface reached it" and "it cannot be reached" are different
   statements.** Conflating them hid a live defect for a commit. Reachability is
   settled by construction, never by inspection.
4. **Every arm here is deterministic.** The LLM arm's recorded figures cannot be
   reproduced in this environment, and no number in this programme is an LLM-arm
   number.

---

## The differ's fourth surface, and why it was added

D7 moved five real answers on the shipped path and the differ reported **343 of
343 identical**. Not a bug — a corpus gap, and the same one twice over:

* its calibration half calls `run_mi_agent_query` directly and is therefore
  always **point-in-time** (B7), so a routed-only change cannot show there;
* its robustness half does route, but is 44 sentences.

**All three standing surfaces were blind to a change that moved five answers.**
The routed surface saw the mechanism, in eighteen hand-picked cases; no
instrument saw the answers.

`answer_diff` now carries a fourth surface — `service_path`, the 350-question
`ere_mi_questions` corpus driven through `execute_governed_mi_query`, the entry
point a user reaches. Recorded at 693 and demonstrated to fail: against the
pre-D7 baseline it reports 689 identical and 4 moved.

**One residual blind spot, stated rather than closed.** `risk_216`
(*"concentration by account status"*) lives in the calibration corpus, whose
questions are never driven through the service. Its movement was measured
directly and no standing instrument reports it. Closing that means driving the
calibration corpus through the service as well — a fifth surface, not attempted
here.

---

## WHAT THE 91.0% MEASURED — a qualifier for every figure derived from the robustness bank

**The robustness bank was always on the shipped path. What was wrong was the
grading.**

`/mi/query` (`mi_agent_api/app.py:1737`) calls `execute_governed_mi_query`, the
same entry point every other surface uses. The 44 variations in `nl_bank.py` were
the right questions, asked the right way, of the right code. The defect was in
`nl_score.grade`, and it had two halves.

### It could not compare a figure to the book

Proved by mutation, not by argument. Take a bank question the grader calls
CORRECT and treble every figure in its answer:

```
Q1.1  How has the profile of our new lending changed over the last few months?

  original -> CORRECT
     New lending (last 1 month): 143 loans at 2026-04-30 against 115 loans at
     2026-06-30. Current Outstanding Balance £21.4m against £18.3m; ...

  EVERY FIGURE TREBLED -> CORRECT
     New lending (last 3.00 month): 429.00 loans at 6,078.00-12.00-90.00 against
     345.00 loans at 6,078.00-18.00-90.00. Current Outstanding Balance £64.20m ...
```

Three of three mutated answers graded **CORRECT, unchanged** — loan counts
trebled, dates mangled into `6,078.00-12.00-90.00`.

### THE SAME RESULT FROM THE OPPOSITE GRADER — matching a haystack is not checking a claim

The first attempt to verify the 44's figures against the book did the opposite of
what `nl_score` did, and reached the identical wrong answer.

It collected every low-cardinality dimension's count and balance into a
**322-value truth set**, then asked whether each answer quotes *at least one*
figure matching any of them within 2%:

```
  20 of 20 answered Q1/Q7/Q8 rows quote at least one figure that matches the book
  CAN-FAIL: 20 of 20 TREBLED answers still match -> the check is WEAK
```

**One grader read no figures. The other read every figure. Both called a trebled
answer correct.**

The reason is the same in both: neither was checking a claim. `nl_score` graded
the route and the plan and never looked at a number; the haystack check looked at
every number and never asked what any of them was *for*. With 322 values and a
2% tolerance, almost any figure finds a neighbour.

> **A tolerance against a large value set is not a check.** A figure is verified
> only when it is bound to the claim it makes — this population, this count,
> this balance.

The result was discarded. It is recorded here rather than deleted because it is
the clearest illustration available of what "verified against the book" must not
mean. What replaced it binds each figure to the population it labels — *"Direct,
7,126 loans: Current Outstanding Balance £1.36bn → £1.39bn"* — and a trebled
answer fails it 8 times out of 8.


The cause was upstream of the grader: `_capture` handed it ten keys and
`executionSummary` was not among them. For the LTV question that returns the
whole book, the service knew `population=11035, filtersApplied=[]` and **the
record dropped both before grading**. It was not a grader that declined to check
the number; it was a grader that was never given one.

### "Safe" meant the refusal sentence was longer than forty characters

```python
if not ok:
    if len(answer) > 40:
        return SAFE_REFUSAL, [], "refused with a stated reason"
```

That was the whole test. A refusal that declined **while holding the answer** —
filter applied, 5,857 loans, correct measure, then a question about how the
reader meant the word "ticket" — graded identically to a refusal that was right
to decline.

### THE QUALIFIER, to be attached to every figure derived from this bank

> **91.0% measured that answers ARRIVED and were SHAPED PLAUSIBLY — that a route
> claimed the question, that the planned capabilities matched the contract, and
> that any refusal was more than forty characters long. It did not measure that
> a single figure was correct, and it could not distinguish a refusal that was
> right to decline from one that declined while holding the answer.**

This applies to the 91.0%/9.0% split in `MI_AGENT_CLIENT_READINESS.md` §6.5, to
the 752-run A/B that reported 0 differences, and to every downstream statement
resting on either.

### What the extension does and does not now verify

The grader now carries the figures and checks **one** of them: the population an
answer covers, and whether a narrowing the sentence states reached it. Under it
the 44 read `CORRECT 32 · UNHELPFUL_REFUSAL 6 · SAFE_REFUSAL 4 · DISCLOSED 2` on
both books — six of the ten refusals were avoidable, and the four that remain
are Q9, where all four phrasings refuse.

### WHICH FIGURE CAME FROM WHICH SCORER — read this before attributing any of them

There are **two** scorers and they are not interchangeable.

| scorer | status | the figures it produced |
|---|---|---|
| `nl_score.py` | **FROZEN**, pinned in the evidence manifest | the baseline/V1 A/B, the **187-to-zero** claim, and the historical **91.0%** |
| `nl_score_extended.py` | extension, added after that comparison | **CORRECT 32 · UNHELPFUL 6 · SAFE 4 · DISCLOSED 2** |

**Never quote a number from one against a comparison run by the other.**

The frozen scorer's guarantee — *"scores baseline and V1 alike"* — is the entire
basis of the A/B: 187-to-zero means something only because **one scorer graded
both sides**. During this programme the frozen file was extended in place. The
figures did not move, but **the guarantee was gone the moment the bytes changed,
whether or not anything moved** — a frozen instrument that has been edited can no
longer certify that two runs were graded alike.

It has been restored to its pinned bytes (sha256 `aade1417…`, verified) and the
extension moved to a separate named scorer. The robustness runner reads the
extended one and says so at the import.

### THE SECOND QUALIFIER, to be attached wherever `CORRECT 32` is quoted

> **`CORRECT 32` means 32 answers arrived correctly SHAPED, of which 8 labelled
> claims are verified against the book.** Q1 and Q7 are unverified. 24 of the 44
> — Q2, Q3, Q4, Q5, Q6 and Q9 — are unverifiable by construction: they ask for a
> forward figure, a run rate or a limit headroom, and there is no expression for
> the right answer to *"when will we reach £100m?"*.

A reader meeting either the 91.0% or the 32 needs both qualifiers, which is why
they sit together here rather than in separate notes.

The 8 verified claims come from the Q8 family, whose answers are structured
enough to bind a figure to the population it labels — *"Direct, 7,126 loans:
Current Outstanding Balance £1.36bn → £1.39bn"*. Each label, count and closing
balance is checked against that population in the book: **8 of 8 exact, and 0 of
8 pass when every figure is trebled.**

### The population check is vacuous across all 44, and what would close it

The check fires only when a question states a threshold. **None of the 44 does.**
Separately and more fundamentally, **the analytical route publishes no
`executionSummary.population` at all**, so even a question that stated one would
have nothing to compare against on that path.

Closing it would mean the analytical composition layer emitting, per composed
capability, the population it measured and the whole-book total it measured
against — the same two numbers the point-in-time path already publishes. That is
a receipt change on the analytical path, not a grader change. **Recorded, not
opened.**

### The same failure mode, found in a surface built to avoid it

The shipped-shapes grader — written specifically to compare figures to the book —
got three of its first seven verdicts wrong, in the product's favour to state
plainly: it marked the product WRONG for being RIGHT.

* It read only the prose when the figure was in the KPI card, so "What are the
  headline numbers?" was called a summary with no balance in it. The card
  carried `current_outstanding_balance_sum = 1964886258.21`.
* Its pandas ground truth used a plain `groupby`, which **drops the two loans
  with a null LTV bucket and £337,343.21 with them**. The product surfaces those
  rows as `Unknown / Missing` and its cross-tab reconciles to the book exactly.
  The grader called that a defect in four cases.

**A grader that marks the product wrong for being right is this surface's own
failure mode, and it is the reason the qualifier matters.** Both directions cost
the same thing — a number nobody can trust — and neither is visible without
probing the grader itself. `shipped_shapes --self-test` now exercises every
grader in both directions, and `_capture` carries the figures rather than the
prose alone.

## A LIMITATION ON EVERY CORPUS-DERIVED FIGURE IN THIS PACK

### The mechanism

**A test family enumerated from a projection cannot exercise that projection's
gap.**

Whenever a question set is written by working outward from an artefact — a field
registry, a view's column list, the segment names a config declares — the
questions inherit that artefact's idea of what exists. The corpus and the code
then share an assumption, and the corpus is blind in exactly the direction the
assumption is wrong. It is not that the corpus is too small. **A larger corpus
built the same way is blind in the same direction**, because size is not what is
missing: an independent source of what a reader might ask is what is missing.

This generalises past the forecast view that made it obvious. Any enumeration
carries it: a family written from the registry's field NAMES misses the plain
words a person uses for those fields; a family written from a config's segment
labels misses the other governed definitions of the same segment; a family
written from a view's columns misses everything the view drops. In each case the
questions are correct, the coverage looks broad, and the one thing that cannot be
reached is the thing the enumeration got wrong.

**The practical consequence for reading this pack: a clean corpus result is
evidence about the enumeration before it is evidence about the product.**

### The measurement, and four instances of it

`balance where account status is active` is about as ordinary as a management
question gets. Across all 693 questions in the four corpora there are **fifteen**
mentions of a named dimension behind a selector preposition, and **not one** is
the construction above. All four cases proving that defect had to be
**constructed**.

The same shape, three more times in the same programme:

| what was missed | how the corpora were built |
|---|---|
| a narrowed question answered over the whole book (`balance where account status is active` → 11,035 loans) | 15 selector-preposition mentions in 693; zero live instances |
| a governed population's whole-book answer (`balance of new lending` → 11,035 loans) | the seasoning family enumerates the SEGMENT names, so the months-on-book windows were unreachable |
| the front-book regression that cost 160 answers | the same family, the same enumeration |
| a field reported unavailable that the book carries | the 27 forecast questions are built from the 12 fields the forecast projection carries, so the projection's own gap cannot be exercised |
| a disclaimed view word choosing the frame (*"the balance by vintage, IGNORING the forecast"* loads the forecast frame) | 113 of 683 corpus questions mention a view word; **not one disclaims it**, so the entire disclaiming class was unreachable |

The last is the mechanism at its most literal — a family enumerated from the very
projection whose gap it would have to reach — but the seasoning instances are the
same shape one step removed: the family was enumerated from the SEGMENT names a
config declares, so the months-on-book windows that name the same population a
different way were unreachable, and the whole-book answer sat behind them.

### A SECOND LIMITATION, ON A CLEAN GREEN RESULT

**A surface that asserts the right things about the wrong number of owners
reports a partial fix as a complete one.**

B21 was diagnosed as one decision — which frame a question loads — and two
owners were named before the work began. There were **four**. The second was
found by enumerating where the answer arrives, before any test failed. The third
and fourth were found only because the constructed cases assert the VERDICT and
not merely the number: with the frame and the dataset both corrected,
*"the balance by vintage, ignoring the forecast"* computed the right figure over
the right 11,035 loans and was **still refused**, because a fourth reader was
raising a forward-projection facet from a word the sentence had ruled out.

The consequence for reading any result in this pack: **"the number is right" and
"the decision has one owner" are different claims, and only the second is what
the contract asserts.** A fix verified by the first alone stops as soon as the
arithmetic agrees, which in B21's case would have been two owners early — with
the question still unanswerable and the receipt still describing a request
nobody made.

### The strongest form: a decision with no corpus coverage at all

The portfolio lens — which cohort of loans an answer covers — **narrows zero of
the 697 corpus questions.** Not few: none. Every question naming both provenance
families resolves to the whole book through a comparison guard, and none names
one alone. Its only coverage anywhere is two unit-test files.

That decision was silently answering *"the balance for loans purchased at
auction"* over 3,909 of 11,035 loans — a complete, correctly formatted figure
over 35% of the book, for a question about how a property was bought. **No
corpus number in this pack moved when that was true, and none moved when it was
fixed.**

Where a figure in this pack rests on constructed cases rather than corpus
coverage, the confidence it carries is bounded by whether the right sentences
were imagined, and that is stated at the point the figure is given.

### What follows for reading this pack

1. **A count of "N of 693 affected" is a lower bound**, and where the construct
   is a plain phrasing rather than a registry term it may be a lower bound of
   zero against a live defect. Three of the four instances above measured zero
   while the defect was live.
2. **Constructed cases are not weaker evidence than corpus cases** in this
   programme; for several defects they are the only evidence that exists. Where a
   figure rests on constructed coverage the document says so.
3. The corpora remain the right instrument for REGRESSION — they are large,
   stable and byte-diffed. They are not an instrument for DISCOVERY.
