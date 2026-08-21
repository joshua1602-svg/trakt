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

## A LIMITATION ON EVERY CORPUS-DERIVED FIGURE IN THIS PACK

**The corpora are built from registry names and enumerated families. They
systematically miss plain phrasings.** Every number in this pack that is
expressed as "N of 693" — or 697, or any corpus denominator — inherits this, and
it is stated here rather than only in the engineering brief because it qualifies
the evidence, not the engineering.

### The measurement

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

The last is the clearest statement of the mechanism: **a family enumerated from a
projection cannot exercise that projection's gap.** The corpus and the defect
share an author's assumption, so the corpus is systematically blind to exactly
the thing the assumption got wrong.

### What follows for reading this pack

1. **A clean corpus result is evidence about COVERAGE before it is evidence about
   the product.** This was already a standing rule; it is restated here because
   the pack's headline figures are corpus-derived.
2. **A count of "N of 693 affected" is a lower bound**, and where the construct
   is a plain phrasing rather than a registry term it may be a lower bound of
   zero against a live defect.
3. **Constructed cases are not weaker evidence than corpus cases** in this
   programme; for several defects they are the only evidence that exists. Where a
   figure rests on constructed coverage the document says so.
4. The corpora remain the right instrument for REGRESSION — they are large,
   stable and byte-diffed. They are not an instrument for DISCOVERY.
