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
