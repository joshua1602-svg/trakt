# The sibling rule — a standing definition

> **A sibling counts only if it shares the SHAPE and the TARGET.**
>
> Two phrasings of the same shape that ask different questions are not the same
> request reworded, and counting them as such claims a routing fix reaches what
> it cannot.

This is a measurement definition, binding on any instrument that reports
reachability. It exists because getting it wrong does not produce an obviously
wrong number — it produces a plausible one that misdirects scope.

---

## The rule

For a phrasing that does **not** deliver, a **sibling** is another phrasing that
**does** deliver and that shares both:

1. **the shape** — the capability being asked for (T1–T8); and
2. **the target** — the thing the answer must be broken down or filtered by
   (`region`, `ltv_band`, `ticket`, `seasoning`, `source_split`, `threshold`).

If such a sibling exists, the failing phrasing is a **ROUTING GAP**: the product
serves this request, and a wording reaches it. If none exists, it is a
**CAPABILITY GAP**: no declared wording reaches this request.

**A breakdown marker outranks a filter mention** when resolving the target.
*"balance over time by region for the front book"* is a **region** request
filtered to the front book — not a seasoning request. Grouping it by the filter
would put it with the wrong siblings.

---

## Why shape alone is not enough

`Which region has grown fastest?` and `Which LTV band moved most between
periods?` are both shape 7. The first delivers; the second refuses. They are not
the same question in different words — one asks about regions, the other about
LTV bands.

Counting the second as *reachable* because the first works would assert that a
routing fix delivers LTV-band rankings. It would not. The capability is wired to
one dimension, and the LTV band is a genuine gap.

Measured on the time-series surface, the difference is not marginal:

| grouping | routing gaps | reads as |
|---|---|---|
| same shape | 7 of 21 | a third of the absences are routing |
| same shape **and target** | **3 of 21** | the absences are real, with three exceptions |

**A third versus a seventh is a different project.** The looser number moves work
out of a build that still has to happen.

---

## The failure mode it prevents

An overstated routing number is not a rounding error. It says *"this already
ships, just phrase it differently"* about something that does not ship. That
claim reaches a client faster than any correction, and it removes work from a
build estimate that the build still owes.

The conservative direction is asymmetric and deliberate: **an understated routing
number costs a rediscovery; an overstated one costs a promise.**

---

## Where it is enforced

`question_interpretation/mi_capability_recontent.py`

* `request_target(question)` resolves the target from an explicit, inspectable
  vocabulary, with breakdown markers outranking filter mentions.
* `sibling_analysis(result)` reports **both** groupings — `routing_gaps` (shape
  only) and `strict_routing_gaps` (shape **and** target). Both are printed. The
  strict number is the one to plan from; the loose number is kept visible so the
  gap between them is never invisible.

`tests/test_mi_capability_recontent.py` pins it:

* `test_a_different_parameter_of_the_same_shape_is_not_a_routing_gap` — asserts
  the strict count is 0 where the loose count is 1;
* `test_the_same_request_in_other_words_is_a_routing_gap` — asserts the strict
  count is 1 where it genuinely is;
* `test_a_breakdown_marker_outranks_a_filter_mention`.

Collapsing the strict grouping to shape-only breaks the first of these.

---

## Standing consequence

Any reachability figure quoted to scope work — a build estimate, a client
statement about what ships, a decision to drop something from a backlog —
**must be the strict number, and must say so.** A figure that does not state its
grouping should be treated as the loose one and re-derived before use.

**One thing the strict number is not:** a ceiling. It is measured against the
declared phrasing bank. A wording nobody has tried may reach a request counted
here as a capability gap, so the honest statement is *"no declared wording
reaches this"*, and widening the bank is the way to find out.
