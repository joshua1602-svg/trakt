# Two mechanisms, one question, different logic — a standing pattern

> **The shape:** two pieces of the system answer the same question about a
> sentence. One reads a **word list**; the other reads **governed values or
> context**. The weaker one **gates the route**. The stronger one **drives the
> refusal**. The user is told the product cannot do something it can.

Every instance has cost a client-facing wrong answer or a confident wrong
refusal, and every instance was invisible until a phrasing crossed the two
mechanisms' disagreement.

`docs/mi_b13_b14_b15_diagnosis.md` records the first three and states the
generalisation: *"every carriage failure recorded in this programme —
point-in-time, evolution, the run-rate path, and now these three — is a different
pair of readers disagreeing."*

---

## Instance 4 — the plural, and the comparison verb

**Recorded here as the fourth instance.** Full diagnosis:
`docs/mi_p0_segment_pair_refusal_fix.md`.

`How have direct and acquired balances moved over the periods?` refused with:

> *"I understood that you asked for Direct and Acquired tracked separately, but
> that could not be applied to the calculation."*

| | the weak reader | the strong reader |
|---|---|---|
| what | `_METRIC_TERMS` / `_COMPARISON_TERMS` | `execution_receipt.segments_named_in` |
| reads | a curated **word list** | the book's **governed dimension values** |
| on this sentence | no metric (`balances` absent); not comparative (no verb) | `["direct", "acquired"]` — correct |
| effect | **gated the route** — no plan composed | **drove the refusal** — named the segments |

The refusal was **true of the route and false of the product**: the same request
with a comparison verb composed and answered completely.

It was a *triple* disagreement — the comparison vocabulary appeared twice, in
`planner._population_pair` and again in `intent.classify`'s own copy, which
`is_comparative`'s docstring had already predicted:

> *"Public because the planner asks it rather than keeping a second comparison
> vocabulary — which is how the same question came to resolve two different ways
> depending on which of the two lists happened to contain its wording."*

**The warning was written, and the drift had already happened underneath it.**
A docstring is not a mechanism.

---

## Instance 5 — three readers of the time axis, found while fixing root 1

**The first instance where the weaker reader would have produced a confident
WRONG ANSWER rather than a refusal.** Full record:
`docs/mi_time_axis_vocabulary_prediction.md`.

Three readers ask *"did this sentence request a time axis?"*:

| reader | what it decides | vocabulary |
|---|---|---|
| `lexical.time_axis_request` | **owns the question** | the full one |
| `llm_query_parser.is_line` | the CHART TYPE | its own inline list |
| `chat_routing._EVOLUTION_MARKERS` | the ROUTE | a third list |

Widening only the first two produced this:

```
balance over time   -> route=evolution   3 rows   [period, value]       correct
balance by period   -> route=None       13 rows   [vintage_year, sum]   WRONG
```

`by period` became a line, missed the evolution route, and the generic executor
answered it as **13 origination vintages** — a cohort distribution presented as a
reporting-period series. The parser's own note at that path had already warned
that *"a VINTAGE is a cohort label, not a point on a time axis"*.

Worse, the measuring instrument scored it **PROVEN**, because `vintage_year` sits
in its own `_TIME_HINTS` list. A wrong answer that passed its own check.

`_is_evolution` now consults the owner, so chart type and route are decided by
one reading.

**The lesson is sharper than the previous four.** Instances 1–4 produced
refusals: the user was told something was impossible. This one produces an
answer to a different question, with no refusal to inspect, and it survived a
purpose-built rater. Counting the readers **before** widening any of them is the
only thing that would have caught it earlier — the third list was found by
tracing a wrong answer backwards.

---

## Why word lists lose

The strong reader resolves against **the book** — the values the data actually
holds. It is complete by construction and cannot miss a synonym, because it is
not matching synonyms.

The weak reader is a hand-maintained list. It is incomplete the moment a lender
types an inflection, a colloquialism, or an ellipsis nobody added. It fails
**silently and asymmetrically**: it does not error, it declines, and the decline
routes the question somewhere that answers a different question or refuses.

`docs/mi_recognition_diagnosis.md` measures the cost across 61 phrasings: **20 of
47 failures are recognition**, and the two dominant roots are both narrow word
lists — the time-axis vocabulary (`by period`, `each month`, `per period`, `over
the periods`, `between periods` all rejected) and the requirement that a measure
be named explicitly.

---

## The standing constraint

> **An instance of this pattern is not closed by adding a phrase to the weak
> list.**

Adding the missing wording fixes the reported sentence and leaves the mechanism
intact, so the next inflection reproduces it. Every instance so far was created
by exactly that move.

`docs/mi_b13_b14_b15_diagnosis.md` already states the constraint a fix must
satisfy — **one reader decides, for both vocabularies, or the drift returns in
the next wording added to either** — and it governs here too.

### The open item this protects

**Elided coordination is NOT to be closed with a vocabulary entry.**

```
"front book and back book movement"                        -> delivers
"how do the front and back books compare over the periods" -> still refused
```

The second shares one noun between two modifiers. `names_both_sides_of_a_pair`
does not recognise it, and **must not be taught this phrase**. Adding
`"front and back books"` to a list would close the example and leave the
mechanism unchanged — the next elision (`"the direct and acquired books"`,
`"our front and back book split"`) fails identically, and the pattern gains a
fifth instance.

Covering it means **parsing coordination** — recognising that two modifiers share
a head noun — which is a mechanism, not an entry. Until that exists the limit
stays open and recorded.

**Any change that closes this by adding a phrase should be rejected in review on
the strength of this document.**

---

## Recognising the next one

Two questions, on any sentence-reading code:

1. **Does anything else in the system read this same sentence for this same
   fact?** If yes, do they share a reader, or do they have a list each?
2. **Which of them gates a route, and which of them writes the message the user
   sees?** When those are different mechanisms, the user can be told the product
   cannot do something it can.

A refusal that names something specific — *"you asked for Direct and Acquired
tracked separately"* — is evidence that **something** understood the sentence. If
the route did not, the two readers have already disagreed.
