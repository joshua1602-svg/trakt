# Two mechanisms, one question, different logic — a standing pattern

> **The shape:** two or more pieces of the system answer the same question about
> a sentence, by different logic. One reads a **word list**; another reads
> **governed values or context**. They disagree, and the disagreement is
> invisible until a phrasing crosses it.

**There are TWO CLASSES, and they are not degrees of the same thing.** The
difference is what the user gets, and it decides how the failure is found.

| | **Class A — the wrong refusal** | **Class B — the wrong answer** |
|---|---|---|
| what the weak reader does | **gates the route** | **gates the route** |
| what the strong reader does | **drives the refusal**, naming what it understood | nothing — no refusal is raised |
| what the user gets | *"that could not be applied"* — a confident refusal of something the product can do | **an answer to a different question**, delivered as if correct |
| how it is found | the refusal quotes the element, so the sentence is provably understood; read the message | nothing announces it; only checking the artifact against the question finds it |
| instances | 1, 2, 3, 4 | **5** |

Class A is recoverable by inspection: the refusal *itself* is the evidence, and
a user who reads it knows something is wrong. **Class B leaves no such trace.**
The answer is well-formed, internally consistent, and wrong, and the only thing
that distinguishes it is whether the object it returns is the one the sentence
asked for.

`docs/mi_b13_b14_b15_diagnosis.md` records the first three and states the
generalisation: *"every carriage failure recorded in this programme —
point-in-time, evolution, the run-rate path, and now these three — is a different
pair of readers disagreeing."* That generalisation holds; what it did not
anticipate is that a pair can disagree **silently**.

---

# CLASS A — the wrong refusal

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

# CLASS B — the wrong answer

**A distinct class, recorded separately because the practice that catches it is
different.** Class A is caught by reading a refusal. Class B produces no
refusal, so nothing prompts anyone to look.

## Instance 5 — three readers of the time axis, found while fixing root 1

Full record: `docs/mi_time_axis_vocabulary_prediction.md`.

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

### The two practices that catch Class B

Neither is a code change. Both are things to do before touching a vocabulary.

> **1. Count the readers of a decision before widening any of them.**
>
> Not "find the list" — **count the lists**. The question *"did this sentence
> request a time axis?"* had three readers; widening two of them is what
> produced the wrong answer. A widening is safe only when every reader of that
> decision moves together, or when all but one have been made to defer to an
> owner.
>
> The third list here was found by tracing a wrong answer backwards. That is the
> expensive order. Counting first is cheap and takes one `grep`.

> **2. A rater whose hint list overlaps the decision under change is COMPROMISED
> for that change.**
>
> The content rater scored the vintage answer **PROVEN** because `vintage_year`
> sits in its own `_TIME_HINTS` — the same kind of list, naming the same kind of
> thing, as the vocabulary being widened. It could not be an independent check
> of a time-axis change, because it holds an opinion about what a time axis is.
>
> When a change touches a concept, any instrument whose own vocabulary names
> that concept must be treated as **part of the change, not as evidence about
> it**. Check such a change against the artifact directly — compare the object
> returned against the object the sentence asked for — or against an instrument
> that shares no vocabulary with the decision.

**Together they are the whole defence.** Practice 1 stops the readers
disagreeing; practice 2 stops the disagreement being ratified by a measurement
that shares the defect.

---

## A coupling that widening inherits — root 1 into root 2

**Recorded here so whoever argues root 2 has it in front of them rather than
discovering it again.**

`llm_query_parser`'s line path contains a standing default:

```python
elif metric is None:
    metric, agg = _balance_metric(semantics), "sum"
```

**A question that reaches the line path and names no measure is answered as
Total Balance.** That default predates all of this work and is untouched by it.

The consequence is the part that matters:

> **Anything that widens what reaches the line path inherits that default.**

It is not a property of the widening — it is a property of the destination. A
change scoped entirely to *how a time axis is recognised* silently became a
change to *what happens when no measure is named*, because the newly-recognised
questions arrived somewhere that already had an opinion about missing metrics.

Measured, when the time-axis vocabulary was widened without a guard:

```
how is the loan book tracking month to month      -> "Total Balance"
how is the front book tracking over the periods   -> "Total Balance"
```

Neither names a measure. Both had refused before. Both now answer, with a metric
nobody asked for.

A guard now confines the default to the pre-existing vocabulary, so a
newly-carried axis cannot make a metric-less question answerable
(`docs/mi_time_axis_vocabulary_prediction.md`). **That guard fixes the blast
radius, not the coupling.** The default is still there, and the next widening
that routes questions into that path will inherit it again unless it carries its
own guard or the default is settled on its merits.

### What this means for the root 2 argument

The standing position is that a question naming no measure should resolve
nothing — the no-silent-substitution rule working, not a defect. **The line
path does not implement that position.** So the root 2 argument is not
"should we add a default?" — it is:

1. **a default already exists** on one path, and has for some time;
2. either it is correct, in which case the rule has an exception that should be
   stated rather than left implicit; or it is wrong, in which case removing it
   is a behaviour change on questions that answer today;
3. and until one of those is settled, **every widening that feeds the line path
   must guard against inheriting it** — which is a tax on unrelated work.

Whoever takes that argument should also check whether other paths carry
comparable defaults; only the line path was examined here, and it was found by
accident rather than by looking.

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

**For Class A — the wrong refusal:**

1. **Does anything else in the system read this same sentence for this same
   fact?** If yes, do they share a reader, or do they have a list each?
2. **Which of them gates a route, and which of them writes the message the user
   sees?** When those are different mechanisms, the user can be told the product
   cannot do something it can.

A refusal that names something specific — *"you asked for Direct and Acquired
tracked separately"* — is evidence that **something** understood the sentence. If
the route did not, the two readers have already disagreed. **The message is the
tell**, and it is free to read.

**For Class B — the wrong answer:** there is no message, so neither question
above fires. Ask instead, before widening anything:

3. **How many readers does this decision have?** Count them. Two of three moving
   is how instance 5 happened.
4. **Does my check share a vocabulary with the thing I am changing?** If the
   rater's own hint list names the concept under change, it cannot adjudicate
   that change.
5. **Where do the newly-recognised questions arrive, and what does that
   destination already do on its own?** A widening inherits every default at its
   destination — see the root 1 / root 2 coupling above.

Question 5 is the one with no natural prompt. Nothing fails, nothing is refused,
and the new behaviour looks like the feature working. **The only reliable check
is to compare the object returned against the object the sentence asked for** —
which is the rule the whole time-series surface was built on, applied to a
change rather than to a product.
