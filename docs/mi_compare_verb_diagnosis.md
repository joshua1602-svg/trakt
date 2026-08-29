# "compare" — diagnosis before any fix

No code changed for the defect. Base: the merged state.

---

## 1. My recorded mechanism was WRONG

The backlog entry said the word `compare` "is resolved as a MEASURE" and the
substitution guard then declines it. **Measured, that is not what happens.**

```
_measure_hits("how does the front book compare with the back book")  ->  []
```

`compare` is not a registry synonym and the measure reader never sees it as a
measure. The parsed spec carries `measures: []` and `metric: None`.

**What actually happens:** the sentence names **no measure at all**, and the
message names the verb instead of saying so.

## 2. The real site, and it is a GOOD guard misfiring

`llm_query_parser._metric_side_residue`. Its own comment states the intent:

> NEVER substitute a different measure for one the user named. A grouped
> question whose metric side carries an unresolvable noun phrase ("the unicorn
> ratio by region") used to default to balance and answer with `ok:true` — a
> confident answer to a question nobody asked.

That guard is right and belongs to the same family as everything closed this
month. The defect is narrow: **its residue vocabulary does not exclude verbs**,
so a sentence whose metric side is purely verbal yields the verb as the
"unresolvable measure the user named" — when the user named none.

```
_metric_side_residue("how does the front book compare with the back book")  ->  'compare'
```

## 3. One decision or several? **The verb reading is one. The family is not.**

Twelve verbs in one frame — *"How does the front book VERB with the back book?"*:

| verb | outcome | mechanism |
|---|---|---|
| compare, contrast, differ, **stack** up, **measure** up, perform | refused, **verb named as the measure** | `_metric_side_residue` |
| break down, split, rank, look | refused, *"Back Book could not be applied"* | a **different** decision — the population is lost |
| trend, move | **answered** | claimed by the movement route |

**Six verbs share the one decision.** Note `stack up` and `measure up` yield
`stack` and `measure` — the residue takes the first word of a phrasal verb.

### The second decision is independent, and this proves it

With a measure named — *"…VERB with the back book **on balance**"*:

```
compare   -> ANSWERS   (bar, 2 groups, grouped by Seasoning Segment)
differ    -> ANSWERS   (same)
contrast  -> still refused: "Back Book could not be applied"
perform   -> still refused: "Back Book could not be applied"
```

Naming a measure fixes `compare` and `differ` and **does not fix** `contrast`
and `perform`. So the "Back Book not applied" failure is a separate defect that
the verb-residue fix will not close, and scoping them as one would close neither
cleanly — the B3/C4 lesson.

## 4. What other verbs are at risk

The residue vocabulary admits **any word on the metric side that resolves to no
governed field**. Verbs a CFO would plausibly use in this frame and that are
therefore at risk: *compare, contrast, differ, stack up, measure up, perform,*
and by the same rule *fare, sit, screen, benchmark, weigh up, shape up*.

The rule is not "these six words". It is: **a sentence whose metric side
contains no measure noun will report its most salient leftover token as an
ungoverned measure**, and for a comparison question that token is the verb.

## 5. Why the corpus scores 4/4 while the bare verb fails

The robustness bank's front-versus-back family scores **4 of 4 correct on both
books**. Its four phrasings are:

```
How does the front book compare with our older lending FROM A RISK PERSPECTIVE?
Are older loans RISKIER than the loans we've originated recently?
How different is the RISK PROFILE of recent originations versus the back book?
Compare the CREDIT PROFILE of the front book with our seasoned loans.
```

**Every one names a measure side.** None uses the bare verb against two named
populations, so none can reach the residue path.

That is the `split by` rule again, and this is its fourth instance: **a phrasing
that appears to work while another carries the defect masks the mechanism.**
`split by` passed a `by`-only splitter because it contains "by"; these four pass
because they name a measure. In both cases the corpus reports a clean family
while the plainest form of the same question fails.

## 6. Scope, not proposed as a fix

* **6A — the residue must not report a verb.** One decision, one owner, six
  measured instances. The honest message for a sentence naming no measure is
  *"you did not say what to compare them on"*, not *"'compare' is not a governed
  measure"*.
* **6B — "Back Book could not be applied" for contrast/perform.** A separate
  decision, independently demonstrated by §3. Diagnose before scoping.

Nothing here is implemented. Both are stated so the next pass starts from a
measured mechanism rather than the wrong one I recorded first.
