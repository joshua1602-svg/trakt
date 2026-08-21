# One seasoning role owner — pre-registered prediction

Written before implementing. Nothing in this document was measured after seeing
a result.

Base: merge-base `4e051f3`; `4e051f3` and `28ece25` both ancestors of HEAD
(`b2c965a`).

Full discipline, because this is the seasoning vocabulary: `32c263a` broke
exactly this family and cost 160 answers and thirty points.

---

## 1. The shape, as directed

**One reader owns the role. Reader 2 consumes the decision and derives none of
its own.**

The owner takes **reader 3's vocabulary** and **reader 1's discipline**:

| taken from | what | why |
|---|---|---|
| reader 3 | `lending_windows_named` — new, recent, front book, back book | its trigger is the defect, not its vocabulary. It is the only reader that knows "new lending" names a population. |
| reader 1 | runs on **every** parse, not only composite plans | B13 is entirely the consequence of a role decision that runs only for composite questions |
| reader 1 | consults the available columns before deciding | a book without the field must not acquire a predicate against it |
| reader 1 | one named window selects; two or more compare | naming both sides is a comparison, and narrowing to one of them answers a different question |

The owner returns a governed **predicate**, not a segment string, because the
windows do not all resolve to the same field:

```
new         -> {months_on_book: {op: le, value: 1}}
recent      -> {months_on_book: {op: le, value: 3}}
front_book  -> {seasoning_segment: Front Book}
back_book   -> {seasoning_segment: Back Book}
```

That is what makes "new lending" narrowable at all: it is a months-on-book
bound, and `_SEGMENT_PHRASES` has no way to express it.

**Reader 2** (`requested_dimension_terms`) stops raising `seasoning_segment`
when the owner has taken the phrase as a population. It raises nothing of its
own about seasoning.

## 2. B14 is NOT fixed by consolidation, and is handled separately

Stated plainly so it is not assumed closed.

Consolidation cannot fix it. B14 is a **column check made against the loaded
frame**: for a run-scoped forecast question the service loads the pipeline
frame, not the funded tape, and `seasoning_segment` is absent from it. Every
reader, consolidated or not, would consult that same frame and reach the same
answer. *"That field is unavailable"* is true of the frame and false of the book
the reader asked about.

The check must be scoped to **the book's schema**, not to whichever frame a
route-scoped load happened to produce. That is a change to what the check reads,
not to how many readers there are. **Not in this commit.** It stays B14.

## 3. Pre-registered prediction

### 3.1 What may move

Nineteen corpus questions have reader 2 raising `seasoning_segment` today.
Sixteen are the seasoning families. Predicted, per group:

**A — a single window, no segment phrase (the B13 shape).** `Q1.1` on both
books, and the constructed *"balance of new lending"*, *"run rate of new
lending"*.
Today: no population, a grouping over the whole book.
Predicted: a population `months_on_book le 1`; reader 2 raises no grouping.
*"Balance of new lending"* narrows instead of returning 11,035 loans.

**B — a single window that IS a segment phrase (the B15 shape).** `pop_253`,
`pop_254`, `pop_255`, and *"show the back book balance by month"*.
Today: a population AND a redundant grouping.
Predicted: the population unchanged; the grouping gone. `pop_253/254/255` keep
their verdicts and their numbers.

**C — two windows named (a comparison).** `Q7.3`, `Q7.4`, `Q8.1–8.4`, both
books.
Predicted: **no change at all.** Two windows is not one, so no population is
selected, exactly as today.

**D — the asymmetric comparison.** `Q7.1`, both books: *"how does the front book
compare with our OLDER LENDING"*. `segments_named` sees one (`Front Book`);
`lending_windows_named` sees two (`front_book`, `back_book`).
Today, on the point-in-time path, reader 1 sees a single segment and injects a
Front Book filter — narrowing a comparison to one of its sides.
Predicted: two windows, so no population. **This is a behaviour change and an
improvement, and it is predicted here rather than explained afterwards.**
On the routed path `analytical_composition` claims Q7.1 and is unaffected.

### 3.2 What must not move

1. **The seasoning families stay 20 of 20 by name**, both books, measured by
   name and never inside an aggregate.
2. **No facet kind other than `grouping_dimension` and `row_population` changes
   kind, status or count.**
3. **The stamping matrix stays at 0 live holes**; B5 stays unreachable.
4. **Answer text moves only on group A and group D.** Any movement on B or C
   stops the work.
5. **The routed surface's 13 cases keep their outcomes**, except where group A
   or D predicts otherwise.

### 3.3 Stop conditions

Stop and report, do not absorb:

* any movement in the seasoning families' by-name counts;
* any answer moving that is not in group A or group D;
* a live hole appearing in the stamping matrix;
* any facet kind outside `grouping_dimension` / `row_population` changing;
* a lexical decision moving — the owner reads the seasoning vocabulary, not the
  lexical one, and must not touch it.

### 3.4 Acceptance

* B13 closed: *"What is the balance of new lending?"* narrows, and does not
  return 11,035 loans.
* B15 closed: no receipt carries the same field as both an applied population
  and a lost grouping.
* All three surfaces run, deterministic arm, both books where the surface has
  two.
* Seasoning families reported by name.

---

## 4. For the record — a comment is not evidence

The source comment delegating the seasoning role decision to the analytical
intent layer describes an intention the code does not honour: that layer runs
only for composite plans, so for every simple question the decision it claims to
own is never taken.

**This is the third comment in this programme documenting a behaviour the code
does not have.** Recorded as a pattern in the standing rules, not as an
incident:

1. the region resolver's docstring claiming a `None` return made validation fail
   clearly;
2. the P1A test whose assertion documented its own defect;
3. this delegation comment.

**A comment stating an invariant is not evidence the invariant holds.** Where a
comment asserts one, the assertion belongs in a test, and until it is in a test
it is a hypothesis about the code rather than a description of it.
