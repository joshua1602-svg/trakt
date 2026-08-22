# Items 3 and 4, re-measured

Measurement and diagnosis only. Nothing proposed, nothing fixed.
Base: HEAD `31e07be`.

Items 3 and 4 were left as *"re-measure rather than schedule"* because item 2
might have closed them. It did not. All three remaining unhelpful refusals stand.

```
   correct              12
   wrong answer          0
   honest refusal        0
   unhelpful refusal     3      B1, B4, A5
```

---

## 1. One cause or three? **Two.**

The same discipline that overturned the B3/C4 pairing, applied before proposing
anything.

| | B1 | B4 | A5 |
|---|---|---|---|
| question | *"What is the LTV for loan **tickets** above £150k?"* | *"For **tickets** larger than £150k, what is the LTV?"* | *"Tell me the basics about this book"* |
| what parsed | measure `Current LTV`, filter `Balance > 150000`, **5,857 loans** | identical | **nothing** |
| refusing layer | the **guard** — `reconcile_facets` | the **guard** | the **parser** — `mi_agent_workflow`, `note == "unmapped"` |
| message | *"I could not tell how you meant ticket"* | identical | *"I couldn't map this question to a governed analytic"* |
| cause | `KIND_UNRESOLVED_ROLE` on the word `ticket` | same | `_SUMMARY_MARKERS` is a nine-phrase literal list |

**B1 and B4 are one cause. A5 is a second, at a different layer.** B1 and B4
differ only in word order and produce byte-identical refusals over identical
resolved state.

## 2. B1/B4 — the cause, precisely

The word `ticket` is a registry synonym for the dimension `ticket_bucket`. D2's
`dimension_role` has four sources — reader 1's filter slot, reader 1's axis slot,
the sentence's grouping cut, and "the book cannot express it" — and **none of
them covers a word that is the SUBJECT of a threshold.**

So the role falls through to `ROLE_UNRESOLVED`, a `KIND_UNRESOLVED_ROLE` facet is
raised, and `reconcile_facets` returns `VERDICT_CLARIFY`.

**Everything else about the question resolved correctly**, which is what makes
this an unhelpful refusal rather than an honest one:

```
  measure  Current LTV
  filters  ['Balance > 150000']
  pop      5857          <- the right population, in hand, then declined
```

Substituting any word that is not a registry dimension answers:

```
  "the LTV for loan tickets above £150k"    ok=False
  "the LTV for tickets above £150k"         ok=False
  "the LTV for loans above £150k"           ok=True
  "the LTV for accounts above £150k"        ok=True
  "the LTV for loan balances above £150k"   ok=True
  "For loans larger than £150k, ..."        ok=True
```

**The trigger is the word, not the sentence shape.**

### The multi-owner shape is already visible

`execution_receipt._THRESHOLD_SUBJECTS` maps `\bticket\b` to `"balance"`. The
receipt module **already holds the fact that `ticket` names the balance in a
threshold context**, and `dimension_role` — in the same module — does not read
it. Two readers of *"what does the word `ticket` refer to here?"*, reaching
opposite conclusions: one says "the balance", the other says "I cannot tell".

That is the shape items 1 and 2 closed, in a third place. Stated as a diagnosis,
not a proposal.

### An adjacent defect found while measuring — recorded, not opened

`_threshold_subject` labels B1's facet **`"LTV over 150000"`** when the filter
execution applied is **`Balance > 150000`**. Its subject list is ordered
`LTV, age, balance` and its look-back window is 42 characters, so it catches the
measure named earlier in the sentence rather than the noun the threshold is on.

The receipt therefore names the wrong subject for a correctly-applied filter.
Not a wrong number; a wrong disclosure. **Recorded.**

## 3. A5 — a different cause, at a different layer

`_is_portfolio_summary` requires one of **nine literal phrases**:

```
summarise the portfolio · summarize the portfolio · portfolio summary
summarise the book      · summarize the book      · summary of the portfolio
overview of the portfolio · overview of the book  · portfolio overview
```

*"Tell me the basics about this book"* contains none, so the route is not
claimed, the deterministic parser finds no metric and no dimension, and
`mi_agent_workflow` reports the question unmapped.

**This is NOT the multi-owner shape.** `_SUMMARY_MARKERS` has exactly one reader.
It is an incomplete list, not a list that disagrees with another one — so the
treatment items 1 and 2 used does not apply here, and saying so is the point of
diagnosing before proposing.

### An observation, not a defect

Summary phrasings reach answers by **two different routes**:

```
  "Please provide a portfolio summary"                route=portfolio_summary
  "Give me a summary of the portfolio"                route=portfolio_summary
  "summarise the book"                                route=portfolio_summary
  "book overview"                                     route=None  (generic executor)
  "key metrics"                                       route=None
  "What are the headline numbers for the portfolio?"  route=None
  "Tell me the basics about this book"                REFUSED
  "Tell me about this book"                           REFUSED
```

The governed route returns regional exposures and the provenance split; the
generic path returns a two-KPI card. Both are correct and they are materially
different answers to the same question. **Not a wrong number. Recorded.**

## 4. The population check is vacuous across all 44

Carried in, and it is worse than "no question states a threshold".

The check fires only on a stated threshold, and none of the 44 states one. But
**the analytical route publishes no `executionSummary.population` at all**, so
even a question that stated a threshold would have nothing to compare against on
that path. The check is not merely unexercised on this bank — on the analytical
route it is unexerciseable.

### What it would take to close

The analytical composition layer would emit, per composed capability, the
population it measured and the whole-book total it measured against — the same
two numbers the point-in-time path already publishes. A receipt change on the
analytical path, not a grader change.

**Not opened.** Stated so the size is known before anyone decides.

## 5. What this changes about the ordering

Items 3 and 4 were ranked third and fourth on the measurement. That still holds,
and the re-measurement sharpens it:

* **Item 3 (B1/B4)** is one cause, is the multi-owner shape in a third place,
  and the answer is already in hand when the refusal is issued. Two questions.
* **Item 4 (A5)** is a nine-phrase list with one reader. One question, no
  consolidation available, and the smallest change of the four.

Neither moves a number. Both convert a refusal into an answer the product can
already compute.
