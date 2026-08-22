# B21, B22, B23 — one shape, two decisions, and a fix that is already written

Diagnosis only. **Nothing is fixed here**, as instructed, and no design is
proposed beyond establishing whether the two share one.

Base: HEAD `cd4a005`.

---

## 1. The shape

> A bare substring anywhere in the question silently changes what the system can
> see.

| | **B21** `workspace.resolve_active_view` | **B22** `portfolio_lens.resolve_lens` |
|---|---|---|
| test | `"forecast" in q` | `_contains_any(low, _DIRECT_TERMS)` |
| vocabulary | `forecast`, `pipeline`, `funded` | `direct`, `organic`, `in-house`, `acquired`, `acquisition`, `purchased`, `inorganic`, `m&a` |
| drops | **columns** — 60 of the book's 71 | **rows** — everything outside one cohort |
| runs | before parsing, before routing | inside the parse |
| default | `funded`, the whole book | `total`, the whole book |
| comparison guard | none | yes |
| **disclosed to the reader** | **no** | **yes** |

Both default to the widest scope, which is right. Neither can tell a question
**about** a thing from a question that **mentions** it.

## 2. Do they share a cause? Yes, and it is narrower than "both use substrings"

The shared cause is not the substring test. It is that **both vocabularies are
words that appear in ordinary English about lending**, and neither decision
requires the word to be doing the job the decision assumes.

`purchased`, `acquired`, `direct`, `funded`, `forecast` are all words a person
uses about a loan, a property, a channel or a projection without naming a book.
The decisions read them as naming a book.

## 3. Do they share a FIX? Yes — and for B22 it is already written, in the same module

**`portfolio_lens` already states the doctrine and implements the test.**

```python
# Only QUALIFIED phrases count. A bare "current" or "entire" is ordinary
# English; it becomes a scope reference when it qualifies a book noun, which is
# what keeps "current LTV" and "current reporting date" out of this vocabulary.
_SCOPE_NOUNS = ("book", "books", "portfolio", "portfolios", "platform", "loan book", "aum")
_SCOPE_QUALIFIERS = ("funded", "unfunded", "whole", "entire", "total", "current",
                     "overall", "consolidated", "combined", "selected", "active",
                     "direct", "acquired", "purchased", "originated", "sponsored")
```

**`_SCOPE_QUALIFIERS` already contains `direct`, `acquired`, `purchased` and
`funded`** — the exact vocabularies of both decisions — and `_SCOPE_PHRASE_RE`
already requires them to qualify a book noun.

`scope_phrase_spans` publishes that test. Measured against every case in this
diagnosis, it discriminates perfectly:

| question | scope span found | `resolve_lens` says |
|---|---|---|
| the **acquired book** | `"of the acquired book"` | Acquired ✓ |
| the **direct book** | `"of the direct book"` | Direct ✓ |
| across the **funded book** | `"across the funded book"` | Total ✓ |
| loans **purchased** at auction | *none* | **Acquired ✗** |
| **directly** held collateral | *none* | **Direct ✗** |
| the borrower **acquired** the property | *none* | **Acquired ✗** |

**Every legitimate case has a span. Every defect has none.** The test that fixes
B22 is a function in the same file, over the same words, that `resolve_lens` does
not call.

`scope_phrase_spans` exists to protect OTHER resolvers from the scope vocabulary
— `mask_scope_phrases` is called by the filter and dimension parsers. It was
never turned on the two decisions that own that vocabulary.

**This is the carriage pattern at its sharpest yet.** Not "produced and unread",
not "built for one field when it generalises to all", but *written as a doctrine,
implemented as a function, applied to everyone except the decision it was written
for.*

### For B21 the doctrine transfers; the vocabulary does not

`forecast` is **not** in `_SCOPE_QUALIFIERS`, and "balance" is not a
`_SCOPE_NOUN`, so `"the forecast funded balance"` yields no span. The view
decision needs its own qualified vocabulary — the same shape, a different noun
set. So:

> **They share a doctrine and a shared second component. B22's fix is available
> today from an existing function; B21's needs its vocabulary written first.**

### The second component, which is new and shared

Qualification alone does not cover **disclaiming**:

```
"balance by vintage, ignoring the forecast"
"Show total balance by region excluding forecast contributions."
"the balance of the front book, before any forecast adjustment"
```

*"forecast contributions"* is a qualified phrase and would still fire. A mention
inside a negative construction — *ignoring*, *excluding*, *setting aside*,
*before any*, *other than*, *net of* — does not select; if anything it selects
the opposite. **Neither decision has this, and both need it.**

## 4. The asymmetry, measured

The work order asks which is more urgent within the pair, from measurement rather
than judgement. Measured on the shipped path, this book:

### B22 changes a number today

```
"What is the balance for loans purchased at auction?"
   receipt: filtersApplied ["Source Portfolio in alp_acquired"]
            population 3,909 of 11,035
```

A complete, correctly formatted answer over **35% of the book**, for a question
about how a property was bought.

### B21 does not change a number on this book — it destroys the answer

```
"What is the balance by vintage, ignoring the forecast?"   -> "vintage_year: field_missing"
"What is the balance by region, setting aside the forecast?" -> refuses on a projection facet
```

And the reason it does not change a number here is worth stating exactly,
because it is a property of the BOOK and not of the code:

```
funded    rows 11,035   sum(current_outstanding_balance) = 1,964,886,258.21
forecast  rows 11,035   sum(current_outstanding_balance) = 1,964,886,258.21
```

`build_forecast_view_frame` puts the **forecast contribution** in
`current_outstanding_balance` — the same column name, a different meaning. On
this book there is no pipeline data, so the contribution equals the funded
balance and the two agree. **On a book with a pipeline they would not, and B21
becomes a wrong number under the same field name, undisclosed.**

### So, within the pair

| | changes a number **now** | disclosed | fix available |
|---|---|---|---|
| **B22** | **yes**, 3,909 of 11,035 | yes, on the receipt | **yes — an existing function** |
| **B21** | not on this book; **yes on any book with pipeline data** | **no** | doctrine yes, vocabulary no |

**B22 is more urgent**: it is the only one changing a number on the book in front
of us, and its fix is a call to a function that already exists. B21 is more
*dangerous* — undisclosed, and a wrong number the moment a pipeline exists — but
its harm here is a destroyed answer rather than a false one.

That B22's harm is visible on the receipt and B21's is not is the accidental
safety this programme has now declined to rely on five times. **It is a reason to
fix B21 too, not a reason to rank it second.**

## 5. B23 is not a third decision; it is B21's missing half

The lens is declared: `filtersApplied` carries `Source Portfolio in
alp_acquired`, and the population count drops. The view is declared nowhere the
reader looks — the receipt for a forecast-view answer reports
`population 11,035, populationTotal 11,035` and says nothing about the twelve
columns.

**B23 collapses into B21.** There is no separate disclosure defect for the lens.

## 6. Coverage — what no surface can see

* **The lens narrows ZERO of the 697 corpus questions.** Its only coverage is
  `tests/test_p1i_scope_resolution.py`. Every question in the corpora that names
  both families resolves to Total through the comparison guard, and none names
  one alone.
* **87 questions take a non-funded view** and none names a field the projection
  drops (measured in D6).

Both figures are the corpus limitation already recorded in the due diligence
pack, arriving again: **a family enumerated from a projection cannot exercise
that projection's gap**, and a family that never asks about provenance cannot
exercise the provenance lens.

**Any work on B21/B22 is constructed coverage from the first line.**

## 7. What this diagnosis does not settle

Stated so it is not assumed settled:

* whether the qualified-mention test should be **one** function over both
  vocabularies or two callers of one helper;
* what a disclaiming construction should DO — select the opposite, or decline to
  select;
* whether `resolve_active_view` should run before parsing at all, which is a
  larger question than the substring test and is not asked here.
