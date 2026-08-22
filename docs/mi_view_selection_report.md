# Is `resolve_active_view` one instance, or one of several?

Asked alongside D8. **Reported, not fixed**, as instructed.

**Answer: one of several. There are two text-driven data-visibility decisions in
this system and both are substring tests. One drops columns, one drops rows, and
the second is the more dangerous of the two by construction while being the safer
by accident.**

---

## 1. The two decisions

| | `workspace.resolve_active_view` | `portfolio_lens.resolve_lens` |
|---|---|---|
| decides | which **frame** is loaded | which **cohort** of rows is in scope |
| trigger | bare substring: `"forecast" in q` | bare substring against phrase families |
| drops | **columns** — 60 of 71 on the forecast view | **rows** — everything outside one cohort |
| runs | before parsing, before routing | inside the parse |
| default when nothing matches | `funded` (the whole book) | `total` (the whole book) |
| guard against a comparison | **none** | yes — both families present falls back to `total` |
| guard against an incidental mention | **none** | **none** |

Both default to the widest scope when they see nothing, which is right. Neither
has any way to tell a question ABOUT a thing from a question that MENTIONS it.

## 2. What a question loses to an incidental mention

Measured on the shipped path. Each pair differs only by a subordinate clause:

```
"What is the balance by vintage?"
   -> 13 groups, 11,035 loans.

"What is the balance by vintage, ignoring the forecast?"
   -> "The query cannot be answered from the prepared data: vintage_year: field_missing"

"What is the balance of the front book?"
   -> Seasoning Segment = Front Book, 1,177 loans.

"What is the balance of the front book, before any forecast adjustment?"
   -> "...: seasoning_segment: field_missing"

"What is the balance by seasoning segment excluding pipeline cases?"
   -> "No governed pipeline data is available for the pipeline view."
```

**The clause that says to IGNORE the forecast is what causes the forecast frame
to be loaded**, and with it sixty of the book's seventy-one columns disappear.

Three things about that failure are worth separating:

1. **It is not disclosed.** `field_missing` reads as a statement about the book.
   It is a statement about a projection the reader never asked for.
2. **D6 did not fix this.** D6 made the RECEIPT honest about availability. This
   fails earlier, at prepared-data validation, before any receipt exists. D6's
   `book_columns` says the field is there; the executor still cannot compute it,
   because the rows in front of it do not carry the column.
3. **The loss is silent in the other direction too.** Nothing tells the reader
   the answer was computed over a twelve-column projection.

### The lens has the same hazard, on rows

`_ACQUIRED_TERMS` contains the bare words `acquired`, `acquisition`, `purchased`,
`inorganic`; `_DIRECT_TERMS` contains `direct`, `organic`, `in-house`. So:

```
"What is the balance for loans purchased at auction?"           -> Acquired cohort
"...where the borrower acquired the property recently?"          -> Acquired cohort
"Show balance by region for directly held collateral"            -> Direct cohort
```

None of the three is about loan provenance. Each silently answers over one
cohort.

## 3. Why the row decision is more dangerous and less harmful

**More dangerous by construction.** Dropping rows changes the NUMBER and leaves
the answer looking complete: "the balance for loans purchased at auction" returns
a figure, correctly formatted, over the wrong population. Dropping columns tends
to produce a refusal, which is wrong but visible.

**Less harmful by accident.** The lens is *declared on the receipt* — the scope
appears in the execution summary and the answer's own prose — and it has a
comparison guard. The view is declared nowhere a reader would see, and has none.

This is the same accidental safety this programme has now declined to rely on
four times.

## 4. So: was D6 one instance?

**No.** D6 fixed one CONSEQUENCE of one of the two decisions — the receipt's
availability claim — and left both decisions and the data loss itself untouched.

What remains open, stated as three separable things rather than one:

* **B21** — the view resolver is a bare substring test with no incidental-mention
  guard, and the resulting data loss is presented as a book limitation.
* **B22** — the lens resolver has the same shape on rows, with bare single-word
  terms, and silently narrows the population.
* **B23** — neither decision is disclosed to the reader. A question answered over
  a twelve-column projection, or over one cohort, says so nowhere the reader
  looks.

**Not scheduled here, and deliberately not folded into D8.** All three are
changes to what data is put in front of the question, which is upstream of every
decision this programme has been consolidating; taking one inside a receipt
commit would mix a data-visibility change into an evidence change.
