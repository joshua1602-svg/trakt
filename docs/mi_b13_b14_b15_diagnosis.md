# B13, B14, B15 — one cause, three presentations

Diagnosis only. Nothing is fixed here and no fix is proposed.

Base: merge-base `4e051f3`; `4e051f3` and `28ece25` both ancestors of HEAD
(`5e76f68`).

---

## The answer

**Yes, they share a cause.** They are not three defects; they are one defect
observed from three angles.

> The ROLE of a seasoning phrase — population or axis — is decided by three
> independent readers, which disagree on the vocabulary they read, on when they
> run, and on whether they consult the columns the book actually has.

| | reader 1 — population injection | reader 2 — dimension terms | reader 3 — analytical intent |
|---|---|---|---|
| where | `llm_query_parser`, after the parse | `execution_receipt.requested_dimension_terms` | `mi_workflows/analytical/*` |
| vocabulary | `_SEGMENT_PHRASES` — front/back book, new origination, newly originated, seasoned book | the registry's dimension vocabulary | `_LENDING_PHRASES` — new lending, new business, new loans, recent lending |
| runs | on every parse | on every receipt build, from the QUESTION TEXT | only when the plan is composite |
| consults available columns | **yes** — declines if the book lacks `seasoning_segment` | **no** | n/a |
| produces | a `spec.filters` entry, and strips the field from `spec.dimension`/`dimensions` | a `KIND_GROUPING` facet | a governed lending window |

Three readers, two vocabularies, one concept. Every one of the three defects is
a different pair of them disagreeing.

---

## B13 — "the balance of new lending" answers over the whole book

*Reader 1's vocabulary does not contain the phrase; reader 3's does but does not
run; reader 2 fires and makes it an axis.*

`"new lending"` is in `_LENDING_PHRASES`, not `_SEGMENT_PHRASES`, so
`resolve_segment_population` returns `None` and no filter is injected. That is
deliberate — the source says so:

> the ruling is explicit that "lending" carries a role that depends on
> analytical context … The role decision is taken by the analytical intent
> layer, which is the only place that context exists.

But the analytical layer runs only when `plan.is_composite`. *"What is the
balance of new lending?"* is not composite, so no route claims it, it falls to
point-in-time, and **the layer that owns the role decision never runs.** Reader 2
then raises a grouping from the question text, and the answer is a two-bar
breakdown of 11,035 loans.

Measured, one word apart on the same route:

```
"balance of the front book"   filter seasoning_segment='Front Book' kept 1177/11035
"balance of new lending"      (no filter warning at all)
```

**The role decision was delegated to a layer that does not run for simple
questions, and nothing covers the gap.**

## B15 — the same field both applied and lost in one receipt

*Readers 1 and 2 both fire, and reader 1's cleanup cannot reach reader 2.*

`"back book"` IS in `_SEGMENT_PHRASES`. Reader 1 injects the filter and then
does exactly the right thing:

```python
# The segment named the population, so it is not also the axis.
for attr in ("dimensions", "hierarchy"): ...remove seasoning_segment...
if spec.dimension == SEASONING_SEGMENT_FIELD: spec.dimension = None
```

It strips the field from the **spec**. But reader 2 does not read the spec — it
reads the **question text**, which still says "back book". So the grouping facet
is raised anyway, and the receipt carries `row_population: applied` from the
injected filter and `grouping_dimension: lost` from the text, for one field.

**A cleanup that edits the spec cannot reach a reader that never consults it.**

## B14 — "the field is not available" on a tape that carries it

*Reader 1 honours the column set; reader 2 does not; and the column set belongs
to a route the question never took.*

```
"Forecast the balance of the front book…"  cols_given=True  seasoning_in_cols=FALSE
"What is the balance of the front book?"   cols_given=True  seasoning_in_cols=TRUE
```

Same tape, same process, same phrase — different column sets. The forecast
question is run-scoped, so the service loads the pipeline/history frame rather
than the funded tape, and that frame has no `seasoning_segment`.

Reader 1's guard then correctly declines:

```python
if available and SEASONING_SEGMENT_FIELD not in available:
    return None          # book carries no seasoning: leave it
```

Reader 2 has no such guard, so `dimension = seasoning_segment` survives onto a
spec that fails validation, and the reader is told *"'Seasoning Segment' is not
available in this dataset"* — true of the frame that was loaded, false of the
book they asked about, and unactionable either way.

**Two readers, one guarded and one not, over a frame chosen for a route the
question did not take.**

---

## Why this matters beyond the three

This is the founding finding of the programme, recurring in a specific
vocabulary. The inventory recorded that **eleven entry points read the raw
question** and that the one real structural conflict is `KIND_GROUPING`
conflating a grouping axis with a filter on a named dimension. Stage 4 resolved
that conflation *for the point-in-time spec*, via
`_split_named_dimension_roles`.

It did not resolve it for the seasoning vocabulary, because the seasoning role
decision does not live in the spec at all — it lives in reader 1, before the
spec is validated, by design ("a role decision, taken once, before the spec is
validated"). Stage 4's split reads `spec.filters` and `spec.dimensions`; readers
1 and 2 both operate outside that.

So the count is now: **the seasoning concept resolves correctly in every
vocabulary it has, and has since the first traces. What fails is that three
readers each own part of the decision and none owns the whole of it.** Every
carriage failure recorded in this programme — point-in-time, evolution, the
run-rate path, and now these three — is a different pair of readers disagreeing.

## What a fix has to satisfy — stated as constraints, not as a design

Recorded so the fix is scoped against them rather than improvised:

1. **One reader decides the role**, for both vocabularies, or the drift returns
   in the next wording added to either.
2. It must run **for simple questions as well as composite ones**, or B13
   recurs for every phrase the analytical layer owns.
3. Its decision must reach **reader 2**, which today consults neither the spec
   nor the column set — so either reader 2 reads the decision, or reader 2 stops
   reading seasoning wording independently.
4. It must be **frame-aware in one place**: the same phrase must not resolve
   differently because a route-scoped frame was loaded.
5. The comparison arm of the receipt already resolves both vocabularies
   correctly via `_governed_population_predicates` and must not be duplicated —
   that is a fourth reader waiting to happen.

Constraint 3 is the one that decides the shape, and it is the same conclusion
the programme reached for the lexical owner: **the remedy for several readers
that agree by maintenance is one reader, not a better copy.**

---

## Not fixed here

No code changed. B12 remains next after this, per the revised order:
B13/B14/B15 → B12 → B10 → B9 → segmented series.
