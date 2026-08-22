# Value-domain resolution — what it would take

Reported, not built, per the contract. Measured on `28ece25` against the real
alderbridge tape (11,035 loans).

## Correction to the earlier statement

I previously reported that "no dimension in the registry declares a value
domain, and the only region list is a hard-coded literal in
`concentration_tests/matching.py:53`". The first half is true and the second is
misleading. **Value resolution does exist**, and `matching.py:53` is not the
mechanism the query path uses — it is a national statistical vocabulary used for
reading covenant documents.

The query path has **two** value-binding paths, and they work in opposite ways.

## Path A — `execution_receipt.geographic_values`: allowlist, from the frame

Profiles the loaded book at runtime and returns `{lowercased value: field key}`.
On the alderbridge tape it resolves **11 tokens**, all for
`collateral_geography` — *london, south east, south west, north east, north
west, east midlands, east of england, scotland, wales, …*

It is correct by construction: a token matches only if the book actually
contains it. It is gated three ways:

| Gate | Effect on this tape |
|---|---|
| field name must contain `geograph`, `region` or `collateral_geo` | **the binding constraint** — restricts it to geography |
| cardinality ≤ 60 | not binding — see below |
| token length ≥ 4, not a stopword | drops "ni", "uk" |

## Path B — `llm_query_parser._CATEGORICAL_FILTER_RE`: denylist, from the wording

`llm_query_parser.py:1820`. Extracts a geography value from a preposition idiom
— *in X*, *exposure to X*, *across X* — and validates it against two
**denylists**, `_CATEGORICAL_STOPWORDS` and `_NON_PLACE_TERMS`. Anything not on
a denylist is accepted as a region.

This is the path that fabricates:

```
"how much is in the good book"  ->  filters={'collateral_geography': 'Good'}
```

"good" appears on neither denylist, so it becomes a region. The comments record
two previous patches to the same denylist — *"when is it expected to complete"*
binding a geography called **Complete**, and *"for joint borrowers"* binding a
borrower predicate to the geography field. Each was a defect that added a term.

**A denylist cannot be completed.** Path A already holds the allowlist that
would settle every one of these cases, and Path B does not consult it.

## What the gate actually costs

Cardinality is not the obstacle. Of 56 registry dimensions, **25 are present in
this tape, and 24 of those have ≤ 13 distinct values** — far inside the existing
cap of 60:

| Field | Distinct values |
|---|---:|
| `account_status` | 2 |
| `seasoning_segment` | 2 |
| `origination_channel` | 2 |
| `portfolio_cohort` | 2 |
| `seasoning_bucket` | 4 |
| `original_ltv_bucket` | 5 |
| `ticket_bucket` | 7 |
| `age_bucket` | 8 |
| `ltv_bucket` | 9 |
| `collateral_geography` | **11 — the only one resolved today** |
| `vintage_year` | 13 |
| `occupancy_type`, `collateral_type`, `interest_rate_type`, … | 1 each |
| `geographic_region_*`, `*_itl3` | 172 |
| `postcode` | 7,118 |

So the profiling mechanism, unchanged, would cover **24 of the 25 dimensions
this book carries**. Only the ITL3 and postcode columns exceed the cap, and the
cap exists deliberately to keep a stray token from matching a code.

The three cases that fail today are all inside that 24:

| Question | Today | Field cardinality |
|---|---|---:|
| *balance where the account status is offer* | binds `account_status` as a **grouping**, discards the value | 2 |
| *balance where the occupancy type is buy to let* | binds `occupancy_type` as a **grouping**, discards the value | 1 |
| *balance for the back book* | binds `seasoning_segment` as a **grouping**, discards the value | 2 |

## What it would take

Four things, in dependency order. Sizes are relative, not estimates.

1. **Widen Path A's field-name gate.** The mechanism, the cardinality cap and
   the stopword guard all stay; only the `geograph|region|collateral_geo`
   substring test changes. Smallest change, largest coverage — 11 tokens to
   roughly 24 fields' worth.
2. **Make Path B consult Path A.** Replace the denylist check with a lookup
   against the resolved value map. This is what removes `'Good'`, and it removes
   the whole class rather than the instance — no future term needs adding.
3. **Decide the tape-shape question, which is the real client-facing work.**
   Profiling resolves what the book *stores*. A raw loan tape carries partial
   postcodes or free-text counties, not ITL codes, so "London" must reach
   whatever that book actually holds. That is a normalisation problem — postcode
   or county to region — sitting upstream of every gate above, and it is the
   only part of this that is genuinely new code rather than a widened
   constraint. It is also the part that is independent of this programme.
4. **Decide where an unresolvable value is reported.** With 1 and 2 done,
   *account status is offer* resolves the field and the value; *the good book*
   resolves the field and **fails** the value. Under this contract that is
   `dimensions[].role = filter` with the value slot filled-but-unresolvable —
   a linguistic claim the interpretation layer can make and a semantic
   conclusion `ResolvedConcept` owns.

Items 1, 2 and 4 are constraint changes inside machinery that already exists and
already works. Item 3 is the client-facing requirement and is properly separate
from this programme, as the contract says.

## Not built

Nothing above has been implemented. This is a report.
