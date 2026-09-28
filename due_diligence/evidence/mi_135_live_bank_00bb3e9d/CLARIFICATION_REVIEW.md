# Clarification review — MI 135 live bank, frozen product `00bb3e9d`

## The finding that governs this whole section

**The product authors no clarification question.** There is no
`clarification_question` field anywhere in the estate, and no harness change
could expose one, because the interpreter never produces a sentence to ask.
What it produces is a `CompileReason{code, subject, detail, spans}` and an
`Ambiguity{slot, note, options, blocking}`. Those are recorded verbatim below.

This bears directly on whether clarification surfacing should be built: today
there is nothing to surface — only reason codes and option lists to render.
The material is unusually good (each ambiguity names the slot, explains why it
blocks, and lists the governed alternatives), but turning it into a question a
lender would recognise is unbuilt work, not a display toggle.

## TOTAL_CLARIFY = 2 of 104 completed

Both fall in the `analytical_intent` family. Note the contrast with the
offline interpretation baseline (run8), which recorded 8 clarifications on the
same bank: clarification is markedly rarer against the live product.
That comparison is indicative only — 25 of 27 `analytical_intent` questions
are in the blocked set, so the live clarify count is measured on 2 of 27.

### Q1.1 — *How has the profile of our new lending changed over the last few months?*

- **CLARIFY_CLASS** = APPROPRIATE_AMBIGUITY
- **Reason codes**: `MODEL_FLAGGED_AMBIGUITY, MISSING_REQUIRED_SLOT, MISSING_REQUIRED_SLOT`
- **Exact interpreter output (verbatim, not paraphrased):**

  - slot `measures` — blocking: `True`
    > 'Profile' does not name a governed quantity or characteristic, and several readings would make Trakt do materially different work: the volume of new lending (pipeline amount / case count), the size and risk characteristics of the cases written (average advance, loan-to-value, borrower age), or the mix of the book by a dimension such as product type, amortisation type or region. I could not choose one, so the measure (and any accompanying grouping) is left empty. Please say which characteristic of new lending you mean.
    options: ['volume of new lending (pipeline amount or case count)', 'risk/size characteristics, e.g. loan-to-value, advance size, borrower age', 'mix by a dimension, e.g. product type, amortisation type, region']
  - slot `capability` — blocking: `False`
    > Read as a generic series over successive reporting periods for the pipeline population. If you meant purely the pipeline's own volume measures, the pipeline capability would own that instead — this follows from whichever measure is confirmed above.
    options: ['generic_analysis', 'pipeline']
  - slot `time` — blocking: `False`
    > 'The last few months' is read as a monthly series over recent governed reporting periods; the exact number of months is not stated and is left to the governed default.

  - `MODEL_FLAGGED_AMBIGUITY`: 'Profile' does not name a governed quantity or characteristic, and several readings would make Trakt do materially different work: the volume of new lending (pipeline amount / case count), the size and risk characteristics of the cases written (average advance, loan-to-value, borrower age), or the mix of the book by a dimension such as product type, amortisation type or region. I could not choose one, so the measure (and any accompanying grouping) is left empty. Please say which characteristic of new lending you mean.
  - `MISSING_REQUIRED_SLOT`: operation 'series' needs at least one measure
  - `MISSING_REQUIRED_SLOT`: operation 'series' needs a measure in every output

- **New/legacy behaviour**: LEGACY_FALLBACK (`CLARIFY_NOT_SERVED_IN_THIS_SLICE`) — the governed path does not serve a
  clarification in this slice, so the legacy path answered instead. The
  reader never saw the clarification.

### Q1.3 — *How does recent lending compare with what we were originating earlier in the year?*

- **CLARIFY_CLASS** = APPROPRIATE_AMBIGUITY
- **Reason codes**: `MODEL_FLAGGED_AMBIGUITY, MODEL_FLAGGED_AMBIGUITY, MODEL_FLAGGED_AMBIGUITY, MISSING_REQUIRED_SLOT, MISSING_REQUIRED_SLOT`
- **Exact interpreter output (verbatim, not paraphrased):**

  - slot `comparison` — blocking: `True`
    > The two sides of the comparison cannot be bound without the reader. 'Recent lending' versus 'what we were originating earlier in the year' could mean (a) a seasoning contrast inside the funded book — recently written loans (front_book) against more seasoned loans (back_book), (b) two explicit periods of this year held against each other, e.g. the most recent months against the opening months, or (c) the pipeline (new lending not yet funded) against loans originated earlier in the year. These execute materially different work, so no side has been asserted.
    options: ['seasoning contrast: front_book versus back_book within the funded population', 'two explicit calendar periods within the current year (recent months versus earlier months)', 'pipeline / new lending versus loans originated earlier in the year']
  - slot `measures` — blocking: `True`
    > 'Lending' names no governed quantity. It could be the amount advanced at origination (Original Balance), the current outstanding balance of those loans, or the number of loans written. No single reading is materially clear, so no measure has been asserted.
    options: ['original_principal_balance (amount advanced at origination), sum', 'current_outstanding_balance, sum', 'count of loans']
  - slot `time` — blocking: `True`
    > 'Earlier in the year' names no governed period boundary — where the year is cut between 'recent' and 'earlier' is not stated. Left unbound pending the comparison basis above.
  - slot `population.base` — blocking: `False`
    > Read as the funded book on the assumption that both sides concern loans actually written; if 'recent lending' means business still in the pipeline, this reading is wrong and is superseded by the comparison ambiguity above.
    options: ['funded', 'pipeline', 'whole_book']

  - `MODEL_FLAGGED_AMBIGUITY`: The two sides of the comparison cannot be bound without the reader. 'Recent lending' versus 'what we were originating earlier in the year' could mean (a) a seasoning contrast inside the funded book — recently written loans (front_book) against more seasoned loans (back_book), (b) two explicit periods of this year held against each other, e.g. the most recent months against the opening months, or (c) the pipeline (new lending not yet funded) against loans originated earlier in the year. These execute materially different work, so no side has been asserted.
  - `MODEL_FLAGGED_AMBIGUITY`: 'Lending' names no governed quantity. It could be the amount advanced at origination (Original Balance), the current outstanding balance of those loans, or the number of loans written. No single reading is materially clear, so no measure has been asserted.
  - `MODEL_FLAGGED_AMBIGUITY`: 'Earlier in the year' names no governed period boundary — where the year is cut between 'recent' and 'earlier' is not stated. Left unbound pending the comparison basis above.
  - `MISSING_REQUIRED_SLOT`: operation 'compare' needs at least one measure
  - `MISSING_REQUIRED_SLOT`: operation 'compare' needs a measure in every output

- **New/legacy behaviour**: LEGACY_FALLBACK (`CLARIFY_NOT_SERVED_IN_THIS_SLICE`) — the governed path does not serve a
  clarification in this slice, so the legacy path answered instead. The
  reader never saw the clarification.

## Adjudication

| | count |
|---|---:|
| APPROPRIATE_AMBIGUITY | 2 |
| UNNECESSARY_CLARIFICATION | 0 |
| PLAN_OR_CONNECTIVITY_GAP_MASQUERADING_AS_CLARIFY | 0 |

Both are genuine. Q1.1 turns on *"profile"*, which names no governed
quantity and admits three materially different pieces of work; Q1.3 asks a
comparison whose two sides, measure and period boundary are all unstated.
Neither would be answerable from governed defaults without choosing for the
reader, and in both cases the follow-up would ask for something genuinely
absent from the first question rather than restating it.

**If clarification surfacing were enabled today**, both of these would improve
the reader's experience: each currently falls through to a legacy answer that
silently picks one of the readings the interpreter refused to choose between.
Neither would expose an unnecessary question. That is 2 of 104 — a real but
small UX gain on present evidence, and the sample is too thin to generalise
while `analytical_intent` is 25/27 unmeasured.

**Recurring pattern**: both clarifications are triggered by an unbound
MEASURE (`MISSING_REQUIRED_SLOT: operation needs at least one measure`)
beneath a vague business noun — "profile", "lending". The measure slot is
where this product asks for help.

