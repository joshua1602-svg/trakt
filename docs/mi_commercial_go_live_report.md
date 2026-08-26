# MI Agent — commercial go-live report

Frozen baseline `0af2d9f` (tagged `mi-launch-baseline-7of7`). Launch work:
`4ca4320`, `1601332`, `b06c17f`, `8910770`.

---

## A. 7/7 frozen core

The baseline was independently re-verified at its HEAD before any change, and
again after all of it.

| | at freeze | after launch work |
|---|---|---|
| post-claim raw-question semantic decision sites | 0 | **0** (across all eight generic paths) |
| route-local semantic vocabularies | 0 | **0** |
| fail-closed detector | 0 of 2 substituted | **0 of 2** |
| independent audit | 10 / 10 | **10 / 10** |
| frozen canary | 0 invariant breaches | **0 breaches**, history untouched |
| unexplained route movements | — | **0** |
| unexplained economic movements | — | **0** |

Live blast across all three closures: **882 corpus questions through
`/mi/query` in both trees against one fixture — 0 questions changed, 0 route
movements**, 381 `ok=True` and 932 artifacts either side. Contract blast: 0 of
882.

## B. Launch defect closure

### Leading filter clause — **CLOSED**

*Root cause, measured, and worse than reported.* Punctuation was not a clause
boundary, so a condition stated FIRST had no end and ran to the end of the
sentence; the field was then resolved from the words it swallowed. Compounding
it, the condition span ended inside its own opener ("for loans" + "with"), and
`metric_slot` truncated at the condition's start — right when the condition is
last, wrong when it is first, because then the subject is what follows.

    "For loans with LTV above 50%, balance by region"
        metric = LTV, filter = balance > 50
    "For loans with borrower age above 70, balance by region"
        filter = balance > 70,000,000,000

*Fix.* Punctuation is a boundary (a comma with digits on both sides is still a
thousands separator); the condition clause ends at the first boundary after its
numeric bound; `metric_slot` removes the clause instead of truncating at it. No
field names, no second resolver, one owner.

*Blast.* 0 of 882 contracts, 0 of 882 live answers. 4 of 15 target questions
changed — exactly the four defect cases, each toward its trailing equivalent.
14 tests; 8 fail against the frozen baseline and the 6 controls pass in both.

### broker_channel — **CORRECTLY GOVERNED, CARRIAGE FIXED**

*Not an MI defect.* The field exists with usable values; the BSR restricts it to
`equity_release`; the asset class is decided by onboarding and carried by the
governed portfolio registry. Diagnosed across three contexts:

| asset class | ranked movement by broker channel |
|---|---|
| unidentified | **refused by name** |
| `equity_release` | **answered and ranked** |
| `residential_mortgage` | **refused by name** |

Point-in-time questions over the same field answer in all three. The gap was
carriage: the example registry a client copies documented nine fields and not
`asset_class`. **No production code changed; applicability was not weakened.**

### concentration_analysis — **INTERPRETATION RESIDUAL CLOSED**

*Before.* `run_concentration_analysis` read the question three times after the
claim — analytical concept, single-name framing, and a portfolio lens whenever
no scope arrived — behind eleven vocabularies of its own; the adapter added a
fourth read.

*After.* The recogniser reads once, pre-claim, and the reading travels; the
scope comes from the contract; a workflow handed no reading **refuses** rather
than reading the sentence. The deterministic calculation — shares, cumulative
shares, ranks, the governed denominator, the currency and scale guards — is
untouched.

*Evidence.* Denominator proven (2,200 book, London 900 = 40.9%), ranking by
share proven, five mutation controls move the published numbers, refusals
preserved. The structural guard fails against the pre-change tree naming the
call it now forbids.

### Two regressions the MI regression caught — **FIXED, NOT EXPLAINED AWAY**

The authoritative run reported exactly two introduced failures. Both were mine.

1. **A silent scope widening I introduced.** `lens_from_contract` returns None
   for every state except FILLED, and the concentration adapter read that None
   as "no scope" — the whole book. "Show product concentration for
   acquired_009" stopped being a controlled failure naming that book and became
   an answer over the entire portfolio. That is the P1L defect class this estate
   removed in Phase 1E, and I put one back. `requested_context_id` now returns
   the name the reader used whether or not it resolves.
2. **The plan module must not import a question resolver**, and putting the
   contract-to-lens mapping in `analytical_plan` made it do exactly that. The
   guard was right; the mapping moved to its own module.

Neither test was adapted to accommodate the regression.

## C. MI-only regression (authoritative)

Denominator: `migration_phase0/MI_REGRESSION_MANIFEST.txt`, **278 modules**,
decided from the import graph. OCC, onboarding, Annex 2, regulatory XML, mail
and demo-platform suites are **outside** it and play no part in this verdict.

| | baseline | final |
|---|---|---|
| modules | 278 | 278 |
| tests executed | 6768 | 6768 |
| passed | 5957 | 5957 |
| failed | 81 | 81 |
| skipped | 711 | 711 |
| xfailed / xpassed | 15 / 0 | 15 / 0 |
| errors | 4 | 4 |
| timeouts | 1 | 1 |
| failing/erroring names | 85 | 85 |

```
INTRODUCED     0
FIXED/REMOVED  0
```

## D. CFO acceptance bank

91 questions, frozen before execution, ten families, expectations written from
the question and the governed data.

```
Total questions              91
Correct delivered            63
Honest governed refusal      25
Wrong / silently incomplete   3
Delivery rate             69.2%   (82.9% of the 76 expected to deliver)
Silent-error rate          3.3%
```

| family | n | correct | refusal | wrong |
|---|---|---|---|---|
| size | 8 | 8 | 0 | 0 |
| composition | 10 | 10 | 0 | 0 |
| trends | 10 | 10 | 0 | 0 |
| comparisons | 6 | 5 | 1 | 0 |
| filters | 12 | 11 | 1 | 0 |
| ranking | 12 | 8 | 4 | 0 |
| concentration | 7 | 3 | 4 | 0 |
| pipeline | 6 | 3 | 3 | 0 |
| specialist | 5 | 5 | 0 | 0 |
| insufficiency | 15 | 0 | 12 | **3** |

**Every critical family has a silent-error rate of 0.** All three wrong answers
are in the insufficiency family — questions the bank expected to be declined.

A classifier correction, recorded: the first run scored 2 delivered answers as
refusals because bare markers fired inside them ("unavailable" in a limits tally
of "3 unavailable"; "no governed" in the standard materiality disclosure). The
markers were tightened to sentence-anchored decline phrases. That removes false
refusals only — it cannot turn a silent wrong answer into a pass, because a
silent wrong answer carries no decline phrase either way.

### The three, examined individually

| question | disclosed? | assessment |
|---|---|---|
| "What changed?" | **yes** — names both dates, the metrics compared, and warns it reports a policy-selected subset | disclosed convention; the bank's REFUSE expectation was strict |
| "Show me the trend." | **partly** — the answer states "Funded balance over 5 period(s)", so the chosen measure and window are visible, but nothing says the question named no measure | weak disclosure |
| **"What will the book be worth in five years?"** | **no** — returns "Forecast funded balance: £173.4m" landing 2026-06 to 2026-08; the five-year horizon is neither honoured nor mentioned anywhere in answer or warnings | **genuine silent substitution — P0** |

## E. Adversarial wording

18 phrasings across 5 core intents. **16 identical governed meaning, 2 drifted —
and both drifted to an explicit refusal, not to a different answer.** Leading and
trailing filter forms are now identical, and five materially different phrasings
of ranked regional movement produce the same dimension, direction, basis, limit,
periods and leader.

The two drifts: "Give me the two biggest regional balance increases since last
month" (the term "regional" is not bound to the dimension) and "What moved in
the book compared with last month?" (the parser reads "Compared" as a geography
value). Both **disclose and refuse**. P1, not P0.

## F. Launch blocker triage

**P0 — launch blocker (1)**

* **Forward-horizon questions answer a short-horizon forecast without disclosing
  it.** "What will the book be worth in five years?" returns a pipeline-horizon
  figure as "Forecast funded balance" with no statement that five years was not
  honoured. A CFO could read £173.4m as the five-year answer. *Not fixed here:*
  the forward horizon is not carried by any governed contract field, so closing
  it needs a new semantic concept — outside this task's bounds, and Phase 10
  directs reporting rather than an unbounded fix.

**P1 — fix soon (safe refusals exist)**

* Filtered ranked movement refuses on the live route: the contract carries the
  predicate but `period_change` selects population by scope, not row predicate,
  so it discloses and declines. Costs the ranking family 2 of 12.
* Concentration by an explicitly named dimension ("show product concentration",
  "show broker concentration") discloses and declines. Costs 4 of 7.
* Pipeline stage and outstanding-offer questions do not map to a governed
  analytic on this data. Costs 3 of 6.
* "Which region has the largest/smallest balance?" routes to `geo_exposure`,
  which needs ITL3 or postcode. Costs 2 of 12 in ranking.
* Two adversarial phrasings above.
* "Show me the trend." chooses a measure; the choice is visible but unflagged.

**P2 — post-launch**

* "What changed?" applies a disclosed latest-versus-previous convention.
* "How many drawdown loans do we have?" is read as a pipeline question.

## G. Verdict

**NOT COMMERCIAL GO-LIVE READY.**

The single blocker: forward-horizon forecast questions return a short-horizon
figure without disclosing that the requested horizon was not honoured. Fixing
that one item — honour the horizon, or state plainly that it was not — changes
the verdict; every other gate is met.

* 7/7 baseline preserved — **yes**
* leading-filter defect closed — **yes**
* broker-channel behaviour correctly governed — **yes**
* concentration interpretation residual closed — **yes**
* authoritative MI regression, 0 unexplained introduced failures — **yes**
* CFO bank, 0 wrong/silently incomplete — **no (1 genuine, of 3 flagged)**
* delivery coverage adequate — 69.2% overall, 82.9% of the expected-deliver
  subset, every critical family at 0 silent errors
