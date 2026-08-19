# P1N — Targeted Statistic Breadth

**Scope:** four additions only — exposure-weighted borrower age, exposure-weighted months on
book, generic MIN, generic MAX. No other statistic or registry expansion.
**Baseline:** `abb9cad` (P1M Statistic Identity: PASS).

---

## 1. Executive verdict

All four capabilities are in, governed rather than special-cased, and every figure reconciles
to independently computed truth at zero variance. The P1M invariant is intact: the new
statistics are protected by the same identity check as the old ones, and median LTV still
refuses.

Nothing was widened beyond the ruling. Both regression banks show **zero changed answers**
against P1M — the commercial bank on both parser paths, and the immutable 40-question bank.

Two things the work surfaced are worth the reader's attention up front:

- A superlative has **two** governed readings, and P1N had to keep them apart. "Largest loan
  balance" is an established loan-level ranking *table*; "maximum loan balance" is the new
  extreme-value *statistic*. My first cut collapsed the first into the second and deleted a
  working capability — caught by the existing ranking and calibration suites, not by my own.
- The registry change shifted which averaging statistic the model chose for LTV, so "average
  LTV in London" came back as a simple mean while the whole-book question stayed weighted.
  That is now normalised (§4).

## 2. Exact capabilities added

| # | Capability | Where it lives |
|---|---|---|
| 1 | `min` as a governed statistic of a measure | aggregation contract → validation → registry permission → executor → P1M facet → receipt |
| 2 | `max` as a governed statistic of a measure | same path |
| 3 | Exposure-weighted borrower age | registry permission only — executor and weighting already existed |
| 4 | Exposure-weighted months on book (portfolio seasoning) | same |

No new engine, parser, ledger or specialist route. `aggregate_series` gained two branches;
everything else is registry permission and vocabulary.

## 3. Measures permitted for min/max

Added by inference rule, not by hand-listing, so the permission follows the measure's *kind*:

| Metric format | min/max? | Reasoning |
|---|---|---|
| currency (balance, principal, valuation, arrears, losses, EAD …) | **yes** | "The largest loan" is a real business figure |
| percent (LTV, interest rate, margin, indexed LTV, protected equity) | **yes** | "The highest LTV" is a real business figure |
| integer — ages, days, terms, months on book | **yes** | "The oldest borrower", "the most seasoned loan" |
| integer — `number_of_*` count metrics | **no** | "The maximum number of properties" is a curiosity, not MI |
| dimensions, flags, dates, identifiers | **no** | An extreme has no coherent reading |
| synthetic `loan_count` | **no** | Not a registry field; a count has no extreme of its own |

**30 metric entries** gained min/max. Verified by test that `number_of_leased_objects`,
`number_of_properties_at_data_cut_off_date`, `collateral_geography`, `account_status` and
`origination_date` did **not**.

Registry changes were made in the build script's CURATION and regenerated — the YAML is
auto-generated and carries a "do not edit by hand" header.

## 4. Weighted-average definitions

Applies to `youngest_borrower_age` and `months_on_book`. No new methodology was created.

| | |
|---|---|
| **Weighting field** | `current_outstanding_balance` |
| **Numerator** | `sum(measure × current_outstanding_balance)` over the resolved population |
| **Denominator** | `sum(current_outstanding_balance)` over the same population |
| **Null treatment** | rows where either the measure or the weight is null are excluded from **both** numerator and denominator (`vals.notna() & w.notna()`) |
| **Population denominator** | after all filters, governed populations and portfolio scopes — the same frame every other statistic sees |
| **Zero-weight guard** | a zero denominator returns NaN rather than a fabricated figure |

`current_outstanding_balance` is the field the existing mechanism already resolves: it is the
registry's `metadata.default_weight_field`, the entry weight for every measure that already had
a weighted average (LTV, interest rate), and position 4 in the executor's own
`resolve_weight_field` fallback hierarchy. It is now **pinned** on these two entries because
validation requires an explicit weight before permitting a weighted average — the value is
unchanged, only its implicitness.

**`default_aggregation` stays `avg` on both.** "Average borrower age" must keep meaning the
simple mean; the weighted figure is reached only by asking for it. Verified by test.

The receipt names the weighting so a reader is not led to believe the answer is a simple mean:

```
Calculated: Weighted-average Borrower Age · entire funded portfolio · 11,035 loans
Calculated: Weighted-average Months on Book · entire funded portfolio · 11,035 loans
```

### A related correction

P1M deliberately relaxed identity so a plain "average" is satisfied by *either* governed
averaging statistic — the field decides which. The commercial bank showed the cost of leaving
that to the parser: **"average LTV in London" returned the simple mean 39.6193 while the
whole-book question returned the governed weighted average 43.1562** — one phrasing, two
statistics, 7% apart, depending on what the model emitted. A bare mean request now normalises
to the field's `default_aggregation`, making the house convention deterministic on both paths.
Measures whose default *is* the simple mean are untouched.

## 5. Composition results

Every population proven by its row count, not claimed.

| Question | Result | Population |
|---|---|---|
| Maximum loan balance | £841,638.96 | 11,035 |
| Minimum loan balance | £0.00 | 11,035 |
| Maximum / minimum LTV | 104.563797 / 0.000000 | 11,035 |
| Maximum / minimum borrower age | 96 / 52 | 11,035 |
| Maximum / minimum valuation | £1,536,316.32 / £95,085.50 | 11,035 |
| Maximum / minimum months on book | 149 / 0 | 11,035 |
| **Max balance for borrowers over 85** | £794,856.41 | **86** |
| **Max LTV for loans over £500k** | 79.558656 | **119** |
| **Max LTV in the back book** | 104.563797 | **9,858** |
| **Max balance in the back book** | £841,638.96 | **9,858** |
| **Max balance in the acquired book** | £684,845.61 | **3,909** |
| **Min borrower age in the direct book** | 52 | **7,126** |
| **Exposure-weighted borrower age** | 72.512509 | 11,035 |
| **Exposure-weighted borrower age, back book** | 72.848625 | **9,858** |
| **Exposure-weighted months on book** | 59.441874 | 11,035 |
| **Exposure-weighted months on book, acquired** | 85.590972 | **3,909** |

P1L holds: no population widened. The exposure-weighted figures are materially different from
their simple means (72.51 vs 71.40 for age; 59.44 vs the unweighted mean), which is the point
of adding them.

### The threshold that must not become the measure (brief §10)

"Max LTV for loans over £500k" resolves to `current_outstanding_balance > 500000` — 119 loans —
with the measure still LTV. The currency marker is what carries it: an **unqualified** number
("loans over 500000") attaches to the requested measure, matches nothing, and **refuses**
rather than widening. That is pre-existing behaviour, it fails safe, and both cases are pinned
by test.

One phrasing asymmetry is recorded rather than fixed: "**highest** LTV for loans over £500k"
loses the threshold and refuses, while "**maximum** LTV for loans over £500k" answers exactly.
Fixing threshold attachment is parser work outside the four additions.

### Joint borrowers (brief §6)

**Data/concept blocked.** The ERE fixture carries only `borrower_identifier` and
`youngest_borrower_age`. `borrower_structure` and `number_of_borrowers` exist in the registry
but are **absent from the tape**, so "maximum loan value for joint borrowers where the youngest
is above 85" cannot be tested. No borrower-structure data or semantics were invented. Generic
MAX is proved instead with the age half of that question — max balance for borrowers over 85 =
**£794,856.41 over 86 loans**.

## 6. Independent truth

Every figure in §5 recomputed with pandas from the fixture; the MI executor was not used as its
own oracle. **26 of 26 acceptance cases at zero variance**, populations exact. Two cases are
expected refusals (median LTV; the unqualified numeric threshold) and returned no figure.

## 7. Genuine-LLM results

Live API, 5 runs per case, provenance captured at the parse seam.

| Case | Distinct outcomes | Provenance | Verdict |
|---|---|---|---|
| Maximum loan balance | **1 of 5** | `llm` ×5 | PASS — £841,638.96 |
| Max LTV in the back book | **1 of 5** | `llm` ×5 | PASS — 104.563797, 9,858 rows |
| Max LTV for loans over £500k | **1 of 5** | `llm` ×5 | PASS — 79.558656, 119 rows |
| Exposure-weighted borrower age | **1 of 5** | `llm` ×5 | PASS — 72.512509 |
| Exposure-weighted months on book | **1 of 5** | `llm` ×5 | PASS — 59.441874 |
| Max balance in the acquired book | **1 of 5** | `llm` ×5 | PASS — £684,845.61, 3,909 rows |
| Median LTV (must refuse) | **1 of 5** | `validation_failed` ×5 | PASS — refused, no KPI |

**Gate: GREEN.** 30 genuine model calls. Every case fully deterministic across repeats.

## 8. P1M regression

| Case | Result |
|---|---|
| Median LTV | **still refuses**, naming the statistic |
| Median loan balance | 156,864.66 — unchanged |
| Median borrower age | 71.0000 — unchanged |
| Average LTV | governed weighted average 43.156246 — unchanged |
| Total balance | 1,964,886,258.21 — sum, unchanged |
| Loan count | 11,035 — count, unchanged |

Statistic identity holds for the new statistics: requested `max` vs executed `avg` / `sum` /
`weighted_avg` / `min`, requested `min` vs executed `avg` / `max`, and requested `weighted_avg`
vs executed `avg` / `median` are all rejected. Tested directly.

## 9. P-gate regression

```
P0 cohort identity · P1C ranked movement · P1D contribution · P1E golden bank ·
P1E measure safety · P1E multi-measure · P1F exposure · P1G measure identity ·
P1I scope · P1J-1 seasoning · P1L population · P1M statistic identity ·
P1N statistic breadth · full mi_agent suite

1,722 passed, 1 skipped, 21 xfailed
```

`tests/test_p1n_statistic_breadth.py` adds **49** tests.

Two existing tests were updated where a **ruling** changed their premise, not their invariant:

| Test | Why | Change |
|---|---|---|
| `test_a_metric_with_no_governed_weighted_average_is_rejected` (P1D) | used borrower age as the measure with no weighted average — P1N grants it one by ruling | moved to balance, where weighting a measure by itself stays degenerate |
| `test_extreme_value_questions_keep_their_existing_governed_behaviour` (P1M) | pinned "maximum LTV" as a refusal, correct when no statistic could express an extreme | now asserts the governed extreme reconciles to the fixture |

Neither weakens a refusal; both record the reasoning in place.

## 10. Commercial Beta bank

| Outcome | Deterministic before → after | Production before → after |
|---|---|---|
| CORRECT | 29 → **29** | 30 → **30** |
| SAFE_REFUSAL | 5 → **5** | 4 → **4** |
| **INCORRECT_SUCCESSFUL** | 0 → **0** | 0 → **0** |
| **SILENT_SEMANTIC_ERROR** | 0 → **0** | 0 → **0** |
| **HARD_FAILURE** | 0 → **0** | 0 → **0** |

**Zero changed answers on both paths.** (An interim run showed one change — C27's simple-mean
LTV — which is the defect corrected in §4; the final run is clean.)

## 11. 40-question bank

| | Before | After |
|---|---|---|
| Answered | **14 / 40** | **14 / 40** |
| Changed answers | — | **0** |

No churn. The bank was not touched and this phase is not justified by it.

## 12. Full repository suite

See §14 — reported from the definitive run.

## 13. Remaining statistic decisions

1. **Median for ratio/rate measures** (LTV, interest rate, margin, indexed LTV, protected
   equity) remains a **product decision**, unchanged by P1N. The three candidate quantities are
   materially different on this book: simple median of loan-level LTVs **39.675692**,
   exposure-weighted median **42.465496**, governed weighted-average **43.156246**. Until a
   methodology is agreed these refuse.
2. **Weighted-average valuation and balance** were explicitly ruled out and are not added.
   `current_valuation_amount`, `original_valuation_amount`, `original_principal_balance` and
   `current_outstanding_balance` remain executable-but-not-governed for a weighted average.
3. **Percentile, quartile, standard deviation, variance, spread** remain outside the governed
   statistic set and are not recognised.
4. **"Oldest" / "youngest"** are deliberately not statistic vocabulary. The measure is literally
   named `youngest_borrower_age`, so "the youngest borrower age" is the field's name rather than
   a statistic over it, and "oldest borrower" would mean the maximum of a field whose own name
   says youngest. Reported rather than guessed; "maximum/minimum borrower age" both work.
5. **Threshold attachment for "highest … £500k"** (§5) — a parser matter, not a statistic one.

---

P1N TARGETED STATISTIC BREADTH: PENDING FULL SUITE
