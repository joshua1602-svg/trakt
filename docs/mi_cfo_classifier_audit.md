# CFO acceptance classifier audit — independent re-adjudication

**Read-only.** No production code, test, bank entry, frozen verdict or existing
acceptance result was modified. Sources: `CFO_ACCEPTANCE_BANK.yaml` (frozen
before execution) and `CFO_ACCEPTANCE_RESULTS.json`, re-adjudicated from the
question text and the response evidence, not from the existing label.

---

## A. Delivered audit

```
CORRECT_DELIVERED originally     63
EXACT                            51
DISCLOSED ASSUMPTION              5
WRONG / SILENT SUBSTITUTION       7
```

**The original classifier was too generous. It graded on "did it deliver and
mention the right words", never on whether the delivered series, period or
figure was the one requested.** Seven answers it passed are answers to a
different question.

### The seven

| # | question | requested | actual | defect class | P0? |
|---|---|---|---|---|---|
| 1 | "Show average balance over time." | measure = **average** balance, monthly | **total** funded balance — the series is byte-identical to "Show funded balance over time" (112.8m, 121.5m, 132.8m …); average balance is ~£269k | **MEASURE SUBSTITUTION**, silent | **yes** |
| 2 | "Show the balance bridge for last month." | period = **last month** (2026-05 → 2026-06) | **2026-02 → 2026-06**, a five-month bridge, +£59.2m instead of +£22.6m | **PERIOD SUBSTITUTION** | **yes** |
| 3 | "What drove the movement in the book last month?" | same | same five-month bridge | **PERIOD SUBSTITUTION** | **yes** |
| 4 | "At the current run rate, when do we reach £250m of loans?" | date at which £250m is reached | *"The book has already reached £250.0m (current funded balance £172.1m)"* — self-contradictory and false; the milestone table stops at £75m | **FALSE STATEMENT** | **yes** |
| 5 | "Summarise the portfolio." | weighted-average interest rate | **0.06%**; true value **6.2644%** — the ratio rendered with a % sign, wrong by a factor of 100, in the flagship CFO summary | **UNIT / SCALING DEFECT**, silent | **yes** |
| 6 | "Give me a portfolio overview." | same | same figure, same defect | **UNIT / SCALING DEFECT**, silent | **yes** (same root cause as 5) |
| 7 | "What is the pipeline balance?" | pipeline population | figure is correct (£3.6m over 8 cases) but described as *"entire funded portfolio"* | **POPULATION MISLABEL** | no — P1 |

Items 2 and 3 state their actual periods in the prose, so a careful reader can
see the substitution; they are graded WRONG because "last month" is explicitly
supplied by the user and was replaced, which the audit rules place outside
DISCLOSED ASSUMPTION. Items 1, 4, 5 and 6 are **silent**: nothing in the answer,
the warnings or the artifacts says the delivered thing is not the requested one.

### The five disclosed assumptions

"What is the average interest rate on the book?" (weighted-average, stated);
"Which broker channel has the largest balance?" and "Which product type has the
largest share of the book?" (the ordered breakdown is delivered — the share
column is present — but the narration does not name the winner); "Show the
largest 10 loan exposures." (ten ranked rows delivered; the prose summarises the
top five); "How many cases are in the pipeline?" (correct count, no population
stated).

### A limitation of my own export, disclosed

`CFO_ACCEPTANCE_RESULTS.json` records artifact `rows` but not `kpis`, so every
single-figure KPI answer shows `max_rows: 0`. Checked live, the figures are
present (£172.1MM total balance; £269K average balance). Those answers are
genuinely delivered — the zero is my exporter, not the product.

## B. Refusal audit

```
HONEST_REFUSAL originally   25
TRUE SAFE REFUSAL           14
FALSE REFUSAL               11
```

Every refusal is safe — none returns a substituted figure. Eleven are
unnecessary, by family:

| family | false refusals | evidence the capability exists |
|---|---|---|
| concentration | 4 — "Show product concentration", "Show broker concentration", "What share of the book is drawdown?", "What proportion of the book is in the acquired portfolio?" | product concentration computes in the workflow suite; "What proportion of the book is in London?" delivers a share |
| pipeline | 3 — "Show the pipeline by stage", "How has the pipeline evolved?", "What is the value of outstanding offers?" | "Show pipeline evolution by stage" **delivers** from the same fixture |
| ranking | 2 — "Which region has the largest/smallest balance?" | routed to `geo_exposure`, which wants ITL3/postcode, while "Show balance by region" delivers seven groups |
| comparisons | 1 — "How does the current month compare with the previous month?" | "Compare this month with last month" delivers |
| filters | 1 — "How many drawdown loans do we have?" | `product_type` is on the funded tape; read as a pipeline question |

**True safe refusals include the two filtered-ranked-movement questions** — see
§C. One true refusal gives a **wrong reason**: "How much is in the Highgate
Mortgages book?" refuses with *"No loans match that filter
(geographic_region_obligor)"*, having read a portfolio name as a geography
value. Safe, but the explanation misleads. P1.

## C. Reconciliation of the targeted capability work

The bank is **not stale**: the results were generated at 13:02 UTC, thirteen
minutes after the last code commit (12:49 UTC), and the refusals reproduce on a
live re-test now.

| capability | proven where | live `/mi/query` |
|---|---|---|
| leading filter clauses | contract + live | **delivered** — both forms identical |
| broker channel | live, with `asset_class` supplied | **delivered** — ranked movement by broker channel |
| concentration | workflow suite | **partly** — origination channel delivers; product and broker refuse |
| **filtered ranked movement** | contract composition, and executed **through the instrument's own plan executor** in `c7_target_plan_proof` | **not delivered** |

Conclusion for filtered ranked movement: **case (2) — the live path still does
not exercise the proven capability.** The contract carries
`current_loan_to_value gt 50.0` and `ordering_of=movement`, and the live route
answers *"this governed capability does not apply a value threshold"*. My
earlier RM3 non-vacuous execution ran through the instrument's executor, not
through the shipped `period_change` route, and the earlier reporting did not
draw that line clearly enough.

## D. Launch-blocker set

Silent wrong-answer P0s — **six questions, five distinct defects**:

1. **Average-over-time measure substitution** — "Show average balance over time"
   returns total funded balance.
2. **Bridge period substitution** — "for last month" returns a five-month
   bridge (two questions).
3. **Forecast milestone false statement** — "already reached £250.0m" at
   £172.1m.
4. **Weighted-average interest rate wrong by 100×** — 0.06% for 6.26%, in the
   portfolio summary and overview (two questions).
5. **Forward-horizon substitution** — the previously reported five-year
   forecast, which returns a pipeline-horizon figure with no disclosure.

Not included: the pipeline population mislabel (figure correct), the eleven
false refusals (coverage, not safety), and the Highgate wrong-reason refusal.
None of the eleven false refusals breaks a core family outright, but
concentration loses four of seven and would be thin in production.

## E. Recommendation

**No — the five-year forecast is not the only P0.** Four further silent
substitutions are present in answers previously graded CORRECT_DELIVERED, and
one of them — the weighted-average interest rate reported as 0.06% instead of
6.26% — is a wrong number on the flagship portfolio summary, which is the single
most likely first question a CFO asks.

The go-live verdict does not change (**NOT COMMERCIAL GO-LIVE READY**), but the
blocker count moves from one to five distinct defects, and the true silent-error
rate over the bank is **10 of 91 (11.0%)**, not 3 of 91.
