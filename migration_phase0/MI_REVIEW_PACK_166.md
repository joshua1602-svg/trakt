# MI Query Agent — full review pack, 166 questions, both arms

Every question in the 75-question acceptance bank and the frozen CFO 91, with the model off and with the concept merge wired and on.

**What the merge column is, exactly.** The supplied key exhausted its credit, so the model is not called: the arm REPLAYS the proposals the model made in the measured Stage 4 run, and the deterministic estate then binds, merges and executes them as it would have. The PROPOSALS are byte-faithful to that run. The ANSWERS are not, and must not be read as if they were — code has shipped since Stage 4, and where it changed an answer this pack shows the current one. That is the point of regenerating it.

Whole book: **640 loans**. Sorted so the likely mis-grades come first.

## What to look at, in order

| bucket | what it is | count |
|---|---|---|
| `0a_` | **WRONG** — a figure that is not the truth, with a receipt. Read these first | **6** |
| `0b_` | The two arms DISAGREE on the grade — compare the pair | **5** |
| `1a_` | CORRECT, answered over the WHOLE BOOK, and the question NARROWS | **0** |
| `1b_` | CORRECT, answered over the whole book, and the question is a whole-book question | **31** |
| `2_n` | The question NAMES a narrowing and NO filter was applied | **5** |
| `3_q` | The answer quotes a figure the truth calculation did not produce | **0** |
| `4_k` | Known grader divergences | **3** |
| `5_r` | The rest | **116** |

## Surface 1 — a CORRECT answer covering more rows than the narrowing implies

**None.** No answer graded CORRECT covers more rows than its question's narrowing implies, on either arm. The 31 entries in bucket `1b_` are CORRECT answers over the whole book to questions that ask about the whole book — listed so they can be eyeballed, not because anything is wrong with them.

## Surface 2 — refusals whose stated reason is a claim about the client's data

A refusal that says something about what the book CONTAINS, rather than about what the system CAN DO, has to be true. 6 do this. They are not all the same thing, and only one class would be a lie:

| id | class | the claim |
|---|---|---|
| `CFO81` | TRUE_about_the_book | I understood that you asked for Highgate Mortgages book, but that could not be applied to the calculation (Highgate Mortgages book — 'Highgate Mortgag |
| `CFO82` | TRUE_about_the_book | 'Risk Grade' is not available in this dataset. This book does not report it, so the question cannot be answered from the current data (no value was fa |
| `CFO83` | TRUE_about_the_book | 'arrears' is not available in this dataset. This book does not report it, so the question cannot be answered from the current data (no value was fabri |
| `CFO85` | TRUE_about_the_book | 'NNEG' is not available in this dataset. This book does not report it, so the question cannot be answered from the current data (no value was fabricat |
| `Q15B` | QUOTES_A_MANGLED_PHRASE | I understood that you asked for Break Direct- book, but that could not be applied to the calculation (Break Direct- book — 'Break Direct- book' is not |
| `Q21C` | TRUE_but_about_a_filter_the_reader_never_asked_for | No loans in this book match that filter ('among'), so there is nothing to calculate. I have not returned a whole-book figure in its place. |



---

# **WRONG** — a figure that is not the truth, with a receipt. Read these first  (`0a_`)


## Q03A · BANK75

**Question as typed:** How many drawdown loans have LTV above 50%?

**Truth:** desc=count, drawdown AND ltv>50, count=45

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `WRONG`**
population: **144** of 640 (whole book 640)
measure `None` · aggregation `count` · narrowed `True`
filters applied: `['Current LTV > 50']`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 50` → applied

> 144 loans · Current Outstanding Balance: £37.1MM.
> 
> Calculated: Count of loans · Current LTV > 50 · 144 loans · as at 30 June 2026.

grade reason: count 45 ABSENT

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **45** of 640 (whole book 640)
measure `None` · aggregation `count` · narrowed `True`
filters applied: `['Current LTV > 50', 'Product Type = drawdown']`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}, "erm_product_type": "drawdown"}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 50` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'filter', 'field': 'erm_product_type', 'operator': None, 'value': 'drawdown'}] · conflicts 0

> 45 loans · Current Outstanding Balance: £11.3MM.
> 
> Calculated: Count of loans · Current LTV > 50 · Product Type = drawdown · 45 loans · as at 30 June 2026.

grade reason: count 45 found


## Q04C · BANK75

**Question as typed:** Show total outstanding balance for London loans in the Direct book with borrower age above 75.

**Truth:** desc=balance, Direct book AND London AND age>75, count=24, balance=7201378.77, direct_rows=441, london_rows_in_direct=60

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `WRONG`**
population: **24** of 640 (whole book 640)
measure `Balance` · aggregation `loan_level` · narrowed `True`
filters applied: `['London', 'Borrower Age > 75', 'Source Portfolio in direct_001']`
spec filters: `{"geographic_region_obligor": "London", "youngest_borrower_age": {"op": "gt", "value": 75.0}, "source_portfolio_id": ["direct_001"]}`
dimensions applied: `none` · spec dimensions: `none`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `borrower age over 75.` → applied; geographic_scope `London` → applied

> Here is the result for your query, covering 24 groups.
> 
> Calculated: Loan-level Balance · London · Borrower Age > 75 · Source Portfolio in direct_001 · 24 groups · 24 loans · as at 30 June 2026.

grade reason: count 24 found; balance 7201378.77 ABSENT

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `WRONG`**
population: **24** of 640 (whole book 640)
measure `Balance` · aggregation `loan_level` · narrowed `True`
filters applied: `['London', 'Borrower Age > 75', 'Source Portfolio in direct_001']`
spec filters: `{"geographic_region_obligor": "London", "youngest_borrower_age": {"op": "gt", "value": 75.0}, "source_portfolio_id": ["direct_001"]}`
dimensions applied: `none` · spec dimensions: `none`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `borrower age over 75.` → applied; geographic_scope `London` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 1

> Here is the result for your query, covering 24 groups.
> 
> Calculated: Loan-level Balance · London · Borrower Age > 75 · Source Portfolio in direct_001 · 24 groups · 24 loans · as at 30 June 2026.

grade reason: count 24 found; balance 7201378.77 ABSENT


## Q05C · BANK75

**Question as typed:** Give me WA LTV for lump sum lending in the Direct portfolio.

**Truth:** desc=WA LTV, Direct book AND lump_sum, count=278, wa_ltv=37.05, balance=73380747.04

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `WRONG`**
population: **441** of 640 (whole book 640)
measure `Current LTV` · aggregation `weighted_avg` · narrowed `True`
filters applied: `['Source Portfolio in direct_001']`
spec filters: `{"source_portfolio_id": ["direct_001"]}`
dimensions applied: `none` · spec dimensions: `none`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> Weighted-average Current LTV: 36.2% · 441 loans.
> 
> Calculated: Weighted-average Current LTV · Source Portfolio in direct_001 · 441 loans · as at 30 June 2026.

grade reason: count 278 ABSENT

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **278** of 640 (whole book 640)
measure `Current LTV` · aggregation `weighted_avg` · narrowed `True`
filters applied: `['Product Type = lump_sum', 'Source Portfolio in direct_001']`
spec filters: `{"erm_product_type": "lump_sum", "source_portfolio_id": ["direct_001"]}`
dimensions applied: `none` · spec dimensions: `none`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `applied` (replayed) · applied [{'kind': 'filter', 'field': 'erm_product_type', 'operator': None, 'value': 'lump_sum'}, {'kind': 'filter', 'field': 'source_portfolio_id', 'operator': None, 'value': 'direct_001'}] · conflicts 0

> Weighted-average Current LTV: 37.0% · 278 loans.
> 
> Calculated: Weighted-average Current LTV · Product Type = lump_sum · Source Portfolio in direct_001 · 278 loans · as at 30 June 2026.

grade reason: count 278 found; wa_ltv 37.05% found


## Q16B · BANK75

**Question as typed:** Break drawdown balance down by both geography and LTV band.

**Truth:** axes=['geographic_region_obligor', 'ltv_bucket'], cells=39, levels={'geographic_region_obligor': 7, 'ltv_bucket': 6}, total=66671398.95

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `WRONG`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Obligor Region (NUTS3)', 'LTV Bucket']` · spec dimensions: `['geographic_region_obligor', 'ltv_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `geography` → applied; grouping_dimension `ltv band` → applied

> Here is the heatmap for your query, covering 42 groups.
> 
> Calculated: Total Balance · grouped by Obligor Region (NUTS3) and LTV Bucket · 42 groups · 640 loans · as at 30 June 2026.

grade reason: cells 39 (artefact rows 42) ABSENT

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **244** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Product Type = drawdown']`
spec filters: `{"erm_product_type": "drawdown"}`
dimensions applied: `['Obligor Region (NUTS3)', 'LTV Bucket']` · spec dimensions: `['geographic_region_obligor', 'ltv_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `geography` → applied; grouping_dimension `ltv band` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'filter', 'field': 'erm_product_type', 'operator': None, 'value': 'drawdown'}] · conflicts 0

> Here is the heatmap for your query, covering 39 groups.
> 
> Calculated: Total Balance · Product Type = drawdown · grouped by Obligor Region (NUTS3) and LTV Bucket · 39 groups · 244 loans · as at 30 June 2026.

grade reason: cells 39 (artefact rows 39) found


## Q17C · BANK75

**Question as typed:** Break Direct portfolio balance down across LTV, ticket size and borrower age.

**Truth:** axes=['ltv_bucket', 'ticket_bucket', 'age_bucket'], cells=143, levels={'ltv_bucket': 6, 'ticket_bucket': 5, 'age_bucket': 7}, total=117356785.33

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `WRONG`**
population: **441** of 640 (whole book 640)
measure `Balance · Average Borrower Age` · aggregation `sum` · narrowed `True`
filters applied: `['Source Portfolio in direct_001']`
spec filters: `{"source_portfolio_id": ["direct_001"]}`
dimensions applied: `['Ticket Size']` · spec dimensions: `['ticket_bucket']`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: multi_measure `more than one measure (balance and age)` → applied; grouping_dimension `ticket size` → applied

> Here is the bar for your query, covering 5 groups.
> 
> Calculated: Balance · Average Borrower Age · Source Portfolio in direct_001 · grouped by Ticket Size · 441 loans · as at 30 June 2026.

grade reason: cells 143 (artefact rows 5) ABSENT

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **441** of 640 (whole book 640)
measure `Balance · Average Borrower Age` · aggregation `sum` · narrowed `True`
filters applied: `['Source Portfolio in direct_001']`
spec filters: `{"source_portfolio_id": ["direct_001"]}`
dimensions applied: `['Age Bucket', 'LTV Bucket', 'Ticket Size']` · spec dimensions: `['age_bucket', 'ltv_bucket', 'ticket_bucket']`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: multi_measure `more than one measure (balance and age)` → applied; grouping_dimension `ticket size` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'dimension', 'field': 'age_bucket'}, {'kind': 'dimension', 'field': 'ltv_bucket'}] · conflicts 0

> Here is the bar for your query, covering 143 groups.
> 
> Calculated: Balance · Average Borrower Age · Source Portfolio in direct_001 · grouped by Age Bucket, LTV Bucket and Ticket Size · 441 loans · as at 30 June 2026.

grade reason: cells 143 (artefact rows 143) found


## Q19A · BANK75

**Question as typed:** How did the Direct book change last month?

> **Known grader divergence:** the frozen human grade says SUBSTANTIVELY CORRECT; the numeric oracle says the £12.4m delta is absent from the answer

**Truth:** open_rows=424, close_rows=441, open=104990413.93, close=117356785.33, delta=12366371.4

**MODEL OFF** — route `cohort_progression` · verdict `answered` · **grade `WRONG`**
population: not published by this route
measure `Cohort progression` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied

> Funded balance for Direct: tracked across 5 reporting period(s) (2026-02 → 2026-06) down.
> 
> Calculated: Cohort progression.

grade reason: delta 12366371.40 ABSENT

**MERGE ON** — route `cohort_progression` · verdict `answered` · **grade `WRONG`**
population: not published by this route
measure `Cohort progression` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Funded balance for Direct: tracked across 5 reporting period(s) (2026-02 → 2026-06) down.
> 
> Calculated: Cohort progression.

grade reason: delta 12366371.40 ABSENT


---

# The two arms DISAGREE on the grade — compare the pair  (`0b_`)


## Q01C · BANK75

**Question as typed:** Count the loans where borrower age is above 55 and current LTV is over 50%.

**Truth:** desc=count, age>55 AND ltv>50, count=144

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `None` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 55` → lost; threshold `LTV over 50` → lost

> I understood that you asked for LTV over 55 and LTV over 50, but that could not be applied to the calculation (LTV over 55 — the threshold was not applied to the calculation; LTV over 50 — the threshold was not applied to the calculation). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **144** of 640 (whole book 640)
measure `None` · aggregation `count` · narrowed `True`
filters applied: `['Current LTV > 50', 'Borrower Age > 55']`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}, "youngest_borrower_age": {"op": "gt", "value": 55.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 55` → applied; threshold `LTV over 50` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'filter', 'field': 'current_loan_to_value', 'operator': 'gt', 'value': 50.0}, {'kind': 'filter', 'field': 'youngest_borrower_age', 'operator': 'gt', 'value': 55.0}] · conflicts 0

> 144 loans · Current Outstanding Balance: £37.1MM.
> 
> Calculated: Count of loans · Current LTV > 50 · Borrower Age > 55 · 144 loans · as at 30 June 2026. Interpretation confidence: medium — check the scope above matches your question.

grade reason: count 144 found


## Q02B · BANK75

**Question as typed:** How much outstanding balance do we have where borrower age exceeds 75 and LTV is over 40%?

**Truth:** desc=balance, age>75 AND ltv>40, count=130, balance=35763779.92

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: **272** of 640 (whole book 640)
measure `Balance · Average Borrower Age` · aggregation `sum` · narrowed `True`
filters applied: `['Current LTV > 40']`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 40.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 75` → applied; threshold `LTV over 40` → lost; multi_measure `more than one measure (balance and age)` → applied

> I understood that you asked for LTV over 40, but that could not be applied to the calculation (LTV over 40 — the threshold was not applied to the calculation). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **130** of 640 (whole book 640)
measure `Balance · Average Borrower Age` · aggregation `sum` · narrowed `True`
filters applied: `['Current LTV > 40', 'Borrower Age > 75']`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 40.0}, "youngest_borrower_age": {"op": "gt", "value": 75.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 75` → applied; threshold `LTV over 40` → applied; multi_measure `more than one measure (balance and age)` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'filter', 'field': 'youngest_borrower_age', 'operator': 'gt', 'value': 75.0}] · conflicts 0

> Balance: £35.76m · Average Borrower Age: 83.5
> 
> Calculated: Balance · Average Borrower Age · Current LTV > 40 · Borrower Age > 75 · 130 loans · as at 30 June 2026.

grade reason: count 130 found; balance 35763779.92 found


## Q03C · BANK75

**Question as typed:** Count drawdown cases where current LTV exceeds 50%.

**Truth:** desc=count, drawdown AND ltv>50, count=45

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: **244** of 640 (whole book 640)
measure `Loans · Weighted-average Current LTV` · aggregation `count` · narrowed `True`
filters applied: `['Product Type = drawdown']`
spec filters: `{"erm_product_type": "drawdown"}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 50` → lost

> I understood that you asked for LTV over 50, but that could not be applied to the calculation (LTV over 50 — the threshold was not applied to the calculation). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **45** of 640 (whole book 640)
measure `Loans · Weighted-average Current LTV` · aggregation `count` · narrowed `True`
filters applied: `['Product Type = drawdown', 'Current LTV > 50']`
spec filters: `{"erm_product_type": "drawdown", "current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 50` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'filter', 'field': 'current_loan_to_value', 'operator': 'gt', 'value': 50.0}] · conflicts 1

> Loans: 45 · Weighted-average Current LTV: 0.56%
> 
> Calculated: Loans · Weighted-average Current LTV · Product Type = drawdown · Current LTV > 50 · 45 loans · as at 30 June 2026.

grade reason: count 45 found


## Q07B · BANK75

**Question as typed:** How do the Direct and Acquired portfolios differ?

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `None` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> 640 loans · Current Outstanding Balance: £172.1MM.
> 
> Calculated: Count of loans · 640 loans · as at 30 June 2026.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['source_portfolio_type', 'source_portfolio_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
concept merge: `applied` (replayed) · applied [{'kind': 'dimension', 'field': 'source_portfolio_type'}] · conflicts 2

> parsed dimension(s) neither applied nor rejected: source_portfolio_type. Refusing to answer with a silently dropped dimension.

grade reason: expected an answer, got a refusal


## Q23A · BANK75

**Question as typed:** When will we reach £100m of funded loans?

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `forecast_extrapolation` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Run-rate extrapolation` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `forecast`
facets: projection `a forward projection` → applied

> The book has already reached £100.0m (current funded balance £172.1m).
> 
> Calculated: Run-rate extrapolation.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `forecast_extrapolation` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `Run-rate extrapolation` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{"current_outstanding_balance": {"op": "eq", "value": 100000000.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `forecast`
facets: projection `a forward projection` → applied; row_population `the population current_outstanding_balance = 100000000.0` → lost
concept merge: `applied` (replayed) · applied [{'kind': 'filter', 'field': 'current_outstanding_balance', 'operator': 'eq', 'value': 100000000.0}] · conflicts 0

> I understood that you asked for loans where Balance is 100000000, but that could not be applied to the calculation (loans where Balance is 100000000 — this analytical route calculated across the whole book; it did not narrow to the requested population). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal


---

# CORRECT, answered over the whole book, and the question is a whole-book question  (`1b_`)


## CFO01 · CFO91

**Question as typed:** What is our total funded balance?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · must contain ['Total Balance']

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> Balance: £172.1MM · 640 loans.
> 
> Calculated: Total Balance · entire funded portfolio · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Balance: £172.1MM · 640 loans.
> 
> Calculated: Total Balance · entire funded portfolio · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO02 · CFO91

**Question as typed:** How many loans do we have?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · must contain ['Count of loans']

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `None` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> 640 loans · Current Outstanding Balance: £172.1MM.
> 
> Calculated: Count of loans · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `None` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> 640 loans · Current Outstanding Balance: £172.1MM.
> 
> Calculated: Count of loans · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO03 · CFO91

**Question as typed:** What is the average loan balance?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `avg` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the average` → applied

> Average Balance: £269K · 640 loans.
> 
> Calculated: Average Balance · entire funded portfolio · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `avg` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the average` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Average Balance: £269K · 640 loans.
> 
> Calculated: Average Balance · entire funded portfolio · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO04 · CFO91

**Question as typed:** What is our weighted average LTV?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Current LTV` · aggregation `weighted_avg` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the weighted average` → applied

> Weighted-average Current LTV: 36.3% · 640 loans.
> 
> Calculated: Weighted-average Current LTV · entire funded portfolio · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Current LTV` · aggregation `weighted_avg` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the weighted average` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Weighted-average Current LTV: 36.3% · 640 loans.
> 
> Calculated: Weighted-average Current LTV · entire funded portfolio · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO05 · CFO91

**Question as typed:** What is the average borrower age?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Borrower Age` · aggregation `avg` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the average` → applied

> Average Borrower Age: 74 · 640 loans.
> 
> Calculated: Average Borrower Age · entire funded portfolio · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Borrower Age` · aggregation `avg` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the average` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Average Borrower Age: 74 · 640 loans.
> 
> Calculated: Average Borrower Age · entire funded portfolio · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO06 · CFO91

**Question as typed:** What is the average interest rate on the book?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Interest Rate` · aggregation `weighted_avg` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the average` → applied

> Weighted-average Interest Rate: 6.3% · 640 loans.
> 
> Calculated: Weighted-average Interest Rate · entire funded portfolio · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Interest Rate` · aggregation `weighted_avg` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the average` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Weighted-average Interest Rate: 6.3% · 640 loans.
> 
> Calculated: Weighted-average Interest Rate · entire funded portfolio · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO07 · CFO91

**Question as typed:** Show balance by region.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · min rows 5

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Obligor Region (NUTS3)']` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `region` → applied

> Here is the bar for your query, covering 7 groups.
> 
> Calculated: Total Balance · grouped by Obligor Region (NUTS3) · 7 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Obligor Region (NUTS3)']` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `region` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the bar for your query, covering 7 groups.
> 
> Calculated: Total Balance · grouped by Obligor Region (NUTS3) · 7 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO08 · CFO91

**Question as typed:** Show loan count by region.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · min rows 5

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `None` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Obligor Region (NUTS3)']` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `region` → applied

> Here is the bar for your query, covering 7 groups.
> 
> Calculated: Count of loans · grouped by Obligor Region (NUTS3) · 7 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `None` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Obligor Region (NUTS3)']` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `region` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the bar for your query, covering 7 groups.
> 
> Calculated: Count of loans · grouped by Obligor Region (NUTS3) · 7 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO09 · CFO91

**Question as typed:** Show balance by product type.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · min rows 2

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Product Type']` · spec dimensions: `['erm_product_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `product type` → applied

> Here is the bar for your query, covering 2 groups.
> 
> Calculated: Total Balance · grouped by Product Type · 2 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Product Type']` · spec dimensions: `['erm_product_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `product type` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the bar for your query, covering 2 groups.
> 
> Calculated: Total Balance · grouped by Product Type · 2 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO10 · CFO91

**Question as typed:** Show balance by broker channel.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · min rows 3

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Broker']` · spec dimensions: `['broker_channel']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `broker channel` → applied

> Here is the bar for your query, covering 4 groups.
> 
> Calculated: Total Balance · grouped by Broker · 4 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Broker']` · spec dimensions: `['broker_channel']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `broker channel` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the bar for your query, covering 4 groups.
> 
> Calculated: Total Balance · grouped by Broker · 4 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO11 · CFO91

**Question as typed:** Show balance by origination channel.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · min rows 2

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Origination Channel']` · spec dimensions: `['origination_channel']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `origination channel` → applied

> Here is the bar for your query, covering 2 groups.
> 
> Calculated: Total Balance · grouped by Origination Channel · 2 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Origination Channel']` · spec dimensions: `['origination_channel']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `origination channel` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the bar for your query, covering 2 groups.
> 
> Calculated: Total Balance · grouped by Origination Channel · 2 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO12 · CFO91

**Question as typed:** Show balance by LTV bucket.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · min rows 2

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket']` · spec dimensions: `['ltv_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv bucket` → applied

> Here is the bar for your query, covering 6 groups.
> 
> Calculated: Total Balance · grouped by LTV Bucket · 6 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket']` · spec dimensions: `['ltv_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv bucket` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the bar for your query, covering 6 groups.
> 
> Calculated: Total Balance · grouped by LTV Bucket · 6 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO13 · CFO91

**Question as typed:** Show balance by age bucket.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · min rows 2

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Age Bucket']` · spec dimensions: `['age_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `age bucket` → applied

> Here is the bar for your query, covering 7 groups.
> 
> Calculated: Total Balance · grouped by Age Bucket · 7 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Age Bucket']` · spec dimensions: `['age_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `age bucket` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the bar for your query, covering 7 groups.
> 
> Calculated: Total Balance · grouped by Age Bucket · 7 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO14 · CFO91

**Question as typed:** Show loan count by product type.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · min rows 2

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `None` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Product Type']` · spec dimensions: `['erm_product_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `product type` → applied

> Here is the bar for your query, covering 2 groups.
> 
> Calculated: Count of loans · grouped by Product Type · 2 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `None` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Product Type']` · spec dimensions: `['erm_product_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `product type` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the bar for your query, covering 2 groups.
> 
> Calculated: Count of loans · grouped by Product Type · 2 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO47 · CFO91

**Question as typed:** Which region has the largest balance?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Obligor Region (NUTS3)']` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: ranking `ranking by region` → applied; grouping_dimension `region` → applied

> Scotland has the highest Balance: £28.9MM (7 groups).
> 
> Calculated: Total Balance · grouped by Obligor Region (NUTS3) · 7 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Obligor Region (NUTS3)']` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: ranking `ranking by region` → applied; grouping_dimension `region` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Scotland has the highest Balance: £28.9MM (7 groups).
> 
> Calculated: Total Balance · grouped by Obligor Region (NUTS3) · 7 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO48 · CFO91

**Question as typed:** Which region has the smallest balance?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Obligor Region (NUTS3)']` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: ranking `ranking by region` → applied; grouping_dimension `region` → applied

> South West has the lowest Balance: £20.5MM (7 groups).
> 
> Calculated: Total Balance · grouped by Obligor Region (NUTS3) · 7 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Obligor Region (NUTS3)']` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: ranking `ranking by region` → applied; grouping_dimension `region` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> South West has the lowest Balance: £20.5MM (7 groups).
> 
> Calculated: Total Balance · grouped by Obligor Region (NUTS3) · 7 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO49 · CFO91

**Question as typed:** Which broker channel has the largest balance?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Broker']` · spec dimensions: `['broker_channel']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: ranking `ranking by broker channel` → applied; grouping_dimension `broker channel` → applied

> Delta Advisers has the highest Balance: £49.1MM (4 groups).
> 
> Calculated: Total Balance · grouped by Broker · 4 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Broker']` · spec dimensions: `['broker_channel']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: ranking `ranking by broker channel` → applied; grouping_dimension `broker channel` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Delta Advisers has the highest Balance: £49.1MM (4 groups).
> 
> Calculated: Total Balance · grouped by Broker · 4 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO62 · CFO91

**Question as typed:** Which product type has the largest share of the book?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Product Type']` · spec dimensions: `['erm_product_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: ranking `ranking by product type` → applied; grouping_dimension `product type` → applied

> lump_sum has the highest Balance: £105.4MM (2 groups).
> 
> Calculated: Total Balance · grouped by Product Type · 2 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Product Type']` · spec dimensions: `['erm_product_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: ranking `ranking by product type` → applied; grouping_dimension `product type` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> lump_sum has the highest Balance: £105.4MM (2 groups).
> 
> Calculated: Total Balance · grouped by Product Type · 2 groups · 640 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO75 · CFO91

**Question as typed:** What is our largest single-name exposure?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `concentration_analysis` · verdict `answered` · **grade `CORRECT`**
population: **640** of — (whole book 640)  ← THE WHOLE BOOK
measure `Largest single-loan current exposure · share of total current exposure` · aggregation `loan_level` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the maximum` → applied

> The largest single-loan exposure is £449k, representing 0.26% of exposure. Top 5 loans account for 1.3% of exposure.
> 
> Calculated: Largest single-loan current exposure · share of total current exposure · 640 loans · as at 2026-06-30.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `concentration_analysis` · verdict `answered` · **grade `CORRECT`**
population: **640** of — (whole book 640)  ← THE WHOLE BOOK
measure `Largest single-loan current exposure · share of total current exposure` · aggregation `loan_level` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the maximum` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> The largest single-loan exposure is £449k, representing 0.26% of exposure. Top 5 loans account for 1.3% of exposure.
> 
> Calculated: Largest single-loan current exposure · share of total current exposure · 640 loans · as at 2026-06-30.

grade reason: answered and met every frozen assertion


## CFO76 · CFO91

**Question as typed:** Show the largest 10 loan exposures.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `concentration_analysis` · verdict `answered` · **grade `CORRECT`**
population: **640** of — (whole book 640)  ← THE WHOLE BOOK
measure `Largest single-loan current exposure · share of total current exposure` · aggregation `loan_level` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the maximum` → applied

> The largest single-loan exposure is £449k, representing 0.26% of exposure. Top 5 loans account for 1.3% of exposure.
> 
> Calculated: Largest single-loan current exposure · share of total current exposure · 640 loans · as at 2026-06-30.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `concentration_analysis` · verdict `answered` · **grade `CORRECT`**
population: **640** of — (whole book 640)  ← THE WHOLE BOOK
measure `Largest single-loan current exposure · share of total current exposure` · aggregation `loan_level` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the maximum` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> The largest single-loan exposure is £449k, representing 0.26% of exposure. Top 5 loans account for 1.3% of exposure.
> 
> Calculated: Largest single-loan current exposure · share of total current exposure · 640 loans · as at 2026-06-30.

grade reason: answered and met every frozen assertion


## Q11A · BANK75

**Question as typed:** Show a table of balance by LTV bucket and ticket-size bucket.

**Truth:** axes=['ltv_bucket', 'ticket_bucket'], cells=30, levels={'ltv_bucket': 6, 'ticket_bucket': 5}, total=172055547.39

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket', 'Ticket Size']` · spec dimensions: `['ltv_bucket', 'ticket_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv bucket` → applied; grouping_dimension `ticket` → applied

> Here is the heatmap for your query, covering 30 groups.
> 
> Calculated: Total Balance · grouped by LTV Bucket and Ticket Size · 30 groups · 640 loans · as at 30 June 2026.

grade reason: cells 30 (artefact rows 30) found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket', 'Ticket Size']` · spec dimensions: `['ltv_bucket', 'ticket_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv bucket` → applied; grouping_dimension `ticket` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the heatmap for your query, covering 30 groups.
> 
> Calculated: Total Balance · grouped by LTV Bucket and Ticket Size · 30 groups · 640 loans · as at 30 June 2026.

grade reason: cells 30 (artefact rows 30) found


## Q11B · BANK75

**Question as typed:** Cross-tab the outstanding balance by LTV band and ticket-size band.

**Truth:** axes=['ltv_bucket', 'ticket_bucket'], cells=30, levels={'ltv_bucket': 6, 'ticket_bucket': 5}, total=172055547.39

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket', 'Ticket Size']` · spec dimensions: `['ltv_bucket', 'ticket_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv band` → applied; grouping_dimension `ticket` → applied

> Here is the heatmap for your query, covering 30 groups.
> 
> Calculated: Total Balance · grouped by LTV Bucket and Ticket Size · 30 groups · 640 loans · as at 30 June 2026.

grade reason: cells 30 (artefact rows 30) found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket', 'Ticket Size']` · spec dimensions: `['ltv_bucket', 'ticket_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv band` → applied; grouping_dimension `ticket` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the heatmap for your query, covering 30 groups.
> 
> Calculated: Total Balance · grouped by LTV Bucket and Ticket Size · 30 groups · 640 loans · as at 30 June 2026.

grade reason: cells 30 (artefact rows 30) found


## Q11C · BANK75

**Question as typed:** Break the balance down by both LTV bucket and ticket-size bucket.

**Truth:** axes=['ltv_bucket', 'ticket_bucket'], cells=30, levels={'ltv_bucket': 6, 'ticket_bucket': 5}, total=172055547.39

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket', 'Ticket Size']` · spec dimensions: `['ltv_bucket', 'ticket_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv bucket` → applied; grouping_dimension `ticket` → applied

> Here is the heatmap for your query, covering 30 groups.
> 
> Calculated: Total Balance · grouped by LTV Bucket and Ticket Size · 30 groups · 640 loans · as at 30 June 2026.

grade reason: cells 30 (artefact rows 30) found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket', 'Ticket Size']` · spec dimensions: `['ltv_bucket', 'ticket_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv bucket` → applied; grouping_dimension `ticket` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the heatmap for your query, covering 30 groups.
> 
> Calculated: Total Balance · grouped by LTV Bucket and Ticket Size · 30 groups · 640 loans · as at 30 June 2026.

grade reason: cells 30 (artefact rows 30) found


## Q12A · BANK75

**Question as typed:** Chart the balance by LTV bucket and borrower-age bucket.

**Truth:** axes=['ltv_bucket', 'age_bucket'], cells=42, levels={'ltv_bucket': 6, 'age_bucket': 7}, total=172055547.39

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket', 'Age Bucket']` · spec dimensions: `['ltv_bucket', 'age_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv bucket` → applied; grouping_dimension `age bucket` → applied

> Here is the heatmap for your query, covering 42 groups.
> 
> Calculated: Total Balance · grouped by LTV Bucket and Age Bucket · 42 groups · 640 loans · as at 30 June 2026.

grade reason: cells 42 (artefact rows 42) found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket', 'Age Bucket']` · spec dimensions: `['ltv_bucket', 'age_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv bucket` → applied; grouping_dimension `age bucket` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the heatmap for your query, covering 42 groups.
> 
> Calculated: Total Balance · grouped by LTV Bucket and Age Bucket · 42 groups · 640 loans · as at 30 June 2026.

grade reason: cells 42 (artefact rows 42) found


## Q12B · BANK75

**Question as typed:** Show me balance split by both LTV band and age band.

**Truth:** axes=['ltv_bucket', 'age_bucket'], cells=42, levels={'ltv_bucket': 6, 'age_bucket': 7}, total=172055547.39

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket', 'Age Bucket']` · spec dimensions: `['ltv_bucket', 'age_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv band` → applied; grouping_dimension `age band` → applied

> Here is the heatmap for your query, covering 42 groups.
> 
> Calculated: Total Balance · grouped by LTV Bucket and Age Bucket · 42 groups · 640 loans · as at 30 June 2026.

grade reason: cells 42 (artefact rows 42) found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket', 'Age Bucket']` · spec dimensions: `['ltv_bucket', 'age_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv band` → applied; grouping_dimension `age band` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the heatmap for your query, covering 42 groups.
> 
> Calculated: Total Balance · grouped by LTV Bucket and Age Bucket · 42 groups · 640 loans · as at 30 June 2026.

grade reason: cells 42 (artefact rows 42) found


## Q13A · BANK75

**Question as typed:** Show a table of balance by LTV bucket and interest-rate bucket.

**Truth:** axes=['ltv_bucket', 'interest_rate_bucket'], cells=30, levels={'ltv_bucket': 6, 'interest_rate_bucket': 5}, total=172055547.39

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket', 'Interest Rate Bucket']` · spec dimensions: `['ltv_bucket', 'interest_rate_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv bucket` → applied; grouping_dimension `rate bucket` → applied

> Here is the heatmap for your query, covering 30 groups.
> 
> Calculated: Total Balance · grouped by LTV Bucket and Interest Rate Bucket · 30 groups · 640 loans · as at 30 June 2026.

grade reason: cells 30 (artefact rows 30) found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket', 'Interest Rate Bucket']` · spec dimensions: `['ltv_bucket', 'interest_rate_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv bucket` → applied; grouping_dimension `rate bucket` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the heatmap for your query, covering 30 groups.
> 
> Calculated: Total Balance · grouped by LTV Bucket and Interest Rate Bucket · 30 groups · 640 loans · as at 30 June 2026.

grade reason: cells 30 (artefact rows 30) found


## Q13B · BANK75

**Question as typed:** Cross-tab balance by LTV band and interest-rate band.

**Truth:** axes=['ltv_bucket', 'interest_rate_bucket'], cells=30, levels={'ltv_bucket': 6, 'interest_rate_bucket': 5}, total=172055547.39

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket', 'Interest Rate Bucket']` · spec dimensions: `['ltv_bucket', 'interest_rate_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv band` → applied; grouping_dimension `rate band` → applied

> Here is the heatmap for your query, covering 30 groups.
> 
> Calculated: Total Balance · grouped by LTV Bucket and Interest Rate Bucket · 30 groups · 640 loans · as at 30 June 2026.

grade reason: cells 30 (artefact rows 30) found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket', 'Interest Rate Bucket']` · spec dimensions: `['ltv_bucket', 'interest_rate_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv band` → applied; grouping_dimension `rate band` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the heatmap for your query, covering 30 groups.
> 
> Calculated: Total Balance · grouped by LTV Bucket and Interest Rate Bucket · 30 groups · 640 loans · as at 30 June 2026.

grade reason: cells 30 (artefact rows 30) found


## Q13C · BANK75

**Question as typed:** Break down outstanding balance by both LTV bucket and rate bucket.

**Truth:** axes=['ltv_bucket', 'interest_rate_bucket'], cells=30, levels={'ltv_bucket': 6, 'interest_rate_bucket': 5}, total=172055547.39

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket', 'Interest Rate Bucket']` · spec dimensions: `['ltv_bucket', 'interest_rate_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv bucket` → applied; grouping_dimension `rate bucket` → applied

> Here is the heatmap for your query, covering 30 groups.
> 
> Calculated: Total Balance · grouped by LTV Bucket and Interest Rate Bucket · 30 groups · 640 loans · as at 30 June 2026.

grade reason: cells 30 (artefact rows 30) found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket', 'Interest Rate Bucket']` · spec dimensions: `['ltv_bucket', 'interest_rate_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv bucket` → applied; grouping_dimension `rate bucket` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the heatmap for your query, covering 30 groups.
> 
> Calculated: Total Balance · grouped by LTV Bucket and Interest Rate Bucket · 30 groups · 640 loans · as at 30 June 2026.

grade reason: cells 30 (artefact rows 30) found


## Q14A · BANK75

**Question as typed:** Show loan count by region and product type.

**Truth:** axes=['geographic_region_obligor', 'erm_product_type'], cells=14, levels={'geographic_region_obligor': 7, 'erm_product_type': 2}, total=640.0, measure=loan count

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `None` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Obligor Region (NUTS3)', 'Product Type']` · spec dimensions: `['geographic_region_obligor', 'erm_product_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `region` → applied; grouping_dimension `product type` → applied

> Here is the heatmap for your query, covering 14 groups.
> 
> Calculated: Count of loans · grouped by Obligor Region (NUTS3) and Product Type · 14 groups · 640 loans · as at 30 June 2026.

grade reason: cells 14 (artefact rows 14) found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `None` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Obligor Region (NUTS3)', 'Product Type']` · spec dimensions: `['geographic_region_obligor', 'erm_product_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `region` → applied; grouping_dimension `product type` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the heatmap for your query, covering 14 groups.
> 
> Calculated: Count of loans · grouped by Obligor Region (NUTS3) and Product Type · 14 groups · 640 loans · as at 30 June 2026.

grade reason: cells 14 (artefact rows 14) found


## Q14B · BANK75

**Question as typed:** Give me a table of loan numbers split by region and loan type.

**Truth:** axes=['geographic_region_obligor', 'erm_product_type'], cells=14, levels={'geographic_region_obligor': 7, 'erm_product_type': 2}, total=640.0, measure=loan count

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `None` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Obligor Region (NUTS3)', 'Product Type']` · spec dimensions: `['geographic_region_obligor', 'erm_product_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `region` → applied; grouping_dimension `loan type` → applied

> Here is the heatmap for your query, covering 14 groups.
> 
> Calculated: Count of loans · grouped by Obligor Region (NUTS3) and Product Type · 14 groups · 640 loans · as at 30 June 2026.

grade reason: cells 14 (artefact rows 14) found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `None` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Obligor Region (NUTS3)', 'Product Type']` · spec dimensions: `['geographic_region_obligor', 'erm_product_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `region` → applied; grouping_dimension `loan type` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the heatmap for your query, covering 14 groups.
> 
> Calculated: Count of loans · grouped by Obligor Region (NUTS3) and Product Type · 14 groups · 640 loans · as at 30 June 2026.

grade reason: cells 14 (artefact rows 14) found


## Q14C · BANK75

**Question as typed:** Break the number of loans down by both geographic region and product type.

**Truth:** axes=['geographic_region_obligor', 'erm_product_type'], cells=14, levels={'geographic_region_obligor': 7, 'erm_product_type': 2}, total=640.0, measure=loan count

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `None` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Obligor Region (NUTS3)', 'Product Type']` · spec dimensions: `['geographic_region_obligor', 'erm_product_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `geographic region` → applied; grouping_dimension `product type` → applied

> Here is the heatmap for your query, covering 14 groups.
> 
> Calculated: Count of loans · grouped by Obligor Region (NUTS3) and Product Type · 14 groups · 640 loans · as at 30 June 2026.

grade reason: cells 14 (artefact rows 14) found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `None` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Obligor Region (NUTS3)', 'Product Type']` · spec dimensions: `['geographic_region_obligor', 'erm_product_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `geographic region` → applied; grouping_dimension `product type` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the heatmap for your query, covering 14 groups.
> 
> Calculated: Count of loans · grouped by Obligor Region (NUTS3) and Product Type · 14 groups · 640 loans · as at 30 June 2026.

grade reason: cells 14 (artefact rows 14) found


---

# The question NAMES a narrowing and NO filter was applied  (`2_n`)


## CFO87 · CFO91

**Question as typed:** How many loans have a Risk Score above 700?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `REFUSE`

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `funded` · reconciled against `None`

> I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Try a metric by a dimension — e.g. 'balance by region', 'weighted average LTV by ticket size', 'ticket size by borrower type' — a cross-period comparison ('compare October and November'), a scale-up forecast ('run-rate to £100m'), risk limits ('are we within limits?'), or 'portfolio summary'.

grade reason: refused, as the bank expects

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 700.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
concept merge: `applied` (replayed) · applied [{'kind': 'filter', 'field': 'current_loan_to_value', 'operator': 'gt', 'value': 700.0}] · conflicts 0

> I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Try a metric by a dimension — e.g. 'balance by region', 'weighted average LTV by ticket size', 'ticket size by borrower type' — a cross-period comparison ('compare October and November'), a scale-up forecast ('run-rate to £100m'), risk limits ('are we within limits?'), or 'portfolio summary'.

grade reason: refused, as the bank expects


## CFO91 · CFO91

**Question as typed:** Which cohort is performing best?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `REFUSE`

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `funded` · reconciled against `None`

> 'best' is not a governed measure in this dataset; no substitute was used. I haven't computed an answer, and I have not substituted a different measure for the one you asked about. Ask for a governed measure — e.g. balance, LTV, interest rate, borrower age or property value — optionally by a dimension.

grade reason: refused, as the bank expects

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{"account_status": "performing"}`
dimensions applied: `none` · spec dimensions: `['source_portfolio_id', 'source_portfolio_id']`
scope `None` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
concept merge: `applied` (replayed) · applied [{'kind': 'dimension', 'field': 'source_portfolio_id'}, {'kind': 'filter', 'field': 'account_status', 'operator': None, 'value': 'performing'}] · conflicts 0

> 'best' is not a governed measure in this dataset; no substitute was used. I haven't computed an answer, and I have not substituted a different measure for the one you asked about. Ask for a governed measure — e.g. balance, LTV, interest rate, borrower age or property value — optionally by a dimension.

grade reason: refused, as the bank expects


## Q17B · BANK75

**Question as typed:** Give me a table of Direct-book balance split by LTV band, ticket-size band and age band.

**Truth:** axes=['ltv_bucket', 'ticket_bucket', 'age_bucket'], cells=143, levels={'ltv_bucket': 6, 'ticket_bucket': 5, 'age_bucket': 7}, total=117356785.33

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket', 'Ticket Size', 'Age Bucket']` · spec dimensions: `['ltv_bucket', 'ticket_bucket', 'age_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: lost_narrowing `Direct` → lost; grouping_dimension `ltv band` → applied; grouping_dimension `ticket` → applied; grouping_dimension `age band` → applied

> I understood that you asked for Direct, but that could not be applied to the calculation (Direct (Source Portfolio Type) — this narrowing was not applied, so the figure covers the whole book rather than only Direct). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket', 'Ticket Size', 'Age Bucket']` · spec dimensions: `['ltv_bucket', 'ticket_bucket', 'age_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: lost_narrowing `Direct` → lost; grouping_dimension `ltv band` → applied; grouping_dimension `ticket` → applied; grouping_dimension `age band` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> I understood that you asked for Direct, but that could not be applied to the calculation (Direct (Source Portfolio Type) — this narrowing was not applied, so the figure covers the whole book rather than only Direct). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal


## Q21B · BANK75

**Question as typed:** For loans over 50% LTV, which region contributed the most balance growth since last month?

**Truth:** open_rows=159, close_rows=144, top_region=Scotland, top_open=4482489.52, top_close=6631018.74, top_delta=2148529.22, runners=[{'region': 'North', 'delta': 1726745.9}, {'region': 'London', 'delta': 546215.15}]

**MODEL OFF** — route `period_change_analysis` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `Governed period change` · aggregation `contribution` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Geographic Region Obligor']` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: threshold `LTV over 50` → lost; comparison_period `comparison period (last month)` → applied; ranking `ranking by region` → applied; grouping_dimension `region` → applied; granularity `month` → applied

> I understood that you asked for LTV over 50, but that could not be applied to the calculation (LTV over 50 — this governed capability does not apply a value threshold, so the figure is not restricted to it). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal

**MERGE ON** — route `period_change_analysis` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `Governed period change` · aggregation `contribution` · narrowed `False`
filters applied: `none`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `['Geographic Region Obligor']` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: threshold `LTV over 50` → applied; comparison_period `comparison period (last month)` → applied; ranking `ranking by region` → applied; grouping_dimension `region` → applied; granularity `month` → applied; row_population `the population current_loan_to_value gt 50.0` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'filter', 'field': 'current_loan_to_value', 'operator': 'gt', 'value': 50.0}] · conflicts 1

> I could not answer this as asked: the answer reports ltv, but the question asked about balance. I have not returned the substituted breakdown.

grade reason: expected an answer, got a refusal


## Q22C · BANK75

**Question as typed:** Which of the Direct and Acquired books drove more of the month-on-month balance increase?

**Truth:** direct={'open': 104990413.93, 'close': 117356785.33, 'delta': 12366371.4}, acquired={'open': 44468825.05, 'close': 54698762.06, 'delta': 10229937.01}, larger=direct

**MODEL OFF** — route `analytical_composition` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `Composed governed capabilities` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded_history`
facets: lost_narrowing `Direct` → lost; comparison_period `comparison period (period on period)` → applied

> I understood that you asked for Direct, but that could not be applied to the calculation (Direct — this narrowing was not applied, so the figure covers the whole book rather than only Direct). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal

**MERGE ON** — route `analytical_composition` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `Composed governed capabilities` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded_history`
facets: lost_narrowing `Direct` → lost; comparison_period `comparison period (period on period)` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 2

> I understood that you asked for Direct, but that could not be applied to the calculation (Direct — this narrowing was not applied, so the figure covers the whole book rather than only Direct). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal


---

# Known grader divergences  (`4_k`)


## CFO60 · CFO91

**Question as typed:** Show product concentration.

> **Known grader divergence:** the frozen run graded TRUE_REFUSAL; the frozen BANK expects DELIVER, so this grader calls it a false refusal

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `concentration_analysis` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `Exposure concentration` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['erm_product_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `product` → lost

> I understood that you asked for product, but that could not be applied to the calculation (product — this answer covers the whole population; it is neither narrowed to nor broken down by product). I have not substituted a broader figure.

grade reason: the bank expects an answer

**MERGE ON** — route `concentration_analysis` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `Exposure concentration` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['erm_product_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `product` → lost
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> I understood that you asked for product, but that could not be applied to the calculation (product — this answer covers the whole population; it is neither narrowed to nor broken down by product). I have not substituted a broader figure.

grade reason: the bank expects an answer


## CFO61 · CFO91

**Question as typed:** Show broker concentration.

> **Known grader divergence:** same as CFO60

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `concentration_analysis` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `Exposure concentration` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `broker` → lost

> I understood that you asked for broker, but that could not be applied to the calculation (broker — this answer covers the whole population; it is neither narrowed to nor broken down by broker). I have not substituted a broader figure.

grade reason: the bank expects an answer

**MERGE ON** — route `concentration_analysis` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `Exposure concentration` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `broker` → lost
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> I understood that you asked for broker, but that could not be applied to the calculation (broker — this answer covers the whole population; it is neither narrowed to nor broken down by broker). I have not substituted a broader figure.

grade reason: the bank expects an answer


## CFO71 · CFO91

**Question as typed:** What is the value of outstanding offers?

> **Known grader divergence:** same as CFO60

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `pipeline` · reconciled against `None`

> 'value' could mean more than one governed measure in this dataset (Balance or Valuation). Say which one and I will answer; no measure was substituted. I haven't computed an answer, and I have not substituted a different measure for the one you asked about. Ask for a governed measure — e.g. balance, LTV, interest rate, borrower age or property value — optionally by a dimension.

grade reason: the bank expects an answer

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `pipeline` · reconciled against `None`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> 'value' could mean more than one governed measure in this dataset (Balance or Valuation). Say which one and I will answer; no measure was substituted. I haven't computed an answer, and I have not substituted a different measure for the one you asked about. Ask for a governed measure — e.g. balance, LTV, interest rate, borrower age or property value — optionally by a dimension.

grade reason: the bank expects an answer


---

# The rest  (`5_r`)


## CFO15 · CFO91

**Question as typed:** What is the balance in the direct book?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **441** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Source Portfolio in direct_001']`
spec filters: `{"source_portfolio_id": ["direct_001"]}`
dimensions applied: `none` · spec dimensions: `none`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> Balance: £117.4MM · 441 loans.
> 
> Calculated: Total Balance · Source Portfolio in direct_001 · 441 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **441** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Source Portfolio in direct_001']`
spec filters: `{"source_portfolio_id": ["direct_001"]}`
dimensions applied: `none` · spec dimensions: `none`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Balance: £117.4MM · 441 loans.
> 
> Calculated: Total Balance · Source Portfolio in direct_001 · 441 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO16 · CFO91

**Question as typed:** What is the balance in the acquired book?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **199** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Source Portfolio in acquired_001']`
spec filters: `{"source_portfolio_id": ["acquired_001"]}`
dimensions applied: `none` · spec dimensions: `none`
scope `acquired` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> Balance: £54.7MM · 199 loans.
> 
> Calculated: Total Balance · Source Portfolio in acquired_001 · 199 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **199** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Source Portfolio in acquired_001']`
spec filters: `{"source_portfolio_id": ["acquired_001"]}`
dimensions applied: `none` · spec dimensions: `none`
scope `acquired` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Balance: £54.7MM · 199 loans.
> 
> Calculated: Total Balance · Source Portfolio in acquired_001 · 199 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO17 · CFO91

**Question as typed:** Summarise the portfolio.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `portfolio_summary` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed analysis` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> At 30 June 2026 the portfolio holds 640 loans with a funded balance of £172.1m. Weighted-average current LTV is 36.3%, the weighted-average interest rate is 6.26% and the average youngest-borrower age is 74.3 years. The largest regional exposures are Scotland (£28.9m, 16.8%), North (£26.8m, 15.6%) and Midlands (£25.1m, 14.6%). By source portfolio: direct_001 £117.4m and acquired_001 £54.7m.
> 
> Calculated: Governed analysis.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `portfolio_summary` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed analysis` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> At 30 June 2026 the portfolio holds 640 loans with a funded balance of £172.1m. Weighted-average current LTV is 36.3%, the weighted-average interest rate is 6.26% and the average youngest-borrower age is 74.3 years. The largest regional exposures are Scotland (£28.9m, 16.8%), North (£26.8m, 15.6%) and Midlands (£25.1m, 14.6%). By source portfolio: direct_001 £117.4m and acquired_001 £54.7m.
> 
> Calculated: Governed analysis.

grade reason: answered and met every frozen assertion


## CFO18 · CFO91

**Question as typed:** Give me a portfolio overview.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `portfolio_summary` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed analysis` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> At 30 June 2026 the portfolio holds 640 loans with a funded balance of £172.1m. Weighted-average current LTV is 36.3%, the weighted-average interest rate is 6.26% and the average youngest-borrower age is 74.3 years. The largest regional exposures are Scotland (£28.9m, 16.8%), North (£26.8m, 15.6%) and Midlands (£25.1m, 14.6%). By source portfolio: direct_001 £117.4m and acquired_001 £54.7m.
> 
> Calculated: Governed analysis.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `portfolio_summary` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed analysis` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> At 30 June 2026 the portfolio holds 640 loans with a funded balance of £172.1m. Weighted-average current LTV is 36.3%, the weighted-average interest rate is 6.26% and the average youngest-borrower age is 74.3 years. The largest regional exposures are Scotland (£28.9m, 16.8%), North (£26.8m, 15.6%) and Midlands (£25.1m, 14.6%). By source portfolio: direct_001 £117.4m and acquired_001 £54.7m.
> 
> Calculated: Governed analysis.

grade reason: answered and met every frozen assertion


## CFO19 · CFO91

**Question as typed:** Show funded balance over time.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · min rows 3

**MODEL OFF** — route `evolution` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Metric evolution` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> Funded balance over 5 period(s): latest £172.1m (up over the window).
> 
> Calculated: Metric evolution.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `evolution` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Metric evolution` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Funded balance over 5 period(s): latest £172.1m (up over the window).
> 
> Calculated: Metric evolution.

grade reason: answered and met every frozen assertion


## CFO20 · CFO91

**Question as typed:** Show loan count over time.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · min rows 3

**MODEL OFF** — route `evolution` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Metric evolution` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> Loan count over 5 period(s): latest 640 (up over the window).
> 
> Calculated: Metric evolution.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `evolution` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Metric evolution` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Loan count over 5 period(s): latest 640 (up over the window).
> 
> Calculated: Metric evolution.

grade reason: answered and met every frozen assertion


## CFO21 · CFO91

**Question as typed:** Show funded balance evolution.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · min rows 3

**MODEL OFF** — route `evolution` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Metric evolution` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> Funded balance over 5 period(s): latest £172.1m (up over the window).
> 
> Calculated: Metric evolution.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `evolution` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Metric evolution` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Funded balance over 5 period(s): latest £172.1m (up over the window).
> 
> Calculated: Metric evolution.

grade reason: answered and met every frozen assertion


## CFO22 · CFO91

**Question as typed:** How has the book grown over the last 3 months?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (change over time)` → applied; granularity `month` → applied

> Between 31 March 2026 and 30 June 2026, 4 of 4 governed metrics could be compared across both snapshots. Largest observed movements measured in counts: Youngest Borrower Age −1 (75 → 74). Largest observed movements measured in currency: Current Outstanding Balance +£50.5m (£121.5m → £172.1m). Largest observed movements measured in percentage points: Current Loan To Value +0.07 pp (36.19% → 36.26%); Current Interest Rate −0.06 pp (6.33% → 6.26%). Against the registry's directionality, 0 metric(s) moved in the improving direction and 1 in the deteriorating direction. The largest observed composition shifts were Broker Channel — Delta Advisers +5.55 pp of count share; Broker Channel — Gamma Direct -4.00 pp of count share; Geographic Region Obligor — South West -3.82 pp of count share. The balance bridge reconciles: opening £121.5m +£91.0m new lending, −£55.8m exits, +£15.3m on continuing loans, closing £172.1m. No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (change over time)` → applied; granularity `month` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Between 31 March 2026 and 30 June 2026, 4 of 4 governed metrics could be compared across both snapshots. Largest observed movements measured in counts: Youngest Borrower Age −1 (75 → 74). Largest observed movements measured in currency: Current Outstanding Balance +£50.5m (£121.5m → £172.1m). Largest observed movements measured in percentage points: Current Loan To Value +0.07 pp (36.19% → 36.26%); Current Interest Rate −0.06 pp (6.33% → 6.26%). Against the registry's directionality, 0 metric(s) moved in the improving direction and 1 in the deteriorating direction. The largest observed composition shifts were Broker Channel — Delta Advisers +5.55 pp of count share; Broker Channel — Gamma Direct -4.00 pp of count share; Geographic Region Obligor — South West -3.82 pp of count share. The balance bridge reconciles: opening £121.5m +£91.0m new lending, −£55.8m exits, +£15.3m on continuing loans, closing £172.1m. No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: answered and met every frozen assertion


## CFO23 · CFO91

**Question as typed:** Show average LTV over time.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `evolution` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Metric evolution` · aggregation `weighted_avg` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the average` → applied

> WA current LTV over 5 period(s): latest 36.3% (down over the window).
> 
> Calculated: Metric evolution.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `evolution` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Metric evolution` · aggregation `weighted_avg` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the average` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> WA current LTV over 5 period(s): latest 36.3% (down over the window).
> 
> Calculated: Metric evolution.

grade reason: answered and met every frozen assertion


## CFO24 · CFO91

**Question as typed:** Show average balance over time.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `evolution` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Metric evolution` · aggregation `avg` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the average` → applied

> Average balance over 5 period(s): latest £269k (up over the window).
> 
> Calculated: Metric evolution.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `evolution` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Metric evolution` · aggregation `avg` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the average` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Average balance over 5 period(s): latest £269k (up over the window).
> 
> Calculated: Metric evolution.

grade reason: answered and met every frozen assertion


## CFO25 · CFO91

**Question as typed:** What has changed since last month?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · must contain ['month']

**MODEL OFF** — route `period_movement` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Month-on-month movement` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied

> Funded balances increased by £22.6m during the month. The largest single regional contribution came from the Scotland (+£12.4m). Weighted-average LTV fell to 36.3%, while average borrower age was unchanged at 74.3 years. Direct_001 contributed approximately +£12.4m and acquired_001 contributed approximately +£10.2m of the movement.
> 
> Calculated: Month-on-month movement.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `period_movement` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Month-on-month movement` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Funded balances increased by £22.6m during the month. The largest single regional contribution came from the Scotland (+£12.4m). Weighted-average LTV fell to 36.3%, while average borrower age was unchanged at 74.3 years. Direct_001 contributed approximately +£12.4m and acquired_001 contributed approximately +£10.2m of the movement.
> 
> Calculated: Month-on-month movement.

grade reason: answered and met every frozen assertion


## CFO26 · CFO91

**Question as typed:** How did the book move last month?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied

> Between 31 May 2026 and 30 June 2026, 4 of 4 governed metrics could be compared across both snapshots. Largest observed movements measured in counts: Youngest Borrower Age −0 (74 → 74). Largest observed movements measured in currency: Current Outstanding Balance +£22.6m (£149.5m → £172.1m). Largest observed movements measured in percentage points: Current Loan To Value −1.32 pp (37.58% → 36.26%); Current Interest Rate −0.05 pp (6.31% → 6.26%). Against the registry's directionality, 1 metric(s) moved in the improving direction and 0 in the deteriorating direction. The largest observed composition shifts were Broker Channel — Delta Advisers +6.22 pp of count share; Geographic Region Obligor — Scotland +4.91 pp of count share; ERM Product Type — lump_sum +4.38 pp of count share. The balance bridge reconciles: opening £149.5m +£83.7m new lending, −£66.7m exits, +£5.6m on continuing loans, closing £172.1m. No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Between 31 May 2026 and 30 June 2026, 4 of 4 governed metrics could be compared across both snapshots. Largest observed movements measured in counts: Youngest Borrower Age −0 (74 → 74). Largest observed movements measured in currency: Current Outstanding Balance +£22.6m (£149.5m → £172.1m). Largest observed movements measured in percentage points: Current Loan To Value −1.32 pp (37.58% → 36.26%); Current Interest Rate −0.05 pp (6.31% → 6.26%). Against the registry's directionality, 1 metric(s) moved in the improving direction and 0 in the deteriorating direction. The largest observed composition shifts were Broker Channel — Delta Advisers +6.22 pp of count share; Geographic Region Obligor — Scotland +4.91 pp of count share; ERM Product Type — lump_sum +4.38 pp of count share. The balance bridge reconciles: opening £149.5m +£83.7m new lending, −£66.7m exits, +£5.6m on continuing loans, closing £172.1m. No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: answered and met every frozen assertion


## CFO27 · CFO91

**Question as typed:** How did balance change since last month?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied

> Between 31 May 2026 and 30 June 2026, 1 of 1 governed metrics could be compared across both snapshots. Largest observed movements measured in currency: Current Outstanding Balance +£22.6m (£149.5m → £172.1m). No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Between 31 May 2026 and 30 June 2026, 1 of 1 governed metrics could be compared across both snapshots. Largest observed movements measured in currency: Current Outstanding Balance +£22.6m (£149.5m → £172.1m). No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: answered and met every frozen assertion


## CFO28 · CFO91

**Question as typed:** How did average LTV change since last month?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `weighted_avg` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; requested_statistic `the average` → applied; granularity `month` → applied

> Between 31 May 2026 and 30 June 2026, 1 of 1 governed metrics could be compared across both snapshots. Largest observed movements measured in percentage points: Current Loan To Value −1.32 pp (37.58% → 36.26%). Against the registry's directionality, 1 metric(s) moved in the improving direction and 0 in the deteriorating direction. No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `weighted_avg` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; requested_statistic `the average` → applied; granularity `month` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Between 31 May 2026 and 30 June 2026, 1 of 1 governed metrics could be compared across both snapshots. Largest observed movements measured in percentage points: Current Loan To Value −1.32 pp (37.58% → 36.26%). Against the registry's directionality, 1 metric(s) moved in the improving direction and 0 in the deteriorating direction. No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: answered and met every frozen assertion


## CFO29 · CFO91

**Question as typed:** Compare this month with last month.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied

> Between 31 May 2026 and 30 June 2026, 4 of 4 governed metrics could be compared across both snapshots. Largest observed movements measured in counts: Youngest Borrower Age −0 (74 → 74). Largest observed movements measured in currency: Current Outstanding Balance +£22.6m (£149.5m → £172.1m). Largest observed movements measured in percentage points: Current Loan To Value −1.32 pp (37.58% → 36.26%); Current Interest Rate −0.05 pp (6.31% → 6.26%). Against the registry's directionality, 1 metric(s) moved in the improving direction and 0 in the deteriorating direction. The largest observed composition shifts were Broker Channel — Delta Advisers +6.22 pp of count share; Geographic Region Obligor — Scotland +4.91 pp of count share; ERM Product Type — lump_sum +4.38 pp of count share. The balance bridge reconciles: opening £149.5m +£83.7m new lending, −£66.7m exits, +£5.6m on continuing loans, closing £172.1m. No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Between 31 May 2026 and 30 June 2026, 4 of 4 governed metrics could be compared across both snapshots. Largest observed movements measured in counts: Youngest Borrower Age −0 (74 → 74). Largest observed movements measured in currency: Current Outstanding Balance +£22.6m (£149.5m → £172.1m). Largest observed movements measured in percentage points: Current Loan To Value −1.32 pp (37.58% → 36.26%); Current Interest Rate −0.05 pp (6.31% → 6.26%). Against the registry's directionality, 1 metric(s) moved in the improving direction and 0 in the deteriorating direction. The largest observed composition shifts were Broker Channel — Delta Advisers +6.22 pp of count share; Geographic Region Obligor — Scotland +4.91 pp of count share; ERM Product Type — lump_sum +4.38 pp of count share. The balance bridge reconciles: opening £149.5m +£83.7m new lending, −£66.7m exits, +£5.6m on continuing loans, closing £172.1m. No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: answered and met every frozen assertion


## CFO30 · CFO91

**Question as typed:** How does the current month compare with the previous month?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied

> Between 31 May 2026 and 30 June 2026, 4 of 4 governed metrics could be compared across both snapshots. Largest observed movements measured in counts: Youngest Borrower Age −0 (74 → 74). Largest observed movements measured in currency: Current Outstanding Balance +£22.6m (£149.5m → £172.1m). Largest observed movements measured in percentage points: Current Loan To Value −1.32 pp (37.58% → 36.26%); Current Interest Rate −0.05 pp (6.31% → 6.26%). Against the registry's directionality, 1 metric(s) moved in the improving direction and 0 in the deteriorating direction. The largest observed composition shifts were Broker Channel — Delta Advisers +6.22 pp of count share; Geographic Region Obligor — Scotland +4.91 pp of count share; ERM Product Type — lump_sum +4.38 pp of count share. The balance bridge reconciles: opening £149.5m +£83.7m new lending, −£66.7m exits, +£5.6m on continuing loans, closing £172.1m. No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Between 31 May 2026 and 30 June 2026, 4 of 4 governed metrics could be compared across both snapshots. Largest observed movements measured in counts: Youngest Borrower Age −0 (74 → 74). Largest observed movements measured in currency: Current Outstanding Balance +£22.6m (£149.5m → £172.1m). Largest observed movements measured in percentage points: Current Loan To Value −1.32 pp (37.58% → 36.26%); Current Interest Rate −0.05 pp (6.31% → 6.26%). Against the registry's directionality, 1 metric(s) moved in the improving direction and 0 in the deteriorating direction. The largest observed composition shifts were Broker Channel — Delta Advisers +6.22 pp of count share; Geographic Region Obligor — Scotland +4.91 pp of count share; ERM Product Type — lump_sum +4.38 pp of count share. The balance bridge reconciles: opening £149.5m +£83.7m new lending, −£66.7m exits, +£5.6m on continuing loans, closing £172.1m. No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: answered and met every frozen assertion


## CFO31 · CFO91

**Question as typed:** Compare the direct and acquired books.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `portfolio_risk_comparison` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Total Current Outstanding Balance · Loan Count · Weighted-average Current Loan To Value · Weighted-average Current Interest Rate · Average Youngest Borrower Age` · aggregation `count` · narrowed `False`
filters applied: `['Direct vs Acquired']`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: cohort_comparison `a comparison by how the loans were sourced` → applied

> Direct has higher observed Current Outstanding Balance than Acquired. Acquired has higher observed Current Loan To Value than Direct. Acquired has higher observed Current Interest Rate than Direct. Direct has higher observed Youngest Borrower Age than Acquired.
> 
> Calculated: Total Current Outstanding Balance · Loan Count · Weighted-average Current Loan To Value · Weighted-average Current Interest Rate · Average Youngest Borrower Age · Direct vs Acquired.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `portfolio_risk_comparison` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Total Current Outstanding Balance · Loan Count · Weighted-average Current Loan To Value · Weighted-average Current Interest Rate · Average Youngest Borrower Age` · aggregation `count` · narrowed `False`
filters applied: `['Direct vs Acquired']`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['source_portfolio_type', 'source_portfolio_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: cohort_comparison `a comparison by how the loans were sourced` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'dimension', 'field': 'source_portfolio_type'}] · conflicts 2

> Direct has higher observed Current Outstanding Balance than Acquired. Acquired has higher observed Current Loan To Value than Direct. Acquired has higher observed Current Interest Rate than Direct. Direct has higher observed Youngest Borrower Age than Acquired.
> 
> Calculated: Total Current Outstanding Balance · Loan Count · Weighted-average Current Loan To Value · Weighted-average Current Interest Rate · Average Youngest Borrower Age · Direct vs Acquired.

grade reason: answered and met every frozen assertion


## CFO32 · CFO91

**Question as typed:** What is the balance difference between this month and last month?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied

> Between 31 May 2026 and 30 June 2026, 1 of 1 governed metrics could be compared across both snapshots. Largest observed movements measured in currency: Current Outstanding Balance +£22.6m (£149.5m → £172.1m). No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Between 31 May 2026 and 30 June 2026, 1 of 1 governed metrics could be compared across both snapshots. Largest observed movements measured in currency: Current Outstanding Balance +£22.6m (£149.5m → £172.1m). No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: answered and met every frozen assertion


## CFO33 · CFO91

**Question as typed:** Show the balance bridge for last month.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `funded_bridge` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Funded balance bridge` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied

> Obligor Region (NUTS3) bridge (Total): funded balance moved from £149.5m in 2026-05 to £172.1m at 2026-06 (latest) — a net change of +£22.6m (up). Largest mover: Scotland (+£12.4m).
> 
> Calculated: Funded balance bridge.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `funded_bridge` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Funded balance bridge` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'measure', 'field': 'current_outstanding_balance'}] · conflicts 0

> Obligor Region (NUTS3) bridge (Total): funded balance moved from £149.5m in 2026-05 to £172.1m at 2026-06 (latest) — a net change of +£22.6m (up). Largest mover: Scotland (+£12.4m).
> 
> Calculated: Funded balance bridge.

grade reason: answered and met every frozen assertion


## CFO34 · CFO91

**Question as typed:** What drove the movement in the book last month?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `funded_bridge` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Funded balance bridge` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied

> Obligor Region (NUTS3) bridge (Total): funded balance moved from £149.5m in 2026-05 to £172.1m at 2026-06 (latest) — a net change of +£22.6m (up). Largest mover: Scotland (+£12.4m).
> 
> Calculated: Funded balance bridge.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `funded_bridge` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Funded balance bridge` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Obligor Region (NUTS3) bridge (Total): funded balance moved from £149.5m in 2026-05 to £172.1m at 2026-06 (latest) — a net change of +£22.6m (up). Largest mover: Scotland (+£12.4m).
> 
> Calculated: Funded balance bridge.

grade reason: answered and met every frozen assertion


## CFO35 · CFO91

**Question as typed:** What is the balance for loans with LTV above 50%?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **144** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Current LTV > 50']`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 50` → applied

> Balance: £37.1MM · 144 loans.
> 
> Calculated: Total Balance · Current LTV > 50 · 144 loans · as at 30 June 2026. Interpretation confidence: medium — check the scope above matches your question.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **144** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Current LTV > 50']`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 50` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Balance: £37.1MM · 144 loans.
> 
> Calculated: Total Balance · Current LTV > 50 · 144 loans · as at 30 June 2026. Interpretation confidence: medium — check the scope above matches your question.

grade reason: answered and met every frozen assertion


## CFO36 · CFO91

**Question as typed:** How many loans have LTV above 50%?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **144** of 640 (whole book 640)
measure `None` · aggregation `count` · narrowed `True`
filters applied: `['Current LTV > 50']`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 50` → applied

> 144 loans · Current Outstanding Balance: £37.1MM.
> 
> Calculated: Count of loans · Current LTV > 50 · 144 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **144** of 640 (whole book 640)
measure `None` · aggregation `count` · narrowed `True`
filters applied: `['Current LTV > 50']`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 50` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 1

> 144 loans · Current Outstanding Balance: £37.1MM.
> 
> Calculated: Count of loans · Current LTV > 50 · 144 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO37 · CFO91

**Question as typed:** Balance by region for loans with LTV above 50%.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · min rows 3

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **144** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Current LTV > 50']`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `['Obligor Region (NUTS3)']` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 50` → applied; grouping_dimension `region` → applied

> Here is the bar for your query, covering 7 groups.
> 
> Calculated: Total Balance · Current LTV > 50 · grouped by Obligor Region (NUTS3) · 7 groups · 144 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **144** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Current LTV > 50']`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `['Obligor Region (NUTS3)']` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 50` → applied; grouping_dimension `region` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the bar for your query, covering 7 groups.
> 
> Calculated: Total Balance · Current LTV > 50 · grouped by Obligor Region (NUTS3) · 7 groups · 144 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO38 · CFO91

**Question as typed:** For loans with LTV above 50%, balance by region

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · min rows 3

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **144** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Current LTV > 50']`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `['Obligor Region (NUTS3)']` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `balance over 50` → applied; grouping_dimension `region` → applied

> Here is the bar for your query, covering 7 groups.
> 
> Calculated: Total Balance · Current LTV > 50 · grouped by Obligor Region (NUTS3) · 7 groups · 144 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **144** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Current LTV > 50']`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `['Obligor Region (NUTS3)']` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `balance over 50` → applied; grouping_dimension `region` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the bar for your query, covering 7 groups.
> 
> Calculated: Total Balance · Current LTV > 50 · grouped by Obligor Region (NUTS3) · 7 groups · 144 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO39 · CFO91

**Question as typed:** What is the balance for loans with borrower age above 75?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **297** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Borrower Age > 75']`
spec filters: `{"youngest_borrower_age": {"op": "gt", "value": 75.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `borrower age over 75` → applied

> Balance: £80.6MM · 297 loans.
> 
> Calculated: Total Balance · Borrower Age > 75 · 297 loans · as at 30 June 2026. Interpretation confidence: medium — check the scope above matches your question.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **297** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Borrower Age > 75']`
spec filters: `{"youngest_borrower_age": {"op": "gt", "value": 75.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `borrower age over 75` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Balance: £80.6MM · 297 loans.
> 
> Calculated: Total Balance · Borrower Age > 75 · 297 loans · as at 30 June 2026. Interpretation confidence: medium — check the scope above matches your question.

grade reason: answered and met every frozen assertion


## CFO40 · CFO91

**Question as typed:** For loans with borrower age above 75, balance by region

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · min rows 3

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **297** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Borrower Age > 75']`
spec filters: `{"youngest_borrower_age": {"op": "gt", "value": 75.0}}`
dimensions applied: `['Obligor Region (NUTS3)']` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `balance over 75,` → applied; grouping_dimension `region` → applied

> Here is the bar for your query, covering 7 groups.
> 
> Calculated: Total Balance · Borrower Age > 75 · grouped by Obligor Region (NUTS3) · 7 groups · 297 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **297** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Borrower Age > 75']`
spec filters: `{"youngest_borrower_age": {"op": "gt", "value": 75.0}}`
dimensions applied: `['Obligor Region (NUTS3)']` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `balance over 75,` → applied; grouping_dimension `region` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the bar for your query, covering 7 groups.
> 
> Calculated: Total Balance · Borrower Age > 75 · grouped by Obligor Region (NUTS3) · 7 groups · 297 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO41 · CFO91

**Question as typed:** Balance by region for loans with borrower age above 75.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · min rows 3

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **297** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Borrower Age > 75']`
spec filters: `{"youngest_borrower_age": {"op": "gt", "value": 75.0}}`
dimensions applied: `['Obligor Region (NUTS3)']` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `borrower age over 75.` → applied; grouping_dimension `region` → applied

> Here is the bar for your query, covering 7 groups.
> 
> Calculated: Total Balance · Borrower Age > 75 · grouped by Obligor Region (NUTS3) · 7 groups · 297 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **297** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Borrower Age > 75']`
spec filters: `{"youngest_borrower_age": {"op": "gt", "value": 75.0}}`
dimensions applied: `['Obligor Region (NUTS3)']` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `borrower age over 75.` → applied; grouping_dimension `region` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the bar for your query, covering 7 groups.
> 
> Calculated: Total Balance · Borrower Age > 75 · grouped by Obligor Region (NUTS3) · 7 groups · 297 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO42 · CFO91

**Question as typed:** How many loans are above £300,000?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **262** of 640 (whole book 640)
measure `None` · aggregation `count` · narrowed `True`
filters applied: `['Balance > 300000']`
spec filters: `{"current_outstanding_balance": {"op": "gt", "value": 300000.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `over 300,000` → applied

> 262 loans · Balance: £97.9MM.
> 
> Calculated: Count of loans · Balance > 300000 · 262 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **262** of 640 (whole book 640)
measure `None` · aggregation `count` · narrowed `True`
filters applied: `['Balance > 300000']`
spec filters: `{"current_outstanding_balance": {"op": "gt", "value": 300000.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `over 300,000` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> 262 loans · Balance: £97.9MM.
> 
> Calculated: Count of loans · Balance > 300000 · 262 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO43 · CFO91

**Question as typed:** What is the balance for loans in London?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **83** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['London']`
spec filters: `{"geographic_region_obligor": "London"}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: geographic_scope `London` → applied

> Balance: £22.4MM · 83 loans.
> 
> Calculated: Total Balance · London · 83 loans · as at 30 June 2026. Interpretation confidence: medium — check the scope above matches your question.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **83** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['London']`
spec filters: `{"geographic_region_obligor": "London"}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: geographic_scope `London` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 1

> Balance: £22.4MM · 83 loans.
> 
> Calculated: Total Balance · London · 83 loans · as at 30 June 2026. Interpretation confidence: medium — check the scope above matches your question.

grade reason: answered and met every frozen assertion


## CFO44 · CFO91

**Question as typed:** How many loans have an interest rate above 7%?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **203** of 640 (whole book 640)
measure `None` · aggregation `count` · narrowed `True`
filters applied: `['Interest Rate > 7']`
spec filters: `{"current_interest_rate": {"op": "gt", "value": 7.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `interest rate over 7` → applied

> 203 loans · Current Outstanding Balance: £55.0MM.
> 
> Calculated: Count of loans · Interest Rate > 7 · 203 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **203** of 640 (whole book 640)
measure `None` · aggregation `count` · narrowed `True`
filters applied: `['Interest Rate > 7']`
spec filters: `{"current_interest_rate": {"op": "gt", "value": 7.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `interest rate over 7` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> 203 loans · Current Outstanding Balance: £55.0MM.
> 
> Calculated: Count of loans · Interest Rate > 7 · 203 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO45 · CFO91

**Question as typed:** Balance by product type for loans with LTV above 40%.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · min rows 2

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **272** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Current LTV > 40']`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 40.0}}`
dimensions applied: `['Product Type']` · spec dimensions: `['erm_product_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 40` → applied; grouping_dimension `product type` → applied

> Here is the bar for your query, covering 2 groups.
> 
> Calculated: Total Balance · Current LTV > 40 · grouped by Product Type · 2 groups · 272 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **272** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Current LTV > 40']`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 40.0}}`
dimensions applied: `['Product Type']` · spec dimensions: `['erm_product_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 40` → applied; grouping_dimension `product type` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the bar for your query, covering 2 groups.
> 
> Calculated: Total Balance · Current LTV > 40 · grouped by Product Type · 2 groups · 272 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO46 · CFO91

**Question as typed:** How many drawdown loans do we have?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **244** of 640 (whole book 640)
measure `None` · aggregation `count` · narrowed `True`
filters applied: `['Product Type = drawdown']`
spec filters: `{"erm_product_type": "drawdown"}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> 244 loans · Current Outstanding Balance: £66.7MM.
> 
> Calculated: Count of loans · Product Type = drawdown · 244 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **244** of 640 (whole book 640)
measure `None` · aggregation `count` · narrowed `True`
filters applied: `['Product Type = drawdown']`
spec filters: `{"erm_product_type": "drawdown"}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 1

> 244 loans · Current Outstanding Balance: £66.7MM.
> 
> Calculated: Count of loans · Product Type = drawdown · 244 loans · as at 30 June 2026.

grade reason: answered and met every frozen assertion


## CFO50 · CFO91

**Question as typed:** Which region added the most balance since last month?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · must contain ['increased the most']

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Geographic Region Obligor']` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; ranking `ranking by region` → applied; grouping_dimension `region` → applied; granularity `month` → applied

> Between 31 May 2026 and 30 June 2026, ranked by absolute balance movement across Geographic Region Obligor, Scotland increased the most: Scotland £16.5m → £28.9m (+£12.4m, +75.0%). Then Wales £15.5m → £24.3m (+£8.8m, +56.6%); South East £21.6m → £24.0m (+£2.5m, +11.4%); North £25.0m → £26.8m (+£1.8m, +7.4%). The full ranking of all 5 ranked categories is in the table below. 2 further categories did not increase on this basis and are not listed.
> 
> Calculated: Governed period change · ranked by Geographic Region Obligor · absolute balance movement, largest increases first · 2026-05-31 → 2026-06-30.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Geographic Region Obligor']` · spec dimensions: `['geographic_region_obligor', 'geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; ranking `ranking by region` → applied; grouping_dimension `region` → applied; granularity `month` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'dimension', 'field': 'geographic_region_obligor'}] · conflicts 0

> Between 31 May 2026 and 30 June 2026, ranked by absolute balance movement across Geographic Region Obligor, Scotland increased the most: Scotland £16.5m → £28.9m (+£12.4m, +75.0%). Then Wales £15.5m → £24.3m (+£8.8m, +56.6%); South East £21.6m → £24.0m (+£2.5m, +11.4%); North £25.0m → £26.8m (+£1.8m, +7.4%). The full ranking of all 5 ranked categories is in the table below. 2 further categories did not increase on this basis and are not listed.
> 
> Calculated: Governed period change · ranked by Geographic Region Obligor · absolute balance movement, largest increases first · 2026-05-31 → 2026-06-30.

grade reason: answered and met every frozen assertion


## CFO51 · CFO91

**Question as typed:** Which region lost the most balance since last month?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Geographic Region Obligor']` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; ranking `ranking by region` → applied; grouping_dimension `region` → applied; granularity `month` → applied

> Between 31 May 2026 and 30 June 2026, ranked by absolute balance movement across Geographic Region Obligor, London decreased the most: London £24.8m → £22.4m (−£2.4m, -9.6%). Then South West £22.6m → £20.5m (−£2.1m, -9.3%). 5 further categories did not decrease on this basis and are not listed.
> 
> Calculated: Governed period change · ranked by Geographic Region Obligor · absolute balance movement, largest decreases first · 2026-05-31 → 2026-06-30.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Geographic Region Obligor']` · spec dimensions: `['geographic_region_obligor', 'geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; ranking `ranking by region` → applied; grouping_dimension `region` → applied; granularity `month` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'dimension', 'field': 'geographic_region_obligor'}] · conflicts 0

> Between 31 May 2026 and 30 June 2026, ranked by absolute balance movement across Geographic Region Obligor, London decreased the most: London £24.8m → £22.4m (−£2.4m, -9.6%). Then South West £22.6m → £20.5m (−£2.1m, -9.3%). 5 further categories did not decrease on this basis and are not listed.
> 
> Calculated: Governed period change · ranked by Geographic Region Obligor · absolute balance movement, largest decreases first · 2026-05-31 → 2026-06-30.

grade reason: answered and met every frozen assertion


## CFO52 · CFO91

**Question as typed:** Which two regions added the most balance since last month?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · must contain ['top 2']

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Geographic Region Obligor']` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; ranking `ranking by regions` → applied; grouping_dimension `regions` → applied; granularity `month` → applied

> Between 31 May 2026 and 30 June 2026, ranked by absolute balance movement across Geographic Region Obligor, Scotland increased the most: Scotland £16.5m → £28.9m (+£12.4m, +75.0%). Then Wales £15.5m → £24.3m (+£8.8m, +56.6%). Showing the top 2 of 7 categories. 2 further categories did not increase on this basis and are not listed.
> 
> Calculated: Governed period change · ranked by Geographic Region Obligor · absolute balance movement, largest increases first, top 2 of 7 · 2026-05-31 → 2026-06-30.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Geographic Region Obligor']` · spec dimensions: `['geographic_region_obligor', 'geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; ranking `ranking by regions` → applied; grouping_dimension `regions` → applied; granularity `month` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'dimension', 'field': 'geographic_region_obligor'}] · conflicts 0

> Between 31 May 2026 and 30 June 2026, ranked by absolute balance movement across Geographic Region Obligor, Scotland increased the most: Scotland £16.5m → £28.9m (+£12.4m, +75.0%). Then Wales £15.5m → £24.3m (+£8.8m, +56.6%). Showing the top 2 of 7 categories. 2 further categories did not increase on this basis and are not listed.
> 
> Calculated: Governed period change · ranked by Geographic Region Obligor · absolute balance movement, largest increases first, top 2 of 7 · 2026-05-31 → 2026-06-30.

grade reason: answered and met every frozen assertion


## CFO53 · CFO91

**Question as typed:** Which three regions added the most balance since last month?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER` · must contain ['top 3']

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Geographic Region Obligor']` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; ranking `ranking by regions` → applied; grouping_dimension `regions` → applied; granularity `month` → applied

> Between 31 May 2026 and 30 June 2026, ranked by absolute balance movement across Geographic Region Obligor, Scotland increased the most: Scotland £16.5m → £28.9m (+£12.4m, +75.0%). Then Wales £15.5m → £24.3m (+£8.8m, +56.6%); South East £21.6m → £24.0m (+£2.5m, +11.4%). Showing the top 3 of 7 categories. 2 further categories did not increase on this basis and are not listed.
> 
> Calculated: Governed period change · ranked by Geographic Region Obligor · absolute balance movement, largest increases first, top 3 of 7 · 2026-05-31 → 2026-06-30.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Geographic Region Obligor']` · spec dimensions: `['geographic_region_obligor', 'geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; ranking `ranking by regions` → applied; grouping_dimension `regions` → applied; granularity `month` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'dimension', 'field': 'geographic_region_obligor'}] · conflicts 0

> Between 31 May 2026 and 30 June 2026, ranked by absolute balance movement across Geographic Region Obligor, Scotland increased the most: Scotland £16.5m → £28.9m (+£12.4m, +75.0%). Then Wales £15.5m → £24.3m (+£8.8m, +56.6%); South East £21.6m → £24.0m (+£2.5m, +11.4%). Showing the top 3 of 7 categories. 2 further categories did not increase on this basis and are not listed.
> 
> Calculated: Governed period change · ranked by Geographic Region Obligor · absolute balance movement, largest increases first, top 3 of 7 · 2026-05-31 → 2026-06-30.

grade reason: answered and met every frozen assertion


## CFO54 · CFO91

**Question as typed:** Which region grew fastest in balance since last month?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Geographic Region Obligor']` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; ranking `ranking by region` → applied; grouping_dimension `region` → applied; granularity `month` → applied

> Between 31 May 2026 and 30 June 2026, ranked by percentage balance movement across Geographic Region Obligor, Scotland increased the most: Scotland £16.5m → £28.9m (+£12.4m, +75.0%). Then Wales £15.5m → £24.3m (+£8.8m, +56.6%); South East £21.6m → £24.0m (+£2.5m, +11.4%); North £25.0m → £26.8m (+£1.8m, +7.4%). The full ranking of all 5 ranked categories is in the table below. 2 further categories did not increase on this basis and are not listed.
> 
> Calculated: Governed period change · ranked by Geographic Region Obligor · percentage balance movement, largest increases first · 2026-05-31 → 2026-06-30.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Geographic Region Obligor']` · spec dimensions: `['geographic_region_obligor', 'geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; ranking `ranking by region` → applied; grouping_dimension `region` → applied; granularity `month` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'dimension', 'field': 'geographic_region_obligor'}] · conflicts 0

> Between 31 May 2026 and 30 June 2026, ranked by percentage balance movement across Geographic Region Obligor, Scotland increased the most: Scotland £16.5m → £28.9m (+£12.4m, +75.0%). Then Wales £15.5m → £24.3m (+£8.8m, +56.6%); South East £21.6m → £24.0m (+£2.5m, +11.4%); North £25.0m → £26.8m (+£1.8m, +7.4%). The full ranking of all 5 ranked categories is in the table below. 2 further categories did not increase on this basis and are not listed.
> 
> Calculated: Governed period change · ranked by Geographic Region Obligor · percentage balance movement, largest increases first · 2026-05-31 → 2026-06-30.

grade reason: answered and met every frozen assertion


## CFO55 · CFO91

**Question as typed:** Which broker channel added the most balance since last month?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Broker Channel']` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; ranking `ranking by broker channel` → applied; grouping_dimension `broker channel` → applied; granularity `month` → applied

> Between 31 May 2026 and 30 June 2026, ranked by absolute balance movement across Broker Channel, Delta Advisers increased the most: Delta Advisers £34.3m → £49.1m (+£14.7m, +42.9%). Then Gamma Direct £37.1m → £40.9m (+£3.8m, +10.1%); Beta Partners £38.0m → £40.5m (+£2.5m, +6.5%); Alpha Network £40.0m → £41.7m (+£1.6m, +4.1%).
> 
> Calculated: Governed period change · ranked by Broker Channel · absolute balance movement, largest increases first · 2026-05-31 → 2026-06-30.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Broker Channel']` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; ranking `ranking by broker channel` → applied; grouping_dimension `broker channel` → applied; granularity `month` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Between 31 May 2026 and 30 June 2026, ranked by absolute balance movement across Broker Channel, Delta Advisers increased the most: Delta Advisers £34.3m → £49.1m (+£14.7m, +42.9%). Then Gamma Direct £37.1m → £40.9m (+£3.8m, +10.1%); Beta Partners £38.0m → £40.5m (+£2.5m, +6.5%); Alpha Network £40.0m → £41.7m (+£1.6m, +4.1%).
> 
> Calculated: Governed period change · ranked by Broker Channel · absolute balance movement, largest increases first · 2026-05-31 → 2026-06-30.

grade reason: answered and met every frozen assertion


## CFO56 · CFO91

**Question as typed:** Which region added the most balance since last month for loans with LTV above 50%?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `['Geographic Region Obligor']` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: threshold `LTV over 50` → applied; comparison_period `comparison period (last month)` → applied; ranking `ranking by region` → applied; grouping_dimension `region` → applied; granularity `month` → applied; row_population `the population current_loan_to_value gt 50.0` → applied

> Between 31 May 2026 and 30 June 2026, ranked by absolute balance movement across Geographic Region Obligor, Scotland increased the most: Scotland £4.5m → £6.6m (+£2.1m, +47.9%). Then North £4.7m → £6.4m (+£1.7m, +36.7%); London £5.4m → £6.0m (+£546k, +10.1%). 4 further categories did not increase on this basis and are not listed.
> 
> Calculated: Governed period change · ranked by Geographic Region Obligor · absolute balance movement, largest increases first · 2026-05-31 → 2026-06-30.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `['Geographic Region Obligor']` · spec dimensions: `['geographic_region_obligor', 'geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: threshold `LTV over 50` → applied; comparison_period `comparison period (last month)` → applied; ranking `ranking by region` → applied; grouping_dimension `region` → applied; granularity `month` → applied; row_population `the population current_loan_to_value gt 50.0` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'dimension', 'field': 'geographic_region_obligor'}] · conflicts 0

> Between 31 May 2026 and 30 June 2026, ranked by absolute balance movement across Geographic Region Obligor, Scotland increased the most: Scotland £4.5m → £6.6m (+£2.1m, +47.9%). Then North £4.7m → £6.4m (+£1.7m, +36.7%); London £5.4m → £6.0m (+£546k, +10.1%). 4 further categories did not increase on this basis and are not listed.
> 
> Calculated: Governed period change · ranked by Geographic Region Obligor · absolute balance movement, largest increases first · 2026-05-31 → 2026-06-30.

grade reason: answered and met every frozen assertion


## CFO57 · CFO91

**Question as typed:** For loans with LTV above 50%, which region added the most balance since last month?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `['Geographic Region Obligor']` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: threshold `LTV over 50` → applied; comparison_period `comparison period (last month)` → applied; ranking `ranking by region` → applied; grouping_dimension `region` → applied; granularity `month` → applied; row_population `the population current_loan_to_value gt 50.0` → applied

> Between 31 May 2026 and 30 June 2026, ranked by absolute balance movement across Geographic Region Obligor, Scotland increased the most: Scotland £4.5m → £6.6m (+£2.1m, +47.9%). Then North £4.7m → £6.4m (+£1.7m, +36.7%); London £5.4m → £6.0m (+£546k, +10.1%). 4 further categories did not increase on this basis and are not listed.
> 
> Calculated: Governed period change · ranked by Geographic Region Obligor · absolute balance movement, largest increases first · 2026-05-31 → 2026-06-30.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `['Geographic Region Obligor']` · spec dimensions: `['geographic_region_obligor', 'geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: threshold `LTV over 50` → applied; comparison_period `comparison period (last month)` → applied; ranking `ranking by region` → applied; grouping_dimension `region` → applied; granularity `month` → applied; row_population `the population current_loan_to_value gt 50.0` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'dimension', 'field': 'geographic_region_obligor'}] · conflicts 0

> Between 31 May 2026 and 30 June 2026, ranked by absolute balance movement across Geographic Region Obligor, Scotland increased the most: Scotland £4.5m → £6.6m (+£2.1m, +47.9%). Then North £4.7m → £6.4m (+£1.7m, +36.7%); London £5.4m → £6.0m (+£546k, +10.1%). 4 further categories did not increase on this basis and are not listed.
> 
> Calculated: Governed period change · ranked by Geographic Region Obligor · absolute balance movement, largest increases first · 2026-05-31 → 2026-06-30.

grade reason: answered and met every frozen assertion


## CFO58 · CFO91

**Question as typed:** Which product type grew the most since last month?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['ERM Product Type']` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; ranking `ranking by product type` → applied; grouping_dimension `product type` → applied; granularity `month` → applied

> Between 31 May 2026 and 30 June 2026, ranked by absolute balance movement across ERM Product Type, lump_sum increased the most: lump_sum £87.3m → £105.4m (+£18.0m, +20.7%). Then drawdown £62.1m → £66.7m (+£4.6m, +7.3%).
> 
> Calculated: Governed period change · ranked by ERM Product Type · absolute balance movement, largest increases first · 2026-05-31 → 2026-06-30.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['ERM Product Type']` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; ranking `ranking by product type` → applied; grouping_dimension `product type` → applied; granularity `month` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Between 31 May 2026 and 30 June 2026, ranked by absolute balance movement across ERM Product Type, lump_sum increased the most: lump_sum £87.3m → £105.4m (+£18.0m, +20.7%). Then drawdown £62.1m → £66.7m (+£4.6m, +7.3%).
> 
> Calculated: Governed period change · ranked by ERM Product Type · absolute balance movement, largest increases first · 2026-05-31 → 2026-06-30.

grade reason: answered and met every frozen assertion


## CFO59 · CFO91

**Question as typed:** What proportion of the book is in London?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **83** of 640 (whole book 640)
measure `Balance` · aggregation `share` · narrowed `True`
filters applied: `['London']`
spec filters: `{"geographic_region_obligor": "London"}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: geographic_scope `London` → applied; share `a proportion of the book` → applied

> Current Outstanding Balance Share Pct: 13.0% · 83 loans · Population Total: 640.
> 
> Calculated: Share of Balance · London · 83 qualifying loans of 640 · as at 30 June 2026. Interpretation confidence: medium — check the scope above matches your question.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **83** of 640 (whole book 640)
measure `Balance` · aggregation `share` · narrowed `True`
filters applied: `['London']`
spec filters: `{"geographic_region_obligor": "London"}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: geographic_scope `London` → applied; share `a proportion of the book` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 1

> Current Outstanding Balance Share Pct: 13.0% · 83 loans · Population Total: 640.
> 
> Calculated: Share of Balance · London · 83 qualifying loans of 640 · as at 30 June 2026. Interpretation confidence: medium — check the scope above matches your question.

grade reason: answered and met every frozen assertion


## CFO63 · CFO91

**Question as typed:** What share of the book is drawdown?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **244** of 640 (whole book 640)
measure `Balance` · aggregation `share` · narrowed `True`
filters applied: `['Product Type = drawdown']`
spec filters: `{"erm_product_type": "drawdown"}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: share `a proportion of the book` → applied

> Current Outstanding Balance Share Pct: 38.7% · 244 loans · Population Total: 640.
> 
> Calculated: Share of Balance · Product Type = drawdown · 244 qualifying loans of 640 · as at 30 June 2026. Interpretation confidence: medium — check the scope above matches your question.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **244** of 640 (whole book 640)
measure `Balance` · aggregation `share` · narrowed `True`
filters applied: `['Product Type = drawdown']`
spec filters: `{"erm_product_type": "drawdown"}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: share `a proportion of the book` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 1

> Current Outstanding Balance Share Pct: 38.7% · 244 loans · Population Total: 640.
> 
> Calculated: Share of Balance · Product Type = drawdown · 244 qualifying loans of 640 · as at 30 June 2026. Interpretation confidence: medium — check the scope above matches your question.

grade reason: answered and met every frozen assertion


## CFO64 · CFO91

**Question as typed:** Show origination channel concentration.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `concentration_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Exposure concentration` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['origination_channel']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `origination channel` → applied

> Largest origination channel exposure is broker (77.8% of exposure).
> 
> Calculated: Exposure concentration.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `concentration_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Exposure concentration` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['origination_channel']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `origination channel` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Largest origination channel exposure is broker (77.8% of exposure).
> 
> Calculated: Exposure concentration.

grade reason: answered and met every frozen assertion


## CFO65 · CFO91

**Question as typed:** What proportion of the book is in the acquired portfolio?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **199** of 640 (whole book 640)
measure `Balance` · aggregation `share` · narrowed `True`
filters applied: `['Source Portfolio in acquired_001']`
spec filters: `{"source_portfolio_id": ["acquired_001"]}`
dimensions applied: `none` · spec dimensions: `none`
scope `acquired` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: share `a proportion of the book` → applied

> Current Outstanding Balance Share Pct: 31.8% · 199 loans · Population Total: 640.
> 
> Calculated: Share of Balance · Source Portfolio in acquired_001 · 199 qualifying loans of 640 · as at 30 June 2026. Interpretation confidence: medium — check the scope above matches your question.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **199** of 640 (whole book 640)
measure `Balance` · aggregation `share` · narrowed `True`
filters applied: `['Source Portfolio Type = acquired', 'Source Portfolio in acquired_001']`
spec filters: `{"source_portfolio_type": "acquired", "source_portfolio_id": ["acquired_001"]}`
dimensions applied: `none` · spec dimensions: `none`
scope `acquired` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: share `a proportion of the book` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'filter', 'field': 'source_portfolio_type', 'operator': None, 'value': 'acquired'}] · conflicts 0

> Current Outstanding Balance Share Pct: 31.8% · 199 loans · Population Total: 640.
> 
> Calculated: Share of Balance · Source Portfolio Type = acquired · Source Portfolio in acquired_001 · 199 qualifying loans of 640 · as at 30 June 2026. Interpretation confidence: medium — check the scope above matches your question.

grade reason: answered and met every frozen assertion


## CFO66 · CFO91

**Question as typed:** What is the pipeline balance?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **8** of 8 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `pipeline` · reconciled against `pipeline`

> Balance: £3.6MM · 8 loans.
> 
> Calculated: Total Balance · entire pipeline · 8 loans.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **8** of 8 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `pipeline` · reconciled against `pipeline`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Balance: £3.6MM · 8 loans.
> 
> Calculated: Total Balance · entire pipeline · 8 loans.

grade reason: answered and met every frozen assertion


## CFO67 · CFO91

**Question as typed:** How many cases are in the pipeline?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **8** of 8 (whole book 640)
measure `None` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `pipeline` · reconciled against `pipeline`

> 8 loans · Current Outstanding Balance: £3.6MM.
> 
> Calculated: Count of loans · 8 loans.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **8** of 8 (whole book 640)
measure `None` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `pipeline` · reconciled against `pipeline`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> 8 loans · Current Outstanding Balance: £3.6MM.
> 
> Calculated: Count of loans · 8 loans.

grade reason: answered and met every frozen assertion


## CFO68 · CFO91

**Question as typed:** Show the pipeline by stage.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **8** of 8 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Pipeline Stage']` · spec dimensions: `['pipeline_stage']`
scope `None` · scopeApplied `None` · dataset context `pipeline` · reconciled against `pipeline`
facets: grouping_dimension `stage` → applied

> Here is the bar for your query, covering 5 groups.
> 
> Calculated: Total Balance · grouped by Pipeline Stage · 5 groups · 8 loans.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **8** of 8 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Pipeline Stage']` · spec dimensions: `['pipeline_stage']`
scope `None` · scopeApplied `None` · dataset context `pipeline` · reconciled against `pipeline`
facets: grouping_dimension `stage` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the bar for your query, covering 5 groups.
> 
> Calculated: Total Balance · grouped by Pipeline Stage · 5 groups · 8 loans.

grade reason: answered and met every frozen assertion


## CFO69 · CFO91

**Question as typed:** How has the pipeline evolved?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `evolution` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Metric evolution` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `pipeline` · reconciled against `pipeline`

> Pipeline amount over 5 period(s): latest £3.6m (up over the window).
> 
> Calculated: Metric evolution.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `evolution` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Metric evolution` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `pipeline` · reconciled against `pipeline`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Pipeline amount over 5 period(s): latest £3.6m (up over the window).
> 
> Calculated: Metric evolution.

grade reason: answered and met every frozen assertion


## CFO70 · CFO91

**Question as typed:** Show pipeline evolution by stage.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `evolution_pipeline_stage` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed analysis` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `pipeline` · reconciled against `pipeline`
facets: grouping_dimension `stage` → applied

> Pipeline amount by stage across 5 period(s): stages APPLICATION, COMPLETED, KFI, OFFER, WITHDRAWN.
> 
> Calculated: Governed analysis.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `evolution_pipeline_stage` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed analysis` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `pipeline` · reconciled against `pipeline`
facets: grouping_dimension `stage` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Pipeline amount by stage across 5 period(s): stages APPLICATION, COMPLETED, KFI, OFFER, WITHDRAWN.
> 
> Calculated: Governed analysis.

grade reason: answered and met every frozen assertion


## CFO72 · CFO91

**Question as typed:** Are any of our concentration limits at risk?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `risk_limits` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Concentration limits vs the governing document` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> 5 passed, 0 warning(s), 6 breach(es), 1 need review, 3 unavailable. Nearest to limit: Top 3 brokers (-31.5 pp headroom). Largest concentration: Top 3 brokers at 76.5%.
> 
> Calculated: Concentration limits vs the governing document.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `risk_limits` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Concentration limits vs the governing document` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> 5 passed, 0 warning(s), 6 breach(es), 1 need review, 3 unavailable. Nearest to limit: Top 3 brokers (-31.5 pp headroom). Largest concentration: Top 3 brokers at 76.5%.
> 
> Calculated: Concentration limits vs the governing document.

grade reason: answered and met every frozen assertion


## CFO73 · CFO91

**Question as typed:** Which of our limits are currently most at risk?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `risk_limits` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Concentration limits vs the governing document` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> 5 passed, 0 warning(s), 6 breach(es), 1 need review, 3 unavailable. Nearest to limit: Top 3 brokers (-31.5 pp headroom). Largest concentration: Top 3 brokers at 76.5%.
> 
> Calculated: Concentration limits vs the governing document.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `risk_limits` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Concentration limits vs the governing document` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> 5 passed, 0 warning(s), 6 breach(es), 1 need review, 3 unavailable. Nearest to limit: Top 3 brokers (-31.5 pp headroom). Largest concentration: Top 3 brokers at 76.5%.
> 
> Calculated: Concentration limits vs the governing document.

grade reason: answered and met every frozen assertion


## CFO74 · CFO91

**Question as typed:** At the current run rate, when do we reach £250m of loans?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `forecast_extrapolation` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Run-rate extrapolation` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `forecast`

> At the current base completion run-rate (~£14.8m/month, £177.7m/year), the book reaches £250.0m around 2026-12 (downside 2027-02, upside 2026-11). Downside/base/upside are indicative scenario bands, not statistically validated confidence intervals.
> 
> Calculated: Run-rate extrapolation.

grade reason: answered and met every frozen assertion

**MERGE ON** — route `forecast_extrapolation` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Run-rate extrapolation` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `forecast`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> At the current base completion run-rate (~£14.8m/month, £177.7m/year), the book reaches £250.0m around 2026-12 (downside 2027-02, upside 2026-11). Downside/base/upside are indicative scenario bands, not statistically validated confidence intervals.
> 
> Calculated: Run-rate extrapolation.

grade reason: answered and met every frozen assertion


## CFO77 · CFO91

**Question as typed:** Which region grew the most?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `REFUSE` · must contain ['period']

**MODEL OFF** — route `period_change` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`

> I can rank region by movement, but this question names no period to compare over, and I have not chosen one for you. Tell me the window — for example “since last month”, “over the last 3 months”, or two named months.

grade reason: refused, as the bank expects

**MERGE ON** — route `period_change` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['geographic_region_obligor', 'geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
concept merge: `applied` (replayed) · applied [{'kind': 'dimension', 'field': 'geographic_region_obligor'}] · conflicts 0

> I can rank region by movement, but this question names no period to compare over, and I have not chosen one for you. Tell me the window — for example “since last month”, “over the last 3 months”, or two named months.

grade reason: refused, as the bank expects


## CFO78 · CFO91

**Question as typed:** What changed?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `REFUSE`

**MODEL OFF** — route `period_change` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`

> I can report what changed, but this question names no period to compare over, and I have not chosen one for you. Tell me the window — for example “since last month”, “over the last 3 months”, or two named months.

grade reason: refused, as the bank expects

**MERGE ON** — route `period_change` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> I can report what changed, but this question names no period to compare over, and I have not chosen one for you. Tell me the window — for example “since last month”, “over the last 3 months”, or two named months.

grade reason: refused, as the bank expects


## CFO79 · CFO91

**Question as typed:** Which region added the most?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `REFUSE`

**MODEL OFF** — route `period_change` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`

> I can rank region by movement, but this question names no period to compare over, and I have not chosen one for you. Tell me the window — for example “since last month”, “over the last 3 months”, or two named months.

grade reason: refused, as the bank expects

**MERGE ON** — route `period_change` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['geographic_region_obligor', 'geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
concept merge: `applied` (replayed) · applied [{'kind': 'dimension', 'field': 'geographic_region_obligor'}] · conflicts 0

> I can rank region by movement, but this question names no period to compare over, and I have not chosen one for you. Tell me the window — for example “since last month”, “over the last 3 months”, or two named months.

grade reason: refused, as the bank expects


## CFO80 · CFO91

**Question as typed:** Show me the trend.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `REFUSE`

**MODEL OFF** — route `evolution` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `Metric evolution` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`

> I can show a trend, but you have not said which metric. For example: funded balance, loan count or weighted-average LTV. No metric has been chosen for you.
> 
> Calculated: Metric evolution.

grade reason: refused, as the bank expects

**MERGE ON** — route `evolution` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `Metric evolution` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> I can show a trend, but you have not said which metric. For example: funded balance, loan count or weighted-average LTV. No metric has been chosen for you.
> 
> Calculated: Metric evolution.

grade reason: refused, as the bank expects


## CFO81 · CFO91

**Question as typed:** How much is in the Highgate Mortgages book?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `REFUSE`

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> I understood that you asked for Highgate Mortgages book, but that could not be applied to the calculation (Highgate Mortgages book — 'Highgate Mortgages book' is not a governed portfolio for this book, so the answer was not narrowed to it). I have not substituted a broader figure.

grade reason: refused, as the bank expects

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> I understood that you asked for Highgate Mortgages book, but that could not be applied to the calculation (Highgate Mortgages book — 'Highgate Mortgages book' is not a governed portfolio for this book, so the answer was not narrowed to it). I have not substituted a broader figure.

grade reason: refused, as the bank expects


## CFO82 · CFO91

**Question as typed:** Show balance by risk grade.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `REFUSE`

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `current_outstanding_balance` · aggregation `sum` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['internal_risk_grade']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`

> 'Risk Grade' is not available in this dataset. This book does not report it, so the question cannot be answered from the current data (no value was fabricated).

grade reason: refused, as the bank expects

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `current_outstanding_balance` · aggregation `sum` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['internal_risk_grade']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> 'Risk Grade' is not available in this dataset. This book does not report it, so the question cannot be answered from the current data (no value was fabricated).

grade reason: refused, as the bank expects


## CFO83 · CFO91

**Question as typed:** What is our arrears rate?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `REFUSE`

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `None` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `funded` · reconciled against `None`

> 'arrears' is not available in this dataset. This book does not report it, so the question cannot be answered from the current data (no value was fabricated).

grade reason: refused, as the bank expects

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `None` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> 'arrears' is not available in this dataset. This book does not report it, so the question cannot be answered from the current data (no value was fabricated).

grade reason: refused, as the bank expects


## CFO84 · CFO91

**Question as typed:** Show the cure rate by vintage.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `REFUSE`

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `funded` · reconciled against `None`

> Cure rate is measured ACROSS governed snapshots and MI Query answers from a single dataset, so it cannot be computed on this path. It is an owned Trakt metric (OBSERVED_CURE_RATE@v1) served by analytics_lib.history.cure_rate; request it through the governed history tools, where the snapshot window is resolved. No value has been computed and no other measure has been substituted for the one you asked about.

grade reason: refused, as the bank expects

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Cure rate is measured ACROSS governed snapshots and MI Query answers from a single dataset, so it cannot be computed on this path. It is an owned Trakt metric (OBSERVED_CURE_RATE@v1) served by analytics_lib.history.cure_rate; request it through the governed history tools, where the snapshot window is resolved. No value has been computed and no other measure has been substituted for the one you asked about.

grade reason: refused, as the bank expects


## CFO85 · CFO91

**Question as typed:** What is the NNEG exposure?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `REFUSE`

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `None` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `funded` · reconciled against `None`

> 'NNEG' is not available in this dataset. This book does not report it, so the question cannot be answered from the current data (no value was fabricated).

grade reason: refused, as the bank expects

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `None` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> 'NNEG' is not available in this dataset. This book does not report it, so the question cannot be answered from the current data (no value was fabricated).

grade reason: refused, as the bank expects


## CFO86 · CFO91

**Question as typed:** Show roll rates by bucket.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `REFUSE`

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `funded` · reconciled against `None`

> Arrears transition / roll analysis is measured ACROSS governed snapshots and MI Query answers from a single dataset, so it cannot be computed on this path. It is an owned Trakt metric (versioned methodology) served by mi_agent.risk_monitor.migration.migration_matrix; request it through the governed history tools, where the snapshot window is resolved. No value has been computed and no other measure has been substituted for the one you asked about.

grade reason: refused, as the bank expects

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['ticket_bucket', 'ticket_bucket']`
scope `None` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
concept merge: `applied` (replayed) · applied [{'kind': 'dimension', 'field': 'ticket_bucket'}] · conflicts 0

> Arrears transition / roll analysis is measured ACROSS governed snapshots and MI Query answers from a single dataset, so it cannot be computed on this path. It is an owned Trakt metric (versioned methodology) served by mi_agent.risk_monitor.migration.migration_matrix; request it through the governed history tools, where the snapshot window is resolved. No value has been computed and no other measure has been substituted for the one you asked about.

grade reason: refused, as the bank expects


## CFO88 · CFO91

**Question as typed:** Compare us with the market.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `REFUSE`

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `funded` · reconciled against `None`

> I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Try a metric by a dimension — e.g. 'balance by region', 'weighted average LTV by ticket size', 'ticket size by borrower type' — a cross-period comparison ('compare October and November'), a scale-up forecast ('run-rate to £100m'), risk limits ('are we within limits?'), or 'portfolio summary'.

grade reason: refused, as the bank expects

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Try a metric by a dimension — e.g. 'balance by region', 'weighted average LTV by ticket size', 'ticket size by borrower type' — a cross-period comparison ('compare October and November'), a scale-up forecast ('run-rate to £100m'), risk limits ('are we within limits?'), or 'portfolio summary'.

grade reason: refused, as the bank expects


## CFO89 · CFO91

**Question as typed:** Show balance by servicer.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `REFUSE`

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `current_outstanding_balance` · aggregation `sum` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`

> I could not build a governed query for this question: bar chart requires a dimension (or x). No substitute figure has been returned.

grade reason: refused, as the bank expects

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: not published by this route
measure `current_outstanding_balance` · aggregation `sum` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> I could not build a governed query for this question: bar chart requires a dimension (or x). No substitute figure has been returned.

grade reason: refused, as the bank expects


## CFO90 · CFO91

**Question as typed:** What will the book be worth in five years?

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `REFUSE`

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `None` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: projection `a forward projection` → lost

> I understood that you asked for a forward projection, but that could not be applied to the calculation (a forward projection — this is a point-in-time calculation; no forward projection was run). I have not substituted a broader figure.

grade reason: refused, as the bank expects

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `TRUE_REFUSAL`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `None` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: projection `a forward projection` → lost
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> I understood that you asked for a forward projection, but that could not be applied to the calculation (a forward projection — this is a point-in-time calculation; no forward projection was run). I have not substituted a broader figure.

grade reason: refused, as the bank expects


## Q01A · BANK75

**Question as typed:** How many loans are to borrowers over 55 with LTV above 50%?

**Truth:** desc=count, age>55 AND ltv>50, count=144

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **144** of 640 (whole book 640)
measure `None` · aggregation `count` · narrowed `True`
filters applied: `['Borrower Age > 55', 'Current LTV > 50']`
spec filters: `{"youngest_borrower_age": {"op": "gt", "value": 55.0}, "current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 55` → applied; threshold `LTV over 50` → applied

> 144 loans · Current Outstanding Balance: £37.1MM.
> 
> Calculated: Count of loans · Borrower Age > 55 · Current LTV > 50 · 144 loans · as at 30 June 2026.

grade reason: count 144 found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **144** of 640 (whole book 640)
measure `None` · aggregation `count` · narrowed `True`
filters applied: `['Borrower Age > 55', 'Current LTV > 50']`
spec filters: `{"youngest_borrower_age": {"op": "gt", "value": 55.0}, "current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 55` → applied; threshold `LTV over 50` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> 144 loans · Current Outstanding Balance: £37.1MM.
> 
> Calculated: Count of loans · Borrower Age > 55 · Current LTV > 50 · 144 loans · as at 30 June 2026.

grade reason: count 144 found


## Q01B · BANK75

**Question as typed:** How many loans have a borrower older than 55 and an LTV greater than 50%?

**Truth:** desc=count, age>55 AND ltv>50, count=144

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: **144** of 640 (whole book 640)
measure `None` · aggregation `count` · narrowed `True`
filters applied: `['Borrower Age > 55', 'Current LTV > 50']`
spec filters: `{"youngest_borrower_age": {"op": "gt", "value": 55.0}, "current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 55` → applied; threshold `LTV over 50` → applied; unresolved_measure `an ltv greater than 50%` → lost

> I understood that you asked for an ltv greater than 50%, but that could not be applied to the calculation (an ltv greater than 50% — this was asked for alongside measures that were calculated, but it is not a governed measure in this dataset, so it was not calculated). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: **144** of 640 (whole book 640)
measure `None` · aggregation `count` · narrowed `True`
filters applied: `['Borrower Age > 55', 'Current LTV > 50']`
spec filters: `{"youngest_borrower_age": {"op": "gt", "value": 55.0}, "current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 55` → applied; threshold `LTV over 50` → applied; unresolved_measure `an ltv greater than 50%` → lost
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> I understood that you asked for an ltv greater than 50%, but that could not be applied to the calculation (an ltv greater than 50% — this was asked for alongside measures that were calculated, but it is not a governed measure in this dataset, so it was not calculated). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal


## Q02A · BANK75

**Question as typed:** What is the balance of loans to borrowers over 75 with LTV above 40%?

**Truth:** desc=balance, age>75 AND ltv>40, count=130, balance=35763779.92

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **130** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Borrower Age > 75', 'Current LTV > 40']`
spec filters: `{"youngest_borrower_age": {"op": "gt", "value": 75.0}, "current_loan_to_value": {"op": "gt", "value": 40.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 75` → applied; threshold `LTV over 40` → applied

> Balance: £35.8MM · 130 loans.
> 
> Calculated: Total Balance · Borrower Age > 75 · Current LTV > 40 · 130 loans · as at 30 June 2026. Interpretation confidence: medium — check the scope above matches your question.

grade reason: count 130 found; balance 35763779.92 found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **130** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Borrower Age > 75', 'Current LTV > 40']`
spec filters: `{"youngest_borrower_age": {"op": "gt", "value": 75.0}, "current_loan_to_value": {"op": "gt", "value": 40.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 75` → applied; threshold `LTV over 40` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Balance: £35.8MM · 130 loans.
> 
> Calculated: Total Balance · Borrower Age > 75 · Current LTV > 40 · 130 loans · as at 30 June 2026. Interpretation confidence: medium — check the scope above matches your question.

grade reason: count 130 found; balance 35763779.92 found


## Q02C · BANK75

**Question as typed:** Show the total balance for loans with borrowers older than 75 and current LTV greater than 40%.

**Truth:** desc=balance, age>75 AND ltv>40, count=130, balance=35763779.92

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **130** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Borrower Age > 75', 'Current LTV > 40']`
spec filters: `{"youngest_borrower_age": {"op": "gt", "value": 75.0}, "current_loan_to_value": {"op": "gt", "value": 40.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 75` → applied; threshold `LTV over 40` → applied

> Balance: £35.8MM · 130 loans.
> 
> Calculated: Total Balance · Borrower Age > 75 · Current LTV > 40 · 130 loans · as at 30 June 2026.

grade reason: count 130 found; balance 35763779.92 found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **130** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Borrower Age > 75', 'Current LTV > 40']`
spec filters: `{"youngest_borrower_age": {"op": "gt", "value": 75.0}, "current_loan_to_value": {"op": "gt", "value": 40.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 75` → applied; threshold `LTV over 40` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Balance: £35.8MM · 130 loans.
> 
> Calculated: Total Balance · Borrower Age > 75 · Current LTV > 40 · 130 loans · as at 30 June 2026.

grade reason: count 130 found; balance 35763779.92 found


## Q03B · BANK75

**Question as typed:** Of the drawdown loans, how many are over 50% LTV?

**Truth:** desc=count, drawdown AND ltv>50, count=45

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **45** of 640 (whole book 640)
measure `None` · aggregation `count` · narrowed `True`
filters applied: `['Product Type = drawdown', 'Current LTV > 50']`
spec filters: `{"erm_product_type": "drawdown", "current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 50` → applied

> 45 loans · Current Outstanding Balance: £11.3MM.
> 
> Calculated: Count of loans · Product Type = drawdown · Current LTV > 50 · 45 loans · as at 30 June 2026.

grade reason: count 45 found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **45** of 640 (whole book 640)
measure `None` · aggregation `count` · narrowed `True`
filters applied: `['Product Type = drawdown', 'Current LTV > 50']`
spec filters: `{"erm_product_type": "drawdown", "current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `LTV over 50` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 1

> 45 loans · Current Outstanding Balance: £11.3MM.
> 
> Calculated: Count of loans · Product Type = drawdown · Current LTV > 50 · 45 loans · as at 30 June 2026.

grade reason: count 45 found


## Q04A · BANK75

**Question as typed:** What is the balance of Direct-book loans in London to borrowers over 75?

**Truth:** desc=balance, Direct book AND London AND age>75, count=24, balance=7201378.77, direct_rows=441, london_rows_in_direct=60

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: **297** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Borrower Age > 75']`
spec filters: `{"youngest_borrower_age": {"op": "gt", "value": 75.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `borrower age over 75` → applied; geographic_scope `London` → lost; lost_narrowing `Direct` → lost

> I understood that you asked for London and Direct, but that could not be applied to the calculation (London (Obligor Region (NUTS3)) — the geographic scope was not applied to the calculation; Direct (Source Portfolio Type) — this narrowing was not applied, so the figure covers the whole book rather than only Direct). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: **32** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Borrower Age > 75', 'London']`
spec filters: `{"youngest_borrower_age": {"op": "gt", "value": 75.0}, "geographic_region_obligor": "London"}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `borrower age over 75` → applied; geographic_scope `London` → applied; lost_narrowing `Direct` → lost
concept merge: `applied` (replayed) · applied [{'kind': 'filter', 'field': 'geographic_region_obligor', 'operator': None, 'value': 'London'}] · conflicts 0

> I understood that you asked for Direct, but that could not be applied to the calculation (Direct (Source Portfolio Type) — this narrowing was not applied, so the figure covers the whole book rather than only Direct). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal


## Q04B · BANK75

**Question as typed:** How much balance is in the Direct portfolio for London loans where the borrower is older than 75?

**Truth:** desc=balance, Direct book AND London AND age>75, count=24, balance=7201378.77, direct_rows=441, london_rows_in_direct=60

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **24** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['London', 'Borrower Age > 75', 'Source Portfolio in direct_001']`
spec filters: `{"geographic_region_obligor": "London", "youngest_borrower_age": {"op": "gt", "value": 75.0}, "source_portfolio_id": ["direct_001"]}`
dimensions applied: `none` · spec dimensions: `none`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `borrower age over 75` → applied; geographic_scope `London` → applied

> Balance: £7.2MM · 24 loans.
> 
> Calculated: Total Balance · London · Borrower Age > 75 · Source Portfolio in direct_001 · 24 loans · as at 30 June 2026.

grade reason: count 24 found; balance 7201378.77 found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **24** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['London', 'Borrower Age > 75', 'Source Portfolio in direct_001']`
spec filters: `{"geographic_region_obligor": "London", "youngest_borrower_age": {"op": "gt", "value": 75.0}, "source_portfolio_id": ["direct_001"]}`
dimensions applied: `none` · spec dimensions: `none`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: threshold `borrower age over 75` → applied; geographic_scope `London` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 1

> Balance: £7.2MM · 24 loans.
> 
> Calculated: Total Balance · London · Borrower Age > 75 · Source Portfolio in direct_001 · 24 loans · as at 30 June 2026.

grade reason: count 24 found; balance 7201378.77 found


## Q05A · BANK75

**Question as typed:** What is the weighted-average LTV of lump sum loans in the Direct book?

**Truth:** desc=WA LTV, Direct book AND lump_sum, count=278, wa_ltv=37.05, balance=73380747.04

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **278** of 640 (whole book 640)
measure `Current LTV` · aggregation `weighted_avg` · narrowed `True`
filters applied: `['Product Type = lump_sum', 'Source Portfolio in direct_001']`
spec filters: `{"erm_product_type": "lump_sum", "source_portfolio_id": ["direct_001"]}`
dimensions applied: `none` · spec dimensions: `none`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the weighted average` → applied

> Weighted-average Current LTV: 37.0% · 278 loans.
> 
> Calculated: Weighted-average Current LTV · Product Type = lump_sum · Source Portfolio in direct_001 · 278 loans · as at 30 June 2026.

grade reason: count 278 found; wa_ltv 37.05% found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **278** of 640 (whole book 640)
measure `Current LTV` · aggregation `weighted_avg` · narrowed `True`
filters applied: `['Product Type = lump_sum', 'Source Portfolio in direct_001']`
spec filters: `{"erm_product_type": "lump_sum", "source_portfolio_id": ["direct_001"]}`
dimensions applied: `none` · spec dimensions: `none`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the weighted average` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 1

> Weighted-average Current LTV: 37.0% · 278 loans.
> 
> Calculated: Weighted-average Current LTV · Product Type = lump_sum · Source Portfolio in direct_001 · 278 loans · as at 30 June 2026.

grade reason: count 278 found; wa_ltv 37.05% found


## Q05B · BANK75

**Question as typed:** For Direct-book lump sum loans, what is the weighted average current LTV?

**Truth:** desc=WA LTV, Direct book AND lump_sum, count=278, wa_ltv=37.05, balance=73380747.04

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: **396** of 640 (whole book 640)
measure `Current LTV` · aggregation `weighted_avg` · narrowed `True`
filters applied: `['Product Type = lump_sum']`
spec filters: `{"erm_product_type": "lump_sum"}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: lost_narrowing `Direct` → lost; requested_statistic `the weighted average` → applied

> I understood that you asked for Direct, but that could not be applied to the calculation (Direct (Source Portfolio Type) — this narrowing was not applied, so the figure covers the whole book rather than only Direct). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: **396** of 640 (whole book 640)
measure `Current LTV` · aggregation `weighted_avg` · narrowed `True`
filters applied: `['Product Type = lump_sum']`
spec filters: `{"erm_product_type": "lump_sum"}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: lost_narrowing `Direct` → lost; requested_statistic `the weighted average` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 1

> I understood that you asked for Direct, but that could not be applied to the calculation (Direct (Source Portfolio Type) — this narrowing was not applied, so the figure covers the whole book rather than only Direct). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal


## Q06A · BANK75

**Question as typed:** Summarise the portfolio.

**Truth:** — no independently computed truth for this case

**Frozen bank expects:** `DELIVER`

**MODEL OFF** — route `portfolio_summary` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Governed analysis` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> At 30 June 2026 the portfolio holds 640 loans with a funded balance of £172.1m. Weighted-average current LTV is 36.3%, the weighted-average interest rate is 6.26% and the average youngest-borrower age is 74.3 years. The largest regional exposures are Scotland (£28.9m, 16.8%), North (£26.8m, 15.6%) and Midlands (£25.1m, 14.6%). By source portfolio: direct_001 £117.4m and acquired_001 £54.7m.
> 
> Calculated: Governed analysis.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `portfolio_summary` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Governed analysis` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> At 30 June 2026 the portfolio holds 640 loans with a funded balance of £172.1m. Weighted-average current LTV is 36.3%, the weighted-average interest rate is 6.26% and the average youngest-borrower age is 74.3 years. The largest regional exposures are Scotland (£28.9m, 16.8%), North (£26.8m, 15.6%) and Midlands (£25.1m, 14.6%). By source portfolio: direct_001 £117.4m and acquired_001 £54.7m.
> 
> Calculated: Governed analysis.

grade reason: no independent truth was computed for this case


## Q06B · BANK75

**Question as typed:** Give me a management summary of the current book.

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `portfolio_summary` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Governed analysis` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> At 30 June 2026 the portfolio holds 640 loans with a funded balance of £172.1m. Weighted-average current LTV is 36.3%, the weighted-average interest rate is 6.26% and the average youngest-borrower age is 74.3 years. The largest regional exposures are Scotland (£28.9m, 16.8%), North (£26.8m, 15.6%) and Midlands (£25.1m, 14.6%). By source portfolio: direct_001 £117.4m and acquired_001 £54.7m.
> 
> Calculated: Governed analysis.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `portfolio_summary` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Governed analysis` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> At 30 June 2026 the portfolio holds 640 loans with a funded balance of £172.1m. Weighted-average current LTV is 36.3%, the weighted-average interest rate is 6.26% and the average youngest-borrower age is 74.3 years. The largest regional exposures are Scotland (£28.9m, 16.8%), North (£26.8m, 15.6%) and Midlands (£25.1m, 14.6%). By source portfolio: direct_001 £117.4m and acquired_001 £54.7m.
> 
> Calculated: Governed analysis.

grade reason: no independent truth was computed for this case


## Q06C · BANK75

**Question as typed:** Give me a concise overview of the funded portfolio.

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `portfolio_summary` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Governed analysis` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> At 30 June 2026 the portfolio holds 640 loans with a funded balance of £172.1m. Weighted-average current LTV is 36.3%, the weighted-average interest rate is 6.26% and the average youngest-borrower age is 74.3 years. The largest regional exposures are Scotland (£28.9m, 16.8%), North (£26.8m, 15.6%) and Midlands (£25.1m, 14.6%). By source portfolio: direct_001 £117.4m and acquired_001 £54.7m.
> 
> Calculated: Governed analysis.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `portfolio_summary` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Governed analysis` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> At 30 June 2026 the portfolio holds 640 loans with a funded balance of £172.1m. Weighted-average current LTV is 36.3%, the weighted-average interest rate is 6.26% and the average youngest-borrower age is 74.3 years. The largest regional exposures are Scotland (£28.9m, 16.8%), North (£26.8m, 15.6%) and Midlands (£25.1m, 14.6%). By source portfolio: direct_001 £117.4m and acquired_001 £54.7m.
> 
> Calculated: Governed analysis.

grade reason: no independent truth was computed for this case


## Q07A · BANK75

**Question as typed:** Compare the Direct and Acquired books.

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `portfolio_risk_comparison` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Total Current Outstanding Balance · Loan Count · Weighted-average Current Loan To Value · Weighted-average Current Interest Rate · Average Youngest Borrower Age` · aggregation `count` · narrowed `False`
filters applied: `['Direct vs Acquired']`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: cohort_comparison `a comparison by how the loans were sourced` → applied

> Direct has higher observed Current Outstanding Balance than Acquired. Acquired has higher observed Current Loan To Value than Direct. Acquired has higher observed Current Interest Rate than Direct. Direct has higher observed Youngest Borrower Age than Acquired.
> 
> Calculated: Total Current Outstanding Balance · Loan Count · Weighted-average Current Loan To Value · Weighted-average Current Interest Rate · Average Youngest Borrower Age · Direct vs Acquired.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `portfolio_risk_comparison` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Total Current Outstanding Balance · Loan Count · Weighted-average Current Loan To Value · Weighted-average Current Interest Rate · Average Youngest Borrower Age` · aggregation `count` · narrowed `False`
filters applied: `['Direct vs Acquired']`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['source_portfolio_type', 'source_portfolio_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: cohort_comparison `a comparison by how the loans were sourced` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'dimension', 'field': 'source_portfolio_type'}] · conflicts 2

> Direct has higher observed Current Outstanding Balance than Acquired. Acquired has higher observed Current Loan To Value than Direct. Acquired has higher observed Current Interest Rate than Direct. Direct has higher observed Youngest Borrower Age than Acquired.
> 
> Calculated: Total Current Outstanding Balance · Loan Count · Weighted-average Current Loan To Value · Weighted-average Current Interest Rate · Average Youngest Borrower Age · Direct vs Acquired.

grade reason: no independent truth was computed for this case


## Q07C · BANK75

**Question as typed:** Give me a side-by-side comparison of Direct versus Acquired.

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `portfolio_risk_comparison` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Total Current Outstanding Balance · Loan Count · Weighted-average Current Loan To Value · Weighted-average Current Interest Rate · Average Youngest Borrower Age` · aggregation `count` · narrowed `False`
filters applied: `['Direct vs Acquired']`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: cohort_comparison `a comparison by how the loans were sourced` → applied

> Direct has higher observed Current Outstanding Balance than Acquired. Acquired has higher observed Current Loan To Value than Direct. Acquired has higher observed Current Interest Rate than Direct. Direct has higher observed Youngest Borrower Age than Acquired.
> 
> Calculated: Total Current Outstanding Balance · Loan Count · Weighted-average Current Loan To Value · Weighted-average Current Interest Rate · Average Youngest Borrower Age · Direct vs Acquired.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `portfolio_risk_comparison` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Total Current Outstanding Balance · Loan Count · Weighted-average Current Loan To Value · Weighted-average Current Interest Rate · Average Youngest Borrower Age` · aggregation `count` · narrowed `False`
filters applied: `['Direct vs Acquired']`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['source_portfolio_type', 'source_portfolio_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: cohort_comparison `a comparison by how the loans were sourced` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'dimension', 'field': 'source_portfolio_type'}] · conflicts 2

> Direct has higher observed Current Outstanding Balance than Acquired. Acquired has higher observed Current Loan To Value than Direct. Acquired has higher observed Current Interest Rate than Direct. Direct has higher observed Youngest Borrower Age than Acquired.
> 
> Calculated: Total Current Outstanding Balance · Loan Count · Weighted-average Current Loan To Value · Weighted-average Current Interest Rate · Average Youngest Borrower Age · Direct vs Acquired.

grade reason: no independent truth was computed for this case


## Q08A · BANK75

**Question as typed:** Where are our largest concentrations today?

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `concentration_analysis` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Exposure concentration` · aggregation `loan_level` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the maximum` → applied

> Largest occupancy type exposure is owner_occupied (92.0% of exposure). Largest exposure currency denomination exposure is GBP (100.0% of exposure). Largest geographic region obligor exposure is Scotland (16.8% of exposure). Largest origination channel exposure is broker (77.8% of exposure). Largest product type exposure is lump_sum (61.4% of exposure).
> 
> Calculated: Exposure concentration.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `concentration_analysis` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Exposure concentration` · aggregation `loan_level` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the maximum` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Largest occupancy type exposure is owner_occupied (92.0% of exposure). Largest exposure currency denomination exposure is GBP (100.0% of exposure). Largest geographic region obligor exposure is Scotland (16.8% of exposure). Largest origination channel exposure is broker (77.8% of exposure). Largest product type exposure is lump_sum (61.4% of exposure).
> 
> Calculated: Exposure concentration.

grade reason: no independent truth was computed for this case


## Q08B · BANK75

**Question as typed:** What are the biggest concentration exposures in the book?

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `concentration_analysis` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Exposure concentration` · aggregation `loan_level` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the maximum` → applied

> Largest occupancy type exposure is owner_occupied (92.0% of exposure). Largest exposure currency denomination exposure is GBP (100.0% of exposure). Largest geographic region obligor exposure is Scotland (16.8% of exposure). Largest origination channel exposure is broker (77.8% of exposure). Largest product type exposure is lump_sum (61.4% of exposure).
> 
> Calculated: Exposure concentration.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `concentration_analysis` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Exposure concentration` · aggregation `loan_level` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: requested_statistic `the maximum` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Largest occupancy type exposure is owner_occupied (92.0% of exposure). Largest exposure currency denomination exposure is GBP (100.0% of exposure). Largest geographic region obligor exposure is Scotland (16.8% of exposure). Largest origination channel exposure is broker (77.8% of exposure). Largest product type exposure is lump_sum (61.4% of exposure).
> 
> Calculated: Exposure concentration.

grade reason: no independent truth was computed for this case


## Q08C · BANK75

**Question as typed:** Summarise the main current portfolio concentrations.

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `concentration_analysis` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Exposure concentration` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> Largest occupancy type exposure is owner_occupied (92.0% of exposure). Largest exposure currency denomination exposure is GBP (100.0% of exposure). Largest geographic region obligor exposure is Scotland (16.8% of exposure). Largest origination channel exposure is broker (77.8% of exposure). Largest product type exposure is lump_sum (61.4% of exposure).
> 
> Calculated: Exposure concentration.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `concentration_analysis` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Exposure concentration` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Largest occupancy type exposure is owner_occupied (92.0% of exposure). Largest exposure currency denomination exposure is GBP (100.0% of exposure). Largest geographic region obligor exposure is Scotland (16.8% of exposure). Largest origination channel exposure is broker (77.8% of exposure). Largest product type exposure is lump_sum (61.4% of exposure).
> 
> Calculated: Exposure concentration.

grade reason: no independent truth was computed for this case


## Q09A · BANK75

**Question as typed:** Are any concentration limits currently breached or close to breach?

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `risk_limits` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Concentration limits vs the governing document` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> 5 passed, 0 warning(s), 6 breach(es), 1 need review, 3 unavailable. Nearest to limit: Top 3 brokers (-31.5 pp headroom). Largest concentration: Top 3 brokers at 76.5%.
> 
> Calculated: Concentration limits vs the governing document.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `risk_limits` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Concentration limits vs the governing document` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> 5 passed, 0 warning(s), 6 breach(es), 1 need review, 3 unavailable. Nearest to limit: Top 3 brokers (-31.5 pp headroom). Largest concentration: Top 3 brokers at 76.5%.
> 
> Calculated: Concentration limits vs the governing document.

grade reason: no independent truth was computed for this case


## Q09B · BANK75

**Question as typed:** Which of our concentration tests are most at risk today?

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `concentration_analysis` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Exposure concentration` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> Largest occupancy type exposure is owner_occupied (92.0% of exposure). Largest exposure currency denomination exposure is GBP (100.0% of exposure). Largest geographic region obligor exposure is Scotland (16.8% of exposure). Largest origination channel exposure is broker (77.8% of exposure). Largest product type exposure is lump_sum (61.4% of exposure).
> 
> Calculated: Exposure concentration.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `concentration_analysis` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Exposure concentration` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Largest occupancy type exposure is owner_occupied (92.0% of exposure). Largest exposure currency denomination exposure is GBP (100.0% of exposure). Largest geographic region obligor exposure is Scotland (16.8% of exposure). Largest origination channel exposure is broker (77.8% of exposure). Largest product type exposure is lump_sum (61.4% of exposure).
> 
> Calculated: Exposure concentration.

grade reason: no independent truth was computed for this case


## Q09C · BANK75

**Question as typed:** Summarise our current position against the concentration limits.

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `risk_limits` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Concentration limits vs the governing document` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> 5 passed, 0 warning(s), 6 breach(es), 1 need review, 3 unavailable. Nearest to limit: Top 3 brokers (-31.5 pp headroom). Largest concentration: Top 3 brokers at 76.5%.
> 
> Calculated: Concentration limits vs the governing document.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `risk_limits` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Concentration limits vs the governing document` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> 5 passed, 0 warning(s), 6 breach(es), 1 need review, 3 unavailable. Nearest to limit: Top 3 brokers (-31.5 pp headroom). Largest concentration: Top 3 brokers at 76.5%.
> 
> Calculated: Concentration limits vs the governing document.

grade reason: no independent truth was computed for this case


## Q10A · BANK75

**Question as typed:** Summarise the current pipeline.

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `portfolio_summary` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Governed analysis` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `pipeline` · reconciled against `funded`

> At 30 June 2026 the portfolio holds 640 loans with a funded balance of £172.1m. Weighted-average current LTV is 36.3%, the weighted-average interest rate is 6.26% and the average youngest-borrower age is 74.3 years. The largest regional exposures are Scotland (£28.9m, 16.8%), North (£26.8m, 15.6%) and Midlands (£25.1m, 14.6%). By source portfolio: direct_001 £117.4m and acquired_001 £54.7m.
> 
> Calculated: Governed analysis.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `portfolio_summary` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Governed analysis` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `pipeline` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> At 30 June 2026 the portfolio holds 640 loans with a funded balance of £172.1m. Weighted-average current LTV is 36.3%, the weighted-average interest rate is 6.26% and the average youngest-borrower age is 74.3 years. The largest regional exposures are Scotland (£28.9m, 16.8%), North (£26.8m, 15.6%) and Midlands (£25.1m, 14.6%). By source portfolio: direct_001 £117.4m and acquired_001 £54.7m.
> 
> Calculated: Governed analysis.

grade reason: no independent truth was computed for this case


## Q10B · BANK75

**Question as typed:** Give me an overview of the pipeline by size and stage.

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: **8** of 8 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Pipeline Stage']` · spec dimensions: `['pipeline_stage']`
scope `None` · scopeApplied `None` · dataset context `pipeline` · reconciled against `pipeline`
facets: grouping_dimension `stage` → applied

> Here is the bar for your query, covering 5 groups.
> 
> Calculated: Total Balance · grouped by Pipeline Stage · 5 groups · 8 loans.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: **8** of 8 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['Ticket Size', 'Pipeline Stage']` · spec dimensions: `['ticket_bucket', 'pipeline_stage']`
scope `None` · scopeApplied `None` · dataset context `pipeline` · reconciled against `pipeline`
facets: grouping_dimension `stage` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'dimension', 'field': 'ticket_bucket'}] · conflicts 0

> Here is the bar for your query, covering 8 groups.
> 
> Calculated: Total Balance · grouped by Ticket Size and Pipeline Stage · 8 groups · 8 loans.

grade reason: no independent truth was computed for this case


## Q10C · BANK75

**Question as typed:** What does the current pipeline look like?

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `pipeline` · reconciled against `None`

> I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Try a metric by a dimension — e.g. 'balance by region', 'weighted average LTV by ticket size', 'ticket size by borrower type' — a cross-period comparison ('compare October and November'), a scale-up forecast ('run-rate to £100m'), risk limits ('are we within limits?'), or 'portfolio summary'.

grade reason: expected an answer, got a refusal

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `pipeline` · reconciled against `None`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Try a metric by a dimension — e.g. 'balance by region', 'weighted average LTV by ticket size', 'ticket size by borrower type' — a cross-period comparison ('compare October and November'), a scale-up forecast ('run-rate to £100m'), risk limits ('are we within limits?'), or 'portfolio summary'.

grade reason: expected an answer, got a refusal


## Q12C · BANK75

**Question as typed:** Plot portfolio balance across LTV buckets and borrower-age buckets.

**Truth:** axes=['ltv_bucket', 'age_bucket'], cells=42, levels={'ltv_bucket': 6, 'age_bucket': 7}, total=172055547.39

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket']` · spec dimensions: `['ltv_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv buckets` → applied; grouping_dimension `age buckets` → lost

> I understood that you asked for age buckets, but that could not be applied to the calculation (age buckets (Age Bucket) — the requested breakdown was not applied). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: **640** of 640 (whole book 640)  ← THE WHOLE BOOK
measure `Balance` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `['LTV Bucket']` · spec dimensions: `['ltv_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv buckets` → applied; grouping_dimension `age buckets` → lost
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> I understood that you asked for age buckets, but that could not be applied to the calculation (age buckets (Age Bucket) — the requested breakdown was not applied). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal


## Q15A · BANK75

**Question as typed:** For the Direct book, show balance by broker and product type.

**Truth:** axes=['broker_channel', 'erm_product_type'], cells=8, levels={'broker_channel': 4, 'erm_product_type': 2}, total=117356785.33

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **441** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Source Portfolio in direct_001']`
spec filters: `{"source_portfolio_id": ["direct_001"]}`
dimensions applied: `['Broker', 'Product Type']` · spec dimensions: `['broker_channel', 'erm_product_type']`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `broker` → applied; grouping_dimension `product type` → applied

> Here is the heatmap for your query, covering 8 groups.
> 
> Calculated: Total Balance · Source Portfolio in direct_001 · grouped by Broker and Product Type · 8 groups · 441 loans · as at 30 June 2026.

grade reason: cells 8 (artefact rows 8) found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **441** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Source Portfolio in direct_001']`
spec filters: `{"source_portfolio_id": ["direct_001"]}`
dimensions applied: `['Broker', 'Product Type']` · spec dimensions: `['broker_channel', 'erm_product_type']`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `broker` → applied; grouping_dimension `product type` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the heatmap for your query, covering 8 groups.
> 
> Calculated: Total Balance · Source Portfolio in direct_001 · grouped by Broker and Product Type · 8 groups · 441 loans · as at 30 June 2026.

grade reason: cells 8 (artefact rows 8) found


## Q15B · BANK75

**Question as typed:** Break Direct-book balance down by both broker channel and loan type.

**Truth:** axes=['broker_channel', 'erm_product_type'], cells=8, levels={'broker_channel': 4, 'erm_product_type': 2}, total=117356785.33

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `current_outstanding_balance` · aggregation `sum` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['broker_channel', 'erm_product_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> I understood that you asked for Break Direct- book, but that could not be applied to the calculation (Break Direct- book — 'Break Direct- book' is not a governed portfolio for this book, so the answer was not narrowed to it). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `current_outstanding_balance` · aggregation `sum` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['broker_channel', 'erm_product_type']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> I understood that you asked for Break Direct- book, but that could not be applied to the calculation (Break Direct- book — 'Break Direct- book' is not a governed portfolio for this book, so the answer was not narrowed to it). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal


## Q15C · BANK75

**Question as typed:** Give me a broker-by-product balance table for the Direct portfolio.

**Truth:** axes=['broker_channel', 'erm_product_type'], cells=8, levels={'broker_channel': 4, 'erm_product_type': 2}, total=117356785.33

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: **441** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Source Portfolio in direct_001']`
spec filters: `{"source_portfolio_id": ["direct_001"]}`
dimensions applied: `['Product Type']` · spec dimensions: `['erm_product_type']`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: unresolved_role `broker` → lost; grouping_dimension `product` → applied

> I could not tell how you meant broker. Did you want the book split by it, or narrowed to one value of it? I have not answered over the whole book in the meantime.

grade reason: expected an answer, got a refusal

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: **441** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Source Portfolio in direct_001']`
spec filters: `{"source_portfolio_id": ["direct_001"]}`
dimensions applied: `['Product Type']` · spec dimensions: `['erm_product_type']`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: unresolved_role `broker` → lost; grouping_dimension `product` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> I could not tell how you meant broker. Did you want the book split by it, or narrowed to one value of it? I have not answered over the whole book in the meantime.

grade reason: expected an answer, got a refusal


## Q16A · BANK75

**Question as typed:** For drawdown loans, show balance by region and LTV bucket.

**Truth:** axes=['geographic_region_obligor', 'ltv_bucket'], cells=39, levels={'geographic_region_obligor': 7, 'ltv_bucket': 6}, total=66671398.95

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **244** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Product Type = drawdown']`
spec filters: `{"erm_product_type": "drawdown"}`
dimensions applied: `['Obligor Region (NUTS3)', 'LTV Bucket']` · spec dimensions: `['geographic_region_obligor', 'ltv_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `region` → applied; grouping_dimension `ltv bucket` → applied

> Here is the heatmap for your query, covering 39 groups.
> 
> Calculated: Total Balance · Product Type = drawdown · grouped by Obligor Region (NUTS3) and LTV Bucket · 39 groups · 244 loans · as at 30 June 2026.

grade reason: cells 39 (artefact rows 39) found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **244** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Product Type = drawdown']`
spec filters: `{"erm_product_type": "drawdown"}`
dimensions applied: `['Obligor Region (NUTS3)', 'LTV Bucket']` · spec dimensions: `['geographic_region_obligor', 'ltv_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `region` → applied; grouping_dimension `ltv bucket` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 1

> Here is the heatmap for your query, covering 39 groups.
> 
> Calculated: Total Balance · Product Type = drawdown · grouped by Obligor Region (NUTS3) and LTV Bucket · 39 groups · 244 loans · as at 30 June 2026.

grade reason: cells 39 (artefact rows 39) found


## Q16C · BANK75

**Question as typed:** Show me the regional balance by LTV bucket for drawdown loans.

**Truth:** axes=['geographic_region_obligor', 'ltv_bucket'], cells=39, levels={'geographic_region_obligor': 7, 'ltv_bucket': 6}, total=66671398.95

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **244** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Product Type = drawdown']`
spec filters: `{"erm_product_type": "drawdown"}`
dimensions applied: `['Obligor Region (NUTS3)', 'LTV Bucket']` · spec dimensions: `['geographic_region_obligor', 'ltv_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `regional` → applied; grouping_dimension `ltv bucket` → applied

> Here is the heatmap for your query, covering 39 groups.
> 
> Calculated: Total Balance · Product Type = drawdown · grouped by Obligor Region (NUTS3) and LTV Bucket · 39 groups · 244 loans · as at 30 June 2026.

grade reason: cells 39 (artefact rows 39) found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **244** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Product Type = drawdown']`
spec filters: `{"erm_product_type": "drawdown"}`
dimensions applied: `['Obligor Region (NUTS3)', 'LTV Bucket']` · spec dimensions: `['geographic_region_obligor', 'ltv_bucket']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `regional` → applied; grouping_dimension `ltv bucket` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 1

> Here is the heatmap for your query, covering 39 groups.
> 
> Calculated: Total Balance · Product Type = drawdown · grouped by Obligor Region (NUTS3) and LTV Bucket · 39 groups · 244 loans · as at 30 June 2026.

grade reason: cells 39 (artefact rows 39) found


## Q17A · BANK75

**Question as typed:** For the Direct book, show balance by LTV bucket, ticket-size bucket and borrower-age bucket.

**Truth:** axes=['ltv_bucket', 'ticket_bucket', 'age_bucket'], cells=143, levels={'ltv_bucket': 6, 'ticket_bucket': 5, 'age_bucket': 7}, total=117356785.33

**MODEL OFF** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **441** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Source Portfolio in direct_001']`
spec filters: `{"source_portfolio_id": ["direct_001"]}`
dimensions applied: `['LTV Bucket', 'Ticket Size', 'Age Bucket']` · spec dimensions: `['ltv_bucket', 'ticket_bucket', 'age_bucket']`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv bucket` → applied; grouping_dimension `ticket` → applied; grouping_dimension `age bucket` → applied

> Here is the result for your query, covering 143 groups.
> 
> Calculated: Total Balance · Source Portfolio in direct_001 · grouped by LTV Bucket, Ticket Size and Age Bucket · 143 groups · 441 loans · as at 30 June 2026.

grade reason: cells 143 (artefact rows 143) found

**MERGE ON** — route `point-in-time` · verdict `answered` · **grade `CORRECT`**
population: **441** of 640 (whole book 640)
measure `Balance` · aggregation `sum` · narrowed `True`
filters applied: `['Source Portfolio in direct_001']`
spec filters: `{"source_portfolio_id": ["direct_001"]}`
dimensions applied: `['LTV Bucket', 'Ticket Size', 'Age Bucket']` · spec dimensions: `['ltv_bucket', 'ticket_bucket', 'age_bucket']`
scope `direct` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: grouping_dimension `ltv bucket` → applied; grouping_dimension `ticket` → applied; grouping_dimension `age bucket` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Here is the result for your query, covering 143 groups.
> 
> Calculated: Total Balance · Source Portfolio in direct_001 · grouped by LTV Bucket, Ticket Size and Age Bucket · 143 groups · 441 loans · as at 30 June 2026.

grade reason: cells 143 (artefact rows 143) found


## Q18A · BANK75

**Question as typed:** How did the book change in the last month?

**Truth:** open_rows=600, close_rows=640, open=149459238.98, close=172055547.39, delta=22596308.41

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied

> Between 31 May 2026 and 30 June 2026, 4 of 4 governed metrics could be compared across both snapshots. Largest observed movements measured in counts: Youngest Borrower Age −0 (74 → 74). Largest observed movements measured in currency: Current Outstanding Balance +£22.6m (£149.5m → £172.1m). Largest observed movements measured in percentage points: Current Loan To Value −1.32 pp (37.58% → 36.26%); Current Interest Rate −0.05 pp (6.31% → 6.26%). Against the registry's directionality, 1 metric(s) moved in the improving direction and 0 in the deteriorating direction. The largest observed composition shifts were Broker Channel — Delta Advisers +6.22 pp of count share; Geographic Region Obligor — Scotland +4.91 pp of count share; ERM Product Type — lump_sum +4.38 pp of count share. The balance bridge reconciles: opening £149.5m +£83.7m new lending, −£66.7m exits, +£5.6m on continuing loans, closing £172.1m. No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: delta 22596308.41 found

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Between 31 May 2026 and 30 June 2026, 4 of 4 governed metrics could be compared across both snapshots. Largest observed movements measured in counts: Youngest Borrower Age −0 (74 → 74). Largest observed movements measured in currency: Current Outstanding Balance +£22.6m (£149.5m → £172.1m). Largest observed movements measured in percentage points: Current Loan To Value −1.32 pp (37.58% → 36.26%); Current Interest Rate −0.05 pp (6.31% → 6.26%). Against the registry's directionality, 1 metric(s) moved in the improving direction and 0 in the deteriorating direction. The largest observed composition shifts were Broker Channel — Delta Advisers +6.22 pp of count share; Geographic Region Obligor — Scotland +4.91 pp of count share; ERM Product Type — lump_sum +4.38 pp of count share. The balance bridge reconciles: opening £149.5m +£83.7m new lending, −£66.7m exits, +£5.6m on continuing loans, closing £172.1m. No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: delta 22596308.41 found


## Q18B · BANK75

**Question as typed:** What changed in the portfolio since last month?

**Truth:** open_rows=600, close_rows=640, open=149459238.98, close=172055547.39, delta=22596308.41

**MODEL OFF** — route `period_movement` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Month-on-month movement` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied

> Funded balances increased by £22.6m during the month. The largest single regional contribution came from the Scotland (+£12.4m). Weighted-average LTV fell to 36.3%, while average borrower age was unchanged at 74.3 years. Direct_001 contributed approximately +£12.4m and acquired_001 contributed approximately +£10.2m of the movement.
> 
> Calculated: Month-on-month movement.

grade reason: delta 22596308.41 found

**MERGE ON** — route `period_movement` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Month-on-month movement` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Funded balances increased by £22.6m during the month. The largest single regional contribution came from the Scotland (+£12.4m). Weighted-average LTV fell to 36.3%, while average borrower age was unchanged at 74.3 years. Direct_001 contributed approximately +£12.4m and acquired_001 contributed approximately +£10.2m of the movement.
> 
> Calculated: Month-on-month movement.

grade reason: delta 22596308.41 found


## Q18C · BANK75

**Question as typed:** Give me a summary of how the funded book moved over the last month.

**Truth:** open_rows=600, close_rows=640, open=149459238.98, close=172055547.39, delta=22596308.41

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied

> Between 31 May 2026 and 30 June 2026, 4 of 4 governed metrics could be compared across both snapshots. Largest observed movements measured in counts: Youngest Borrower Age −0 (74 → 74). Largest observed movements measured in currency: Current Outstanding Balance +£22.6m (£149.5m → £172.1m). Largest observed movements measured in percentage points: Current Loan To Value −1.32 pp (37.58% → 36.26%); Current Interest Rate −0.05 pp (6.31% → 6.26%). Against the registry's directionality, 1 metric(s) moved in the improving direction and 0 in the deteriorating direction. The largest observed composition shifts were Broker Channel — Delta Advisers +6.22 pp of count share; Geographic Region Obligor — Scotland +4.91 pp of count share; ERM Product Type — lump_sum +4.38 pp of count share. The balance bridge reconciles: opening £149.5m +£83.7m new lending, −£66.7m exits, +£5.6m on continuing loans, closing £172.1m. No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: delta 22596308.41 found

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Between 31 May 2026 and 30 June 2026, 4 of 4 governed metrics could be compared across both snapshots. Largest observed movements measured in counts: Youngest Borrower Age −0 (74 → 74). Largest observed movements measured in currency: Current Outstanding Balance +£22.6m (£149.5m → £172.1m). Largest observed movements measured in percentage points: Current Loan To Value −1.32 pp (37.58% → 36.26%); Current Interest Rate −0.05 pp (6.31% → 6.26%). Against the registry's directionality, 1 metric(s) moved in the improving direction and 0 in the deteriorating direction. The largest observed composition shifts were Broker Channel — Delta Advisers +6.22 pp of count share; Geographic Region Obligor — Scotland +4.91 pp of count share; ERM Product Type — lump_sum +4.38 pp of count share. The balance bridge reconciles: opening £149.5m +£83.7m new lending, −£66.7m exits, +£5.6m on continuing loans, closing £172.1m. No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: delta 22596308.41 found


## Q19B · BANK75

**Question as typed:** What changed in the Direct portfolio since last month?

**Truth:** open_rows=424, close_rows=441, open=104990413.93, close=117356785.33, delta=12366371.4

**MODEL OFF** — route `period_movement` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Month-on-month movement` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `direct` · scopeApplied `direct_001` · dataset context `funded` · reconciled against `funded`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied

> Funded balances increased by £12.4m during the month. The largest single regional contribution came from the Scotland (+£7.8m). Weighted-average LTV fell to 36.2%, while average borrower age decreased slightly to 74.3 years.
> 
> Calculated: Month-on-month movement.

grade reason: delta 12366371.40 found

**MERGE ON** — route `period_movement` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Month-on-month movement` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `direct` · scopeApplied `direct_001` · dataset context `funded` · reconciled against `funded`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Funded balances increased by £12.4m during the month. The largest single regional contribution came from the Scotland (+£7.8m). Weighted-average LTV fell to 36.2%, while average borrower age decreased slightly to 74.3 years.
> 
> Calculated: Month-on-month movement.

grade reason: delta 12366371.40 found


## Q19C · BANK75

**Question as typed:** Summarise the month-on-month movement in the Direct book.

**Truth:** open_rows=424, close_rows=441, open=104990413.93, close=117356785.33, delta=12366371.4

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `direct` · scopeApplied `direct_001` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (period on period)` → applied; granularity `month` → applied

> Between 31 May 2026 and 30 June 2026, 1 of 1 governed metrics could be compared across both snapshots. Largest observed movements measured in currency: Current Outstanding Balance +£12.4m (£105.0m → £117.4m). No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: delta 12366371.40 found

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `direct` · scopeApplied `direct_001` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (period on period)` → applied; granularity `month` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> Between 31 May 2026 and 30 June 2026, 1 of 1 governed metrics could be compared across both snapshots. Largest observed movements measured in currency: Current Outstanding Balance +£12.4m (£105.0m → £117.4m). No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: delta 12366371.40 found


## Q20A · BANK75

**Question as typed:** How did drawdown loans change last month?

**Truth:** open_rows=255, close_rows=244, open=62115206.98, close=66671398.95, delta=4556191.97

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{"erm_product_type": "drawdown"}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied; row_population `the population erm_product_type = drawdown` → applied

> Between 31 May 2026 and 30 June 2026, 4 of 4 governed metrics could be compared across both snapshots. Largest observed movements measured in counts: Youngest Borrower Age +0 (74 → 74). Largest observed movements measured in currency: Current Outstanding Balance +£4.6m (£62.1m → £66.7m). Largest observed movements measured in percentage points: Current Loan To Value −1.69 pp (36.28% → 34.59%); Current Interest Rate +0.02 pp (6.29% → 6.31%). Against the registry's directionality, 1 metric(s) moved in the improving direction and 0 in the deteriorating direction. The largest observed composition shifts were Broker Channel — Gamma Direct -7.33 pp of count share; Geographic Region Obligor — London -6.17 pp of count share; Broker Channel — Beta Partners +5.43 pp of count share. The balance bridge reconciles: opening £62.1m +£51.8m new lending, −£49.2m exits, +£2.0m on continuing loans, closing £66.7m. No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: delta 4556191.97 found

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{"erm_product_type": "drawdown"}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied; row_population `the population erm_product_type = drawdown` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 1

> Between 31 May 2026 and 30 June 2026, 4 of 4 governed metrics could be compared across both snapshots. Largest observed movements measured in counts: Youngest Borrower Age +0 (74 → 74). Largest observed movements measured in currency: Current Outstanding Balance +£4.6m (£62.1m → £66.7m). Largest observed movements measured in percentage points: Current Loan To Value −1.69 pp (36.28% → 34.59%); Current Interest Rate +0.02 pp (6.29% → 6.31%). Against the registry's directionality, 1 metric(s) moved in the improving direction and 0 in the deteriorating direction. The largest observed composition shifts were Broker Channel — Gamma Direct -7.33 pp of count share; Geographic Region Obligor — London -6.17 pp of count share; Broker Channel — Beta Partners +5.43 pp of count share. The balance bridge reconciles: opening £62.1m +£51.8m new lending, −£49.2m exits, +£2.0m on continuing loans, closing £66.7m. No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: delta 4556191.97 found


## Q20B · BANK75

**Question as typed:** What changed in the drawdown book since last month?

**Truth:** open_rows=255, close_rows=244, open=62115206.98, close=66671398.95, delta=4556191.97

**MODEL OFF** — route `period_movement` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `Month-on-month movement` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{"erm_product_type": "drawdown"}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied; row_population `the population erm_product_type = drawdown` → lost

> I understood that you asked for loans where Product Type is drawdown, but that could not be applied to the calculation (loans where Product Type is drawdown — this analytical route calculated across the whole book; it did not narrow to the requested population). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal

**MERGE ON** — route `period_movement` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `Month-on-month movement` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{"erm_product_type": "drawdown"}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: comparison_period `comparison period (last month)` → applied; granularity `month` → applied; row_population `the population erm_product_type = drawdown` → lost
concept merge: `no_change` (replayed) · applied [] · conflicts 1

> I understood that you asked for loans where Product Type is drawdown, but that could not be applied to the calculation (loans where Product Type is drawdown — this analytical route calculated across the whole book; it did not narrow to the requested population). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal


## Q20C · BANK75

**Question as typed:** Summarise the month-on-month movement for drawdown loans.

**Truth:** open_rows=255, close_rows=244, open=62115206.98, close=66671398.95, delta=4556191.97

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{"erm_product_type": "drawdown"}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (period on period)` → applied; granularity `month` → applied; row_population `the population erm_product_type = drawdown` → applied

> Between 31 May 2026 and 30 June 2026, 1 of 1 governed metrics could be compared across both snapshots. Largest observed movements measured in currency: Current Outstanding Balance +£4.6m (£62.1m → £66.7m). No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: delta 4556191.97 found

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{"erm_product_type": "drawdown"}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: comparison_period `comparison period (period on period)` → applied; granularity `month` → applied; row_population `the population erm_product_type = drawdown` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 1

> Between 31 May 2026 and 30 June 2026, 1 of 1 governed metrics could be compared across both snapshots. Largest observed movements measured in currency: Current Outstanding Balance +£4.6m (£62.1m → £66.7m). No governed materiality threshold is configured for this portfolio, so movements are ranked by observed size only, within each unit of measurement. No movement is described as material, significant, a breach or high risk.
> 
> Calculated: Governed period change.

grade reason: delta 4556191.97 found


## Q21A · BANK75

**Question as typed:** Which region added the most balance last month for loans with LTV above 50%?

**Truth:** open_rows=159, close_rows=144, top_region=Scotland, top_open=4482489.52, top_close=6631018.74, top_delta=2148529.22, runners=[{'region': 'North', 'delta': 1726745.9}, {'region': 'London', 'delta': 546215.15}]

**MODEL OFF** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `['Geographic Region Obligor']` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: threshold `LTV over 50` → applied; comparison_period `comparison period (last month)` → applied; ranking `ranking by region` → applied; grouping_dimension `region` → applied; granularity `month` → applied; row_population `the population current_loan_to_value gt 50.0` → applied

> Between 31 May 2026 and 30 June 2026, ranked by absolute balance movement across Geographic Region Obligor, Scotland increased the most: Scotland £4.5m → £6.6m (+£2.1m, +47.9%). Then North £4.7m → £6.4m (+£1.7m, +36.7%); London £5.4m → £6.0m (+£546k, +10.1%). 4 further categories did not increase on this basis and are not listed.
> 
> Calculated: Governed period change · ranked by Geographic Region Obligor · absolute balance movement, largest increases first · 2026-05-31 → 2026-06-30.

grade reason: top region Scotland found

**MERGE ON** — route `period_change_analysis` · verdict `answered` · **grade `CORRECT`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `['Geographic Region Obligor']` · spec dimensions: `['geographic_region_obligor', 'geographic_region_obligor']`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: threshold `LTV over 50` → applied; comparison_period `comparison period (last month)` → applied; ranking `ranking by region` → applied; grouping_dimension `region` → applied; granularity `month` → applied; row_population `the population current_loan_to_value gt 50.0` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'dimension', 'field': 'geographic_region_obligor'}] · conflicts 0

> Between 31 May 2026 and 30 June 2026, ranked by absolute balance movement across Geographic Region Obligor, Scotland increased the most: Scotland £4.5m → £6.6m (+£2.1m, +47.9%). Then North £4.7m → £6.4m (+£1.7m, +36.7%); London £5.4m → £6.0m (+£546k, +10.1%). 4 further categories did not increase on this basis and are not listed.
> 
> Calculated: Governed period change · ranked by Geographic Region Obligor · absolute balance movement, largest increases first · 2026-05-31 → 2026-06-30.

grade reason: top region Scotland found


## Q21C · BANK75

**Question as typed:** Among loans with current LTV above 50%, where did balance increase the most over the last month?

**Truth:** open_rows=159, close_rows=144, top_region=Scotland, top_open=4482489.52, top_close=6631018.74, top_delta=2148529.22, runners=[{'region': 'North', 'delta': 1726745.9}, {'region': 'London', 'delta': 546215.15}]

**MODEL OFF** — route `period_change_analysis` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: threshold `LTV over 50` → applied; comparison_period `comparison period (last month)` → applied; granularity `month` → applied; row_population `the population current_loan_to_value gt 50.0` → applied

> No loans in this book match that filter ('among'), so there is nothing to calculate. I have not returned a whole-book figure in its place.

grade reason: expected an answer, got a refusal

**MERGE ON** — route `period_change_analysis` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `Governed period change` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{"current_loan_to_value": {"op": "gt", "value": 50.0}}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
facets: threshold `LTV over 50` → applied; comparison_period `comparison period (last month)` → applied; granularity `month` → applied; row_population `the population current_loan_to_value gt 50.0` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> No loans in this book match that filter ('among'), so there is nothing to calculate. I have not returned a whole-book figure in its place.

grade reason: expected an answer, got a refusal


## Q22A · BANK75

**Question as typed:** Which source portfolio contributed most to balance growth last month?

**Truth:** direct={'open': 104990413.93, 'close': 117356785.33, 'delta': 12366371.4}, acquired={'open': 44468825.05, 'close': 54698762.06, 'delta': 10229937.01}, larger=direct

**MODEL OFF** — route `funded_bridge` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Funded balance bridge` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: comparison_period `comparison period (last month)` → applied; ranking `ranking by source portfolio` → applied; grouping_dimension `source portfolio` → applied; granularity `month` → applied

> Source Portfolio bridge (Total): funded balance moved from £149.5m in 2026-05 to £172.1m at 2026-06 (latest) — a net change of +£22.6m (up). Largest mover: direct_001 (+£12.4m).
> 
> Calculated: Funded balance bridge.

grade reason: truth carries no checkable figure

**MERGE ON** — route `funded_bridge` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Funded balance bridge` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
facets: comparison_period `comparison period (last month)` → applied; ranking `ranking by source portfolio` → applied; grouping_dimension `source portfolio` → applied; granularity `month` → applied
concept merge: `applied` (replayed) · applied [{'kind': 'measure', 'field': 'current_outstanding_balance'}] · conflicts 0

> Source Portfolio bridge (Total): funded balance moved from £149.5m in 2026-05 to £172.1m at 2026-06 (latest) — a net change of +£22.6m (up). Largest mover: direct_001 (+£12.4m).
> 
> Calculated: Funded balance bridge.

grade reason: truth carries no checkable figure


## Q22B · BANK75

**Question as typed:** Did Direct or Acquired add more balance during the last month?

**Truth:** direct={'open': 104990413.93, 'close': 117356785.33, 'delta': 12366371.4}, acquired={'open': 44468825.05, 'close': 54698762.06, 'delta': 10229937.01}, larger=direct

**MODEL OFF** — route `analytical_composition` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `Composed governed capabilities` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded_history`
facets: comparison_period `comparison period (last month)` → applied; grouping_dimension `direct or acquired` → lost

> I understood that you asked for direct or acquired, but that could not be applied to the calculation (direct or acquired — this answer covers the whole population; it is neither narrowed to nor broken down by direct or acquired). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal

**MERGE ON** — route `analytical_composition` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `Composed governed capabilities` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded_history`
facets: comparison_period `comparison period (last month)` → applied; grouping_dimension `direct or acquired` → lost
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> I understood that you asked for direct or acquired, but that could not be applied to the calculation (direct or acquired — this answer covers the whole population; it is neither narrowed to nor broken down by direct or acquired). I have not substituted a broader figure.

grade reason: expected an answer, got a refusal


## Q23B · BANK75

**Question as typed:** At the current trajectory, when do we get to £100 million?

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `funded` · reconciled against `None`

> I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Try a metric by a dimension — e.g. 'balance by region', 'weighted average LTV by ticket size', 'ticket size by borrower type' — a cross-period comparison ('compare October and November'), a scale-up forecast ('run-rate to £100m'), risk limits ('are we within limits?'), or 'portfolio summary'.

grade reason: expected an answer, got a refusal

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Try a metric by a dimension — e.g. 'balance by region', 'weighted average LTV by ticket size', 'ticket size by borrower type' — a cross-period comparison ('compare October and November'), a scale-up forecast ('run-rate to £100m'), risk limits ('are we within limits?'), or 'portfolio summary'.

grade reason: expected an answer, got a refusal


## Q23C · BANK75

**Question as typed:** When does the funded book reach the £100m milestone?

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `forecast_extrapolation` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Run-rate extrapolation` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `forecast`

> The book has already reached £100.0m (current funded balance £172.1m).
> 
> Calculated: Run-rate extrapolation.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `forecast_extrapolation` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Run-rate extrapolation` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `forecast`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> The book has already reached £100.0m (current funded balance £172.1m).
> 
> Calculated: Run-rate extrapolation.

grade reason: no independent truth was computed for this case


## Q24A · BANK75

**Question as typed:** At the current run rate, when will we reach £250m?

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `forecast_extrapolation` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Run-rate extrapolation` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `forecast`
facets: projection `a forward projection` → applied

> At the current base completion run-rate (~£14.8m/month, £177.7m/year), the book reaches £250.0m around 2026-12 (downside 2027-02, upside 2026-11). Downside/base/upside are indicative scenario bands, not statistically validated confidence intervals.
> 
> Calculated: Run-rate extrapolation.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `forecast_extrapolation` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Run-rate extrapolation` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `forecast`
facets: projection `a forward projection` → applied
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> At the current base completion run-rate (~£14.8m/month, £177.7m/year), the book reaches £250.0m around 2026-12 (downside 2027-02, upside 2026-11). Downside/base/upside are indicative scenario bands, not statistically validated confidence intervals.
> 
> Calculated: Run-rate extrapolation.

grade reason: no independent truth was computed for this case


## Q24B · BANK75

**Question as typed:** When are we expected to get to £250 million of funded loans?

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `funded` · reconciled against `None`

> I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Try a metric by a dimension — e.g. 'balance by region', 'weighted average LTV by ticket size', 'ticket size by borrower type' — a cross-period comparison ('compare October and November'), a scale-up forecast ('run-rate to £100m'), risk limits ('are we within limits?'), or 'portfolio summary'.

grade reason: expected an answer, got a refusal

**MERGE ON** — route `point-in-time` · verdict `refused` · **grade `FALSE_REFUSAL`**
population: not published by this route
measure `None` · aggregation `count` · narrowed `None`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `None` · scopeApplied `None` · dataset context `funded` · reconciled against `None`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Try a metric by a dimension — e.g. 'balance by region', 'weighted average LTV by ticket size', 'ticket size by borrower type' — a cross-period comparison ('compare October and November'), a scale-up forecast ('run-rate to £100m'), risk limits ('are we within limits?'), or 'portfolio summary'.

grade reason: expected an answer, got a refusal


## Q24C · BANK75

**Question as typed:** Based on the current run rate, when does the book reach £250m?

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `forecast_extrapolation` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Run-rate extrapolation` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `forecast`

> At the current base completion run-rate (~£14.8m/month, £177.7m/year), the book reaches £250.0m around 2026-12 (downside 2027-02, upside 2026-11). Downside/base/upside are indicative scenario bands, not statistically validated confidence intervals.
> 
> Calculated: Run-rate extrapolation.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `forecast_extrapolation` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Run-rate extrapolation` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `forecast`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> At the current base completion run-rate (~£14.8m/month, £177.7m/year), the book reaches £250.0m around 2026-12 (downside 2027-02, upside 2026-11). Downside/base/upside are indicative scenario bands, not statistically validated confidence intervals.
> 
> Calculated: Run-rate extrapolation.

grade reason: no independent truth was computed for this case


## Q25A · BANK75

**Question as typed:** Do we expect to breach any concentration tests?

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `risk_limits` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Concentration limits vs the governing document` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> 5 passed, 0 warning(s), 6 breach(es), 1 need review, 3 unavailable. Nearest to limit: Top 3 brokers (-31.5 pp headroom). Largest concentration: Top 3 brokers at 76.5%.
> 
> Calculated: Concentration limits vs the governing document.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `risk_limits` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Concentration limits vs the governing document` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> 5 passed, 0 warning(s), 6 breach(es), 1 need review, 3 unavailable. Nearest to limit: Top 3 brokers (-31.5 pp headroom). Largest concentration: Top 3 brokers at 76.5%.
> 
> Calculated: Concentration limits vs the governing document.

grade reason: no independent truth was computed for this case


## Q25B · BANK75

**Question as typed:** Are any concentration limits likely to be breached as the book grows?

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `risk_limits` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Concentration limits vs the governing document` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`

> 5 passed, 0 warning(s), 6 breach(es), 1 need review, 3 unavailable. Nearest to limit: Top 3 brokers (-31.5 pp headroom). Largest concentration: Top 3 brokers at 76.5%.
> 
> Calculated: Concentration limits vs the governing document.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `risk_limits` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Concentration limits vs the governing document` · aggregation `count` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `none`
scope `total` · scopeApplied `None` · dataset context `funded` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> 5 passed, 0 warning(s), 6 breach(es), 1 need review, 3 unavailable. Nearest to limit: Top 3 brokers (-31.5 pp headroom). Largest concentration: Top 3 brokers at 76.5%.
> 
> Calculated: Concentration limits vs the governing document.

grade reason: no independent truth was computed for this case


## Q25C · BANK75

**Question as typed:** Based on the current book and forward pipeline, which concentration tests are we at risk of breaching?

**Truth:** — no independently computed truth for this case

**MODEL OFF** — route `risk_limits` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Concentration limits vs the governing document` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['collateral_geography']`
scope `total` · scopeApplied `None` · dataset context `pipeline` · reconciled against `funded`

> 5 passed, 0 warning(s), 6 breach(es), 1 need review, 3 unavailable. Nearest to limit: Top 3 brokers (-31.5 pp headroom). Largest concentration: Top 3 brokers at 76.5%.
> 
> Calculated: Concentration limits vs the governing document.

grade reason: no independent truth was computed for this case

**MERGE ON** — route `risk_limits` · verdict `answered` · **grade `NO_COMPUTABLE_TRUTH`**
population: not published by this route
measure `Concentration limits vs the governing document` · aggregation `sum` · narrowed `False`
filters applied: `none`
spec filters: `{}`
dimensions applied: `none` · spec dimensions: `['collateral_geography']`
scope `total` · scopeApplied `None` · dataset context `pipeline` · reconciled against `funded`
concept merge: `no_change` (replayed) · applied [] · conflicts 0

> 5 passed, 0 warning(s), 6 breach(es), 1 need review, 3 unavailable. Nearest to limit: Top 3 brokers (-31.5 pp headroom). Largest concentration: Top 3 brokers at 76.5%.
> 
> Calculated: Concentration limits vs the governing document.

grade reason: no independent truth was computed for this case
