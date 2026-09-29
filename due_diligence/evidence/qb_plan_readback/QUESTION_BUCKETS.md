# The question bank in three buckets — DRAFT for owner review

The pass mark for the proof stage is **every *must answer* question answered correctly, and no question answered wrongly** — not 135 out of 135. A clear "I can't answer that" or "did you mean…?" is a correct outcome for a question in the third bucket.

Machine-readable copy: `qb_question_buckets.json` (the full-bank run is scored against it).

| bucket | questions | new path answers today | falls to old path / declines |
|---|---|---|---|
| Must answer — correctly, every time | 88 | 76 | 12 |
| Nice to have | 31 | 20 | 11 |
| Fine to decline or ask back | 16 | 0 | 16 |

"Today" is the latest check of each question: 64 were re-asked on today's builds; the rest are from this morning's full run, before batches 1–2, and are re-checked by the next full run.

## The must-answer questions the new path does not answer yet (12)

| # | question | what it needs |
|---|---|---|
| 49, 134 | pipeline conversion rate; offer-to-completion pull-through | **Not built.** Conversion is owned by stage movement (D2a); the one real build item in this bucket |
| 82, 83 | latest vs prior pipeline; pipeline growth October to November | **A gap.** The pipeline path gives the figure at each date but not the change between them |
| 112 | current completion run rate | Answered correctly today by the old path; the new path's hold is settled before the old path is switched off |
| 122, 128 | time to (securitisation) scale | **Your configuration, not code:** the portfolio's stage must be set (D9) before 'scale' has a number |
| 62 | pipeline stage distribution | The model asks back 'by amount or by number of cases?' — acceptable, or default to amount |
| 69, 76, 132, 135 | this month's completions; expected completions by month; weighted pipeline; when cases complete | Last checked **before** batch 1 added these concepts — expected to be answered now; the full run confirms |

## Where I'd most like your view

- **Conversion / pull-through (49, 134): must answer?** It is the standard mortgage-pipeline KPI, and the only item here that needs new build (D2a). If it can wait, it moves to nice to have.
- **Week-on-week and month-on-month pipeline change (82, 83): must answer?** I think yes — it is a routine management question.
- **The strat tables (region, broker, LTV band, borrower age, product, property type, single/joint, account status):** I have put the balance and risk cuts (WA LTV) in *must*, and the secondary measures of each (average balance, WA rate by region …) in *nice*. Move any you disagree with.
- **Method questions (98–101, 124):** I have treated 'which stages use historical rates' and similar as fine to decline — they are about how the forecast is built, which the Forecast tab's notes show. A model-risk reviewer might want them; say if so.

## Must answer — correctly, every time (88)

| # | question | why | status |
|---|---|---|---|
| 1 | What is the funded balance? | Headline balance | ✓ new path |
| 2 | What is the current funded balance? | Headline balance | ✓ new path |
| 3 | What is total current outstanding balance? | Headline balance | ✓ new path |
| 4 | How many funded loans are there? | Loan count | ✓ new path |
| 5 | How many loans are in the funded book? | Loan count | ✓ new path |
| 6 | What is the weighted average current LTV? | WA LTV — the key ERM risk figure | ✓ new path |
| 7 | What is WA LTV? | WA LTV | ✓ new path |
| 8 | What is the weighted average interest rate? | WA interest rate | ✓ new path |
| 9 | What is WA current interest rate? | WA interest rate | ✓ new path |
| 11 | What is the average borrower age? | Borrower age drives ERM risk | ✓ new path |
| 12 | What is the weighted average borrower age? | WA age is a standard pool statistic | ✓ new path |
| 13 | What is the average loan balance? | Average loan size | ✓ new path |
| 15 | What is the largest loan? | Largest loan — single-name concentration | ✓ new path |
| 17 | What is the total original balance? | Original balance — standard pool figure | ✓ new path |
| 18 | What is the current valuation amount? | Property value — collateral cover | ✓ new path |
| 20 | What is the weighted average current valuation? | The dashboard's valuation tile (D10) | ✓ new path |
| 21 | Show balance by region. | Standard strat table | ✓ new path |
| 22 | Show funded balance by collateral region. | Same as balance by region (property location) | ✓ new path |
| 24 | Show loan count by region. | Standard strat table | ✓ new path |
| 26 | Show WA LTV by region. | Risk concentration by region | ✓ new path |
| 29 | Show balance by broker. | Broker concentration | ✓ new path |
| 34 | Show balance by LTV bucket. | Standard strat table | ✓ new path |
| 35 | Show loan count by LTV bucket. | Standard strat table | ✓ new path |
| 37 | Show balance by borrower age bucket. | Standard ERM strat table | ✓ new path |
| 39 | Show WA LTV by borrower age bucket. | LTV by age is the core ERM (no-negative-equity) risk view | ✓ new path |
| 41 | Show balance by product type. | Standard strat table | ✓ new path |
| 42 | Show balance by account status. | Performance / status of the book | ✓ new path |
| 43 | Show balance by borrower structure. | Single vs joint — standard ERM strat | ✓ new path |
| 44 | Show balance by property type. | Standard ERM strat table | ✓ new path |
| 46 | What is the pipeline amount? | Headline pipeline | ✓ new path |
| 47 | How many pipeline cases are there? | Headline pipeline | ✓ new path |
| 48 | What is the weighted expected funded amount? | Weighted pipeline — the Pipeline tab's headline | ✓ new path |
| 49 | What is the pipeline conversion rate? | Conversion is a core pipeline KPI — NOT BUILT yet (D2a) | ↺ old path / declined |
| 50 | What is the pipeline amount by stage? | Pipeline by stage | ✓ new path |
| 51 | What is the pipeline case count by stage? | Pipeline by stage | ✓ new path |
| 52 | What is the pipeline amount by broker? | Broker concentration | ✓ new path |
| 53 | What is the pipeline amount by region? | Pipeline by region | ✓ new path |
| 54 | What is the pipeline amount by expected completion month? | When the pipeline lands | ✓ new path |
| 55 | Which broker has the largest pipeline? | Top broker | ✓ new path |
| 56 | Which stage has the largest pipeline? | Largest stage | ✓ new path |
| 57 | How much pipeline is expected to complete next month? | Next month's completions | ✓ new path |
| 58 | How much pipeline is overdue? | Overdue pipeline | ✓ new path |
| 59 | How much pipeline is current month? | This month's completions | ✓ new path |
| 61 | Show pipeline by expected completion date. | When the pipeline lands | ✓ new path |
| 62 | Show pipeline stage distribution. | Same as pipeline by stage | ↺ old path / declined |
| 63 | Show pipeline weighted expected amount by stage. | Weighted pipeline by stage | ✓ new path |
| 65 | Show pipeline amount by product. | Pipeline by product | ✓ new path |
| 66 | Show pipeline amount by LTV bucket. | Pipeline by LTV band | ✓ new path |
| 69 | Show current month pipeline completions expected. | This month's completions | ↺ old path / declined |
| 70 | Show pipeline amount evolution by week. | Pipeline trend | ✓ new path |
| 71 | Show pipeline amount evolution by month. | Pipeline trend | ✓ new path |
| 72 | Show pipeline case count evolution by week. | Pipeline trend | ✓ new path |
| 73 | Show pipeline case count evolution by month. | Pipeline trend | ✓ new path |
| 76 | Show expected completions by month. | When the pipeline lands | ↺ old path / declined |
| 77 | Show weighted expected funded amount by month. | Weighted pipeline by month | ✓ new path |
| 80 | Show pipeline by stage for October and November. | Month-on-month pipeline | ✓ new path |
| 81 | Compare October and November pipeline amount. | Month-on-month pipeline | ✓ new path |
| 82 | Compare latest pipeline with prior pipeline. | Week-on-week change — a routine management question | ↺ old path / declined |
| 83 | Show pipeline growth from October to November. | Month-on-month growth — a routine management question | ↺ old path / declined |
| 85 | What is the forecast funded balance? | Forecast funded balance | ✓ new path |
| 86 | What is funded balance plus weighted pipeline? | Forecast funded balance | ✓ new path |
| 87 | What is the expected funded balance? | Forecast funded balance | ✓ new path |
| 88 | What is the weighted expected pipeline contribution? | Pipeline's part of the forecast | ✓ new path |
| 94 | How much of the forecast comes from funded book? | The forecast bridge | ✓ new path |
| 95 | How much of the forecast comes from pipeline? | The forecast bridge | ✓ new path |
| 96 | What is the forecast bridge? | The forecast bridge | ✓ new path |
| 97 | Show funded vs pipeline contribution. | The forecast bridge | ✓ new path |
| 107 | When do we reach £25m funded balance? | Scale milestone | ✓ new path |
| 108 | When do we reach £50m funded balance? | Scale milestone | ✓ new path |
| 109 | When do we reach £75m funded balance? | Scale milestone | ✓ new path |
| 110 | When do we reach £100m funded balance? | Scale milestone | ✓ new path |
| 111 | When do we reach £150m funded balance? | Scale milestone | ✓ new path |
| 112 | What is the current completion run rate? | Run-rate — the old path answers it today; the new path is on hold | ↺ old path / declined |
| 113 | What is the annualised completion run rate? | Run-rate | ✓ new path |
| 114 | Show the funded balance extrapolation curve. | Scale-up curve | ✓ new path |
| 115 | Show the scale-up forecast. | Scale-up curve | ✓ new path |
| 116 | What is the downside forecast? | Scenario band | ✓ new path |
| 117 | What is the base forecast? | Scenario band | ✓ new path |
| 118 | What is the upside forecast? | Scenario band | ✓ new path |
| 122 | What is the expected time to securitisation scale? | Time to securitisation scale — needs the portfolio's stage set (D9) | ↺ old path / declined |
| 123 | Show milestone dates to funding thresholds. | Milestone table | ✓ new path |
| 127 | Project funded balance over the next twelve months. | 12-month projection | ✓ new path |
| 128 | When does the book reach scale? | Time to scale — needs the portfolio's stage set (D9) | ↺ old path / declined |
| 129 | Show pipeline amount by product. | Pipeline by product | ✓ new path |
| 130 | Show pipeline amount by LTV band. | Pipeline by LTV band | ✓ new path |
| 132 | What is the weighted expected pipeline? | Weighted pipeline | ↺ old path / declined |
| 134 | What is the offer to completion pull-through rate? | Pull-through is the standard mortgage pipeline KPI — NOT BUILT yet (D2a) | ↺ old path / declined |
| 135 | When are pipeline cases expected to complete? | When the pipeline lands | ↺ old path / declined |

## Nice to have (31)

| # | question | why | status |
|---|---|---|---|
| 10 | What is the simple average interest rate? | The weighted average is the standard; a simple average is secondary | ✓ new path |
| 14 | What is the median loan balance? | Useful, secondary to the average | ✓ new path |
| 16 | What is the smallest loan? | Rarely decision-relevant | ✓ new path |
| 19 | What is the average current valuation? | Secondary to the weighted average | ✓ new path |
| 25 | Show average loan balance by region. | Secondary cut | ✓ new path |
| 27 | Show WA interest rate by region. | Secondary cut | ✓ new path |
| 28 | Show average borrower age by region. | Secondary cut | ✓ new path |
| 30 | Show loan count by broker. | Secondary cut | ✓ new path |
| 31 | Show average loan balance by broker. | Secondary cut | ✓ new path |
| 32 | Show WA LTV by broker. | Secondary cut | ✓ new path |
| 33 | Show WA interest rate by broker. | Secondary cut | ✓ new path |
| 36 | Show WA interest rate by LTV bucket. | Secondary cut | ✓ new path |
| 38 | Show loan count by borrower age bucket. | Secondary cut | ✓ new path |
| 40 | Show balance by origination channel. | Secondary cut | ✓ new path |
| 64 | Show pipeline weighted expected amount by broker. | Secondary cut | ✓ new path |
| 74 | Show pipeline amount by stage over time. | Analyst trend view | ↺ old path / declined |
| 75 | Show pipeline cases by stage over time. | Analyst trend view | ↺ old path / declined |
| 79 | Show pipeline by broker over time. | Analyst trend view | ↺ old path / declined |
| 84 | Show stage migration over time. | Stage migration — analyst view (stage movement is built) | ↺ old path / declined |
| 89 | What is the forecast loan count? | Secondary forecast figure | ✓ new path |
| 90 | Show forecast balance by region. | Secondary forecast cut | ✓ new path |
| 91 | Show forecast balance by broker. | Not on the Forecast tab | ↺ old path / declined |
| 92 | Show forecast balance by LTV bucket. | Secondary forecast cut | ✓ new path |
| 93 | Show forecast balance by expected completion month. | Secondary forecast cut | ↺ old path / declined |
| 103 | How much pipeline is excluded because of missing probability? | Forecast disclosure detail | ✓ new path |
| 119 | How much pipeline is needed to reach £100m? | A new calculation (pipeline needed for a target) | ↺ old path / declined |
| 120 | What happens if completion run rate falls by 25%? | What-if scenario — NOT BUILT (D2b) | ↺ old path / declined |
| 121 | What completion rate is assumed from KFI to completion? | The forecast's assumed conversion — NOT BUILT (D2a) | ↺ old path / declined |
| 125 | What is the 8-week completion run rate? | Run-rate variant | ↺ old path / declined |
| 126 | What is the 12-week completion run rate? | Run-rate variant | ↺ old path / declined |
| 131 | What is the pipeline case count by product? | Secondary cut | ✓ new path |

## Fine to decline or ask back (16)

| # | question | why | status |
|---|---|---|---|
| 23 | Show balance by obligor region. | The book does not hold the borrower's region; declining is correct | ↺ old path / declined |
| 45 | Show balance by occupancy type. | The book does not hold occupancy; ERM loans are owner-occupied by design | ↺ old path / declined |
| 60 | Show overdue pipeline cases. | Asks for a list of cases; the agent gives totals (the overdue amount), not case lists | ↺ old path / declined |
| 67 | Show pipeline amount by age bucket. | Unclear whose age (borrower or case) — asking back is right | ↺ old path / declined |
| 68 | Show pipeline amount by source file. | A technical field (which file), not an MI question | ↺ old path / declined |
| 78 | Show pipeline conversion basis over time. | Method detail over time — not an MI question | ↺ old path / declined |
| 98 | Show forecast completion basis. | Vague ('basis' of what?) — asking back is right | ↺ old path / declined |
| 99 | Show historical conversion basis. | Method detail, not an MI figure | ↺ old path / declined |
| 100 | Which stages use historical rates? | Method configuration detail | ↺ old path / declined |
| 101 | Which stages use config fallback rates? | Method configuration detail | ↺ old path / declined |
| 102 | How much active pipeline is excluded from weighting? | Analyst jargon ('active' exclusions) | ↺ old path / declined |
| 104 | How much pipeline is excluded because of withdrawn status? | Analyst jargon; 'pipeline by stage' covers it | ↺ old path / declined |
| 105 | Show forecast balance by stage. | The funded book has no stage — asking back is right | ↺ old path / declined |
| 106 | Show forecast balance by broker and stage. | The funded book has no stage or broker split in the forecast | ↺ old path / declined |
| 124 | Compare current weighted pipeline forecast with run-rate extrapolation. | A method comparison, not an MI figure | ↺ old path / declined |
| 133 | How much of the pipeline has lapsed past its stage window? | Analyst jargon ('stage window') | ↺ old path / declined |

