# MI Agent — Analytical Intent V1: ten runs traced end to end

*Extracted from `due_diligence/MI_AGENT_ANALYTICAL_INTENT_V1.md` §13.2–13.3.
Branch `claude/mi-analytical-capability-layer-vlkjfw`, commit `8c9d04e`.*

Ten of the 752 measured runs, followed from the words the user typed to the
figure returned, with an independent recomputation of every number alongside.

**What to look for.** Each trace has eight steps, and each step states where its
content came from:

| Marker | Meaning |
|---|---|
| **RECORDED** | read verbatim out of the measured run file — not recomputed, not reconstructed |
| **DERIVED** | the boundary's own classifier re-run on the same question. `intent.classify` reads nothing but the question and the parse, so this is what the boundary saw |
| **TRUTH** | computed inside the trace from the fixture CSVs with pandas, referencing nothing the agent produced |

The **agrees** column compares the delivered figure against the TRUTH column. It
is the whole point of the exercise: a trace where the narrative reads well and
the numbers do not agree is a failure, however plausible the prose.

---

### 13.2 Ten runs traced end to end

Ten measured runs, chosen to cover every distinct mechanism rather than
ten easy wins: five per book, two under the forced-model arm, one refusal,
two answered with no analytical plan at all, and the three-resolver control
in full. Each is the FIRST recorded run of that variation — not a
hand-picked repeat.

Provenance is stated per section, because that is what makes the trace
auditable rather than decorative:

* **RECORDED** — read verbatim out of the measured run file; not recomputed.
* **DERIVED** — the boundary's own classifier re-run on the same question.
  `intent.classify` reads nothing but the question and the parse, so this is
  what the boundary saw.
* **TRUTH** — computed in the trace from the fixture CSVs with pandas,
  referencing nothing the agent produced.

### Q1.1 — alderbridge / production
*Chosen because: the governed lending ruling: NEW = L1M, on the question that failed 100% of the time in the baseline.*

**1. Question** (RECORDED)

> How has the profile of our new lending changed over the last few months?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `llm` |
| mode detail | `llm_repaired` |
| LLM failure | `None` |
| repeats in this arm | 5, all reaching `analytical_composition` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | MIX_PROFILE, MOVEMENT_TREND, VINTAGE_COHORT |
| operations | SNAPSHOT, CHANGE, ATTRIBUTION, DELTA, EVOLUTION |
| concept signals matched | change_trend |
| lending windows named | new |
| lending role | population |
| structural requirements | period_comparison |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | months_on_book le 1 |
| predicates unavailable | none |
| rows before → after | 11035 → 115 |
| periods compared | 2026-04-30, 2026-06-30 |
| narrowed to | — |

**5. Deterministic capabilities invoked** (RECORDED)

| | |
|---|---|
| route | `analytical_composition` |
| plan intent | `origination_profile_change` |
| capabilities | `period_movement`, `population_profile` |
| required finding kinds | movement |
| composition version | 1.0.0 · plan origin deterministic |

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

| # | kind | measure | population (rows) | delivered | **independent truth** | agrees |
|---|---|---|---|---|---|---|
| 1 | movement | `current_outstanding_balance` | New lending (last 1 month) (115) | £18.3m | **£18.3m** | ✅ |
| 1p | prior | `current_outstanding_balance` @ 2026-04-30 | New lending (last 1 month) (143) | £21.4m | **£21.4m** | ✅ |
| 2 | movement | `current_loan_to_value` | New lending (last 1 month) (115) | 34.7259% | **34.7259%** | ✅ |
| 2p | prior | `current_loan_to_value` @ 2026-04-30 | New lending (last 1 month) (143) | 34.4568% | **34.4568%** | ✅ |
| 3 | movement | `current_interest_rate` | New lending (last 1 month) (115) | 6.3391% | **6.3391%** | ✅ |
| 3p | prior | `current_interest_rate` @ 2026-04-30 | New lending (last 1 month) (143) | 6.4597% | **6.4597%** | ✅ |

*(plus 15 composition / cohort / forecast / timing findings on this run, reconciled in §12.4 but not tabulated here)*

**8. Final answer** (RECORDED, verbatim)

> Across 2026-04-30 → 2026-06-30, New lending (last 1 month), 115 loans: Current Outstanding Balance £21.4m → £18.3m (−£3.2m); Current Loan To Value 34.46% → 34.73% (+0.27pp); Current Interest Rate 6.46% → 6.34% (-0.12pp). Current profile: Region — South East 43.7%, London 23.4%, South West 11.4%; LTV band — 30-40% 47.7%, 20-30% 25.0%, 40-50% 23.5%; Borrower age band — 70-75 36.7%, 65-70 30.6%, 60-65 12.7%; Ticket size — 200-300k 31.9%, 150-200k 26.5%, 100-150k 22.4%; Interest rate band — 6-7% 50.7%, 5-6% 25.7%, 7-8% 16.9%.

Calculated: Composed governed capabilities.

| | |
|---|---|
| ok | `True` |
| controlled refusal | `False` |
| semantic guard | `ok` [('comparison_period', 'applied'), ('grouping_dimension', 'applied')] |

---

### Q1.2 — alderbridge / forced_llm
*Chosen because: the same family reached by a different window (RECENT = L3M) and a different phrasing, under a forced model parse.*

**1. Question** (RECORDED)

> Are we originating different types of loans now compared with a few months ago?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `llm` |
| mode detail | `llm_repaired` |
| LLM failure | `None` |
| repeats in this arm | 5, all reaching `analytical_composition` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | MIX_PROFILE, MOVEMENT_TREND, VINTAGE_COHORT |
| operations | SNAPSHOT, COMPARISON, CHANGE, ATTRIBUTION, DIVERGENCE, DELTA, RANKING, EVOLUTION |
| concept signals matched | comparison, change_trend |
| lending windows named | recent |
| lending role | population |
| structural requirements | period_comparison |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | months_on_book le 3 |
| predicates unavailable | none |
| rows before → after | 11035 → 258 |
| periods compared | 2026-04-30, 2026-06-30 |
| narrowed to | — |

**5. Deterministic capabilities invoked** (RECORDED)

| | |
|---|---|
| route | `analytical_composition` |
| plan intent | `origination_profile_change` |
| capabilities | `period_movement`, `population_profile` |
| required finding kinds | movement |
| composition version | 1.0.0 · plan origin deterministic |

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

| # | kind | measure | population (rows) | delivered | **independent truth** | agrees |
|---|---|---|---|---|---|---|
| 1 | movement | `current_outstanding_balance` | Recent lending (last 3 months) (258) | £39.7m | **£39.7m** | ✅ |
| 1p | prior | `current_outstanding_balance` @ 2026-04-30 | Recent lending (last 3 months) (367) | £53.2m | **£53.2m** | ✅ |
| 2 | movement | `current_loan_to_value` | Recent lending (last 3 months) (258) | 34.7274% | **34.7274%** | ✅ |
| 2p | prior | `current_loan_to_value` @ 2026-04-30 | Recent lending (last 3 months) (367) | 34.1750% | **34.1750%** | ✅ |
| 3 | movement | `current_interest_rate` | Recent lending (last 3 months) (258) | 6.3961% | **6.3961%** | ✅ |
| 3p | prior | `current_interest_rate` @ 2026-04-30 | Recent lending (last 3 months) (367) | 6.4768% | **6.4768%** | ✅ |

*(plus 15 composition / cohort / forecast / timing findings on this run, reconciled in §12.4 but not tabulated here)*

**8. Final answer** (RECORDED, verbatim)

> Across 2026-04-30 → 2026-06-30, Recent lending (last 3 months), 258 loans: Current Outstanding Balance £53.2m → £39.7m (−£13.5m); Current Loan To Value 34.17% → 34.73% (+0.55pp); Current Interest Rate 6.48% → 6.40% (-0.08pp). Current profile: Region — South East 37.6%, London 22.7%, South West 12.8%; LTV band — 30-40% 49.3%, 20-30% 25.4%, 40-50% 22.5%; Borrower age band — 70-75 34.4%, 65-70 29.9%, 60-65 15.2%; Ticket size — 200-300k 30.8%, 150-200k 23.0%, 100-150k 23.0%; Interest rate band — 6-7% 51.2%, 5-6% 23.9%, 7-8% 19.5%.

Calculated: Composed governed capabilities.

| | |
|---|---|
| ok | `True` |
| controlled refusal | `False` |
| semantic guard | `None` [] |

---

### Q3.1 — alderbridge / production
*Chosen because: a DIFFERENT GOVERNED DATASET — the pipeline extract, not the loan tape.*

**1. Question** (RECORDED)

> How much do we currently have at offer and how much of it is likely to complete?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `llm` |
| mode detail | `validation_failed` |
| LLM failure | `parse_failure` |
| repeats in this arm | 5, all reaching `analytical_composition` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | PIPELINE, FORECAST_PROJECTION |
| operations | CONVERSION, RUN_RATE, EXPECTED_COMPLETION, PROJECT_VALUE |
| concept signals matched | forecast |
| lending windows named | — |
| lending role | — (none named) |
| structural requirements | pipeline_dataset, period_comparison, forecast |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | — |
| predicates unavailable | none |
| rows before → after | None → None |
| periods compared | — |
| narrowed to | [{"field": "pipeline_stage", "value": "OFFER", "rows": 157, "dataset": "pipeline"}] |

**5. Deterministic capabilities invoked** (RECORDED)

| | |
|---|---|
| route | `analytical_composition` |
| plan intent | `pipeline_offer_outlook` |
| capabilities | `pipeline_stock`, `pipeline_completion_forecast` |
| required finding kinds | measure, forecast |
| composition version | 1.0.0 · plan origin deterministic |

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

| # | kind | measure | population (rows) | delivered | **independent truth** | agrees |
|---|---|---|---|---|---|---|

*(plus 3 composition / cohort / forecast / timing findings on this run, reconciled in §12.4 but not tabulated here)*

**8. Final answer** (RECORDED, verbatim)

> Offer stage pipeline is £29.4m across 157 case(s) as at 2026-06-29. Expected completion amount from pipeline cases at Offer stage: £5.0m. Expected to land: 2026-07 £5.0m.

Calculated: Composed governed capabilities.

| | |
|---|---|
| ok | `True` |
| controlled refusal | `False` |
| semantic guard | `ok` [('row_population', 'unsupported')] |

---

### Q4.2 — alderbridge / production
*Chosen because: THE FAIL-CLOSED REFUSAL. Baseline answered '11,035 loans'.*

**1. Question** (RECORDED)

> How many loans are we completing at the moment?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `llm` |
| mode detail | `llm_repaired` |
| LLM failure | `None` |
| repeats in this arm | 3, all reaching `None` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | PIPELINE |
| operations | MOVEMENT, RUN_RATE |
| concept signals matched | — |
| lending windows named | — |
| lending role | — (none named) |
| structural requirements | pipeline_dataset, period_comparison |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | — |
| predicates unavailable | none |
| rows before → after | None → None |
| periods compared | — |
| narrowed to | — |

**5. Deterministic capabilities invoked** (RECORDED)

**None — no analytical plan was built.** Route: `(none: the point-in-time path)`. This is the correct outcome for this question; see the note below.

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

**No findings — nothing was computed.** That is the point of this trace.

**8. Final answer** (RECORDED, verbatim)

> I understood this as a pipeline question, but I have not answered it: this asks about the pipeline (applications, offers or completions), which is a different governed dataset from the funded book; and this asks how something changed, which needs two governed reporting snapshots to compare. I have NOT substituted a current-position figure, because that would answer a different question from the one you asked. Ask for the governed analytic directly — for example the concentration limit tests, the completion run-rate, the pipeline at a named stage, or a named measure compared across two reporting periods — and I will compute it.

| | |
|---|---|
| ok | `False` |
| controlled refusal | `True` |
| semantic guard | `refuse` [('pipeline_dataset', 'unavailable'), ('period_comparison', 'unavailable')] |

---

### Q6.2 — kestrelmoor / production
*Chosen because: governed FLAG SETTLING: no analytical plan at all, the existing limits route answers. Second book.*

**1. Question** (RECORDED)

> Where are we closest to our limits?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `llm` |
| mode detail | `llm` |
| LLM failure | `None` |
| repeats in this arm | 3, all reaching `risk_limits` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | LIMITS_CONCENTRATION |
| operations | STATUS, HEADROOM, RANKING, CONCENTRATION |
| concept signals matched | limits |
| lending windows named | — |
| lending role | — (none named) |
| structural requirements | limit_evidence |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | — |
| predicates unavailable | none |
| rows before → after | None → None |
| periods compared | — |
| narrowed to | — |

**5. Deterministic capabilities invoked** (RECORDED)

**None — no analytical plan was built.** Route: `risk_limits`. This is the correct outcome for this question; see the note below.

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

**No findings — nothing was computed.** That is the point of this trace.

**8. Final answer** (RECORDED, verbatim)

> Contractual risk limits are unavailable for this portfolio (No Schedule 8 limits available — extraction required.). I can show observed concentrations once limits are provided.

Calculated: Concentration limits vs the governing document.

| | |
|---|---|
| ok | `True` |
| controlled refusal | `False` |
| semantic guard | `None` [] |

---

### Q7.2 — alderbridge / production
*Chosen because: an ASYMMETRIC pair — a segment against a window (Back Book vs RECENT).*

**1. Question** (RECORDED)

> Are older loans riskier than the loans we've originated recently?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `llm` |
| mode detail | `llm_repaired` |
| LLM failure | `None` |
| repeats in this arm | 5, all reaching `analytical_composition` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | MIX_PROFILE, VINTAGE_COHORT |
| operations | SNAPSHOT, COMPARISON, DIVERGENCE |
| concept signals matched | comparison |
| lending windows named | back_book, recent |
| lending role | population |
| structural requirements | population_comparison |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | seasoning_segment = Back Book, months_on_book le 3 |
| predicates unavailable | none |
| rows before → after | 11035 → 9858 |
| periods compared | — |
| narrowed to | [{"field": "seasoning_segment", "value": "Back Book", "rows": 9858, "dataset": "funded"}] |

**5. Deterministic capabilities invoked** (RECORDED)

| | |
|---|---|
| route | `analytical_composition` |
| plan intent | `vintage_risk_comparison` |
| capabilities | `portfolio_snapshot`, `portfolio_snapshot`, `vintage_analysis` |
| required finding kinds | comparison, cohort |
| composition version | 1.0.0 · plan origin deterministic |

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

| # | kind | measure | population (rows) | delivered | **independent truth** | agrees |
|---|---|---|---|---|---|---|
| 1 | measure | `loan_count` | Back Book (13+ months) (9858) | 9,858 | **9,858** | ✅ |
| 2 | measure | `current_outstanding_balance` | Back Book (13+ months) (9858) | £1.79bn | **£1.79bn** | ✅ |
| 3 | measure | `current_loan_to_value` | Back Book (13+ months) (9858) | 43.9657% | **43.9657%** | ✅ |
| 4 | measure | `current_interest_rate` | Back Book (13+ months) (9858) | 6.5699% | **6.5699%** | ✅ |
| 5 | measure | `youngest_borrower_age` | Back Book (13+ months) (9858) | 71.7988 | **71.7988** | ✅ |
| 6 | measure | `loan_count` | Recent lending (last 3 months) (258) | 258 | **258** | ✅ |
| 7 | measure | `current_outstanding_balance` | Recent lending (last 3 months) (258) | £39.7m | **£39.7m** | ✅ |
| 8 | measure | `current_loan_to_value` | Recent lending (last 3 months) (258) | 34.7274% | **34.7274%** | ✅ |
| 9 | measure | `current_interest_rate` | Recent lending (last 3 months) (258) | 6.3961% | **6.3961%** | ✅ |
| 10 | measure | `youngest_borrower_age` | Recent lending (last 3 months) (258) | 68.3527 | **68.3527** | ✅ |
| 11 | comparison | `loan_count` | Back Book (13+ months) (9858) | 9,858 | **9,858** | ✅ |
| 11c | comparand | `loan_count` | Recent lending (last 3 months) (258) | 258 | **258** | ✅ |
| 12 | comparison | `current_outstanding_balance` | Back Book (13+ months) (9858) | £1.79bn | **£1.79bn** | ✅ |
| 12c | comparand | `current_outstanding_balance` | Recent lending (last 3 months) (258) | £39.7m | **£39.7m** | ✅ |
| 13 | comparison | `current_loan_to_value` | Back Book (13+ months) (9858) | 43.9657% | **43.9657%** | ✅ |
| 13c | comparand | `current_loan_to_value` | Recent lending (last 3 months) (258) | 34.7274% | **34.7274%** | ✅ |
| 14 | comparison | `current_interest_rate` | Back Book (13+ months) (9858) | 6.5699% | **6.5699%** | ✅ |
| 14c | comparand | `current_interest_rate` | Recent lending (last 3 months) (258) | 6.3961% | **6.3961%** | ✅ |
| 15 | comparison | `youngest_borrower_age` | Back Book (13+ months) (9858) | 71.7988 | **71.7988** | ✅ |
| 15c | comparand | `youngest_borrower_age` | Recent lending (last 3 months) (258) | 68.3527 | **68.3527** | ✅ |

*(plus 13 composition / cohort / forecast / timing findings on this run, reconciled in §12.4 but not tabulated here)*

**8. Final answer** (RECORDED, verbatim)

> Back Book (13+ months) against Recent lending (last 3 months) (9,858 vs 258 loans): Loan Count 9,858 vs 258 (+9,600); Current Outstanding Balance £1.79bn vs £39.7m (+£1.75bn); Current Loan To Value 43.97% vs 34.73% (+9.24pp); Current Interest Rate 6.57% vs 6.40% (+0.17pp); Youngest Borrower Age 71.8 vs 68.4 (+3.4). Across 13 governed origination vintage(s), 2014 holds £67.6m at 54.47% weighted-average LTV and 2026 holds £71.8m at 34.59%.

Calculated: Composed governed capabilities.

| | |
|---|---|
| ok | `True` |
| controlled refusal | `False` |
| semantic guard | `None` [] |

---

### Q8.3/provenance — kestrelmoor / production
*Chosen because: §8 control, resolver 1 of 3.*

**1. Question** (RECORDED)

> Are direct and acquired balances developing differently over time?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `llm` |
| mode detail | `llm_repaired` |
| LLM failure | `None` |
| repeats in this arm | 5, all reaching `analytical_composition` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | MOVEMENT_TREND |
| operations | DELTA, TREND, RANKING, ATTRIBUTION |
| concept signals matched | comparison, change_trend |
| lending windows named | — |
| lending role | — (none named) |
| structural requirements | period_comparison |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | — |
| predicates unavailable | none |
| rows before → after | None → None |
| periods compared | 2026-04-30, 2026-06-30 |
| narrowed to | — |

**5. Deterministic capabilities invoked** (RECORDED)

| | |
|---|---|
| route | `analytical_composition` |
| plan intent | `population_movement_comparison` |
| capabilities | `period_movement`, `period_movement` |
| required finding kinds | movement, comparison |
| composition version | 1.0.0 · plan origin deterministic |

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

| # | kind | measure | population (rows) | delivered | **independent truth** | agrees |
|---|---|---|---|---|---|---|
| 1 | movement | `current_outstanding_balance` | Direct (5612) | £558.0m | **£558.0m** | ✅ |
| 1p | prior | `current_outstanding_balance` @ 2026-04-30 | Direct (5302) | £530.4m | **£530.4m** | ✅ |
| 2 | movement | `current_outstanding_balance` | Acquired (6643) | £1.21bn | **£1.21bn** | ✅ |
| 2p | prior | `current_outstanding_balance` @ 2026-04-30 | Acquired (6425) | £1.17bn | **£1.17bn** | ✅ |
| 3 | comparison | `current_outstanding_balance` | Direct (5612) | £558.0m | **£558.0m** | ✅ |
| 3c | comparand | `current_outstanding_balance` | Acquired (6643) | £1.21bn | **£1.21bn** | ✅ |

**8. Final answer** (RECORDED, verbatim)

> Across 2026-04-30 → 2026-06-30, Direct, 5,612 loans: Current Outstanding Balance £530.4m → £558.0m (+£27.7m). Across 2026-04-30 → 2026-06-30, Acquired, 6,643 loans: Current Outstanding Balance £1.17bn → £1.21bn (+£40.0m). Direct against Acquired (5,612 vs 6,643 loans): Current Outstanding Balance £558.0m vs £1.21bn (−£656.4m).

Calculated: Composed governed capabilities.

| | |
|---|---|
| ok | `True` |
| controlled refusal | `False` |
| semantic guard | `None` [] |

---

### Q8.3/seasoning — kestrelmoor / production
*Chosen because: §8 control, resolver 2 of 3.*

**1. Question** (RECORDED)

> Are the front book and the back book balances developing differently over time?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `llm` |
| mode detail | `llm_repaired` |
| LLM failure | `None` |
| repeats in this arm | 5, all reaching `analytical_composition` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | MOVEMENT_TREND, VINTAGE_COHORT |
| operations | DELTA, TREND, RANKING, ATTRIBUTION, SNAPSHOT, COMPARISON, DIVERGENCE, EVOLUTION |
| concept signals matched | comparison, change_trend |
| lending windows named | front_book, back_book |
| lending role | population |
| structural requirements | period_comparison, population_comparison |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | seasoning_segment = Front Book, seasoning_segment = Back Book |
| predicates unavailable | none |
| rows before → after | 12255 → 3020 |
| periods compared | 2026-04-30, 2026-06-30 |
| narrowed to | [{"field": "seasoning_segment", "value": "Front Book", "rows": 3020, "dataset": "funded"}, {"field": "seasoning_segment", "value": "Back Book", "rows": 9235, "dataset": "funded"}] |

**5. Deterministic capabilities invoked** (RECORDED)

| | |
|---|---|
| route | `analytical_composition` |
| plan intent | `population_movement_comparison` |
| capabilities | `period_movement`, `period_movement` |
| required finding kinds | movement, comparison |
| composition version | 1.0.0 · plan origin deterministic |

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

| # | kind | measure | population (rows) | delivered | **independent truth** | agrees |
|---|---|---|---|---|---|---|
| 1 | movement | `current_outstanding_balance` | Front Book (0-12 months) (3020) | £299.9m | **£299.9m** | ✅ |
| 1p | prior | `current_outstanding_balance` @ 2026-04-30 | Front Book (0-12 months) (2626) | £262.4m | **£262.4m** | ✅ |
| 2 | movement | `current_outstanding_balance` | Back Book (13+ months) (9235) | £1.47bn | **£1.47bn** | ✅ |
| 2p | prior | `current_outstanding_balance` @ 2026-04-30 | Back Book (13+ months) (9101) | £1.44bn | **£1.44bn** | ✅ |
| 3 | comparison | `current_outstanding_balance` | Front Book (0-12 months) (3020) | £299.9m | **£299.9m** | ✅ |
| 3c | comparand | `current_outstanding_balance` | Back Book (13+ months) (9235) | £1.47bn | **£1.47bn** | ✅ |

**8. Final answer** (RECORDED, verbatim)

> Across 2026-04-30 → 2026-06-30, Front Book (0-12 months), 3,020 loans: Current Outstanding Balance £262.4m → £299.9m (+£37.6m). Across 2026-04-30 → 2026-06-30, Back Book (13+ months), 9,235 loans: Current Outstanding Balance £1.44bn → £1.47bn (+£30.1m). Front Book (0-12 months) against Back Book (13+ months) (3,020 vs 9,235 loans): Current Outstanding Balance £299.9m vs £1.47bn (−£1.17bn).

Calculated: Composed governed capabilities.

| | |
|---|---|
| ok | `True` |
| controlled refusal | `False` |
| semantic guard | `ok` [('grouping_dimension', 'applied')] |

---

### Q8.3/dimension_value — kestrelmoor / production
*Chosen because: §8 control, resolver 3 of 3.*

**1. Question** (RECORDED)

> Are North West and Scotland balances developing differently over time?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `deterministic_fallback_after_llm_failure` |
| mode detail | `deterministic_fallback` |
| LLM failure | `parse_failure` |
| repeats in this arm | 5, all reaching `analytical_composition` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | MOVEMENT_TREND |
| operations | DELTA, TREND, RANKING, ATTRIBUTION |
| concept signals matched | comparison, change_trend |
| lending windows named | — |
| lending role | — (none named) |
| structural requirements | period_comparison |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | collateral_geography = North West, collateral_geography = Scotland |
| predicates unavailable | none |
| rows before → after | 12255 → 2935 |
| periods compared | 2026-04-30, 2026-06-30 |
| narrowed to | [{"field": "collateral_geography", "value": "North West", "rows": 2935, "dataset": "funded"}, {"field": "collateral_geography", "value": "Scotland", "rows": 1987, "dataset": "funded"}] |

**5. Deterministic capabilities invoked** (RECORDED)

| | |
|---|---|
| route | `analytical_composition` |
| plan intent | `population_movement_comparison` |
| capabilities | `period_movement`, `period_movement` |
| required finding kinds | movement, comparison |
| composition version | 1.0.0 · plan origin deterministic |

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

| # | kind | measure | population (rows) | delivered | **independent truth** | agrees |
|---|---|---|---|---|---|---|
| 1 | movement | `current_outstanding_balance` | Region North West (2935) | £416.7m | **£416.7m** | ✅ |
| 1p | prior | `current_outstanding_balance` @ 2026-04-30 | Region North West (2815) | £403.0m | **£403.0m** | ✅ |
| 2 | movement | `current_outstanding_balance` | Region Scotland (1987) | £291.7m | **£291.7m** | ✅ |
| 2p | prior | `current_outstanding_balance` @ 2026-04-30 | Region Scotland (1897) | £279.7m | **£279.7m** | ✅ |
| 3 | comparison | `current_outstanding_balance` | Region North West (2935) | £416.7m | **£416.7m** | ✅ |
| 3c | comparand | `current_outstanding_balance` | Region Scotland (1987) | £291.7m | **£291.7m** | ✅ |

**8. Final answer** (RECORDED, verbatim)

> Across 2026-04-30 → 2026-06-30, Region North West, 2,935 loans: Current Outstanding Balance £403.0m → £416.7m (+£13.8m). Across 2026-04-30 → 2026-06-30, Region Scotland, 1,987 loans: Current Outstanding Balance £279.7m → £291.7m (+£12.1m). Region North West against Region Scotland (2,935 vs 1,987 loans): Current Outstanding Balance £416.7m vs £291.7m (+£125.0m).

Calculated: Composed governed capabilities.

| | |
|---|---|
| ok | `True` |
| controlled refusal | `False` |
| semantic guard | `ok` [('geographic_scope', 'applied'), ('geographic_scope', 'applied')] |

---

### Q9.3 — kestrelmoor / forced_llm
*Chosen because: a composed FORECAST on the second book, under a forced model parse.*

**1. Question** (RECORDED)

> If the current pipeline converts as expected, what will our funded balance be?

**2. Parser provenance** (RECORDED)

| | |
|---|---|
| parser used | `deterministic_fallback_after_llm_failure` |
| mode detail | `deterministic_fallback` |
| LLM failure | `parse_failure` |
| repeats in this arm | 3, all reaching `analytical_composition` |

**3. Recognised family / operation** (DERIVED — `intent.classify`, a pure function of the question)

| | |
|---|---|
| families | PIPELINE, FORECAST_PROJECTION |
| operations | CONVERSION, EXPECTED_COMPLETION, PROJECT_VALUE |
| concept signals matched | forecast |
| lending windows named | — |
| lending role | — (none named) |
| structural requirements | pipeline_dataset, forecast |
| materially analytical | **True** |

**4. Resolved population and period** (RECORDED — the governed population ledger, i.e. what execution reported)

| | |
|---|---|
| predicates applied | — |
| predicates unavailable | none |
| rows before → after | None → None |
| periods compared | — |
| narrowed to | — |

**5. Deterministic capabilities invoked** (RECORDED)

| | |
|---|---|
| route | `analytical_composition` |
| plan intent | `funded_balance_outlook` |
| capabilities | `funded_balance_forecast`, `pipeline_completion_forecast` |
| required finding kinds | forecast |
| composition version | 1.0.0 · plan origin deterministic |

**6. Structured findings and 7. independently calculated truth**
  (findings RECORDED; truth computed HERE from the fixture CSVs with pandas, referencing nothing the agent produced)

| # | kind | measure | population (rows) | delivered | **independent truth** | agrees |
|---|---|---|---|---|---|---|
| 1 | measure | `current_outstanding_balance` | the whole funded book (12255) | £1.77bn | **£1.77bn** | ✅ |

*(plus 7 composition / cohort / forecast / timing findings on this run, reconciled in §12.4 but not tabulated here)*

**8. Final answer** (RECORDED, verbatim)

> Current funded balance is £1.77bn as at 2026-06-30. Gross pipeline in the governed extract is £76.7m as at 2026-06-29. Expected completions from the pipeline: £12.5m. Forecast funded balance: £1.78bn. Expected completion amount from the open pipeline: £7.7m. Expected to land: 2026-07 £3.9m; 2026-08 £1.9m; 2026-09 £1.8m.

Calculated: Composed governed capabilities.

| | |
|---|---|
| ok | `True` |
| controlled refusal | `False` |
| semantic guard | `ok` [('projection', 'applied')] |



### 13.3 The raw evidence, and how to recompute it

Every number in this report is recomputable by someone who does not trust the
narration. The measurement artefacts are committed at
`due_diligence/evidence/analytical_intent_v1/` — the four run files hold all 752
responses verbatim as the production endpoint returned them, including the ones
that refused.

| File (as committed) | Bytes | SHA-256 of the **uncompressed** JSON |
|---|---|---|
| `v1_nl_alderbridge_production.json.gz` | 94,841 (from 2,478,770) | `50f4b38e85f121b98a9f6e6f4cfeba84053c1b932527199c0c48204956905514` |
| `v1_nl_alderbridge_forced_llm.json.gz` | 95,083 (from 2,482,727) | `e08fdeb250226b644be1db7423fd60bef6e6a89d56f54a29ac4f1547abaea286` |
| `v1_nl_kestrelmoor_production.json.gz` | 96,416 (from 2,495,500) | `2d2511babd0139274eece83acc25e283c79dbbda68d6348851bb147b55038e3b` |
| `v1_nl_kestrelmoor_forced_llm.json.gz` | 96,287 (from 2,498,863) | `eb89bdf72b48c8544b9ab79ba6fe373d8e22880df31f1e4603ab8e423877b3a5` |
| `nl_bank.py` | 6,877 | `f37729113df3a6734d661ac63f862f9963b592d13bbffe9e7c38566020bd8874` |
| `nl_harness.py` | 6,285 | `de059ef3fd07357092d8109747d0c46405f4329413cd17e4e9a01ec9477e8a9c` |
| `nl_score.py` | 10,490 | `aade14173ab3623196434e833b90a1d112d4106c526381cade6c0cbc4c7164f7` |
| `nl_reconcile.py` | 5,455 | `4802c50be93ff55ec923ab473250e6e30c4a5233456362dae063be769f4b3d39` |
| `v1_final.py` | 5,466 | `786ca261bf33702ceac57f0985e7621ba2643a3fd8b65f3ea7f68a774cd48ea5` |
| `v1_score_run.py` | 3,397 | `3c625636ef0da0ea48e58f2284c85fa08c417a854808f1e619cbcd4c7a4f2b86` |
| `v1_trace.py` | 10,653 | `3306224571fec5b9a4b9c922eb8155a28af8a5bb23bb19476ae7c3b46577dfb8` |

```bash
cd due_diligence/evidence/analytical_intent_v1 && gunzip -k v1_nl_*.json.gz

python v1_final.py       # the distribution, the baseline comparison, the gate
python nl_reconcile.py v1_nl_*.json   # recompute every figure from the CSVs
python v1_trace.py       # regenerate the ten traces in §13.2
python nl_harness.py alderbridge production out.json   # reproduce from scratch
```

**The control that makes the comparison honest:** `nl_score.py` is the scorer
used for *both* the frozen baseline and this run, unmodified. `v1_final.py`
recomputes the baseline column of every table in §12 from the baseline's own run
files, and reproduces the previously published figures exactly — 405 correct,
147 silent semantic errors, 40 incorrect-successful, 187 unsafe. A scorer bent
to flatter this change would no longer reproduce them.

**What the run files contain that a summary cannot:** the parser provenance per
run (`llm`, `deterministic`, or `deterministic_fallback_after_llm_failure`), the
governed population ledger with row counts before and after, every structured
finding with its period and population, the semantic-guard verdict and facets,
and the answer text. §13.2 traces ten of them; the other 742 are in the files.
