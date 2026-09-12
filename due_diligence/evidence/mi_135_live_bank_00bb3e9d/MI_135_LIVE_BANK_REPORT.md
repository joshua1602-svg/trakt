# MI 135 live acceptance bank — frozen product `00bb3e9d`

**STATUS: INCOMPLETE.** 104 of 135 questions carry semantic evidence. The
remaining 31 could not be interpreted: the Anthropic API key the product uses
reached its usage limit mid-bank —

```
MODEL_UNAVAILABLE  BadRequestError 400
"You have reached your specified API usage limits.
 You will regain access on 2026-10-01 at 00:00 UTC."
```

The product behaved correctly under that failure: it refused rather than
answering from a partial reading — *"I could not complete the
language-understanding step for this question, so I have not answered it."*

The 31 are **not** randomly distributed, which is why no bank-level rate is
claimed here: they are `analytical_intent` (25) and `borrowing_base` (6) —
two whole families.

## Provenance

```
PRODUCT_SHA_UNDER_TEST = 00bb3e9d8175a395b6643772379866d6bc6169eb
DEPLOYED_SHA_CONFIRMED = 00bb3e9d8175a395b6643772379866d6bc6169eb  (2 stable reads)
PRODUCT_FILES_CHANGED  = 0
BANK_SOURCE  = mi_agent/interpretation_v2/banks/interpretation_bank_135.yaml
BANK_SHA256  = 1f71e8444dc4bc149bc8028527c5fe077ce49690a95d27d2a2273a20813fdab2
QUESTIONS    = 135   (45 canonical x 3 variants)
COMPLETED    = 104
BLOCKED      = 31  (API usage limit, resets 2026-10-01)
canary principal matched = true; serving mode = canary
```

## Output 1 — scorecard (of the 104 completed)

```
FULLY_CORRECT                                         52
PARTIALLY_CORRECT                                     35
WRONG                                                  0
APPROPRIATE_CLARIFICATION                              0
UNNECESSARY_CLARIFICATION                              0
PLAN_OR_CONNECTIVITY_GAP_MASQUERADING_AS_CLARIFY       0
HONEST_REFUSAL                                        15
BAD_REFUSAL                                            0
INCONCLUSIVE                                           2

NEW_SERVED                                            22
LEGACY_FALLBACK                                       82
NEW_PATH_SERVE_RATE                                   21.2%

PLAN                                                 102
CLARIFY                                                2
REFUSE                                                 0
INTERPRETER_FAILURE                                    0
```

## Output 2 — safety / semantic scorecard

```
SILENT_MEASURE_DROPS        0      PIPELINE_TO_FUNDED_MISROUTES  0
SILENT_DIMENSION_DROPS      0      FUNDED_TO_PIPELINE_MISROUTES  0
SILENT_FILTER_DROPS         0      EXECUTION_ERRORS              0
SILENT_PERIOD_DROPS         0      EXCEPTION_ESCAPES             0
SILENT_CAPABILITY_DROPS     0
SILENT_POPULATION_DROPS     0
SILENT_SOURCE_SCOPE_DROPS   0
SILENT_COMPARISON_DROPS     0
SILENT_WIDENINGS            0
```

**Zero silent semantic errors and zero misroutes across 104 live questions.**
Every failure in this bank is a visible one: a reading that differs from the
fixture, or a refusal that states its reason.

## Output 3 — question families

| Family | n | Fully correct | Other |
|---|---:|---:|---|
| pipeline_stage_movement | 27 | 26 (96%) | partially correct 1 |
| multi_dimensional_artefact | 21 | 2 (10%) | partially correct 18, honest refusal 1 |
| multi_filter_facts | 15 | 8 (53%) | partially correct 4, honest refusal 3 |
| synthesis | 15 | 8 (53%) | honest refusal 6, partially correct 1 |
| movement | 15 | 2 (13%) | partially correct 10, honest refusal 3 |
| strategic | 9 | 6 (67%) | honest refusal 2, partially correct 1 |
| analytical_intent | 2 | 0 (0%) | inconclusive 2 |

## Output 4 — where the partial answers lose the question

Dimension that diverges from the fixture on the 35 PARTIALLY_CORRECT:

```
statistic             20
operation              8
capability             7
temporal               3
weight                 2
output_structure       1
```

## Output 5 — why the governed path did not serve

```
 31  INELIGIBLE:POPULATION_NOT_EXECUTABLE
 29  INELIGIBLE:CAPABILITY_NOT_GENERIC
  9  INELIGIBLE:GEOGRAPHY_REQUESTED
  3  PLAN_RECEIPT_RECONCILIATION_FAILED:the measured population is empty
  2  INELIGIBLE:MEASURE_NOT_SUPPORTED
  2  INELIGIBLE:TOO_MANY_DIMENSIONS
  2  INELIGIBLE:OPERATION_NOT_TEMPORAL
  2  CLARIFY_NOT_SERVED_IN_THIS_SLICE
  1  INELIGIBLE:DIMENSION_NOT_SUPPORTED
  1  INELIGIBLE:NOT_SINGLE_OUTPUT
```

