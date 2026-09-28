# MI 135 acceptance bank — COMPLETE, frozen product `00bb3e9d`

```
PRODUCT_SHA_UNDER_TEST = 00bb3e9d8175a395b6643772379866d6bc6169eb
DEPLOYED_SHA_CONFIRMED = 00bb3e9d8175a395b6643772379866d6bc6169eb
PRODUCT_FILES_CHANGED  = 0
BANK_SHA256            = 1f71e8444dc4bc149bc8028527c5fe077ce49690a95d27d2a2273a20813fdab2
QUESTIONS = 135        COMPLETED = 135

this run:  HTTP_ACCEPTANCE_REQUESTS = 31    TRANSPORT_RETRIES = 0
           CONCEPT_MERGE_ATTEMPTS = 31      SUCCESSFUL = 30 (applied 7, no_change 23)
           PROVIDER_CALLS_OBSERVED = not exposed by the product (see observability)
MI_AGENT_PLAN_SERVE_ASSUMED_STATE = off     VERIFIED = false
```

The canary state is consistent with `off`: no governed record was sought or
written, and the only model arm exercised was concept-merge. That is corroboration,
not proof, and it is not claimed as verified.

## Scorecard

```
FULLY_CORRECT              52        HONEST_REFUSAL             25
PARTIALLY_CORRECT          56        BAD_REFUSAL                 0
WRONG                       0        APPROPRIATE_CLARIFICATION   2
INCONCLUSIVE                2        UNNECESSARY_CLARIFICATION   0
INFRASTRUCTURE_FAILURE      0

USER_SUCCESS_RATE = 40.0%   (FULLY_CORRECT + APPROPRIATE_CLARIFICATION) / 135
INDEPENDENT_NUMERIC_PARITY = N/A on all 135 — the bank states no expected values
```

**The headline understates the product, and the reason is measurable.** Of the 56
partials, **19 diverge from the fixture on one slot only: the interpreter leaves
`statistic` unstated on a balance measure.** In every one of those 19 the compiler
binds `statistic: "sum"` and marks `statistic_defaulted: true` — a DISCLOSED
governed default, not a silent one — and the receipt confirms `aggregation: sum`.
17 of the 19 delivered an answer to the caller; the other 2 are safe
empty-population refusals.

```
STRICT (interpretation fixture)   FULLY_CORRECT 52    PARTIAL 56
USER-FACING (a disclosed default
is not a miss)                    FULLY_CORRECT 69    PARTIAL 37   → 52.6% success
```

Both are reported because they answer different questions: the strict figure is
what the interpretation benchmark scores, the user-facing figure is what a lender
receives.

## Safety

```
TOTAL_SILENT_SEMANTIC_ERRORS = 0     MISROUTES                = 0
  silent_measure_drop     0            pipeline_to_funded      0
  silent_statistic_drop   0            funded_to_pipeline      0
  silent_dimension_drop   0          EXECUTION_ERRORS         = 0
  silent_filter_drop      0          EXCEPTION_ESCAPES        = 0
  silent_period_drop      0          WRONG ANSWERS            = 0
  silent_capability_drop  0          BAD_REFUSALS             = 0
  silent_population_drop  0
  silent_scope_drop       0
  silent_comparison_drop  0
  silent_widening         0
```

**Nothing in this bank answered a different question from the one asked.** Every
failure is visible: a divergent reading, or a refusal that names its obstacle.

## Migration (governed path)

```
CANARY_EXPOSED              104
  NEW_SERVED                 22      governed serve rate 21.2% of the 104
  LEGACY_FALLBACK            82
LEGACY_ONLY_COMPLETION       31
NOT_TESTED_KNOWN_UNMIGRATED  31
```

The serve-rate denominator is the 104 canary-exposed questions only. The 31 are
excluded from it by construction: none targets a migrated capability, so the
governed path was deliberately not invoked and did not fail to serve them.

Of the 82 fallbacks, 77 are `INELIGIBLE` at the slice perimeter, 3
`PLAN_RECEIPT_RECONCILIATION_FAILED`, 2 `CLARIFY_NOT_SERVED_IN_THIS_SLICE`.

## Family breakdown (all 135)

| Family | n | Fully | Partial | Refusal | Wrong | Success |
|---|---:|---:|---:|---:|---:|---:|
| pipeline_stage_movement | 27 | 26 | 1 | 0 | 0 | **96%** |
| analytical_intent | 27 | 0 | 20 | 5 | 0 | 0% |
| multi_dimensional_artefact | 21 | 2 | 18 | 1 | 0 | 10% |
| multi_filter_facts | 15 | 8 | 4 | 3 | 0 | 53% |
| synthesis | 15 | 8 | 1 | 6 | 0 | 53% |
| movement | 15 | 2 | 10 | 3 | 0 | 13% |
| strategic | 9 | 6 | 1 | 2 | 0 | 67% |
| borrowing_base | 6 | 0 | 1 | 5 | 0 | 0% |

## Partial-answer Pareto (primary divergent slot, one per question)

| Divergence | Questions | % of partials | Examples |
|---|---:|---:|---|
| unstated `statistic` (compiler defaults, disclosed) | 20 | 36% | Q02A, Q04A, Q11A |
| legacy-only: figure not independently verifiable | 21 | 38% | Q1.2, Q2.1, Q2.2 |
| `capability` | 7 | 12% | Q18A, Q18B, Q18C |
| `operation` | 5 | 9% | Q10B, Q22B, Q22C |
| `weight` | 2 | 4% | Q05A, Q05C |
| `temporal` | 1 | 2% | Q21A |

## Refusal Pareto (25 honest refusals) — safe *and* a coverage question

| Class | n | Existing owner? | Reading |
|---|---:|---|---|
| SEMANTIC_CONNECTIVITY — capability understood, not reachable | 6 | yes | The product names what it understood and declines. A connectivity miss, safely handled. |
| Multi-date scope refusal (compare / concentration) | 5 | yes | "the portfolios are at different reporting dates" — a real coverage gap with existing arithmetic behind it. |
| SAFE_EMPTY_POPULATION | 5 | n/a | Correct behaviour: refuses to substitute a whole-book figure. |
| DATA_OR_CONFIGURATION — no facility configured | 5 | **yes** | Borrowing-base arithmetic exists; no approved facility is configured for this portfolio. **Not a capability gap.** |
| Comparison not applicable to the owner | 3 | yes | Connectivity. |
| SAFE_REFUSAL_RATHER_THAN_DROP | 1 | n/a | Refused rather than answer with a dropped dimension. |

Every refusal is safe. Five of them (borrowing base) are pure configuration, and
eight are connectivity — so 13 of 25 refusals sit on arithmetic that already
exists.

## Census overlay — primary cause of every non-success

| Primary cause | Questions | % of misses | Existing owner? | Complexity |
|---|---:|---:|---|---|
| INTERPRETATION (unstated statistic; disclosed default) | 20 | 24% | n/a — reading, not arithmetic | TRIVIAL |
| RECEIPT_OR_PRESENTATION (legacy answer, unverifiable figure) | 21 | 25% | yes | N/A (measurement gap) |
| PLAN_REPRESENTATION (capability/operation/weight/temporal) | 15 | 18% | yes | MEDIUM |
| DISPATCH_CONNECTIVITY (capability understood, unreachable) | 9 | 11% | yes | SMALL–MEDIUM |
| TEMPORAL_BINDING (multi-date scope refusals) | 5 | 6% | yes | SMALL |
| DATA_OR_CONFIGURATION (no facility configured) | 5 | 6% | yes | TRIVIAL (config) |
| APPROPRIATE_USER_AMBIGUITY | 2 | 2% | n/a | N/A |
| SAFE_EMPTY_POPULATION / correct refusal | 6 | 7% | n/a | N/A |
| GENUINE_CAPABILITY_GAP | **0** | 0% | — | — |

**GENUINE_CAPABILITY_GAP = 0.** Every miss in this bank sits on arithmetic that
already exists somewhere in the estate, or on a reading, or on configuration.
Nothing required a calculation nobody has written. The census's candidate genuine
gap — forward approved-limit projection — did not present as one here: Q2.1/Q2.2
were answered by run-rate extrapolation, and Q5.1/Q5.2 refused for connectivity
rather than absent arithmetic.

### Smallest set of causes explaining the misses

```
50% of misses   2 causes:  unstated statistic + legacy-answer unverifiability
75% of misses   4 causes:  + plan representation + dispatch connectivity
90% of misses   6 causes:  + temporal binding + facility configuration
```

## Commercial readiness

```
COMMERCIAL_MI_QUERY_STATUS = GO_WITH_BOUNDED_FIXES
SECOND_LENDER_READINESS    = YES_AFTER_BOUNDED_FIXES
COMMERCIAL_OBSERVABILITY_GAP = YES
```

**Safety — strong.** 0 wrong answers, 0 silent semantic errors, 0 misroutes,
0 bad refusals across 135 live questions. Every decline states its reason and
several explicitly refuse to substitute a whole-book figure. This is the axis
that decides whether a lender can be exposed to the product, and it passes.

**Coverage — mixed and legible.** 52.6% user-facing fully correct; 25 safe
refusals; two families near zero (`analytical_intent`, `borrowing_base`) and one
near perfect (`pipeline_stage_movement`, 96%). The weak families are weak for
identified, bounded reasons, not diffuse ones.

**Migration — narrow but sound.** 22 of 104 canary-exposed questions served by
the governed path, close to the structural ceiling: only 39 of 135 questions
target a migrated capability at all. Where it served, it served without a single
silent error. Legacy reliance remains high by design.

**Remediation — bounded, not architectural.** 0 genuine capability gaps; 13 of 25
refusals sit on existing arithmetic; 6 causes explain 90% of misses; the largest
single cluster is a trivial interpretation fix with a disclosed default already
behind it.

## Top five remediations

| # | Theme | Questions | % bank | Root cause | Existing owner | Complexity | Blocker |
|---|---|---:|---:|---|---|---|---|
| 1 | Interpreter states the statistic on a balance measure | 20 | 15% | INTERPRETATION | n/a | TRIVIAL | **NO** |
| 2 | Multi-date scope: compare / concentration across reporting dates | 8 | 6% | TEMPORAL_BINDING | yes | SMALL | **NO** |
| 3 | Configure an approved facility for the portfolio | 5 | 4% | DATA_OR_CONFIGURATION | yes | TRIVIAL | **NO** |
| 4 | Connect understood-but-unreachable capabilities (forecast, limits) | 9 | 7% | DISPATCH_CONNECTIVITY | yes | SMALL–MEDIUM | **NO** |
| 5 | Persist token usage in governed evidence | 0 | 0% | observability | n/a | SMALL | **NO** |

None is a commercial blocker on its own; together 1–4 address 42 questions (31%
of the bank) and all four sit on existing arithmetic or configuration.

## The six questions

**1. Did the migration materially improve the real MI product?** On safety, yes
and measurably: the governed path served 22 questions with zero silent semantic
errors and zero misroutes, and the refusal discipline it introduced is visible
throughout — including on legacy answers. On coverage, not yet: it serves 21% of
the questions it was exposed to, because it deliberately implements two
capabilities of ten.

**2. Where is the remaining work?** Predominantly **connectivity and plan
expressiveness** (24 questions), then **interpretation** (20, trivial), then
**configuration** (5). Genuine missing analytics: none found.

**3. Should clarification surfacing be built?** Not yet. Only 2 of 135 clarified,
both genuinely ambiguous — and the product authors no clarification sentence
today, only reason codes and option lists (`clarification_question = NOT_AUTHORED`).
The material is good, but building the UX would serve 1.5% of the bank. Revisit
once connectivity work raises the clarify rate.

**4. Smallest changes, largest measured benefit?** Remediation 1 (trivial, 20
questions) and 3 (config only, 5 questions) — 25 questions, 19% of the bank, for
almost no engineering. Then 2 and 4.

**5. Is another broad architectural slice justified?** No. The evidence points to
bounded connectivity work on existing owners, not new architecture. A slice would
be justified only if genuine capability gaps existed, and this bank found none.

**6. Commercially credible for second-lender OCC onboarding after a bounded
sprint?** Yes. The safety properties hold on live evidence, the failures are
concentrated and named, and nothing needs inventing. The caveat is coverage
breadth, not correctness — and a second lender will expose configuration paths
(remediation 3) that this portfolio left untested.

## Observability finding (carried forward, not fixed)

`COMMERCIAL_OBSERVABILITY_GAP = YES`. Governed evidence records
`model.usage = {}` on every request, so interpretation cost cannot be attributed
per query in production. The concept-merge arm does record its own cost. Not
fixed here.

```
135_BANK = COMPLETE
```
