# Tranche D — four reconciliations for the boundary report

## 1. The permissive `any` declarations, all of them

Adding `expected_answer_type` closed a hole where 31 count questions had nothing
to check against. A value of `any` can never fail, so leaving it anywhere it is
not forced reopens part of that hole by declaration. Where it stood, and where
it stands now:

| | first cut | after pinning |
|---|---|---|
| `any` | 21 | **7** |
| pinned to a real type | — | 14 |

**All 21 were `parse_only` cases** — twelve pipeline, nine forecast — validated
at parse level because the synthetic 400-row tape carries no pipeline data.
None was an executed answer case, so none was silently unchecked at run time.
Their answer type is derivable from the wording regardless, and declaring it
costs nothing now and becomes enforceable the moment Tranche E gives them
executable data.

Fourteen were pinned by extending the classifier with the governed pipeline and
forecast measure nouns — `expected funded`, `weighted expected`, `pipeline
amount`, `conversion`, `applications`, `cases` — which live in the pipeline
contract rather than the funded field registry and so never reached a type from
a declared format.

**The seven that keep `any`, each with its reason.** Every one is genuinely
ambiguous between an amount and a case count, and the ambiguity is in the
question, not in the classifier:

| case | question | why `any` |
|---|---|---|
| `pipe_185` | pipeline by broker | "pipeline" alone is either the gross amount or the case count |
| `pipe_186` | pipeline by stage | same |
| `pipe_191` | pipeline by stage for broker Alpha | same |
| `pipe_187` | completions by month | completions counted, or the amount completing |
| `fcast_197` | show projected completions | same ambiguity, forward |
| `fcast_198` | forecast by completion month | names no measure at all |
| `fcast_202` | expected completions over the next three months | count or amount |

Pinning any of these would declare an expectation the question does not carry.
The sibling cases that DO disambiguate — `pipeline amount by stage`,
`how many cases are in the pipeline` — are pinned, so the distinction is
visible in the bank rather than argued here.

## 2. The coverage counts, as a partition

The earlier figures did not tie because the classes overlapped: the `any` cases
were counted separately AND again inside `parse_only`. As a partition, where
every case falls in exactly one class:

| class | cases |
|---|---|
| typed and executed — the sweep's reach | 207 |
| `parse_only` — validated at parse level, never executed | 21 |
| refuse / clarify — answer type is `none` | 24 |
| **total** | **252** |

Which ties to the bank, and to the collection count established earlier:
252 parametrised cases + 6 module-level tests = 258 collected. Two further
module-level tests were added with the new field, so the bank now collects 260.

## 3. The 44-bank answer-type counts

"168 of 176 match, 0 diverge" invited a subtraction, and the subtraction was
wrong because the two numbers came from different run sets. Within each set the
counts tie exactly:

| run set | declared-typed runs | matched | refused | no typed finding |
|---|---|---|---|---|
| `llm_post` | 224 | **168** | 20 | 36 |
| `det_post` | 224 | **176** | 12 | 36 |

The third state is **no typed finding**: 36 runs in both sets, where the route
answers without emitting a typed finding — the specialist-route gap of §4
below. The eight-run difference between the sets is exactly the eight extra
refusals on the LLM path (20 against 12), which is the parser instability D4
addresses, not a difference in typing.

## 4. The subject-side rule, as a standing rule

The same defect has now appeared **four** times, in four independent places:

| where | what it did |
|---|---|
| `llm_query_parser._detect_metric` | "balance where LTV above 50%" resolved to weighted-average LTV |
| `llm_query_parser.wants_balance_too` | "how many loans have a balance above £250k" answered with a balance |
| `answer_type.asked`, first version | typed "balance by region where borrower age is over 70" as an AGE question |
| `answer_type.asked`, second version | typed "balance by LTV bucket" as a RATE question |

Four occurrences is a design rule, not four bugs:

> **Anywhere wording is scanned to fill a slot, it reads the SUBJECT SIDE, never
> the whole string.** A measure named inside a condition is the field being
> filtered on. A measure named inside a grouping clause is the axis, not the
> answer. Both belong to clauses that have already been consumed by another
> slot, and a scanner that sees them is competing for a word that is spoken for.

Two helpers now implement it — `llm_query_parser._metric_slot` and
`answer_type.subject_side` — and they agree on the conservative rule for
conditions: an opener counts only when a numeric bound follows it, so "loans
with LTV above 50%" is cut while "regions with the highest LTV" is not.

**Where the pattern may still exist, unaudited.** Every one of these was found
by a defect rather than by a search. The shape to look for is a regex or
vocabulary scan over the whole question that decides a single slot:

* `_explicit_dimensions` — scans the whole question for dimension terms. Partly
  guarded (`mask_scope_phrases`, and the aggregator-before-bucket mask added in
  D2), not audited end to end.
* `_parse_filters` and the categorical-value scan — anchored at end-of-string,
  which is a different guard, but the same shape.
* `_detect_periods`, `_relative_mode`, `_forecast_question_kind`,
  `_risk_limit_category` — each scans the whole question for one decision.
* `detect_measure_set` / `unresolved_measure_slots` — multi-measure detection,
  which is where "high age borrower exposure" picked up the age measure.

None of these is asserted to be defective. They are the population a systematic
audit should cover, and that audit is named in the backlog rather than done
here.
