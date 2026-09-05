# Composition sprint — architecture map and root inventory

Recorded before any production change, per §2 and §16.

```
HEAD    4aed376a86ae174edfb3f9b13133323d1f77f1a5
branch  claude/mi-query-agent-defects-27s4ys   (clean, in sync with origin)
```

Everything below is traced from the running code, not recalled.

---

## 1. The path, as it exists

```
question
  │
  ├─ mi_agent_api/mi_service.py            request → client/frame/lens resolution
  ├─ mi_agent/llm_query_parser.py          _deterministic_parse_unchecked
  │     ├─ ERE analytical-intent recognisers        (l.3911)
  │     ├─ _cohort_progression_recognizer           (l.3932)
  │     ├─ _risk_limit_recognizer                   (l.3940)
  │     ├─ _measure_set_recognizer                  (l.3946)  ← multi-measure
  │     ├─ filtered count / balance branch          (l.3952)  ← single-output
  │     ├─ "show loans where" drill-through         (l.4030)
  │     ├─ line / trend branch                      (l.4282)
  │     ├─ grouped-bar branch                       (l.4728)
  │     └─ terminal no-metric region (summary / ambiguous / amount / unmapped)
  │  → MIQuerySpec
  ├─ mi_agent/mi_query_executor.py         P1E measure-set dispatch (l.1646)
  ├─ mi_agent/execution_receipt.py         facets, receipt, guard
  ├─ question_interpretation/completeness.py   concept-carried check
  └─ governance envelope
```

**`MIQuerySpec.measures` and the P1E executor path already work.** A verified
live example produces, and executes, three measures over one population:

```
loan_count                  → count
current_outstanding_balance → sum
current_loan_to_value       → weighted_avg
```

Grouped measure sets work too (`_execute_grouped_measure_set`). §7's instruction
not to build a second analytics engine is correct: the engine is there.

---

## 2. Current owners, by concept

| Concept | Owners today | Count |
|---|---|---|
| **"a count was requested"** | `_COUNT_INTENT_RE` + `_wants_count()`; `_COUNT_MEASURE_RE`; `_SHARE_COUNT_RE` + `_counts_a_row_noun()`; `is_count_q` (inline regex, l.3956); `wants_balance_too` in the filtered branch | **5** |
| requested output set | `detect_measure_set` / `_measure_hits` (multi only) | 1, partial |
| aggregation / statistic | `_detect_metric`, `_apply_agg_intent`, `_local_aggregation_intent`, per-branch literals (`"count"`, `"sum"`, `"weighted_avg"`) | 4+ |
| measure resolution | `_detect_metric` (single) and `_measure_hits` (set) — two entry points, one vocabulary | 2 |
| filters | `_parse_filters` → `_FILTER_COMPARATORS` (prefix) + `_POSTFIX_COMPARATORS`; `_grouped_value_filters`; `_parse_categorical_filter` | 3 paths, 1 vocabulary since Phase 1 |
| predicate-subject exclusion | `question_interpretation.lexical.is_filter_subject` | 1 (consolidated in Phase 1) |
| grouping dimensions | `_explicit_dimensions`, `_grouping_regions`, `_DIMENSION_SUFFIX_RE` | 3 |
| semantic coverage | `question_interpretation/completeness.py` | 1 |
| measure completeness | **none machine-readable** — `measures_unavailable` is disclosed in prose by `_measure_set_answer` | 0 |
| clause-local scope | **none** | 0 |
| conversational scope | **none** | 0 |

---

## 3. Root causes, with evidence

### R1 — Five owners of "count", three of them adjacency-bound

One adjective between "how many" and the row noun splits them:

```
question                                       wants_count  COUNT_MEASURE  is_count_q  row_noun
how many loans are there                          True          True         True        True
how many funded loans are there                  False         False         True        True
how many pipeline cases are there                False         False         True        True
how many joint loans are there                   False         False         True        True
how many funded loans are to joint borrowers     False         False         True        True
```

`_COUNT_INTENT_RE` and `_COUNT_MEASURE_RE` both spell it
`how\s+many\s+(?:loans|cases|accounts…)` — adjacent only. `is_count_q` is
`\bhow many\b` — modifier-tolerant. So the two vocabularies that feed the
**measure set** are blind to exactly the phrasings the bank uses.

Consequence, traced end to end:

```
"How many funded loans are to joint borrowers, and what is their funded balance?"
    detect_measure_set → []                    (count not found → only 1 measure)
    → falls past _measure_set_recognizer
    → filtered-count branch → aggregation=count, filters={borrower_type: Joint}
    → the BALANCE is never represented anywhere

"What is the funded balance for joint borrowers, and how many loans are there?"
    detect_measure_set → [balance/sum, loan_count/count]   ← adjacency holds
    → multi_measure, both outputs execute
```

The two sentences ask the same thing. This is failure family A, whole.

**Leverage:** `_measure_set_recognizer` runs at l.3946, *before* the filtered
branch at l.3952. Making count discovery modifier-tolerant routes family A into
the multi-measure execution that already works — no new engine, per §7.

### R2 — A span already governed as filter / bucket / dimension is re-discovered as the measure

```
"How many loans are in the 60-70% LTV bucket?"
    → metric = current_loan_to_value, aggregation = weighted_avg,
      dimension = ltv_bucket, filters = {}
```

`_wants_count` is **True** here. It is never consulted, because the grouped-bar
branch only asks `if metric is None`, and `_detect_metric` had already claimed
`LTV` from the bucket phrase. The count request loses to a measure word that was
identifying a population.

`is_filter_subject` (consolidated in Phase 1) already owns this exclusion for
*numeric predicates*. It does not cover bucket / dimension / entity-qualifier
positions. §5 asks for the general mechanism; this is the gap it must close.

### R3 — Terminal branches build a fresh spec and drop resolved filters

The `dimension is None and metric is None` region returns from four places
(`share`, `wants_summary`, `_ambiguous`, `amount`), each constructing a new
`MIQuerySpec`. Only the `share` branch computes filters.

```
"total pipeline amount for cases with an interest rate above 6%"   filters={}
"total pipeline balance for cases with an interest rate above 6%"  filters={rate>6}
```

`_parse_filters` resolves the bound correctly in **both**; the word `amount`
routes to a branch that discards it, and the facet guard then refuses honestly.
This is §14C: the comparator is parsed and then not carried, so parsing and
execution disagree about the same object.

### R4 — No clause-local scope

```
"How many joint loans are there, what is their balance,
 and how much of that balance has LTV above 40%?"

  → note=filtered_count_and_balance
    filters={borrower_type: Joint, current_loan_to_value: {gt: 40}}
```

Both predicates are global. The third clause's narrowing silently mutates the
population of the first two. There is no representation in which it could be
otherwise: `MIQuerySpec` has one `filters` dict for the whole request.

### R5 — Completeness cannot be enforced because requests are not recorded

`measures_requested` is written **only inside the P1E branch**, i.e. only when
the set was already discovered. A request whose outputs were never discovered
records nothing, so there is nothing for a completeness invariant to compare.
§9 therefore depends on §4: the requested-output record must exist first.

### R6 — There is no conversational state at all

`mi_agent_api/app.py::QueryRequest` carries `question`, `portfolio`,
`portfolioId`, `asOfDate`, `filters`, `datasetContext`, `context`,
`sourcePortfolioLens`. **No conversation or session identifier, no prior scope.**
Every turn is independent by construction.

§10–13 therefore need a transport change as well as a scope model. That is a
decision, not an implementation detail, and it is raised rather than assumed.

---

## 4. What the roots imply for the sprint's shape

| § | Requirement | Depends on |
|---|---|---|
| 4 | one owner for requested outputs | R1 |
| 5 | role ownership before measure detection | R2 |
| 14C | one comparator contract end to end | R3 |
| 3 / 8 | clause-local scope | R4 |
| 9 | completeness | R5, and therefore R1 |
| 10–13 | conversational scope | R6 + a transport decision |

The order is forced: **R1 → R5 → R4**, because completeness needs a requested
record and clause-local scope needs outputs to attach narrowings to. R2 and R3
are independent and can land on their own invariants.

## 5. Debt this sprint should remove, not add to

- five count owners → one
- two measure-resolution entry points (`_detect_metric`, `_measure_hits`) → one
  vocabulary with two callers, or one owner
- per-branch aggregation literals → the requested-output record carries it
- `is_count_q` (an inline regex in the middle of the parse) → deleted

If the sprint ends with a `QueryPlan` layered *on top of* five count owners, it
has failed §5 of its own brief.
