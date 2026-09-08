# Atomic perimeter — Phase 0 root-cause characterisation

100-question live perimeter run, ERE/2026-06-30. 68 VERIFIED, 6 SUSPECT,
26 NOT_ANSWERED. This file characterises all 32 non-verified rows. **No
production code was changed to produce it.**

Every finding below was reproduced OFFLINE against
`tests/fixtures/pipeline_transition_2w` (copied, with the product and rate
columns varied so thresholds and grouping have a population) plus the funded
tape `mi_agent_api/tests/test_stage_movement_query` writes. The live wording is
reproduced verbatim in each case, so none of this rests on the live receipts.

---

## A. The four families

Thirty of the thirty-two rows are four defects. Each is the estate's recurring
shape: **one concept with two owners.**

| # | Family | Rows | Class | First wrong decision |
|---|--------|------|-------|----------------------|
| 1 | The prepared pipeline frame speaks a vocabulary the MI registry does not carry | 16 | D | registry lookup, before the parse |
| 2 | A stated numeric threshold is dropped from the spec | 4 | A | `_deterministic_parse` |
| 3 | Ranking language picks money when the reader asked for population | 2 | A | `_deterministic_parse`, ranking branch |
| 4 | "headroom"/"exposure" are claimed by the risk-limit route ahead of any NNEG owner | 8 | A / F | recogniser claim, priority 100 |
| — | Not reproduced offline (P024, P026) | 2 | unknown | needs the live receipt |

---

## Family 1 — the pipeline frame's vocabulary is not the registry's

**16 rows: P028, P031–P035, P041–P045, P046, P047, P048, P049, P050.**
(One of them, P048, is a SUSPECT — see why below.)

`prepare_pipeline_mi_dataset` produces 36 columns. Run on the fixture, these are
present and populated:

```
product_type                     10/10   'Lifetime Mortgage Lump Sum', 'Drawdown Lifetime Mortgage'
pipeline_case_age_days           10/10   35, 35, 35
expected_funded_amount           10/10   100000.0, 220000.0, 280000.0
weighted_expected_funded_amount   9/10   20000.0, 44000.0, 126000.0
completion_probability            9/10   0.2, 0.2, 0.45
expected_completion_month        10/10   '2026-09', '2026-09', '2026-08'
days_to_expected_completion      10/10   90, 90, 60
```

**None of the seven is in `mi_agent/mi_semantics_field_registry.yaml.`** The
registry is the parser's and the executor's only vocabulary: a column absent from
it can never be named, bound, grouped or filtered by an MI question, however
faithfully `pipeline_prep` computes it and however plainly
`config/mi/pipeline_field_contract.yaml` declares it.

The corroboration is exact and falsifiable. The pipeline fields that ARE
registered — `pipeline_stage`, `broker_channel`, `current_interest_rate`,
`current_loan_to_value`, `youngest_borrower_age`, `current_outstanding_balance` —
are precisely the themes that verified (stage 49/49, broker 4/5, rate buckets,
LTV, age 20/20). The seven that are not registered are precisely the themes that
scored zero. Nothing else separates them.

### Where the two vocabularies disagree

| prepared pipeline frame | MI registry | note |
|---|---|---|
| `product_type` | `erm_product_type` | contract already declares `funded_correlation: [erm_product_type, …]`; nothing reads it at bind time |
| `completion_probability` | `forecast_funding_probability` | registry entry is `virtual`, `source_criteria: ["forecast"]` — a state layer, not this frame |
| `weighted_expected_funded_amount` | `forecast_funded_balance` | same |
| `expected_completion_month` | `forecast_funding_date` | same, and a month is not a date |
| `expected_funded_amount` | — | no registry entry at all |
| `pipeline_case_age_days` | — | no registry entry at all |
| `days_to_expected_completion` | — | no registry entry at all |

`mi_agent_pptx/pipeline_prep.py:93` already reads `funded_correlation` to bridge
the two names. The MI query surface does not.

### Why one of these 16 is a SILENT wrong answer

P048 "What is the average pipeline case age in days?" answers
**"Average Borrower Age: 74 · 10 loans"**.

`pipeline_case_age_days` has no registered name, and `youngest_borrower_age`
carries the bare synonym `age`. The absence does not produce a refusal — it
produces a substitution, because a neighbouring field claims the word. P049 and
P050 ("older than 30 days") bind the threshold onto `youngest_borrower_age` and
are then caught by the facet guard, which refuses honestly. So the same root
gives a silent wrong answer on the mean and a fail-closed refusal on the
threshold, decided only by whether the guard had a facet to check.

**Filling the registry gap removes the silent substitution.** It is not a
synonym-priority problem.

### What this means for the forecast cluster (P041–P045)

The economics already exist, deterministically and under governance:
`completion_probability` comes from the stage probabilities in
`config/client/pipeline_expected_funding.yaml` (with empirical override from
`pipeline_history`), `expected_funded_amount` from the case facility, and
`weighted_expected_funded_amount` from the two. **No new forecast economics are
required to answer P041–P045.** That makes them class D, not class E — but
exposing a forward-looking measure on the MI query surface is still a product
decision, and it is deferred to Phase 3 rather than taken here.

---

## Family 2 — a stated threshold is dropped from the spec

**4 rows: F032, P030 (SUSPECT); P025, P029 (NOT_ANSWERED).** Two mechanisms, one
symptom.

### 2a. The postfix comparator has its own number grammar (F032, P030)

`_POSTFIX_COMPARATORS` (`mi_agent/llm_query_parser.py:3087`) matches
`(-?\d+(?:\.\d+)?)\s*(?:years?|yrs?)?\s*(?:\+|\bor (?:above|over|…)\b)`. It has
**no `%`, no currency symbol and no k/m/bn multiplier**, and hard-codes `years?`
as the only unit — while the prefix grammar `_VALUE` allows all three.

```
85 or older      → ge 85       ✓
200000 or more   → ge 200000   ✓
7 or more        → ge 7        ✓
7% or more       → LOST        ✗
50% or more      → LOST        ✗
£200k or more    → LOST        ✗   (never exercised by this bank)
```

The threshold is not mis-applied, it is **never recognised**. With no filter and
a rate word in the sentence, the parser falls through to the weighted-average
rate over the whole book. Reproduced:

> "How many pipeline cases have an interest rate of 7% or more?"
> → *"Weighted-average Interest Rate: 6.9% · 10 loans · entire pipeline"*

This is why F032/P030 are SILENT rather than refused: the facet guard can only
report a requested facet that something recorded as requested, and nothing did.
`or more` is in the governed `COMPARATOR_PHRASES` and maps to `ge` correctly —
the failure is entirely the number format beside it. The seam is
frame-independent; funded and pipeline share it.

### 2b. The no-metric terminal region discards row filters (P025, P029)

In `_deterministic_parse`, the block guarded by `if dimension is None and metric
is None:` is a set of terminal returns, each of which constructs a **fresh**
`MIQuerySpec`. Only the `share` branch computes filters. The `amount` branch
(added 2026-09-04 for the product owner's "amount defaults to balance" rule), the
`wants_summary` branch and the `_ambiguous` branch all return specs with
`filters` unset — silently discarding every row filter the question stated.

The trigger is the word **amount**, not the word rate, and it drops filters of
any kind:

```
"total pipeline amount for cases with an interest rate above 6%"  → filters={}
"total pipeline balance for cases with an interest rate above 6%" → filters={'current_interest_rate': {'op':'gt','value':6.0}}
"total pipeline amount for cases with an LTV above 40%"           → filters={}
"total funded balance for loans with an interest rate below 5%"   → filters={'current_interest_rate': {'op':'lt','value':5.0}}
```

"balance" resolves through `_detect_metric`, so it never enters the block at all.
The funded half of the bank said "balance" and the pipeline half said "amount" —
which is the only reason this reads as a pipeline defect. It is not.

Here the guard DID have a facet, so it refused, reproduced verbatim:

> *"I understood that you asked for interest rate over 6, but that could not be
> applied to the calculation … I have not substituted a broader figure."*

Fail-closed behaviour is working correctly in both 2a and 2b. The defect is
upstream of it in both.

### P024 and P026 — not reproduced

Both answer correctly offline:

* P024 "How many pipeline cases have an interest rate above 6%?" →
  *"5 loans · Calculated: Count of loans · Interest Rate > 6 · 5 loans"* — correct.
* P026 "weighted average LTV for pipeline cases with an interest rate above 6%" →
  *"Weighted-average Current LTV: 50.0% · 5 loans"* — correct.

I will not assign these a class on the strength of a paraphrase of the live
question. They need the live receipt for the exact bank string before Phase 2
touches anything on their account.

---

## Family 3 — ranking language picks money over population

**2 rows: F039, P040 (both SUSPECT).**

```
"Which product type has the most funded loans?"   metric=current_outstanding_balance agg=sum dim=erm_product_type
"Which product type has the largest funded balance?"  ← byte-identical spec
```

Two different questions, one spec. Reproduced end-to-end:

> "Which broker has the most pipeline cases?"
> → *"Balance: £4.7MM · Broker: … · Calculated: Total Balance · grouped by Broker"*

`_RANK_DESC` is `("largest","biggest","highest","greatest","top ")` — "most" is
deliberately absent, so the sentence is not even seen as a ranking; it becomes a
grouped balance summary whose top row is read as the answer.

**The discriminator already exists and is already governed.**
`_counts_a_row_noun` (line 1402) reads the bare row noun standing as the subject,
and excludes anything carrying a money word via `_DEFAULTED_MEASURE_RE`:

```
"most funded loans"      → True    "largest funded balance"  → False
"most pipeline cases"    → True    "largest pipeline amount" → False
```

It is already wired into the trend branch at line 4260 for exactly this reason
("a trend of things is a count of them"). The ranking branch is the same
question asked with a superlative instead of a period, and it does not consult
it. `metric_defaulted` is also `False` on these specs despite no measure being
named — the same disclosure defect already fixed for trends.

---

## Family 4 — "headroom" is owned by the risk-limit route

**8 rows: F044 (SUSPECT), F043, F045, F046, F047, F048, F049, F050.**

`_RISK_LIMIT_RE` (`llm_query_parser.py:1738`) contains a bare `\bheadroom\b`. It
sets `risk_limit_query=True`, and the `risk_limits` recogniser
(`chat_routing.py:4021`, **priority 100**) claims the question. Reproduced:

> "What is the current NNEG headroom on the funded book?"
> → route `risk_limits`, *"4 passed, 0 warning(s), 7 breach(es) … Nearest to
>   limit: Top 3 brokers (-55.0 pp headroom)"*

That is a governing-document concentration report answering a question about
collateral shortfall on a lifetime book. It is the worst row in the bank: not a
wrong number, a wrong subject, delivered confidently. F045 behaves identically.
F046–F050 are claimed the same way (`risk_limits` / `concentration_analysis`)
and then refuse on the grouping dimension — an honest refusal reached for the
wrong reason.

### The secondary defect: the refusal names a field that cannot exist

`mi_agent_workflow._UNSUPPORTED_CONCEPTS` maps NNEG →`["nneg_flag"]`. **There is
no `nneg_flag` anywhere in the estate** — not in `mi_semantics_field_registry.yaml`
(the registered name is `negative_equity_guarantee`), not in the pipeline
contract, not in any tape. It appears only in this list and in two golden-question
fixtures that were built from it.

So F043's refusal — *"'NNEG' is not available in this dataset. This book does not
report it"* — is unconditional and its stated reason is unfalsifiable. It is not
evidence that the ERE tape lacks the NNEG inputs. It cannot be.

---

## B. NNEG inventory — the seven questions, answered

**1. Is NNEG exposure already computed deterministically?** Yes, in two places.
`mi_agent_api/evolution.py::_nneg_metrics` (per reporting period, feeding the
progression series) and `mi_agent_api/snapshots.py::_risk_tile` (point-in-time,
the dashboard's ERM risk tile).

**2. Is NNEG headroom already computed deterministically?** Yes, in one.
`_nneg_metrics` emits `nneg_headroom` and `nneg_headroom_pct`. `_risk_tile`
emits exposure only.

**3. What economic definition is used?**

```
exposure     = Σ max(0, balance − value)                 both owners agree
headroom     = Σ (value − balance)                       signed, aggregate
headroom_pct = 1 − (Σ balance / Σ value)                 value-weighted, aggregate
```

`value` is the first present of `indexed_valuation_amount`,
`current_valuation_amount`, `indexed_value`, `original_valuation_amount`; rows
with a null or non-positive valuation are excluded from the basis. `_risk_tile`
uses `current_valuation_amount` only and also publishes the count of loans above
valuation. The two definitions of exposure are arithmetically identical.

**4. Loan-level, aggregate, scenario-derived or point-in-time?** Row-wise then
summed to an **aggregate**, **point-in-time** figure at a single reporting date.
No loan-level NNEG column is materialised anywhere. Not scenario-derived — no
projection, no stress, no roll-forward.

**5. Can it safely be grouped by borrower type / age / product?** Economically,
yes: exposure and headroom are sums of a row-wise function of two loan-level
columns, so a partition is well defined. `headroom_pct` is a **ratio of sums** and
must be recomputed within each group, never averaged across groups. A grouped
precedent exists — `mi_agent_pptx/cohorts.py` carries `nneg_headroom_pct` per
vintage — but only along the vintage/period axis, because `_nneg_metrics` takes a
whole frame and the caller does the slicing. **No general grouped NNEG primitive
exists today.**

**6. What reporting date does it require?** One funded reporting cut, with
balance and valuation measured at the same date. Both owners are single-date by
construction; neither can span dates.

**7. Does the production tape carry every required input?** **Unknown from here,
and the live run does not answer it** — F043's refusal came from the phantom
`nneg_flag` gate, which would fire on any tape. The check is one line on the ERE
frame: `current_outstanding_balance` and `current_valuation_amount` both present
and non-empty (`snapshots._has_values`). The dashboard's ERM risk tile already
performs it and already fails closed with the missing field names, so whether the
ERE dashboard renders a figure in that tile settles it without new code.

---

## C. Phase 3 positions (recorded, not taken)

**NNEG — EXISTING PRIMITIVE for the ungrouped figures; NEW CAPABILITY for the
grouped ones.** `nneg_exposure`, `nneg_headroom` and `nneg_headroom_pct` at a
single date are governed and computed today; exposing them needs no new
economics. F046–F050 ask for them **by borrower type / age bucket / product**,
and no grouped primitive exists — building one means deciding, at minimum, how
`headroom_pct` behaves per group and how rows with no valuation are treated in a
partition. That is a business definition, and it is not remediation.

**Pipeline forecast — EXISTING PRIMITIVE, not exposed.** All three measures and
their assumptions already exist under governance. Exposing them on the MI query
surface is a scope decision, not an economics one.

**Whatever is decided, `\bheadroom\b` must stop being claimed unconditionally by
the risk-limit route, and `nneg_flag` must stop being the estate's stated reason
for declining NNEG.** Both are true independently of any capability decision, and
family 4's silent wrong answer is not fixed without the first of them.

---

## D. Blast radius, per family

| Family | Owner | Expected blast |
|---|---|---|
| 1 | `build_mi_semantics_registry.py` + regenerated YAML | **Wide.** The registry is read by the parser, executor, coverage ledger, receipt and every route. Adding fields widens the alias pool; `pipeline_case_age_days` competes with `youngest_borrower_age` for "age". Needs the full wide diff and an alias-collision census before, not after. |
| 2a | `_POSTFIX_COMPARATORS` | Narrow, but it is corpus-wide: any question with a postfix comparator. Both frames. |
| 2b | the no-metric terminal region of `_deterministic_parse` | Narrow in code, wide in reach — every filtered question that names no resolvable measure. Some currently answer whole-book and would begin to refuse or narrow; that is the correction, and it must be measured as a SET. |
| 3 | ranking branch of `_deterministic_parse` | Narrow. Reuses `_counts_a_row_noun`; the risk is questions that name a row noun AND want money, which `_DEFAULTED_MEASURE_RE` already excludes. |
| 4 | `_RISK_LIMIT_RE`, `_UNSUPPORTED_CONCEPTS` | Narrow for `nneg_flag`. Qualifying `\bheadroom\b` touches the whole risk-limit route, which is 49-question-bank territory — every "headroom" question in the 115 bank must be held. |

Families 2a, 2b and 3 are Phase 1 (they are five of the six SUSPECT rows and two
honest refusals). Family 4's routing fix is the sixth SUSPECT. Family 1 is Phase 2
and is the one that needs the wide diff.
