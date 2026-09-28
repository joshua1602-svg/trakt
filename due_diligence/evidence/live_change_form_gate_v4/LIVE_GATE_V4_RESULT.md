# Live gate v4 — FAIL on one count; both measured contracts clean

```
PRODUCT_HEAD                    dea97562  (approved f22d0e46 + this gate's workflow only)
PRODUCT_CODE_CHANGED_DURING_RUN NO        (asserted in CI, not claimed)
BASELINE                        665176864d2f5dbceedc277e351c5d2cd978777f
BANK_ID                         live_change_form_gate_v4
BANK_SHA256                     26345762e04537246183d3218b82f787669c844c157ad4f1df9f1d0ee4762dce
MODEL                           claude-opus-5   (requested AND served, all 17)
RUN                             https://github.com/joshua1602-svg/trakt/actions/runs/34747532767

LIVE_MODEL_CALLS_ATTEMPTED      17
LIVE_MODEL_CALLS_SUCCESSFUL     17
PROVIDER_FAILURES               0
LIVE_MI_QUERY_CALLS             0     DEPLOYMENTS 0
```

## Both contracts under measurement came back clean

```
CHANGE_FORM_TOTAL = 17 / 17       ok
  material_summary 4/4   metric_delta 5/5   attribution 4/4   level_comparison 4/4

TEMPORAL_ROUTE_VALID = 17 / 17   ok
  EXPLICIT_PAIR     11 cases   11 pass
  EXPLICIT_ANCHOR    2 cases    2 pass
  OWNER_DEFAULT      4 cases    4 pass
  routes taken: 12 pair, 2 anchor, 3 absent
  TIME_STATED: 14 true, 3 false
```

**The presence repair works live.** Three readings (W10, W12, W13 — all attribution)
stated no temporal form at all, and all three were recorded as
`TIME_STATED = false, INTERPRETED_TIME_FORM = null` and accepted as
`OWNER_DEFAULT_ABSENT`. Not one was reported as an explicit `current`. Under v3's
contract those same three would have been counted as temporal failures; under the
repaired one they are correct readings whose window the deterministic owner supplies.

**And the discipline held where it should.** `metric_delta` and `level_comparison`
stated a pair in all nine of their cases — neither inherited the default.

| secondary gate | result | |
|---|---|---|
| EXPLICIT_MEASURE_PRESERVED | 13 / 13 | ok |
| MATERIAL_SUMMARY_MEASURE_INVENTED | 0 / 4 | ok |
| SCOPE_PRESERVED | 4 / 4 | ok |
| UNAUTHORISED_FILTERS | 0 | ok |
| INVENTED_DIMENSIONS | 0 | ok |
| SILENT_SEMANTIC_DROPS | 0 | ok |
| **UNNECESSARY_CLARIFICATIONS** | **1** | **FAIL** |

16 of 17 cases pass. Compile: 16 COMPILED, 1 CLARIFY, reaching the four distinct
contracts `period_movement/summary`, `period_movement/movement`,
`funded_bridge/bridge`, `generic_analysis/compare`.

```
LIVE_INTERPRETATION_GATE_V4 = FAIL
```

## The one failure, and what it actually shows

**W06** — *"What is the movement in the number of loans in arrears since the prior
reporting date?"*

Everything else is right: `change_form = metric_delta`, measure `loan` + `count`,
`time.form = relative_pair`, population funded. It invented **no** filter. What it
did was raise a **blocking** ambiguity on `filters`:

> "The question restricts to loans 'in arrears'. The governed arrears status concept
> (`interest_in_arrears`) and the arrears band concept (`arrears_bucket`) both carry
> no governed value list, so no arrears value can be asserted without guessing.
> Please confirm how 'in arrears' should be expressed…"

**That claim is true.** Checked against the registry:

```
interest_in_arrears        values = (NONE PUBLISHED)
arrears_bucket             values = (NONE PUBLISHED)
number_of_days_in_arrears  values = (NONE PUBLISHED)
arrears_balance            values = (NONE PUBLISHED)
account_status             values = Active / Redeemed   <- does not express arrears
```

So the clarification is **necessary**, not unnecessary. Asking rather than
substituting is the behaviour this whole architecture exists to produce. The gate
counts it only because the bank pinned `UNNECESSARY_CLARIFICATIONS = 0` globally
while containing a question that cannot be expressed in the governed vocabulary at
all.

## The smallest evidence-supported residual defect

**Governed arrears concepts are exposed to the interpreter as filterable while
publishing no value vocabulary, so a question that restricts on arrears cannot be
expressed and must necessarily clarify.**

Four concepts — `interest_in_arrears`, `arrears_bucket`, `number_of_days_in_arrears`,
`arrears_balance` — are offered as filter targets with empty `values` and an empty
`values_source`. A reader asking about "loans in arrears" is asking something the
registry cannot currently express, and the only correct outcome is to ask back.

This is a **registry value-coverage gap**, not a defect in the change_form contract
or the temporal contract. Both of those measured clean at 17/17.

Not implemented. This run is measurement only.

## My third authoring defect in the same family, stated plainly

This one is worth naming because it was introduced while *fixing* the previous one.

- **v3/V06** taught: a restricting adjective ("live") is not an invented filter. So
  v4 added `filters_permitted` per case.
- **v4/W06** shows that was necessary but **not sufficient**: a concept can be
  *permitted* and still have no value vocabulary, so permitting the concept does not
  make the restriction expressible. Authoring a question whose restriction the
  registry cannot express, while pinning zero clarifications, is a bank defect.

The gate is scored exactly as pinned regardless, and nothing was relaxed after the
result was seen.

## Integrity

```
v1 manifest hashes to its pin    OK   (immutable, 17/18)
v2 manifest hashes to its pin    OK   (immutable, retired UNRUN)
v3 manifest hashes to its pin    OK   (immutable, change_form 18/18)
v4 manifest hashes to its pin    OK   (this measurement)
```

v1's, v2's and v3's hashes were asserted **in CI before the first call**, not just
checked afterwards. No expectation changed after seeing results, no question was
asked twice, no case re-run, no model substituted, and no product code changed during
the run — CI proved that by diffing against the approved head.

One CI-plumbing bug was caught by those assertions before any spend: deriving this
workflow from v3's renamed every occurrence of the v3 filename except the one inside
the allowlist grep, where the dot is escaped, so the workflow flagged its own file as
a stray change. Fixed; no evidence or expectation was touched.

The full verbatim `CandidateIntent` is persisted for all seventeen in
`change_form_v4_result.json`, with each case's score, temporal route, time-stated
flag, compile outcome, served model, bank hash and baseline SHA.

This gate measures interpretation. It computes no answer and claims no actual FROM/TO
snapshot values.
