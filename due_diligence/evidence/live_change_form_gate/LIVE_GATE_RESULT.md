# Live change_form interpretation gate — FAIL (17/18)

```
BASELINE_SHA          665176864d2f5dbceedc277e351c5d2cd978777f
TEST_BRANCH           claude/live-change-form-gate
TESTED_HEAD           0d3f0ca943b257447291b441a454d84a7afdc22f
PRODUCT_CODE_CHANGED  NO   (asserted in CI, not claimed)
BANK_ID               live_change_form_gate_v1
BANK_SHA256           4e6fca5920ea07b8c5e56321657ae47de9f2953c3c639184ab910570afd87c59
MODEL                 claude-opus-5   (requested and served)
RUN                   https://github.com/joshua1602-svg/trakt/actions/runs/34715249945

LIVE_MODEL_CALLS_ATTEMPTED    18
LIVE_MODEL_CALLS_SUCCESSFUL   18
PROVIDER_FAILURES             0
LIVE_MI_QUERY_CALLS           0      DEPLOYMENTS  0
```

The bank completed. This is a measurement, not a provider block.

## The gate

| metric | result | required | |
|---|---|---|---|
| CHANGE_FORM_TOTAL | **17 / 18** | 18 | **FAIL** |
| MATERIAL_SUMMARY | 5 / 5 | 5 | ok |
| METRIC_DELTA | **4 / 5** | 5 | **FAIL** |
| ATTRIBUTION | 4 / 4 | 4 | ok |
| LEVEL_COMPARISON | 4 / 4 | 4 | ok |
| EXPLICIT_MEASURE_PRESERVED | 13 / 13 | 13 | ok |
| MATERIAL_SUMMARY_SPECIFIC_MEASURE_INVENTED | 0 / 5 | 0 | ok |
| SCOPE_PRESERVED | 4 / 4 | 4 | ok |
| UNNECESSARY_CLARIFICATIONS | 0 | 0 | ok |
| INVENTED_FILTERS | 0 | 0 | ok |
| INVENTED_DIMENSIONS | 0 | 0 | ok |
| SILENT_SEMANTIC_DROPS | **2** | 0 | **FAIL** |
| *(recorded, not gated)* TEMPORAL_PAIR_PRESERVED | 16 / 18 | — | |

`LIVE_INTERPRETATION_GATE = FAIL`

What the run establishes positively is worth stating plainly, because it is most of
the sprint's thesis: the broad "what changed" family was read as
`material_summary` **5 times out of 5 with no measure invented even once**, the
Direct/Acquired scope survived **4 times out of 4**, every explicitly named
measure survived **13 of 13**, and nothing was clarified, filtered or grouped that
the reader did not ask for. All eighteen readings compiled to PLAN.

## The three failures

### CF07 — `change_form` absent entirely (class A)

> "How did weighted average LTV move over the latest reporting period?"

```
change_form  null                  <- expected metric_delta
capability   generic_analysis      operation  movement
measures     current_loan_to_value, statistic weighted_average
time.form    relative_pair
compiled     PLAN  generic_analysis / movement
```

The measure, its statistic and the period pair are all correct. The slot under
measurement is simply empty. The model wrote four disclosure notes on this
question — one of them reasoning explicitly about which capability the question
implies — and still never populated `change_form`.

### CF05 and CF11 — the period pair collapsed to a single state (class C)

> CF05 "What materially changed in the direct funded portfolio over the latest reporting period?"
> CF11 "What drove the change in funded balance this month?"

```
CF05  change_form material_summary  capability period_movement  operation summary
      lens direct (preserved)       time.form current   <- expected a pair
      compiled PLAN  period_movement / summary  scope [source_portfolio_type=direct]

CF11  change_form attribution       capability funded_bridge    operation bridge
      measures funded_balance_movement
      time.form current, grain monthly  <- expected a pair
      compiled PLAN  funded_bridge / bridge
```

Both got the form right. Neither is careless: CF11 states its reasoning outright —

> "'this month' read as the current governed monthly reporting period; **the
> bridge defines its own comparison period**."

— and that is the same principle Sprint A wrote down for the measure axis, applied
by the model to the temporal axis. `plan_material_summary.period_request` does in
fact resolve an absent grain to `current_vs_previous`, so for CF05 the
capability really does own the pair. The readings are defensible. They still fail
the pre-registered expectation, and the expectation was not relaxed to accommodate
them.

## The smallest general semantic defect the evidence supports

**The change-intelligence contract has no completeness obligation, so an intent
that under-specifies it still compiles to PLAN, and the under-specification is
invisible at the governed boundary.**

One defect, one shape, covering all three failures:

- `change_form` is `Optional` and nothing downstream may require it — so CF07's
  empty slot compiled to a normal plan rather than being caught.
- There is no slot that can say **"the capability owns the period pair"**. CF05 and
  CF11 needed to express exactly that, had nowhere to put it, and so stated
  `time.form = current` and explained the rest in prose.
- All three put their real reasoning in `ambiguity` notes, which are free text the
  compiler does not read. None raised a *blocking* ambiguity, so nothing asked.

The consequence is the one `change_form` was introduced to eliminate, reappearing
one axis over: the same business question can reach the deterministic layer with
two different readings — `time.form = current` or `relative_pair`, `change_form`
set or null — and **both plan**. `change_form` fixed this for "who owns the
candidate set of measures". Nothing yet fixes it for "who owns the period pair",
and nothing obliges the form to be stated at all.

Not implemented. Per the brief, this run is measurement only.

## One harness gap, recorded

The runner stores every semantic slot of each reading but does **not** persist the
verbatim `raw_payload`, so the per-case JSON in `change_form_result.json` carries
`raw_payload: null`. Every field needed to score and to diagnose is present — form,
capability, operation, measures with canonical resolution, population, lens,
temporal, filters, dimensions, both kinds of ambiguity — and the three failures are
fully characterised above from it. But the byte-exact model output for this run was
not kept. The fix is one line in the runner; it was **not** applied, because doing
so would invalidate the pinned harness this measurement was taken with and a re-run
would spend eighteen more calls to recover data the gate did not turn on.

## What was not done

No gate was relaxed after seeing results. No question was asked twice. No failed
case was re-run. No cheaper model was substituted. No product code was changed —
CI asserts that by diffing against the baseline, not by assertion in prose. The
bank and its expectations are byte-identical to the pre-registration in `480f491c`.
