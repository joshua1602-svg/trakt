# metric_delta output-contract closeout

    PRODUCT_BASELINE = 5d84a6a31ccf0f255c6c9170de97b01532fa35da
    PRODUCT_HEAD     = (this commit)

Two bounded repairs, both at the layer that already owns the thing being
repaired, and both proved against the live evidence that found them.

---

# OBJECTIVE A — M01, proved from the record

Read from the sink record `plan_serving_canary` wrote during run 34756499729
(correlation `shadow_f6bdfe6a04404100bd07`). No question was asked, no model
called, no endpoint touched. Nothing below is inferred from the English.

    M01_CANDIDATE_OPERATION   = "compare"      model.raw_payload.operation
    M01_NORMALISED_OPERATION  = "compare"      interpretation.candidate_intent
    M01_COMPILED_OPERATION    = "compare"      compiler.plan.operation
    CHANGE_FORM               = "metric_delta" claimed AND compiler-bound
    CAPABILITY                = "period_movement"
    MODE                      = "requested_metric"
    compile outcome           = PLAN, reason_codes []

The normalisation trace the compiler recorded carries exactly one rule:

    change_form: capability 'generic_analysis' -> 'period_movement'
                 (derived from change_form 'metric_delta')

Rule 5 — the operation rule — did not appear, because it did not fire.

## Why a valid metric_delta reached the adapter with an operation it refused

`normalise.canonical_intent` rule 5 collapses a form's linguistic variants to its
canonical operation, and it is driven entirely by two central tables:
`CHANGE_FORM_CANONICAL_OPERATION` and `CHANGE_FORM_OPERATION_VARIANTS`. Both
carried `material_summary` and nothing else. So for `metric_delta` the rule had
no canonical operation to collapse into and left `compare` standing. The compiler
then checked `compare` against `CAPABILITY_OPERATIONS["period_movement"]`, which
admits it, and emitted a plan with no reasons. `plan_metric_delta.OPERATIONS`
admits `movement` alone, so the adapter refused, the legacy path served, and the
reader got a refusal about a window they never mentioned.

    M01_ROOT_CAUSE = C — BOTH

**A — missing central canonicalisation.** The form was absent from the tables
that exist precisely to collapse a reader's spelling of an action into the one
shape its owner executes.

**B — the adapter restating a perimeter owned centrally.** `OPERATIONS =
{"movement"}` is a second statement of the form's executable shape. It is the
correct set; it was simply the only place the rule was written down, so it acted
as a gate instead of as defence in depth.

## `compare` is a variant of this form, not another form's action

The one thing that could have made this "genuinely incompatible — STOP" is if
another change form owned `compare`. `level_comparison` is that candidate: "two
states, compared as levels, without movement intelligence". It runs
`generic_analysis`. A `compare` that arrives under `period_movement` is there
because the form stated alongside it owns that capability, which is
`CHANGE_FORM_CAPABILITY`'s own rule applying. **No form owns a `compare` over
`period_movement` with a named measure**, so admitting it takes nothing from
anyone, and `test_E9` pins that separation.

## The repair, and where it is not

    CENTRAL_OPERATION_CONTRACT_CHANGED      = YES  (2 table entries)
    ADAPTER_OPERATION_REINTERPRETATION_ADDED = NO  (0 lines)

```python
CHANGE_FORM_CANONICAL_OPERATION = {"material_summary": "summary",
                                   "metric_delta": "movement"}
CHANGE_FORM_OPERATION_VARIANTS  = {"material_summary": {"summary","movement","compare"},
                                   "metric_delta":     {"movement","compare"}}
```

`normalise.py` is untouched — the rule was already generic over both tables. The
adapter is untouched except for two comments that had become false. Measured
across the whole capability operation set, on M01's exact slots:

| stated | compiled | where it lands |
|---|---|---|
| `compare` | **`movement`** | rule 5 collapses it — **eligible, serves** |
| `movement` | `movement` | unchanged — eligible |
| `rank` | — | **UNSUPPORTED_COMPOSITION at the compiler** |
| `breakdown` | — | **UNSUPPORTED_COMPOSITION at the compiler** |
| `series` | `series` | carried through, refused by the adapter |
| `summary` | `summary` | carried through, refused by the adapter |

The set stops at `compare` on purpose: `rank`, `breakdown`, `series` and
`summary` state result shapes the requested-metric mode does not produce, so each
is left exactly as stated and refused rather than flattened.

## Two pinned controls asserted the opposite, and were rewritten

This is the significant act in this sprint and it is not buried.

**`test_the_operation_is_not_normalised_by_the_form`** asserted
`movement.plan.operation != compared.plan.operation`, reasoning that "`compare`
and `movement` are two different requests of the same owner, and collapsing them
would be deciding for the reader". The estate contains no such pair of requests:
`run_period_change_analysis` in `requested_metric` mode has one shape for a named
metric over a governed pair, and nothing anywhere executes a `compare`
differently. The control now asserts the collapse AND keeps the half that
survives — `series` and `summary` are still carried through unrewritten.

**`test_Q20_metric_delta_routing_is_unchanged[compare]`** asserted
`plan.operation == operation`, on the reasoning that "this form has no canonical
operation of its own to collapse into". It has one. The control now asserts
`movement`, still guards against collapse into `summary`, and gained
`test_Q20_a_shape_this_owner_does_not_produce_is_still_not_collapsed`.

No other pinned control changed. These two were the only failures my change
introduced anywhere.

---

# OBJECTIVE B — M03, traced to the owner that sets the status

## What is recorded, and what is not

    START_ROWS         = 73     resolved_start_snapshot.row_count  (2025-11-30)
    END_ROWS           = 958    resolved_end_snapshot.row_count    (2026-06-30)
    WEIGHT_FIELD       = current_outstanding_balance
                                config/business_semantics_registry.yaml,
                                current_interest_rate.weight_field
    START_VALUE_ROWS   = NOT RECORDED
    END_VALUE_ROWS     = NOT RECORDED
    START_WEIGHT_ROWS  = NOT RECORDED
    END_WEIGHT_ROWS    = NOT RECORDED
    START_COVERAGE     = NOT RECORDED
    END_COVERAGE       = NOT RECORDED

The counts are NOT in the artefact, NOT in the sink record and NOT in the
receipt. `MetricChange.to_dict()` carries `valid_population`,
`excluded_population`, `start_detail` and `end_detail`, but
`plan_metric_delta.receipt` projected only field, aggregation, the two values,
the movement, the unit and the status — so the one question "why was this
qualified?" was the question its own evidence could not answer. **That is fixed
here and is part of the repair, not a note about it.** They are recorded from now
on; they cannot be recovered for the spent run without asking again, and this
sprint asks nothing.

## PARTIAL_REASON

    PARTIAL_REASON = rows carrying no numerically valid `current_interest_rate`
                     in at least one of the two snapshots

`calculations._aggregate_weighted` masks on `values.notna() & weights.notna()`,
sets `excluded = total_rows - valid`, and returns
`STATUS_AVAILABLE if excluded == 0 else STATUS_PARTIALLY_AVAILABLE`. The value is
computed either way. `metric_change` then finds both sides `.ok` — because
`AggregateOutcome.ok` admits `partially_available` — computes
`movement = end - start`, and downgrades the pair's status to
`partially_available`.

**The weight was not the cause, and that is derived rather than assumed.** The
same run's M04 overview listed Current Outstanding Balance among its "largest
observed movements", a list built from `comparable`, which is `available` only —
so the balance had zero exclusions at both dates, so every row carried a valid
weight. The exclusion was therefore on the rate itself. Derived from two recorded
facts; the counts remain unrecorded.

## The publishability decision, read out of existing contracts

    M03_EXISTING_POLICY   = PUBLISHABLE_WITH_CAVEAT
    M03_TARGET_DISPOSITION = QUALIFIED

Four existing owners already say so, and no new policy is created:

1. `AggregateOutcome.ok` admits `STATUS_PARTIALLY_AVAILABLE`.
2. `metric_change` computes and KEEPS `movement_value` for such a pair —
   contrast `NOT_COMPARABLE_DUE_TO_AVAILABILITY`, which leaves it `None`.
3. `period_change_route._metric_rows` publishes every metric change whatever its
   status, so **the −3.21 pp was already in the served table**.
4. `period_change_route` line 839 gates a distribution on
   `status in (AVAILABLE, PARTIALLY_AVAILABLE)` — the route's own statement that
   partially available is publishable.

What `partially_available` DOES cost is comparability:
`MetricChange.comparable` requires `AVAILABLE` strictly, and that governs ranking
and the "N of M compared" count. Publication and comparability are two different
gates in this estate, and only the second excluded M03.

## The defect, exactly

`build_answer` narrates `result.summary`. Every clause it had was built from
`comparable`: the count, `top_movements_by_unit`, the improvement/deterioration
split. A partially available metric falls out of all three. In
`portfolio_overview` mode that is correct and complete — the reader asked what
moved and "nothing comparably" is the answer. In `requested_metric` mode the
reader named one metric, all three clauses go silent about it, and the only
surviving clause was the balance bridge. The route executed perfectly and
answered a different question.

## The repair

    REQUESTED_METRIC_CAN_BE_REPLACED_BY_BRIDGE = NO

`build_summary` gains `requested_metrics`, present only when
`selection.mode == MODE_REQUESTED_METRIC`, copied field by field off the owner's
own `MetricChange` — values, status, weight field, valid and excluded
populations, and the owner's notes. Nothing is recomputed. Each row carries a
disposition that NAMES what the status already means:

    ANSWERED          a movement that stands unqualified
    QUALIFIED         a movement that stands with an exclusion behind it
    GOVERNED_REFUSAL  the owner produced no movement at all

`build_answer` leads with those rows and then continues into the overview
narrative unchanged, so the bridge is supplementary and never a substitute. One
presenter, one calculation: `test_E_owner_parity_of_the_requested_metric_block`
asserts the served answer is byte-identical to `build_answer` run over an
independently executed owner result.

---

# Certification

    CERTIFICATION_SCORER_FIXED = YES
    SPENT_CANARY_MODIFIED      = NO

`certification.score_requested_metric` establishes, from structured evidence:

    REQUESTED_FIELD              receipt.requested_fields
    EXECUTED_FIELD               receipt.selected_measures
    SERVED_METRIC_FIELD          the served "Metric movements" artefact
    REQUESTED_METRIC_DISPOSITION receipt.requested_metric_disposition

and then reads the served ANSWER for the metric's name and, where a figure
exists, that figure — because the receipt and the table were both RIGHT in M03
and the prose was not, so a scorer that stops at structured evidence cannot see
this class of defect at all. The tokens it looks for are derived from the receipt
and rendered by the estate's own `_format_movement`, so no governed wording is
pinned and rewording the narrative cannot fail it.

Its `--self-test` rejects the answer M03 was actually served, accepts a corrected
one, rejects an answer that names the metric without its figure, and accepts a
governed refusal that names the metric. Three further controls exercise it
against real served envelopes rather than fixtures, including one that replaces
only the `answer` with the overview narrative and requires a FAIL.

---

# Offline controls

    OFFLINE_CONTROLS = 86 (65 before, +21)

| # | control | proves |
|---|---|---|
| 1 | E1 | metric_delta + canonical `movement` → requested_metric owner |
| 2 | E2 | M01's exact slots — `compare` canonicalised centrally, then served |
| 3 | E3 / E3b | `series`, `summary` refused by the adapter; `rank`, `breakdown` refused earlier, by the compiler |
| 4 | E4 | a fully available requested metric is named and stated in the answer |
| 5 | E5 | a partially available one is published with its exclusion counts, and still excluded from the comparability count |
| 6 | E6 | an unavailable one substitutes nothing — no other metric's name appears |
| 7 | E7 / E7b | material_summary unchanged, and the overview narrative unedited |
| 8 | E8 | attribution unchanged |
| 9 | E9 | level_comparison still claimed by no adapter; its `compare` uncollapsed |
| 10 | E10 | `build_answer` takes a result and nothing else; no `question`, no `re` in any new function |
| — | E11 a/b/c | the certification gate against real served output, both directions |

```
OWNER_PARITY = 3 / 3    NUMERICAL 0 · PERIOD 0 · SCOPE 0 · OWNER 0
```

Presentation qualifies a value according to its status; it never alters one. Each
served figure is compared against the same figure on an independently executed
`run_period_change_analysis`, and the served answer against `build_answer` run
over that independent result.

```
NEW_ANALYTICAL_CALCULATION_OWNERS = 0
RAW_QUESTION_ROUTING_ADDED        = 0
PRODUCT_EXECUTABLE_LOC            = +126 / −10   (budget ~150)
```

---

# Regression

Every baseline below was taken by stashing this change and re-running the same
command in the same container, so the comparison is against this environment
rather than against a number recorded in an earlier sprint.

| suite | before | after | failing ids |
|---|---|---|---|
| `tests/interpretation_v2/test_change_intelligence_serving.py` | 65 passed | **86 passed** | 0 → 0 |
| `tests/interpretation_v2/` | 565 passed, 3 failed | **586 passed, 3 failed** | **identical 3**, all pre-existing |
| `mi_agent_api/tests/` | 1907 passed, 26 failed | **1907 passed, 26 failed** | **identical 17 ids** |
| `mi_agent/tests/` | 1768 passed, 22 failed | **1768 passed, 22 failed** | **identical 19 ids** |

The three pre-existing `interpretation_v2` failures — `test_bank_and_shadow_boundary`,
`test_contract_normalisation::test_only_contract_modules_moved`,
`test_interpretation_policy::test_production_surfaces_are_untouched` — fail
identically on the pristine tree and are untouched by this change.

```
NEW_UNEXPLAINED_FAILURES = 0
```

Two controls DID change, and they are not counted as "unexplained" because they
are the subject of the sprint rather than collateral: both asserted that
`metric_delta` has no canonical operation, and both were rewritten with their
reasoning replaced rather than their assertions merely flipped. They are set out
in full under Objective A above.

Forms explicitly re-proved unchanged: `material_summary` (E7, E7b),
`attribution` (E8), `level_comparison` (E9).

---

# Fresh canary — prepared, NOT run

    FRESH_CANARY_PREPARED = YES
    FRESH_CANARY_BANK_ID  = metric_delta_output_contract_canary
    FRESH_CANARY_SHA256   = 6465f4ed962f058d9c9e9f63ecb941cb19dcf60a218e27339cf60d92e39a6996
    authorised_live_calls = 5, calls_per_case = 1, retries = 0

| case | form | canonical op | measure | temporal | owner | disposition | user-facing outcome |
|---|---|---|---|---|---|---|---|
| N01 | metric_delta | `movement` | funded/outstanding balance | EXPLICIT_PAIR | `run_period_change_analysis`, requested_metric | ANSWERED | names the metric, states the movement |
| N02 | metric_delta | `movement` | weighted-average LTV | EXPLICIT_PAIR | same | ANSWERED | same |
| N03 | metric_delta | `movement` | weighted-average interest rate | EXPLICIT_PAIR | same | QUALIFIED *or* ANSWERED | same, plus exclusion counts when QUALIFIED |
| N04 | metric_delta | `movement` | principal balance | EXPLICIT_PAIR | same | ANSWERED | same |
| N05 | material_summary | `summary` | none named | OWNER_DEFAULT | + `insight_funded.compose` | none | the overview narrative, no requested-metric block |

N01 and N04 are the live proof of the M01 repair, in wording M01 did not use;
N03 replaces M03 and is the live proof of the output contract. Wording
uniqueness was verified programmatically against all six historical manifests,
each of which was re-hashed against its pin in the same pass, before the file was
written.

**The operation is pinned as the CANONICAL value the compiled plan must carry,
not as the spelling the model emits.** The contract collapses `compare` and
`movement`; pinning which one Opus produces would be pinning a coin toss.

**N03's disposition is an accept-set, deliberately.** Its metric returned
`partially_available` on this book in the last live run, which is why it is in
the bank — but availability is a fact about the data on the day, not about the
code, and a book that improves between runs would turn a correct `ANSWERED` into
a false FAIL. Pinning a data condition as though it were a contract is the
v4/W06 authoring defect, and it is not repeated. What is pinned is the invariant
that belongs to this sprint: the figure reaches the reader either way.

**What is deliberately absent.** No unavailable-metric case: the
anti-substitution rule is proved offline against a fixture built for it, and no
field is KNOWN to be single-dated on this book, so a live case would pin a guess.
No S09 retest — unchanged data history. No attribution case — nothing in this
sprint touches its adapter, owner or envelope, whereas `material_summary` shares
the presenter that changed, which is why it is the regression case.

**The runner is not built.** The bank is hashed and unspent; wiring
`certification.score_requested_metric` into a canary runner and a workflow is the
first step of any live run and is not done here, because this sprint was told not
to run one.

One authoring hazard caught and recorded: `.gitignore` carries `*out*.json`,
which silently matched the manifest's first filename. The hash file would have
been committed and the manifest it pins would not, so no CI checkout could have
verified the bank. Caught before anything was spent; every other evidence file in
the programme was then audited and none is affected.
