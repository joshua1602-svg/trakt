# Confirmation canary — the five, spent once

    bank        metric_delta_confirmation_canary
    sha256      9705705c7c66a605ccd3f16a08275b33f1b05b201c5f09f99bcd24386eb7d065
    run         34756499729, 2026-09-13T12:14:43Z → 12:18:10Z
    served      5d84a6a31ccf0f255c6c9170de97b01532fa35da
    asked       5 of 5, once each, no retries, no rephrasing
    VERDICT     FAIL — 4 PASS, 1 FAIL

The bank is SPENT. The result was committed by the job itself before it exited,
so the next checkout finds it spent rather than trusting a runner-local file.

## The flag, verified behaviourally rather than asserted

Every one of the five evidence records carries:

    serving.mode        = "canary"
    principal_matched   = true

against the single allow-listed oid recorded in the result JSON. No wildcard was
matched and no principal other than the bearer's own subject appears. That is
what `MI_AGENT_PLAN_SERVE = canary` plus an exact-match allow-list MEANS at the
point of service, and it is measured on every record rather than read back from
a setting this repository cannot see.

## What the four passes establish

| case | form | route | served from | mode | owner |
| --- | --- | --- | --- | --- | --- |
| M02 | metric_delta | `governed_plan_metric_delta` | NEW | requested_metric | `run_period_change_analysis` |
| M03 | metric_delta | `governed_plan_metric_delta` | NEW | requested_metric | `run_period_change_analysis` |
| M04 | material_summary | `governed_plan_material_summary` | NEW | portfolio_overview | `run_period_change_analysis` + `insight_funded.compose` |
| M05 | attribution | `governed_plan_attribution` | NEW | — | `period_change.bridge.balance_bridge` |

**S03 is closed.** S03 read its question flawlessly — metric_delta,
period_movement/movement, `current_outstanding_balance`, a stated relative pair,
compiled to a plan — and reached no owner, because no adapter claimed the form.
M02 and M03 are the same shape of question and they now reach
`run_period_change_analysis` in `requested_metric` mode and are served from the
governed plan. Neither is a legacy answer that happens to look right: both record
`response_served_from = NEW`.

**The field the plan named is the field the owner analysed.** M02 requested
`current_loan_to_value` and selected exactly that, nothing excluded; M03
requested `current_interest_rate` and selected exactly that, nothing excluded.
No candidate stood in for the field the reader asked about.

**Connecting metric_delta disturbed neither form already serving.** M04 and M05
came back on their own routes, their own owners and their own modes, with
`bridge_reconciles = true` on the bridge.

Both metric deltas record `temporal_default_applied = false`: the pair was stated
and no default was invented for it.

## M01 — FAIL, and it is a real product gap

    question   "Compare outstanding balance across the two most recent
                reporting dates and tell me the size of the shift."
    read       change_form claimed = metric_delta
               change_form compiled = metric_delta
               temporal route = relative_pair, stated
               measure = current_outstanding_balance
    served     LEGACY_FALLBACK
    reason     INELIGIBLE:OPERATION_NOT_ADMITTED
    outcome    REFUSE

The reading was correct in every slot. The plan compiled. The adapter then
refused it, and the legacy path answered with a refusal about a window the
reader never asked for ("You asked about the last 3 months…" — the reader asked
for the two most recent reporting dates).

### The cause, stated as far as it is measured

`plan_metric_delta.OPERATIONS = frozenset({"movement"})`. The refusal code means
the compiled operation was not `movement`; the record's receipt is empty because
nothing executed, so the actual value is NOT in this artefact and I am not
claiming one. `CAPABILITY_OPERATIONS["period_movement"]` admits
`{movement, rank, breakdown, compare, …}`, and for a question whose verb is
"compare", `compare` is the obvious candidate — unmeasured. It can be confirmed
for free from the sink record for M01, which carries the compiled plan; that read
needs no model call and no `/mi/query`.

### Why the other two adapters do not have this problem, and why this one does

`vocabulary.CHANGE_FORM_OPERATION_VARIANTS` canonicalises a form's linguistic
variants to its own operation — and it deliberately contains `material_summary`
only. Its own comment says why `metric_delta`, `attribution` and
`level_comparison` are absent. So for `material_summary`, `normalise` rule 5
collapses `compare` and `movement` to `summary` before a plan exists and the
adapter's single-operation perimeter is never tested in production. For
`metric_delta` nothing canonicalises, so whatever the capability admits arrives
at the adapter — and the adapter admits one value.

This is the restatement failure this programme is supposed to catch. The commit
that added this adapter took care to read the MODE from
`vocabulary.CHANGE_FORM_MODE` "rather than restated", and then hard-coded the
OPERATION perimeter beside it. `compare` is a shape
`run_period_change_analysis` in `requested_metric` mode genuinely produces — a
start value, an end value and the movement between them per requested field —
so refusing it is not the owner declining a shape it cannot make. `rank`,
`breakdown` and `series` are different: those state shapes this owner does not
produce and refusing them is correct.

### Why 65 offline controls did not catch it

C13b asks exactly this question — what an adapter does with a non-canonical
operation — and asks it of `material_summary`, where the answer is correctly
"fail closed", because canonicalisation upstream means such a plan cannot occur.
There is no equivalent control for `metric_delta`, the one form where such a plan
CAN occur. The controls tested the case that cannot happen and not the case that
can. That is a gap in my test design, not bad luck.

## What is NOT claimed

**No independent numeric parity.** CI has no governed book, so the owner cannot
be run independently over the same snapshots here. The figures below are recorded
so a reader with data access can reconcile them; they are not certified by this
run. Offline parity against fixtures is separate, passing evidence.

    M02  current_loan_to_value    weighted_average  39.5118 → 47.0383  +7.5266 pp
    M03  current_interest_rate    weighted_average   9.5334 →  6.3187  −3.2147 pp

## One observation that passed every pinned gate and still deserves an answer

M03 asked whether the weighted average interest rate had gone up or down and by
how much. Every gate this bank pins held: governed route, right owner, right
mode, `current_interest_rate` requested and selected. But the owner marked the
movement `status = "partially_available"`, the renderer therefore reported
"0 of 1 governed metrics could be compared across both snapshots", and the
published answer described the balance bridge instead of stating the −3.21 pp
the receipt holds.

Two readings, and this run does not choose between them: either the suppression
is correct governed caution about a weighted average whose weights are
incomplete across the pair, or a figure the owner computed is being withheld from
a reader who asked for it precisely. Recorded for the operator; not diagnosed
further and not repaired here.

Related and separate: all three period resolutions note the two snapshots cover
30 and 212 days, so period-flow fields are reported but not compared. That is
existing governed behaviour on this book, not something this sprint changed.

## Nothing was repaired during this run

Per the stop policy: the failed question was not retried, not rephrased and not
substituted; no expectation was changed; no product code was touched; the 135
was not run.
