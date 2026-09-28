# P0 target state — the governed population perimeter

    STATUS    design only. Nothing implemented, nothing changed.
    BASIS     production question-bank run of 135 questions (review at 036778a1)
              product code at 5c436961ebe279bc0820ab006867b9f8a869bde2
    SCOPE     the funded / pipeline / forecast population bases.
              `whole_book` is explicitly out of scope — see §11.

## 0. Decisions taken

    D1  VINTAGE SKEW POLICY                                        SETTLED 2026-09-28
        Caveat, with a 45-day ceiling, the same ceiling the funded book uses.
        Beyond the ceiling: refuse. Within it: answer, with both vintages stated
        on the face of the answer. Owner decision; recorded, not inferred.

    D2  FORECAST OPERATION MAPPING (scenario, cohort_conversion)    OPEN — see §6
        Blocks writing plan_forecast_runtime. Nothing else.

    D3  SEQUENCING                                                 OPEN — see §11
        Recommendation: P0 and P1 committed as one piece of work.

## 1. What the evidence asks this design to fix

    80 of 135 questions are pipeline or forecast and fell back to legacy
    26 fell back specifically on INELIGIBLE:POPULATION_NOT_EXECUTABLE
     0 of 48 forecast questions reached a governed plan
     4 of 15 pipeline_evolution questions reached one

And the capability is not missing. These legacy owners answered correctly on
production data during the same run:

    17  forecast_extrapolation      run-rate, milestones, scenario bands
     4  analytical_composition      composed funded + pipeline forecast
     3  cohort_conversion           9.4% cumulative cohort conversion
     3  evolution_pipeline_stage    stage series across 90 extracts
     3  temporal_compare            governed period comparison
     1  evolution, 1 scenario, 1 concentration_analysis

So this is a **connectivity** design, not a capability design. Nothing below adds
arithmetic.

## 2. What already exists — do not rebuild any of it

The perimeter primitive and the declaration pattern are both already in the tree.
This design generalises them; it does not invent them.

**`plan_runtime_adapter.check_population_base(plan, executed_population, executable)`**
is already parameterised on both the executable set and the population the runtime
says it actually loaded, and already returns three distinct refusals:

    POPULATION_NOT_EXECUTABLE   the plan's base is outside this runtime's set
    POPULATION_BASE_MISMATCH    requested and executed disagree
    EXECUTED_POPULATION_UNPROVEN the runtime declared nothing, so nothing is proven

**`plan_pipeline_runtime`** already declares its own population, in the shape this
design makes the rule:

    CAPABILITY = "pipeline"
    EXECUTABLE_POPULATIONS = frozenset({"pipeline"})
    EXECUTION_POPULATION   = "pipeline"

**`plan_serving_canary`** already dispatches specialist owners before the funded
perimeter runs, with a comment that states the principle exactly: *"A pipeline plan
is not the funded runtime's to refuse."* And it already iterates a registry for the
change_form owners: `for owner in (material_summary, attribution, metric_delta)`.

**`plan_attribution`** is the precedent for how a new owner is added — *"Plan
connectivity … THE MISSING ARROW ONLY … It contains no bridge arithmetic."*

## 3. The defect in one line

`{"funded"}` is **restated in five places** as the answer to "which populations does
this runtime execute":

    plan_material_summary.py:63   POPULATION_BASES        = frozenset({"funded"})
    plan_attribution.py:72        POPULATION_BASES        = frozenset({"funded"})
    plan_metric_delta.py:69       POPULATION_BASES        = frozenset({"funded"})
    plan_temporal_runtime.py:81   SLICE_2_POPULATION_BASES= frozenset({"funded"})
    plan_runtime_adapter.py:169   EXECUTABLE_POPULATIONS  = frozenset({"funded"})

Five copies of one fact, and one of them — the adapter's — is the *default* every
other caller silently inherits. There is no single place that answers "which
populations can this estate execute, and by which owner", so widening the estate
means editing five files and hoping none was missed. That is the property to remove.

## 4. Target state — the runtime declaration contract

**One rule: a runtime declares which populations it executes, and which one it
actually loaded. Nothing else may assert either.**

Every governed runtime exposes the same four names — three of which
`plan_pipeline_runtime` already exposes:

    CAPABILITY              str            the plan capability this owner claims
    EXECUTABLE_POPULATIONS  FrozenSet[str] what it will accept
    EXECUTION_POPULATION    str            what it loads, for the mismatch guard
    POPULATION_INPUTS       Mapping        NEW — see §4.1, derived populations only

and the same two functions it already exposes: `claims(plan)` and an eligibility
check that ends by delegating structure to the adapter.

`plan_runtime_adapter.EXECUTABLE_POPULATIONS` stops being a default other modules
inherit and becomes what it honestly is: **the generic funded executor's own
declaration**, renamed so it cannot be mistaken for an estate-wide constant. Callers
pass `executable=` explicitly or the call is a bug.

The four restatements in §3 are deleted and replaced by each module's own
declaration. Three of them (`material_summary`, `attribution`, `metric_delta`) keep
the value `{"funded"}` — those forms genuinely are funded-only — but they now *own*
that statement instead of repeating a constant, which is what makes a later change
to one of them local.

### 4.1 A derived population must declare its inputs, and its skew

`forecast` is not a dataset. The production evidence shows what it actually is:

    "Current funded balance is £87.1m as at 2026-08-31. Gross pipeline in the
     governed extract is £1.18bn as at 2026-09-24. Expected completions from the
     open pipeline: £6.9m. … Forecast funded balance: £94.1m."

Two datasets, at two different cut-off dates, composed. `EXECUTION_POPULATION =
"forecast"` alone would satisfy the existing mismatch guard while proving almost
nothing, so a derived population additionally declares its lineage:

    POPULATION_INPUTS = {
        "funded":   {"required": True, "as_of": "governed funded snapshot"},
        "pipeline": {"required": True, "as_of": "governed weekly extract"},
    }

and the receipt carries the resolved `as_of` for each input.

**D1 — the skew rule, and it introduces no new number.** The ceiling is not a
constant in this module. It is read from the policy that already owns the funded
book's ceiling:

    mi_agent/period_change/selection.py:100
        SelectionPolicy.max_snapshot_gap_days(context_id)
    config/period_change_selection.yaml:47
        max_snapshot_gap_days: 45
        max_snapshot_gap_days_by_portfolio: {}   # per-portfolio overrides

That method is already per-portfolio, with a documented reason — *"A book on a
quarterly cadence legitimately needs a wider ceiling than a monthly one, so the value
is per-portfolio configuration rather than a constant."* The forecast runtime calls
the same method with the same `context_id`. **No new config key, no second 45.** A
lender that widens its ceiling widens it once, for both.

    inputs within the ceiling   ANSWER, both vintages stated on the answer
    inputs beyond the ceiling   REFUSE, naming both dates and the ceiling
    a required input missing    REFUSE  (POPULATION_INPUT_UNRESOLVED, §9)

**Two distances, one ceiling — and the receipt must say which.** The funded rule
measures the distance from a REQUESTED period to the nearest available snapshot
("you asked for 31 May, the nearest snapshot is 30 Nov, 182 days"). The forecast rule
measures the distance BETWEEN TWO DATASETS' cut-offs ("funded 31 Aug, pipeline
24 Sep, 24 days"). Both answer the same question — how far apart may two dates be
before the answer stops meaning what it says — and one ceiling over both is the
right call. But they are not the same measurement, so the receipt names the distance
it checked (`input_vintage_skew_days` alongside the existing `gap_days`), or a reader
will later mistake a forecast caveat for a snapshot-gap caveat.

**What D1 means for production today.** Funded 2026-08-31 against pipeline
2026-09-24 is **24 days, inside the 45-day ceiling**. So the current book produces a
forecast that ANSWERS and carries both dates. D1 does not block today's forecast; it
makes it state its own basis. The ceiling bites only if the pipeline extract goes
stale — which is exactly when a forecast should stop being published quietly.

**If the two ceilings should ever differ,** the extension point is the policy method
(a named distance argument), not a new constant in the forecast module. Stated so the
next person does not reach for the constant.

## 5. Change 1 — one registry, and dispatch derived from it

Replace the hand-written dispatch with one ordered registry that the canary reads:

    GOVERNED_RUNTIMES = (pipeline_rt, forecast_rt, material_summary,
                         attribution, metric_delta, temporal_rt, generic_rt)

Order is the existing order and is load-bearing: specialist capability owners first,
change_form owners next, the temporal runtime after them (it claims on period form,
which is broader), the generic funded executor last. `level_comparison` stays absent
by the same reasoning already written into `_change_form_owner`.

Dispatch then becomes: the first runtime whose `claims(plan)` is true owns the plan,
and the population gate runs **with that runtime's own declaration**. Today the gate
runs with the funded default for anything the if-chain did not catch, which is why a
forecast plan is told the funded runtime cannot execute it.

The estate's executable set stops being a constant and becomes derivable:
`⋃ rt.EXECUTABLE_POPULATIONS for rt in GOVERNED_RUNTIMES`. One place to read, and a
test can assert it covers `POPULATION_BASES` minus whatever is deliberately
unsupported.

## 6. Change 2 — the forecast connectivity module

`mi_agent/plan_forecast_runtime.py`, built exactly as `plan_attribution` was: the
missing arrow and nothing else.

    CAPABILITY             = "forecast"
    EXECUTABLE_POPULATIONS = frozenset({"forecast"})
    EXECUTION_POPULATION   = "forecast"
    POPULATION_INPUTS      = {"funded": …, "pipeline": …}

It dispatches on the plan's operation to owners that already exist, mirroring how
`plan_pipeline_runtime` handles its four:

    forecast_projection  -> the composed funded + weighted-pipeline owner
                            (legacy `analytical_composition`)
    forecast_milestone   -> the run-rate extrapolation owner
                            (legacy `forecast_extrapolation`)
    point_in_time        -> the run-rate owner's current-rate answer
    series               -> the extrapolation curve

**Two mapping questions this design raises and does not decide.** `scenario`
("what happens if run rate falls 25%") and `cohort_conversion` (9.4% cumulative
conversion) both answered well on legacy, and neither is obviously a `forecast`
operation — conversion looks like `pipeline_stage_movement`, and scenario may warrant
its own operation rather than being folded into `forecast_projection`. Settle the
mapping before writing the module; guessing it here would be the operation-vocabulary
mistake this programme has already made once.

## 7. Change 3 — widen the temporal runtime to the pipeline population

`SLICE_2_POPULATION_BASES = frozenset({"funded"})` with the comment *"slice 2 SELECTS
its own frames, so the population base becomes this module's to honour"* — which is
correct, and is precisely why it can widen: the module owns frame selection, so it
can select pipeline extracts as well as funded snapshots.

    EXECUTABLE_POPULATIONS = frozenset({"funded", "pipeline"})
    EXECUTION_POPULATION   -> resolved per request from the frames actually loaded

That last line matters: with two possible populations the declaration can no longer
be a module constant, and `POPULATION_BASE_MISMATCH` becomes a live guard instead of
a dead one. The five `pipeline_evolution` fallbacks routed to `temporal_compare`,
`evolution`, `evolution_funnel` and `evolution_pipeline_stage` are what this change
addresses.

## 8. The negative contract — what must not change

Stated as hard constraints because each is a way this design could go wrong:

1. **Widening the perimeter must not weaken a single refusal.** A plan whose base no
   registered runtime executes still refuses `POPULATION_NOT_EXECUTABLE`, and the
   refusal still says which populations *are* executable.
2. **No answer without a proven population.** `EXECUTED_POPULATION_UNPROVEN` stays a
   refusal. A new runtime that forgets to declare `EXECUTION_POPULATION` must fail
   closed, never default to the plan's request.
3. **The gate keeps its position.** The canary's comment states the reason: a refusal
   above execution means *zero rows were touched*, so a pipeline question cannot
   produce a funded number even transiently, in the evidence sink or anywhere else.
   Dispatch-then-gate preserves this; gate-then-dispatch would not.
4. **No question text after compilation.** Dispatch reads `plan.capability` and
   `plan.population.base`. No runtime may consult wording to decide it owns a plan.
5. **No duplicated arithmetic.** Every new module is an arrow to an existing owner.
   If a forecast figure is computed in `plan_forecast_runtime`, the design has failed.
6. **No silent narrowing or widening.** A plan asking for a filtered or grouped
   population must not be answered over a broader one — the failure the run shows
   seven times on legacy and twice on the governed path.

## 9. Reason-code contract

No new reason codes. The three existing population codes carry the whole design, and
each keeps its exact current meaning:

    POPULATION_NOT_EXECUTABLE     no registered runtime executes this base
    POPULATION_BASE_MISMATCH      a runtime loaded a different population
    EXECUTED_POPULATION_UNPROVEN  a runtime declared nothing

A derived population adds two, both from D1:

    POPULATION_INPUT_UNRESOLVED   a required input (§4.1) has no governed snapshot.
                                  A refusal, not a caveat — a forecast with no
                                  pipeline extract is not a forecast with a footnote.
    POPULATION_VINTAGE_SKEW       the inputs are further apart than the portfolio's
                                  `max_snapshot_gap_days`. A refusal, and it names
                                  both dates and the ceiling, so the operator can
                                  see whether the fix is a fresher extract or a
                                  wider configured ceiling.

Within the ceiling there is no reason code, because there is no refusal — but the
caveat is not optional either. Both input vintages appear on the answer, in the same
governed fields that already carry `sourceNotes` and `warnings`. An answer that
composes two datasets and states one date has not satisfied D1.

## 10. Acceptance — measured against this same bank, re-run once

The design is accepted only on evidence, and the bank is the instrument.

    governed serving overall          41/135 (30%)  ->  target >= 85/135 (63%)
    forecast served by governed plan   0/48         ->  target >= 30/48
    pipeline_evolution                 4/15         ->  target >= 10/15
    POPULATION_NOT_EXECUTABLE         26            ->  target <= 4
                                                        (whole_book only)

    HARD GATES, any failure blocks
      WRONG answers                    0            ->  must remain 0
      silent substitutions             9            ->  must fall, never rise
      refusals losing their reason     0            ->  must remain 0
      funded categories               35/45 NEW     ->  must not regress
      answers with an unproven population           ->  must be 0
      composed answers not stating both vintages    ->  must be 0   (D1)

The last line is the one that matters most: widening the perimeter without the proof
obligation would convert honest refusals into unprovable answers, which is worse than
the state we are in. If governed serving rises and unproven populations rise with it,
the change is a regression however good the headline looks.

## 11. Out of scope, deliberately

- **`whole_book`.** One question ([Q25C]) asks for it, it needs the limit schedule,
  and Concentration Limits are not established by the client. It stays unexecutable
  and keeps refusing, and §8.1 is what makes that safe.
- **Output composition.** The governed pipeline route's value-free, measure-blind
  narration is real and is the P1 in the review. Bringing forecast inside the
  perimeter without fixing composition would produce more governed answers that do
  not state their numbers. **Sequence P1 immediately after this, not later.**
- **Interpretation.** The unmatched-wording defect is P2 and independent.
- **Latency.** 38s median is not addressed here and is not made worse by it.

## 12. Risks

    Widening admits a plan the owner cannot actually serve, converting a clean
    legacy answer into a governed refusal. Mitigated by the registry order and by
    §10's hard gate that funded categories must not regress — but this is the
    likeliest way the bank's answered count falls while its governed count rises.
    Watch both numbers, not one.

    The forecast operation mapping (§6) is guessed rather than settled, producing a
    second CHANGE_FORM_OPERATION_VARIANTS-shaped dispute. Mitigated by settling it
    before the module is written.

    POPULATION_INPUTS (§4.1) is the only new obligation, and a new obligation on a
    path that currently has none will surface skew that was previously invisible —
    the 24-day funded/pipeline gap will start refusing or caveating answers that
    used to publish silently. That is the design working, and it should be expected
    rather than treated as a regression.
