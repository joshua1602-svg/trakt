# P0 target state — the governed population perimeter

    STATUS    in build. Change 1 built (ed0f6a9e); Change 2a built (§14.3);
              Change 2b (scenario), Change 3 and P1 not yet started.
    BASIS     production question-bank run of 135 questions (review at 036778a1)
              product code at 5c436961ebe279bc0820ab006867b9f8a869bde2
    SCOPE     the funded / pipeline / forecast population bases.
              `whole_book` is explicitly out of scope — see §11.

## 0. Decisions taken

    D1  VINTAGE SKEW POLICY                                        SETTLED 2026-09-28
        Caveat, with a 45-day ceiling, the same ceiling the funded book uses.
        Beyond the ceiling: refuse. Within it: answer, with both vintages stated
        on the face of the answer. Owner decision; recorded, not inferred.

    D2a COHORT CONVERSION OWNERSHIP                   SETTLED 2026-09-28, see §6.1
        Owner's reading: it belongs in stage movement, but where it pertains to
        COMPLETIONS it carries a forward-looking funded angle. Resolution proposed:
        `pipeline_stage_movement` owns the CALCULATION; `forecast` CONSUMES it as a
        declared input and never recomputes it. Confirmed by the owner.

    D2b SCENARIO OPERATION MAPPING                       SETTLED 2026-09-28, §6.2
        Its own operation, because it takes an assumption. Owner decision.
        Consequence: a new typed `assumption` intent slot — see §6.2 for why it
        cannot share `target`.

    D5  ASSUMPTION ON THE ANSWER                        SETTLED 2026-09-28, §6.2
        A scenario states its assumption in the sentence, alongside the measure
        and as-at that D4 already makes mandatory. Extension of D4, confirmed by
        the owner.

    D4  ANSWER PROVENANCE POLICY                         SETTLED 2026-09-28, see §13
        Measure name and as-at ALWAYS on the answer itself. Everything else
        published in the envelope and disclosed by the surface on demand.
        Owner decision; recorded, not inferred.

    D6  WHICH OWNER DEFINES "FORECAST FUNDED BALANCE"         SETTLED 2026-09-28
        The analytical composer — `forecast_bridge.compute_forecast_bridge` over
        the LATEST weekly extract, the figure the React Forecast tab shows
        (£94.1m on the production book). Owner decision.

        Where the other figure comes from, corrected: the £96.2m is the
        scale-up forecast's "current weighted pipeline forecast" (Model C),
        read from `evolution.forecast_evolution`. Same formula, same pipeline
        preparation — it is NOT double counting, as first suspected. It pairs
        the August funded book with the LAST AUGUST extract (the calendar-month
        join of §14.2) instead of the latest extract (24 Sep): £9.1m of
        expected completions instead of £6.9m. The same series feeds
        `/mi/evolution/forecast`, so the dashboard's forecast-over-time chart
        most likely ends at ~£96.2m beside the tab's £94.1m — to be checked on
        the deployed surface; not changed by P0.

    D7  WHICH WEEKLY EXTRACT A NAMED MONTH MEANS              SETTLED 2026-09-28
        The last weekly report in the named month. Owner decision. Applied
        year-aware: a bare month the catalogue holds in two different years
        still clarifies, exactly as it does for funded snapshots.

    D8  WHICH PIPELINE "THE PIPELINE" MEANS                  SETTLED 2026-09-29
        The live cases — KFI, Application, Offer — by default, and the dashboard
        and the query agent use the same pipeline. Owner decision. The 2026-09-29
        production run found the agent summing the whole weekly extract
        (£1.18bn, completed and withdrawn included) while the Pipeline tab showed
        the live pipeline (#505). Both answer paths now call the tab's own
        functions (`pipeline_contract.live_pipeline_scope`, `open_totals`,
        `excluded_from_open`) and state the exclusion in the tab's words; a
        question that names a stage itself is answered over the whole extract.
        Pinned by `tests/test_the_agent_and_the_dashboard_share_one_pipeline.py`.

    D9  WHAT "SCALE" MEANS                                   SETTLED 2026-09-29
        Specific to the portfolio / SPV, never a figure the model invents. A
        client whose total assets under management exceed £200MM is AT SCALE,
        and the answer says so. For a new SPV before securitisation, scale is
        £100MM (at £80MM, scale is £100MM, £20MM away). A question naming
        "scale" or "securitisation scale" resolves to the portfolio's configured
        threshold; with none configured the agent asks. Owner decision.
        Assets under management = the funded balance across all the client's
        portfolios (owner, 2026-09-29). A portfolio's stage — pre-securitisation
        SPV or established — is recorded in the client configuration as
        `portfolio.stage` (owner, 2026-09-29). ERE is a pre-securitisation SPV
        (owner, 2026-09-29).
        BUILT: `config/system/scale_policy.yaml` holds the two thresholds;
        `mi_agent_api/scale_policy.py` resolves them from the stage; the model
        names the word `scale` as a milestone target (vocabulary 2.5.0, the
        figure never shown to it); the compiler binds it as a governed named
        threshold; the forecast runtime resolves it after interpretation and
        refuses SCALE_NOT_CONFIGURED when no stage is recorded; the milestone
        rule publishes the gap still to go. The live service reads the
        ACTIVATED client configuration, so the stage reaches production through
        onboarding's standing field, not the repository file.
        DEPRIORITISED (owner, 2026-09-29): the production stage is not being
        set now; "scale" is not a priority question and remains an accepted
        gap — a scale question refuses SCALE_NOT_CONFIGURED until it is.

    D10 WEIGHTED-AVERAGE VALUATION                           SETTLED 2026-09-29
        Answered where the book carries valuations: it is the dashboard's
        "Weighted avg property value" tile and the LTV denominator. Defined as
        the tile defines it — balance-weighted current valuation. Registry entry
        `current_valuation_amount` permits `weighted_avg`, weight
        `current_outstanding_balance`; vocabulary 2.4.0. Pinned by
        `tests/test_weighted_average_valuation_is_the_dashboards.py`.

    D11 WHAT "OVERDUE" MEANS                                  SETTLED 2026-09-29
        Two different questions share the word. About the PIPELINE, it is the
        Pipeline tab's own definition: live cases whose expected completion
        month is before the extract's month (`overdueExpectedCompletion*`).
        About LOANS, it is arrears — answered only where the tape carries the
        arrears fields the registry already defines, refused honestly where it
        does not. When the question does not say which, the agent asks.
        Owner agreed.

    D15 THE PIPELINE'S "PREVIOUS"                             SETTLED 2026-09-30
        "Week on week is difficult because the reporting of the pipeline is
        adhoc — so it should strictly be between the two most recent pipeline
        snapshots." Previous, prior, last week and week on week of the
        pipeline are the snapshot before the latest, whatever the gap; the
        answer names both dates and the gap. §26.1.

    D16 "EXPECTED TO COMPLETE IN A MONTH"                     SETTLED 2026-09-30
        The pipeline amount of the cases due then, at face value, with the
        weighted figure alongside. §26.2.

    D17 THE EXPECTED COMPLETION DATE                          SETTLED 2026-09-30
        Leaves out lapsed cases — the forecast's own rule: a case past its
        stage's validity window carries no weight and is not dated; the
        answer says how many are lapsed. §26.3.

    D18 NO LEGACY FALLBACK                                    SETTLED 2026-09-30
        "Do not use the old system." For a principal the governed path
        serves, the answer is the governed answer or the governed decline —
        what the question was understood as and why it is not answered, with
        no figure — and the legacy path is never asked. §27.1.

    D19 ONE ACTIVATED CLIENT                                  SETTLED 2026-09-30
        "There should only be one single activated client in trakt — it's
        already plugged into the mi dashboard. Any clients that are not
        active are just test / dummy runs." The tenant a single-tenant MI
        deployment serves reads the estate's single activated OCC
        configuration; zero or several activated clients read nothing.
        §27.2.

    D12 PIPELINE "BY REGION"                                  SETTLED 2026-09-29
        The client's reporting regions, on the Pipeline tab and in an agent
        answer alike. The pipeline is harmonised with the funded book's region
        engine; a region with no governed mapping is disclosed, not placed.
        Built — §15.2.

    D3  SEQUENCING                                     RECOMMENDATION HARDENED, §11
        P1 is now a DEPENDENCY of D1, not a preference. No governed answer states
        any vintage today, so "state both vintages" cannot be satisfied until the
        governed path has an answer composer. P0 and P1 are one piece of work.

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

### 6.1 Cohort conversion — D2a

The owner's reading is that conversion is a stage-movement measurement, and that its
appearance in a forecast is a different thing from its calculation. That resolves
cleanly against the negative contract already in §8.5 (*no duplicated arithmetic*):

    pipeline_stage_movement   OWNS the calculation. Conversion is a measured
                              historical fact about stage transitions — 9.4% of the
                              original KFI cohort has funded to date. Nothing
                              forward-looking about the measurement itself.

    forecast                  CONSUMES it, as a declared input alongside funded and
                              pipeline (§4.1), and never recomputes it. The
                              forward-looking angle is what the FORECAST does with a
                              historical rate, not a second definition of the rate.

The practical test that this is right: the live answer already says *"This is the
single definition of conversion used on the KPI card, the funnel and the forecast."*
One definition, three consumers. Two owners would break that sentence.

So `POPULATION_INPUTS` for forecast gains a third entry, and a forecast answer citing
a conversion rate attributes it to the stage-movement owner in the receipt rather
than claiming it.

### 6.2 Scenario — D2b

The owner's decision is that scenario is its own operation because it takes an
assumption. That is correct, and it has one structural consequence the rest of this
section establishes: **the intent needs somewhere to put the assumption.**

**What exists today.**

    OPERATIONS                           no `scenario`
    CAPABILITY_OPERATIONS["forecast"]    forecast_milestone, forecast_projection,
                                         point_in_time, series
    intent slots                         no slot can carry an assumption

**Why it cannot share `target`.** `target` is the nearest slot, and its own schema
description rules it out: *"A threshold the question names as a GOAL … It is NOT a
filter — it does not narrow the population."* An assumption is neither a goal nor a
filter; it perturbs a driver. More decisively, the deterministic engine that owns the
arithmetic already treats them as two independent inputs:

    mi_agent_api/scenario.py:41
      apply_scenario(*, current_balance, base_monthly_run_rate, reporting_period,
                     run_rate_multiplier=1.0, target_value=None, …)

and the legacy router calls it with both at once — *"perturb the completion run-rate
… and re-solve the milestone date to a target."* A single question can carry both:
*"if run rate falls 25%, when do we reach £100m?"* One slot cannot hold two inputs the
engine keeps separate.

**What the legacy path does instead — and why the governed path must not.** The
assumption is extracted by regex over the raw question, after the routing decision:

    mi_agent_api/chat_routing.py:2263  _scenario_multiplier(question)
        re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent|…)", q)
        down = any(w in q for w in ("decreas", "fall", "fell", "drop", …))

That is exactly the post-compilation wording read §8.4 forbids. Note what it is NOT:
the engine is not the problem. `scenario.py` describes itself as *"PURE and
side-effect free … no NL parsing … takes a small set of typed overrides."* It was
built to be called with a typed assumption and has been fed by a regex. **The
governed path replaces the extraction, not the engine.**

**The slot.**

    assumption: {
      lever:   a governed driver concept          e.g. completion_run_rate
      change:  "relative_pct" | "multiplier"
      value:   signed number                      e.g. -25   or   0.5
    }

Direction lives in the **sign of one number**, not in a word. Legacy holds direction
in one place (the verb list), magnitude in another (the regex) and narrates it from a
third — a hardcoded template at `chat_routing.py:2370` that says *"lifts"* whatever the
sign. That is the production answer [120]: *"A -25% completion run-rate LIFTS the
monthly run-rate from £1.2m to £925k."* The arithmetic was right. The verb came from
a place that did not know the direction. One signed value leaves nowhere for that to
recur, and D4's composer narrates from it.

**The lever set is what the engine owns, and nothing else.** Today that is one lever:
the completion run-rate, with conversion as the engine's own declared proxy (*"a
conversion change maps proportionally to the completion run-rate"*). So:

    lever = completion_run_rate    ANSWER
    lever = conversion             ANSWER, via the engine's stated proxy, and the
                                   proxy assumption is stated on the answer
    any other lever                REFUSE — the engine does not own it

Legacy's guard is lexical — `_SCENARIO_LEVERS = ("conversion", "convert", "run rate",
"run-rate", "completion")` matched as substrings of the whole question. So *"what if
completion TIMES fall 10%"* matches `completion` + `fall` + `10%` and would be applied
to run-rate VOLUME, a different quantity. A governed lever concept closes that.

**A scoped assumption refuses.** *"If Direct-book conversion falls 5%"* names a
population the engine cannot honour — it perturbs the whole run-rate. Applying it
book-wide is the silent widening §8.6 forbids, so it refuses and says why.

**D5, confirmed.** A scenario answer states its assumption in the sentence, with the
measure and as-at that D4 already requires: *"Funded balance reaches £100m around
2028-03 under a −25% completion run-rate, as at 31 Aug 2026."* The reasoning: a
scenario answer without its assumption is unfalsifiable — "£100m by 2028-03" is a
different claim from the base forecast and reads identically without it. Settled as an
extension of D4 by the owner.

**Scope of change for D2b:** `scenario` added to `OPERATIONS` and to
`CAPABILITY_OPERATIONS["forecast"]`; the `assumption` slot added to the intent schema
and parser with a bounded lever vocabulary; `plan_forecast_runtime` maps
`operation = scenario` to `scenario.apply_scenario` unchanged. No arithmetic moves.

**Both mapping questions are now resolved.** This paragraph originally left
`scenario` and `cohort_conversion` open, warning that guessing them would repeat the
operation-vocabulary mistake this programme has already made once. Neither was
guessed: cohort conversion is D2a (§6.1, proposed) and scenario is D2b (§6.2,
settled). Both are confirmed; nothing in this section blocks writing
`plan_forecast_runtime`.

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
- **Output composition — OUT OF SCOPE BUT A HARD DEPENDENCY, not a preference.**
  Re-measured on the production bank after the review:

        answers carrying a "Calculated:" provenance line
          governed (NEW)   0 of 41
          legacy          54 of 94
        answers stating an as-at date anywhere in the answer
          governed (NEW)   0 of 41
          legacy          15 of 94

  The governed path — the one built to be provable — publishes LESS provenance to
  the reader than the legacy path it replaces. The proof exists in the receipt
  (`period_from`, `period_to`, `snapshot_references`, `calculation_owner`) and never
  reaches the sentence. That is why **D1 cannot be implemented without P1**: "state
  both input vintages on the face of the answer" is unsatisfiable on a path where no
  answer states any date. P0 and P1 ship together.
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


## 13. The answer-composition contract — D4

P1's build is out of scope here, but D1 needs a defined place for a vintage to land,
so the contract it lands in is stated.

**Two things are never optional, on every governed answer, of every shape:**

    the MEASURE, by its governed display name — "Total Balance", not `balance`,
      and never a raw canonical field. The production bank shows why: three
      different pipeline questions returned one identical sentence because the
      measure was absent from it.
    the AS-AT BASIS — see the four shapes below.

**Everything else is published, not omitted.** `on demand` means disclosed by the
surface, never withheld from the envelope. The fields already exist and already carry
this: `sourceNotes` today reads `Governed snapshots · 2025-11-30 → 2026-06-30`,
`warnings` carries the period-length caveat, `governance.snapshot` carries the
content hash, row count and approval state. So D4 needs no new mechanism — the
composer puts two things in the sentence, and the surface reveals the rest from
fields that are populated today.

The distinction matters for one reason: an answer leaves the product as a screenshot,
an export or a pasted line far more often than as a live envelope. If provenance is
only in a panel nobody expands, it is not provenance. Hence measure and as-at travel
**in the sentence**.

### The as-at basis has four shapes, and the composer is written for all four

Written out because a composer built for the point case and patched afterwards is how
the current stub happened.

    POINT        one snapshot            "as at 1 September 2026"
    PAIR         a movement or compare   "1 August 2026 vs 1 September 2026"
    SPAN         a series or evolution   "across 90 weekly extracts to 24 Sep 2026"
    COMPOSED     a derived population    BOTH input vintages, per D1 —
                                         "funded as at 31 Aug 2026, pipeline as at
                                          24 Sep 2026"

`COMPOSED` is where D1 and D4 meet: D1 requires both vintages on the face of the
answer, and this is the shape that carries them. Neither decision is implementable
without the other, which is §11's dependency restated from the other end.

### Grouped and charted answers are not exempt

"Here is the bar for your query, covering 6 groups" satisfies neither rule. A grouped
answer names its measure, its grouping and its as-at like any other:
*"Total Balance by Pipeline Stage — 6 stages, as at 1 September 2026."* The chart is
the artifact; the sentence is still the answer.

### Acceptance

Added to §10's hard gates, and measured on the same bank:

    governed answers naming their measure          0 of 41  ->  41 of 41
    governed answers stating an as-at basis        0 of 41  ->  41 of 41
    governed answers leaking a raw field name      3 of 41  ->  0
    distinct questions returning identical text    3        ->  0


## 14. Implementation notes — what building it changed in this design

### 14.1 Change 1's premise was partly overstated, and is corrected here

§3 called `{"funded"}` "restated in five places" with one of them "the default
every other caller silently inherits". Reading the code to build it showed a more
precise picture, and the implementation follows the precise one:

- **The four per-module sets were already locally owned.** They were four local
  declarations holding the same value under three different names — not four
  copies of a shared constant. What was missing was a *uniform name* a registry
  could read, and any runtime but pipeline saying which population it LOADS.
- **The adapter's default was correct for its one production use.** The funded
  gate guards material summary, attribution, metric delta, the temporal runtime
  and the generic executor — all funded. So `{"funded"}` was the right set; the
  defect was that the gate did not NAME whose declaration it checked.
- **The real hazard was a comment, and it was worse than a default.** The
  adapter's docstring told the next owner to *"declare `pipeline` here"* — in the
  generic funded executor's own set. An owner following it would have had its
  base admitted by the funded gate into funded-only runtimes and executed over a
  funded frame. Pipeline happened to take the other route the comment offered.

So Change 1 as built: a uniform `EXECUTABLE_POPULATIONS` / `EXECUTION_POPULATION`
on every runtime; `plan_runtime_registry` partitioning runtimes into those
dispatched ABOVE the funded gate and those guarded BY it; the gate passing
`executable=runtime_registry.FUNDED_GATE_POPULATIONS` explicitly; the comment
corrected. The adapter keeps its name and value — tests rightly pin both, and
renaming it would have been churn rather than repair.

Guarded by `tests/interpretation_v2/test_plan_runtime_registry.py`, 23 tests, each
of the three hazards mutation-tested to fail: dropping `executable=` from the gate
(2 fail), widening the generic executor to `forecast` (3 fail), reintroducing an
old perimeter name (1 fails).

### 14.2 D1 exposes a real lineage defect in the forecast owner

`evolution.forecast_evolution` joins funded to pipeline **by calendar-month
string**:

    weighted_by_month[ym] = w          # "later extract overwrites -> latest wins"
    wpipe = weighted_by_month.get(ym)  # ym = the FUNDED period's year-month

The pipeline `extract_date` is available at the join and **discarded**; the output
period carries the funded `reporting_date` and no pipeline date. So:

- the composed forecast never records which pipeline extract fed it;
- a funded period can only pair with an extract from the same calendar month —
  August funded pairs with an August extract or with **nothing**, in which case
  `weighted_expected_pipeline` is None and the forecast silently equals funded;
- `build_extrapolation` then returns `reportingPeriod` (funded) with no pipeline
  vintage at all.

**D1 cannot be satisfied without the pipeline vintage, and the owner currently
throws it away.** The fix is additive lineage, not arithmetic: record the
`extract_date` that fed each period's `weighted_expected_pipeline`, and surface it
through `build_extrapolation`. No figure changes; an existing value stops being
discarded. The month-join itself — whether pairing by calendar month rather than
by nearest extract is correct — is a *calculation* question and is deliberately
NOT changed by P0. It is recorded here because D1 will make it visible, and the
first composed answer that states both dates will show whether the pairing is the
one a reader expects.

### 14.3 Change 2a as built — what it serves, and four things building it found

**Served** by `mi_agent/plan_forecast_runtime.py`, dispatched above the funded
gate from `plan_runtime_registry.POPULATION_OWNING_RUNTIMES`:

    forecast_milestone / forecast_milestone_date   the run-rate owner's
        + target forecast_funded_balance >= X      `milestone_answer`
    point_in_time / forecast_completion_rate       the owner's base run-rate
    point_in_time | forecast_projection /          the analytical composer,
        forecast_funded_balance                    `funded_balance_forecast`
                                                   -> compute_forecast_bridge
                                                   (added once D6 settled)

**Refused by name**, legacy serves: a `forecast_funded_balance` SERIES (the only
series in the estate is the month-joined one D6 did not choose); `scenario`
(Change 2b); a `pipeline` or `whole_book` base; any lens, seasoning or
named source; filters, axes, geography, comparisons; a `gt` threshold (the
owner's rule is `>=`); a run-rate over any window but the current one.

**1. Normalisation rule 6.** The production bank spelled one milestone question's
population two ways — five `forecast`, four `funded`. Under the forecast
capability `funded` (and the schema default, which is `funded`) now normalises to
`forecast`; `pipeline` deliberately does not ("of the offer pipeline, how much
converts" is the pipeline's contribution alone, not the whole forecast). Replaying
the signed-off 135: five plans move, four of them onto the exact plan a sibling
already had (Q23C and NL2A onto Q23A's, Q24B onto Q24A's), and the replay guard
pins them as an authorised migration with a proof that the base is the only slot
that moved.

**2. `POPULATION_INPUTS["pipeline"]` is required WHEN it is the completion
signal, not always.** §4.1 declared it required. The run-rate owner falls back to
funded balance growth when the pipeline has no observed completion flow — its own
documented rule, with its own sufficiency floor — and in that case the pipeline
fed nothing and has no vintage to state. So the receipt records which inputs each
answer USED: a projected milestone uses both; "already reached" reads the funded
balance alone; the run-rate uses whichever signal the owner took. D1's skew check
runs over the inputs used. The owner now says which signal it took as a code
(`completionSignalKind`), not only as prose, so this needs no sentence parsed.

**3. The run-rate's pipeline vintage is a different extract from Model C's.**
§14.2 added the extract that fed the weighted pipeline. The run-rate's pipeline
input is the funnel's trailing completion flow, which ends at the funnel's latest
extract — surfaced as `completionFlowExtractDate`. On the production book that is
the 24-day gap §4.1 described (funded 2026-08-31, pipeline 2026-09-24).

**4. The legacy run-rate answer most likely misstates its own basis.**
Production [112]: *"~£1.2m/month … based on 7 month(s) of funded growth"*. The
±25% bands (£925k / £1.5m) are what the observed-completion-flow branch produces;
the funded-growth proxy produces them only if fewer than three of its seven
observations survive step-change screening, and otherwise uses percentiles. So
the figure most likely came from the pipeline's completion flow, and the legacy
sentence hard-codes "funded growth" whichever branch ran. Not proven from the
bank text alone; the governed answer removes the question by taking the signal
from the owner's own description.

**Also found, and fixed separately (90a32d42):** `plan_serving_canary.serve`
had accepted the client's source registry since f244e211 and, from e6e16c63,
stopped forwarding it — every governed compilation ran without the registry, so
a question naming a portfolio was refused on the governed path. A structural
test now asserts every input `serve` shares with `_attempt` is passed through.

**The target is now proved on the way out.** `requested_semantics` transcribes
the plan's target and `mi_service._governed_plan_coverage` requires the executed
receipt to state the same concept, comparator and value — so an answer for a
different threshold (the £250m defect) is UNACCOUNTED even if a future runtime
reintroduced it.

Guarded by `tests/interpretation_v2/test_specialist_runtime_forecast.py` and the
rule 6 tests in `test_contract_normalisation.py`. Mutation-tested: disabling the
skew check fails 3; dropping the target proof fails 2; dispatching forecast below
the funded gate fails 4.

**D6 as built.** The balance is served through the composer's own executor with
the context the legacy analytical route builds — the funded frame from the same
`_routed_frame` resolver `mi_service` hands that route, the latest governed
weekly extract, no lens. A test calls `/mi/forecast/snapshot` (the React Forecast
tab's endpoint) for the same book and asserts the governed figure, the funded
balance and the expected completions all equal the tab's, and that the answer
states the tab's two dates.

**Broad regression for Change 2a (ad931b70).** mi_agent + mi_agent_api, 3,992
tests: 70 failed / 3,655 passed against 69 / 3,654 at Change 1. One failing ID
differed, `test_serve_threads_the_registry_to_the_compiler_seam`, and it was an
artefact of the run: `plan_serving_canary.py` was edited three minutes into it,
so `inspect.getsource(_attempt)` read shifted lines and returned the next
function. On HEAD it passes. Zero new failures.

### 14.4 Change 3 is a store and a decision, not a widened constant

§7 read Change 3 as "widen `EXECUTABLE_POPULATIONS` to include pipeline, because
the temporal runtime owns frame selection". Reading the code before building it:

- **The temporal runtime selects frames through `SnapshotStore`, and the only
  store production builds is funded.** `governed_snapshot_store` serves
  `FUNDED_ROUTE` alone and refuses any other route by design, and
  `plan_pipeline_runtime` states the rule that follows: pipeline time is weekly
  extracts under a different owner, and answering a pipeline question from the
  funded catalogue "is the single worst outcome available". Widening the
  constant without a weekly pipeline store would admit pipeline plans into a
  runtime that can only load funded frames.
- **What fails in production is pair comparison, not series.** Weekly series
  already serve through the pipeline runtime ([70]–[73] NEW). The refusals are
  [80]/[81] "October and November pipeline" (PERIOD_NOT_SUPPORTED) and [82]/[83]
  "latest vs prior", "growth from October to November"
  (POPULATION_NOT_EXECUTABLE).
- **No governed rule says which weekly extract a named month is.** The temporal
  runtime matches a month EXACTLY and refuses when more than one snapshot falls
  in it (`PERIOD_LABEL_AMBIGUOUS`) — right for monthly funded snapshots, and it
  would refuse every month against weekly extracts, which carry four or five.
  The legacy owner (`temporal_compare._match_period`) takes the latest extract
  whose month NUMBER matches, in any year — so "October" can silently mean an
  October from a different year. Neither can be reused as it stands.

**D7 — WHICH WEEKLY EXTRACT A NAMED MONTH MEANS (settled: the last weekly report
in the month).** As recommended: the last weekly extract dated within the named
month, year-aware (a bare month that
occurs in two years stays ambiguous and clarifies, as it does for funded), with
the extract date stated on the answer (e.g. "October 2025, weekly extract of <date>").
It is the month-end position, it is what legacy intends minus the year defect,
and stating the date makes the choice auditable. Relative forms need no
decision: "latest vs prior" on a weekly grain is the two most recent extracts.

So Change 3 as it will be built: a governed WEEKLY pipeline store implementing
the same `SnapshotStore` protocol (headers from the weekly extract inventory,
`cadence=weekly`, `load_loans` returning the prepared pipeline frame), handed to
the temporal runtime for pipeline plans; the D7 rule applied only on the weekly
route; `EXECUTION_POPULATION` resolved per request from the store actually used;
the pipeline amount bound to `pipeline_prep.PIPELINE_AMOUNT_FIELD` exactly as
the pipeline runtime binds it. Relative pairs can be built before D7 is settled;
named months wait for it.

`pipeline_stage_movement` (27 certification cases refused
POPULATION_NOT_EXECUTABLE, [84] in production) is a separate population-owning
runtime over the stage-movement owner, not part of Change 3, and is recorded
here so it is not mistaken for one.

**Step 1 as built (owner go-ahead 2026-09-28).** The pipeline runtime — not the
temporal runtime — now serves a pipeline plan whose period is a named month
(`explicit_period`) or "latest against previous" (`relative_pair`, weekly or
monthly grain). It asks the weekly owner (`evolution.pipeline_evolution`) for
the history, chooses the extracts — D7 for a month, extract order for weeks —
and returns those extracts' own figures, total or by stage. It computes no
change between them: a movement is a different operation. Each chosen extract
and the rule that chose it are on the receipt (`period_resolution`), in the
sentence and in the source notes. The month-label reader moved to a neutral
`mi_agent/period_labels.py` so this runtime reads "October 2025" by the funded
runtime's rule without importing the module that owns the funded catalogue.
Guarded by `tests/interpretation_v2/test_specialist_runtime_pipeline_dated.py`.

### 14.5 Stage movement as built — the largest single POPULATION_NOT_EXECUTABLE family

Not in the original P0 scope; added with the owner's go-ahead because it moves
more questions than Change 3. `mi_agent/plan_stage_movement_runtime.py`, a
population-owning runtime dispatched above the funded gate, serves
`pipeline_stage_movement` plans — transition, arrivals, stayers, departures (by
destination) and reconciliation — over the latest governed pair of weekly
extracts. It translates the plan's stage filters into the owner's own reading
(`stage_movement_query.StageMovement`), takes every figure from
`movement_detail.resolve_stage_transition_detail`, and every word from the
owner's pure `compose`. It reads no question and does no arithmetic.

Proof, against the legacy route as oracle on its own fixture: for 26 of the 27
recorded certification plans the governed reading equals the reading legacy
built from the sentence, and the governed answer equals the legacy answer word
for word. The 27th (SM03C, "arrivals" that also name an origin stage) is refused
by name — a new arrival has no origin, and legacy's answer to it was adjudicated
PARTIALLY_CORRECT.

One shared change: `requested_semantics` now transcribes a capability-owned
filter or axis by its concept when it has no canonical field (`origin_stage`,
`destination_stage`), so the coverage owner proves the stage axis by name rather
than matching an empty field.

### 14.6 The production run's own plans, replayed — and what that tightened

The canary had recorded every plan of the owner's 2026-09-28 production run in
its evidence sink. `qb-plan-readback.yml` (read only) pulled all 135 back; each
record's serving decision matches `qb_full.txt` exactly, and today's compiler
reproduces 132 of the 135 plans byte for byte — the other three differ only by
rule 6's base rewrite. So the perimeter can be measured against the real plans
offline, and `tests/interpretation_v2/test_production_bank_perimeter.py` does.

**What it showed.** The first cut of the forecast perimeter would have admitted
19 questions that legacy served. Eight of those plans do not mean what their
shape says: the model put "expected completions by month", "the extrapolation
curve", "the base scenario", "how much of the forecast comes from the funded
book", "the forecast's method" and "the KFI-to-completion conversion rate" into
the same two shapes as the balance and the run-rate. The runtime answers the
shape, so those eight would have been served a different figure — WRONG answers
on the governed path, which §10 makes a hard gate.

**What changed.**

- A plan with a period grain is refused: "by month" asks for a figure per
  period, and every forecast figure served is a point. Checked first, because it
  must still refuse after the hold below lifts.
- `forecast_projection / forecast_funded_balance` and `point_in_time /
  forecast_completion_rate` are HELD (`AMBIGUOUS_READING`). The owner computes
  them; a plan cannot yet be trusted to mean them, because the vocabulary names
  these measures and defines none of them. Three correct readings are held with
  the misreads ([87], [112], [113]) and legacy answers them correctly meanwhile.

**Admitted, pinned by bank number:** [80], [81] (pipeline at named months, D7),
[85] (forecast funded balance, D6), [107]–[111] (milestones). Eight questions,
each a reading whose plan cannot mean anything else.

**What lifts the hold** is a P2 item and model-facing: give the forecast
measures definitions the model can separate them by (the balance is one point
from the latest extract; the run-rate is £ per month, not a conversion rate; a
curve is a series), then re-run and re-read. The replay test then says, by bank
number, which readings separated.

**Also read from the plans, for Change 3's remaining steps:** [82] is a
`portfolio_summary` level comparison over the pipeline; [83] a
`generic_analysis` metric delta over the pipeline for named months; [84] a
stage movement transition by origin and destination (the full matrix); [74],
[75], [79] never became plans (AMBIGUOUS_PERIOD: "over time" with no grain).

## 15. Direction of travel — owner-approved 2026-09-29

The 2026-09-29 production bank (135 questions, build ce5d1276, vocabulary 2.3.0)
was the first measured against the dashboard's NUMBERS rather than only against
the reading. Of 135 answers: 45 right and clear, 23 right but weakly presented,
13 wrong and already fixed (D8, the live pipeline), 9 wrong and not yet fixed,
5 that did not answer, 5 honest refusals, and 35 refusals of questions the
dashboard already answers. The new engine gave 51 answers (41 on 2026-09-28);
the old engine gave 84, including 6 of the 9 open wrong answers and all 5
non-answers. The model's own clarification notes name the cause of most of the
rest: "the registry carries no governed concept for …" — weighted expected
pipeline, overdue, scenarios, the forecast's parts, the projection curve — all
of which the dashboard already computes.

### 15.1 The direction — market-standard practice only

    1  ONE CATALOGUE for the dashboard and the agent (a semantic layer). Every
       figure defined once — name, meaning, one calculation — and read by both.
       The agent can only ask for catalogue items. Nothing is invented: each
       new item is wired to the calculation the dashboard already uses. D8 and
       D10 are its first two instances.
    2  ONE ENGINE. The new engine answers or explains why not; the old engine
       is retired area by area, running only as a comparison until switched
       off. An area switches when the new engine matches the old engine's
       right answers with zero wrong numbers.
    3  ASK WHEN GENUINELY UNCLEAR. Clarifications the model writes are served
       to the reader with options, not discarded.
    4  ONE ANSWER STANDARD (P1, §13): the figure, what it is, as-at date(s),
       what is included and excluded, how it was calculated; breakdowns name
       their leaders.
    5  GOLDEN ANSWERS WITH NUMBERS, checked on every release, reconciled to the
       dashboard; verified readings reused at runtime.
    6  SPEED: one interpretation per question, history loaded at start, caching.

    GUARDRAILS. No per-question patches or keyword rules; the model's refusal
    discipline is not loosened; the model never calculates. Every release must
    show zero numbers that differ from the dashboard, no area losing right
    answers, and a reason on every refusal — measured by the bank before
    anything is switched on.

    PRIORITY (owner, 2026-09-29): narrow the gap as far as possible,
    accepting some gaps. The backlog below is built in order of how many of
    the 135 bank questions each item closes, largest first.

### 15.2 The catalogue backlog, from the run — each wired to an existing owner

    concept (what the model asked for)          the dashboard's existing owner
    ------------------------------------------  ----------------------------------------------
    expected completion month; overdue /        DONE — batch 1, vocabulary 2.6.0:
      this month / next month (pipeline, D11)     _expected_completion_breakdown / _summary
    weighted expected pipeline, total and by    DONE — batch 1: open_totals()["weighted"];
      stage / broker / product / LTV band          _dimension_breakdown weightedExpectedFundedAmount
    product and LTV band on the pipeline        DONE — batch 1: productBreakdownFull, ltvBreakdown
    region on the pipeline (D12)                DONE — reporting taxonomy on the tab and agent
    what carries no forecast weight, and why    pipeline_prep.completion_probability_summary
      (withdrawn, lapsed, missing probability)     by_source
    the forecast's parts: funded, pipeline      forecast_bridge.compute_forecast_bridge
                                                   fundedBalance / weightedExpectedFundedAmount
    base / downside / upside scenario           forecast_extrapolation.run_rate_model scenarios
    the month-by-month projection curve         forecast_extrapolation._project_series
    conversion basis; which stages use          pipeline_history.historical_model_evidence
      historical or fallback rates
    KFI-to-completion conversion                cohort conversion (stage movement, D2a)
    scale threshold (D9)                        client / portfolio configuration (new field)
    weighted-average valuation (D10)            DONE — registry entry, vocabulary 2.4.0

    Batch 1 (2026-09-29) reads every figure off the Pipeline tab's own
    functions for the same live rows; nothing is grouped or summed in the
    agent. Pinned by `tests/interpretation_v2/test_pipeline_catalogue_batch1.py`
    against the tab's snapshot. Bank questions it gives a home include
    pipeline_003, _007, _009, _010, _012, _013, _015, _016, _018–_021,
    pipeline_strat_001–_003 and pipeline_evolution_007/_008 when read as
    expected-completion months — whether the model reads each one this way is
    what the spot check measures.

    REGION ON THE PIPELINE (D12, owner 2026-09-29: "the Pipeline tab should
    also group by the reporting regions" — YES). "By region" governs to the
    client's reporting taxonomy (`canonical_region_reporting`); the tab used
    to group the extract's raw spelling. Now the pipeline's preparation stamps
    the reporting region with the funded book's own engine
    (`engine.region_taxonomy`, `pipeline_prep._apply_region_taxonomy`), the
    tab's region chart groups by it (the raw spelling kept in
    `regionSourceBreakdownFull` for audit), and the agent reads that chart. A
    case whose region has no governed mapping is placed in no region and
    disclosed on the chart and in the answer (`regionBasis`). Without the
    reporting column the agent refuses rather than serve the raw spelling; a
    region FILTER and other geography levels remain refused. Pinned by
    `tests/interpretation_v2/test_the_pipeline_region_is_the_reporting_region.py`.

    The forecast-balance hold (§14.6) is released by this backlog, not before
    it: once the curve, the scenarios and the forecast's parts each have their
    own concept, the questions that were misread into the balance have a home
    of their own. Re-run evidence (2026-09-29): lifting it today admits none of
    the 135 questions and would remove the only guard on the 2.2.0 misreads,
    because a forward-looking balance is a permitted period form.

## 16. Catalogue batch 2 — the forecast semantic model (owner direction 2026-09-29)

"Start batch 2. Adhere to the target state, scalable architecture, Cortex-based
design principles, and no local / tactical fixes."

### 16.1 What the evidence asks for

Under vocabulary 2.3.0 the model stopped MISREADING the forecast questions and
started NAMING what the catalogue lacks. Its own blocking notes on the
2026-09-29 run, verbatim in substance:

    [114] "no governed measure for such a curve"
    [116-118] "no scenario concept or dimension: there is no base/downside/upside"
    [94, 95] the forecast is "NOT decomposable into a part of itself"
    [102-104] "no concept for weighting, or an exclusion from weighting"
    [100, 101] "'historical rates' cannot be bound to any governed concept"
    [123] "the registry governs no default threshold list"
    [89] "the registry carries no forecast-owned loan-count measure"
    [127] a twelve-month horizon had nowhere to go but a label

Every one of these is a figure the Forecast tab ALREADY shows. The gap is the
catalogue, not the engine.

### 16.2 The architecture — a semantic model, one engine (Cortex's shape)

Cortex Analyst answers from a SEMANTIC MODEL: one YAML per subject declaring
each measure and dimension — its description (what the model reads) and its
expression (what executes). The model is grounded in the file; the warehouse
computes. Trakt's equivalent, for the forecast capability:

    config/mi/semantic_model/forecast.yaml     ONE file per capability
      views       the owners whose published output answers — the Forecast
                  tab (`forecast_view.compose_forecast_view`) and its scale-up
                  panel (`forecast_extrapolation.build_extrapolation`)
      measures    definition (shown to the model) + unit + view + the PATH of
                  the figure in that view's output
      dimensions  definition + governed values + how each measure's view
                  publishes the figure per value (members or rows)

    mi_agent/semantic_model.py                  load + validate + read a path
    vocabulary                                  the forecast concepts, their
                                                definitions and values come
                                                FROM the file (no second copy)
    plan_forecast_runtime                       one generic reader: scalar,
                                                one governed member, a
                                                breakdown, or a series — each a
                                                lookup in the owner's output

What this buys, and why it is the target state rather than a patch:

  - ONE DEFINITION. What the model is told a measure is, and what executes for
    it, are the same entry. They cannot drift.
  - ONE ENGINE. Every figure is read from the payload the tab renders, built by
    the same function for the same inputs. The runtime still computes nothing
    (the AST rule in `test_the_runtime_computes_no_forecast_figure` holds).
  - SCALES BY DATA. A new forecast figure is an entry in the file and a test
    pinning it to the tab — not a new branch in the runtime.
  - THE FORECAST TAB'S SNAPSHOT BECOMES A FUNCTION. It was assembled inline in
    the `/mi/forecast/snapshot` route, so nothing else could call it. It moves
    to `mi_agent_api/forecast_view.py`; the route and the agent both call it.

The pipeline runtime's batch-1 maps (`TAB_COLUMN`, `_TAB_TIMING_KEY`) are the
same idea written in code; they move to `semantic_model/pipeline.yaml` as a
follow-up, so both specialist capabilities have one form.

### 16.3 The concepts (vocabulary 2.7.0)

    measure                          the tab's figure (view: path)
    -------------------------------  --------------------------------------------
    forecast_funded_balance          forecast_view: forecastBridge.forecastFundedBalance
      by forecast_component            funded_book = fundedBalance,
                                       weighted_pipeline = weightedExpectedFundedAmount
      by canonical_region_reporting    forecastBreakdowns.byRegion  (D12 on both books)
      by ltv_bucket                    forecastBreakdowns.byLtvBucket
    forecast_loan_count              forecast_view: forecastBridge.forecastLoanCount
    weighting_excluded_amount        forecast_view: forecastBridge.excludedFromWeightingAmount
      by weighting_exclusion_reason    forecastBridge.excludedByReason (owner-classified)
    projected_funded_balance         scale_up: completionRunRateForecast.projectedBalances
      (a monthly series)               one column per forecast_scenario
    forecast_completion_rate         scale_up: completionRunRateForecast.baseMonthlyRunRate
      by forecast_scenario             scenarioMonthlyRunRate.{downside,base,upside}
    annualised_completion_run_rate   scale_up: completionRunRateForecast.annualisedRunRate
    forecast_milestone_date          scale_up: milestones (existing target path unchanged)
      by funding_threshold             the governed ladder, one row per threshold

    dimension                        values
    -------------------------------  --------------------------------------------
    forecast_component               funded_book, weighted_pipeline
    forecast_scenario                downside, base, upside
    funding_threshold                the owner's ladder (£25m ... £150m)
    weighting_exclusion_reason       completed, withdrawn, not_forecast, lapsed,
                                     missing_probability

Two intent/compiler changes, both market-standard and both general:

  - `time.periods_ahead` — the forward horizon ("next twelve months" = 12),
    the mirror of `periods_back`. The curve is SELECTED to it, never
    re-projected; a horizon beyond the owner's published one refuses.
  - A milestone grouped by `funding_threshold` needs no single target: it asks
    for the ladder, which the owner publishes.

### 16.4 What stays refused, and why

    forecast by broker / stage        the tab publishes no such breakdown
    "if the run-rate falls 25%"       D2b: its own `scenario` operation with a
                                      typed `assumption` slot (§6.2) — batch 2c
    stage probabilities, basis         batch 2b: the owner publishes stage lists,
                                      not per-stage probabilities (run-off makes
                                      them per case); owner disclosure first
    the 8-/12-week run-rate           not published (the proxy's lookbacks are
                                      monthly windows)
    KFI-to-completion conversion      D2a: stage movement owns it; the Model B
                                      projection is withdrawn by its owner
    the D6 holds (§14.6)              kept until a live run shows the readings
                                      separate onto the new concepts

### 16.5 As built (2026-09-29)

    config/mi/semantic_model/forecast.yaml   8 measures, 4 dimensions, 2 views
    mi_agent/semantic_model.py               load + validate + read a path; no
                                             arithmetic (AST-pinned)
    mi_agent_api/forecast_view.py            the Forecast tab's snapshot as one
                                             function; the route, the analytical
                                             composer and the runtime all call it
    mi_agent/plan_reading.py                 the plan-reading helpers both
                                             specialist runtimes share
    mi_agent/plan_forecast_runtime.py        two paths: the semantic model's
                                             generic reader, and the milestone
                                             for a stated threshold (its rule)
    time.periods_ahead                       intent + plan + compiler; absent,
                                             it is left out of plan identity, so
                                             every recorded plan_id still replays

Owner changes this needed, each one a disclosure the tab now shows too:

  - the weighting exclusion BY REASON (`pipeline_prep.EXCLUSION_REASONS`,
    `forecastBridge.excludedByReason`), every governed reason published, with
    zeros, and summing to the excluded total;
  - the forecast by region in the REPORTING taxonomy when both books carry it
    (D12 extended to the Forecast tab), with `regionBasis` and the amount it
    cannot place. Before, the tab added the funded book's raw region spelling
    to the pipeline's, so "YORKSHIRE" and "Yorkshire" were two bars. Where a
    book carries no harmonised region (the legacy central-tape path) the tab
    falls back as before, and the agent refuses rather than answer from it.

The balance and the run-rate moved onto the generic reader: one path for every
lookup. The balance's receipt now names the Forecast tab's own view function
and the path of the figure (`forecastBridge.forecastFundedBalance`), D6 kept.

Pinned by `tests/interpretation_v2/test_forecast_semantic_model.py`: every
declared path resolves in the owners' real output, every figure equals the
tab's or the scale-up owner's for the same book and run, the curve is selected
and never extended, and the answer's coverage accounts for the member, axis and
region asked for.

The D6 holds stay: `point_in_time/forecast_completion_rate` and
`forecast_projection/forecast_funded_balance` are released only when the spot
check shows the 2.7.0 readings have moved onto the new concepts. The scenario
run-rates are declared and read the owner's figure, and wait on that release.

## 17. Breadth — the funded book's remaining gaps (2026-09-29)

From the 2026-09-29 run, 12 funded questions fell back. Each is closed at its
owner, with the rule that closes it stated once:

    "by region" (7)          INELIGIBLE:GEOGRAPHY_REQUESTED. The compiler had
                             already resolved each to ONE field through the
                             governed geography contract ("by region" is the
                             client's reporting taxonomy); the funded adapter
                             refused any geography wholesale. It now carries the
                             binding to the executor — a grouping as an axis, a
                             restriction ("in London") as a predicate through the
                             one predicate accessor (`plan_predicates`) — and the
                             coverage owner proves the axis on every receipt
                             (`_geography_coverage`, shared with the specialist
                             runtimes). The temporal path still refuses geography
                             until it declares it carries one.
    median / largest /       INELIGIBLE:MEASURE_NOT_GENERIC. The executor has
    smallest (3)             always computed them; the adapter's statistic map
                             lacked them. The answer names them ("Maximum Balance:
                             £274K") — the label and money-format suffix lists
                             gained `_min` / `_max`.
    borrower structure (1)   EXECUTION_FAILED. A legacy concept for a band no book
                             materialises; the registry already said "prefer
                             borrower_type". Registry `superseded_by` folds a
                             legacy name into its governed successor as an alias
                             (vocabulary 2.8.0): one concept per meaning.
    occupancy type (1)       EXECUTION_FAILED on the production book, which does
                             not carry the field. A field the book lacks is now
                             recorded as the book's (`FIELD_NOT_IN_BOOK`, named by
                             the executor's own validator before executing), not
                             as a crash — and never answered from a neighbour.
    weighted-average         already closed (D10, 2.4.0).
    valuation (1)

On a production book whose preparation stamps the reporting region (the
platform path does), "by region" now answers from the governed field; on a
book that does not, it is FIELD_NOT_IN_BOOK. Serving that refusal to the reader
— instead of falling back — belongs with served clarifications (§15, next).
Pinned by `tests/interpretation_v2/test_funded_breadth.py`.

## 18. Every funded answer says what it is (D4) — brought forward (2026-09-29)

Owner instruction: "Fix the wording before proof stage otherwise we will forget
and have to go back." A funded breakdown read "Here is the bar for your query,
covering 10 groups" — no measure, no grouping, no date — so a text channel, and
the evidence a bank run is judged on, carried no answer.

One composer, both paths. `adapters._answer` is the lead sentence every funded
answer uses, legacy and governed alike; it now composes:

    a breakdown    "Balance by Region — largest: East Midlands £831K, East of
                   England £761K, Yorkshire and The Humber £752K, and 7 more
                   (10 groups)." ("highest" for an average or a ratio;
                   "Number of loans" for a count)
    a series       "Balance over 5 reporting dates, 2025-07-31 to 2025-11-30:
                   from £9.4MM to £12.1MM." — read in time order, never ranked;
                   a grouped series names its leaders at the latest date
    one figure     unchanged ("Median Balance: £162K · 36 loans.")

— from the same rows, labels and formatters the table and chart are built
from; the ordering is presentation, nothing is computed.

The governed funded answer now carries the same execution receipt the legacy
answer does (`execution_receipt.build_receipt`, one owner): "Calculated: Total
Balance · grouped by Region · 10 groups · 36 loans · as at 30 November 2025" —
built from what ran and the book's own cut-off date (`reporting_date_label`),
never from the question. Forecast and stage-movement answers already stated
their measure and dates (their own renderers, D4). Pipeline answers stated
the measure but not the extract they were read from — "The live pipeline
amount is £1.1m." — and now say "as at the weekly extract of <date>" (a grouped
series names its first and last extract).

Pinned by `tests/interpretation_v2/test_answer_wording.py`.

## 19. The combined spot check, and what it changed (2026-09-29)

Run on 438932ac (vocabulary 2.8.0), 51 questions from 15:14:42 UTC; read back
by run 36594305684. **45 of 51 were served by the governed path; on the
morning's run the same 51 had 1.** Every figure cross-checked adds up: the
forecast by region and by LTV band each total the £94.1m forecast funded
balance, and the 12-month curve plus six months at the run-rate is the
18-month curve.

The six that fell to the old path, and what each needed:

| question | reading | why it fell | change |
|---|---|---|---|
| forecast_018/019/020 — pipeline excluded from weighting (all / missing probability / withdrawn) | correct: forecast, **pipeline** base, `weighting_excluded_amount` (+ reason) | the forecast runtime executed the `forecast` population only; the old path then answered 019 with the whole live pipeline (4,835 cases, £969.8m) | §19.1 |
| forecast_runoff_002 — lapsed past its stage window | clarify: "no governed concept for a stage window" | `lapsed` was defined in five words | §19.2 |
| funded_breakdown_1d_003 — by obligor region | `geographic_region_obligor` | the production book does not carry it: FIELD_NOT_IN_BOOK, and the old path's refusal says so | none — correct |
| funded_breakdown_1d_025 — by occupancy type | `occupancy_type` | as above | none — correct |

### 19.1 Each figure declares the population it is measured over

The semantic model now says, per measure, which population the figure is
measured over: the capability's own (`population: forecast`) unless the entry
names one of its view's inputs and reads that input alone —
`weighting_excluded_amount` / `_case_count`, `population: pipeline`. The
forecast runtime reads its executable populations and each plan's execution
population from the model; the population gate and the receipt prove the
plan's base against that declaration. A plan naming another population is
refused `POPULATION_NOT_MEASURED`, never answered from it; a model declaring a
funded figure is refused at load (the runtime sits above the funded gate).

### 19.2 Definitions (vocabulary 2.9.0)

`lapsed` is defined in the pipeline's own terms — an open case in its stage
longer than that stage's validity window, measured from the book's history
("lapsed", "expired", "past its stage window") — and the exclusion measure
says it is measured over the pipeline. The bank runner's minimum vocabulary
is 2.9.0.

### 19.3 One answer standard (`mi_agent.answer_standard`)

The spot check stated the one book's funded balance as "£87.1MM" in a funded
answer and "£87.1m" in a forecast answer; a pipeline breakdown named ten
brokers where a funded one named three; answers said "month(s)", "case(s)",
"the owner's 18-month horizon" and showed the taxonomy id "(uk_itl1)"; the
pipeline and forecast sentences hard-coded "£". One module now decides those
words, presentation only:

    money()           the platform formatter (mi_agent_api.currency), chat
                      suffixes bn/m/k — tiles keep BN/MM/K — in the client's
                      reporting currency
    plural()          a count agreed with its noun
    breakdown_lead()  measure, grouping, the three leaders, how many groups
    region_note()     which regions, on which location, and what is in none

A timeline (expected completion month) and an owner-ordered axis (scenario,
threshold ladder) keep their order and name every row.

### 19.4 Regions say which location they are (D12, completed)

The harmonisation (`engine.region_taxonomy.apply`) records, per row, the
source column its raw region came from (`region_source_field`), and one
`disclosure()` serves every surface. Every region answer — funded, pipeline,
forecast — now states the basis from that record through `mi_geography`'s
basis of the column: "Regions are the client's reporting regions, by the
property's location". A funded region breakdown also discloses loans the
taxonomy cannot place, as the pipeline and forecast answers already did.

Found on the way, at its source: the pipeline's `_apply_group_aliases` copies
the property's region into the borrower column when the extract has none, and
the harmonisation read that column first, so the pipeline's regions were
recorded as the borrower's address. Its harmonisation now reads the book's
own basis first, as the funded book's does. The figures are unchanged; the
record of what they rest on is now true.

Pinned by `test_forecast_semantic_model.py`, `test_answer_standard.py`,
`test_funded_breadth.py` and `test_the_pipeline_region_is_the_reporting_region.py`.

## 20. The pass mark, conversion, pipeline change — on one semantic engine (2026-09-29)

**D13 — owner decision, 2026-09-29** ("execute 1-3 on target state cortex
architecture"). The proof stage's pass mark is the question bank in three
buckets (`due_diligence/evidence/qb_plan_readback/qb_question_buckets.json`):
every MUST ANSWER question answered correctly and no question answered
wrongly — not 135/135. Decided on the three points put to the owner:

1. the pipeline conversion rate and the offer-to-completion pull-through
   ([49], [134]) are must answer;
2. week-on-week and month-on-month pipeline change ([82], [83]) are must
   answer;
3. the strat split stands as drafted (balance and risk cuts must, secondary
   measures of each cut nice to have).

### 20.1 One semantic engine

`mi_agent.semantic_engine` is the ONE reader every capability's semantic
model is served by — the Cortex shape: the model file declares what a figure
is and where its owner publishes it; the engine resolves a plan against that
declaration and reads the figure; a new figure is a new entry in a file.

    check(...)            the catalogue perimeter every declared figure shares
    serve_figure(...)     value / member / breakdown / curve, by lookup
    inputs_* / receipt    the view's dated inputs and the receipt core
    period_change(...)    THE one place a change between two governed
                          figures is computed (pinned: no other arithmetic
                          in the engine)

The forecast runtime's own copy of the reader is gone; it keeps only what is
the forecast's (its held readings, the milestone rule, the input vintage
ceiling). Two capabilities now declare their figures in files: forecast and
pipeline stage movement.

### 20.2 Conversion (D2a) as data

`config/mi/semantic_model/pipeline_stage_movement.yaml`, over the owner D2a
names — the pipeline's case history (`pipeline_history.build_historical_
completion_model`), which the Pipeline tab's conversion card and the
forecast's stage rates already read:

    cohort_conversion       the % of the original KFI cohort funded to date;
                            by destination_stage, the funnel at the latest week
    stage_pull_through      of the cases that left a stage, the share that
                            advanced (Offer → Completion is origin_stage OFFER)
    stage_completion_rate   the forecast's per-stage completion rate — [121]
                            'what completion rate is assumed from KFI to
                            completion' now has its own concept

The capability's origin/destination stage dimensions are defined in that file.
The owner's per-figure sufficiency flag and counts travel with the figure: a
rate measured on too few cases is answered as provisional, with its evidence —
never a bare 100%. Units gain `pct` and `ratio`.

### 20.3 Pipeline change between two dated extracts

The pipeline capability takes `movement` and `compare`, and
`CAPABILITY_CHANGE_FORMS` declares that it implements `metric_delta` and
`level_comparison` over its own measures, so normalisation keeps the
measure's owner (a funded metric delta still goes to `period_movement`). The
two figures are read exactly as the dated shape reads them — the weekly
owner's, at D7's or the extract order's extracts — and the change is the
engine's `period_change`, overall or per stage:

    The live pipeline amount fell by £400k (-14.3%), from £2.8m at the weekly
    extract of 2025-10-30 (October 2025) to £2.4m at the weekly extract of
    2025-11-27 (November 2025).

### 20.4 The run-rate hold

Its two misreads now have their own concepts ([121] → stage_completion_rate,
[134] → stage_pull_through), and [98] had already left the shape. The hold on
`point_in_time/forecast_completion_rate` is released on the next live run's
evidence that they read there, not before.

Pinned by `test_stage_conversion.py` and `test_pipeline_change.py`.

## 21. The full bank; one model call; the smallest loan (2026-09-29)

### 21.1 What the full bank showed

The 21:21 UTC full bank on b1ef994f served 64 of 135 on the governed path.
Every one of the 68 questions from [68] on failed at the language step with
the provider's "credit balance is too low" — the account ran out mid-run; the
readback now prints the interpreter's own failure reason, so this reads as
what it was. Of the 67 asked before that, must answer 48/48 and nice to have
15/15 were served (this morning, on the same 67: 37). One must-answer was
answered wrongly — [59] "How much pipeline is current month?" was read as the
current period and answered with the whole live pipeline; vocabulary 2.11.0
defines a month named as the pipeline's own attribute as its expected
completion timing.

### 21.2 One model call, not a retrieval loop

Owner direction (2026-09-29): "we definitely want to move away from 3 calls
for speed and cost."

The interpreter looked the governed catalogue up through eight read-only
metadata tools. The readback's token counts measured the loop: a median of
about 37k cached prompt tokens read — three sequential model calls, each
re-reading the ~12k prefix plus the tool results — and ~780 output tokens, for
metadata that is identical for every question. That was the right design
against the alternative it replaced (a 24k-token flat vocabulary, sent
uncached, that under-described every concept). It is not against a cached
prefix.

Now the model reads the whole semantic model up front, the way a semantic-layer
analyst product reads its semantic model on every request:

    system  [ rules | orientation | GOVERNED CATALOGUE ]   cached, shared
            [ CLIENT CONTEXT ]                              per request
    tools   emit_candidate_intent only; tool_choice forced; max_rounds = 1

  * The catalogue is `metadata.governed_catalogue` — the metadata service's
    OWN views (`get_concept_metadata`, `get_allowed_values`,
    `search_capabilities`, the asset and portfolio context), over the same
    index the compiler binds against. It can advertise nothing the tools could
    not, and `test_model_sees_no_data` walks it with the rest of the prompt.
  * A client's source portfolio NAMES are per request (`client_context`),
    after the cache breakpoint — a registry belongs to one client and a cached
    prefix is shared.
  * The same model; the same rules; the same compiler re-validating every
    concept. What changes is only where the model reads the catalogue from.
  * Notes in the intent are asked to be one short sentence each (output tokens
    are the slowest and dearest part of a call).
  * Every outcome records `model_calls` and `model_ms`, and the readback
    projects them, so the saving is measured rather than asserted.

The interpreter moved, so its behaviour must be measured again before its
numbers describe it (`test_the_measured_policy_has_not_moved_since_it_was_
measured` is repointed at the change): the next bank run on the deployed
change is that measurement.

### 21.3 D14 — the smallest loan has a balance

Owner decision (2026-09-29): "smallest loan should have a simple rule where it
must be > 0." A redeemed loan's balance is zeroed on purpose
(`closed_account.zero_fields`), so the minimum over the funded book was £0.

  * The rule is the REGISTRY's: `statistic_scope: {min: {comparator: gt,
    value: 0}}` on current_outstanding_balance, in the build script's
    curation (`build_mi_semantics_registry.py`), regenerated — the build
    script also now carries the two earlier hand edits (D10's balance-weighted
    valuation, 2.8.0's `superseded_by`), so regeneration reproduces the file.
  * The compiler writes it onto the plan as a predicate on the figure's own
    output. The executor applies it like any filter, the receipt discloses it
    ("Minimum Balance · Balance > 0 · N loans"), and the coverage ledger
    proves it. Nothing downstream knows the rule exists.
  * A question that already restricts the balance ("the smallest loan over
    £50k") stands on its own. A minimum asked beside another figure in one
    output is refused rather than answered — the predicate would change the
    other figure too.

Pinned by `test_smallest_loan_has_a_balance.py`,
`test_governed_metadata_access.py` (the catalogue is the tools' own views; one
call) and `test_model_sees_no_data.py`.

## 22. The one-call full bank (2026-09-30)

622351a2 (vocabulary 2.12.0, one model call), 135 questions, read back by run
36689644793 (`qb_recorded_intents_20260930.json` keeps the model's readings,
no figures).

    served by the governed path   110 / 135   (last night 64; yesterday 51)
    must answer                    83 / 88
    nice to have                   21 / 31
    fine to decline                 6 / 16    (10 declined or asked back)

The measurement of §21.2:

    model calls per question       1  (all 135)
    model time                     5.5s median, 7.2s p90
    fresh input tokens             5,520 -> 352 (median)
    output tokens                  784 -> 383
    relative cost per question     13.1k -> 7.4k input-token equivalents
    end to end, same 63 questions  30.3s -> 24.0s median

The model is now a fifth of a question's time. The rest is the legacy answer,
which `mi_service` still computes FIRST on every question — its own model
call included — and keeps as the fallback, before the governed attempt runs.

### 22.1 What was wrong, and fixed at its definition (vocabulary 2.13.0)

  * [87] "What is the expected funded balance?" (must answer) was read as the
    pipeline's weighted amount (£6.9m) where the retrieval-loop interpreter
    had read the forecast funded balance (£94.1m). Neither definition named
    the phrase; the forecast funded balance's now does, and the weighted
    amount's rules it out.
  * [102] "How much ACTIVE pipeline is excluded from weighting?" (fine to
    decline) dropped 'active' and was answered with the whole extract's
    exclusion (£1.17bn, more than the live pipeline). The definition now says
    the figure covers the whole extract, so a live restriction is kept — and
    the runtime declines it.
  * [125, 126] the 8- and 12-week run-rates arrived in the held run-rate shape
    with their window dropped. The run-rate's definition now says it is
    measured over the forecast's own window; a question naming another keeps
    it in `time`, and the runtime refuses it (pinned with the hold off).

### 22.2 What changed in the runtimes

  * [82] "Compare latest pipeline with prior pipeline" (must answer) was a
    relative pair with no grain, refused. The pipeline's periods ARE its
    weekly extracts, so 'prior' with no grain is the previous extract, and
    the receipt says the grain was the pipeline's own
    (`NATIVE_GRAIN_RULE`).
  * The run-rate hold is NOT released. Every reading it was built for moved
    to its own concept ([113], [121], [134]; [98] asks back), but [125, 126]
    arrived in the shape with their window dropped — served, they would be
    answered with the forecast's own window. It comes off when a live run
    shows the shape arriving for [112] alone
    (`test_why_the_run_rate_hold_stays_on_the_latest_readings`). Legacy
    answers [112] correctly meanwhile.

### 22.3 Open

  * [122, 128] time to scale — D9 needs ERE's `portfolio.stage`
    (`pre_securitisation_spv`: scale £100m; `established`: £200m). An owner
    fact, not a code change.
  * [135] "When are pipeline cases expected to complete?" was read as the
    amount AND the case count by expected completion month; the pipeline
    runtime serves one figure per answer, and legacy refused it too. Two
    figures on one axis is answer composition (P1, D4/D5), not a pipeline
    branch.
  * Legacy first: the governed attempt could run first and the legacy answer
    be computed only when it declines — the step towards switching legacy
    off, and most of the remaining time and model cost per question.

## 23. Governed first; scale at £250MM; when the pipeline completes (2026-09-30)

Owner decisions (2026-09-30), on §22's open items:

    1  Scale for an ESTABLISHED client is £250MM (was £200MM). ERE is NOT
       established: it is a new SPV before securitisation, at scale at £100MM
       (owner correction, same day — the stage was briefly recorded as
       established in a1da9582's parent commits).
    2  The governed attempt runs FIRST; the legacy path only when it declines.
    3  "When are pipeline cases expected to complete?" is a DATE, from the
       book's own history of how long cases take to complete.

### 23.1 Scale (D9, amended)

`config/system/scale_policy.yaml`: `established` threshold 250,000,000; a
pre-securitisation SPV stays at 100,000,000. ERE's authored configuration
records `portfolio.stage: pre_securitisation_spv`. The live service reads the
configuration OCC ACTIVATED (`currency.client_config_path`), never the
repository copy — production answered [122, 128] "scale not configured"
because the activated configuration carries no stage. They answer (against
£100MM) once OCC activates one that does: Client onboarding → Amend →
Securitisation stage "New SPV before securitisation" → Approve → Activate.

### 23.2 Governed first

`mi_service._run_analysis` made the governed attempt AFTER the legacy parse
(its own model call) and the legacy answer, which it handed over as the
fallback and recorded beside the governed value. §22 measured the cost: the
model is a fifth of a question's time; the legacy path is most of the rest.

The attempt needs nothing the legacy path computes, so it now runs once,
before the legacy parse, for an allow-listed principal. A question it answers
returns at once — no legacy parse, no router, no legacy computation. A
question it declines is answered by exactly the legacy path it always was,
from the parse on; the attempt is not repeated. The fallback is still
incapable of failing; it is computed only when used, and a governed answer's
evidence record says `legacy_result_available: false`. Pinned by
`test_slice2_serving_repair.py` (`test_a_governed_answer_never_runs_the_
legacy_path`; a declined request keeps the legacy answer unchanged).

This is the step before switching the legacy path off: when the governed path
declines only what should be declined, the fallback is a refusal.

### 23.3 When the live pipeline is expected to complete

The history owner already measured, per stage, the median days from a case
first being seen at the stage to completing (`historicalCompletionTiming
ByStage`, over the cases that DID complete). It now applies that to every live
case at the latest extract and publishes `expectedCompletionByStage` (median
expected date, live cases, median days, completions measured, live cases
already past it, sufficiency) and `expectedCompletion` (the median over all
live cases). The stage-movement semantic model declares
`expected_completion_date` (unit `date`); the one engine reads it:

    Expected completion date: 2026-06-08 — the median over the 5 live cases of
    the date each was first seen at its stage plus the book's median time to
    complete from that stage, measured on the cases that did complete — a date
    for the cases that complete, not a promise that they will.

It is conditional on completing, and says so; the share that complete is
`stage_completion_rate`. The extract's own expected-completion month keeps
'by expected completion month/date' and no longer claims 'when are cases
expected to complete'. Vocabulary 2.14.0. Pinned by
`test_expected_completion_date.py`.

## 24. The 10:26 check; a grouped figure is a breakdown; history on demand (2026-09-30)

The 11-question check on b43cd904 (vocabulary 2.14.0, governed first):
[87] £94.1m and [82] +£2.9m served correctly; [102] and [125, 126] declined
as they should; [122, 128] wait on OCC activating ERE's stage (PR #510).
[135] "When are pipeline cases expected to complete?" was refused.

### 24.1 A grouped figure is a breakdown (normalisation rule 7)

The model's reading of [135] was right in everything that carries meaning:
`pipeline_stage_movement`, `expected_completion_date`, grouped by
`origin_stage`. It labelled the operation `point_in_time`, and the compiler
refuses a grouping on a single figure — with a reason that names the meaning
it refused: "a grouping makes output 'primary' a breakdown, not a
point_in_time". The operation label and the grouping state one thing twice
(how many figures the answer is), so the label has one canonical value. That
is a representational redundancy of the kind `normalise.py` exists to remove, not
a reading to correct:

    point_in_time + every output grouped  ->  breakdown
        only when the capability produces a breakdown (limit_assessment does
        not, and keeps its refusal), and only when EVERY output groups (a
        mixed intent keeps its refusal — rewriting it would make the
        ungrouped output the unsupported one).

It adds no dimension, drops no filter, picks no member and infers no ranking.
It is not reached for the one question: over the whole catalogue — 75
breakdowns across the funded book, the pipeline, the forecast and stage
movement — the grouped single figure compiles to exactly the plan the stated
breakdown does, and where the breakdown does not compile neither does the
label (`test_every_breakdown_the_catalogue_can_make_reads_the_same_labelled_a_
figure`).

It SUPERSEDES an earlier fail-closed position (`test_cde_fail_closed.py`): "a
grouped point-in-time is a breakdown; silently promoting it would be the
compiler deciding what the question meant". The slots decide it — the grouping
is stated and admits one plan — and the rewrite is not silent. The replay of
the 2026-09-28 run shows what that refusal was also doing: [105, 106]
"forecast balance by stage" are now plans, and are refused by the forecast
runtime, whose owner publishes no split by pipeline stage — the refusal moved
from a label to the owner of the figure, which is where it belongs. [100,
101], stage groupings with no measure and a blocking ambiguity, now ask back
rather than refuse. Both pinned in `test_production_bank_perimeter.py`.
A question about one member keeps its filter, so its breakdown is that
member's row. The model's label still travels in `intent_claims`; the rewrite
is recorded in plan provenance. Normal form 1.1; the plan identity is
unchanged (provenance is outside it). Pinned by
`test_grouped_figure_is_a_breakdown.py` and, on the recorded reading, by
`test_expected_completion_date.py`.

The same answer by stage exposed two presentation faults, fixed in the
semantic model rather than the renderer. The axis read "from stage" (the
rates' name for `origin_stage`); a binding may now name its axis, and the date
names it "current stage". The "not a promise" caveat went only with the single
figure, because it sat inside `explain`, which carries whole-pipeline
companion figures; a measure may now declare a `caveat` that goes with every
shape:

    Expected completion date by current stage: KFI 2026-05-22, Application
    2026-06-08, Offer 2026-06-08. A date for the cases that complete, not a
    promise that they will. As at pipeline history 2026-05-29.

A stage's median date can fall before the as-at date: the stage's live cases
are, at the median, already past the book's typical time to complete from it.
The table carries `pastTypical` beside each date. Whether such cases should be
dated differently is a methodology decision for the owner, and is not taken
here.

### 24.2 Speed: the history on demand, and where the rest goes

Governed answers still took ~27-32s with the legacy path gone (§23.2), against
a median model time of 5.5s. The production seam built the governed inputs for
every attempt, the pipeline's case history among them — a listing of every
weekly extract and a copy of the cached model — and a funded question never
reads it. The seam now hands the canary a provider, resolved only by a
pipeline, stage-movement or forecast plan (`plan_serving_canary._history`;
`test_history_is_read_only_when_needed.py`).

The rest is measured rather than guessed: every bank question records its
per-stage timing (85c3fd41), now with each governed input timed on its own —
`mi_query.governed_inputs.snapshot_store`, `.source_registry`, `.pipeline`,
`governed.interpret_and_compile`, `governed.pipeline_history`. The first
trend question after a deploy (346s on [82]) builds the weekly history cold;
precomputing it when an extract arrives is the next step once the timings name
the order of the remaining costs.

## 25. No bank wording in the model's view; held-out variants (2026-09-30)

Owner direction, 2026-09-30: "you must not make local / tactical fixes just to
pass the 135 bank. These MUST be supportive of other natural language
variants."

### 25.1 The audit

Everything the model sees — rules, orientation, the governed catalogue with
every definition, CLIENT CONTEXT and the intent tool — was checked against
every question bank in the repository (15 files, 1,184 questions).

    verbatim   three bank questions were quoted word for word, each put
               there by a definition fix: 'how much pipeline is overdue' and
               'how much pipeline is current month' (the pipeline's timing,
               2.11.0), 'what completion rate is assumed from KFI to
               completion' (the stage completion rate), 'when are pipeline
               cases expected to complete' (2.14.0).
    restated   six more reproduced a bank question with a word or two
               changed: 'how much of the forecast comes from the funded
               book', 'funded vs pipeline contribution', 'milestone dates to
               funding thresholds', 'project the funded balance over the next
               N months', 'how much pipeline has lapsed past its stage
               window', 'funded balance plus weighted pipeline'.

A definition that quotes the question teaches the model the sentence, and the
next reader who words it differently gets whatever the definition said
before. Each now states the meaning it rests on, keeping term-level synonyms
('current month pipeline', 'the expected funded balance') — the names a
reader uses for a figure, which a catalogue must carry. Vocabulary 2.15.0.

The other changes since 2026-09-29 were checked the same way. Each is at the
level of meaning, not of a sentence: a statistic's scope on the registry (D14),
a runtime rule for a relative pair with no grain (§22.2), a restriction the
weighting exclusion must keep ([102]), a run-rate window kept ([125, 126]),
normalisation rule 7 over the whole catalogue (§24.1), a new governed measure
the owner decided (§23.3).

### 25.2 The guards

`test_no_bank_question_is_in_the_models_view.py`, over every bank:

    - no bank question of four words or more appears in the model's view;
    - no phrase in it carries 85% of a bank question's meaning-bearing words
      with at most two others, once the governed concepts' own names (label,
      id, aliases) are taken out — a question that IS a concept's name is not
      quoted by naming the concept;
    - both checks are shown to catch what 2.14.0 said.

### 25.3 Held-out variants

`config/mi/golden_questions/holdout_variants_20260930.yaml`: 105
questions, each asking exactly what one bank question asks in other words
(other verbs and nouns, order, form, contractions, a reader's synonyms) — one
for every must-answer question, two or three for the questions a definition,
rule or runtime was changed for since 2026-09-29 (`holdout_recent`), and a
sample of nice-to-have and fine-to-decline. Written after vocabulary 2.14.0
and normal form 1.1, and not consulted building either; the guard keeps every
one of them out of the model's view.

The test is paraphrase invariance, not a second bank to fit:
`score_variants.py` compares each variant with its bank question on the same
deploy — SAME outcome, path and headline figure, or a finding (DIFFERENT,
PATH, LOST). A finding is fixed at the meaning it exposes, and once one is
fixed the file is spent: the next proof needs variants written after that fix.

The proof the pass mark (D13) asks for is then both: every must-answer bank
question answered correctly, nothing wrong, AND the held-out variants
answering as their bank questions do.

## 26. The held-out variants check; D15, D16, D17 (2026-09-30)

The 13:49 check on 2d6df14d (vocabulary 2.15.0): the eighteen questions changed
for since 2026-09-29, each followed by its held-out variants (42 asked; the
readback found all 42). 17 of 21 variants compared answered exactly as their
bank question. The four that did not:

    [82]  x2   "How has the pipeline changed since the previous extract?",
               "Latest pipeline against the one before it: what moved?" —
               read as a `material_summary` with no measure, sent to the
               FUNDED book's owner of the form, refused for the population.
    [57]       "What's due to complete out of the pipeline next month?" £7.8m
               (face value) against the bank question's £4.8m (weighted):
               two readings of one question.
    [122]      "How long until we're big enough to securitise?" — both
               readings wait on OCC activating ERE's stage (PR #510).

And [135] answered 2026-04-01 for a pipeline as at 2026-09-24: most live cases
are KFIs sat far past their stage's window.

### 26.1 D15 — the pipeline's "previous" is the snapshot before the latest

A change form's owner was chosen from the form alone: `material_summary` is
the funded book's. With no measure to name an owner, the POPULATION now does
(`vocabulary.POPULATION_CHANGE_OWNER`, normalisation rule 4): a change in the
pipeline is the pipeline's, where it implements the form. The pipeline now
implements `material_summary` — what moved between two snapshots: the change
in each headline figure (amount, case count, weighted amount), from the same
weekly owner, through the semantic engine's `period_change`
(`plan_pipeline_runtime._dated_summary`). "Previous", "prior", "last week" and
"week on week" select the latest snapshot and the one before it whatever the
gap, and the answer names both and the gap — "between the previous snapshot
(2026-09-21) and the latest snapshot (2026-09-24), 3 days apart" — never "a
week before". Pinned by `test_pipeline_previous_snapshot.py`.

### 26.2 D16 — expected to complete in a month is face value

The Pipeline tab publishes, for each month bucket, the amount, the case count
and the weighted amount of the same cases. The model is told the month's
"how much" is the amount at face value and the weighted figure only when
weighting is asked for; every answer states the figures it did not lead with
(`receipt["timing_figures"]`). Pinned by
`test_expected_in_a_month_is_face_value.py`.

### 26.3 D17 — lapsed cases are not dated

`pipeline_prep.stage_validity_windows` is now the one definition of lapsed,
read by the forecast's weighting and by the history owner's expected date:
a live case whose time in its stage (from the stage's entry date) exceeds the
stage's window — measured by the run-off model where history supports it,
else the configured fallback — is counted and left out of the date. Each
stage publishes its live, lapsed and dated cases and the window applied.
Pinned by `test_expected_completion_date.py`.

Vocabulary 2.16.0. The variants that exposed D15–D17 are now spent: the next
proof needs variants written after this build.


## 27. The second held-out check; D18, D19 (2026-09-30)

The 15:53 check on 2e4ab2f2 (vocabulary 2.16.0): 18 of 21 variants answered
exactly as their bank question. Two findings are the subject of this section.

    hv2_pipeline_013_1  "Is any of the pipeline overdue to complete, and how
                        much?" — read correctly (the count and the amount of
                        the overdue cases), declined by the pipeline runtime
                        for asking two figures at once, and then ANSWERED BY
                        THE LEGACY PATH with the whole pipeline (£969.8m),
                        "overdue" dropped. The right answer is £0.
    [122] and variants  "scale" declined as not configured: the MI knows its
                        tenant as `client_001`, OCC activated the client as
                        `ERE`, and the MI read none of the activated
                        configuration.

### 27.1 D18 — a decline is the answer

`plan_serving_canary.respond` is the service's entry point: the governed
answer, or the governed decline (`mi_agent/plan_decline.py`), and None only
for a principal the governed path does not serve. `mi_service` calls it
before the legacy parse, and for a served principal returns whatever it
gives — the legacy parse, router and runner are not reached.
(`serve` keeps its contract — None for a decline — for the offline harnesses
that measure the attempt on its own; the service does not call it.)

The decline reads the evidence record the attempt already wrote, and nothing
else: "I understood this as pipeline case count and pipeline amount, for the
pipeline, where expected completion timing is overdue, but I have not
answered it: that figure, or that combination of figures, is not one I can
produce in a single answer yet. Nothing was guessed, and no other figure was
put in its place." A runtime's reason is worded by its FAMILY (the leading
word of the code — MEASURE, PERIOD, POPULATION, …), so a reason added later
is still worded truthfully; a test fails for any declared reason that falls
through to the generic sentence. Each decline carries its kind for the
operator's record — `unsupported`, `clarify`, `model_unavailable`,
`unavailable`, `failed` — which `mi_service` maps to UNSUPPORTED_QUESTION,
AMBIGUOUS_QUESTION, SEMANTIC_MODEL_UNAVAILABLE and CALCULATION_FAILED, all at
HTTP 200. A decline is not measured by the legacy coverage reader (it
answered nothing). The record states `DECLINED` and the sentence given.

Principals the governed path does not yet serve are unchanged: the legacy
path answers them. Moving everybody is a serving-mode change, taken
separately.

### 27.2 D19 — the served tenant reads the one activated client

`client_config.get_single_activated_client_config`: the clients OCC holds an
activated configuration for (onboarded, with a current version — OCC's own
"active"); exactly one resolves, zero or several resolve to nothing and are
logged. `currency.client_config_path` — the MI's one locator — uses it when
the client asked about has no activation of its own AND is the one tenant a
single-tenant deployment serves (`dependencies.serves_only`: no explicit
tenancy registry, and the deployment's `default_tenant_id`). Any other
client resolves to its own activation or to nothing, exactly as before. The
resolution is made once per request.

What changes for the served tenant, which until now read no client
configuration at all: the portfolio stage (so "scale" has its D9
threshold), the governed reporting currency, the client-level asset class
and geography basis where the portfolio registry does not state one, and the
cache fingerprint (one invalidation on deploy). The dashboard's client NAME
is unchanged: `client_identity` names a tenant only from a configuration
addressed to it, and the activated configuration declares its own client id
(`ere_funding_uk`), not `client_001`.

Pinned by `test_governed_decline.py`, the D18 class in
`test_plan_serving_canary.py`, `test_slice2_serving_repair.py` and
`test_single_activated_client.py`.


## 28. Several figures in one answer; what moved in the whole pipeline (2026-09-30)

The two remaining findings of the 15:53 check, built as the design had already
placed them.

### 28.1 Answer composition — several figures of one population (P1, D4)

§22.3 settled it: "two figures on one axis is answer composition (P1, D4/D5),
not a pipeline branch". `mi_agent/plan_composition.py` is that composer, above
every runtime:

    SPLIT      a plan whose one output names several measures becomes one plan
               per figure — the same population, filters, grouping and period.
               Structural; nothing is re-read or re-decided.
    SERVE      each part through `_serve_plan`, the same function a one-figure
               question passes through: the runtime's perimeter, the population
               proof, execution, reconciliation and rendering.
    ALL OR     every figure is answered or the question is declined, naming the
    NOTHING    figure that could not be produced (D18's decline).
    ONE DATA   every part must declare the same data (the identity its receipt
               states); otherwise the answer is withheld
               (`COMPOSED_FIGURES_NOT_ALIGNED`), never shown side by side.
    COMPOSE    every figure's own governed sentence in the order asked (each
               names its measure and as-at, D4), then what they say about the
               population once; every KPI in one card; one table per shared
               axis, rows matched by member, a column per figure.

The coverage owner proves a composed answer as the conjunction of its parts:
each part's ledger exactly as if that figure were asked alone, plus one entry
per figure asked for, resolved only by a fully accounted part stating it
(`mi_service._composed_plan_coverage`). A runtime that states several figures
from one owner payload (stage movement's reconciliation and summary) says so
(`serves_figures_together`) and is not split. A renderer that states companion
figures (D16) does not restate one a sibling part states.

It applies to the funded book and the pipeline alike ("how many loans and what
balance, by region" composes by the same rule as "is any of the pipeline
overdue, and how much"). Pinned by `test_several_figures_one_answer.py`.

### 28.2 What moved in the whole pipeline (vocabulary 2.17.0)

The movement owner publishes, for its latest pair of extracts, every case
classified once as arrived, moved stage, left or stayed, with counts, amounts
and a reconciliation to both extracts. Stage movement now implements the
`material_summary` form over that (`CAPABILITY_CHANGE_FORMS`), so normalisation
rule 4 keeps the form with the owner of the figures instead of binding it to
the funded book's owner. The runtime serves it when no stage is named (one
stage's movement is that stage's reconciliation), the figures named are ones
the summary states, and the period is the latest extract and the one before
it — D15's "previous"; any other pair is refused. The owner's wording
(`stage_movement_query`, subtype `summary`) reads its own event totals and
per-stage rows and states its residuals rather than hiding them. Pinned by
`test_pipeline_movement_summary.py`.
