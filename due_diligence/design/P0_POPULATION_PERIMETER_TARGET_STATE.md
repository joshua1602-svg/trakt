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
        Open for the build: AUM proposed as the funded balance across the
        client's portfolios, and the SPV's stage recorded in client config.

    D10 WEIGHTED-AVERAGE VALUATION                           SETTLED 2026-09-29
        Answered where the book carries valuations: it is the dashboard's
        "Weighted avg property value" tile and the LTV denominator. Defined as
        the tile defines it — balance-weighted current valuation. Registry entry
        `current_valuation_amount` permits `weighted_avg`, weight
        `current_outstanding_balance`; vocabulary 2.4.0. Pinned by
        `tests/test_weighted_average_valuation_is_the_dashboards.py`.

    D11 WHAT "OVERDUE" MEANS                                 PROPOSED 2026-09-29
        Two different questions share the word. About the PIPELINE, it is the
        Pipeline tab's own definition: live cases whose expected completion
        month is before the extract's month (`overdueExpectedCompletion*`).
        About LOANS, it is arrears — answered only where the tape carries the
        arrears fields the registry already defines, refused honestly where it
        does not. When the question does not say which, the agent asks.
        Awaiting owner confirmation.

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

### 15.2 The catalogue backlog, from the run — each wired to an existing owner

    concept (what the model asked for)          the dashboard's existing owner
    ------------------------------------------  ----------------------------------------------
    expected completion month; overdue /        pipeline_contract._expected_completion_breakdown
      this month / next month (pipeline, D11)     and _expected_completion_summary
    weighted expected pipeline, total and by    open_totals()["weighted"]; _dimension_breakdown
      stage / broker / region / product            weightedExpectedFundedAmount
    product on the pipeline                     productBreakdown (product_type)
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

    The forecast-balance hold (§14.6) is released by this backlog, not before
    it: once the curve, the scenarios and the forecast's parts each have their
    own concept, the questions that were misread into the balance have a home
    of their own. Re-run evidence (2026-09-29): lifting it today admits none of
    the 135 questions and would remove the only guard on the 2.2.0 misreads,
    because a forward-looking balance is a permitted period form.
