# Production question-bank review — current state vs target state

    SOURCE   operator-supplied fresh run, 135 questions, production data
             funded 568 loans / £87.1m as at 2026-08-31
             pipeline 6,165 cases / £1.18bn as at 2026-09-24
    SCOPE    Funded, Pipeline, Forecast only. Concentration Limits and Borrowing
             Base are not established by the client, and no question in this bank
             exercises them — so nothing here is attributable to their absence.
    READ-ONLY. No code, config or data changed.

## The run, counted

    OUTCOME    ANSWERED 95    REFUSED 40
    SERVED     NEW 41 (30%)   LEGACY_FALLBACK 94 (70%)

    category              n   answered      served NEW
    funded_kpi           20   19 ( 95%)     16 ( 80%)
    funded_breakdown_1d  25   23 ( 92%)     15 ( 60%)
    pipeline             27   15 ( 56%)      6 ( 22%)
    pipeline_evolution   15   13 ( 87%)      4 ( 27%)
    forecast             26    7 ( 27%)      0 (  0%)
    forecast_scale       22   18 ( 82%)      0 (  0%)

    top fallback reasons
      31  CLARIFY_NOT_SERVED_IN_THIS_SLICE
      26  INELIGIBLE:POPULATION_NOT_EXECUTABLE
       9  INELIGIBLE:DIMENSION_NOT_SUPPORTED
       8  INELIGIBLE:GEOGRAPHY_REQUESTED
       8  REFUSE_NOT_SERVED_IN_THIS_SLICE

    latency  median 37.9s  mean 42.1s  max 295.3s
             40 of 135 over 45s, 11 over 60s

Governed serving is up from the certified estate (25 of 134, 18.7%) to 41 of 135
(30%), and the whole of that gain is in the funded slice plus a new
`governed_plan_pipeline` route that did not exist at `5c436961`.

## Where the estate sits against target state

**Funded is close to target.** 80% of KPI questions and 60% of one-dimensional
breakdowns are served by the governed plan, and the answers are clean and
self-describing: "Balance: £87.1MM · 568 loans", "Weighted-average Current LTV:
44.4%". The refusals are the good kind — [20] declines a weighted-average valuation
because weighted average is not a governed statistic for that measure and says it
did not substitute the total.

**Pipeline is routed but not finished.** The governed route exists and fires, and
the pipeline data is real and correct. Its output composition is not.

**Forecast is entirely outside the governed contract — 0 of 48 across both forecast
categories — while the forecast capability demonstrably works.** `forecast_scale`
answered 18 of 22 with real numbers: run-rate £1.2m/month, £100m reached around
2027-07 with downside/base/upside bands, a -25% scenario, "£12.9m of additional
completions ≈ 10 months". Cohort conversion, temporal comparison, analytical
composition and run-rate extrapolation all produced good governed figures. None of
it went through a governed plan.

## The failure that matters: answers published over the wrong thing

Sixteen of the 95 answers (17%) are materially wrong or are not answers. They are
one class: the system computed over a different population, measure, period or
shape than was asked, and published the result as the answer.

    [ 58] LEGACY  "How much pipeline is overdue?"
                  -> 6,165 loans · £1.18BN            (the whole pipeline)
    [103] LEGACY  "How much pipeline is excluded because of missing probability?"
                  -> 6,165 loans · £1.18BN            (the whole pipeline)
    [ 86] LEGACY  "What is funded balance plus weighted pipeline?"
                  -> Balance: £1.18BN · entire pipeline (funded half dropped,
                     weighting dropped)
    [ 82] LEGACY  "Compare latest pipeline with prior pipeline"
                  -> "moved from £1.18bn in 2026-09 to £1.18bn in 2026-09"
                     (a period compared with itself, reported as +0.43%)
    [ 77] LEGACY  "Show weighted expected funded amount by month"
                  -> "Funded balance over 8 period(s)"  (wrong measure)
    [ 80] LEGACY  "Show pipeline by stage for October and November"
                  -> all 6,165 loans by stage          (period ignored)
    [ 16] LEGACY  "What is the smallest loan?"  -> a 10-group table
    [134] LEGACY  "offer to completion pull-through rate" -> amounts, no rate
    [120] LEGACY  "if completion run rate FALLS by 25%" -> "LIFTS the monthly
                  run-rate from £1.2m to £925k"        (direction word wrong)

    [ 44] NEW     "Show balance by property type"
                  -> "Balance: £87.1MM · Collateral Type: RBLD · 568 loans"
                     (a GROUP BY silently became a FILTER)
    [ 59] NEW     "How much pipeline is current month?"
                  -> the full £1.18bn                  (temporal narrowing dropped)

    [ 50] NEW     "pipeline amount by stage"    -> "Pipeline by pipeline_stage
    [ 51] NEW     "pipeline case count by stage"   across 6 governed stage(s)"
    [ 56] NEW     "which stage has the largest"     — all three identical, no
                                                      values, measure-blind, and
                                                      [56] is a rank question
    [ 70] NEW     "amount evolution by week"  -> "Pipeline pipeline amount across
    [ 72] NEW     "case count evolution by week"  90 governed weekly extract(s)"
                                                   — no figures; doubled word

**Split by path, honestly:**

    silent substitutions   2 of 41 NEW (4.9%)   7 of 94 LEGACY (7.4%)
    narration stubs        5 of 41 NEW          2 of 94 LEGACY

The governed path roughly halves the substitution rate. **It does not eliminate it**,
and the two that got through — a group-by turned into a filter, and a dropped
temporal narrowing — are dimensions the plan-receipt reconciliation already claims
to cover. That is the more uncomfortable half of this finding.

## Testing the four candidate diagnoses

**Routing — NOT the gap.** Routing is behaving correctly. `POPULATION_NOT_EXECUTABLE`
26 times is the governed runtime declining to execute a population base it does not
own (`SLICE_2_POPULATION_BASES = {"funded"}`), and refusing is the right response to
that. The problem is the SIZE of the executable set, not the logic that consults it.

**Capability — NOT the gap.** Forecast, cohort conversion, run-rate, scenario and
temporal comparison all produced correct, well-qualified figures on production data.
Nothing here needs building.

**Interpretation — a real but small and honest defect.** The parser lifts unmatched
English as though it were a governed concept: "'excluded because' is not a governed
measure", "'use historical rates' is not a governed measure", "no loans in this book
match that filter ('weighting', 'active')", "('lapsed')". It fails safely — nothing
is fabricated — but the refusals are not usable by a client, and the same pattern
produced the ('both') and ('among') refusals in the certified estate.

**The actual gap — the governed contract's coverage.** The estate has two tiers: one
with a proof obligation and one without. Quality tracks the tier, not the
capability. Funded is 80% inside and its answers are trustworthy; forecast is 0%
inside and 73% of its questions cannot be answered at all, while the same
capabilities answer fine outside the contract with no obligation to prove what they
computed.

## Recommendation — three non-tactical enhancements, in order

**P0 — Extend the governed execution perimeter to the pipeline and forecast
population bases.** One declared set gates 26 of the 94 fallbacks and all 48
forecast questions. The owners already exist and already produce correct figures;
this brings them inside the contract rather than building anything. It is the single
change that moves the most questions, and it needs no new capability, no new
interpretation and no prompt change.

**P0 — Make plan-receipt reconciliation a precondition of ANSWERED on every path,
and close its gaps inside the perimeter.** No answer should publish unless the
receipt proves the population, measure, period and shape the plan asked for were the
ones executed. This is what makes the funded refusals good, and its absence is what
lets £1.18bn be published as the answer to "how much pipeline is overdue". [44] and
[59] show the obligation is incomplete even where it applies: group-by must not
degrade to a filter, and a temporal narrowing must not vanish.

**P1 — Finish the `governed_plan_pipeline` output composition.** Every grouped and
series answer on the newest governed route is value-free and measure-blind, and
"Pipeline pipeline amount" is a formatting defect in the same composer. It is one
owner, it is contained, and it is the first thing a client sees on the route the
programme just built.

**P2 — Stop the interpreter naming unmatched wording as a concept.** Contained, and
it converts a class of unusable refusals into honest ones.

**Not a model enhancement, but it will decide adoption:** median 37.9s per question,
40 of 135 over 45 seconds, one at 295 seconds. No MI user waits 38 seconds for
"what is the funded balance". This belongs on the same roadmap as the above.

## What NOT to do

Do not chase the individual wrong answers. Each of the sixteen looks like a
one-line fix in a different place, and fixing them case by case would add sixteen
narrow rules and leave the property that produced them untouched. The property is
that an answer can be published without proving what it measured. Fix the property.
