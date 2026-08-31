# MI Query Agent — pipeline stage-movement question bank and capability audit

**Audit only. No production code was changed.**

| | |
|---|---|
| Starting SHA | `fe4af9e` (merge of PR #380 to `main`) |
| Branch | `claude/mi-query-agent-c7-2tlhr6` |
| Question banks added | `migration_phase0/STAGE_MOVEMENT_BANK.yaml` (88), `migration_phase0/STAGE_MOVEMENT_NEAR_NEIGHBOURS.yaml` (28) |
| Path exercised | `POST /mi/query` on the real FastAPI app — no mocks, no bypass |
| Fixture | `tests/fixtures/pipeline_history_5w` (five governed weekly extracts, 2026-05-01 → 2026-05-29) |
| Configuration run | Governed engine alone (language layer off). See §5.1 — the language layer could not be exercised in this environment, and cannot change the capability findings. |

---

## 1. Executive conclusion

**The premise of the brief does not hold in this repository.** The brief states that
pipeline stage-to-stage movement "already exists as a governed analytical capability"
with ENGINE: YES, REACT: YES, PPTX: YES. It does not. The token `stage_movement`
appears nowhere in the tree, and no payload on any surface carries a source →
destination transition, gross arrivals, gross departures, departures by destination,
per-stage stayers, or an amount amendment on persisting cases. §8 traces the actual
payloads.

What exists is a different analytic: **per-stage stock, and the NET week-on-week change
in that stock.** A net stage flow is not an arrival count. Between the 2026-05-15 and
2026-05-22 extracts the OFFER stock moved 3 → 1, a net of −2, while the gross movement
was two departures to COMPLETED and zero arrivals. The two numbers coincide only when
one side of the flow happens to be empty.

So the gap is **not** primarily recognition, binding or routing. It is **analytics**:
the number does not exist to be routed to. Recognition and binding failures sit on top
of that, and they are the dangerous part.

**How much works already: almost nothing, and worse, it does not fail safely.**
Of 88 stage-movement questions, the agent **answered 28 with a figure that is not the
answer to the question asked** — a 32% silent-error rate on this family. The dominant
pattern is precise and reproducible:

> The parser binds **only the SOURCE stage**, as a point-in-time equality filter, and
> silently discards the destination.

"How many cases went from Application to Offer?" compiles to
`filters: {pipeline_stage: APPLICATION}` and returns *"1 loans · £500K"*. The truth is
one case moved (case 2002, **£200,000**). The count coincides; the money does not; and
nothing in the answer tells the reader a transition was never computed.

**Not one of the 54 refusals refuses because stage movement is unsupported.** Three are
deliberate governed refusals, and all three are about *dataset interpretation* ("how many
cases completed" — funded or pipeline?), not about movement. The other 51 refuse because
an unrelated guard happened to fire: an unknown measure noun, a missing period, an empty
filter match, a chart needing a dimension. Whether a reader gets a wrong number or a
refusal is therefore arbitrary from the reader's point of view: "moved from KFI to
Application" refuses, "went from Application to Offer" answers, and the two are the same
question.

**Is a narrow additive implementation safe? Yes, for a defined subset, and the
discriminator is measured, not asserted.** A discriminator built on *named governed
stage tokens in a directional construction* — deliberately **not** on the word
"movement" — captures **55 of the 82** unsupported questions with **zero** collisions
against all 28 adversarial near neighbours and **zero** against the six governed
stage-stock questions (§12). The word "movement" is owned by five other governed
analytics and broadening it is the pre-registered abort condition.

**Recommendation: IMPLEMENT ONLY A DEFINED SUBSET** (§16).

---

## 2. Existing Query architecture

`POST /mi/query` → `mi_agent_api.mi_service` →

1. **One parse.** `mi_agent.parsed_question.ParsedQuestion.parse` compiles the question
   into a single governed spec. Every downstream consumer reads that spec; a recogniser
   must never re-parse.
2. **Dataset resolution.** `mi_agent_api.workspace.resolve_dataset` decides funded /
   pipeline / forecast from the question, not from a UI tab.
3. **Optional language layer.** `mi_agent_api.concept_merge_arm.apply` proposes concept
   fills of six kinds only — `category_value`, `threshold`, `measure`, `dimension`,
   `source_book`, `dataset` — which are bound and merged into the SAME spec. It can
   change a filter, a measure or a dataset. **It cannot create a route, a recogniser or
   a governed calculation.**
4. **Capability routing.** `mi_agent_api.recogniser_registry.REGISTRY` — a declarative
   registry of 14 `Recogniser` entries ordered by `(-confidence, priority,
   registration_index)`. Each declares a `recognise` predicate, a `handle` callable, an
   optional governed `capability`, and free-form `metadata`. A handler returning `None`
   falls through to the next candidate. Registration lives in
   `chat_routing._register_default_recognisers`.
5. **Execution.** The winning handler calls an internal governed service — `evolution`,
   `temporal_compare`, `forecast_extrapolation`, `risk_limits`, `movement_summary`,
   `mi_workflows.*` — and never computes the analytic itself.
6. **Deterministic composition.** The result is shaped into the same artifact union the
   React chat already renders (chart | table | risk | kpi), with a reconciliation block
   and a semantic-coverage ledger.
7. **Fall-through.** `try_route` returning `None` defers to the point-in-time engine
   (`run_mi_agent_query` + `adapt_workflow_result`).

**How stage movement would need to fit it:** as one more `Recogniser` in that registry,
whose handler delegates to a governed calculation that does not exist yet, and whose
recognition is strict enough to return `None` for everything the fourteen existing
recognisers already own. That is the shape of `mi_workflows.concentration_analysis`,
which is the most recent capability added and is the pattern to copy.

---

## 3. Existing question-bank architecture

The mechanism extended is `migration_phase0/CFO_ACCEPTANCE_BANK.yaml`, graded by
`migration_phase0.pack_grader.grade_cfo`. Its schema is six keys:

```yaml
- {q: "...", family: <name>, expect: DELIVER|REFUSE, must: [...], must_not: [...], rows: N}
```

and its verdict vocabulary is `CORRECT` / `TRUE_REFUSAL` (correctly declined) /
`FALSE_REFUSAL` (declined but answerable) / `WRONG` / `NO_ORACLE`.

Both new bank files use **that schema and no other key**, and were graded by **that
function, unmodified**.

> **NEW ORACLE FRAMEWORK: NO**
> **NEW QUESTION-BANK SCHEMA: NO**

Per §3 of the brief, the diagnostic classification, first-failure attribution and
truth figures this audit needed are recorded **here in the report**, not bolted onto
the bank schema.

`expect` was written from the question and the governed data before execution, exactly
as the CFO bank's header requires — independently of what the agent returns. §8
establishes what the governed data can actually produce; that is what set every
`expect`.

Neither new file is reachable by any test: `completeness_calibration.py` and
`data_claim_audit.py` name the banks they read literally, and nothing globs
`migration_phase0/`. The frozen regression manifest therefore cannot move on account of
these files.

---

## 4. Question set

| Family | n |
|---|--:|
| A source → destination, case count | 12 |
| B source → destination, amount | 8 |
| C departures | 7 |
| D arrivals | 6 |
| E stayers / persisting cases | 6 |
| F amendments on persisting cases | 6 |
| G completions | 9 |
| H withdrawals / terminal exits | 9 |
| I stage reconciliation | 7 |
| J largest / most material movement | 6 |
| K period comparison | 6 |
| L broad stage-movement summary | 6 |
| **Stage movement total** | **88** |
| Adversarial near neighbours | 28 |
| **Grand total** | **116** |

Natural-language variation is carried on genuine semantic distinctions, not mechanical
permutation: movement verbs (*moved, progressed, advanced, went, transitioned, flowed*),
arrival (*arrived, entered, came into, moved into*), departure (*left, departed, exited,
moved out*), persistence (*stayed, remained, persisting*), terminal (*completed,
withdrawn, dropped out*), subject (*cases, applications, pipeline cases, deals*), value
(*balance, amount, value, exposure*), direction (*from X to Y, X into Y, out of X into Y,
between the last two extracts*) and time (*this period, month on month, previous
period*).

---

## 5. Harness proof

Before interpreting a single stage-movement failure, the same path was proven on basic
pipeline questions.

| Probe | Result |
|---|---|
| "Show pipeline amount by stage." | **Answered.** 5 groups, 8 cases, dataset `pipeline`. |
| "How many pipeline cases are in each stage?" | **Answered**, 5 groups — but rendered as *Total Balance* grouped by stage rather than a count. A measure-binding observation, recorded, not a harness fault. |

The fixture is fit for purpose and its movements are the point:

| case | 05-01 | 05-08 | 05-15 | 05-22 | 05-29 |
|---|---|---|---|---|---|
| 2001 | KFI | APPLICATION | OFFER | COMPLETED | COMPLETED |
| 2002 | KFI | KFI | APPLICATION | APPLICATION | OFFER |
| 2003 | OFFER | OFFER | OFFER | OFFER | OFFER |
| 2004 | APPLICATION | APPLICATION | APPLICATION | WITHDRAWN | WITHDRAWN |
| 2005 | – | KFI | KFI | APPLICATION | APPLICATION |
| 2006 | KFI | KFI | KFI | KFI | KFI |
| 2007 | APPLICATION | OFFER | OFFER | COMPLETED | COMPLETED |
| 2008 | – | – | KFI | KFI | OFFER |

Governed truth for the latest window (2026-05-22 → 2026-05-29), computed from
`pipeline_contract.load_prepared_pipeline` — the loader `evolution.pipeline_evolution`
itself uses:

| from | to | cases | amount |
|---|---|--:|--:|
| APPLICATION | OFFER | 1 | £200,000 |
| KFI | OFFER | 1 | £800,000 |
| APPLICATION | APPLICATION (stayed) | 1 | £500,000 |
| KFI | KFI (stayed) | 1 | £600,000 |
| OFFER | OFFER (stayed) | 1 | £300,000 |
| COMPLETED | COMPLETED | 2 | £800,000 |
| WITHDRAWN | WITHDRAWN | 1 | £400,000 |

**Fixture limitation, recorded not worked around:** every case carries a constant loan
amount across all five extracts. There is therefore **no amount amendment on any
persisting case anywhere in the fixture**, and family F cannot be given a numeric target
even in principle. Family F is still asked, because whether the agent *invents* an
amendment is itself worth measuring — it does not.

### 5.1 Configuration caveat

The shipping configuration runs the deterministic engine **plus** the concept-merge
language layer. The API key retained in this environment has been rotated and returns
`401 authentication_error`, so the language layer could not be exercised and the bank
was run against the **governed engine alone**. This is a real production configuration
— it is the one the acceptance bank scores at 127 correct / 4 wrong — but it is not the
shipping one, and the limit must be stated rather than papered over.

**It does not change the capability findings.** The language arm's six proposal kinds
bind values, thresholds, measures, dimensions, books and datasets into the existing
spec. None of them can create a recogniser, a route, or a governed calculation. A
capability that does not exist cannot be reached by a better reading of the question.
What the language layer *could* change is the vocabulary and measure-binding subset —
so the 9 `VOCABULARY_GAP` and 3 `measure binding` attributions in §7 should be read as
an **upper bound** on that layer's contribution, and everything else as unaffected.

---

## 6. Current Query results

**88 stage-movement questions, graded by `pack_grader.grade_cfo`:**

| Verdict | n | share |
|---|--:|--:|
| CORRECT | 3 | 3.4% |
| CORRECTLY DECLINED | 54 | 61.4% |
| DECLINED BUT ANSWERABLE | 3 | 3.4% |
| **WRONG** | **28** | **31.8%** |

The three CORRECT are governed stage-**stock** questions that were never the point:
"How many pipeline cases are withdrawn?" (1, £400K), "What is the withdrawn balance in
the pipeline?" (£400K), "What is the balance at the Completion stage of the pipeline?"
(£800K, 2 cases). **Zero** stage-movement questions were answered correctly, because
none can be.

### 6.1 The 28 wrong answers, by pattern

| Pattern | n |
|---|--:|
| A stage **stock** returned for a question about **flow** | 15 |
| The **source** stage bound as a point-in-time filter, the **destination** silently dropped | 6 |
| A **forward forecast** substituted for a historical movement | 4 |
| The **wrong dataset** — the funded book answering a pipeline question | 3 |

(The same source-bound/destination-dropped binding also causes four of the 54 refusals,
where the transition phrase matched an empty set rather than a populated one. It is the
single most common defect in this family; whether it surfaces as a number or a refusal
is incidental.)

Worked examples across the first two patterns — what was asked, what the spec became,
what came back, and what is true:

| Question | Compiled to | Returned | Truth |
|---|---|---|---|
| How many cases went from Application to Offer? | `pipeline_stage = APPLICATION` | 1 loans · £500K | 1 case · **£200,000** |
| How many deals transitioned out of KFI into Application? | `pipeline_stage = KFI` | 1 loans · £600K | **0 cases** |
| How much exposure transitioned from KFI to Application? | `pipeline_stage = KFI` | £600K | **£0** |
| Show balance transferred from Application to Offer. | `pipeline_stage = APPLICATION` | £500K | **£200,000** |
| How many cases arrived in Application? | `pipeline_stage = APPLICATION` | 1 loans · £500K | **0 arrivals** |
| How many cases stayed in Application? | `pipeline_stage = APPLICATION` | 1 loans · £500K | 1 case · £500,000 (coincides) |
| How many Offer cases remained at Offer? | `pipeline_stage = OFFER` | 3 loans · £1.3MM | **1 case · £300,000** |

The last two are the most instructive pair. The stayer question about Application is
*accidentally right* — because nothing arrived in Application that week, so stock and
stayers coincide. The identical question about Offer is wrong by a factor of three,
because two cases arrived. The agent has no way to tell those two situations apart, and
neither does the reader.

**A forward forecast substituted for a historical movement (4).** Four questions route
to `analytical_composition` and come back with a *projection*:

> "How many offers moved to Completion?" → *"Offer stage pipeline is £1.3m across 3
> case(s) as at 2026-05-29. Expected completion amount from pipeline cases at Offer
> stage: £975k. Expected to land: 2026-06 £975k."*

The reader asked what happened; they were told what is expected to happen. The truth for
that window is zero cases and £0.

**Wrong dataset (4).** "How many cases were withdrawn?" resolves to the **funded** book
and answers *"640 loans · £172.1MM"*. "Which transition moved the most balance?" routes
to `period_change_analysis` on the funded book and reports *"+£22.6m (£149.5m →
£172.1m)"*. Neither figure has anything to do with the pipeline.

**A stage stock returned for a flow question (15 including the above overlap),** and one
case — "How many Offer cases were withdrawn?" — where the word *withdrawn* was dropped
entirely and the OFFER stock returned instead.

### 6.2 Why the 54 refusals are not protection

Clustered by refusal text:

| n | Refusal reason | Example |
|--:|---|---|
| 19 | "couldn't map this question to a governed analytic" | Show case movement from Application into Offer. |
| 8 | "needs two governed reporting snapshots to compare" | How many cases moved from KFI to Application? |
| 9 | "'X' is not a governed measure in this dataset" (*departures, arrivals, withdrawals, reconcile, movement, changed, increase decrease, explain movement*) | Where did Offer-stage departures go? |
| 3 | "No loans in this book match that filter ('offer to completion')" | How many cases progressed from Offer to Completion? |
| 3 | "'amount' / 'value' could mean more than one governed measure" | What value progressed from Application to Offer? |
| 3 | "could not be applied to the calculation" | What drove the change in Application-stage balance? |
| 2 | "names no period to compare over" | Which stage had the most movement? |
| 2 | "bar chart requires a dimension" | What balance exited KFI by destination? |
| 1 | "'Pipeline Stage' is not available in this dataset" | What was the largest stage transition? |
| 3 | governed dataset-interpretation refusal | How many cases completed? |

Only the last three are a deliberate governed refusal, and none of them is *about stage
movement*. The three "no loans match that filter" cases are the clearest illustration:
the phrase **"offer to completion"** was bound as a single category value, matched
nothing, and the empty set produced a refusal. That is luck wearing the costume of a
guard.

---

## 7. First-failure decomposition

The first point at which the current path fails, per question:

| First failure | n | share |
|---|--:|--:|
| Intent / construction recognition | 35 | 40% |
| Capability routing | 12 | 14% |
| Lexical / vocabulary recognition | 9 | 10% |
| Stage extraction / binding | 8 | 9% |
| Direction / source–destination binding | 6 | 7% |
| Dataset selection | 4 | 5% |
| Measure binding | 3 | 3% |
| Time binding | 3 | 3% |
| Result shaping | 2 | 2% |
| n/a (answered correctly, or a deliberate governed refusal) | 6 | 7% |

Read this carefully, because the headline number is misleading on its own. **Intent /
construction recognition** dominates not because the recogniser is weak but because
*there is nothing for it to recognise into*: the spec has no representation of a
transition. `mi_agent.mi_query_spec` carries `filters`, `dimensions`, `metric`,
`aggregation`, `compare_periods`, `bridge_dimension`, `cohort_progression`,
`forecast_mode`, `risk_limit_query` — and no source/destination pair. The moment a
question is parsed, the destination has nowhere to live, so it is dropped. Every
downstream layer is then failing on a spec that already lost the question.

That is why **direction / source–destination binding** attributes only 6: it is rarely
the *first* failure, because the destination is usually gone before binding is reached.

---

## 8. Existing stage-movement capability — traced, not inferred

### 8.1 `GET /mi/evolution/funnel` → `evolution.pipeline_funnel_evolution`

Keys actually present in the live payload:

```
stages, stageLabels, weeks, sourceFiles, uniqueWeeklyExtractsUsed,
series.{KFI|APPLICATION|OFFER|COMPLETED}       -- STOCK: value, count per week
flowSeries.{KFI|APPLICATION|OFFER|COMPLETED}   -- NET week-on-week change in that stock
summary.{STAGE}.{latestFlowValue, latestFlowCount, priorFlowValue, priorFlowCount,
                 fiveWeekAvgFlowValue, fiveWeekAvgFlowCount, deltaFlowValue,
                 deltaFlowCount, latestStockValue, latestStockCount,
                 fiveWeekAvgStockValue}
conversionLagWeeks, lineage, cohortProgression, cumulativeCohortConversion
```

Note `_FUNNEL_STAGES = ("KFI", "APPLICATION", "OFFER", "COMPLETED")` — **WITHDRAWN is
not in the funnel at all**, so no attrition analytic exists on this surface.

### 8.2 `GET /mi/evolution/pipeline` → `evolution.pipeline_evolution`

```
periods[].metrics.{pipeline_amount, pipeline_case_count,
                   weighted_expected_funded_amount}
byStage[].{period, week, stage, value, count}
fiveWeekAverage, pipelineTiming, lineage
```

Stage **stock** per week. No transition.

### 8.3 `mi_agent_api.movement_detail` — the closest thing, and the decisive evidence

`movement_components(current, prior, dims)` joins the two governed prepared frames on
`pipeline_case_identifier` and, in the joined frame, holds **both stages per case**:
`_stage` and `_stage_prior`, alongside `_measure` and `_measure_prior`. It already
resolves the case key, excludes and reports unkeyed rows, reports duplicates, and sums
duplicates so the decomposition reconciles.

It then uses those two stage columns **for exactly one test** — `was_active &
now_terminal` → the `progressed_out` component — and **discards them**. The returned
frame carries only `delta`, `component` and the dimension columns.

The consequence is measurable. Running the live endpoint over the 2026-05-22 →
2026-05-29 window:

```json
"components": {
  "new":            {"amount": 0.0, "cases": 0},
  "removed":        {"amount": 0.0, "cases": 0},
  "progressed_out": {"amount": 0.0, "cases": 0},
  "increased":      {"amount": 0.0, "cases": 0},
  "decreased":      {"amount": 0.0, "cases": 0},
  "unchanged":      {"amount": 0.0, "cases": 8}
}
```

**All eight cases are reported `unchanged`** — including case 2002 (APPLICATION → OFFER)
and case 2008 (KFI → OFFER), which both moved stage. `progressed_out` fires only on a
move into a TERMINAL stage, so every non-terminal transition is invisible here. Its
contributor dimensions are `brokers` and `regions`; **stage is not one of them**. The
endpoint is additionally off unless `TRAKT_MI_ENHANCED_HOVERS` is set.

### 8.4 `mi_agent_api.pipeline_history.build_historical_completion_model`

Tracks each case across all five extracts by `pipeline_case_identifier` (falling back to
`application_identifier`) and builds per-case stage timelines. What it *emits* is
per-stage empirical completion rates and timings, and a cumulative cohort progression
(monotonic % of the KFI cohort reaching each milestone). The timeline itself records
only the **earliest** date at each stage — `if stage not in t["stages"]` — and **carries
no balance at all**. It cannot answer a money question about a transition.

### 8.5 Field-by-field verdict against the brief's §20 list

| Field the brief expects | Present? | Where |
|---|---|---|
| opening count | **yes** | `funnel.series[stage][i].count` |
| opening amount | **yes** | `funnel.series[stage][i].value` |
| closing count | **yes** | same series, next index |
| closing amount | **yes** | same |
| arrivals count / amount | **no — NET only** | `flowSeries` holds arrivals − departures |
| departures count / amount | **no — NET only** | as above |
| departures by destination | **no** | nowhere |
| persisting / staying cases | **no** | derivable in `movement_components`, not derived |
| persisting amount | **no** | as above |
| amount amendment on stayers | **partial, not per stage** | `increased`/`decreased` components, whole pipeline |
| reconciliation residual | **partial** | `movement_detail` reconciles the total, not per stage |
| reporting window | **yes** | `source_dates`, `lineage` |
| stable case identifier | **yes** | `pipeline_case_identifier`, `CASE_KEY` |
| live-stage semantics | **yes** | `_OPEN_PIPELINE_STAGES`, `ACTIVE_STAGES`, `_STAGE_CANON` |
| completion handling | **yes** | `COMPLETED` stage, terminal test |
| withdrawal / terminal handling | **yes as a stage, no as an exit** | `WITHDRAWN` in `TERMINAL_STAGES`; not in the funnel |

---

## 9. Capability mapping

| Question family | Verdict against the existing governed capability set |
|---|---|
| A source → destination, count | **NOT AVAILABLE.** Composable *in the engine* from `movement_components`' joined frame; nothing computes or publishes it. |
| B source → destination, amount | **NOT AVAILABLE**, same join, same absence. |
| C departures by destination | **NOT AVAILABLE.** No payload records a departing case's next stage. |
| D arrivals | **NOT AVAILABLE as a gross figure.** Net stage flow is available and is a different number. |
| E stayers | **NOT AVAILABLE.** One `groupby` away from `movement_components`; not performed. |
| F amendments on stayers | **NOT AVAILABLE per stage.** Whole-pipeline `increased`/`decreased` exists. No fixture variation to measure against. |
| G completion stock | **DIRECTLY AVAILABLE** — and the agent gets it right. |
| G completion flow | **COMPOSABLE** from `funnel.flowSeries.COMPLETED`; the Query layer does not reach it. |
| H withdrawn stock | **DIRECTLY AVAILABLE** — and the agent gets it right. |
| H attrition by origin stage | **NOT AVAILABLE.** WITHDRAWN is not even a funnel stage. |
| I stage reconciliation | **NOT AVAILABLE.** Opening and closing are governed; the arrivals and departures between them are not, so the bridge cannot be drawn without inventing its middle. |
| J largest / most material | **NOT AVAILABLE**, and ranking must not be added to the Query layer (brief §13). |
| K period comparison | **NOT AVAILABLE.** A comparison cannot be more available than the thing compared. |
| L broad summary | **PARTIALLY COMPOSABLE** as a per-stage net-change summary from `funnel.summary`; not as stage movement. |

The brief's §21 worked example — *"if the existing payload already contains KFI
departure → destination APPLICATION → case count, then Query should retrieve that
governed value"* — has no antecedent. The payload does not contain it.

---

## 10. Near-neighbour protection

All 28 ran through the same path. **The protected asset is the route, not only the
verdict**: an implementation that moves any of these has broadened generic "movement"
and must be stopped, even if the verdict still passes.

Eight distinct governed owners are in play — `point_in_time_engine`, `evolution`,
`cohort_conversion`, `period_change_analysis`, `period_change`, `funded_bridge`,
`analytical_composition`, `forecast_extrapolation`. The full route baseline is in §17.

21 answered correctly. The seven refusals are **pre-existing behaviour, recorded as the
baseline, not defects introduced here** — and three of them are correct conduct:

| Question | Route | Why it declined |
|---|---|---|
| What is the current pipeline amount? | point-in-time | *"'amount' could mean Balance or Valuation"* — governed disambiguation |
| How many funded loans were added? | `period_change` | *"names no period to compare over"* — governed clarification |
| How has regional concentration moved? | `period_change` | same |
| How has pipeline balance moved this month? | point-in-time | needs two snapshots; a real pipeline-movement gap |
| How has the pipeline grown? | point-in-time | generic decline; a real gap |
| What is the KFI to completion conversion rate? | `cohort_conversion` | reached the right owner, then could not confirm KFI was applied |
| How much of the pipeline is expected to complete? | point-in-time | forecast capability refusal |

---

## 11. Surface parity

| Question family | Engine | API | React | PPTX | MI Query today | Query gap |
|---|---|---|---|---|---|---|
| source → destination count | COMPOSABLE_FROM_EXISTING_CAPABILITY | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | the analytic itself |
| source → destination amount | COMPOSABLE_FROM_EXISTING_CAPABILITY | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | the analytic itself |
| departures by destination | COMPOSABLE_FROM_EXISTING_CAPABILITY | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | the analytic itself |
| arrivals (gross) | COMPOSABLE_FROM_EXISTING_CAPABILITY | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | the analytic itself |
| arrivals / departures (net) | SUPPORTED | SUPPORTED | SUPPORTED | SUPPORTED | NOT_SUPPORTED | routing only |
| stayers | COMPOSABLE_FROM_EXISTING_CAPABILITY | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | the analytic itself |
| amount amendments (whole pipeline) | SUPPORTED | SUPPORTED (flagged) | SUPPORTED (flagged) | NOT_SUPPORTED | NOT_SUPPORTED | routing only |
| amount amendments per stage | COMPOSABLE_FROM_EXISTING_CAPABILITY | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | the analytic itself |
| completion count (stock) | SUPPORTED | SUPPORTED | SUPPORTED | SUPPORTED | **SUPPORTED** | none |
| completion amount (stock) | SUPPORTED | SUPPORTED | SUPPORTED | SUPPORTED | **SUPPORTED** | none |
| completion flow (net) | SUPPORTED | SUPPORTED | SUPPORTED | SUPPORTED | NOT_SUPPORTED | routing only |
| withdrawals (stock) | SUPPORTED | SUPPORTED | NOT_SUPPORTED (not a funnel stage) | NOT_SUPPORTED | **SUPPORTED** | none |
| stage reconciliation | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | the analytic itself |
| largest transition | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | the analytic itself |
| prior-period comparison of movement | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED | the analytic itself |
| broad movement summary | SUPPORTED (net, per stage) | SUPPORTED | SUPPORTED | SUPPORTED | NOT_SUPPORTED | routing only |

React consumers: `EvolutionPanel.tsx`, `PipelineSnapshotPanel.tsx`,
`insight/InsightStageCard.tsx`, `insight/ConversionContext.tsx`,
`insight/EnhancedMetricTooltip.tsx` (flagged). PPTX: `deck.slide_funnel`, spec id
`funnel` / `origination_flow`.

**The brief's parity claim is not borne out.** Engine / React / PPTX have the
*origination funnel* — stock and net flow. None has stage-to-stage movement.

---

## 12. Recommended implementation

### 12.1 The analytic belongs in `movement_detail`, not in a new module

`movement_components` is already the only production function that holds, per case,
both the prior and current stage across two governed extracts, keyed on the governed
identifier, with duplicate and unkeyed rows already handled. The smallest honest
addition is to **stop discarding the two stage columns** and add one grouping beside
`component_summary`:

```
stage_transitions(components) -> {(from_stage, to_stage): {cases, amount}}
```

Opening and closing per stage already exist in `funnel.series`, so the identity
`opening + arrivals − departures ± amendment = closing` closes without inventing a
second source of either end.

**Query must call this. Query must not compute it.**

### 12.2 The one genuinely new spec field

A transition needs a source and a destination. The existing spec cannot represent it:
`filters` is a point-in-time predicate by construction, and putting the destination
there would silently change the population — which is precisely the defect §6.1
measures. `dimensions` groups one frame and cannot express a pair across two.

So **two scalars on the existing spec** — a source stage and a destination stage, plus
a movement basis (`transition` | `arrivals` | `departures` | `stayers`) — are the
minimum addition, and they belong in `mi_agent.mi_query_spec` next to the existing
`compare_periods` / `cohort_progression` / `forecast_mode` mode fields, which are the
same shape of thing. Nothing else new is required: no new intent enum, no new plan
structure, no parallel parser, no second executor.

### 12.3 The narrowest reliable discriminator — measured

**Not the word "movement".** That word is owned by funded movement, period change,
pipeline evolution, LTV movement and concentration movement, and broadening it is the
abort condition.

The discriminator that the bank evidence supports:

1. **defer** when the question also carries `conversion | forecast | expected to |
   projection | run rate` — these have governed owners, and a KFI→completion phrasing is
   `cohort_conversion`'s own canonical wording;
2. **fire** on an explicit intent phrase — `stage movement`, `stage transition`,
   `movement between stages`, `stage-to-stage`;
3. **fire** on **two governed stage tokens in a directional construction** — *from X to
   Y*, *X into Y*, *out of X into Y*, *leaving X for Y*, *entering Y from X*, *X → Y*,
   *X became Y*;
4. **fire** on **one governed stage token plus a gross-movement event term** —
   departures / arrivals / stayers / left / remained / exited / attrition / amendment.

Stage tokens come from `pipeline_prep._STAGE_CANON`, the existing governed stage
vocabulary — not a new list.

Measured against both banks:

| | result |
|---|---|
| Unsupported stage-movement questions captured | **55 of 82** |
| Adversarial near-neighbour collisions | **0 of 28** |
| Governed stage-**stock** questions captured in error | **0 of 6** |

The deference rule was not decoration: without it, *"What is the KFI to completion
conversion rate?"* is captured and stolen from `cohort_conversion`. That collision was
found by measurement, and the fix is the same deference pattern
`concentration_analysis` already uses to hand ITL3 questions to `geo_exposure`.

The 27 not captured — completion/withdrawal wording with no explicit transition, the
ranking family, period comparison of movement, and arrivals-with-origin phrasing — fall
through to exactly today's behaviour and are **deliberately out of scope** for a first
implementation (§16).

### 12.4 Routing

One `Recogniser` in `chat_routing._register_default_recognisers`:

```
name="pipeline_stage_movement", capability=CAP_PIPELINE, priority=25
```

between `cohort_conversion` (20) and `forecast_extrapolation` (30) — after conversion so
that owner keeps its questions, and well before `evolution` (110) and `period_change`
(85) so a movement phrasing carrying two stage tokens is not swallowed by a generic
period comparison. `recognise` implements §12.3 and returns `False` otherwise; `handle`
returns `None` when the two extracts or the stable identifier are unavailable, so the
question falls through to today's behaviour rather than failing.

### 12.5 Refusal

The refusal must be *about stage movement*, not the incidental guards of §6.2:
"I can report the pipeline position at each stage and its net weekly change, but I have
not answered how many cases moved from Application to Offer — that requires tracking
each case between two extracts, which this book does not report." The existing
controlled-refusal machinery already carries that shape.

**Summary sentence, in the form the brief asked for:** *stage movement fits the existing
Query architecture by extending the governed spec with a source/destination pair,
recognising it through one new registry `Recogniser` gated on the existing
`_STAGE_CANON` vocabulary, and dispatching to a new grouping inside the existing
`movement_detail` governed calculation.*

---

## 13. Production change estimate

| # | Module / function | Reason | Est. LOC | Regression sensitivity |
|---|---|---|---|---|
| 1 | `mi_agent_api/movement_detail.py` — retain `_stage`/`_stage_prior` on the returned frame; new `stage_transitions()` | the governed calculation; the join already exists | 60–80 | **LOW.** Additive: existing components and payload keys unchanged. Guard: `component_summary` output must be byte-identical. |
| 2 | `mi_agent_api/app.py` — publish the matrix on the existing `/mi/insight/movement-detail` | expose the governed value | 10–15 | **LOW.** Endpoint is already feature-flagged off by default. |
| 3 | `mi_agent/mi_query_spec.py` — `stage_from`, `stage_to`, `stage_movement_basis` | the one thing the spec cannot represent | 10–15 | **MEDIUM.** The spec is serialised into the response and compared field-by-field by the acceptance harness; new fields must default to `None` and appear on every existing answer unchanged. |
| 4 | `mi_agent/llm_query_parser.py` — directional construction + event-term recognition over `_STAGE_CANON` | stop binding the source as a stock filter and the phrase as a category value | 90–130 | **HIGH.** This is the single most sensitive file in the estimate. It owns value binding for every dataset; the three "no loans match that filter ('offer to completion')" refusals show transition phrases currently reaching the category-value binder. Must be gated on the pipeline dataset. |
| 5 | `mi_agent_api/chat_routing.py` — one `Recogniser` + handler | routing and delegation | 60–90 | **MEDIUM.** Registry insertion is declarative and ordered; the risk is the predicate, not the mechanism. |
| 6 | `mi_agent_api/adapters.py` / presenters — render the matrix as the existing table artifact | reuse, no new renderer | 20–30 | **LOW.** |
| 7 | Tests — `movement_detail` transition truth, recogniser deference, near-neighbour route locks, refusal wording | pin the discriminator and the 28 protected routes | 200–260 | n/a |

**Production LOC (items 1–6): approximately 250–360.** With tests: **450–620.**

**IMPLEMENTATION COMPLEXITY: MEDIUM.**
The analytic is one `groupby` on a frame that already exists, and the routing is one
declarative registry entry. The cost is concentrated in item 4 — teaching the parser a
construction it has never had, inside the file that binds values for every other
dataset.

**REGRESSION RISK: MEDIUM, and it is not evenly spread.**
Items 1, 2 and 6 are additive and low-risk. Item 3 touches a structure the acceptance
harness compares field-by-field. Item 4 is the real exposure: it is the file whose
current behaviour produces the 10 source-bound-destination-dropped answers, and the
same code path binds category values for the funded book. The measured discriminator
(0/28 and 0/6 collisions) is what makes this a *medium* rather than a *high*: the risk
is bounded by evidence, not by intuition.

---

## 14. Existing Query baseline

Re-measured at `fe4af9e`, not cited. All 166 authoritative questions were re-run through
`POST /mi/query` in the deterministic configuration and their answer surface compared
byte for byte against the frozen record in
`migration_phase0/MI_ACCEPTANCE_BANK_ANSWERS.json`.

**166 of 166 identical.** The recorded grades therefore stand unchanged:

| Verdict | Governed engine alone (re-measured) | Shipping configuration (recorded) |
|---|--:|--:|
| CORRECT | **127** | 136 |
| CORRECTLY DECLINED | **16** | 16 |
| DECLINED BUT ANSWERABLE | **19** | 12 |
| **WRONG** | **4** | **2** |

Named existing failures — deterministic (4):

| id | Question | Why |
|---|---|---|
| Q04C | Show total outstanding balance for London loans in the Direct book with LTV over 50% | correct 24-loan population, balance 7,201,378.77 absent — wrong output grain |
| Q10B | Give me an overview of the pipeline by size and stage | expected 8 cells, artefact carried 5 |
| Q17C | Break Direct portfolio balance down across LTV, ticket size and borrower age | expected 143 cells, artefact carried 5 |
| Q19A | How did the Direct book change last month? | delta 12,366,371.40 absent — wrong temporal shape |

Under the shipping configuration only Q04C and Q19A remain wrong; Q10B and Q17C are
recovered by the language layer.

### 14.1 Frozen regression manifest

The manifest cannot have moved, by construction: no production file was touched, and
neither new bank file is reachable by any test — `completeness_calibration.py` and
`data_claim_audit.py` name the banks they read literally, and nothing globs
`migration_phase0/`.

Verified rather than assumed. `mi_agent_api/tests`, `mi_agent/tests` and
`tests/test_pipeline_history_fixture.py` were run at `fe4af9e` with the new files in
place: **2,256 passed, 19 failed, 265 skipped, 7 xfailed**, and **all 19 failing names
are inside the frozen 85** — nothing new, nothing recovered.

Nothing was repaired. This is the baseline any stage-movement implementation is judged
against.

---

## 15. Pre-registered implementation acceptance

Frozen before any implementation begins.

1. Zero regression in the authoritative 166: **CORRECT ≥ 127 deterministic / ≥ 136
   shipping; WRONG ≤ 4 deterministic / ≤ 2 shipping.**
2. No previously correct question becomes wrong — checked name by name, not in aggregate.
3. No correctly declined question becomes incorrectly answered.
4. **All 28 adversarial near neighbours keep their exact route** (§17). A verdict that
   still passes on a changed route is a failure.
5. Funded-movement questions do not become stage movement: `funded_bridge` and
   `period_change_analysis` keep every question they own today.
6. Generic pipeline-evolution questions do not become stage movement: `evolution` keeps
   its three.
7. Current stage-stock questions preserve their route: all six governed stock questions
   stay on the point-in-time engine and keep their figures.
8. Conversion questions preserve their route: `cohort_conversion` keeps all three,
   including *"What is the KFI to completion conversion rate?"* — the one measured
   collision.
9. Source-stage binding correct on every family-A/B question that the discriminator
   captures.
10. Destination-stage binding correct on the same set — and never bound into `filters`.
11. Count vs amount binding correct: the £200,000 in the worked example must be the
    APPLICATION→OFFER transition amount, never the APPLICATION stock.
12. Ambiguous questions refuse rather than guess — in particular the bare "How many
    cases completed?", which must not silently pick a dataset.
13. A missing stable pipeline identifier produces the existing governed
    unavailable/refusal behaviour, never a whole-book fallback.
14. **Query performs no stage-movement arithmetic.** Asserted structurally, not by
    inspection: no `groupby`, join or subtraction over pipeline frames anywhere under
    `mi_agent_api/chat_routing.py` or the new handler.
15. Query delegates to the governed capability in `movement_detail`.
16. Returned answers reconcile to the same values `movement_detail` serves to
    React/PPTX — the same primitive, not a parallel one that agrees today.
17. The frozen regression manifest stays at exactly 85 names.
18. The refusal for an uncaptured stage-movement question is *about stage movement*, not
    an incidental guard.

### Pre-registered ABORT conditions

Stop, and do not trade Query stability for this feature, if any of these occur:

- **A.** Any of the 28 near-neighbour routes changes.
- **B.** The authoritative 166 loses a single CORRECT, or gains a single WRONG.
- **C.** The change to `llm_query_parser` affects value binding for the **funded**
  dataset in any measurable way.
- **D.** Making a family answerable requires broadening the generic sense of "movement",
  "change", "flow" or "progression" beyond the stage-token discriminator.
- **E.** Query ends up computing any part of the transition itself.
- **F.** Total production LOC exceeds ~400, or item 4 exceeds ~150 — either indicates the
  discriminator has stopped being narrow.

---

## 16. Recommendation

> ## IMPLEMENT ONLY A DEFINED SUBSET

**In scope** — the 55 questions the measured discriminator captures with zero
collisions: family A (source→destination count), family B (source→destination amount),
family C (departures, including by destination), the directional part of family D
(arrivals from a named stage), family E (stayers), family F (amendments on stayers, with
the fixture caveat), and the explicit "stage movement" summary phrasings in family L.

**Explicitly out of scope for this implementation:**

- **Family J (largest / most material).** Requires ranking in the Query layer, which the
  brief forbids and which would collide with `concentration_analysis`.
- **Family K (period comparison of movement).** A second-order analytic on a first-order
  one that does not exist yet. Revisit only once the matrix is shipped and stable.
- **The completion / withdrawal wording without an explicit transition** ("How many cases
  completed?", "How many cases were withdrawn?"). These are the pipeline-vs-funded
  dataset-interpretation problem, not stage movement, and one of them currently answers
  *640 loans · £172.1MM* from the funded book. **That is a live defect worth fixing on
  its own, separately, and it should not be smuggled into a stage-movement change.**

**Why not "safe to implement narrowly" without qualification:** because item 4 in §13
sits inside the file that binds category values for every dataset, and because the
measured evidence bounds the risk without eliminating it. And why not "defer": because
the current behaviour is not a neutral absence. It answers 32% of these questions with a
figure that looks like an answer, is not, and carries no warning — including a forward
forecast presented as historical fact and a funded-book total presented as a pipeline
withdrawal. Leaving that in place is not the conservative option.

---

## 17. Appendix — full results

### Verdicts by family

| Family | n | CORRECT | CORRECTLY DECLINED | DECLINED BUT ANSWERABLE | WRONG |
|---|--:|--:|--:|--:|--:|
| A source→destination, count | 12 | 0 | 7 | 0 | 5 |
| B source→destination, amount | 8 | 0 | 4 | 0 | 4 |
| C departures | 7 | 0 | 6 | 0 | 1 |
| D arrivals | 6 | 0 | 4 | 0 | 2 |
| E stayers | 6 | 0 | 2 | 0 | 4 |
| F amendments on stayers | 6 | 0 | 6 | 0 | 0 |
| G completions | 9 | 1 | 5 | 1 | 2 |
| H withdrawals / terminal | 9 | 2 | 2 | 0 | 5 |
| I stage reconciliation | 7 | 0 | 6 | 0 | 1 |
| J largest / most material | 6 | 0 | 2 | 0 | 4 |
| K period comparison | 6 | 0 | 6 | 0 | 0 |
| L broad summary | 6 | 0 | 4 | 2 | 0 |
| **Total** | **88** | **3** | **54** | **3** | **28** |

### Diagnostic classification

| Classification | n |
|---|--:|
| UNSUPPORTED_CAPABILITY | 20 |
| ANSWERED_WRONG | 15 |
| RECOGNISED_NO_EXECUTION_ROUTE | 14 |
| WRONG_STAGE_BINDING | 10 |
| VOCABULARY_GAP | 9 |
| AMBIGUOUS | 6 |
| RECOGNISED_BUT_WRONG_ROUTE | 4 |
| WRONG_DATASET | 4 |
| CORRECTLY_REFUSED | 3 |
| CORRECT | 3 |

### First-failure layer

| First point at which the current path fails | n |
|---|--:|
| intent / construction recognition | 35 |
| capability routing | 12 |
| lexical / vocabulary recognition | 9 |
| stage extraction / binding | 8 |
| direction / source-destination binding | 6 |
| n/a | 6 |
| dataset selection | 4 |
| measure binding | 3 |
| time binding | 3 |
| result shaping | 2 |

### Near-neighbour route baseline

| Question | Route today | Dataset | Verdict |
|---|---|---|---|
| How has pipeline balance moved this month? | `point_in_time_engine` | pipeline | FALSE_REFUSAL |
| Show pipeline evolution. | `evolution` | pipeline | CORRECT |
| What is pipeline by stage? | `point_in_time_engine` | pipeline | CORRECT |
| Show pipeline amount by stage. | `point_in_time_engine` | pipeline | CORRECT |
| How much pipeline is currently in Offer? | `point_in_time_engine` | pipeline | CORRECT |
| What is the current pipeline amount? | `point_in_time_engine` | — | FALSE_REFUSAL |
| How much balance is in Application? | `point_in_time_engine` | pipeline | CORRECT |
| What is the Offer-stage balance? | `point_in_time_engine` | pipeline | CORRECT |
| What percentage of pipeline is in KFI? | `point_in_time_engine` | pipeline | CORRECT |
| Show weekly pipeline cases. | `evolution` | pipeline | CORRECT |
| How has the pipeline grown? | `point_in_time_engine` | — | FALSE_REFUSAL |
| Show pipeline case count over time. | `evolution` | pipeline | CORRECT |
| What is the conversion rate? | `cohort_conversion` | pipeline | CORRECT |
| How has conversion changed? | `cohort_conversion` | pipeline | CORRECT |
| What is the KFI to completion conversion rate? | `cohort_conversion` | pipeline | FALSE_REFUSAL |
| What is funded balance movement? | `period_change_analysis` | — | CORRECT |
| Why did funded balance increase? | `period_change_analysis` | — | CORRECT |
| Show movement by region. | `funded_bridge` | funded | CORRECT |
| Show balance movement by portfolio. | `funded_bridge` | funded | CORRECT |
| How many funded loans were added? | `period_change` | — | FALSE_REFUSAL |
| How did the book move last month? | `period_change_analysis` | — | CORRECT |
| What is movement in LTV? | `period_change_analysis` | — | CORRECT |
| How has regional concentration moved? | `period_change` | — | FALSE_REFUSAL |
| How has average LTV changed since last month? | `period_change_analysis` | — | CORRECT |
| Show balance by region. | `point_in_time_engine` | funded | CORRECT |
| What is the forecast funded balance? | `analytical_composition` | funded+pipeline | CORRECT |
| How much of the pipeline is expected to complete? | `point_in_time_engine` | pipeline | FALSE_REFUSAL |
| When will we reach 700 loans? | `forecast_extrapolation` | funded+pipeline | CORRECT |

### Every stage-movement question, as answered today

| Family | Question | Verdict | Diagnostic | First failure | What came back |
|---|---|---|---|---|---|
| A | How many cases moved from KFI to Application? | TRUE_REFUSAL | RECOGNISED_NO_EXECUTION_ROUTE | capability routing | I understood this as a pipeline, movement trend question, but I have not answered it: this asks how something  |
| A | How many KFI cases progressed to Application? | TRUE_REFUSAL | RECOGNISED_NO_EXECUTION_ROUTE | capability routing | I understood this as a pipeline, movement trend question, but I have not answered it: this asks how something  |
| A | How many cases went from Application to Offer? | WRONG | WRONG_STAGE_BINDING | direction / source-destination binding | 1 loans · Current Outstanding Balance: £500K. |
| A | How many applications became offers? | WRONG | ANSWERED_WRONG | intent / construction recognition | 1 loans · Current Outstanding Balance: £500K. |
| A | How many offers moved to Completion? | WRONG | RECOGNISED_BUT_WRONG_ROUTE | capability routing | Offer stage pipeline is £1.3m across 3 case(s) as at 2026-05-29. Expected completion amount from pipeline case |
| A | How many cases progressed from Offer to Completion? | TRUE_REFUSAL | WRONG_STAGE_BINDING | stage extraction / binding | No loans in this book match that filter ('offer to completion'), so there is nothing to calculate. I have not  |
| A | What number of cases moved KFI to Application? | TRUE_REFUSAL | RECOGNISED_NO_EXECUTION_ROUTE | capability routing | I understood this as a pipeline, movement trend question, but I have not answered it: this asks how something  |
| A | Show case movement from Application into Offer. | TRUE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| A | Cases progressing from KFI to Application this period? | TRUE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| A | How many pipeline cases advanced from Offer to Completion? | TRUE_REFUSAL | WRONG_STAGE_BINDING | stage extraction / binding | No loans in this book match that filter ('offer to completion'), so there is nothing to calculate. I have not  |
| A | How many deals transitioned out of KFI into Application? | WRONG | WRONG_STAGE_BINDING | direction / source-destination binding | 1 loans · Current Outstanding Balance: £600K. |
| A | How many cases flowed from Application to Offer between the last two extracts? | WRONG | WRONG_STAGE_BINDING | direction / source-destination binding | 1 loans · Current Outstanding Balance: £500K. |
| B | How much balance moved from KFI to Application? | TRUE_REFUSAL | RECOGNISED_NO_EXECUTION_ROUTE | capability routing | I understood this as a pipeline, movement trend question, but I have not answered it: this asks how something  |
| B | What value progressed from Application to Offer? | TRUE_REFUSAL | AMBIGUOUS | measure binding | 'value' could mean more than one governed measure in this dataset (Balance or Valuation). Say which one and I  |
| B | How much pipeline moved from Offer to Completion? | TRUE_REFUSAL | WRONG_STAGE_BINDING | stage extraction / binding | No loans in this book match that filter ('offer to completion'), so there is nothing to calculate. I have not  |
| B | What was the balance of cases moving from Application into Offer? | TRUE_REFUSAL | RECOGNISED_NO_EXECUTION_ROUTE | capability routing | I understood this as a pipeline, movement trend question, but I have not answered it: this asks how something  |
| B | How much exposure transitioned from KFI to Application? | WRONG | WRONG_STAGE_BINDING | direction / source-destination binding | Balance: £600K · 1 loans. |
| B | What amount moved from Offer to Completion? | WRONG | RECOGNISED_BUT_WRONG_ROUTE | capability routing | Offer stage pipeline is £1.3m across 3 case(s) as at 2026-05-29. Expected completion amount from pipeline case |
| B | Show balance transferred from Application to Offer. | WRONG | WRONG_STAGE_BINDING | direction / source-destination binding | Balance: £500K · 1 loans. |
| B | What exposure went from Offer to Completion this period? | WRONG | RECOGNISED_BUT_WRONG_ROUTE | capability routing | Offer stage pipeline is £1.3m across 3 case(s) as at 2026-05-29. Expected completion amount from pipeline case |
| C | Where did Offer-stage departures go? | TRUE_REFUSAL | VOCABULARY_GAP | lexical / vocabulary recognition | 'departures' is not a governed measure in this dataset; no substitute was used. I haven't computed an answer,  |
| C | Where did cases leaving Application move to? | TRUE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| C | What happened to cases that left KFI? | TRUE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| C | Break down departures from Offer by destination. | TRUE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| C | How many cases left Application and where did they go? | WRONG | ANSWERED_WRONG | intent / construction recognition | 1 loans · Current Outstanding Balance: £500K. |
| C | What balance exited KFI by destination? | TRUE_REFUSAL | RECOGNISED_NO_EXECUTION_ROUTE | result shaping | I could not build a governed query for this question: bar chart requires a dimension (or x). No substitute fig |
| C | Show destination mix for cases departing Offer. | TRUE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| D | How many cases arrived in Application? | WRONG | ANSWERED_WRONG | intent / construction recognition | 1 loans · Current Outstanding Balance: £500K. |
| D | Where did new Offer cases come from? | TRUE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| D | What balance moved into Offer? | TRUE_REFUSAL | RECOGNISED_NO_EXECUTION_ROUTE | capability routing | I understood this as a pipeline, movement trend question, but I have not answered it: this asks how something  |
| D | Show arrivals into Completion by prior stage. | TRUE_REFUSAL | VOCABULARY_GAP | lexical / vocabulary recognition | 'arrivals completion' is not a governed measure in this dataset; no substitute was used. I haven't computed an |
| D | How much entered Application during the period? | WRONG | ANSWERED_WRONG | intent / construction recognition | 1 loans · Current Outstanding Balance: £500K. |
| D | Which stages contributed cases into Offer? | TRUE_REFUSAL | RECOGNISED_NO_EXECUTION_ROUTE | stage extraction / binding | I understood that you asked for stages, but that could not be applied to the calculation (stages — this answer |
| E | How many cases stayed in Application? | WRONG | ANSWERED_WRONG | intent / construction recognition | 1 loans · Current Outstanding Balance: £500K. |
| E | How much balance remained in Offer? | WRONG | ANSWERED_WRONG | intent / construction recognition | Balance: £1.3MM · 3 loans. |
| E | What happened to cases that stayed at KFI? | TRUE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| E | Show persisting Application cases. | TRUE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| E | How many Offer cases remained at Offer? | WRONG | ANSWERED_WRONG | intent / construction recognition | 3 loans · Current Outstanding Balance: £1.3MM. |
| E | What was the balance of cases staying in Application? | WRONG | ANSWERED_WRONG | intent / construction recognition | Balance: £500K · 1 loans. |
| F | What was the amount amendment on cases that stayed in Application? | TRUE_REFUSAL | AMBIGUOUS | measure binding | 'amount' could mean more than one governed measure in this dataset (Balance or Valuation). Say which one and I |
| F | How much did persisting Offer cases change in value? | TRUE_REFUSAL | RECOGNISED_NO_EXECUTION_ROUTE | capability routing | I understood this as a pipeline, movement trend question, but I have not answered it: this asks how something  |
| F | What balance change occurred on cases remaining at KFI? | TRUE_REFUSAL | RECOGNISED_NO_EXECUTION_ROUTE | capability routing | I understood this as a pipeline, movement trend question, but I have not answered it: this asks how something  |
| F | Did Application-stage cases increase or decrease in amount? | TRUE_REFUSAL | VOCABULARY_GAP | lexical / vocabulary recognition | 'increase decrease' is not a governed measure in this dataset; no substitute was used. I haven't computed an a |
| F | Show amount movement on cases that stayed in Offer. | TRUE_REFUSAL | AMBIGUOUS | measure binding | 'amount' could mean more than one governed measure in this dataset (Balance or Valuation). Say which one and I |
| F | What was the net balance amendment for KFI stayers? | TRUE_REFUSAL | WRONG_STAGE_BINDING | stage extraction / binding | No loans in this book match that filter ('kfi stayers'), so there is nothing to calculate. I have not returned |
| G | How many cases completed? | TRUE_REFUSAL | CORRECTLY_REFUSED | n/a | I understood this as a pipeline question, but I have not answered it: this asks about the pipeline (applicatio |
| G | How much balance completed? | TRUE_REFUSAL | CORRECTLY_REFUSED | n/a | I understood this as a pipeline question, but I have not answered it: this asks about the pipeline (applicatio |
| G | How many pipeline cases are at the Completion stage? | FALSE_REFUSAL | VOCABULARY_GAP | lexical / vocabulary recognition | 'completion' is not a governed measure in this dataset; no substitute was used. I haven't computed an answer,  |
| G | What is the balance at the Completion stage of the pipeline? | CORRECT | CORRECT | n/a | Balance: £800K · Pipeline Stage: COMPLETED · 2 loans. |
| G | How many pipeline cases reached Completion? | WRONG | ANSWERED_WRONG | intent / construction recognition | 2 loans · Current Outstanding Balance: £800K. |
| G | What value completed this period? | TRUE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| G | What was completion flow by count? | TRUE_REFUSAL | CORRECTLY_REFUSED | n/a | I understood this as a pipeline question, but I have not answered it: this asks about the pipeline (applicatio |
| G | What was completion flow by balance? | TRUE_REFUSAL | RECOGNISED_NO_EXECUTION_ROUTE | result shaping | I could not build a governed query for this question: bar chart requires a dimension (or x). No substitute fig |
| G | How many Offer cases completed? | WRONG | RECOGNISED_BUT_WRONG_ROUTE | capability routing | Offer stage pipeline is £1.3m across 3 case(s) as at 2026-05-29. Expected completion amount from pipeline case |
| H | How many pipeline cases are withdrawn? | CORRECT | CORRECT | n/a | 1 loans · Current Outstanding Balance: £400K. |
| H | What is the withdrawn balance in the pipeline? | CORRECT | CORRECT | n/a | Balance: £400K · 1 loans. |
| H | How many cases were withdrawn? | WRONG | WRONG_DATASET | dataset selection | 640 loans · Current Outstanding Balance: £172.1MM. |
| H | How much pipeline was withdrawn? | WRONG | ANSWERED_WRONG | intent / construction recognition | Balance: £400K · 1 loans. |
| H | Where did pipeline drop out? | TRUE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| H | What stage had the most withdrawals? | TRUE_REFUSAL | VOCABULARY_GAP | lexical / vocabulary recognition | 'withdrawals' is not a governed measure in this dataset; no substitute was used. I haven't computed an answer, |
| H | How many Offer cases were withdrawn? | WRONG | WRONG_STAGE_BINDING | direction / source-destination binding | 3 loans · Current Outstanding Balance: £1.3MM. |
| H | What balance left the pipeline without completing? | WRONG | ANSWERED_WRONG | intent / construction recognition | Balance: £3.6MM · 8 loans. |
| H | Where was the greatest pipeline attrition? | WRONG | ANSWERED_WRONG | intent / construction recognition | Here is the result for your query, covering 8 groups. |
| I | Reconcile Application stage this period. | TRUE_REFUSAL | VOCABULARY_GAP | lexical / vocabulary recognition | 'reconcile' is not a governed measure in this dataset; no substitute was used. I haven't computed an answer, a |
| I | Explain the movement in Offer-stage cases. | TRUE_REFUSAL | VOCABULARY_GAP | lexical / vocabulary recognition | 'explain movement' is not a governed measure in this dataset; no substitute was used. I haven't computed an an |
| I | Why did KFI cases change from opening to closing? | TRUE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| I | Show opening, arrivals, departures and closing for Application. | TRUE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| I | Reconcile Offer balance between the two extracts. | WRONG | ANSWERED_WRONG | intent / construction recognition | Balance: £1.3MM · 3 loans. |
| I | What drove the change in Application-stage balance? | TRUE_REFUSAL | RECOGNISED_NO_EXECUTION_ROUTE | stage extraction / binding | I understood that you asked for Application, but that could not be applied to the calculation (Application — t |
| I | Reconcile Application cases from opening to closing. | TRUE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| J | Which stage had the most movement? | TRUE_REFUSAL | AMBIGUOUS | time binding | I can report what changed, but this question names no period to compare over, and I have not chosen one for yo |
| J | What was the largest stage transition? | TRUE_REFUSAL | WRONG_DATASET | dataset selection | 'Pipeline Stage' is not available in this dataset. This book does not report it, so the question cannot be ans |
| J | Where did the most cases progress? | WRONG | WRONG_DATASET | dataset selection | Here is the bar for your query, covering 7 groups. |
| J | Which transition moved the most balance? | WRONG | ANSWERED_WRONG | intent / construction recognition | Between 31 May 2026 and 30 June 2026, 1 of 1 governed metrics could be compared across both snapshots. Largest |
| J | Where was pipeline attrition greatest? | WRONG | ANSWERED_WRONG | intent / construction recognition | Here is the result for your query, covering 8 groups. |
| J | What was the biggest departure destination? | WRONG | WRONG_DATASET | dataset selection | Here is the result for your query, covering 10 groups. |
| K | Compare stage movement with the prior period. | TRUE_REFUSAL | RECOGNISED_NO_EXECUTION_ROUTE | stage extraction / binding | I understood that you asked for stage, but that could not be applied to the calculation (stage — field is unav |
| K | Did more cases move from Application to Offer this period? | TRUE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| K | How has KFI-to-Application movement changed? | TRUE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| K | Was completion flow higher than last period? | TRUE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| K | Compare Offer departures with the previous reporting period. | TRUE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| K | Has pipeline progression improved month on month? | TRUE_REFUSAL | RECOGNISED_NO_EXECUTION_ROUTE | stage extraction / binding | I understood that you asked for month, but that could not be applied to the calculation (month — this answer i |
| L | Summarise pipeline stage movement this period. | TRUE_REFUSAL | VOCABULARY_GAP | lexical / vocabulary recognition | 'movement' is not a governed measure in this dataset; no substitute was used. I haven't computed an answer, an |
| L | Give me the stage movement summary. | TRUE_REFUSAL | AMBIGUOUS | time binding | I can report what changed, but this question names no period to compare over, and I have not chosen one for yo |
| L | What changed in pipeline stages? | TRUE_REFUSAL | VOCABULARY_GAP | lexical / vocabulary recognition | 'changed' is not a governed measure in this dataset; no substitute was used. I haven't computed an answer, and |
| L | What happened in the pipeline between the last two extracts? | TRUE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| L | Show pipeline progression. | FALSE_REFUSAL | UNSUPPORTED_CAPABILITY | intent / construction recognition | I couldn't map this question to a governed analytic, so I haven't computed an answer (nothing was guessed). Tr |
| L | How did cases move through the funnel? | FALSE_REFUSAL | AMBIGUOUS | time binding | I can report what changed, but this question names no period to compare over, and I have not chosen one for yo |
