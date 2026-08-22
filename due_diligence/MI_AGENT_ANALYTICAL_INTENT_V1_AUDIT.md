# Analytical Intent V1 evidence pack — external review response

**Investigated at** `bbd5929`, production code unchanged since `9125e77`.
**Nothing under "product changes" below has been implemented.**

Rules observed: `nl_score.py` and the baseline run files were not touched; no
independent recomputation was moved toward a delivered figure; every new
assertion carries a provenance marker.

---

## 1. Q9.3 — £12.5m against £7.7m

### What the two figures are

**TRUTH** — recomputed in `audit/truth_pipeline.py`, standalone pandas over the
raw weekly M2L extracts, importing nothing from the forecast modules.

| | £12.5m | £7.7m |
|---|---|---|
| capability | `funded_balance_forecast` | `pipeline_completion_forecast` |
| engine | `forecast_bridge.compute_forecast_bridge` | `pipeline_contract._dimension_breakdown` |
| population | **every case in the weekly extract** — 732 | **open pipeline** — 686 |
| stage filter | none | `pipeline_stage in [KFI, APPLICATION, OFFER]` |
| gross | £76,677,094.14 | £71,924,889.05 |
| probability set | historical rates + config; **COMPLETED at 1.00** | historical rates, open stages only |
| horizon | unbounded | 3 months, `untimedExpectedAmount = 0.0` |

**They are different concepts, and they reconcile exactly** (RECORDED, from the
run file's own evidence blocks):

```
12,459,455.42  whole extract
−  7,707,250.33  open pipeline
=  4,752,205.09  settled-stage component  ← equals settledStageComponent exactly
                                          ← and equals gross(all) − gross(open) exactly
landing profile 3,945,480.97 + 1,928,752.13 + 1,833,017.23 = 7,707,250.33 exactly
```

So: **not competing forecasts of the same quantity**, and **not horizon
truncation** — the landing profile is complete for the open pipeline.

### But the £12.5m figure is itself wrong as an addend to the funded balance

This is the part the review's dichotomy did not anticipate, and it is a blocker.

**TRUTH** — from the raw extract: the 46 cases the open view excludes all carry
`Status = "Completed"`, gross **£4,752,205.09** exactly, and all 46 have
`Date Funds Released` populated between **2026-06-22 and 2026-06-29** — i.e.
before the funded cut-off of 2026-06-30.

**RECORDED** — from the governed config
`config/client/pipeline_expected_funding.yaml`:

```yaml
exclude_stages: [WITHDRAWN, COMPLETED]
exclude_reconciled_rows: true
stage_probabilities: { …, COMPLETED: 1.00 }
```

**RECORDED** — from the code:

* `mi_agent_api/pipeline_prep.py:403` maps `COMPLETED` to
  `pipeline_status = "funded"`. The system's own classification says these cases
  are already on the book.
* `mi_agent_api/pipeline_prep.py:501` excludes `WITHDRAWN` from weighting.
  **There is no equivalent tier for `COMPLETED`.** Those rows fall through to
  the configured-stage tier and pick up probability **1.00**.
* `mi_agent_api/pipeline_prep.py:686` — `excluded_sources` is
  `{excluded_withdrawn, missing_stage, unavailable}`. `COMPLETED` is not in it.
* `analytics/pipeline_expected_funding.py:90-95` — a **second, governed**
  implementation of the same concept **does** honour `include_stages`,
  `exclude_stages` and `exclude_reconciled_rows`. It is used by the orchestrator
  and the demo script. **The MI serving path does not use it.**

So the forecast funded balance adds, at 100% probability, cases the system
classifies as already funded, on a basis the governed config explicitly
forbids.

| | |
|---|---|
| delivered forecast balance | £1,784,930,793.81 |
| forecast balance on the governed open-pipeline basis | £1,780,178,588.72 |
| **overstatement** | **£4,752,205.09 (0.268% of the book)** |
| Alderbridge equivalent | £5,550,243.24 |

**Limit of the evidence, stated precisely.** On these fixtures the funded tape
and the pipeline extract are independently synthesised: zero of the 46 completed
cases' amounts appear in the funded tape, so this data set does **not**
demonstrate a numeric double count. What it demonstrates is that the serving
path includes a stage the governance config excludes and the code itself calls
"funded". On production data, where the pipeline and the tape describe one book,
that is a double count.

### The disclosure exists but never reaches the reader

**RECORDED** — `mi_workflows/analytical/executors.py:880` emits a warning naming
the settled component, and it is on the envelope
(`warnings[0]`: *"£4.8m of the expected completions sits on cases the pipeline
extract already shows as completed…"*). It does **not** appear in the answer
text, and the trace in §13.2 prints the answer without printing the warnings
array. A reader of either the answer or the trace cannot see it.

### Q3.1 — horizon is correct, not truncated

**RECORDED**: `months: 1`, `untimedExpectedAmount: 0.0`, single stage `OFFER`.
**TRUTH**: all 157 OFFER cases share the config lag of 30 days from the
2026-06-29 extract → 2026-07-29, one landing month. The profile is complete for
the population the question asked about.

Two observations worth carrying forward, neither a defect:

* the stage-lag model lands an entire stage's expectation in a **single month**;
  it is a point estimate presented as a profile.
* the probability applied to OFFER is **0.1693**, not the config's 0.75 — an
  empirical rate derived from 12 weekly extracts overrides config wherever
  history is sufficient. On both books all three open stages have sufficient
  history, so **`stage_probabilities` in the config is inert for the MI serving
  path** except as a fallback. Anyone tuning that file expecting the answer to
  move would be disappointed.

---

## 2. Independent verification of the pipeline forecasts

`audit/truth_pipeline.py` — standalone pandas, no import from any forecast
module, every assumption an explicit constant naming its source
(`CONFIG_STAGE_PROBABILITY`, `CONFIG_STAGE_DAYS_TO_FUND`, `ACTIVE_STAGES`,
`MIN_OBSERVATIONS = 12`, the status→stage map, the case-identity column). No
assumption had to be substituted; nothing was unrecoverable.

**TRUTH — Kestrelmoor, as at 2026-06-29** (12 weekly extracts, 732 cases tracked)

| stage | cases | gross | probability | basis | expected | lands | agrees |
|---|---:|---:|---:|---|---:|---|---|
| OFFER | 229 | £23,583,269.40 | 0.1673 | historical | £3,945,480.97 | 2026-07 | ✅ |
| APPLICATION | 182 | £19,153,447.16 | 0.1007 | historical | £1,928,752.13 | 2026-08 | ✅ |
| KFI | 275 | £29,188,172.49 | 0.0628 | historical | £1,833,017.23 | 2026-09 | ✅ |
| **open pipeline** | **686** | **£71,924,889.05** | | | **£7,707,250.33** | | ✅ |
| COMPLETED *(config says exclude)* | 46 | £4,752,205.09 | 1.0000 | config | £4,752,205.09 | 2026-06 | ✅ |
| **whole extract** | **732** | **£76,677,094.14** | | | **£12,459,455.42** | | ✅ |

**TRUTH — Alderbridge, as at 2026-06-29** (12 weekly extracts, 504 cases tracked)

| stage | cases | gross | probability | basis | expected | lands | agrees |
|---|---:|---:|---:|---|---:|---|---|
| OFFER | 157 | £29,407,505.06 | 0.1693 | historical | £4,978,690.61 | 2026-07 | ✅ |
| open pipeline | 472 | £88,808,167.37 | | | £9,625,160.91 | | ✅ |
| COMPLETED *(config says exclude)* | 32 | £5,550,243.24 | 1.0000 | config | £5,550,243.24 | 2026-06 | ✅ |

**Every figure agrees to the penny, including the empirical rates.** The
arithmetic is not in question anywhere; the population feeding the headline is.

**A modelling caveat surfaced by the recomputation.** The empirical rate for a
stage is *(cases ever seen at that stage that ever completed) ÷ (cases ever seen
at that stage)*. `completed` is therefore the same count for all three stages —
every completion passed through all three — and the rates are **right-censored**:
recent cases have not had time to complete. Expected completions are a floor,
not a central estimate. The codebase acknowledges this for KFI conversion
elsewhere; the composed forecast answers do not say it.

---

## 3. The rolling-cohort narrative

**TRUTH** — membership recomputed from the fixture tapes by `loan_identifier`:

| population | 2026-04-30 | 2026-06-30 | in both | left | joined |
|---|---:|---:|---:|---:|---:|
| **NEW (mob ≤ 1)** — Q1.1, Q1.4 | 143 | 115 | **0** | 143 | 115 |
| RECENT (mob ≤ 3) — Q1.2, Q1.3 | 367 | 258 | 143 | 224 | 115 |
| Front Book — Q8.3/seasoning | 2,626 | 3,020 | 2,492 | 134 | 528 |
| Back Book — Q8.3/seasoning | 9,101 | 9,235 | 9,101 | 0 | 134 |
| *control:* whole book | 10,996 | 11,035 | 10,920 | 76 | 115 |

The review understated it. For Q1.1 the two periods share **not one loan**. The
delivered sentence — *"New lending (last 1 month), 115 loans: £21.4m → £18.3m
(−£3.2m)"* — reports a £3.2m decline in a population that has no members in
common between the two dates. Nothing ran off; two different monthly cohorts
were measured. Front/Back is the same effect in milder form: the 134 loans that
left the front book are exactly the 134 that joined the back book.

The control line matters: for a population whose membership is *not*
time-relative, movement narration is sound.

### Blast radius — enumerated, not estimated

**RECORDED** — only two plan builders emit a movement over a funded population:

| plan intent | capability | affected when |
|---|---|---|
| `origination_profile_change` (`planner.py:341`) | `period_movement` | **always** — its population is always a lending window |
| `population_movement_comparison` (`planner.py:550,554`) | `period_movement` ×2 | only when the pair is a lending window / seasoning segment |

**Population kinds** — `KIND_SEASONING` (all four windows: `new`, `recent`,
`front_book`, `back_book`) is time-relative. `KIND_PROVENANCE`,
`KIND_DIMENSION_VALUE` and `KIND_TOTAL` are not. So the detection rule is
mechanical: *finding kind is MOVEMENT and population kind is `KIND_SEASONING`*.

Reachable beyond the traced ten — *"Show the month on month movement for the
back book"* takes the same path (verified, `analytical_composition`, `ok=True`).

**This is a product defect in answer composition, not a documentation defect.**
It is not repaired by rewording the trace.

---

## 4. Semantic truth control

`audit/expected_semantics.yaml` — expected family, operation, population and
period for each traced question, derived from the question text and the governed
definitions, each with its derivation written out. `audit/check_semantics.py`
compares.

**The file discloses a weakness in itself**: the brief asks that it be written
before reading the agent's resolutions, and it was not — items 1–3 required
reading them first. That is stated in the file header rather than glossed. The
mitigations are that every expectation carries a derivation checkable against
the question text alone, and that the fixture records four tensions and one
outright disagreement rather than agreeing throughout.

**Result: family and operation agree on all ten. Population disagrees on one.**

> **Q9.3 — SEMANTICS DISAGREES.**
> Expected addend population: `pipeline_stage in [KFI, APPLICATION, OFFER]`.
> Actual addend population: *"every case in the weekly extract"*.
> Expected addend £7,707,250.33; actual £12,459,455.42.

This is the control earning its place: the numeric TRUTH column recomputes using
the agent's own population and therefore **agrees** on this run. Only an
expectation written from the question could catch it. A first version of the
checker passed Q9.3 on a substring coincidence — the composed answer *also*
reports an open-pipeline figure — so the check was tightened to test the
population of the finding that feeds the headline. A control that passes when it
should not is worse than none.

**Tensions recorded, not resolved:**

| question | tension |
|---|---|
| Q1.1 | *"our new lending"* → NEW = L1M by the governed ruling, but the question says *"over the last few months"*. A one-month population answering a several-month question is a narrow reading. The convention is defensible; the tension is real. |
| Q1.2 | *"originating"* maps to RECENT (L3M); the adverb *"now"* argues for NEW (L1M). |
| Q7.2 | *"older loans"* → BACK BOOK (>12m) vs *"originated recently"* → RECENT (≤3m). **These are not complementary**: loans originated 4–12 months ago are in neither side, and the answer does not say so. |
| Q8.3/seasoning | membership is time-relative — see item 3. |

---

## 5. Reproducibility

**Done.** `due_diligence/evidence/analytical_intent_v1/MANIFEST.json` — 52
artefacts, none missing, each with SHA-256, role and group:

| group | count | why it is in the manifest |
|---|---:|---|
| `fixture:funded:*` | 6 | input to **both** the agent and the TRUTH column |
| `fixture:pipeline:*` | 24 | the 12 weekly extracts per book |
| `code+config` | 9 | `intent.py`, planner, executors, registry, `seasoning.py`, buckets, expected-funding, BSR, field registry |
| `measurement` | 7 | bank, harness, frozen scorer, reconciler, comparison, tracer |
| `run-file` | 4 | uncompressed SHA-256 of all 752 responses |
| `audit` | 2 | this review's recomputations |

**Provenance corrected — the report is currently wrong about this.** The run
files were written 22:23:35–22:49:40. The last commit before that window is
`044d13b` (22:15:31); production code was last changed by `9125e77` (22:14:54).
`git diff --name-only 044d13b..HEAD` returns **only** paths under
`due_diligence/`. So the runs describe production code at `9125e77`, which is
identical to that at `8c9d04e` — but the pack should say so rather than assert
`8c9d04e`.

**Harness change made** (test asset, not production, not the frozen scorer):
`nl_harness.py` now records `metadata.analyticalIntent` and the built plan's
calls and inputs into the run file. Step 3 of a trace becomes RECORDED instead
of DERIVED. **This takes effect on the next measurement** — the committed run
files predate it, and re-running is deliberately deferred so that any product
fix arising from item 1 or item 3 is measured once rather than twice.

---

## 6. Decomposing the headline

**Mechanical rule, applied by `audit/classify_substance.py`. It never reads
answer text.**

| class | rule |
|---|---|
| `CONTROLLED_REFUSAL` | `ok is False` |
| `INFORMATIONAL_NO_COMPUTE` | `ok is True` **and** no artifact **and** no structured finding **and** no reconciliation block |
| `SUBSTANTIVE_CALCULATED` | `ok is True` **and** at least one artifact or structured finding |
| `OTHER` | anything else |

| | V1 | baseline |
|---|---:|---:|
| SUBSTANTIVE_CALCULATED | 618 (82.2%) | 570 (75.8%) |
| INFORMATIONAL_NO_COMPUTE | 54 (7.2%) | 42 (5.6%) |
| CONTROLLED_REFUSAL | 80 (10.6%) | 140 (18.6%) |
| OTHER | 0 | 0 |
| **headline correctness** | **89.4%** | **56.5%** |
| **substantive correctness** | **82.2%** | **50.9%** |

**Per book and arm** — the whole effect is one book:

| book / arm | headline | substantive |
|---|---:|---:|
| alderbridge / production | 89.9% | 89.9% |
| alderbridge / forced-LLM | 89.4% | 89.4% |
| kestrelmoor / production | 88.8% | **74.5%** |
| kestrelmoor / forced-LLM | 89.4% | **75.0%** |

**The corrected headline is 82.2% substantive-calculated correctness, against a
baseline of 50.9%** — the gap between it and 89.4% is 54 runs on Kestrelmoor's
Q5 and Q6, where the portfolio has no Schedule 8 limits document and the limits
route correctly reports that instead of computing.

Two things this decomposition shows that the single figure hid. The improvement
is +31.3pp substantive against +32.9pp headline, so it is **not** an artefact of
informational responses. And V1 has 12 *more* informational responses than the
baseline — because Q6.2/Q6.3 now reach the limits route and report the gap,
where the baseline sent them to the generic executor and got a confident wrong
answer. Some of that 7.2% is a wrong answer converted into an honest one.

---

# BLOCKERS

**B1 — the forecast funded balance is computed on a basis the governed config
forbids.** Item 1. `£1.78bn` overstates by £4,752,205.09 on Kestrelmoor and
£5,550,243.24 on Alderbridge, by adding cases the extract shows as COMPLETED at
probability 1.00 and the code labels `pipeline_status = "funded"`. Which figure
is wrong: **£12.5m is wrong as an addend to the funded balance**; £7.7m is the
correct forward expectation. A second, governed implementation that honours the
exclusion already exists and is not used by the serving path.
*Do not circulate the pack with §13.2's Q9.3 trace presented as verified.*

**B2 — the rolling-cohort narrative reports movement in populations that do not
persist.** Item 3. Q1.1's two periods share zero loans. Affects every
`origination_profile_change` answer and every seasoning-pair
`population_movement_comparison`, not only the traced ten.

**B3 — the settled-stage disclosure is suppressed.** The warning exists on the
envelope and reaches neither the answer prose nor the trace. Cheapest of the
three to fix and the one that would have let a reader catch B1 unaided.

---

# PROPOSED CHANGES

## (a) Documentation — can be made now, no product behaviour touched

| # | change | scope |
|---|---|---|
| D1 | Add the reconciling bridge to §13.2's Q9.3 trace: £12.5m − £4.75m settled = £7.7m open, with both populations named. | one trace |
| D2 | Print the `warnings` array in trace step 8. The settled-stage disclosure is on the envelope and the trace hides it. | tracer, all ten |
| D3 | Add the item-2 TRUTH rows for Q3.1 and Q9.3 to those two traces. | two traces |
| D4 | Add the `semantics agrees` column and the tension notes from item 4. | all ten |
| D5 | Correct the provenance line: runs produced at tree `044d13b`, production code `9125e77`; state the diff-verified equivalence to `8c9d04e`. | front matter, §13.3 |
| D6 | Replace §12's single headline with the item-6 decomposition, baseline included. | §1, §12 |
| D7 | Record the two modelling caveats: right-censored empirical rates; config `stage_probabilities` inert wherever history is sufficient. | §13 |
| D8 | Mark the Q9.3 trace **not verified** pending B1. | §13.2 |

## (b) Product — approval required, nothing implemented

| # | change | blast radius |
|---|---|---|
| **P1** | Honour `exclude_stages` in the serving path: give `COMPLETED` an exclusion tier in `pipeline_prep._derive_probabilities_and_amounts` alongside `WITHDRAWN`, and add it to `excluded_sources`. | Changes `weighted_expected_funded_amount` for **every** consumer of the bridge — MI answers, PPTX metric resolver, notifications, evolution. Forecast balances fall by the settled component. Needs a decision on whether `forecast_loan_count` drops the settled cases too. **The alternative — converge the serving path onto `analytics/pipeline_expected_funding.py` — is the better long-term fix and a larger change.** |
| **P2** | Surface the settled-stage warning in the composed answer prose, not only the envelope. | Narrative only; no figure changes. Smallest fix, largest reader benefit. |
| **P3** | Relabel both forecast figures so scope is explicit on their face. Proposed wording: *"Expected completions from the open pipeline (KFI, Application, Offer; 686 cases): £7.7m, expected to land 2026-07 £3.9m; 2026-08 £1.9m; 2026-09 £1.8m."* and *"The extract also carries 46 cases already at Completed stage worth £4.8m, which are excluded from the forward expectation."* | Narrative only. Pairs with P1; without P1 the labels describe a figure that should not be in the headline at all. |
| **P4** | Cohort-aware movement composition. Where `population.kind == KIND_SEASONING`, state both period populations and label the result a cohort comparison. Where identifiers permit, decompose into joined / left / persisted. Proposed wording: *"New lending (last 1 month): 143 loans £21.4m at 2026-04-30 against 115 loans £18.3m at 2026-06-30. These are different cohorts — no loan is in both — so this is a comparison of two months' originations, not a movement in a single population."* | `narrative.py` plus a `membership` field on the movement finding. Two plan intents. Changes the prose of every seasoning movement answer; changes no figure. |
| **P5** | Disclose the non-complementary pair in Q7.2-shaped comparisons (BACK vs RECENT leaves 4–12 months in neither side). | Narrative only. |
| **P6** | Author semantic expectations at bank-definition time so the item-4 control is genuinely blind. | Test assets only. |

**Recommended order: P2 (disclosure, no figures move) → P1 (the defect) → P3
(labels, now describing a correct figure) → P4 → P5 → P6.**

Re-measurement of the 752-run bank is required after P1 or P4, since both change
answer content. One re-run should cover both.
