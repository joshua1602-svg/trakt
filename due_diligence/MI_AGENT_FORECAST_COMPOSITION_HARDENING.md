# MI Query Agent — Forecast & Composition Hardening

Closing the product defects found by the Analytical Intent V1 audit, without
widening MI capability. Three tranches, worked strictly in order so that every
changed answer is attributable to exactly one of them.

**Status: Tranche A complete and recorded. Tranche B in progress. Tranche C not
started.**

---

## 1. Executive verdict

*(completed after Tranche B)*

### The qualification that governs how every reconciliation number in this pack should be read

**6,856 / 6,856 findings reconciled proves ARITHMETIC fidelity against the
population the agent EXECUTED. It does not prove that population was the
semantically correct one.**

Those are two different claims and only the first was ever measured by
reconciliation. An answer can be arithmetically perfect over the wrong rows —
the V1 audit found exactly that in Q9.3, where every figure reconciled to the
penny while the population feeding the headline forecast included 46 cases the
governed config excludes. The frozen expectation file (§2) is the separate
control for the second claim, and it is reported separately in §11.

## 2. Baseline

| | |
|---|---|
| Branch | `claude/mi-analytical-capability-layer-vlkjfw` |
| Pre-work commit | `49e00b5` |
| Accepted V1 evidence | 752-run bank; unsafe 187 → 0; CORRECT 56.5% → 89.4%; substantive CORRECT 50.9% → 82.2%; 6,856/6,856 reconciled; full suite 9,061 passed |

### 2.1 The frozen expectation file

`due_diligence/evidence/forecast_composition_hardening/frozen_expectations.yaml`
— 44 expectations, one per bank variation, authored from the question text and
the governed definitions alone and hashed into the manifest **before any code
change in this sprint**. It is not edited for the duration; a disagreement
between it and agent behaviour is reported as a finding, never reconciled by
amending the file.

It records four rolling populations, one expected refusal (Q4.2 — no governed
count run-rate exists), and three tensions where the question genuinely supports
a second reading (Q1.1, Q1.2, Q7.2). Prior exposure on ten of the forty-four is
disclosed in the file header: the control is strongest on the other thirty-four.

### 2.2 The pinned comparator, and a provenance error corrected

The 56.5% → 89.4% claim was previously unverifiable by a third party, because
the manifest pinned this run's artefacts and not the comparator's. Now hashed:
the four baseline run files, the bank, and the harness revision that produced
them.

**The harness divergence is real and is recorded rather than papered over.** The
manifest listed `nl_harness.py` at 7,074 bytes / `131071ad…` beside run files
that were in fact produced by the 6,285-byte / `de059ef3…` revision — the audit
had modified the harness and the manifest hashed the current file. Both the
baseline and the V1 run files were produced by `de059ef3…`, which is now pinned
at `evidence/analytical_intent_v1/baseline/nl_harness_that_produced_runs.py`.
The newer revision has produced no run file yet; re-measurement is deferred so
one re-run covers both it and Tranche B.

**Standing rule adopted:** harness hash and run-file hash move together. Every
run file in the manifest names its producing harness, and
`due_diligence/evidence/verify_manifest.py` exits non-zero if a run file names a
harness the manifest does not pin.

## 3. Tranche A — generic composition contracts

No calculation changed. Every value these contracts print was already in the
structured findings.

**A1 — scope on the finding.** `PopulationRef` gained `rows_prior` (membership at
the start of a compared period) and `time_relative` (whether membership is set by
a months-on-book window or seasoning segment). Both were **already computed and
discarded**: the per-snapshot row counts existed on the narrowed frames, the
population kind on the spec. No second semantic model, no recalculation.

**A2 — competing scopes are refused, not explained.** Two findings naming the
same measure over different populations with different values are one question
answered twice. The narrator declines to print either, names both scopes with
their case counts, and says why. It does not adjudicate: which population is
correct is a governance question, not a presentation one.

Two design points matter for the constraint in the brief:

* the test is on **values**, not labels — two scopes agreeing on a figure are the
  same quantity described twice and are left alone. This is why the contract will
  still hold unchanged after Tranche B removes the contaminated figure.
* a `KIND_COMPARISON` finding pairing two populations **is** a reconciliation, so
  a question that deliberately sets two populations against each other keeps both
  numbers. A first implementation lacked this and wrongly flagged Q8's
  direct-vs-acquired pair; the fix keys on structure that already existed.

Nothing in A2 asserts that COMPLETED cases belong in forward completions. It
refuses to choose.

**A3 — rolling cohorts are named as such.** Where a movement compares a
time-relative population whose membership changed, the answer states both
populations with their dates and calls the result a cohort comparison:

> New lending (last 1 month): 143 loans at 2026-04-30 against 115 loans at
> 2026-06-30. Current Outstanding Balance £21.4m against £18.3m … These are
> rolling cohorts — membership is set by the origination window, so a loan joins
> or leaves it with the passage of time. This is a comparison of two cohorts, not
> movement within one population.

The discriminator is the population KIND, not merely a changed row count:
provenance counts also move between snapshots (5,302 → 5,612) but only because
loans are originated, which reads correctly as book movement.

**A4 — decomposition.** Entries/exits decomposition was **not** built. Row counts
at both dates are disclosed because the architecture already computed them;
identifying which loans joined or left would be a new cohort engine, which the
brief puts out of scope. The answer discloses that the populations differ.

## 4. Proof that Tranche A was numerically neutral

Whole 44-variation bank, run deterministically against both checkouts, comparing
the FIGURES in each answer rather than the prose. Script:
`evidence/forecast_composition_hardening/neutrality2.py`; output alongside it.

| | |
|---|---|
| Questions compared | 44 |
| Figure sets identical | 33 |
| **Values changed** | **0** |
| **Figures printed that no finding holds** | **0** |
| **TRANCHE A NUMERICAL NEUTRALITY** | **PASS** |

Every one of the 11 differences falls into an intended category:

| category | count | figures |
|---|---|---|
| withheld by A2 refusal | 6 | £15.2m, £9.6m |
| delta not narrated (A3) | 20 | −£3.2m, −£13.5m, +£43.1m, −£10.6m, ±pp deltas |
| newly printed, already held by a finding | 18 | 143, 367, 472, 504, 1,282, 9,714 |

**Two judgement calls inside A, stated rather than buried.**

*A3 withholds the delta.* For a rolling cohort the answer prints "£21.4m against
£18.3m" and not "(−£3.2m)". The finding still holds `change`; the narrator
declines to narrate it. For a cohort pair sharing zero loans the delta is the
most misleading figure in the sentence, and the brief's own A3 example omits it.
It is nonetheless a withdrawal of a previously printed figure.

*Q9's dependent figure is untouched.* The answer still prints "Forecast funded
balance: £1.98bn", which is built on the £15.2m addend A2 refuses. Suppressing a
component while printing its consequence is an odd state, and it is deliberate:
correcting that figure is Tranche B's job, and doing it in A would have made the
changed number unattributable to one tranche.

### 4.1 Tranche A regression gates

| gate | result |
|---|---|
| analytical layer, intent boundary, P1J-1, P1L, fabricated-population, golden bank, P1I, P1M, P1N, 252-question calibration, all mi_agent_api tests | **1,989 passed, 13 xfailed, 0 failed** |
| 30-question simple-MI bank | **0 of 30 changed** |

## 5. Tranche B — one authoritative pipeline population

### 5.1 The defect, stated as a defect

`config/client/pipeline_expected_funding.yaml` has always carried:

```yaml
include_stages: [KFI, APPLICATION, OFFER]
exclude_stages: [WITHDRAWN, COMPLETED]
stage_probabilities: {KFI: 0.20, APPLICATION: 0.45, OFFER: 0.75, COMPLETED: 1.00}
```

The serving path honoured half of the exclusion. `pipeline_prep.py` excluded
WITHDRAWN by a hard-coded literal test; COMPLETED matched no exclusion, fell
through to the configured-probability tier, and was weighted at **1.00** — full
value, certainty — into the **forward** expectation. The same module classifies a
COMPLETED case as `pipeline_status = "funded"`. The forecast of what is still to
come was therefore adding, at certainty, cases the system already knew were on
the book. The current funded balance already contains them.

This is a product defect, not a documentation defect: no wording could make the
number right.

### 5.2 What changed

| # | change | file |
|---|---|---|
| B1 | exclusion set read from the config that declares it, replacing the hard-coded WITHDRAWN literal; the row-level source names **which** stage excluded the row (`excluded_withdrawn`, `excluded_completed`) so a receipt can say why | `mi_agent_api/pipeline_prep.py` |
| B1 | `completion_probability_summary` matches excluded rows on the `excluded_` prefix, so adding a stage to `exclude_stages` needs no code change | `mi_agent_api/pipeline_prep.py` |
| B2 | forward **count** now describes the same population as the forward **amount**: `forecastLoanCount = fundedLoanCount + eligibleCaseCount`, with `eligibleCaseCount = pipelineCaseCount − excludedCaseCount` published on the bridge | `mi_agent_api/forecast_bridge.py` |
| B3 | the forward finding carries a `PopulationRef` labelled "the open pipeline", and the answer states the exclusion instead of the stale warning that claimed the excluded cases were *inside* the expectation | `mi_workflows/analytical/executors.py` |

Historical calibration is untouched. `pipeline_history` reads the raw weekly
extracts directly and needs COMPLETED observations to estimate a rate at all;
the exclusion governs the forward population only. This is stated in the
docstring of `_excluded_stages` so the next reader does not "fix" it.

**B2 goes one clause beyond the brief's literal wording, and that is flagged
rather than absorbed.** The brief named the amount. Leaving the count alone would
have shipped a forecast whose two halves described different populations — the
book grows by 32 cases whose balance was deliberately excluded. Both halves were
moved; if the intent was amount-only, B2 is the line to revert.

### 5.3 What was deliberately not consolidated

There are still two forward findings carrying the same number under two labels
("Expected completions from the open pipeline", "Expected completion amount from
the open pipeline"). Before Tranche B they carried **different** numbers, which
is what made them competing scopes. They now agree, so A2's contract — which
tests values, not labels — correctly stops refusing without being touched, and
the narrator's dedup prints the figure once. Collapsing the two findings into one
is a structural change to the forecast capability with no effect on any delivered
figure, so it was left for a maintenance pass rather than smuggled into a
correctness tranche.

## 6. What Tranche B changed numerically

### 6.1 Prep layer, both books, as-of extract 2026-06-29

RECORDED. "Before" is the pre-Tranche-B tier reproduced in-process by restricting
the exclusion set to WITHDRAWN — the one thing B1 changed — with no production
file edited (`tb_prep_before.py`).

| | Alderbridge before | Alderbridge after | Kestrelmoor before | Kestrelmoor after |
|---|---|---|---|---|
| probability basis | `mixed_historical_and_config` | `historical_observed` | `mixed_historical_and_config` | `historical_observed` |
| rows on `configured_stage_rate` | 32 (£5,550,243.24) | **0** | 46 (£4,752,205.09) | **0** |
| rows on `excluded_completed` | 0 | **32 (£5,550,243.24)** | 0 | **46 (£4,752,205.09)** |
| gross weighted over | £94,358,410.61 | £88,808,167.37 | £76,677,094.14 | £71,924,889.05 |
| blended conversion | 0.1608 | **0.1084** | 0.1625 | **0.1072** |

Neither book's latest extract carries a WITHDRAWN case, so on this data the
entire effect of B1 is the COMPLETED exclusion.

### 6.2 Delivered figures, Q9 on both books

RECORDED from the answer and its findings.

| | Alderbridge before | Alderbridge after | Kestrelmoor before | Kestrelmoor after |
|---|---|---|---|---|
| expected completions | £15,175,404.15 | **£9,625,160.91** | £12,459,455.42 | **£7,707,250.33** |
| forecast funded balance | £1,980,061,662.36 | **£1,974,511,419.12** | £1,784,930,793.81 | **£1,780,178,588.72** |
| forecast loan count | 11,539 | **11,507** | 12,987 | **12,941** |
| competing forward figures in one answer | 2 (£15.2m and £9.6m) | **1** | 2 | **1** |

### 6.3 The independent check

TRUTH. `evidence/forecast_composition_hardening/truth_pipeline.py` recomputes the
whole forecast from the twelve raw weekly M2L extracts. It imports nothing from
`pipeline_prep`, `pipeline_history` or `forecast_bridge`; every assumption is an
explicit constant naming the config key or source line it came from. **It was
written during the V1 audit, before Tranche B existed, and has not been edited
since** — so it cannot have been moved toward the delivered figure.

| | independent truth | delivered | agree |
|---|---|---|---|
| Alderbridge open-pipeline expectation | 9,625,160.91 | 9,625,160.91 | ✔ |
| Alderbridge excluded component | 5,550,243.24 (32 cases) | 5,550,243.24 (32 cases) | ✔ |
| Alderbridge forecast balance | 1,964,886,258.21 + 9,625,160.91 = 1,974,511,419.12 | 1,974,511,419.12 | ✔ |
| Kestrelmoor open-pipeline expectation | 7,707,250.33 | 7,707,250.33 | ✔ |
| Kestrelmoor excluded component | 4,752,205.09 (46 cases) | 4,752,205.09 (46 cases) | ✔ |
| Kestrelmoor forecast balance | 1,772,471,338.39 + 7,707,250.33 = 1,780,178,588.72 | 1,780,178,588.72 | ✔ |
| landing months (OFFER→07, APPLICATION→08, KFI→09), both books | 4,978,690.61 / 2,332,075.32 / 2,314,394.98 and 3,945,480.97 / 1,928,752.13 / 1,833,017.23 | identical | ✔ |

Kestrelmoor's £7,707,250.33 is the figure the **frozen expectation file predicted
for Q9.3 before any code in this sprint was written**. The file was hashed into
the manifest at `49e00b5`; the number arrived at `eaa1f5e`.

### 6.4 The answer, before and after

Alderbridge Q9 — "Based on the current pipeline, what is the forecast funded
balance?"

> **Before.** Current funded balance is £1.96bn as at 2026-06-30. Gross pipeline
> in the governed extract is £94.4m as at 2026-06-29. Expected completions from
> the pipeline: **£15.2m**. Forecast funded balance: £1.98bn. Expected completion
> amount from the open pipeline: **£9.6m**. Expected to land: …

> **After.** Current funded balance is £1.96bn as at 2026-06-30. Gross pipeline in
> the governed extract is £94.4m as at 2026-06-29. Expected completions from the
> open pipeline: **£9.6m**. This excludes **32 case(s) worth £5.6m** the extract
> already shows as completed or withdrawn. Forecast funded balance: £1.97bn.
> Expected to land: …

One forward figure, the population it covers named, and what it leaves out
disclosed with its size.

### 6.5 A governance observation, recorded not acted on

After B1 the `by_source` breakdown on both books is **zero rows on
`configured_stage_rate`**. Every weighted row takes an empirical rate, so the
configured `stage_probabilities` block — 0.20 / 0.45 / 0.75 — is now entirely
inert for these two books' forward forecast. It survives only as the fallback for
a stage with insufficient history. That is the correct precedence, but it means
the shipped config values are no longer load-bearing anywhere a reader can see,
and it is exactly the calibration question Tranche C investigates. Nothing was
changed for it.

## 7–17

*(Tranche B measurement, semantic assurance, Tranche C and the launch
recommendation follow.)*
