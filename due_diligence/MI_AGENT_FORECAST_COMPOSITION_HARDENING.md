# MI Query Agent — Forecast & Composition Hardening

Closing the product defects found by the Analytical Intent V1 audit, without
widening MI capability. Three tranches, worked strictly in order so that every
changed answer is attributable to exactly one of them.

**Status: Tranche A and Tranche B complete, measured and recorded. Tranche C
investigated; nothing implemented and no approval assumed.**

---

## 1. Executive verdict

Both forecast defects the Analytical Intent V1 audit found are closed, and the
measurement that shows it separates three things the previous evidence pack
collapsed into one.

| | pre-sprint `49e00b5` | post-Tranche-B |
|---|---|---|
| unsafe outcomes over 752 runs | 0 | **0** |
| numeric findings reconciled independently | 6,856 / 6,856 | **6,856 / 6,856** |
| semantic agreement with the frozen expectation | 92.0% | **98.4%** |
| **headline forecast built from the population the question named** | **0 of 48** | **48 of 48** |
| controlled refusals | 56 | **56** |
| substantive calculated answers | 85.4% | **85.4%** |
| grade changes across 752 like-for-like runs | — | **0** |

Alderbridge's expected completions fall from £15,175,404.15 to £9,625,160.91 and
Kestrelmoor's from £12,459,455.42 to £7,707,250.33, both reconciling to the penny
against an independent recomputation that predates the change.

Two qualifications carry equal weight with the result. The production LLM parser
could not be remeasured — the supplied API key ran out of credit mid-sprint, so
the comparator is a deterministic-parse A/B (§7.1). And Tranche B made the
empirical conversion rates the *sole* basis of the forward forecast, at which
point §11 becomes load-bearing: those rates rest on a 77-day observation window
against funding lags of 30 to 90 days, and not one stage on either book has a
matured sample at its own configured lag.

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
The newer revision has since produced run files — the `run-file:llm-degraded:*`
groups — so the divergence is closed as a provenance matter. It is **not** closed
as a measurement matter: those runs are excluded from every before/after claim in
this report because the API key was exhausted while they ran (§7.1). The
comparable remeasurement is still owed.

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

## 7. How Tranche B was measured, and one comparator that had to be replaced

### 7.1 The LLM arm ran out of credit mid-sprint

RECORDED. The V1 evidence was produced with a live LLM parser: 645 of its 752
runs parsed by LLM. The remeasurement had to run on the same bank. It did not
get the same parser.

The first remeasurement of all four arms completed and was **lost to an operator
error of mine** — the harness `chdir`s to the repository root before writing, and
I passed a relative output path, so four completed 752-run arms failed at the
final write. Relaunching consumed the remaining balance on the supplied API key,
which is now exhausted:

> `{"type":"invalid_request_error","message":"Your credit balance is too low to
> access the Anthropic API."}`

The relaunched arms therefore ran with the LLM parser mostly failing over to the
deterministic parser: 145 LLM parses out of 752, against the baseline's 645.
That run is kept and published — it is real measurement — but **it cannot be
compared to the V1 baseline**, because a difference in the parser mix moves the
result more than the code does.

Demonstrated, not asserted. Running the *pre-sprint* tree on the same exhausted
key gives 92.6% correct against the post-sprint tree's 91.5% — the *later* tree
scoring lower. The two runs did not share a parser mix either (60 LLM parses
against 145), which is exactly the problem. All eight differing runs are the same
variation, Q1.3, and in every one of the eight the parser differed between the
two runs: LLM after, deterministic-fallback before. The LLM parse
produced the spec population `origination_date ge 2024-01-01`, which the
analytical route cannot apply, so P1L's population-propagation guard refused. A
pre-existing V1 guard, fired by parser variation, in a run where neither tree's
code was responsible. Full detail: `tranche_b/ab_llm_arm_degraded.txt`.

### 7.2 The comparator that replaced it

A run whose parser mix is not reproducible cannot answer "what did the code
change?". So the A/B was rerun with the LLM parser **off on both sides**, which
makes the parse a pure function of the question text and the parser mix identical
by construction — 752 of 752 deterministic on each side.

`nl_harness_det.py` is a **separate instrument**, not an edit: `nl_harness.py` is
pinned in the manifest and was not touched, so the run files it produced still
verify against the harness the manifest names. Apart from its own docstring, the
deterministic harness differs in three environment lines — `MI_AGENT_LLM_PARSER`
off, `MI_AGENT_LLM_ENABLED` 0, and the API key removed from the environment. The
bank, the repeat counts, the request payload and the capture are the same code.

What it does not do: exercise the LLM parse path. It is the controlled
comparator, not a replacement for the LLM arm. **A full-credit LLM remeasurement
is still outstanding** and is listed as such in §13.

### 7.3 The like-for-like result

Both revisions, same 752 runs, same bank, same parser. `tranche_b/ab_deterministic.txt`.

| | pre-sprint `49e00b5` | post-Tranche-B | |
|---|---|---|---|
| unsafe outcomes (INCORRECT_SUCCESSFUL / SILENT_SEMANTIC_ERROR / HARD_FAILURE) | **0** | **0** | ✔ |
| CORRECT / disclosed | 696 (92.6%) | 696 (92.6%) | unchanged |
| SUBSTANTIVE_CALCULATED | 642 (85.4%) | 642 (85.4%) | unchanged |
| CONTROLLED_REFUSAL | 56 (7.4%) | 56 (7.4%) | **did not rise** |
| grade differences across 752 matched runs | — | **0** | |
| answer-text differences | — | 196 | all attributable |
| parser mix | 752 deterministic | 752 deterministic | identical |

The 196 changed answers fall in exactly the variations the two tranches touched
and nowhere else:

| variations | runs | tranche | what changed |
|---|---|---|---|
| Q1.1–Q1.4 | 80 | A3 | rolling-cohort disclosure on new/recent lending |
| Q8.1–Q8.4 `/seasoning` | 80 | A3 | rolling-cohort disclosure on front/back book |
| Q9.2–Q9.4 | 36 | B | one forward figure over the open pipeline, exclusion disclosed |

No other variation moved a character.

Two things the table does not show, both checked directly. **A2's
competing-scopes refusal fires on zero runs after Tranche B** — it withheld six
figures during Tranche A and stops of its own accord once the two forward figures
agree, which is the end state it was built for and not a special case added for
it. And **Q9.1 refuses on both trees**, identically: it asks for a change over
time that the forecast route cannot supply, and P1L's guard has always said so.
That refusal is not new and is not A2's.

## 8. Semantic assurance — three axes, scored separately

The brief required the scorer to distinguish **semantic**, **arithmetic** and
**presentation** correctness rather than collapsing them into one grade. The
frozen scorer `nl_score.py` does not do that and was not touched; a separate
instrument does, in `three_axis.py`.

### 8.1 Why three axes and not one number

An answer can be arithmetically perfect over the wrong rows. That is not a
hypothetical: it is precisely the state the V1 evidence pack shipped in, where
6,856 / 6,856 findings reconciled while the headline forecast was composed over a
population the governed config excludes. One grade cannot express "every number
is right and the answer is wrong".

| axis | question | measured by |
|---|---|---|
| SEMANTIC | did the agent read the question the way the frozen expectation says it should be read — family, operation, **population**, and the **population the headline was built from**? | `three_axis.py` against `frozen_expectations.yaml` |
| ARITHMETIC | is every numeric finding right? | **delegated** to `nl_reconcile.py`, which recomputes each from the fixture CSVs |
| PRESENTATION | are the required disclosures present, and does the answer print only figures the findings hold? | `three_axis.py` |

Arithmetic is delegated rather than reimplemented. Two matchers would be two
chances to be wrong, and the reconciler is the one already run against a baseline.

### 8.2 The headline-population check, which is the one that matters

The population axis asks whether the expected population appears **anywhere** in
the answer. That is a weaker question than it looks, and the weakness is exactly
the V1 defect: the pre-sprint Q9 answer *named* "the open pipeline" on a secondary
finding while its headline was composed over the whole extract. On that axis
alone the pre-sprint tree scores MATCH.

So the frozen file's `headline_finding` and `headline_addend_population` fields
are used for a second, stronger check: decompose the headline
(`headline = base + addend`) and identify **which finding's population the addend
actually came from**. It compares values, so a relabelling cannot satisfy it.

Run against the pre-sprint tree, the scorer states the defect in its own words:

```
Q9.3   DIVERGE    MATCH    DIVERGE
  ! headline addend 15,175,404.15 came from 'the governed pipeline extract'
    (pipeline:extract), expected pipeline_stage in [KFI, APPLICATION, OFFER]
```

### 8.3 Result

Both revisions, 752 runs each. `tranche_b/three_axis_det_pre.txt`,
`tranche_b/three_axis_det_post.txt`.

| axis | pre-sprint `49e00b5` | post-Tranche-B |
|---|---|---|
| SEMANTIC agreement with the frozen expectation | 692 / 752 (**92.0%**) | 740 / 752 (**98.4%**) |
| headline built from the expected population | **0 match, 48 diverge** | **48 match, 0 diverge** |
| population resolved as expected (where checkable) | 436 match, **0 diverge** | 436 match, **0 diverge** |
| ARITHMETIC — findings reconciled independently | **6,856 / 6,856** | **6,856 / 6,856** |
| PRESENTATION — missing disclosures | 0 | 0 |
| PRESENTATION — figures this revision prints that the other does not, and no finding holds | 0 | **0** |

The last row was measured **in both directions** — each revision scored against
the other as comparator — so neither "the new answers invented a figure" nor "the
old answers did" is left untested.

The arithmetic axis is identical on both sides and always was. That is the point
of separating the axes: **the sprint moved the semantic axis by 6.4 points and
the arithmetic axis by nothing**, because nothing was ever wrong with the
arithmetic. Reporting one blended number would have hidden both facts.

### 8.4 The 12 runs that still diverge, listed

All twelve are **Q2.3** — *"When are we forecast to cross £100m of funded
balances?"* — on both books, in both arms (with the LLM parser off the two arms
are identical by construction, so this is 3 repeats × 2 books × 2 arms). The parser maps no governed analytic
to this phrasing, so the agent refuses and says so:

> I couldn't map this question to a governed analytic, so I haven't computed an
> answer (nothing was guessed).

The frozen expectation says a correct answer is a date or horizon. This is a
**capability gap, honestly declared** — not a wrong answer — and it is unchanged
from before the sprint. Its sibling phrasings Q2.1, Q2.2 and Q2.4 are recognised
and answered. Widening the parser to catch it is an NL-coverage change, which
the brief's out-of-scope list excludes.

### 8.5 Two scorer rules that were wrong, and were corrected before publishing

Disclosed because the alternative is a scorer tuned in silence. Neither touched
the frozen expectation file, and neither moved a product figure.

1. **"a date or horizon, not a balance"** was first implemented as *"the answer
   must contain a date"*. On a £100m milestone against a £1.96bn book, the
   correct answer is *"The book has already reached £100.0m"* — a milestone
   behind you has no future date. The rule was wrong, not the product. Raw score
   before the correction: 704 / 752 (93.6%).
2. **The pipeline-exclusion disclosure** was demanded wherever the expectation
   named the open pipeline. Q2.4 names the pipeline in the *question* and is
   correctly answered "already reached" without reaching for it. The rule now
   applies only where the answer actually presents a pipeline-derived figure.

A third rule was written and **thrown away entirely**: an absolute "every printed
figure is held by a finding" check. It flags bucket labels (`30-40%`, `70-75`,
`0-12 months`) and rounded renderings (`£2.3m` for 2,332,075.32) as unheld
figures — it measures the extractor, not the product. The published check is
differential — each revision scored with the other as comparator — so every label
artefact cancels, because labels are identical on both sides. A fourth fix
followed from the same discipline: the tolerance now comes from the token's own
printed precision (`£752k` stands for [751.5k, 752.5k), `£2.3m` for
[2.25m, 2.35m)), because a flat percentage band rejected a faithful rendering of
751,834 as unheld.

## 9. Tranche B regression gates

All run on the shipped tree at `85c212a`/`6f61365`, with no checkout swapped
underneath them.

| gate | required | measured |
|---|---|---|
| affected pipeline / forecast tests (`test_forecast_bridge`, `test_pipeline_phase3_refinement`, `test_pipeline_prep_vectorisation`) | green | **46 passed** |
| analytical layer, intent boundary, P1I, P1J-1, P1L, P1M, P1N, fabricated-population, golden bank, 252-question calibration bank, all `mi_agent_api` tests | green | **1,989 passed, 13 xfailed, 0 failed** |
| 30-question simple-MI bank | no change | **0 of 30 changed**, `ok` and route byte-identical |
| 80-question wide bank | changes attributable | **2 of 80 changed**, both the A3 rolling-cohort disclosure; 65/80 `ok` unchanged |
| nine canonical CFO questions, both books | all green | **18 / 18 `ok=True`**; 2 of 9 changed on each book — Q1 (A3 disclosure) and Q9 (Tranche B) |
| 752-run NL bank, unsafe outcomes | 0 | **0** before and after |
| 752-run NL bank, grade differences pre vs post (like-for-like) | attributable | **0** |
| numeric findings reconciled independently | all | **6,856 / 6,856**, 0 mismatches |
| full repository suite | green | *running at the time of this commit — filled in below* |

### 9.1 The three tests that changed, and why each was not simply re-pinned

Each assertion was checked against the fixture data before it was touched.

**`test_prep_uses_historical_then_config`** carried the line

```python
self.assertIn("configured_stage_rate", srcs)   # COMPLETED via config 1.0
```

which **asserted the defect verbatim** — a settled case weighted at certainty
inside a forward forecast. It is replaced by the governed behaviour: COMPLETED is
excluded, names the stage that excluded it, and carries no probability. The
config-fallback tier that line also covered is independently covered by
`test_pipeline_prep_vectorisation`, which was checked before the line was
removed.

**`test_discloses_gross_excluded_and_basis`** moved from £80,000 / 1 case to
£170,000 / 2 cases. The fixture holds one withdrawn case at £80,000 and one
completed case at £90,000; £170,000 across 2 cases is what the governed exclusion
set actually removes from it.

**`test_forecast_loan_count_is_funded_plus_pipeline`** became
`..._plus_eligible_pipeline` and now also pins
`eligibleCaseCount == pipelineCaseCount − excludedCaseCount`, so the count and the
amount cannot drift apart again.

## 10. The anti-gaming floor

The brief set a floor to stop the result being improved by refusing more.

| rule | measured |
|---|---|
| substantive calculated-answer rate must not fall below **82.2%** | **85.4%** (642 / 752), unchanged from the pre-sprint tree measured the same way |
| controlled-refusal count must not rise | **56 → 56**, unchanged |
| the 752-run bank must not be edited to improve the result | `nl_bank.py` untouched — sha256 `f37729113df3a673…`, identical to the copy pinned in the manifest since V1 |
| the frozen baseline scorer must not be modified | `nl_score.py` untouched — sha256 `aade14173ab36231…`, identical to the manifest's; it graded both sides of every comparison in this report |
| the frozen expectation file must not be edited | untouched since `49e00b5`; the two rules corrected in §8.5 are in the *scorer*, and both are disclosed |

**No refusal needs individual justification, because no refusal was added.** The
eight extra refusals visible in the degraded LLM arm are not in this column: they
are parser-mix artefacts, demonstrated in §7.1, and they appear on the
post-sprint side of a comparison whose two sides did not run the same parser.
Under the like-for-like comparator the refusal count is identical.

The one honest qualification on the floor: 85.4% is measured with the LLM parser
off. The V1 82.2% was measured with it on. The two are not the same measurement,
and the floor is therefore satisfied on a deterministic-parse basis, not on the
basis it was originally set. §13 lists the outstanding LLM remeasurement.

## 11. Tranche C — investigation only, nothing implemented

No production file was changed for this section. `censoring.py` lives in
`due_diligence/evidence/` and no serving path imports it. Output:
`tranche_c_censoring.txt`.

### 11.1 The question

Tranche B made the empirical stage rates the **sole** basis of the forward
forecast on both books — `configured_stage_rate` now carries zero rows (§6.5).
So the question of whether those rates are sound stopped being academic in the
same commit that made them authoritative.

The shipped estimator computes, per stage:

```
rate(S) = |cases ever seen at S that were ever seen COMPLETED| / |cases ever seen at S|
```

and trusts it when the denominator reaches `MIN_OBSERVATIONS = 12`. Every case
ever seen at S is in that denominator, including one first seen in the final
week. If completion takes time — and the governed config says 30 to 90 days —
then the denominator contains cases that **could not yet have converted**, and
the rate is biased downward. `MIN_OBSERVATIONS` counts observations; it does not
ask whether any of them have had time to mature.

### 11.2 What the data can support

Independent recomputation from the twelve raw weekly extracts. Observation
window **2026-04-13 → 2026-06-29 = 77 days**.

| stage | config lag | config rate | shipped (naive) rate | cases with a FULL lag of observation |
|---|---|---|---|---|
| KFI | 90d | 0.20 | 0.0635 (32/504) A · 0.0628 (46/732) K | **none — the window is shorter than the lag** |
| APPLICATION | 60d | 0.45 | 0.1016 (32/315) A · 0.1007 (46/457) K | **0 of 315** A · **0 of 457** K |
| OFFER | 30d | 0.75 | 0.1693 (32/189) A · 0.1673 (46/275) K | 10 of 189 A · 15 of 275 K |

Not one stage on either book has a matured sample at its own configured lag.
OFFER has the only matured cases at all, and Alderbridge's 10 is **below the
`MIN_OBSERVATIONS` threshold of 12** that the shipped model itself requires.

A Kaplan–Meier cumulative incidence was computed as the second check, treating an
open case as censored at the last extract and a withdrawal as a competing failure.
It is reported but **not quoted as an estimate**, because it is not identified:
the risk set reaches zero before every horizon of interest.

| stage | horizon | at risk at the horizon | estimate |
|---|---|---|---|
| KFI | 56d | 126 | 0.0000 |
| KFI | 90d | **0** | 0.5078 — artefact |
| APPLICATION | 42d | 32 | 0.0000 |
| APPLICATION | 60d | **0** | 1.0000 — artefact |
| OFFER | 14d | 63 | 0.0000 |
| OFFER | 30d | **0** | 1.0000 — artefact |

Both estimators fail, in opposite directions, for the same reason. The honest
statement is not "the true rate is higher than 0.0635"; it is **the data does not
identify a completion rate at the horizons the forecast assumes.**

### 11.3 Why these fixtures cannot settle the magnitude

The demo extracts have a structure a real book would not:

* every observed completion happened at **exactly** the same elapsed time —
  KFI→completed 70 days (min 70, max 77), APPLICATION→completed 49 days for all
  32 / 46 cases, OFFER→completed 28 days for all of them;
* the **same 32 (Alderbridge) / 46 (Kestrelmoor) cases** are the completions
  counted at all three stages — the three "stage rates" share one numerator and
  differ only in their denominators;
* every completion falls in the **last two weekly extracts** (2026-06-22 and
  2026-06-29);
* there is **not one withdrawal** in the entire twelve-extract history.

This is a single cohort walking a fixed conveyor, not a conversion process. So
the magnitudes above are a property of the generator, and **no claim about a real
client's conversion rate should be drawn from them.**

The structural finding does transfer, because it does not depend on the data:
**a sufficiency gate that counts exposures rather than matured exposures will
license a censored rate on any book whose extract history is shorter than its
funding lag** — and it will do so silently, because `historical_observed` reads
as the more trustworthy basis.

### 11.4 Client-specificity

| element | client-specific? | where it lives |
|---|---|---|
| stage probabilities, lags, include/exclude stages | yes | `config/client/pipeline_expected_funding.yaml` |
| `MIN_OBSERVATIONS = 12` | **no — a module constant** | `pipeline_history.py:31` |
| the precedence rule (empirical beats config where sufficient) | **no — hard-coded** | `pipeline_prep._derive_probabilities_and_amounts` |
| the maturity requirement | **does not exist** | — |

Tranche B moved the exclusion set out of code and into the config that declares
it. The two governing quantities above did not move, and a client whose funding
cycle is longer than another's has no way to say so.

### 11.5 What I would propose, and am not doing

Recorded for a decision, not implemented; the brief withholds approval and none
is assumed.

1. **Maturity-aware sufficiency.** Count, per stage, the cases observed for at
   least the stage's configured lag. Require that count — not the raw
   observation count — to clear `MIN_OBSERVATIONS` before the empirical rate is
   trusted. On this data that alone would send all three stages back to config.
2. **Say which it used, and why.** `completion_probability_basis` already
   distinguishes `historical_observed` from `mixed_historical_and_config`.
   Neither value tells a reader that the empirical rate rests on a window shorter
   than the lag. That belongs in the disclosure, in the same sentence as the
   figure.
3. **Make the threshold and the horizon client config.** Both are constants in a
   shared module today.
4. **Do not ship a censored-estimator correction.** On 77 days of data no
   estimator is identified, and substituting one artefact for another would be
   worse than the honest naive rate — which at least does not claim more than it
   has seen.

The order matters: (1) and (2) make the current behaviour honest without
inventing a number. (4) is the recommendation to stop there.

## 12. Judgement calls and open items, stated rather than buried

Six things in this sprint are decisions rather than deductions. Each is reversible
and each is named so it can be reversed.

1. **B2 moved the forward count as well as the amount.** The brief named the
   amount. Leaving the count alone would have shipped a forecast whose two halves
   described different populations. If amount-only was the intent, the line to
   revert is `forecast_loan_count = funded_loan_count + eligible_case_count`.

2. **B3 was not consolidated.** Two forward findings still carry the same number
   under two labels. They agree now, so nothing a reader sees is wrong, and A2
   correctly stops refusing without being touched. Collapsing them is a
   structural change with no effect on a delivered figure, so it was kept out of
   a correctness tranche.

3. **A3 still withholds the rolling-cohort delta.** The answer prints "£21.4m
   against £18.3m" and not "(−£3.2m)". For a cohort pair sharing few or no loans
   the delta is the most misleading figure in the sentence. The finding still
   holds it. It remains a withdrawal of a previously printed figure.

4. **The LLM arm is unmeasured at full credit.** §7.1. The like-for-like
   comparison is sound; the production parser path is not remeasured since V1.

5. **Three scorer rules were corrected mid-measurement** (§8.5) and a fourth was
   thrown away. Each correction was to my instrument, never to the frozen
   expectation and never to a product figure; the raw score before the first
   correction is published alongside the corrected one.

6. **Tranche C is not implemented and no approval is assumed.** §11.5 is a
   proposal.

## 13. What is not done

| item | status |
|---|---|
| full-credit LLM remeasurement of the 752-run bank | **outstanding** — the supplied API key is exhausted; the deterministic comparator stands in |
| manifest divergence between `nl_harness.py` and its run files | **resolved** — that revision has now produced the `run-file:llm-degraded:*` groups, which name it; but those runs are excluded from every before/after claim (§7.1), so the *comparable* remeasurement is still outstanding |
| Q2.3 phrasing (*"When are we forecast to cross £100m…"*) | **open, out of scope** — an NL-coverage change, refused honestly today |
| Tranche C implementation | **not started, awaiting a decision on §11.5** |
| entries/exits decomposition for rolling cohorts | **out of scope** — would be a new cohort engine |
| collapsing the duplicate forward finding | **deferred to maintenance** |

## 14. Verdict

The two product defects the V1 audit found in the forecast are closed.

The one that mattered is B1. The governed config had said
`exclude_stages: [WITHDRAWN, COMPLETED]` all along; the serving path honoured half
of it, and cases the system itself classified as already funded were weighted at
certainty into the forecast of what is still to come. Alderbridge's expected
completions fall from £15,175,404.15 to £9,625,160.91 and Kestrelmoor's from
£12,459,455.42 to £7,707,250.33 — both reconciling to the penny against a
recomputation that imports nothing from the forecast modules and predates the
change, and Kestrelmoor landing exactly on the figure the frozen expectation
recorded before any code in this sprint was written.

The measurement that matters is the separation of axes. Arithmetic was
6,856 / 6,856 before this sprint and is 6,856 / 6,856 after it; nothing was ever
wrong with the arithmetic. Semantic agreement with the frozen expectation moved
from 92.0% to 98.4%, and the headline-population check — did the forecast get
built from the rows the question asked about — moved from **0 of 48** to
**48 of 48**. A single blended grade would have shown neither.

Nothing was traded for it: zero unsafe outcomes on both sides, zero grade
changes across 752 like-for-like runs, refusals unchanged at 56, substantive
answers unchanged at 85.4%, and 196 changed answers confined to the eleven
variations the two tranches touched.

The qualification that belongs in the same breath is §11. Tranche B made the
empirical conversion rates the sole basis of the forward forecast, and those
rates rest on a 77-day window against funding lags of 30 to 90 days. Not one
stage on either book has a matured sample at its own configured lag. The forecast
is now composed over the right population; whether it is weighted by a defensible
rate is the open question, and it is more open after Tranche B than before it.

**FORECAST & COMPOSITION HARDENING: TRANCHE A AND B COMPLETE. TRANCHE C
INVESTIGATED, NOT IMPLEMENTED.**
