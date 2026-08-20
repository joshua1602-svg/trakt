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

## 5–17

*(Tranche B, its measurement, Tranche C and the launch recommendation follow.)*
