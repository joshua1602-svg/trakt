# MI Query Agent — final close-out and go-live freeze

| | |
|---|---|
| **Production SHA (intended for deployment)** | `23804de` |
| **Acceptance / oracle-only SHA** | `c1ca905` |
| **Production tree at `c1ca905` vs `23804de`** | **identical** — `git diff 23804de -- mi_agent mi_agent_api mi_workflows question_interpretation engine config configs trakt_core analytics` is empty |

Everything committed after `23804de` is under `migration_phase0/`: the oracle, the
grader, and the evidence record. No production file was opened during this
close-out.

---

## 1 · Frozen baseline, re-confirmed

| shipping bank, 166 questions | at `23804de` | after close-out |
|---|---:|---:|
| CORRECT | 135 | **136** |
| CORRECTLY DECLINED | 16 | 16 |
| NO CHECKABLE TRUTH | 1 | **0** |
| DECLINED BUT ANSWERABLE | 12 | 12 |
| **WRONG** | **2** | **2** |

The single movement is Q10A, and it is an oracle closure on an unchanged answer
(§2). Governed engine alone: 126 → **127** correct.

Frozen regression manifest: **85 failing/erroring names, name for name.**
Provider-unavailable: **0 answered, 0 wrong, 0 whole-book fallback** across five
injected failure modes. Semantic coverage census: **1,612 questions, 0 answering
questions carry an unaccounted concept.**

The two accepted residual defects are unchanged and were not touched:
**Q04C** — correct 24-loan population, loan-level groups where a scalar total was
asked for. **Q19A** — five-period progression where a last-month delta was asked
for. Both 6/6 wrong across the stability run, i.e. deterministically wrong rather
than intermittently.

---

## 2 · Q10A — closed, oracle only

Truth computed directly from the governed extract, not from any agent output:

```
tests/fixtures/pipeline_history_5w/2026-05-29/M2L_KFI_and_Pipeline_2026_05_29.csv
rows            8
Loan Amount     100,000 + 200,000 + … + 800,000  =  3,600,000.00
latest of       five weekly extracts (2026-05-01 … 2026-05-29)
```

The agent answers **8 loans, £3.6MM**, reconciled against `pipeline`. CFO66
("what is the pipeline balance") and CFO67 ("how many cases are in the pipeline")
read the same file, reconcile against the same dataset and are already graded
correct against the CFO bank's own assertions — so all three are measuring one
extract.

Q10A had been NO_COMPUTABLE_TRUTH because its frozen human grade
("WRONG / SILENT") was recorded against an earlier, different answer and the
grader refuses to apply a stale verdict. That is the fidelity rule working, not a
gap in the answer.

**Answer unchanged.** Q10A, CFO66 and CFO67 were re-asked at `c1ca905` and are
byte-identical to the frozen capture.

The oracle also gained a generic `dataset` key, checked against the envelope's
reconciliation record rather than the prose — a pipeline figure and a funded
figure read alike in a sentence. This is the check that would have caught the
pipeline summary answering from the funded book.

---

## 3 · Reach denominator — stated conservatively

> **8 of the original 16 existing-capability reach cases recovered — 50%.**

Recorded separately, and not netted off the headline:

> CFO60 / CFO61 may ultimately be reclassified as governed methodology refusals
> if the cross-originator comparability rule is confirmed as the authoritative
> product requirement. The registry declares those dimensions' categories
> originator-specific, so a book-level concentration over them would present
> unaligned categories as one exposure, and the methodology declines with that
> reason stated. **The acceptance bank still expects an answer to both**, so they
> remain in the denominator until that product decision is made.

No production work was done on either in this close-out.

---

## 4 · Q22A — the six-run ledger, read not re-run

| run | grade | arm call | model |
|---:|---|---|---|
| 1 | CORRECT | applied | claude-opus-5 |
| 2 | CORRECT | applied | claude-opus-5 |
| 3 | CORRECT | applied | claude-opus-5 |
| 4 | CORRECT | applied | claude-opus-5 |
| **5** | **FALSE_REFUSAL** | **proposal_unavailable** | **none — no call succeeded** |
| 6 | CORRECT | applied | claude-opus-5 |

Run 5 returned the controlled availability refusal verbatim. So:

> **5 of 5 answer-bearing runs correct; one provider-unavailable controlled
> refusal; 0 wrong.**

It was not a semantic refusal and not an incorrect answer. No production change.

Q22A is adjudicated on winner = Direct and Direct delta = +£12,366,371.40 because
its question asks which portfolio contributed most; Q22B and Q22C name both books
and are still adjudicated on both sides.

---

## 5 · Live ERE data pre-flight

**This section could not be completed, and the reason is data availability in
this environment — not a defect in the client's data and not a defect in the
Query Agent.** Stating that precisely matters: I have not seen ERE production MI
data, so I make no claim about whether it satisfies the contract.

### What is actually present

One ERE-shaped artefact:
`ERE_Portfolio_122025_ESMA_Annex2_canonical_ESMA_Annex2_typed.csv` — 33 rows,
130 columns, cut-off 2025-10-31, originator `more2life`, Omni product family,
real broker names. Genuine ERE shape.

**It is an ESMA Annex 2 regulatory OUTPUT, not an MI input.**
`engine/gate_4_projection/regime_projector.py` fills *blank* fields with ND codes
for regulatory submission, which is exactly what this file carries: 73 of 130
columns are ND-coded on every row, 27 are entirely null, 30 carry values. It sits
downstream of the MI curation layer, not upstream of it.

**No curated ERE MI tape exists anywhere on this machine.** Every
`18_central_lender_tape.csv` present is a synthetic test fixture of 60–73 rows;
none carries an ERE signature. The 640-loan book used for all acceptance
measurement is synthetic (`L00000…`, `Alpha Network`, `Beta Partners`).

### 5A · The contract check that could be run

Against the ERE product profile's `base_mi` requirement:

| required field | column | usable value |
|---|---|---|
| loan_identifier | **absent** | — |
| current_principal_balance | present | **ND5 on all 33 rows** |
| current_interest_rate | present | yes — 9.29–9.56, points scale ✓ |
| origination_date | present | yes |
| current_valuation_amount | present | yes |
| reporting_date | **absent** | — (the file carries `data_cut_off_date`) |

The balance equivalence group — `current_principal_balance` /
`current_outstanding_balance` / `current_loan_balance` — is **unsatisfied on all
three members**. `current_outstanding_balance` is declared `source_criteria:
curated` in the registry: it is produced by curation, never read raw, and the
ERE profile derives it `from_field: current_principal_balance`.

Fields the smoke bank would need: `source_portfolio_type`, `source_portfolio_id`,
`product_type`, `collateral_geography`, `loan_identifier` and `pipeline_stage`
are **absent**; `erm_product_type` is **entirely null**; `occupancy_type` is
ND-coded throughout; `geographic_region_obligor` is `GBZZZ` (ESMA "not
available") on all 33 rows. A single cut-off date means **no history**, so
movement and forecast questions could only refuse.

Where a value *is* present, the scales are correct and match what the profiler
expects: LTV as a fraction (median 0.4000), interest rate in points (median
9.550), ages 57–79, 33 distinct identifiers over 33 rows.

### 5B / 5C · Not executable

The independent live truth pack and the 20–30 question live smoke both require a
curated live tape. Without one, running them against the ESMA output artefact
would measure a file the Query Agent was never designed to read, and any result —
pass or fail — would be misleading. **They were not run and no substitute was
invented.**

### A secondary observation, classified and not fixed

`ProductProfile.derive_current_outstanding_balance` accepts a source value when
it is `not in (None, "")`. The ESMA sentinel `ND5` passes that test, so running
the governed derivation over this artefact produced a balance on **33 of 33
rows, every value the string `ND5`**. The docstring's "never fabricated" holds —
it is propagating, not inventing — but a no-data sentinel would reach the
governed balance field as though it were a value.

Classification per §6: **CANONICAL TRANSFORM**. Not a Query Agent defect. Its
reachability on the real onboarding path depends on whether ND-coded input can
arrive at curation, which I cannot establish here. **Recorded, not fixed** — the
brief permits an upstream fix only where it is clearly within the existing
contract, and I cannot prove that from this environment.

---

## 6 · What would clear the pre-flight

The exact upstream blocker:

> A curated ERE MI tape — `18_central_lender_tape.csv` for the ERE client, with
> `current_outstanding_balance` populated numerically and a `reporting_date` — is
> not present in this environment. The only ERE artefact available is an ESMA
> Annex 2 regulatory output whose `current_principal_balance` is ND-coded on
> every row, and which the MI contract does not read.

To clear it: run the onboarding/curation pipeline over the live ERE source
extract, place the resulting governed tape (and any pipeline extract) where
`MI_AGENT_ONBOARDING_OUTPUT_ROOT` and `MI_AGENT_PIPELINE_ROOT` resolve, load the
ERE client configuration and its Schedule 8, then re-run §5A–5C unchanged. The
Query Agent needs no modification for that to happen.

---

## 7 · Accepted post-go-live backlog

| item | class |
|---|---|
| Q04C — correct population, wrong output grain | accepted residual defect |
| Q19A — five-period progression for a last-month delta | accepted residual defect |
| Q25A/B/C — forward concentration forecasting | genuinely new capability |
| Q07B, Q20B — narrowing reaches the contract, execution does not apply it | safe reach gap |
| Q21B, Q21C — field placed in the wrong slot by interpretation | safe reach gap |
| Q10C — no governed analytic matches the formulation | safe reach gap |
| Q15C, CFO71 — a word with two governed meanings | interactive clarification UX |
| CFO60 / CFO61 methodology decision | product/governance decision |
| `derive_current_outstanding_balance` accepts an ND sentinel | canonical transform, upstream |
| Live ERE curated tape and smoke | go-live pre-flight, blocked on data |

---

## 8 · Decision

Synthetic acceptance is at or above the frozen result (136 ≥ 135, wrong = 2).
The conservative 8/16 reach recovery is preserved. The frozen manifest is exactly
85. No production semantic architecture was altered — the production tree is
byte-identical to `23804de`.

The Query Agent itself is sound and frozen. What cannot be demonstrated is the
live-data half of go-live, because the live curated data is not in this
environment.

GO-LIVE DATA CONTRACT BLOCKER — QUERY ARCHITECTURE REMAINS FROZEN
