# TRAKT MI Agent — 40-Question Pre-Launch Baseline

**Status: LAUNCH GATE FAILED — 1 silent semantic error (B25).**
No fix was attempted; this was a verification-only task.

| | |
| --- | --- |
| Date | 18 August 2026 |
| Branch | `claude/mi-query-agent-review-n8d33r` |
| HEAD SHA tested | `4c550ee4d0a94193c73132e7919b57345efbfab1` |
| Working tree | clean |
| Local vs remote | **NOT in sync — 7 commits unpushed** (remote at `25190c1`) |
| Entrypoint | `mi_agent_api.mi_service.execute_governed_mi_query` |
| Question bank | `config/mi/golden_questions/business_semantic_questions.yaml` v1, 40 questions, sha256 `e0fc0b61…3194` |
| Fixture | `demo_platform` / alderbridge — 11,035 loans, £1,964,886,258.21, snapshots 2026-04-30 / 05-31 / 06-30, combined sha256 `34e20a43…1e06` |
| Semantics registry | sha256 `c29b555b…18a7` |

> **Correction to the brief's premise.** The task states "the branch has been pushed but not
> merged". It has **not** been pushed. `origin/claude/mi-query-agent-review-n8d33r` is at
> `25190c1` (P1A era); every commit from P1B onward — including P1C and the B11 fix — exists
> only locally. This baseline therefore records a **local** HEAD.

---

## 1. Summary

| Outcome | Deterministic | Genuine LLM |
| --- | ---: | ---: |
| CORRECT | 7 | 7 |
| EXPLICIT_PARTIAL | 3 | 1 |
| SAFE_REFUSAL | 29 | 31 |
| INCORRECT_SUCCESSFUL | 0 | 0 |
| **SILENT_SEMANTIC_ERROR** | **1** | **1** |
| HARD_FAILURE | 0 | 0 |
| **Total** | **40** | **40** |

| Measure | Value |
| --- | --- |
| Route agreement | **40 / 40** |
| Genuine LLM parses | **32 / 40** |
| Deterministic zero-cost short-circuit | 8 (A5, B05, B09, B11, B13, B16, B20, B28) |
| `deterministic_fallback_after_llm_failure` | **0** |
| LLM specs rejected by the governed validator | 18 (reported as `parse_failure`) |
| Receipt coverage (CORRECT + EXPLICIT_PARTIAL) | 10/10 deterministic, 8/8 LLM |
| Numerically reconciled to independent truth | 10 deterministic / 8 LLM substantive answers |
| No numerical truth (refusal or qualitative by design) | 29 deterministic / 31 LLM |
| Refusals leaking substantive results | **0** |
| Unhandled exceptions | **0** |

---

## 2. LAUNCH GATE

| Criterion | Required | Deterministic | LLM | Result |
| --- | --- | ---: | ---: | --- |
| Incorrect successful answers | 0 | 0 | 0 | PASS |
| Silent semantic errors | 0 | **1** | **1** | **FAIL** |
| Hard failures | 0 | 0 | 0 | PASS |

**MI 40-QUESTION PRE-LAUNCH BASELINE: FAIL**

### The failure — B25

**Question:** *"How does the direct book compare with the acquired book on borrower age?"*

**Expected behaviour:** report the borrower-age comparison between the two books, or refuse
and state that borrower age was not among the governed comparison indicators.

**Actual behaviour:** `ok: true`, route `portfolio_risk_comparison`, **zero artifacts**, and
the assertion:

> "Compared Direct with Acquired at the current reporting date: no governed directional
> differences were observed across the selected indicators."

**Execution evidence:**

* `spec.metric = "youngest_borrower_age"`, `spec.aggregation = "avg"` — the parser resolved
  the requested measure correctly;
* `workflow.mode = "requested_metric"` — the workflow was asked for that specific metric;
* `workflow.metric_comparisons = []` — **nothing was compared**;
* `workflow.distribution_comparisons = []`, `workflow.summary = []`, `evidence = []`;
* scopes resolved correctly: direct 7,126 rows, acquired 3,909 rows (= 11,035);
* receipt: `"Calculated: Portfolio comparison."` — no measure, no population, no indicator
  list, and no statement that the requested metric was not compared;
* P0 facet ledger carries a single facet, `cohort_comparison: applied`. No facet exists for
  the requested measure, so the guard had nothing to adjudicate and returned `ok`.

**Independent truth** (pandas, straight from the fixture):

| Book | Mean `youngest_borrower_age` | Rows |
| --- | ---: | ---: |
| Direct | **72.188** | 7,126 |
| Acquired | **69.957** | 3,909 |
| Difference | **2.231 years** | |

**Why this is SILENT_SEMANTIC_ERROR and not EXPLICIT_PARTIAL:** the reader asked about
borrower age and is told no differences were observed. Nothing in the answer, the receipt or
the warnings discloses that borrower age was never among "the selected indicators". The
system answered a materially different question — "were there directional differences across
the governed indicator set (which was empty)?" — and the substitution is not disclosed. The
same route answers A8 with a populated comparison table, so the shape is available; here it
returned a null finding instead.

**Severity: HIGH.** It is a confident negative assertion contradicted by the data, on both
parser paths, and it passes the P0 guard.

**Not a regression.** Byte-identical to the previous P1D run. What changed is the
**classification**, not the behaviour: §3 of this task required independent numerical
reconciliation of every substantive answer, which had not been applied to B25 before. The
earlier baseline recorded it as EXPLICIT_PARTIAL ("honest, but gives no age figures"); that
was too generous, and this report supersedes it.

---

## 3. Complete 40-question table

Route shown once — deterministic and LLM routes agreed on all 40.

| ID | Short question | Det | LLM | Route | Truth reconciled? | Receipt valid? | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | avg LTV + age + borrower type in London | SAFE_REFUSAL | SAFE_REFUSAL | — | n/a | n/a | Det: multi-measure refusal. LLM: borrower_type unavailable |
| A2a | Which region grew most last month | **CORRECT** | **CORRECT** | period_change_analysis | ✅ South East +£7,840,963.14 | ✅ | ranked movement, basis + both dates stated |
| A2b | Which broker grew most | SAFE_REFUSAL | SAFE_REFUSAL | period_change_analysis | n/a | n/a | broker not a governed period-change dimension |
| A2c | Which product grew most | SAFE_REFUSAL | SAFE_REFUSAL | period_change_analysis | n/a | n/a | dimension named in refusal |
| A2d | Which borrower type grew most | SAFE_REFUSAL | SAFE_REFUSAL | period_change_analysis | n/a | n/a | dimension named in refusal |
| A3 | balance by LTV by borrower type | SAFE_REFUSAL | SAFE_REFUSAL | — | n/a | n/a | Det refuses the Amortisation Type substitution |
| A4 | balance by region by borrower type | EXPLICIT_PARTIAL | SAFE_REFUSAL | — | ✅ SE £516,214,136.58 / 2,420; London £413,804,467.49 / 1,380 | ✅ | "Not applied: borrower type — field is unavailable" |
| A5 | balance by borrower type by product | SAFE_REFUSAL | SAFE_REFUSAL | — | n/a | n/a | field-unavailable refusal |
| A6 | Close to breaching concentration limits? | **CORRECT** | **CORRECT** | risk_limits | ✅ all 12 tests (below) | ✅ | specialist route reachable |
| A7 | When will funded loans be £100MM | SAFE_REFUSAL | SAFE_REFUSAL | — | n/a | n/a | Det: unmapped. LLM: funded_status unavailable |
| A8 | avg LTV direct vs acquired | **CORRECT** | **CORRECT** | portfolio_risk_comparison | ✅ 43.35 / 42.68 vs truth 43.3535 / 42.6846 | ✅ | prose states direction; table carries both values |
| A9 | avg collateral value since inception | SAFE_REFUSAL | SAFE_REFUSAL | period_change_analysis | n/a | n/a | no eligible governed field in both snapshots |
| B01 | Most concentrated + headroom | **CORRECT** | **CORRECT** | risk_limits | ✅ identical to A6 | ✅ | |
| B02 | Segments driving growth this quarter | SAFE_REFUSAL | SAFE_REFUSAL | period_change_analysis | n/a | n/a | period-grain refusal: 2 months ≠ "this quarter" |
| B03 | Over-reliant on any single broker | SAFE_REFUSAL | SAFE_REFUSAL | — | n/a | n/a | |
| B04 | Credit quality: new vs back book | SAFE_REFUSAL | SAFE_REFUSAL | — | n/a | n/a | Det: cohort comparison lost. LLM: PD unavailable |
| B05 | Breach 75% LTV cap if HPI −10% | SAFE_REFUSAL | SAFE_REFUSAL | period_change_analysis | n/a | n/a | stress + threshold both named as unapplied |
| B06 | Exposure to borrowers over 85 | **CORRECT** | **CORRECT** | — | ✅ 86 loans / £19,428,730.79 | ✅ | `Borrower Age > 85`, identical predicate on both paths |
| B07 | Headroom before London limit binds | SAFE_REFUSAL | SAFE_REFUSAL | risk_limits | n/a | n/a | London scope could not be applied → refuse |
| B08 | Run rate of new lending | **CORRECT** | **CORRECT** | forecast_extrapolation | ✅ £16.3m/mo vs truth mean £16,287,633.51 | ✅ | specialist route reachable |
| B09 | Which vintages have highest LTV | SAFE_REFUSAL | SAFE_REFUSAL | — | n/a | n/a | vintage unavailable |
| B10 | Arrears share + concentration | SAFE_REFUSAL | SAFE_REFUSAL | concentration_analysis | n/a | n/a | measure substitution refused |
| B11 | Region contributing most to WA LTV | **CORRECT** | **CORRECT** | — | ✅ South East 11.3360 of 43.1562 | ✅ | contribution ≠ highest LTV; both stated |
| B12 | Diversification vs last quarter | SAFE_REFUSAL | SAFE_REFUSAL | portfolio_risk_comparison | n/a | n/a | |
| B13 | Product type with highest ticket | SAFE_REFUSAL | SAFE_REFUSAL | — | n/a | n/a | |
| B14 | Acquired converging with direct on LTV | SAFE_REFUSAL | SAFE_REFUSAL | — | n/a | n/a | |
| B15 | Proportion eligible for 75% LTV securitisation | SAFE_REFUSAL | SAFE_REFUSAL | — | n/a | n/a | threshold not applied → refuse (no WA LTV substitute) |
| B16 | Brokers bringing highest LTV | SAFE_REFUSAL | SAFE_REFUSAL | — | n/a | n/a | |
| B17 | Driving change in WA LTV since inception | SAFE_REFUSAL | SAFE_REFUSAL | funded_bridge | n/a | n/a | measure substitution refused |
| B18 | Regional mix shift over last quarter | SAFE_REFUSAL | SAFE_REFUSAL | concentration_analysis | n/a | n/a | |
| B19 | Balance at year end at current rate | SAFE_REFUSAL | SAFE_REFUSAL | — | n/a | n/a | projection refused; no £1.96bn KPI leaked |
| B20 | Cohorts closest to NNEG risk | SAFE_REFUSAL | SAFE_REFUSAL | — | n/a | n/a | |
| B21 | Largest single-loan exposure + share | EXPLICIT_PARTIAL ⚑ | SAFE_REFUSAL | — | ✅ £841,638.96 (row 1) | ✅ | share-of-book half not answered — see §6 |
| B22 | Concentration in top 10 postcodes | EXPLICIT_PARTIAL | EXPLICIT_PARTIAL | geo_exposure | ✅ TLI35 £83,379,049.41 / 4.2435% / 172 areas | ✅ | granularity substitution disclosed |
| B23 | Older borrowers, bigger loans vs value | SAFE_REFUSAL | SAFE_REFUSAL | — | n/a | n/a | |
| B24 | Fastest-growing part by loan count | SAFE_REFUSAL | SAFE_REFUSAL | period_change_analysis | n/a | n/a | dimension "part" unresolvable |
| **B25** | **Direct vs acquired on borrower age** | **SILENT_SEMANTIC_ERROR** | **SILENT_SEMANTIC_ERROR** | portfolio_risk_comparison | ❌ 72.188 vs 69.957 — 2.231y difference | ❌ receipt omits the uncompared metric | **GATE FAILURE — see §2** |
| B26 | 10% HPI fall on WA LTV | SAFE_REFUSAL | SAFE_REFUSAL | — | n/a | n/a | stated as unstressed |
| B27 | Regions most exposed vs last month | SAFE_REFUSAL | SAFE_REFUSAL | geo_exposure | n/a | n/a | correctly refused — exposure ranking ≠ growth ranking |
| B28 | Book quality by origination vintage | SAFE_REFUSAL | SAFE_REFUSAL | — | n/a | n/a | |

⚑ = judgement call, reasoning in §6.

---

## 4. Historical-defect regression checks (§5)

| # | Check | Result | Evidence |
| --- | --- | --- | --- |
| A | London filtering does not return whole-book | **PASS** | Probe "weighted average LTV in London": `Weighted-average Current LTV · London · 1,380 loans`, value **42.7453396482** vs truth **42.745339648169605**. Not 11,035 loans, not 43.16% |
| B | "over 85" → `> 85`, both paths identical | **PASS** | B06 both paths: `Borrower Age > 85 · 86 loans`, £19.4MM. Truth `>85` = 86 / £19,428,730.79; `>=85` would be 136 / £31,115,676.60 |
| C | LTV bound not substituted by portfolio WA LTV | **PASS** | B15 refuses, states the threshold was not applied, returns no figure and no artifacts |
| D | Unstressed result not presented as stressed | **PASS** | B26 and B05 both refuse and state "no governed stress or scenario calculation was run, so this figure is unstressed"; zero artifacts |
| E | `risk_limits` reachable | **PASS** | A6 and B01, both parser paths |
| F | `forecast_extrapolation` reachable | **PASS** | B08, both parser paths |
| G | Ranked period movement + deterministic ranking | **PASS** | A2a → `period_change_analysis`; South East £508,373,173.44 → £516,214,136.58 (+£7,840,963.14) matches truth exactly |
| H | Superlative single-loan exposure ≠ whole-book | **PASS** | B21 returns a 10-row loan-level table, row 1 `ALP_ORIGINATION-006359` at **£841,638.96** = truth max. £1.96bn is never presented as the answer |
| I | B11 contribution vs highest-LTV neighbour | **PASS** | "contributes most" → **South East** (11.3360). "highest weighted-average LTV" → **West Midlands** (43.9477). Distinct calculations, both reconciled |
| J | No refusal leaks a substantive result | **PASS** | 29 det / 31 LLM refusals inspected. Only artifact type present is `validation` (0 rows). Zero charts, tables or KPIs with content |

### A6 / B01 limit reconciliation (all 12 tests independently checked)

| Test | Reported | Independent truth | ✓ |
| --- | --- | --- | --- |
| London | 21.1% vs 25.0% | 413,804,467.49 / 1,964,886,258.21 = 21.06% | ✅ |
| South East | 26.3% vs 30.0% | 516,214,136.58 / 1,964,886,258.21 = 26.27% | ✅ |
| Largest other region (South West) | 12.1% vs 15.0% | 12.08% | ✅ |
| Scotland | 3.3% vs 8.0% | 3.27% | ✅ |
| Largest single loan | 0.0% vs 1.0% | 841,638.96 / book = 0.0428% | ✅ |
| WA current LTV | 43.2% vs 50.0% | 43.1562% | ✅ |
| Loans above 75.0% LTV | 0.3% vs 75.0% | 100 − 99.6699 = 0.3301% | ✅ |
| **Borrowers aged over 85** | **1.0% vs 0.8%, headroom −0.2, RED** | 19,428,730.79 / book = **0.9888%**; 0.8 − 0.9888 = **−0.1888** | ✅ |
| 3 further tests | `unavailable` (missing fields) | correctly disclosed | ✅ |

---

## 5. Parser provenance (§2)

| Provenance | Count | Questions |
| --- | ---: | --- |
| `llm` — genuine live model parse | **32** | all except those below |
| `deterministic` — zero-cost short-circuit (intentional) | 8 | A5, B05, B09, B11, B13, B16, B20, B28 |
| `deterministic_fallback_after_llm_failure` | **0** | — |

No deterministic result is counted as a genuine LLM parse anywhere in this report.
18 questions had the model's spec rejected by the governed validator (`parse_failure`); in
every case the carried/deterministic spec was used and the provenance recorded honestly.

---

## 6. Judgement calls, stated openly

**B21 — classified EXPLICIT_PARTIAL, flagged.** The question has two halves. The largest
single-loan exposure is present and exactly correct (row 1, £841,638.96); the share of the
book is neither computed nor narrated as unanswered. The receipt does accurately state what
executed (`Loan-level Balance · entire funded portfolio · 10 groups · 11,035 loans`), which
is this system's disclosure mechanism, and nothing false is asserted — so EXPLICIT_PARTIAL.
A stricter reading would call it an undisclosed partial. It does not affect the gate verdict.
On the LLM path the same question is refused outright.

**B22 — EXPLICIT_PARTIAL.** Granularity substitution (postcode → ITL3) is disclosed
explicitly. The *top-10 aggregate* share is not stated, though the 15-row table permits it.

**A8 — CORRECT.** The prose states only the direction; the comparison table carries both
weighted averages (43.35 / 42.68), which reconcile to truth. Classified on the substantive
content, not the sentence.

---

## 7. Receipt validation (§7)

All 10 deterministic and 8 LLM substantive answers carry a receipt. Metric, filters,
grouping, comparison period, ranking basis, aggregation and population were each checked
against what executed:

* **A2a** — `Governed period change · ranked by Collateral Geography · absolute balance
  movement, largest increases first · 2026-05-31 → 2026-06-30`: dimension, basis, direction
  and both dates all accurate.
* **B06** — `Total Balance · Borrower Age > 85 · 86 loans · as at 30 June 2026`: predicate
  and population exact.
* **B11** — `Contribution to portfolio weighted-average Current LTV · grouped by Region ·
  11,035 loans`: aggregation and grouping accurate.
* **A4 / B22** — carry an explicit `Not applied:` line naming the unhonoured facet.
* **A6 / B01 / A8 / B08** — routed receipts naming the governed capability. Minimal but
  accurate; none claims semantics that did not execute.

**One receipt inconsistency: B25.** `"Calculated: Portfolio comparison."` is not false, but
it omits that the requested metric (`youngest_borrower_age`) was parsed and then compared
against nothing. A receipt whose purpose is to expose a scope error does not expose this one.

---

## 8. Comparison with the previous reported baseline (§11)

| | Previously reported | This run | Change |
| --- | --- | --- | --- |
| Det — Correct | 6 | **7** | reclassification (A8) |
| Det — Explicit partial | 5 | **3** | A8 → CORRECT; B25 → SILENT_SEMANTIC_ERROR |
| Det — Safe refusal | 29 | 29 | unchanged |
| Det — Silent semantic error | 0 | **1** | **B25 reclassified** |
| LLM — Correct | 6 | **7** | reclassification (A8) |
| LLM — Explicit partial | 3 | **1** | A8 → CORRECT; B25 → SILENT_SEMANTIC_ERROR |
| LLM — Safe refusal | 31 | 31 | unchanged |
| LLM — Silent semantic error | 0 | **1** | **B25 reclassified** |
| Route agreement | 40/40 | 40/40 | unchanged |

**The system's behaviour did not change.** A field-level diff of the deterministic
transcript against the previous P1D run shows **no question changed** on `ok`, `route`,
`answer`, `error` or spec. On the LLM path two questions differ in the model's *spec* only
(A2a's spurious filter column, B18 parsing genuinely instead of falling back) with
**identical answers**; B18 is a small improvement in provenance.

Every classification change is a change in **my assessment**, produced by applying §3's
requirement to reconcile each substantive answer against independent truth — a step not
performed on A8 or B25 before. The earlier report's B25 entry is superseded.

* Previously CORRECT questions no longer CORRECT: **none**.
* Previously EXPLICIT_PARTIAL that degraded: **B25** (to SILENT_SEMANTIC_ERROR, on
  re-assessment, not on behaviour change).
* Specialist routes that became unreachable: **none** (`risk_limits`,
  `forecast_extrapolation`, `funded_bridge`, `period_change_analysis`,
  `concentration_analysis`, `geo_exposure`, `portfolio_risk_comparison` all reached).
* Deterministic/LLM semantic disagreement: **none** — where both answer, the answers are
  identical. A4 and B21 differ only in strictness (deterministic answers with disclosure,
  LLM refuses); neither is wrong.
* Receipt inconsistencies: **1** (B25).
* Refusals containing leaked substantive results: **0**.

---

## 9. Harness verification (§14)

The acceptance and evaluation suites were run to establish the harness itself executed
correctly — **312 passed, 0 failed**:

```
mi_agent/tests/test_p0_execution_receipt.py
mi_agent/tests/test_p1a_single_filter.py
mi_agent/tests/test_p1b_route_precedence.py
mi_agent/tests/test_p1c_ranked_movement.py
mi_agent/tests/test_p1d_aggregate_contribution.py
mi_agent/tests/test_mi_filter_normalisation.py
tests/test_p1c_ranked_movement_e2e.py
tests/test_p1d_aggregate_contribution_e2e.py
```

No unhandled exception occurred in either bank run (0 `HARD_FAILURE`).

---

## 10. Saved artifacts

| File | Contents |
| --- | --- |
| `due_diligence/MI_40Q_PRELAUNCH_BASELINE.md` | this report |
| `due_diligence/MI_40Q_PRELAUNCH_BASELINE_deterministic.json` | full deterministic transcript, 40 questions |
| `due_diligence/MI_40Q_PRELAUNCH_BASELINE_llm.json` | full genuine-LLM transcript, 40 questions |
| `due_diligence/MI_40Q_PRELAUNCH_BASELINE_manifest.json` | HEAD SHA, remote state, bank hash, fixture hashes, registry hash, timestamp, parser provenance, gate verdict |

Both transcripts were scanned for credentials: **clean**. No API key was written to disk,
committed, or printed at any point.

---

## 11. Verdict

**MI 40-QUESTION PRE-LAUNCH BASELINE: FAIL**

One silent semantic error (**B25**), present on both parser paths, pre-existing and not a
regression. No fix was attempted, per §15. Everything else in the bank is correct, disclosed
or refused, with zero incorrect successful answers and zero hard failures.
