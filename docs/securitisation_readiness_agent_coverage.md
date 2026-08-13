# Securitisation Readiness Agent — capability coverage

*Sprint 2, Part 13. What the agent could ask today, what it could not, and what
each gap actually needs. Sprint 2 does not build the agent; it builds what the
agent consumes.*

The test applied to every row: **could an agent get a defensible answer using
only published tools, with every figure computed by Trakt and evidenced?** An
answer the agent would have to compute itself is scored as *not covered*, even
where the inputs are available — because a number produced in a model's context
window has no governed definition, no provenance and no reproducibility.

Legend: **✅ covered** · **◐ partial** · **❌ not covered**

---

## 1. Portfolio understanding

| Question | Status | How |
|---|---|---|
| How big is this book, and what is in it? | ✅ | `portfolio_summary` |
| Balance-weighted LTV, rate, term | ✅ | `portfolio_summary.weighted_averages` |
| Composition by region / status / stage / collateral type | ✅ | `portfolio_summary.composition`, `stratify` |
| Break it down by any canonical dimension | ✅ | `stratify` (bounded, discloses truncation) |
| Cross-tabulate two dimensions | ❌ | One dimension per call. See gap **G1** |
| Arrears and delinquency profile | ✅ | `portfolio_summary.arrears`, `stratify` on `number_of_days_in_arrears` |
| Vintage / cohort analysis | ◐ | `stratify` on `portfolio_cohort` where the tape carries it; no date bucketing. **G2** |

## 2. Concentration and limits

| Question | Status | How |
|---|---|---|
| How concentrated is the book, by any dimension? | ✅ | `concentration` |
| Top-N borrower / region / product exposure | ✅ | `concentration` with `top_n` |
| Is a concentration limit breached? | ✅ | `evaluate_covenants` — the *approved* threshold, its approver and configuration version |
| By how much, and how much headroom is left? | ✅ | `evaluate_covenants` (utilisation, headroom, breach amount) |
| Which loans cause the breach? | ✅ | `covenant_drillthrough`, reconciled to the numerator |
| Has a test deteriorated since last period? | ✅ | `evaluate_covenants` (prior value, status transition, `deteriorated`) |
| Would this pool pass a *proposed* covenant not yet approved? | ❌ | Deliberate. A test that nobody approved has no governed definition. **G3** |

## 3. Loan-level investigation

| Question | Status | How |
|---|---|---|
| Show me these twenty loans | ✅ | `get_loans`, batch-first |
| As a structured credit object rather than 130 columns | ✅ | `get_loans(shape="structured")` |
| With the full valuation history | ✅ | `include=["valuations"]` |
| Which loans are worst on some metric? | ✅ | `rank_loans` — without retrieving the population |
| All loans matching a condition | ◐ | `rank_loans` + `filters` gives the extremes; there is no "return every loan where X". **G4** — and it is bounded on purpose |

## 4. Data quality and evidence

| Question | Status | How |
|---|---|---|
| What does this tape actually carry? | ✅ | `data_completeness` |
| Which fields are materially incomplete? | ✅ | `data_completeness(max_completeness_pct=…)` |
| What failed validation? | ✅ | `list_validation_exceptions` (field-level, which is the grain it is recorded at) |
| Where did this number come from? | ✅ | `explain_values` — source dataset, field, snapshot, content hash, mapping method and version |
| Was it mapped or derived? | ✅ | `explain_values.origin`, `transformation.derivation_rule` |
| **Why** is the LTV that number? | ✅ | `explain_values.derivation` — numerator, selected valuation by id, selection policy version, rejected observations with reasons |
| Is the valuation behind it stale? | ✅ | Same block: the rejection reason states the age and the limit |
| Which snapshot am I looking at? | ✅ | Every envelope carries `SnapshotRef` |

## 5. Movement

| Question | Status | How |
|---|---|---|
| What changed since last period? | ✅ | `period_change` |
| How did the balance move — redemptions, additions, amortisation? | ✅ | `period_change.balance_bridge` |
| Are these two periods comparable? | ✅ | `period_change.period_resolution` and `limitations` — Trakt resolves it, the agent must not assume it |
| Trend over more than two periods | ❌ | Two-period comparison only. **G5** |

## 6. Securitisation readiness proper

| Question | Status | How |
|---|---|---|
| Is the data complete enough to issue? | ◐ | `data_completeness` gives coverage; the *threshold* for "enough" is not encoded. **G6** |
| Does it meet Annex-required field coverage? | ❌ | `engine/gate_4_projection` knows the Annex 12 shape, but no tool exposes readiness against it. **G7** — the single highest-value gap |
| Are there eligibility-criteria failures? | ◐ | Where encoded as approved concentration tests, yes. Loan-level eligibility is not modelled. **G8** |
| Is the pool within its proposed covenants? | ✅ | `evaluate_covenants` |
| Produce the diligence evidence pack | ❌ | Every ingredient exists per call; nothing accumulates them. **G9** |
| What could I not determine? | ◐ | Every tool reports its own gaps in `warnings`; nothing aggregates them into one statement. **G10** |

---

## Coverage

**32 questions: 20 covered, 6 partial, 6 not covered.**

Everything in section 4 is covered, which matters more than the headline: the
questions an agent cannot answer are mostly about *scope and packaging*, not
about whether Trakt can evidence a figure. Sprint 2 moved provenance from "which
dataset" to "which valuation, under which policy version, and why not the others"
— which is the difference between a number and a defensible number.

---

## The gaps, and what each actually needs

| | Gap | What it needs | Size |
|---|---|---|---|
| **G7** | Annex-coverage readiness | A tool over the existing `gate_4_projection` field requirements: required fields, coverage, blocking gaps | **Small — highest value** |
| **G9** | Evidence pack | The enquiry lifecycle in `docs/a2a_readiness_design.md` (correlation and audit already exist per call) | Medium |
| **G6** | Readiness thresholds | Governed configuration for "complete enough", reviewed like a covenant rather than chosen by an agent | Small |
| **G10** | Aggregated unknowns | A closing tool that collects warnings and refusals across an enquiry — depends on G9 | Small after G9 |
| **G1** | Two-dimension stratification | Extend `analytics_lib.stratify` to a second dimension; both tools then get it | Small |
| **G2** | Vintage bucketing | `analytics_lib.buckets` already does date bucketing; wire it to `stratify` | Small |
| **G5** | Multi-period trend | The period-change engine is pairwise; a trend is a different query shape | Medium |
| **G4** | Filtered loan retrieval | Deliberately absent. Reconsider only with a hard bound and a stated reason | Deliberate |
| **G8** | Loan-level eligibility | A registered rule set, reviewed as code — never a formula language | Medium |
| **G3** | Unapproved covenant simulation | Would need a sandbox that can never be mistaken for an approved result | Deliberate |

Two of these (**G3**, **G4**) are *decisions*, not backlog. They are listed so
that a future reader knows they were considered and declined, rather than
overlooked.

---

## What Sprint 2 changed

Before Sprint 2 the agent could evaluate covenants and nothing else — one tool.
It can now answer every question in sections 1–5 that is marked covered, using
thirteen tools over the same governed path.

The two changes that matter most for a *securitisation* agent specifically:

1. **`explain_values` answers "why".** "The LTV is 60.0" is a number. This is a
   diligence answer, and it is one call
   (`explain_value(LN-I-005, current_loan_to_value)`, run against the planted
   portfolio in `tests/planted_portfolio.py`):

   > `current_loan_to_value = 60.0` percentage points, under `CURRENT_LTV@v1`,
   > **not recomputed by Trakt** — the canonical tape owns the figure.
   > Numerator `current_principal_balance = 180,000`. Denominator: valuation
   > `val_de9b64715fff4a3b`, an **indexed** observation of £300,000 dated
   > 2026-06-30, selected under `CURRENT_LTV_SELECTION@v1` as the most recent
   > qualifying indexed observation.
   > Rejected: the **full** valuation — *66 months old at 2026-07-31, and the
   > policy allows 24*; the **purchase price** — *a qualifying 'indexed'
   > valuation was available, which this policy prefers*.
   > Validation: `failed`, rules `LTV001`/`LTV004`, 2 errors.

   A buyer's agent can act on that. It can see that the reported LTV rests on an
   index rather than a physical inspection, that the last physical valuation is
   five and a half years old, and that the field carries validation failures —
   three findings, none of which are visible in "60.0".

2. **The aggregate tools mean the agent stops computing.** Before, "weighted
   average LTV of the London exposure" would have been answered by retrieving
   London and averaging it in the model — no governed definition, no provenance.
   Now it is one `stratify` call whose number is Trakt's.
