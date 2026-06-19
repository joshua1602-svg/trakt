# Trakt MI Runtime — Product Proof Pack

**Audience:** product, commercial, and delivery stakeholders.
**Type:** business-readable proof summary of what the Phase 6B / 6C MI runtime
already demonstrates. Documentation only — no new runtime features, no Azure, no
onboarding orchestration, no M&A Agent, no Annex 2/regulatory changes.

**Date:** 2026-06-18

---

## 1. What Trakt now proves

Trakt can take **fragmented, real-world-shaped source data** and turn it into
**governed, recurring management information** — end to end, deterministically:

- **Fragmented artefacts can be consolidated.** Separate borrower, loan,
  collateral, cashflow/arrears, portfolio/SPV-mapping and pipeline files are
  joined into a single coherent view per reporting period.
- **Canonical recurring MI snapshots can be created.** Each reporting date
  becomes one canonical snapshot — the building block for "this month vs last
  month" reporting.
- **Snapshots can be stored and queried.** They are persisted and retrieved
  through the snapshot store (`LocalFsSnapshotStore`), so history is preserved
  and re-queried rather than recomputed ad hoc.
- **A single governed query entry point answers the MI questions that matter.**
  `run_mi_query` answers funded, pipeline, forecast-funded, trend, concentration
  and risk-migration questions over that history.
- **The existing point-in-time MI still works.** The original flat single-CSV MI
  path is unchanged and continues to work alongside the new capabilities.

In short: **fragmented inputs → canonical snapshots → governed MI answers**, with
backward compatibility preserved.

---

## 2. Source artefacts used (synthetic)

The proof uses small, deterministic **synthetic** files (no real client data),
mirroring how lenders/servicers actually hold data in separate systems:

| Artefact | Represents |
|---|---|
| `borrowers.csv` | Borrower attributes (age, single/joint, region) |
| `loans.csv` | Loan facts per reporting date (balance, rate, funded status, product, amortisation, dates, risk grade, IFRS 9 stage, PD bucket) |
| `collateral.csv` | Security/property (value, LTV, property region) |
| `cashflows.csv` | Period cashflow & arrears (due/paid, arrears balance & status) |
| `portfolio_map.csv` | Trakt portfolio / SPV / acquired-portfolio mapping |
| `pipeline.csv` | Unfunded pipeline opportunities (expected balance, stage, conversion probability, broker/channel) |

Across **three reporting dates** (`2024-01-31`, `2024-02-29`, `2024-03-31`) so
movement and trends can be shown.

---

## 3. Consolidation logic (deterministic)

The artefacts are joined with simple, explicit rules:

- **`borrower_id`** — borrowers join onto loans (and onto pipeline opportunities
  where a borrower is known).
- **`loan_id`** — collateral and portfolio mapping join onto loans.
- **`loan_id` + `reporting_date`** — cashflows/arrears join onto the loan for the
  correct period.
- **Portfolio mapping** — each loan is tagged with its Trakt `portfolio_id` /
  `portfolio_name`, `spv_id`, and `acquired_portfolio_id`.
- **Pipeline opportunity namespace** — pipeline opportunities are kept
  **separate** from funded loans (an opportunity id is never confused with a
  funded loan id), so funded and forecast views never double-count.

The output is **one canonical MI snapshot per reporting date**, carrying the
fields the MI runtime needs (balances, rate, LTV, region, product, risk grade,
IFRS 9 stage, PD bucket, arrears, portfolio/SPV, pipeline stage, conversion
probability, and a derived forecast-funded amount and time-on-book).

---

## 4. MI questions proven

Each of these is produced through the **same governed `run_mi_query`** entry
point over the consolidated snapshots:

| Question | What it shows |
|---|---|
| **Total funded** | The funded book at the latest reporting date |
| **Total pipeline** | The unfunded pipeline at the latest reporting date |
| **Total forecast-funded** | Funded book + probability-weighted pipeline |
| **Funded trend** | Funded balance across the three reporting dates |
| **Pipeline trend** | Pipeline balance across the three reporting dates |
| **Forecast-funded trend** | Forecast-funded balance across the three dates |
| **Funded by portfolio** | Funded balance split by Trakt portfolio |
| **Funded by region** | Funded balance split by region |
| **Pipeline by stage** | Pipeline split by funnel stage (KFI/application/offer/…) |
| **Concentration warning** | Group share vs limits, with green/amber/red status |
| **Risk grade migration** | How internal risk grades moved between two dates |
| **IFRS 9 migration** | How IFRS 9 stages moved between two dates |
| **PD bucket migration** | How PD buckets moved between two dates |

Migrations correctly identify **deterioration** (e.g. a loan moving grade B→C,
IFRS 9 Stage 1→Stage 2, and to a worse PD bucket) using configured orderings.

---

## 5. Governance proof

The proof is deliberately **governed and auditable**, not a black box:

- **Structured issues, not crashes.** Missing optional artefacts, unmatched rows
  (e.g. collateral referencing an unknown loan), missing keys, etc. are reported
  as explicit, typed issues — the pipeline degrades gracefully instead of
  failing silently.
- **Lineage metadata.** Each key consolidated field records which artefact it
  came from (e.g. LTV ← collateral, arrears ← cashflows, portfolio ← portfolio
  map, pipeline stage ← pipeline) — provenance is captured.
- **The snapshot layer is never bypassed.** Every state/temporal/risk query
  resolves and loads through the snapshot store; a query with no store fails with
  a clear `snapshot_store_required` issue.
- **Governed chart factory only.** Charts are requested only through the existing
  permitted MI chart library (e.g. bar for distributions, line for trends); no
  free-form chart types were introduced.
- **No Streamlit chart code copied.** The legacy dashboard code was not lifted in.
- **No Azure dependency.** Everything runs locally; no cloud SDK is imported.
- **No Annex 2 / regulatory changes.** The regulatory delivery path is untouched.
- **No production-engine claim.** The consolidation joins and lineage are
  hard-wired for the synthetic fixture to prove the *shape* of the flow — this is
  a proof, not a production mapper.

---

## 6. What remains unproven (explicit)

This pack does **not** yet prove, and does not claim:

- **Production source mapping** — handling arbitrary client schemas, column-name
  drift, and multi-file fan-in per artefact type.
- **Onboarding orchestration** — the automated process that ingests, validates
  and assigns Trakt portfolio references to real client artefacts.
- **Azure ingestion** — cloud upload, Event Grid/blob triggers, or a cloud
  snapshot adapter.
- **Full UI** — a product user interface; outputs here are governed data +
  chart instructions, not a rendered app.
- **M&A Agent runtime** — the point-in-time diligence agent.
- **Production lineage engine** — lineage here is a lightweight metadata dict,
  not a full field-level lineage/audit system.
- **Production data-quality workflow** — exception triage, remediation, and
  sign-off beyond the structured-issue surface.

---

## 7. Suggested demo script

A simple, honest 5-step narrative for a live or recorded demo:

1. **"Here is how the data really arrives."** Show the six separate synthetic
   artefacts (borrowers, loans, collateral, cashflows, portfolio map, pipeline)
   across three months — fragmented, exactly as lenders hold it.
2. **"Trakt consolidates it into a clean monthly view."** Run the consolidation
   step; show one canonical snapshot per reporting date and the lineage note for
   a couple of fields ("LTV came from collateral; arrears from cashflows").
3. **"It's stored as recurring history."** Show three snapshots registered in the
   snapshot store and that we can pick the latest, a point in time, or a range.
4. **"Now ask the MI questions."** Through one governed entry point:
   - current state: **total funded / pipeline / forecast-funded**;
   - **trends** over the three months (funded, pipeline, forecast-funded);
   - **breakdowns**: funded by portfolio, funded by region, pipeline by stage.
5. **"And the risk/early-warning view."** Show a **concentration warning**
   (green/amber/red) and **risk migration** (grade / IFRS 9 / PD) between two
   months — pointing out the deteriorating loan.
6. **"Explain the controls."** Close on governance: structured issues instead of
   silent failures, lineage, the snapshot store as the single source of history,
   the governed chart library, and that the existing point-in-time MI still
   works unchanged.

> Honesty note for the demo: state clearly that the consolidation is a
> deterministic synthetic proof, not the production mapper — what is proven is
> the *flow and governance*, not production source-handling.

---

## Appendix — technical references

| Reference | What it is |
|---|---|
| [`docs/phase6b_mi_runtime_smoke_pack.md`](./phase6b_mi_runtime_smoke_pack.md) | Phase 6B — MI runtime smoke pack over canonical snapshot frames |
| [`docs/phase6c_multi_artifact_consolidation_proof.md`](./phase6c_multi_artifact_consolidation_proof.md) | Phase 6C — multi-artefact consolidation proof |
| `tests/test_phase6b_mi_runtime_smoke_pack.py` | 6B runtime smoke tests (states, trends, concentration, migration, flat) |
| `tests/test_phase6c_multi_artifact_consolidation.py` | 6C consolidation + runtime tests |
| `tests/helpers/phase6c_consolidation.py` | Deterministic consolidation helper (load → join → lineage → register) |
| `tests/fixtures/phase6c_multi_artifact/*.csv` | Synthetic source artefacts (borrowers, loans, collateral, cashflows, portfolio_map, pipeline) |
| `docs/phase6_mi_runtime_integration.md` | Phase 6 — the governed MI runtime boundary and Step 0 semantics |

*Documentation only. No runtime, Azure, onboarding, M&A, or Annex 2/regulatory
code was added or changed.*
