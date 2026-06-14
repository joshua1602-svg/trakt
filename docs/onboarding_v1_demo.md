# Onboarding Agent v1 — Review Workbench, Mapping Memory & End-to-End Demo

This guide explains, in plain language, what the v1 Onboarding Agent demo does,
how the review workbench works, and how client-specific mapping memory cuts down
repeat work. No prior knowledge of the internals is required.

> **Not included yet (by design):** monthly drift detection and the live Azure
> Blob / Event Grid wrapper. This pass produces *Azure-ready* manifests and a
> trigger JSON, but never uploads to Azure or runs the downstream Gates 1–5.

---

## The story in one paragraph

A lender sends us a data room — a folder of CSVs and a funding agreement. The
Onboarding Agent reads those files, works out which business **domains** they
cover (loan, borrower, collateral, cashflow, pipeline, warehouse terms), proposes
how each source column maps to our canonical fields, and raises **gap questions**
about anything it cannot decide on its own. A junior analyst opens the run in the
**workbench**, answers the questions, fixes any mappings, and saves the important
decisions into **client memory**. We then build a consolidated **central lender
tape** and **central pipeline tape**, write the lineage and remaining gaps, and
produce an **Azure-ready trigger** that a future pipeline could pick up. Next
month, the same client's run reuses the saved memory and asks fewer questions.

---

## What the demo does

`engine/onboarding_agent/demo_onboarding_v1.py` runs the whole story on synthetic
data and prints a plain-English summary.

```bash
python -m engine.onboarding_agent.demo_onboarding_v1 \
  --output-dir onboarding_output/demo_onboarding_v1 \
  --client-id demo_client \
  --run-id demo_run_001
```

Step by step, the demo:

1. **Cleans** a demo project directory.
2. **Runs onboarding** on `synthetic_onboarding_pack_domain_based/scenario_a_combined`.
3. **Generates** the review pack (`08_onboarding_review_pack.html`) and gap
   questions (`07_gap_questions.yaml`).
4. **Applies a demo answers file** (`25_workbench_answers.yaml`) that closes the
   key gaps:
   - chooses the authoritative reporting date;
   - picks the source of truth for the balance conflict;
   - handles `employment_status = "manual"` (treat as missing) and `PART_TIME`
     (map to `OTHR`);
   - accepts / marks-unavailable the missing valuation / rate / originator fields;
   - confirms the ESMA UK geography policy.
5. **Saves selected decisions** into client mapping memory (under
   `onboarding_output/demo_client/client_memory/`).
6. **Re-runs mapping with memory applied** to show fewer unresolved gaps.
7. **Promotes (dry-run)** — builds the central tapes, lineage, gaps, promotion
   manifest and pipeline trigger.
8. **Prints a final summary.**

Example output:

```
Demo completed.
Input files: 4
Domains detected: borrower, cashflow, collateral, loan, pipeline
Central lender tape rows: 8
Central pipeline rows: 4
Blocking gaps before answers: 8
Blocking gaps after answers: 0
Unresolved mapping gaps before memory: 10
Unresolved mapping gaps after memory: 7
Client memory entries saved: 4
Pipeline trigger written: .../output/manifests/23_pipeline_trigger.json
Readiness: ready_for_pipeline
Ready for MI: yes
Ready for regulatory projection: yes
```

---

## What files the demo reads

From `scenario_a_combined`:

| File | What it carries |
| --- | --- |
| `master_loan_collateral_tape.csv` | Loan, borrower **and** collateral fields in one tape |
| `cashflow_report.csv` | Per-period payment / balance schedule |
| `pipeline_report.csv` | Applications / origination pipeline (some not yet funded) |
| `warehouse_funding_agreement.md` | Warehouse facility terms (advance rate, margin, …) |

## What domains it detects

`loan`, `borrower`, `collateral`, `cashflow`, `pipeline` (and `warehouse_terms` /
`securitisation_terms`, which are out of scope for `regulatory_mi`). Domains
follow the **canonical fields**, not the files — one combined master tape can
cover loan + borrower + collateral at once.

## What mappings it applies

Deterministic-first: exact / normalised / alias matches from the existing Gate 1
engine, then context hints, then the regulatory-preference ambiguity rule.
Anything still uncertain becomes a gap question rather than a guess.

## What gaps it finds (and how answers close them)

Typical gaps for scenario A:

- **Reporting date** — two dates seen (`2026-01-31` loan tape vs `2026-02-01`
  cashflow). The answer picks the authoritative one.
- **Balance source of truth** — `current_balance` vs `principal_outstanding`
  disagree for some loans. The answer chooses a primary source; the choice is
  saved as **source-precedence memory**.
- **Enum issues** — `employment_status` contains `manual` (a placeholder) and
  `PART_TIME` (not a canonical code). Answers map / drop them; saved as **enum
  memory**.
- **Missing core fields** — e.g. `originator_name`, `interest_rate_type`. Answers
  mark them unavailable or supply a source.

Answers are written to `25_workbench_answers.yaml` and fed through the existing
answer-ingestion logic, producing the approved artefacts `10`–`15`.

## How mapping memory reduces repeat work

Approved decisions are saved per client under:

```
onboarding_output/{client_id}/client_memory/
  mapping_memory.yaml            # mapping_override / validation_only / out_of_scope / mark_unavailable
  source_precedence_memory.yaml  # which source wins a conflict
  enum_memory.yaml               # how odd enum values are handled
  ignored_columns.yaml           # columns to ignore
```

On the **next** run for the same client, memory is applied **after** any
project/run-approved overrides but **before** the generic alias/registry mapping:

1. project/run-approved overrides
2. **client mapping memory**
3. alias/registry deterministic mapping
4. ambiguity rules
5. gap generation / optional LLM

Memory is **client-scoped** (never global, never shared with other clients),
**mode-aware and field-scope-safe** (a remembered mapping to a field that is out
of scope for the current mode is rejected, not applied), and **never silently
overrides a material conflict** — if the new data no longer matches the evidence
that justified a remembered mapping, the agent keeps the mapping but flags it and
raises a warning gap:

> *Client memory says loan_amount maps to original_principal_balance, but values
> no longer match expected validation sources.*

The run summary and review pack report: *Client mapping memory loaded: yes/no*,
*Memory entries applied: N*, *Memory entries rejected/warned: N*.

---

## What the central tapes mean

- **Central lender tape** (`18_central_lender_tape.csv`) — one row per funded /
  live loan, with the consolidated loan + borrower + collateral fields and full
  lineage (`18b_*`) and remaining gaps (`18c_*`).
- **Central pipeline tape** (`18a_central_pipeline_tape.csv`) — one row per
  application in the origination pipeline. Application-only rows (no funded loan
  id) stay here and are never forced into the lender tape.

## What the Azure-ready trigger JSON is for

`23_pipeline_trigger.json` is a small Event-Grid-friendly message describing a
*ready* (or *blocked*) handoff: the client id, run id, mode, the central tape
URIs and the readiness flags. A future Azure trigger could consume it to start
the downstream Trakt pipeline. In this pass it is **only written locally** — no
upload happens.

---

## The review workbench

A lightweight Streamlit app for analysts:

```bash
python -m streamlit run engine/onboarding_agent/streamlit_onboarding_workbench.py
```

Sidebar inputs: `project_dir`, `client_id`, `run_id`, `mode`,
`regulatory_reporting_enabled`. It is **file/artefact based** — no database.

Sections: **Overview · Domains · Mappings · Gaps · Conflicts · Source precedence ·
Enums · Client memory · Actions · Readiness**.

Actions (each writes project-scoped artefacts and is recorded in
`26_workbench_action_log.json`):

- **Save pending decisions** → `24_workbench_pending_decisions.yaml`
- **Generate answers YAML** → `25_workbench_answers.yaml`
- **Ingest answers** → approved artefacts `10`–`15`
- **Save selected decisions to client memory**
- **Promote dry-run** → central tapes + manifests + trigger
- **Apply client memory and rerun mapping**
- **Refresh review pack**

The "change mapping" dropdown only offers canonical fields **in scope for the
selected mode** — MI-only will not offer regulatory non-core fields as targets.

---

## Running the tests

```bash
pytest tests/test_onboarding_workbench_smoke.py -q
pytest tests/test_onboarding_mapping_memory.py -q
pytest tests/test_onboarding_demo_v1.py -q
```

These cover the workbench loaders/serialisers, mapping-memory save/load/apply
(including client scoping, mode/field-scope safety and conflict warnings), and
the full end-to-end demo.
