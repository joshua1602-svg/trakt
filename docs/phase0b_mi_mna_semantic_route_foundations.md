# Phase 0B — MI/M&A Semantic Registry & Route-Contract Foundations

**Status:** Implemented (config / registry / tests / docs only). No orchestration,
no snapshot/history layer, no MI states, no M&A Agent, no Streamlit chart
migration, no legacy analytics refactor.

**Date:** 2026-06-18

This phase lays the **declarative foundations** for the MI and M&A routes
described in `docs/mi_mna_target_architecture_and_build_plan.md`. It builds on
Phase 0 (`docs/phase0_risk_field_registry_changes.md`), which added the generic
canonical risk fields and analytics aliases. Phase 0B curates those fields into
the MI semantic layer and adds the route/state/stratification/bucket/scorecard
**config skeletons** that later phases will execute against.

---

## 1. What was added

### 1.1 MI semantic registry curation
Curated **27** MI semantic entries (build script `CURATION` →
`mi_agent/mi_semantics_field_registry.yaml`, regenerated; version `0.2.2 → 0.3.0`,
72 → **99** fields):

- **Risk-model fields (canonical, from Phase 0):** `amortisation_type`,
  `internal_risk_grade`, `internal_risk_score`, `internal_risk_stage`,
  `ifrs9_stage`, `probability_of_default`, `loss_given_default`,
  `exposure_at_default`.
- **Risk/borrower derived bands:** `pd_bucket`, `lgd_bucket`, `ead_bucket`,
  `borrower_structure` (single vs joint) — registered as first-class derived
  dimensions, consistent with the existing `ltv_bucket`/`age_bucket` pattern.
- **Segmentation / snapshot / state dimensions (VIRTUAL):** `portfolio_id`,
  `spv_id`, `acquired_portfolio_id`, `acquisition_date`, `spv_transfer_date`,
  `reporting_date`, `cut_off_date`, `upload_timestamp`, `pipeline_stage`,
  `funded_status`, `forecast_funding_date`, `forecast_funding_probability`,
  `forecast_funded_balance`, `number_of_borrowers`, `months_on_book`.

The build script (`mi_agent/build_mi_semantics_registry.py`) gained a small,
additive `virtual: True` concept (alongside the existing `derived: True`) plus an
explicit per-field `source_criteria` override. **Virtual** fields are recognised
MI dimensions that are **not** loan-level canonical fields — they belong to the
snapshot/state layer (build-plan Phase 2) or are derived in the state layer — and
are flagged so consumers can tell them apart from materialised loan-level fields.
A new `virtual_field_count` is emitted in the registry metadata.

### 1.2 Bucket & stratification config skeletons
- `config/mi/buckets.yaml` — edge/scale definitions for **LTV**, **borrower age**,
  **youngest borrower age**, **interest rate**, **PD**, **LGD**, **EAD**,
  **balance band**, and **time on book** buckets (seeded from legacy
  `analytics/mi_prep.py` definitions as a reference, not imported).
- `config/mi/stratification_catalogue.yaml` — each dimension → semantic field →
  bucket rule → applicable states / asset classes.

### 1.3 Route config skeletons
`config/routes/{mi,mna,regulatory_annex2,regulatory_and_mi}_route.yaml`. Each
declares `route_id`, `temporality`, `requires_history`, `allowed_states`,
`allowed_dimensions`, `temporal_modes`, and the `risk_monitor` / `forecast` /
`exceptions_scorecard` capability switches (plus `allowed_segments` and, for the
combined route, a `composes` list).

### 1.4 MI config skeletons
`config/mi/state_library.yaml` (portfolio-state definitions) and
`config/mi/risk_monitor.yaml` (early-warning rule shapes), alongside the bucket
and stratification files above.

### 1.5 M&A config skeleton
`config/mna/diligence_scorecard.yaml` — field-completeness, exception-materiality
(reusing the existing exception engine as a read interface), buyer concentration
limits, and acquired-portfolio segmentation.

### 1.6 Tests
`tests/test_phase0b_mi_mna_foundations.py` (53 tests, yaml + stdlib only) proves:
new configs parse; route configs carry the required top-level keys; the registry
recognises every curated field; no duplicate route/state/dimension/bucket keys
(via a duplicate-key-detecting YAML loader); and no Streamlit/chart code is
imported or copied into any Phase 0B file. Existing registry tests
(`test_mi_semantics_buckets.py`, `test_mi_semantics_cleanup.py`) were updated for
the new counts/version. Full suite: **188 passed**.

---

## 2. What was intentionally NOT built

Per the brief's scope guardrails:

- **No orchestration / route layer code** — routes are declarative YAML only;
  nothing resolves a `RouteContract` at runtime yet.
- **No snapshot / history layer** — segmentation/date/state fields are registered
  as *virtual* semantic dimensions; the `SnapshotStore`, headers and persistence
  are deferred to build-plan Phase 2. They were deliberately **not** added to the
  loan-level canonical `fields_registry.yaml`.
- **No MI states** — `state_library.yaml` declares states; no state assembler
  exists.
- **No M&A Agent** — `diligence_scorecard.yaml` is a shape only.
- **No bucketing engine** — `buckets.yaml` declares edges; no code reads them.
- **No Streamlit chart migration** and **no legacy analytics refactor** —
  `analytics/` is untouched; legacy definitions were used only as a reference.

---

## 3. How this prepares for later phases

- **Phase 1 (shared analytics library, `analytics_lib/`):** `buckets.yaml` and
  `stratification_catalogue.yaml` give the bucketing/stratification modules their
  edge definitions and dimension→field→bucket mapping up front, so the library can
  be built and unit-tested against a fixed contract (with parity tests vs legacy
  `mi_prep` definitions).
- **Phase 2 (snapshot layer):** the *virtual* segmentation/snapshot dimensions
  (`portfolio_id`, `spv_id`, `acquired_portfolio_id`, `reporting_date`,
  `cut_off_date`, `upload_timestamp`, `acquisition_date`, `spv_transfer_date`)
  pre-name the snapshot header/loan-row contract from §6.1 of the target
  architecture, so the snapshot model can materialise them without re-litigating
  naming or semantics.
- **Phases 3-6 (MI states, temporal MI, risk monitor, M&A agent):** the route
  contracts, `state_library.yaml`, `risk_monitor.yaml` and
  `diligence_scorecard.yaml` define the envelopes those phases implement against;
  the validator's future contract check (target architecture §8) has concrete
  config to read.

*Config / registry / tests / docs only. No source code paths were wired to these
files; the legacy app and regulatory Annex 2 contract are unchanged.*
