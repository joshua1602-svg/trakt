# Onboarding date semantics — funded-book reporting date vs pipeline snapshot date

## Problem

Onboarding treated `reporting_date` too generically. In an MI package the loan,
collateral and cashflow tapes share one funded-book reporting date, but the
origination-pipeline file can legitimately carry a *different* snapshot date
(it drives forward exposure / forecast). The flow must separate the two,
asset-agnostically, without weakening `regulatory_mi` or the existing
`reporting_date` / `data_cut_off_date` semantics.

## Design summary

1. **`reporting_date` / `funded_reporting_date`.** `reporting_date` stays the
   canonical base field. `funded_reporting_date` and friends are added as
   **synonyms** of `reporting_date` (not a new canonical field): in the MI
   semantics registry and in the date-semantics alias set —
   `funded_reporting_date`, `funded book reporting date`, `funded as-of date`,
   `loan tape reporting date`, `loan extract reporting date`, `book date`,
   `cut-off date`, `data cut-off date`.

2. **`pipeline_snapshot_date`.** A new, separate **MI/pipeline** date in the MI
   semantics registry (`source_criteria: pipeline_state`, `role: date`,
   `virtual: true`) — *not* a regulatory core field. Synonyms: `pipeline snapshot
   date`, `pipeline as-of date`, `pipeline extract date`, `pipeline report date`,
   `application pipeline date`, `kfi pipeline date`.

3. **Artefact date semantics** (`engine/onboarding_agent/date_semantics.py`):
   * `current_loan_report` / `historical_loan_report` → funded basis →
     `reporting_date`;
   * `collateral_report` → funded basis (must align to funded reporting date);
   * `cashflow_report` → funded basis;
   * `pipeline_report` → `pipeline_snapshot_date`.

4. **Consistency validation.** Funded loan/collateral/cashflow artefacts must
   share one `funded_reporting_date`; conflicting dates raise a **blocking**
   `date_basis_mismatch` (downgraded to a recorded, non-blocking note only when
   explicitly approved). A differing `pipeline_snapshot_date` is **allowed** and
   recorded as a non-blocking `pipeline_vs_funded_difference`. The funded vs
   pipeline difference is written into the summary explicitly.

5. **Inference behaviour.** Date alias/inference is role-aware: a date in a
   pipeline file / filename maps to `pipeline_snapshot_date`; a date in a
   loan/collateral/cashflow file maps to `reporting_date`. A funded-style token
   inside a pipeline artefact still resolves to `pipeline_snapshot_date`.
   Pipeline `2025-12-01` with a funded package `2025-11-30` is valid; collateral
   `2025-12-01` with loan `2025-11-30` is flagged unless approved.

6. **Folder / manifest conventions.** Both supported and never assume one parent
   folder date applies to all artefacts:
   * role/date folders — `input/funded/2025-11-30/`, `input/pipeline/2025-12-01/`;
   * explicit run manifest — `mi_package: { funded_reporting_date,
     pipeline_snapshot_date }`. Per-artefact resolution precedence: manifest →
     role/date folder → detected column date → file name, each auditable.

7. **Safety.** `regulatory_mi` and the regulatory `data_cut_off_date` aliases are
   untouched; `reporting_date` semantics are preserved; `pipeline_snapshot_date`
   is never forced onto funded records; funded date mismatches never pass
   silently; every inferred/defaulted date carries source + confidence + evidence.

## Files changed

* `mi_agent/build_mi_semantics_registry.py` — add `pipeline_snapshot_date` virtual
  field; add funded-book reporting-date synonyms to `reporting_date`.
* `mi_agent/mi_semantics_field_registry.yaml` — regenerated (99 → 100 fields;
  core 67 → 68).
* `engine/onboarding_agent/date_semantics.py` *(new)* — role basis, alias
  resolution, folder/manifest parsing, per-artefact date assignment, consistency
  validation.
* `tests/test_onboarding_date_semantics.py` *(new)* — 21 tests.
* `mi_agent/tests/test_mi_semantics_buckets.py` — field-count expectation updated
  (99 → 100, core 67 → 68) for the intentional new field.

## Tests run

* `tests/test_onboarding_date_semantics.py` — **21 passed**.
* `mi_agent` full suite — **135 passed** (incl. updated bucket counts).
* Combined new/touched onboarding tests (date_semantics + run_context +
  product_profile) — **76 passed**.
* Full `test_onboarding_*` suite — **576 tests, 11 failures identical to the
  pre-existing baseline** (environment/version drift, e.g. Annex 2
  `target_fields_total` and category counts); **no new failures introduced**.

## Before / after on the October/November shape

| Scenario | Before | After |
|---|---|---|
| funded loan/collateral/cashflow all `2025-11-30` | single generic `reporting_date`; pipeline conflated | one `funded_reporting_date = 2025-11-30`; **passes** |
| pipeline `2025-12-01` with funded `2025-11-30` | risked a date conflict/block or silent overwrite | `pipeline_snapshot_date = 2025-12-01` kept **separate**; **allowed**, recorded non-blocking |
| collateral `2025-12-01` vs loan `2025-11-30` | could pass silently under a single generic date | **blocking** `date_basis_mismatch` (unless explicitly approved) |
| date in pipeline filename | mapped to `reporting_date` | mapped to `pipeline_snapshot_date` |

(There is no shipped `input/2025-10` / `2025-11` pack in the repo; the behaviour
is proven on this shape via unit tests against the live resolver/validator.)

## Remaining notes

* The 11 pre-existing onboarding-suite failures are unrelated to this change
  (present on a clean checkout under the same interpreter/deps).
* `regulatory_mi` stays strict; `data_cut_off_date` (RREL6) regulatory aliases are
  unchanged.
