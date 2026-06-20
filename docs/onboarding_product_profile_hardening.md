# Onboarding hardening — product-profile-aware defaults & capability readiness

## Problem

The `mi_only` onboarding flow blocked or asked for fields that are not required
for **base MI** when the product is equity release / lifetime mortgage —
`maturity_date`, `amortisation_type`, `funded_status`,
`current_outstanding_balance` (when `current_principal_balance` exists), risk
fields, and acquisition/SPV fields. The fix had to stay **asset-agnostic** and
**config-driven** — no ERE/M2L-specific code, no client names, no source-file or
tape-specific hacks — and must not weaken other asset classes or regulatory MI.

## Design summary

A new, config-driven **product-profile** layer expresses the structural
characteristics of a product family and a **capability-based readiness** layer
replaces "all-or-nothing" promotion.

1. **Product profile abstraction** — `config/asset/product_profiles.yaml`.
   Profiles are keyed by a stable `profile_id` (e.g.
   `equity_release_lifetime_mortgage`), matched by asset-class / product-type /
   evidence tokens (never file or client names). Each profile declares per-field
   **base-MI policies** (`not_applicable | derived | defaulted | optional |
   required`) and **capability field contracts**. The engine never removes a
   field from the registry and never relaxes `regulatory_reporting`.

2. **Detection / confirmation** — `product_profile.resolve_product_profile`.
   * Explicit config/operator selection → trusted, applied outright.
   * Detected ≥ `apply_confidence` (0.80) → applied, evidence recorded.
   * Detected in `[confirm_confidence (0.55), apply_confidence)` → *proposed for
     confirmation*, **not** applied.
   * Below `confirm_confidence` → no profile (generic stricter behaviour).

3. **Artefact-role distinction.** Funded loan extract / loan tape rows
   (`current_loan_report` / `historical_loan_report`) derive `funded_status =
   funded` unless a source status clearly contradicts it; `pipeline_stage` comes
   from a `pipeline_report` artefact. `pipeline_mi` only activates when a
   pipeline artefact is present. Every derivation records its source.

4. **Equity-release defaults/derivations** (applied only when the profile is
   active), all returned as auditable `DerivationRecord`s
   (field/method/value/source/confidence/rationale):
   * `maturity_date` → not applicable for base MI;
   * `amortisation_type` → defaulted to canonical enum `OTHR`
     (capitalising/roll-up/no scheduled amortisation);
   * `current_outstanding_balance` → satisfied by `current_principal_balance`
     (or equivalent current-balance field) because interest capitalises;
   * `funded_status` → from the funded-extract artefact role;
   * `months_on_book` → from `origination_date` + `reporting_date`;
   * `number_of_borrowers` → from borrower/person/applicant fields only, **never**
     a unique loan-id count; non-blocking when unavailable;
   * risk fields → non-blocking for base MI; required only for
     `risk_migration` / `risk_monitor`;
   * `acquired_portfolio_id` / `acquisition_date` / `spv_id` → non-blocking unless
     M&A / SPV segmentation is required;
   * `portfolio_id` → may default to the client/run portfolio.

5. **Reporting-date inference** — `run_context.extract_data_cut_off_date` gains
   folder-period and run-id tiers: tape date → file name → folder period
   (`input/2025-10` → `2025-10-31`) → run id (`mi_2025_10` → `2025-10-31`) →
   config → CLI fallback. Conflicts/missing are still surfaced, never invented;
   every candidate is recorded with source + confidence.

6. **Capability-based readiness** — `capability_readiness.py` emits readiness per
   capability (`base_mi`, `pipeline_mi`, `risk_migration`, `risk_monitor`,
   `spv_segmentation`, `mna_segmentation`, `regulatory_reporting`). `mi_only`
   promotes when `base_mi` (and `pipeline_mi` where a pipeline artefact exists)
   is ready, even if `risk_migration` is unavailable. Missing optional fields
   stay visible as non-blocking gaps.

7. **Safety.** `regulatory_mi` is never relaxed (profiles only touch base/optional
   MI capabilities; `regulatory_reporting` keeps its own regime contract). Risk
   capabilities are reported `unavailable` when risk inputs are absent so
   downstream cannot run risk-migration queries. Nothing is fabricated silently;
   every default/derivation carries rationale + source + confidence. The registry
   is untouched.

## Integration

The profile plugs into the existing coverage seam: `product_profile.profile_overlay_rules`
emits overlay rules shape-compatible with `target_coverage.load_mi_applicability_overlay`,
which now merges an **applied** profile's policies (gap-fill only — the static
YAML overlay always wins). A no-source field the profile marks
not_applicable / derived / defaulted / optional is therefore no longer reported
as `missing_required`, so it stops being a base-MI blocker.

## Files changed

* `config/asset/product_profiles.yaml` *(new)* — config-driven profile registry.
* `engine/onboarding_agent/product_profile.py` *(new)* — loader, resolver,
  defaults/derivations, overlay-rule emitter; all auditable.
* `engine/onboarding_agent/capability_readiness.py` *(new)* — per-capability
  readiness + promotion decision.
* `engine/onboarding_agent/run_context.py` — folder-period / run-id reporting-date
  tiers.
* `engine/onboarding_agent/target_coverage.py` — `load_mi_applicability_overlay`
  merges an applied product profile (gap-fill, additive).
* `tests/test_onboarding_product_profile.py` *(new)* — 32 tests.
* `tests/test_run_context_date.py` — +7 folder/run-id inference tests.

## Tests run

* `tests/test_onboarding_product_profile.py` + `tests/test_run_context_date.py` —
  **55 passed**.
* Full `test_onboarding_*` suite — **555 tests, 11 failures, identical to the
  baseline before this change** (pre-existing environment/version drift, e.g.
  registry-count assertions like `99 != 72`); **no new failures introduced**.

## Before / after blocker count

`tests/test_onboarding_product_profile.py::test_before_after_blocker_count`
quantifies it on the canonical equity-release base-MI fields
(`maturity_date`, `amortisation_type`, `funded_status`,
`current_outstanding_balance`, `ifrs9_stage`, `spv_id`):

| | blockers |
|---|---|
| generic (no profile) | 6 / 6 |
| equity-release profile applied | **0 / 6** |

A dedicated October/November input pack is not present in the repository, so the
reduction is demonstrated on the canonical field set plus the live coverage
classifier (`target_coverage._classify`) rather than a shipped pack.

## Remaining blockers / notes

* No bundled `input/2025-10` (October) or `2025-11` (November) pack exists in the
  repo; the period-inference behaviour is proven by unit tests instead.
* The 11 pre-existing onboarding-suite failures are unrelated to this change
  (present on a clean checkout under the same interpreter) and are left as-is.
* `regulatory_mi` deliberately remains strict — promotion there still requires
  `regulatory_reporting` readiness, which no profile relaxes.
