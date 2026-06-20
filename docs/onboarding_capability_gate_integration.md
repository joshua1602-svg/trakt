# Onboarding hardening — wiring product-profile / capability readiness into the blocking gates

## Problem

The product-profile, capability-readiness and date-semantics layers existed but
were **standalone** — they did not control the live blocking gates. On the real
Oct/Nov run the workflow still reported 56 blocking Gate-4 decisions and 11
blocking `core_field` gaps, because:

* `target_coverage.load_mi_target_contract` marks **every `core`-tier MI field
  `required`**, so every unmapped MI dimension became a blocking
  `missing_required` decision (28c / 34); and
* `run_target_first_coverage` loaded the applicability overlay **without** the
  resolved product profile; and
* `gap_analyzer._missing_core_field_questions` marked every in-scope core
  canonical field blocking, ignoring product-profile policy.

## Fix

The standalone layers are now bound into both gates, asset-agnostically and
config-driven.

### System B — Gate 4 / 28c / 34 (`target_coverage.run_target_first_coverage`)

1. Resolve the product profile from the run context (explicit `product_profile`
   else evidence-based detection) and pass it to `load_mi_applicability_overlay`
   so explicit field policies (`not_applicable` / `derived` / `defaulted` /
   `optional`) classify no-source fields as **non-blocking**.
2. New `apply_mi_capability_scope`: for an **applied** profile in an MI mode, a
   missing target is a base-MI blocker **only** if it is in the profile's
   `base_mi` capability contract. Every other still-`required` core MI field is
   downgraded to `optional` (visible, non-blocking), tagged with the capability
   that actually needs it.
3. New audit artefact `28d_product_profile_scope.json`: profile resolution
   (decision/confidence/evidence), every capability-scope change with rationale,
   and the per-capability readiness + promotion decision. The result is also
   returned in-band so 28c/34 reflect it (not just a standalone artefact).

### System A — 07 gap questions (`gap_analyzer`)

`analyze_gaps` / `_missing_core_field_questions` take the resolved profile; an
in-scope core field the profile marks not_applicable/defaulted/derived/optional
is demoted from `blocking` to a visible `high` gap (rationale recorded). Demotion
is **gated to MI modes only** — `regulatory_mi` keeps strict core requirements.

### Profile resolution guard

`onboarding_context.detect_context` now returns `asset_signal_strength`;
`resolve_product_profile` refuses to apply a profile when that strength is `0`
(asset_class was a default guess, not evidenced), so a generic non-equity pack
keeps stricter generic behaviour. The strength is propagated through the context
backstop so System B honours it too.

## Specific field behaviour (mi_only + equity_release)

| Field | Base-MI treatment |
|---|---|
| `maturity_date` | not applicable → non-blocking |
| `amortisation_type` | defaulted (`OTHR`) → non-blocking |
| `funded_status` / `account_status` | derived / optional → non-blocking |
| `current_outstanding_balance` | satisfied by `current_principal_balance` (equiv group) |
| `months_on_book` | derived from `origination_date` + `reporting_date` |
| `pipeline_snapshot_date` | not a base-MI blocker; required only for `pipeline_mi` when a pipeline artefact exists; inferred from pipeline folder/date/filename |
| risk fields (ifrs9_stage, PD, LGD, EAD, internal grade) | non-blocking; disable `risk_migration`/`risk_monitor`, not base MI |
| `spv_id`, `acquisition_date`, `acquired_portfolio_id` | non-blocking unless that segmentation is required |
| `originator_name` / LEI / `interest_rate_type` | non-blocking for mi_only |
| loan id, principal/current balance, currency, reporting date | **remain genuine base-MI blockers** |

## Safety

* `regulatory_mi` untouched — capability scope is a no-op for non-MI contracts,
  the MI overlay never loads for Annex 2, and core-field demotion is gated out
  for `regulatory_mi` (verified: all core gaps stay blocking).
* No fields removed from the registry; not all missing MI fields marked
  required; profile only applied with positive evidence/explicit selection.
* Every default/derivation/scope change is auditable (28d + gap rationales).

## Files changed

* `engine/onboarding_agent/target_coverage.py` — `apply_mi_capability_scope`;
  profile resolution + overlay merge + 28d audit in `run_target_first_coverage`.
* `engine/onboarding_agent/gap_analyzer.py` — profile-aware core-field severity
  (MI-mode-gated).
* `engine/onboarding_agent/onboarding_context.py` — `asset_signal_strength`
  (detector + backstop propagation).
* `engine/onboarding_agent/product_profile.py` — `base_mi_required_fields`,
  no-evidence guard.
* `engine/onboarding_agent/onboarding_orchestrator.py` — deterministic profile
  resolution wired into `analyze_gaps`; `product_profile` run param.
* `config/asset/product_profiles.yaml` — added non-blocking policies for
  `originator_name`, `originator_legal_entity_identifier`, `interest_rate_type`,
  `account_status`.
* `tests/test_onboarding_capability_gate.py` *(new)* — 10 integration tests.
* `tests/test_onboarding_modes.py` — `test_mi_only_emits_missing_core_field_gaps`
  updated to the capability-aware contract (base fields block; profile-demoted
  fields visible/non-blocking).

## Tests run

* `tests/test_onboarding_capability_gate.py` — **10 passed**.
* `test_onboarding_modes` + `capability_gate` + `product_profile` +
  `date_semantics` — **74 passed**; `regulatory_mi` core gaps verified all
  still blocking.
* Full `test_onboarding_*` suite — **586 tests, 9 failures**: the established
  pre-existing baseline of 11 **minus the 2 this change fixes**
  (`test_blocking_materially_reduced_and_remaining_explainable`,
  `test_headline_status_from_28c_not_legacy`), with **no new failures**. The
  remaining 9 are stale registry-count / annex2-alignment assertions that
  predate this work (registry grew 72→100 earlier).

## Before / after on the Oct/Nov shape

Driving `run_target_first_coverage` on a mi_only equity-release pack with the base
fields mapped:

| | Gate-4 blocking decisions |
|---|---|
| generic context (no profile) | 50 |
| equity-release profile applied | 0 (only genuine, here all base fields mapped) |
| capability-scope downgrades recorded | 48 |

On the full synthetic equity pack the 07 `core_field` gaps drop from "all
blocking" to: genuine base blockers (`data_cut_off_date`, currency, …) blocking,
`amortisation_type`/`maturity_date`/originator/etc. demoted to visible. Remaining
blockers are only genuine base-MI fields, exactly as required.
