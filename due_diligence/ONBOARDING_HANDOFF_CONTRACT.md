# Onboarding Handoff Contract

## Purpose

The Onboarding Agent produces a **governed canonical handoff package**. It is the
controlled, lineage-rich, approved boundary between onboarding and the rest of the
pipeline:

```text
Raw client files
→ Onboarding Agent
→ governed canonical handoff package        ← THIS DOCUMENT
→ Transformation & Validation Agent
→ Projection Agent
→ Delivery / XML / XSD / reporting outputs
```

The Onboarding Agent **does not**:

- own Annex 2 XML input generation;
- create bespoke MI / regulatory / investor delivery tapes;
- re-run raw Gate 1 source canonicalisation on its own central tape.

It **stops** after producing a package the Transformation & Validation Agent can
consume without rerunning raw source discovery or fuzzy mapping.

## Handoff artefacts

Produced under the run output at:

```text
onboarding_output/<client_id>/<run_id>/output/handoff/
```

| Artefact | Meaning |
| --- | --- |
| `24_onboarding_handoff_manifest.json` / `.yaml` | Governed manifest: identity, governance flags, artefact references, downstream counts, readiness. |
| `25_onboarding_handoff_readiness.json` / `.md` | Handoff readiness (separate from Annex 2 XML readiness). |
| `26_onboarding_handoff_field_contract.csv` / `.json` | Row-level contract for every target field in 28a, with a downstream classification + owner. |
| `27_onboarding_handoff_lineage.json` | Source → canonical → target lineage for each material field, with operator/LLM decision links. |

These are **additive**. They never mutate existing onboarding outputs
(`28a/28c/34/35/40/42–50`, `08_onboarding_review_pack.html`,
`output/central/18_central_lender_tape.csv`).

## The manifest declares what the package IS and IS NOT

```text
handoff_type                       = canonical_onboarding_package
handoff_stage                      = post_onboarding_pre_transformation_validation
next_agent                         = transformation_validation
not_raw_source                     = true
not_xml_ready                      = true   (until downstream produces an XML-ready frame)
do_not_rerun_gate1_on_central_tape = true
```

## Readiness logic (separate from XML readiness)

`ready_for_transformation_validation = true` when:

- the central canonical tape exists;
- `28a_target_coverage_matrix` exists;
- the target universe has loaded;
- registry gaps are zero or explicitly classified;
- there are no blocking onboarding decisions;
- unresolved items are classified for downstream handling;
- LLM recommendations, if present, are advisory-only and not required.

`ready_for_projection = false` while there are pending regime rules, unresolved
semantic derivations, not-yet-materialised transformation outputs, or unresolved
target-specific validations.

`ready_for_xml_delivery = false` unless the Transformation & Validation and
Projection stages have produced an XML-ready target frame. **Onboarding never
sets this true.**

## Field classification vocabulary

`handoff_classification` (controlled): `source_mapped`,
`source_mapped_with_alternatives`, `operator_decision_pending`,
`approved_decision_applied`, `configured_static`, `default_downstream`,
`nd_default_downstream`, `pending_regime_rule`, `source_absent`, `alias_mismatch`,
`semantic_derivation_required`, `transformation_required`, `projection_required`,
`delivery_required`, `not_applicable`.

`downstream_owner` (controlled): `onboarding`, `transformation_validation`,
`projection`, `delivery`, `operator`.

Examples:

- A valid asset default (`exposure_currency_denomination = GBP`,
  `interest_rate_type = Fixed`, `amortisation_type = Bullet`) with no source →
  `default_downstream` / `transformation_validation` /
  `materialise_default_from_asset_config`.
- A valid ND default → `nd_default_downstream` / `transformation_validation` /
  `materialise_nd_default_if_still_unmapped`.
- A field needing a regime rule not yet implemented → `pending_regime_rule` /
  `projection` / `implement_or_defer_regime_rule`.
- `current_outstanding_balance -> current_principal_balance` is **not** silently
  aliased. It is `semantic_derivation_required` /
  `transformation_validation` / `define_approved_ERM_balance_derivation_or_operator_decision`,
  because equity-release outstanding balance may include rolled-up interest,
  fees, advances and product-specific accrual mechanics.

## IMPORTANT — preventing the previous failure mode

The central lender tape is a **canonical onboarding handoff artefact**. It is not
raw source input and not an XML-ready regulatory delivery tape.

> **Do not run raw Gate 1 canonicalisation on
> `output/central/18_central_lender_tape.csv`.**
>
> Use the onboarding handoff package
> (`output/handoff/24_onboarding_handoff_manifest.json`) as input to the
> Transformation & Validation Agent.

Running `trakt_run` (raw Gate 1) on `18_central_lender_tape.csv` as if it were raw
client input re-runs canonicalisation and fails before product defaults, ND
defaults, validation and projection logic can be applied — producing misleading
"missing core field" errors (e.g. `amortisation_type`, `interest_rate_type`,
`exposure_currency_denomination`, `maturity_date`, `current_principal_balance`,
`property_post_code`). Several of those are resolved downstream by the
Transformation & Validation Agent via product/regime defaults; others are
alias/derivation decisions. They must not be treated as raw client-file failures
during onboarding — the field contract (`26_*`) classifies each one for its
correct downstream owner.

The manifest carries `do_not_rerun_gate1_on_central_tape = true` so an
orchestrator can assert this guard programmatically before invoking raw Gate 1.
