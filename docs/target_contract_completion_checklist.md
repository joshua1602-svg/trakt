# Target Contract Completion Checklist / Target Field Disposition

**Onboarding owns target-field disposition. Downstream agents execute it.**

This is the architectural keystone that the later pipeline stages exposed: fields
such as `RREL40 debt_to_income_ratio`, `RREL24 maturity_date`,
`RREL1/RREL2 (formal identifiers)` should **not** first become understandable at
Projection time. For a chosen target contract, the Onboarding Agent now produces a
complete, field-by-field **disposition** answering, up front:

> For each Annex 2 target field, do we have — source data, an asset default, a
> client/lender policy default, a valid ND treatment, a derivation/calculation
> rule, a formal client onboarding requirement, an operator review decision, a
> projection-only rule, or a true unresolved/blocking gap?

```text
Onboarding   → decides the expected disposition for each target field
Transformation → materialises source / default / context values
Validation   → validates values + whether dispositions are acceptable/blocking
Projection   → constructs the target frame using approved dispositions
Delivery/XML → applies XSD / XML structure   (NOT in scope here)
```

## Artefacts

Written under `onboarding_output/<client>/<run>/`:

| Artefact | Content |
| --- | --- |
| `29_target_contract_completion_checklist.csv` | one row per target field / ESMA code, with the full disposition |
| `29_target_contract_completion_checklist.json` | same rows + a disposition-count summary |
| `29_target_contract_completion_checklist.md` | human-readable disposition + review-bench summary |
| `29a_target_contract_review_bench.csv` / `.json` | the unresolved rows, categorised for the future review bench/UI |

The handoff manifest (`24_*`) references the checklist via
`target_contract_completion_checklist_path` + `target_contract_review_bench_path`,
and the handoff field contract (`26_*`) carries the disposition columns so the
downstream agents consume them.

Module: `engine/onboarding_agent/target_contract_completion.py`.

## Disposition vocabulary (one per field)

`source_supplied`, `source_mapped_with_review`, `asset_default_supplied`,
`client_policy_default_supplied`, `configured_static_supplied`,
`nd_policy_selected`, `derivation_configured`, `calculation_configured`,
`projection_rule_required`, `client_onboarding_required`,
`operator_review_required`, `config_mapping_required`, `not_applicable`,
`unresolved_gap`.

Each row also carries secondary flags: `requires_client_input`,
`requires_operator_review`, `requires_config`, `requires_projection_rule`,
`requires_derivation`, and `blocking_for_{onboarding_handoff,transformation,validation,projection}`.

## ND allowed ≠ ND selected

This distinction is enforced:

* **ND allowed** = regulatory permission (`nd_allowed` in the regime / workbook).
* **ND selected** = an actual asset/client/config **policy** decision.

A field is **not** completed merely because ND is allowed. It is completed only
when a policy layer (client → asset → registry asset-applicability → regime
configured default) has actually *selected* a valid ND/default, or it is
explicitly deferred to projection with a known rule. Otherwise it is an
`unresolved_gap` and goes to the review bench — it does **not** silently ND.

## Asset policy vs client policy vs generic regime

ERM behaviour is **never** hard-coded into the engine or into generic Annex 2
regime logic. It is read from layered config:

```text
config/asset/product_defaults_ERM.yaml        # asset-class policy (reporting_policy:)
config/client/<client>_reporting_policy.yaml  # optional client/lender override (same shape)
config/system/fields_registry.yaml            # per-field applicability.<asset_class>
config/regime/annex2_delivery_rules.yaml      # regime envelope (nd_allowed / default_value)
config/regime/annex2_field_universe.yaml      # authoritative 107-field universe
```

Layer precedence for an ND/default selection: **client policy → asset
policy/defaults → registry asset applicability → regime configured default**.
The `disposition_source` column records which layer made the decision, so an
asset-driven ND is distinguishable from a generic regime fallback.

The asset `reporting_policy:` block added to `product_defaults_ERM.yaml`:

```yaml
reporting_policy:
  formal_client_onboarding_required:        # not expected in an ordinary loan tape
    - unique_identifier                      # RREL1
    - original_underlying_exposure_identifier # RREL2
  nd_policy:                                 # ND SELECTED for this asset class
    debt_to_income_ratio: ND5                # RREL40 — DTI not captured for ERM
    maturity_date: ND5                       # RREL24 — lifetime mortgage, no term
  not_applicable: []
```

## Worked examples

### RREL40 `debt_to_income_ratio`
For this equity-release lender, borrower income / DTI is not captured and is not a
relevant metric. RREL40 permits ND and is a percentage field.

* ERM (asset `nd_policy: ND5`) → `nd_policy_selected`, `disposition_source = asset_config`, not blocking.
* No asset policy, but the regime configures `default_value: ND5` → `nd_policy_selected`, `disposition_source = regime_config`.
* A field with **no** selection at any layer → `unresolved_gap`, `requires_config = true`, surfaced on the review bench.

It is **not** hard-coded to ND5: the value comes from the asset/client/regime
config, and the source layer is recorded.

### RREL24 `maturity_date`
* ERM / lifetime mortgage → `nd_policy_selected` (`ND5`) via the asset policy / registry `applicability.equity_release`.
* Traditional amortising asset (no policy) → `source_supplied` / `derivation_configured` / `unresolved_gap` depending on source/config — **never** generically ND5.

### RREL1 / RREL2 (formal identifiers)
Not present in uploaded files and not inferable from ordinary loan IDs →
`client_onboarding_required`, `requires_client_input = true`,
`blocking_for_validation = true`, `blocking_for_projection = true`,
`owner = client_onboarding`. Recommended action: request the formal regulatory
exposure identifiers from the client.

### RREC9 `property_type` / RREL27 `purpose`
* source present + enum mapping complete → `source_supplied`;
* multiple plausible source columns → `operator_review_required`;
* source values present but enum mapping incomplete → `config_mapping_required`
  (a **config** blocker, never misclassified as a data failure);
* no source and a selected ND/default policy → `nd_policy_selected`; otherwise
  `operator_review_required` / `unresolved_gap`.

## Review bench

Every unresolved/review disposition becomes a `29a` review-bench item, categorised
so the future UI can distinguish:

| Category | Meaning |
| --- | --- |
| `operator_decision` | human mapping ambiguity / confirmation |
| `client_required_input` | formal client onboarding field missing |
| `asset_policy_required` | asset/client policy not yet selected |
| `config_required` | enum/config mapping missing |
| `projection_rule_required` | projection-only rule not yet implemented |

## Downstream consumption (execute the disposition)

The disposition flows `26 (handoff contract) → 32 (transformation contract) →
Validation / Projection`. Pure mappers in the module translate a disposition into
each agent's vocabulary (`transformation_action_for_disposition`,
`validation_classification_for_disposition`, `projection_status_for_disposition`):

* **Transformation** — `asset_default_supplied` → materialise the asset default;
  `nd_policy_selected` → materialise the selected ND; `client_onboarding_required`
  → do not source-map/default, carry forward as client input required. The
  disposition is written into `32_transformation_field_contract.csv`
  (`field_disposition` / `disposition_action`).
* **Validation** — `config_mapping_required` → config blocker (not a data
  failure); `client_onboarding_required` & missing → validation blocker owned by
  `client_onboarding`; `nd_policy_selected` & materialised → validate ND allowed.
* **Projection** — `source_supplied` → project the source value;
  `nd_policy_selected` → project ND/default; `projection_rule_required` → carry as
  a projection blocker; `client_onboarding_required` → carry as a
  `blocked_client_onboarding_dependency` (see `56_projection_blocker_resolution`),
  never rediscovered as a generic source-mapping gap.

## Guardrails

* Additive — never mutates upstream onboarding artefacts; the checklist build is
  wrapped so any config/load failure yields an empty checklist and the handoff
  still succeeds.
* No XML, no delivery normalisation, no UI in this layer — `29a` only *prepares*
  the future review bench/UI by making the categories explicit.
* Asset behaviour stays in config, never in generic engine/regime code.

## Tests

```bash
pytest tests/test_target_contract_completion_checklist.py -q
```
