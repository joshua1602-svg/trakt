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
`formal_identifier_policy_required`, `operator_review_required`,
`config_mapping_required`, `asset_policy_required`, `not_applicable`,
`unresolved_gap`.

Each row also carries secondary flags: `requires_client_input`,
`requires_operator_review`, `requires_config`, `requires_projection_rule`,
`requires_derivation`, `requires_enum_mapping`,
`requires_formal_identifier_policy`, `requires_asset_policy`, and
`blocking_for_{onboarding_handoff,transformation,validation,projection}`.

## Candidate found ≠ approved field treatment complete

The classifier separates these concepts and never confuses a *candidate* for a
*completed* treatment:

```text
source_candidate_found → source_selected → source_approved_for_target_field
                       → source_projectable → field_disposition_complete
```

A field is only `source_supplied` when the source is **selected, approved for
that ESMA field, sufficiently mapped/normalised, and not dependent on an
unresolved enum/config/projection rule**. Otherwise a more accurate disposition
is used: `source_mapped_with_review`, `config_mapping_required` (enum/projection
mapping incomplete), `projection_rule_required` (no rule maps the source yet),
`operator_review_required` (multiple plausible columns),
`formal_identifier_policy_required` / `client_onboarding_required`,
`asset_policy_required`, or `unresolved_gap`.

## Classifier priority order

```text
0. not_applicable
1. formal regulatory identifier without approved policy → formal_identifier_policy_required
2. deliberate client/asset policy selected (reporting_policy / registry applicability)
   — OVERRIDES source ambiguity (a field the product does not capture is ND-selected)
3. approved, complete, projectable source → source_supplied
   (enum incomplete → config_mapping_required; no regime rule → projection_rule_required)
4. derivation configured → derivation_configured
5. fallback configured default / ND (asset defaults / nd_defaults / regime default_value)
6. ambiguous (source_mapped_alt) → operator_review_required;
   single source flagged → source_mapped_with_review
7. selected ND/default outside the allowed envelope → config_mapping_required
8. deferred / pending → projection_rule_required
9. ND allowed but no policy selected → asset_policy_required   (ND allowed ≠ ND selected)
10. nothing → unresolved_gap
```

`source ambiguity` never overrides a deliberate asset/client policy that says the
field is not captured and should be ND-selected.

## ND allowed ≠ ND selected

This distinction is enforced:

* **ND allowed** = regulatory permission (`nd_allowed` in the regime / workbook).
* **ND selected** = an actual asset/client/config **policy** decision.

A field is **not** completed merely because ND is allowed. It is completed only
when a policy layer (client → asset → registry asset-applicability → regime
configured default) has actually *selected* a valid ND/default, or it is
explicitly deferred to projection with a known rule. Otherwise it is an
`asset_policy_required` item that goes to the review bench — it does **not**
silently ND.

**ND permission is taken from the authoritative field universe, merged with the
regime envelope.** The `nd_allowed` shown on the checklist is the union of the
workbook universe permission and the regime projection rule. (Diagnosis: the
regime `annex2_delivery_rules.yaml` had narrowed RREL40 to `[ND5]`, which was
narrower than the authoritative universe — RREL40 permits ND1–ND5 — and caused a
spurious "ND1 not in nd_allowed [ND5]" conflict. The union fixes this without
hard-coding.)

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
  formal_client_onboarding_required:         # not expected in an ordinary loan tape
    - unique_identifier                       # RREL1
    - original_underlying_exposure_identifier # RREL2
  # ordinary source columns explicitly APPROVED to satisfy a formal identifier
  # (otherwise a formal identifier stays client_onboarding-required):
  formal_identifier_source_approved: []
  nd_policy:                                 # ND SELECTED for this asset class
    debt_to_income_ratio: ND5                # RREL40 — DTI not captured for ERM
    maturity_date: ND5                       # RREL24 — lifetime mortgage, no term
  not_applicable: []
```

## Worked examples

### RREL40 `debt_to_income_ratio`
For this equity-release lender, borrower income / DTI is not captured and is not a
relevant metric. RREL40 permits ND (ND1–ND5 per the universe) and is a percentage
field.

* ERM (asset `reporting_policy.nd_policy: ND5`) → `nd_policy_selected`,
  `disposition_source = asset_config`, `requires_operator_review = false`,
  `blocking_for_projection = false`. This holds **even when the coverage matrix
  flags an ambiguity** — the deliberate "not captured" policy wins, so RREL40 is
  never an operator mapping mystery.
* Traditional lender that **captures** DTI → `source_supplied` (use the source,
  not ND5).
* No source and **no** asset/client policy (and no regime default) →
  `asset_policy_required`, `requires_asset_policy = true`, surfaced on the review
  bench.

It is **not** hard-coded to ND5: the value comes from the asset/client/regime
config, and the deciding layer is recorded in `disposition_source`.

### RREL24 `maturity_date`
* ERM / lifetime mortgage → `nd_policy_selected` (`ND5`) via the asset policy / registry `applicability.equity_release`.
* Traditional amortising asset (no policy) → `source_supplied` / `derivation_configured` / `unresolved_gap` depending on source/config — **never** generically ND5.

### RREL1 / RREL2 (formal identifiers)
Formal regulatory identifiers are **not** satisfied by ordinary loan / account /
policy IDs merely because such a column exists. By default →
`formal_identifier_policy_required`, `requires_client_input = true`,
`requires_formal_identifier_policy = true`, `blocking_for_validation = true`,
`blocking_for_projection = true`, `owner = client_onboarding`. Recommended action:
request/approve a formal regulatory identifier policy.

Only when an explicit approval exists (asset/client
`reporting_policy.formal_identifier_source_approved: [unique_identifier, …]`) is
`source_supplied` accepted for the ordinary column.

### RREL27 `purpose`
The source exists (e.g. `Remortgage`) but the ESMA enum mapping is not confirmed
complete, so the value is not yet safely projectable → `config_mapping_required`,
`requires_config = true`, `requires_enum_mapping = true`, `owner = config_policy`.
It is **not** marked `source_supplied` until the value is both selected and
safely projectable (enum/projection mapping present). An enum/LIST field that is
source-mapped is only `source_supplied` when the coverage confirms the enum
mapping is complete.

### RREC9 `property_type`
* source present + enum mapping confirmed complete → `source_supplied`;
* multiple plausible source columns → `operator_review_required`
  (`disposition_source = source_ambiguity`);
* source present but enum mapping not confirmed → `config_mapping_required`.

The same `operator_review_required` outcome is expected for RREL43, RREC13 and
RREC17 when multiple plausible rate/valuation columns remain.

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
