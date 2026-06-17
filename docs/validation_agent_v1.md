# Validation Agent v1

The Validation Agent is the **control gate after Transformation**. It consumes
the Transformation Agent output package, validates the transformed canonical
values and the transformation issue classifications, and produces a governed
**validation-readiness package** for the Projection Agent.

```text
Raw client files
  → Onboarding Agent
  → governed canonical handoff package
  → Transformation Agent
  → Validation Agent            ← THIS AGENT
  → Projection Agent
  → Delivery/XML/XSD/reporting outputs
```

It answers:

- Is the transformed canonical dataset valid enough to pass to Projection?
- Which issues are true validation failures vs warnings?
- Which issues require operator action? config/policy action?
- Which issues are acceptable downstream projection gaps?

## Purpose

A deterministic, configurable validation layer over the **canonical** transformed
tape — *not* an Annex 2 / XML delivery validator. It validates value quality and
canonical cross-field business rules, then classifies every carried-forward
transformation issue with a clear downstream owner and action.

## Inputs

Loaded from the run's `output/transformation/`:

| Input | Use |
| --- | --- |
| `30_transformation_manifest.json` | Validated for readiness + governance flags. |
| `31_transformed_canonical_tape.csv` | The dataset under validation. |
| `32_transformation_field_contract.csv` | Field-level control layer. |
| `34_transformation_lineage.json` | Carried forward + extended. |
| `35_transformation_issues.csv` | Carried forward, re-validated, reclassified. |

It also reads (referenced via the manifest, read-only):
`config/system/fields_registry.yaml`, `config/regime/annex2_delivery_rules.yaml`,
`config/asset/product_defaults_ERM.yaml`, `config/system/enum_synonyms*.yaml`.

## Outputs

Written under `onboarding_output/<client_id>/<run_id>/output/validation/`.
The **40–45** block is validation-specific (Onboarding owns the project-root 40s
under the *run* dir; Transformation owns 30–35 under `output/transformation/`;
these 40–45 live under `output/validation/`, so there is no path collision):

| Artefact | Purpose |
| --- | --- |
| `40_validation_manifest.json` / `.yaml` | Run manifest, governance flags, counts, readiness, `next_agent`. |
| `41_validation_results.csv` / `.json` | One row per field/check: status, counts, sample failures. |
| `42_validation_readiness.json` / `.md` | Operator-readable readiness verdict + next action. |
| `43_validation_issues.csv` / `.json` | Carried-forward + new issues with classification, ownership, blocking flags. |
| `44_validation_lineage.json` | Transformation lineage carried forward + validation lineage. |
| `45_validation_summary.md` | Human-readable pass/warn/fail summary. |

## Readiness flags (three distinct)

- **`ready_for_validation_complete`** — transformed tape loaded, all
  validation-owned checks ran, **no validation_failure marked
  blocking_for_validation**, all remaining issues classified with a downstream
  owner/action, and no uncontrolled parser/type/enum exceptions.
- **`ready_for_projection`** — only if validation is complete *and* there are no
  `operator_required`, `config_required`, `semantic_derivation_required`, or
  `pending_projection_rule` items outstanding (unless explicitly deferred by
  projection policy).
- **`ready_for_xml_delivery`** — always `false` at this stage.

`next_agent` = `projection` when `ready_for_projection` is true; otherwise
`operator_config_projection_remediation`.

## Issue classifications

Every issue carries one of:

`validation_pass`, `validation_warning`, `validation_failure`,
`operator_required`, `config_required`, `projection_required`,
`acceptable_downstream_gap`, `semantic_derivation_required`.

Carry-forward rules for transformation issues:

| Transformation issue | Validation classification | blocks validation | blocks projection |
| --- | --- | --- | --- |
| `pending_projection_rule` | `projection_required` | no | yes |
| `operator_decision_pending` | `operator_required` | no | yes |
| `semantic_derivation_required` | `semantic_derivation_required` | no | yes |
| `source_absent` | `config_required` (if a default is allowed) else `validation_warning` | no | yes |
| `enum_unmapped` | `config_required` (or `validation_failure` if mandatory+enforce_presence) | only if mandatory | yes |
| `invalid_default` / `invalid_nd_default` | `validation_failure` | yes | yes |
| `*_parse_failed` | `validation_failure` | yes | yes |

Value-level checks add new issue types: `missing_required_value`, `invalid_type`,
`invalid_date`, `invalid_number`, `invalid_rate`, `invalid_boolean`,
`invalid_enum`, `invalid_country_code`, `invalid_lei`, `duplicate_identifier`,
`cross_field_rule_failed`.

A blank in a **mandatory** field is a blocking `validation_failure` only when the
field permits **no** ND/default (e.g. an identifier). Where the regime allows an
ND value or default, a blank is a downstream config gap (blocks projection, not
validation). A mandatory field with no ND/default permitted that is **entirely
absent** from the tape is also a blocking failure.

Run/source context fields (e.g. `data_cut_off_date` / RREL6) are extracted by
Onboarding and materialised by Transformation into every row; Validation only
checks presence + ISO format and never fills the value. See
[`run_context_fields.md`](run_context_fields.md).

## Checks performed

- **Value-level** (per field, registry-driven): presence/nullability, data type,
  ISO date format, numeric format, percentage/rate bounds, boolean values, enum
  membership (canonical enum library), country code, LEI format, identifier
  uniqueness / duplicate exposure identifiers, regime regex validators (as
  warnings — authoritative at projection).
- **Cross-field business rules** (canonical, configurable): valuation amounts ≥ 0,
  current LTV ≥ 0, `data_cut_off_date` present + parseable, loan/unique
  identifier not null, `redemption_date >= origination_date` where both present.

These reuse `engine.gate_3_validation` primitives (`is_blank`, `is_nd`,
`coerce_decimal`, `coerce_date_iso`, `coerce_bool_yn`, `load_registry`,
`load_enum_library`) through `engine/validation_agent/rules_adapter.py`.

## CLI usage

```bash
python -m engine.validation_agent.workflow \
  --transformation-manifest onboarding_output/<client_id>/<run_id>/output/transformation/30_transformation_manifest.json
```

or the subcommand form:

```bash
python -m engine.validation_agent.cli validate \
  --transformation-manifest .../output/transformation/30_transformation_manifest.json
```

Configs default to the values recorded in the transformation manifest and can be
overridden with `--registry`, `--regime-config`, `--asset-config`,
`--enum-config-dir`.

## Relation to Onboarding and Transformation

- **Onboarding** produces the governed canonical handoff package.
- **Transformation** materialises defaults, normalises types/enums, and emits the
  transformed canonical tape + issue ledger.
- **Validation** (this agent) validates that tape and the issue classifications,
  and decides whether the dataset is fit to pass to Projection. It refuses to run
  unless the transformation manifest reports `ready_for_validation = true` and
  `ready_for_xml_delivery = false`.

## What this agent deliberately does NOT do

- No raw Gate 1 re-run, source discovery, or fuzzy matching.
- No mutation of Onboarding or Transformation artefacts (writes only under
  `output/validation/`).
- No projection output, no Annex 2 XML, and never claims XML readiness.
- No silent resolution of operator decisions and no silent addition of enum
  mappings or defaults — unresolved work is surfaced with explicit ownership.

## Tests

```bash
pytest tests/test_validation_agent_workflow.py -q
```
