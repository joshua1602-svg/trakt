# Transformation Agent v1

The Transformation Agent is the **deterministic bridge between the Onboarding
Agent and the Validation Agent**. It is *not* a projection or regulatory
delivery agent.

```text
Raw client files
  → Onboarding Agent
  → governed canonical handoff package      (output/handoff/24..27, output/central/18)
  → Transformation Agent          ← THIS AGENT
  → Validation Agent
  → Projection Agent
  → Delivery/XML/XSD/reporting outputs
```

## What it does

1. Loads and **validates** the onboarding handoff manifest
   (`24_onboarding_handoff_manifest.json`). It refuses to run unless:
   - `handoff_type == canonical_onboarding_package`
   - `not_raw_source == true`
   - `ready_for_transformation_validation == true`
   - `next_agent == transformation_validation`
2. Loads the **central canonical tape** (`output/central/18_central_lender_tape.csv`)
   — it never re-runs raw Gate 1 canonicalisation or source discovery.
3. Uses the **handoff field contract** (`26_*`) as the control layer for what to
   transform, default, type, normalise or defer.
4. Reuses the existing deterministic `engine.gate_2_transform` logic through a
   clean adapter (`engine/transformation_agent/gate2_adapter.py`):
   - `apply_types` — date / number / rate / currency / boolean normalisation;
   - `resolve_canonical_enum_normalization` + `apply_canonical_enum_normalization`
     — internal enum standardisation (NOT regime / ESMA projection);
   - `apply_config_defaults` — config-driven default fill.
5. Materialises asset-class defaults, ND defaults and configured-static values
   (precedence: handoff contract → asset config → regime config). Blanks only;
   existing source values are never overwritten.
6. Classifies every contract field with a controlled `transformation_status`
   and surfaces unresolved items as transformation issues with clear ownership.

## What it does NOT do (guardrails)

- No raw Gate 1 re-run on the central tape.
- No fuzzy source matching / source discovery.
- No projection to Annex 2 XML and **never** claims XML readiness.
- No mutation of any Onboarding Agent artefact (writes only under
  `output/transformation/`).
- Projection-specific gaps are classified `pending_projection_rule` and handed
  to the Projection Agent — they are never treated as transformation failures.

## CLI

```bash
python -m engine.transformation_agent.workflow \
  --handoff-manifest onboarding_output/<client_id>/<run_id>/output/handoff/24_onboarding_handoff_manifest.json
```

or the subcommand form:

```bash
python -m engine.transformation_agent.cli transform \
  --handoff-manifest .../output/handoff/24_onboarding_handoff_manifest.json
```

Configs default to the values recorded in the handoff manifest and can be
overridden with `--asset-config`, `--regime-config`, `--registry`,
`--enum-mapping`, `--no-dayfirst`.

## Output artefacts (numbering convention)

Written under `onboarding_output/<client_id>/<run_id>/output/transformation/`.
The Onboarding Agent owns artefacts up to the 40s; the Transformation Agent
takes the dedicated **30–35** block (no conflict with onboarding numbering):

| Artefact | Purpose |
| --- | --- |
| `30_transformation_manifest.json` / `.yaml` | Run manifest, governance flags, counts, readiness. |
| `31_transformed_canonical_tape.csv` / `.json` | Normalised, defaulted, validation-ready canonical tape. |
| `32_transformation_field_contract.csv` / `.json` | Per-field transformation status + value source + casts. |
| `33_transformation_readiness.json` / `.md` | Distinct validation / projection / XML readiness flags. |
| `34_transformation_lineage.json` | Onboarding lineage carried forward + transformation lineage. |
| `35_transformation_issues.csv` / `.json` | Controlled issue ledger with ownership and blocking flags. |

## Readiness semantics (distinct flags)

- `ready_for_validation = true` when the central tape is loaded, the handoff is
  valid, there are no blocking onboarding decisions, and there are **no
  uncontrolled type/enum/date parse failures**. Every field carries a controlled
  status (including explicit `source_absent` / `semantic_derivation_required` /
  `operator_decision_pending`).
- `ready_for_projection = false` while any `pending_projection_rule`,
  `semantic_derivation_required` or `operator_decision_pending` remain.
- `ready_for_xml_delivery = false` always at this stage.

## Transformation status vocabulary

`transformed`, `copied`, `default_materialised`, `nd_default_materialised`,
`configured_static_materialised`, `source_context_materialised`,
`run_context_materialised`, `enum_normalized`, `type_normalized`,
`derived`, `pending_projection_rule`, `source_absent`,
`semantic_derivation_required`, `operator_decision_pending`,
`validation_required`, `not_applicable`, `failed_transformation`.

## Run / source context fields

Portfolio-level fields such as `data_cut_off_date` (RREL6) are resolved by the
Onboarding Agent and carried in the handoff as `source_context_mapped` /
`run_context_mapped`. The Transformation Agent materialises the single value into
**every row** (ISO `YYYY-MM-DD`), with status `source_context_materialised` /
`run_context_materialised`. See [`run_context_fields.md`](run_context_fields.md).

## Tests

```bash
pytest tests/test_transformation_agent_workflow.py -q
```


## Consuming the Onboarding target-field disposition

Onboarding now owns the per-field **disposition** (see
`docs/target_contract_completion_checklist.md`). The Transformation Agent
**executes** it rather than rediscovering each field's treatment. The disposition
is carried on the handoff field contract (`26_*`) and propagated into
`32_transformation_field_contract.csv` via the `field_disposition`,
`disposition_source` and `disposition_action` columns
(`transformation_action_for_disposition`):

- `asset_default_supplied` / `configured_static_supplied` → materialise the configured value;
- `nd_policy_selected` → materialise the selected (allowed) ND;
- `derivation_configured` → apply the derivation rule;
- `client_onboarding_required` → do **not** source-map/default; carry forward as client input required;
- `projection_rule_required` → carry forward for projection.
