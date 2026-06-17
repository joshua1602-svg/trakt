# Projection Blocker Diagnostics

After the Validation Agent completes its main pass, a diagnostic step refines
every **projection-blocking issue** from the coarse `projection_required` /
`blocking_for_projection` flag into one of **six precise subtypes**. This gives
operators and the Projection Agent a clear, actionable explanation of _why_ each
field is blocked and _what_ needs to happen to unblock it.

```text
Transformation Agent
  → Validation Agent (artefacts 40–45)
  → Projection Blocker Diagnostics  ← THIS STEP
  → 46_projection_blocker_diagnostics.csv / .json / .md
```

## When it runs

Automatically as the last step of `build_validation_package` in
`engine/validation_agent/validation_agent.py`, after all issues are collected.
It reads the validated issue list and the transformed canonical tape; it writes
only under `output/validation/` and never mutates any upstream artefact.

## Subtypes

| Subtype | Trigger | Recommended action |
| --- | --- | --- |
| `materialised_projection_pending` | Field IS in the tape with at least one non-blank value | Implement the regime projection rule to map/validate for Annex 2 output |
| `not_materialised_projection_pending` | Field absent or entirely blank; no ND/default allowed; no related source fields present | Supply a source value, or configure a ND/default if the regime allows it |
| `nd_or_default_rule_pending` | Field absent/blank but the regime or asset config allows a ND-value or default | Implement the ND-value or default rule at the projection layer |
| `source_mapping_pending` | Field absent/blank but related canonical fields (sharing conceptual name tokens) are present and non-blank | Implement a derivation rule using the related source fields |
| `operator_or_config_dependency` | Issue IS `operator_required` / `config_required`, or a peer issue for the same field is | Resolve the linked operator/config issue first, then re-validate |
| `unknown_projection_dependency` | Fallback — none of the above patterns matched | Investigate field origin and projection requirements manually |

## Classification logic

For each issue where `blocking_for_projection = true`:

1. **Operator/config direct**: if the issue's `validation_classification` is
   `operator_required` or `config_required` → `operator_or_config_dependency`.
2. **Peer dependency**: if another issue for the same `canonical_field` is
   `operator_required` / `config_required` → `operator_or_config_dependency`.
3. **Materialised**: if the field is in the transformed tape and has at least one
   non-blank row value → `materialised_projection_pending`.
4. **ND or default allowed**: field is absent/blank but `nd_allowed` or
   `default_allowed` in `annex2_delivery_rules.yaml` → `nd_or_default_rule_pending`.
5. **Related source fields**: field is absent/blank but other tape columns sharing
   name tokens (e.g. `income` → `secondary_income`) are present and non-blank →
   `source_mapping_pending`.
6. **Nothing found**: field absent/blank, no ND/default, no related fields →
   `not_materialised_projection_pending`.
7. **Fallback**: edge cases → `unknown_projection_dependency`.

## Outputs

Written under `onboarding_output/<client_id>/<run_id>/output/validation/`:

| Artefact | Content |
| --- | --- |
| `46_projection_blocker_diagnostics.csv` | One row per projection-blocking issue with subtype and rationale |
| `46_projection_blocker_diagnostics.json` | Same data + subtype counts + subtype descriptions |
| `46_projection_blocker_diagnostics.md` | Human-readable summary grouped by subtype |

The diagnostic counts are also embedded in:
- `40_validation_manifest.json` — `projection_blocker_diagnostic_count` +
  `projection_blocker_subtype_counts`
- `42_validation_readiness.md` — a "Projection blocker diagnostics" section with
  a subtype count table

## Guardrails

- **No upstream mutation**: writes only under `output/validation/`.
- **No readiness auto-resolution**: the existing `ready_for_projection` /
  `ready_for_validation_complete` booleans are not changed by diagnostics.
- **No projection or XML output**: the diagnostic step has no knowledge of Annex 2
  XML schema or projection rules.
- **Conservative blocking**: every issue that was `blocking_for_projection = true`
  before diagnostics remains blocking after.

## Module

`engine/validation_agent/projection_blocker_diagnostics.py`

Public API:
- `classify_projection_blockers(issues, df, tx_contract, regime_index)` → list of diagnostic rows
- `write_blocker_diagnostics(out_dir, diagnostic_rows, *, client_id, run_id, target_contract_id)` → path dict
- `subtype_counts(diagnostic_rows)` → `{subtype: count}` for all 6 known subtypes
- `SUBTYPES` — ordered tuple of the six subtype string constants

## Tests

```bash
pytest tests/test_projection_blocker_diagnostics.py -v
```

Covers 34 assertions across unit (classify subtypes, write artefacts) and
integration (full Transformation → Validation pipeline producing artefact 46).
