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

## ND/default eligibility sources (important)

ND/default eligibility is **not** read from `annex2_delivery_rules.yaml →
field_rules` alone. `field_rules` is the *active runtime rule set*, but many
authoritative Annex 2 codes have no hand-authored rule there yet (e.g. RREL15
`customer_type`, RREL24 `maturity_date`). The diagnostic resolves eligibility, in
priority order, from **four** sources and records which one confirmed it in the
`nd_or_default_source` column:

1. **`field_rules`** — the runtime regime index (`nd_allowed` / `default_allowed`);
2. **`field_universe`** — the authoritative workbook ND envelope in
   `annex2_field_universe.yaml` (`nd1_4_allowed` / `nd5_allowed`), merged into the
   regime index as fallback metadata by `rules_adapter.build_regime_index`;
3. **`asset_config`** — a concrete default/ND default in `product_defaults_*.yaml`
   (`defaults` / `nd_defaults`), e.g. ERM `maturity_date: ND5`;
4. **`transformation_contract`** — `default_rule` / `default_value` / `nd_allowed`
   columns on `32_transformation_field_contract`.

This is why RREL15 (ND1–ND4 in the universe) now resolves as
`nd_or_default_rule_pending` instead of `not_materialised_projection_pending`, even
though it has no `field_rules` entry. The diagnostic never invents a default — it
only reports that one is *permitted*; `asset_default_possible` is true only when a
concrete asset value is configured.

## Classification logic

For each issue where `blocking_for_projection = true`:

1. **Operator/config direct**: if the issue's own `validation_classification` is
   `operator_required` or `config_required` → `operator_or_config_dependency`.
2. **Materialised**: if the field is in the transformed tape and has at least one
   non-blank row value → `materialised_projection_pending`.
3. **ND or default allowed**: field is absent/blank but ND/default is permitted by
   any of the four sources above → `nd_or_default_rule_pending`. **A peer
   operator/config dependency no longer masks this** — the dependency is surfaced
   via the `operator_dependency_present` / `config_dependency_present` evidence
   columns instead of overriding the more actionable subtype.
4. **Peer dependency (no ND/default)**: field absent/blank, not ND/default
   eligible, but a peer issue for the same field is operator/config-owned →
   `operator_or_config_dependency`.
5. **Related source fields**: field is absent/blank but other tape columns sharing
   name tokens (e.g. `income` → `secondary_income`) are present and non-blank →
   `source_mapping_pending`.
6. **Nothing found**: field absent/blank, no ND/default, no related fields →
   `not_materialised_projection_pending`.
7. **Fallback**: edge cases → `unknown_projection_dependency`.

### Evidence columns

Every diagnostic row also carries auditable evidence, independent of the chosen
subtype: `nd_or_default_allowed`, `nd_or_default_source`, `nd_default_possible`,
`asset_default_possible`, `operator_dependency_present`,
`config_dependency_present`.

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
- `classify_projection_blockers(issues, df, tx_contract, regime_index, *, field_universe=None, asset_defaults=None, asset_nd_defaults=None)` → list of diagnostic rows
- `write_blocker_diagnostics(out_dir, diagnostic_rows, *, client_id, run_id, target_contract_id)` → path dict
- `subtype_counts(diagnostic_rows)` → `{subtype: count}` for all 6 known subtypes
- `SUBTYPES` — ordered tuple of the six subtype string constants

### Runtime ND visibility — design choice

`rules_adapter.build_regime_index(regime_cfg, *, field_universe, code_to_canonical)`
**merges** the workbook universe into the regime index rather than requiring every
code to be hand-authored in `field_rules`. `field_rules` always wins where present;
the universe is fallback metadata (empty `nd_allowed` is backfilled, and
universe-only codes get a synthesised entry tagged `rule_source: field_universe`).
This was chosen over adding `field_rules` stubs because adding entries would also
change Onboarding's `pending_regime_rule` classification (a broader, riskier change),
whereas the merge keeps `field_rules` as the single active rule set while making the
authoritative ND envelope visible at runtime.

## Tests

```bash
pytest tests/test_projection_blocker_diagnostics.py -v
```

Covers 34 assertions across unit (classify subtypes, write artefacts) and
integration (full Transformation → Validation pipeline producing artefact 46).
