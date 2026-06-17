# Run / source context fields (e.g. `data_cut_off_date` / RREL6)

Some regulatory fields are **portfolio-level run/source context** — a single
value for the whole tape, not a per-loan attribute. The reporting cut-off date
`data_cut_off_date` (ESMA Annex 2 `RREL6` / `CutOffDt`) is the canonical example.

These must be **extracted from the client source pack / config by the Onboarding
Agent**, carried through the handoff, **materialised by the Transformation Agent**
into every row, and then **validated** — never hard-coded, never filled by
Validation, never silently defaulted.

## End-to-end flow

```text
Onboarding Agent        extract a single portfolio-level value with evidence
  → handoff package     carry value + source evidence + classification
  → Transformation      materialise the value into every row (ISO YYYY-MM-DD)
  → Validation          validate presence + ISO format (does NOT fill it)
```

## 1. Onboarding extraction (`engine/onboarding_agent/run_context.py`)

`extract_data_cut_off_date(...)` resolves a single value deterministically and
auditable, with this priority:

1. `--override-reporting-date` operator override (recorded as `cli_override`);
2. **source column** — the canonical `data_cut_off_date` column values and the
   file profiler's `detected_reporting_date` (`01_file_inventory.csv`);
3. **file name** — a date embedded in the source file name
   (`..._012026.csv` → `2026-01-31`, `..._2026-01-31.csv`, `January 2026`, …);
4. **config** — a configured static reporting date in asset/regime config;
5. `--reporting-date` plain CLI fallback (only when nothing above is found).

Resolution rules:

- **Agree** → accept, with source evidence.
- **Conflict** within the chosen tier → surfaced as a **blocking operator item**
  (`data_cut_off_date_conflict = true`); never silently resolved.
- **Missing** → surfaced as `data_cut_off_date_missing = true`; no date is
  invented, so downstream Validation can still fail.

All values are normalised to ISO `YYYY-MM-DD`.

## 2. Handoff package

The Onboarding handoff records the resolved value + evidence:

- `24_onboarding_handoff_manifest.json` adds:
  `run_context_fields`, `source_context_fields`, `data_cut_off_date`,
  `data_cut_off_date_source`, `data_cut_off_date_source_file`,
  `data_cut_off_date_source_column_or_location`, `data_cut_off_date_confidence`,
  `data_cut_off_date_conflict`, `data_cut_off_date_missing`,
  `data_cut_off_date_candidates` (full evidence list).
- `26_onboarding_handoff_field_contract` classifies the `data_cut_off_date` row
  as:
  - `source_context_mapped` — value came from a source column;
  - `run_context_mapped` — value came from file name / config / CLI;
  - `operator_decision_pending` (blocking) — on conflict.
- `27_onboarding_handoff_lineage.json` records a lineage row with source/value
  evidence.

## 3. Transformation materialisation

The Transformation Agent recognises `source_context_mapped` /
`run_context_mapped` fields and materialises the single value into **every row**
of `31_transformed_canonical_tape.csv`, normalised to ISO `YYYY-MM-DD`.

- status in `32_transformation_field_contract`:
  `source_context_materialised` / `run_context_materialised`;
- origin recorded in `34_transformation_lineage.json`
  (`default_source = handoff_source_context` / `handoff_run_context`);
- if the value cannot be parsed to ISO, it raises a controlled transformation
  issue (so Validation fails) — never a guessed date.

## 4. Validation

Validation is unchanged in principle: `data_cut_off_date` must be **present** and
**ISO-parseable**. It does not fill the value. `RREL6` permits no ND/default, so:

- after source-derived materialisation → checks **pass**;
- when absent/unresolved (column blank or missing) → a **blocking
  `validation_failure`** (`missing_required_value`).

## CLI (optional fallback only)

```bash
# Source-derived value is preferred; this is only used when none is found:
python -m engine.onboarding_agent.workflow ... --reporting-date 2026-01-31

# Force the CLI value to win over a source-derived value (audited as cli_override):
python -m engine.onboarding_agent.workflow ... \
    --reporting-date 2026-01-31 --override-reporting-date
```

## Guarantees

- No hard-coded date anywhere.
- Validation never fills the value.
- Conflicts and missing values are surfaced explicitly, never silently resolved.
- No Projection / XML logic is involved at any stage.
