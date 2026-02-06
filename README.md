# Trakt

Data pipeline and analytics platform for Equity Release Mortgage (ERM) portfolio management with ESMA Annex 12 regulatory compliance.

Trakt ingests raw loan tape data, normalises it into a canonical format, validates it against schema and business rules, and produces regime-specific outputs for MI dashboards, ESMA Annex 12 investor reporting, or Annex 2-9 regulatory submissions.

## Quick start

```bash
pip install -r requirements.txt

# MI mode — produce dashboard-ready canonical for Streamlit
python engine/orchestrator/trakt_run.py \
  --mode mi \
  --input loan_portfolio_112025.csv

# Annex 12 — investor reporting (deal-level XML)
python engine/orchestrator/trakt_run.py \
  --mode annex12 \
  --input loan_portfolio_112025.csv \
  --config config/client/config_client_annex12.yaml

# Regulatory — exposure-level Annex 2-9
python engine/orchestrator/trakt_run.py \
  --mode regulatory \
  --input loan_portfolio_112025.csv \
  --regime ESMA_Annex2

# Launch the analytics dashboard
streamlit run analytics/streamlit_app_erm.py
```

## Pipeline modes

`trakt_run.py` supports three modes. Gates 1-3 are common to all modes; Gates 4-5 are mode-specific.

| Mode | Gates | Output |
|------|-------|--------|
| `mi` | 1-3 | `canonical_typed.csv` for Streamlit dashboard |
| `annex12` | 1-5 | ESMA Annex 12 investor XML (deal-level) |
| `regulatory` | 1-5 | ESMA Annex 2-9 regime projection + XML (exposure-level) |

### Gate sequence

| Gate | Script | Purpose |
|------|--------|---------|
| 1 - Semantic alignment | `semantic_alignment.py` | Fuzzy-matches raw loan tape columns to the canonical field registry |
| &mdash; Transform | `canonical_transform.py` | Standardises formats, enriches geography (NUTS/ITL), derives fields (LTV, classifications) |
| 2 - Canonical validation | `validate_canonical.py` | Schema and format validation against the field registry |
| 2.5 - Lineage | `lineage_tracker.py` | Tracks field-level and value-level data lineage |
| 3 - Business rules | `validate_business_rules.py` | Cross-field business rule validation |
| 4a - Annex 12 projection | `annex12_projector.py` | Projects canonical data into the ESMA Annex 12 schema (annex12 mode) |
| 4b - Regime projection | `regime_projector.py` | Projects canonical data into ESMA Annex 2-9 schemas (regulatory mode) |
| 5 - XML + XSD validation | `xml_builder_investor.py` | Generates ESMA-compliant XML and validates against the XSD schema |

A JSON run manifest (`out/run_manifest.json`) is produced at the end of every run with gate statuses, artefact paths, and timing.

## Blob storage trigger

`blob_trigger.py` provides automatic pipeline execution when a data tape is uploaded to cloud blob storage (Azure Blob, AWS S3, or local filesystem for testing).

Upload path convention determines the mode:
```
uploads/{client_id}/mi/tape.csv          → MI mode
uploads/{client_id}/annex12/tape.csv     → Annex 12 mode
uploads/{client_id}/regulatory/tape.csv  → Regulatory mode (regime from filename)
```

Local testing:
```bash
python engine/orchestrator/blob_trigger.py \
  --provider local --path tape.csv --mode mi
```

## Analytics dashboard

`streamlit_app_erm.py` provides an interactive dashboard with three tabs:

- **Stratifications** -- portfolio breakdowns by LTV, region, ticket size, interest rate, borrower age, and origination vintage.
- **Scenario Analysis** -- cashflow projections under configurable HPI, prepayment, mortality, and interest rate assumptions (requires `scenario_engine` module).
- **Static Pools** -- cohort-based performance tracking with prepayment and risk segmentation.

Optional modules (`risk_monitor.py`, `risk_limits_config.py`) add concentration-limit monitoring when present.

## Configuration

| File | Role |
|------|------|
| `config_ERM_UK.yaml` | Master client config -- identity, transformations, enrichment rules, UI branding |
| `config_ere_annex12.yaml` | ESMA Annex 12 deal metadata and structural overlay |
| `fields_registry.yaml` | Canonical field definitions |
| `annex12_field_constraints.yaml` | Field-level validation constraints |
| `annex12_rules.yaml` | Business rule definitions |
| `product_defaults_ERM.yaml` | Default values for equity release mortgage fields |
| `aliases/` | Field alias mappings for data reconciliation |

## Key outputs

| Artefact | Description |
|----------|-------------|
| `*_canonical_full.csv` | Mapped canonical output (pre-typing) |
| `*_canonical_typed.csv` | Typed and enriched canonical output |
| `annex12_projected.csv` | Full Annex 12 record set |
| `annex12_final.xml` | ESMA-compliant investor XML report |
| `out/run_manifest.json` | Pipeline run manifest with gate results |
| `out/field_lineage.json` | Field-level data lineage |

## Project structure

```
trakt/
  engine/
    orchestrator/
      trakt_run.py                   # Pipeline orchestrator (entry point)
      blob_trigger.py                # Cloud blob-storage trigger (Azure/AWS/local)
    gate_1_alignment/
      semantic_alignment.py          # Gate 1: semantic alignment
      aliases/
        alias_builder.py             # TF-IDF alias generation
    gate_2_transform/
      canonical_transform.py         # Transform: typing & derivation
      lineage_tracker.py             # Gate 2.5: data lineage
      delta_manifest.py              # Run manifest / SHA256 hashing
    gate_3_validation/
      validate_canonical.py          # Gate 2: canonical validation
      validate_business_rules.py     # Gate 3: business rule validation
      aggregate_validation_results.py # Validation results aggregation
      validate_only.py               # Standalone validation utility
    gate_4_projection/
      annex12_projector.py           # Gate 4: regime projection
      regime_projector.py            # Alternative regime projector
    gate_5_delivery/
      xml_builder_investor.py        # Gate 5: XML generation
      xml_builder.py                 # Alternative XML builder
  analytics/
    streamlit_app_erm.py             # Analytics dashboard (entry point)
    mi_prep.py                       # Dashboard data preparation layer
    charts_plotly.py                 # Plotly chart factories
    scenario_engine.py               # Cashflow projection engine
    static_pools_core.py             # Static pool analysis engine
    risk_monitor.py                  # Concentration-limit monitoring
  config/
    system/
      fields_registry.yaml           # Canonical field definitions
    client/
      config_client_ERM_UK.yaml      # Master client configuration
      config_client_annex12.yaml     # ESMA Annex 12 configuration
    asset/                           # Product defaults and policies
    regime/                          # Regulatory regime configurations
  requirements.txt                   # Python dependencies
```
