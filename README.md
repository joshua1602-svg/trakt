# Trakt

Data pipeline and analytics platform for Equity Release Mortgage (ERM) portfolio management with ESMA Annex 12 regulatory compliance.

Trakt ingests raw loan tape data, normalises it into a canonical format, validates it against schema and business rules, projects it into the ESMA Annex 12 schema, and produces investor-ready XML reports.

## Quick start

```bash
pip install -r requirements.txt

# Run the full pipeline
python trakt_run.py \
  --input loan_portfolio_112025.csv \
  --config config_ere_annex12.yaml

# Launch the analytics dashboard
streamlit run streamlit_app_erm.py
```

## Pipeline stages

`trakt_run.py` orchestrates the following gates in sequence:

| Gate | Script | Purpose |
|------|--------|---------|
| 1 - Semantic alignment | `semantic_alignment.py` | Fuzzy-matches raw loan tape columns to the canonical field registry |
| &mdash; Transform | `canonical_transform.py` | Standardises formats, enriches geography (NUTS/ITL), derives fields (LTV, classifications) |
| 2 - Canonical validation | `validate_canonical.py` | Schema and format validation against the field registry |
| 2.5 - Lineage | `lineage_tracker.py` | Tracks field-level and value-level data lineage |
| 3 - Business rules | `validate_business_rules.py` | Cross-field business rule validation |
| 4 - Regime projection | `annex12_projector.py` | Projects canonical data into the full ESMA Annex 12 schema |
| 5 - XML + XSD validation | `xml_builder_investor.py` | Generates ESMA-compliant XML and validates against the XSD schema |

A JSON run manifest (`out/run_manifest.json`) is produced at the end of every run with gate statuses, artefact paths, and timing.

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
