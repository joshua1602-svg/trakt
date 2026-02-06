# Trakt

Data pipeline and analytics platform for Equity Release Mortgage (ERM) portfolio management with ESMA Annex 12 regulatory compliance.

Trakt ingests raw loan tape data, normalises it into a canonical format, validates it against schema and business rules, projects it into the ESMA Annex 12 schema, and produces investor-ready XML reports.

## Quick start

```bash
pip install -r requirements.txt

# Run the full pipeline
python trakt_orchestrator.py \
  --input loan_portfolio_112025.csv \
  --config client_config_annex_12.yaml

# Launch the analytics dashboard
streamlit run streamlit_app_erm.py
```

## Pipeline stages

`trakt_orchestrator.py` orchestrates the following gates in sequence:

| Gate | Script | Purpose |
|------|--------|---------|
| 1 - Semantic alignment | `alignment_engine.py` | Fuzzy-matches raw loan tape columns to the canonical field registry |
| &mdash; Transform | `portfolio_synthesizer.py` | Standardises formats, enriches geography (NUTS/ITL), derives fields (LTV, classifications) |
| 2 - Canonical validation | `gatekeeper.py` | Schema and format validation against the field registry |
| 2.5 - Lineage | `lineage_JSON.py` | Tracks field-level and value-level data lineage |
| 3 - Business rules | `validate_business_rules_aligned_v1_2.py` | Cross-field business rule validation |
| 4 - Regime projection | `esma_investor_regime_adapter.py` | Projects canonical data into the full ESMA Annex 12 schema |
| 5 - XML + XSD validation | `esma_investor_disclosure_generator.py` | Generates ESMA-compliant XML and validates against the XSD schema |

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
| `asset_policy_uk.yaml` | Master client config -- identity, transformations, enrichment rules, UI branding |
| `client_config_annex_12.yaml` | ESMA Annex 12 deal metadata and structural overlay |
| `data_standard_definition.yaml` | Canonical field definitions |
| `esma_12_integrity_rules.yaml` | Field-level validation constraints |
| `esma_12_disclosure_logic.yaml` | Business rule definitions |
| `submission_schema_layout.yaml` | ESMA code ordering for XML output |
| `materiality_framework.yaml` | Issue severity and materiality policy |
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
  trakt_orchestrator.py                       # Pipeline orchestrator (entry point)
  streamlit_app_erm.py                        # Analytics dashboard (entry point)
  alignment_engine.py                         # Gate 1: semantic alignment
  portfolio_synthesizer.py                    # Transform: typing & derivation
  gatekeeper.py                               # Gate 2: canonical validation
  validate_business_rules_aligned_v1_2.py     # Gate 3: business rule validation
  lineage_JSON.py                             # Gate 2.5: data lineage
  esma_investor_regime_adapter.py             # Gate 4: regime projection
  esma_investor_disclosure_generator.py       # Gate 5: XML generation
  mi_prep.py                                  # Dashboard data preparation layer
  charts_plotly.py                            # Plotly chart factories
  static_pools_core.py                        # Static pool analysis engine
  risk_monitor.py                             # Concentration-limit monitoring
  alias_builder.py                            # TF-IDF alias generation
  delta_json.py                               # Run manifest / SHA256 hashing
  asset_policy_uk.yaml                        # Master client configuration
  client_config_annex_12.yaml                 # ESMA Annex 12 configuration
  data_standard_definition.yaml               # Canonical field definitions
  esma_12_integrity_rules.yaml                # Field-level validation constraints
  esma_12_disclosure_logic.yaml               # Business rule definitions
  submission_schema_layout.yaml               # ESMA code ordering
  materiality_framework.yaml                  # Issue materiality policy
  aliases/                                    # Field alias YAML files
  enum/                                       # Enumeration mapping files
  requirements.txt                            # Python dependencies
```
