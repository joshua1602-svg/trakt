# Config Duplication & Misalignment Inventory (Pass 1)

## Scope
Concrete inventory of current drift/duplication candidates across runtime, dashboard, risk, regime, and client layers.

This is a **Pass 1 planning artifact** and does not change runtime behavior.

## Severity legend
- **Blocker**: must address before clean multi-client onboarding safety.
- **Near-term**: should address in next cleanup pass.
- **Later**: acceptable temporarily, but should be standardized.

---

## Inventory table

| # | Concern | Current location(s) | Intended owner layer | Risk level | Suggested cleanup pass | Priority bucket |
|---|---|---|---|---|---|---|
| 1 | ERM default portfolio type in core runtime | `engine/orchestrator/trakt_run.py` (`--portfolio-type` default `equity_release`) | Platform runtime defaults / resolved config selection | High | Pass 2 | Blocker |
| 2 | ERM client config path default in core runtime | `engine/orchestrator/trakt_run.py` (`--master-config` default ERM UK file) | Runtime context selection + platform default profile | High | Pass 2 | Blocker |
| 3 | Hardcoded ERM client config lookup in dashboard | `analytics/streamlit_app_erm.py` (`config_client_ERM_UK.yaml` paths) | Client config resolution through resolver | High | Pass 2 | Blocker |
| 4 | Hardcoded static pool config filename | `analytics/streamlit_app_erm.py` (`static_pools_config_erm.yaml`) | Asset chart pack selected by resolver | Medium | Pass 2 | Near-term |
| 5 | In-code chart fallback definitions | `analytics/streamlit_app_erm.py` fallback `charts = [...]` | Standard chart pack + asset/client overlays | Medium | Pass 2 | Near-term |
| 6 | Theme defaults duplicated in code + YAML | `streamlit_app_erm.py`, `charts_plotly.py`, `config/system/config.py`, client YAML | Client config + optional platform theme defaults | Medium | Pass 2 | Near-term |
| 7 | Client legal/reporting constants duplicated | `config/client/config_client_ERM_UK.yaml`, `config/client/config_client_annex12.yaml` | Client config (single source) + regime injection | High | Pass 2 | Blocker |
| 8 | Conflicting static/reporting dates | `config_client_ERM_UK.yaml` vs `config_client_annex12.yaml` | Client deterministic constants + explicit override rules | High | Pass 2 | Blocker |
| 9 | Client/regime overlay mixing in Annex 12 file | `config/client/config_client_annex12.yaml` | Clear split: regime rules in regime layer, client constants in client layer | Medium | Pass 2 | Near-term |
| 10 | Risk thresholds in Python constants | `config/client/risk_limits_config.py` | Client config (declarative) with optional asset defaults | Medium | Pass 3 | Later |
| 11 | UK region map hardcoded in risk module | `analytics/risk_monitor.py` | Asset or geography mapping config | Medium | Pass 3 | Later |
| 12 | Legacy shared config module duplicates declarative config roles | `config/system/config.py` | Platform/client/chart layers declaratively | Low/Med | Pass 3 | Later |
| 13 | Chart ownership not explicit by layer (standard/asset/client) | dashboard wiring + asset file | Standard chart pack + asset chart pack + client overrides | High | Pass 2 | Blocker |
| 14 | Resolved runtime config artifact missing | no canonical `resolved_config` artifact | Runtime/resolved config layer | Medium | Pass 2 | Near-term |
| 15 | Provenance for overridden values not captured | no per-key source tracking artifact | Runtime/resolved config layer | Medium | Pass 2 | Near-term |
| 16 | Regime output tooling split across builders with uneven structure | `xml_builder.py` vs `xml_builder_investor.py` | Regime/output layer standardization | Low/Med | Pass 3 | Later |

---

## Highest urgency items for onboarding safety

1. Remove hidden ERM defaults from shared runtime selection path.
2. Stop hardcoded dashboard config path assumptions.
3. Normalize chart ownership layering (standard/asset/client).
4. Consolidate duplicated client constants (LEI/legal/reporting date ownership).
5. Introduce resolved runtime config/provenance artifact.

---

## Notes for implementation sequencing

- **Pass 1**: document and scaffold only (current pass).
- **Pass 2**: introduce resolver usage in controlled entrypoints and chart loading.
- **Pass 3**: convert remaining Python-embedded policy constants (risk/region) and legacy modules.

