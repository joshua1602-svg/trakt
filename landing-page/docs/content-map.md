# Content map — every claim on the page, and what backs it

The purpose of this document is to stop marketing running ahead of the product.
Every visible claim is classified:

| Class | Meaning |
|---|---|
| **Evidenced** | Implemented in this repository. The cited file is the proof. |
| **Demonstration** | Describes what the public demo does, here and now. |
| **Deployment** | True of a configured client deployment, not a platform guarantee. Worded to say so. |
| **Positioning** | A statement of what Trakt is for. Not a capability claim. |
| **Excluded** | Considered and deliberately **not** said, because the repository does not support it. |

Anyone editing page copy should add a row here, or delete one.

---

## A. Navigation

| Copy | Class | Evidence |
|---|---|---|
| Product · Capabilities · How it works · Governance · Book a demo | Positioning | — |

---

## B. Hero

| Copy | Class | Evidence |
|---|---|---|
| "Portfolio intelligence. Wherever you work." | Positioning | Brief-specified proposition. |
| "Trakt turns fragmented portfolio data into trusted answers, automated reporting and governed workflows." | Evidenced | Ingest→canonical→validate→output pipeline: `engine/orchestrator/trakt_run.py`, gates in `README.md`. "Trusted answers" = the deterministic MI engine (`mi_agent/`). "Automated reporting" = `mi_agent_pptx/`, `configs/pptx/investor_pack.yaml`. |
| "Use Trakt through Microsoft 365 Copilot, Teams, the Trakt workspace or automated reporting processes." | Evidenced (with one qualification) | Copilot: `deploy/copilot-agent/` + `mi_agent_api/copilot_actions.py` (three shipped actions). Workspace: `frontend/mi-agent-ui/`. Automated reporting: `mi_agent_pptx/cli.py`, `apps/blob_trigger_app`. **Teams**: the declarative agent is packaged as a Teams app (`deploy/copilot-agent/manifest.json`) and sideloaded through Teams — that is the extent of the Teams claim. No standalone Teams bot exists. |
| Interface preview showing £5,382,463 / 36 exposures / regional bars | Evidenced | Rendered from `data/demo-pack.json`, produced by the engine from `synthetic_demo/output/…canonical_typed.csv`. Not a mock-up. |
| "The public demonstration uses a wholly synthetic portfolio. No client or consumer information is displayed, and no upload is accepted." | Demonstration | True by construction — no upload endpoint exists; the pack is the only data source. |

---

## C. Overview video

| Copy | Class | Evidence |
|---|---|---|
| "Two minutes on what Trakt does" / "A short walkthrough of the governed data layer…" | Positioning | — |
| The placeholder card | Demonstration | **No video asset exists in this repository.** Component documents the drop-in location; README § "Replacing the demo video". |

---

## D. Interactive demonstration

| Copy | Class | Evidence |
|---|---|---|
| "This is the Trakt Copilot experience… Answers are computed by the same deterministic engine that serves the Trakt workspace and Microsoft 365 Copilot." | Evidenced | Both channels are thin adapters over one service (`docs/mi_shared_service_architecture.md`); the pack is generated through `mi_agent.mi_agent_workflow.run_mi_agent_query` + `mi_agent_api.adapters.adapt_workflow_result` — the same chain. |
| Client: **Synthetic Demo Lender**, ERE Funding Limited, UK equity release, GBP | Evidenced | `synthetic_demo/config/config_client_SYNTHETIC_ERM.yaml`. |
| "…a lifetime-mortgage portfolio with interest roll-up, funded through a warehouse facility and reported to ESMA Annex 2 for securitisation." | Evidenced | Interest roll-up: `amortisation_type` = "Interest roll-up" for all 36 rows. Warehouse funding: `synthetic_onboarding_pack/warehouse_funding_agreement.md`. Annex 2: `default_regime: ESMA_Annex2` and the delivery report. |
| 36 exposures · £5,382,463 · as at 30 November 2025 | Evidenced | Computed from the canonical CSV; asserted to the penny by `tests/demo_pack_reproducible_test.py`. |
| Every KPI, chart and table in an answer | Evidenced | Engine output. Redacted, never recomputed — see `scripts/build_demo_pack.py`. |
| "Balance coverage 100.0% of the funded book" | Evidenced | The engine's own `reconciliation.coverage_by_balance_pct`. |
| Governance answer: "62 source headers mapped… 0 parse failures… 2 exceptions… Annex 2 preflight passed" | Evidenced | Read directly from `…_header_mapping_report.json`, `…_transform_report.json`, `validation/…_field_summary.csv`, `…_ESMA_Annex2_delivery_report.json`. |
| "The exceptions are real, and Trakt reports them rather than resolving them for you." | Evidenced | The two `ENUM_INVALID` / BLOCKING exceptions are genuinely present and unresolved in the committed validation output. |
| Refusal: "Temporal comparison needs two governed reporting periods…" | Demonstration | Accurate — one snapshot is published. |
| Refusal: "…Trakt holds a governed snapshot history and answers month-on-month…" | Evidenced | `mi_agent_api/snapshots.py`, `temporal_compare.py`, `evolution.py`, `docs/phase4_temporal_mi_foundations.md`. |
| Refusal: "…Trakt runs pipeline snapshots, the origination funnel, conversion analysis and expected-funding forecasts…" | Evidenced | `mi_agent_api/pipeline_prep.py`, `forecast_bridge.py`, `forecast_extrapolation.py`, `analytics/pipeline_expected_funding.py`. |
| Refusal: "Arrears, default, forbearance and loss analytics are standard Trakt portfolio analytics **where the portfolio's own data supports them**." | Evidenced (qualified) | Arrears/default/loss fields are in the canonical registry and the ESMA projections; the qualification is deliberate — this synthetic book carries none. |
| Refusal: "Exposure-level drill-through exists in the Trakt workspace, governed by role-based access within the client environment." | Evidenced + Deployment | Drill-through: `frontend/mi-agent-ui/src/components/DrillThroughPanel.tsx`, `mi_agent/tests/test_mi_drill_filters.py`. Role-based access is a deployment arrangement (`mi_agent_api/auth.py`) — worded as such. |
| Session limit message | Demonstration | `DEMO_MAX_QUESTIONS_PER_SESSION` / `DEMO_MAX_REPORTS_PER_SESSION`. |

### Report previews

| Copy | Class | Evidence |
|---|---|---|
| Document title "Investor & Funder MI Pack" | Evidenced | `configs/pptx/investor_pack.yaml` → `deck.name`. |
| Page order: Executive summary → Stratifications → Geographic exposure → Methodology & coverage | Evidenced | Mirrors the real slide order in the same file. |
| "In a client environment the same pack is generated as a branded PowerPoint from this identical governed dataset and delivered on schedule to each funding partner" | Evidenced + Deployment | Generation: `mi_agent_pptx/` + `mi_agent_api/decks.py`. Scheduled delivery is a deployment arrangement. |
| "The production pack also carries funded evolution, vintage cohort progression, pipeline, origination funnel, forecast bridge and risk limit pages, which need reporting history and a pipeline dataset that this public demonstration does not publish." | Evidenced | Those slide types exist in `investor_pack.yaml` and `mi_agent_pptx/deck.py`; the stated reason for their absence here is accurate. |
| "preview only, no document is produced" | Demonstration | No document generation, no download link, no storage path — asserted by unit and E2E tests. |

---

## E. Demo scope

| Copy | Class | Evidence |
|---|---|---|
| "This demonstration shows Trakt's conversational interface. In production, the same governed data layer powers portfolio analytics, reporting, monitoring and operational workflows across the organisation." | Evidenced | Analytics `analytics/` + `mi_agent/`; reporting `mi_agent_pptx/`; monitoring `mi_agent/risk_monitor/`; workflows `engine/orchestrator/`. |
| "The demonstration uses a wholly synthetic portfolio. No client or consumer information is displayed." | Demonstration | — |

---

## F. Omnichannel

| Card | Class | Evidence |
|---|---|---|
| **Microsoft 365 Copilot** — "Ask portfolio questions, request reports and retrieve current management information directly through Copilot." | Evidenced | Exactly the three shipped actions: `askTraktMi`, `getLatestInvestorDeck`, `getLatestCanonicalTape` (`docs/copilot_v1_implementation.md`). |
| **Teams** — "Access governed portfolio intelligence within existing team workflows and conversations." | Evidenced (narrow) | The declarative agent is distributed as a Teams app and is used inside Teams (`deploy/copilot-agent/manifest.json`, sideload instructions in `docs/copilot_v1_implementation.md`). Deliberately no claim of a bespoke Teams bot, adaptive cards or proactive messaging. |
| **Trakt Workspace** — "dashboards, drill-through, monitoring and portfolio investigation." | Evidenced | `frontend/mi-agent-ui/src/components/` — `DrillThroughPanel`, `RiskLimitsPanel`, `FundedSnapshotPanel`, `GeographyPanel`, `EvolutionPanel`, `ForecastView`. |
| **Automated Delivery** — "scheduled reports, alerts, data outputs and governance artefacts without manual intervention." | Evidenced | `apps/blob_trigger_app` (blob-triggered pipeline), `mi_agent_pptx/cli.py`, `export_audit_pack.py`, `mi_agent/risk_monitor/monitor.py`. |
| "The Copilot demonstration above is one way into Trakt, not the whole product." | Positioning | — |

---

## G. Capability stack

| Card | Class | Evidence |
|---|---|---|
| **Portfolio Integration** — ingestion, mapping, standardisation, deduplication, portfolio migration, acquisition integration | Evidenced | `engine/` Gate 1 semantic alignment + canonical transform; `agents/onboarding_agent.py`; `analytics_lib/migration.py`; `config/mna/diligence_scorecard.yaml`; `docs/mi_mna_target_architecture_and_build_plan.md`. |
| **Portfolio Analytics** — dashboards, natural-language analysis, drill-through, cohort, trend, concentration | Evidenced | `analytics/streamlit_app_erm.py`, `frontend/mi-agent-ui/`, `mi_agent/mi_query_executor.py`, `analytics_lib/cohort.py`, `analytics_lib/concentration.py`, `mi_agent_api/evolution.py`. |
| **Management Reporting** — KPI packs, board reporting, variance analysis, management commentary, scheduled MI | Evidenced | `analytics/mi_prep.py`, `mi_agent_pptx/`, `mi_agent_api/temporal_compare.py` (variance), `mi_agent_pptx/insight_resolver.py` (commentary), `config/mi/state_library.yaml`. |
| **Investor Reporting** — investor packs, stratifications, performance commentary, scheduled delivery, funding-partner reporting | Evidenced | `configs/pptx/investor_pack.yaml`, `analytics_lib/stratify.py`, `config/mi/stratification_catalogue.yaml`, `mi_agent_api/decks.py`. |
| **Regulatory Reporting** — **"ESMA Annex 2 reporting", "ESMA Annex 12 reporting"**, field validation, rule application, **"submission-ready XML"** | Evidenced | `config/regime/annex2_*`, `config/regime/annex12_*`, `engine/` regime projector + `engine/gate_5_delivery/xml_builder_investor.py` validated against the committed XSDs. **The generic word "Annex reporting" was replaced with the two regimes actually implemented, and "submission-ready outputs" narrowed to XML, which is what the pipeline emits.** |
| **Governance and Audit** — provenance, validation evidence, exception logs, versioning, approval records, reproducible outputs | Evidenced | `engine/gate_2_transform/lineage_tracker.py` (Gate 2.5), `exception_db.py`, `exception_queue.py`, `export_audit_pack.py`, `out/run_manifest.json`, `agents/review_schemas.py`. |
| **Portfolio Monitoring** — risk alerts, covenant monitoring, concentration limits, exception management, data-quality monitoring | Evidenced | `mi_agent/risk_monitor/`, `config/mi/risk_monitor.yaml`, `config/client/risk_limits_config.py`, `mi_agent_api/risk_limits.py`, `ingest_violations.py`. |
| **Omnichannel Intelligence** — conversational access, workflow actions, role-based delivery, reusable intelligence, scheduled outputs | Evidenced | As per section F, plus `mi_agent_api/auth.py` and `copilot_auth.py` for role-based delivery. |
| "Eight capability areas, connected through one data layer — not eight products bolted together." | Evidenced | One canonical model and one shared analytical service feed every output (`docs/mi_shared_service_architecture.md`, `docs/spine_audit_single_source_of_truth.md`). |

---

## H. Connected operating model

| Copy | Class | Evidence |
|---|---|---|
| Source data → governed data layer → analytics and business rules → Copilot · Workspace · Reports · Regulatory outputs · Alerts | Evidenced | The gate sequence in `README.md` and `engine/orchestrator/trakt_run.py`, then the four output paths above. |
| "Trakt calculates the answer once, governs it centrally and distributes it through every required channel." | Evidenced | `mi_agent_api/mi_service.py` is the single analytical implementation behind both channels; `mi_agent_api/tests/test_channel_parity.py` asserts the two channels agree across the golden-question library. |
| "…the figure in a board pack, an investor report, a regulatory submission and a Copilot answer is the same figure — reconciled by construction rather than by comparison." | Evidenced | Deck slides render from the same MI payloads as the dashboard (`configs/pptx/investor_pack.yaml` header, `mi_agent_pptx/mi_api.py`); channel parity is tested. |

---

## I. Portfolio lifecycle

| Stage | Class | Evidence |
|---|---|---|
| Launch or acquire a portfolio | Evidenced | `agents/onboarding_agent.py`, `docs/onboarding_v1_demo.md`, `synthetic_onboarding_pack/`. |
| Run the portfolio | Evidenced | `mi_agent/`, `analytics/`, scheduled MI. |
| Finance the portfolio | Evidenced | `configs/pptx/investor_pack.yaml`, `synthetic_onboarding_pack/warehouse_funding_agreement.md`. |
| Securitise or refinance | Evidenced | `config/regime/annex2_*`, `annex12_*`, `analytics_lib/stratify.py`, `synthetic_onboarding_pack/synthetic_securitisation_summary.md`. |
| Integrate additional portfolios | Evidenced | `analytics/pipeline_snapshot_selector.py`, multi-client config under `config/clients/`, `docs/mi_mna_target_architecture_and_build_plan.md`. |

---

## J. Governance and trust

| Claim | Class | Evidence / wording note |
|---|---|---|
| "Synthetic public demonstration" | Demonstration | — |
| "Deterministic calculation logic… The same question returns the same number, every time and in every channel." | Evidenced | Deterministic parser + executor (`mi_agent/llm_query_parser.py` deterministic path, `mi_agent/mi_query_executor.py`); parity tested (`test_channel_parity.py`); the demo pack's reproducibility is itself asserted by a test. |
| "Traceable source-to-output lineage… any published figure can be traced back to source." | Evidenced | `engine/gate_2_transform/lineage_tracker.py`, the four committed gate reports, `out/run_manifest.json`. |
| "Role-based access in production. Client deployments sit behind Microsoft Entra ID; Copilot actions require a validated bearer token and **fail closed** when they are not configured." | Deployment + Evidenced | `mi_agent_api/copilot_auth.py` defaults to `entra` and returns 503 unconfigured; `mi_agent_api/auth.py` Easy-Auth guard. Deliberately says "in production" — the landing page itself is public. |
| "Client-isolated environments. Trakt runs one deployment per client…" | Deployment | `docs/copilot_v1_implementation.md` § "Tenancy": one deployment per client, selected by `MI_AGENT_CLIENT_ID` / `MI_AGENT_PLATFORM_URI`. |
| "Configurable data retention… configured per client environment rather than fixed by the platform." | Deployment | Storage roots and retention are environment variables (`deploy/trakt-mi-api/app_settings.example.json`). No retention guarantee is claimed. |
| "Controlled document and report delivery… Storage credentials and storage paths never leave the server." | Evidenced | `copilot_actions.py`: HMAC-signed, short-lived (default 300 s) tokens redeemed server-side; the API streams the bytes; no SAS token or account key ever appears in a URL. |
| "Deployment, hosting, retention and access-control arrangements are configured for each client environment, and are agreed as part of onboarding rather than being fixed by the platform." | Deployment | The required note. |

### Excluded — considered and deliberately not said

| Not claimed | Why |
|---|---|
| SOC 2, ISO 27001, or any certification | Trakt does not hold them. No certification is named anywhere on the page. |
| "GDPR compliant" | Compliance is an organisational assessment, not a code property. The page describes retention and access as *configurable*, and says nothing more. |
| A hosting geography ("UK-hosted", "EU data residency") | Region is a deployment choice; no region is fixed in repository configuration. |
| Penetration tested / independently assessed | No evidence of a completed test. |
| Regulatory approval, authorisation or endorsement | Trakt produces reporting *artefacts*; it holds no permission. |
| "Bank-grade", "military-grade", "zero-risk", "fully secure" | Unfalsifiable. Absent. |
| "AI-powered", "AI for loan tapes", "chat with your CSV", "upload your loan tape" | Explicitly avoided. The demo's answers are deterministic, not generative, and the page says so. |
| Uptime or SLA figures | None agreed. |
| Named clients, logos, testimonials, customer counts | None available. |
| "Real-time" | The pipeline is batch and snapshot-based. |
| Any figure not produced by the engine | Every number on the page comes from `data/demo-pack.json`. |

---

## K. Final CTA and footer

| Copy | Class | Evidence |
|---|---|---|
| "See Trakt applied to your operating model" / "We will demonstrate how Trakt can integrate with your existing portfolio data, reporting requirements and Microsoft 365 environment." | Positioning | An offer to demonstrate, not a capability claim. Microsoft 365 integration is evidenced (`deploy/copilot-agent/`). |
| "We use your details only to respond, and do not add you to a marketing list." | Deployment | **Operational commitment.** True of the delivery adapters as implemented (one destination, no list subscription). Honouring it is a business undertaking — if lead handling changes, this line must change with it. |
| Footer: "specialist lenders, non-bank lenders, private-credit managers, servicing businesses and securitisation participants" | Positioning | The audience the brief specifies; consistent with an ERM/securitisation codebase. |

---

## Future capabilities — do not claim yet

Present in the repository as scaffolding, design documents or partial
implementations. None is mentioned on the page.

| Capability | Status |
|---|---|
| Scenario modelling / stress testing | `mi_agent_api/scenario.py`, `analytics/scenario_engine.py` exist, but no client-facing scenario product is described. |
| M&A due-diligence scorecard | `config/mna/diligence_scorecard.yaml`, `due_diligence/` — early. |
| LLM-assisted query parsing | `mi_agent/llm_query_parser.py` supports an Anthropic path, but the deterministic parser is what ships and what the page describes. The demo uses no model at all. |
| Approval workflow / artefact versioning | `docs/copilot_v1_implementation.md` § "Known v1 limitations" states there is no approval state or artefact versioning. The page's "Approval records" bullet under Governance and Audit refers to onboarding review records (`agents/review_schemas.py`, `cli/onboarding_review_cli.py`), which do exist — **it must not be read as a report-approval workflow.** |
| Annex 12 investor XML at production maturity | Implemented, but `docs/xml_readiness_remediation_roadmap.md` lists open items. The page claims the reporting capability, not production certification. |
| Public API access for clients | "APIs" appears only inside the Omnichannel Intelligence capability list, matching the existing internal FastAPI surface. No public developer API is offered. |
