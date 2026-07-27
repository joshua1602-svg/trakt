# Content map — every claim on the page, and what backs it

The purpose of this document is to stop marketing running ahead of the product.
Every visible claim is classified:

| Class | Meaning |
|---|---|
| **Evidenced** | Implemented in this repository. The cited file is the proof. |
| **Demonstration** | Describes what the public demo does, here and now. |
| **Deployment** | True of a configured client deployment, not a platform guarantee. Worded to say so. |
| **Roadmap** | Not shipped. Labelled as roadmap on the page itself, never mixed in with live capability. |
| **Positioning** | A statement of what Trakt is for. Not a capability claim. |
| **Excluded** | Considered and deliberately **not** said, because the repository does not support it. |

Anyone editing page copy should add a row here, or delete one.

The page is seven sections, in this order: value proposition · capabilities ·
delivery model · how it works · where it applies · example · contact.

---

## A. Navigation

| Copy | Class | Evidence |
|---|---|---|
| Capabilities · Delivery · How it works · Example · Book a demo | Positioning | — |

---

## 1. Value proposition

| Copy | Class | Evidence |
|---|---|---|
| "One governed portfolio dataset. Every book, every report." | Positioning | The operating-system claim. Leads on the governed single source rather than on channel — channel breadth is the easiest claim for an incumbent to copy, so it lives in section 3 instead. |
| "Trakt normalises loan tapes, servicing extracts, valuations and funding data into one governed model that drives management, investor and regulatory reporting." | Evidenced | Ingest→canonical→validate→output pipeline: `engine/orchestrator/trakt_run.py`, gates in `README.md`. Reporting outputs: `mi_agent_pptx/`, `configs/pptx/investor_pack.yaml`, `engine/gate_5_delivery/`. |
| "Every figure is reconciled by construction rather than by comparison." | Evidenced | One analytical implementation behind every channel (`mi_agent_api/mi_service.py`); deck slides render from the same MI payloads as the dashboard (`mi_agent_pptx/mi_api.py`); parity asserted by `mi_agent_api/tests/test_channel_parity.py` across the golden-question library. **Moved here from "Delivered as", where it was buried.** |
| Proof point: "Deterministic engine — same question, same number, every channel" | Evidenced | Deterministic parser + executor (`mi_agent/llm_query_parser.py` deterministic path, `mi_agent/mi_query_executor.py`); parity tested as above; the demo pack's reproducibility is itself asserted by `tests/demo_pack_reproducible_test.py`. |
| Proof point: "Traceable lineage — every published figure ties back to source" | Evidenced | `engine/gate_2_transform/lineage_tracker.py` (Gate 2.5), the four committed gate reports, `out/run_manifest.json`. |
| Proof point: "Client-isolated environments and controlled data handling" | Deployment | One deployment per client, selected by `MI_AGENT_CLIENT_ID` / `MI_AGENT_PLATFORM_URI` (`docs/copilot_v1_implementation.md` § "Tenancy"). Document delivery is HMAC-signed and server-redeemed (`mi_agent_api/copilot_actions.py`). |
| Interface preview showing £5,382,463 / 36 exposures / regional bars | Evidenced | Rendered from `data/demo-pack.json`, produced by the engine from `synthetic_demo/output/…canonical_typed.csv`. Not a mock-up. |
| Preview footer "Deterministic · As at … · Synthetic portfolio" | Demonstration | A provenance label travelling with the figures, not the page's disclaimer — that appears once, in section 6. |

**Removed from this section:** the synthetic-portfolio disclaimer paragraph, the
"Use Trakt through Microsoft 365 Copilot, Teams…" channel line (now section 3),
and the preview caption "The same governed answer reaches Teams, the Trakt
workspace and scheduled reports."

---

## 2. Capabilities

Six tiles, one sentence each. Reduced from eight; the "Illustrative
capabilities" disclosure under every tile is gone, because "illustrative" tells
a buyer the capability may not exist.

| Tile | Class | Evidence |
|---|---|---|
| **Multi-source portfolio ingestion** (lead tile) — "Any number of books — direct originations, acquired back books, sponsored securitisations — held in one governed model, each reportable on its own and in aggregate." | Evidenced (with one qualification) | `trakt_core/portfolio.py` is the governed portfolio contract: `PortfolioRegistry`, `resolve_scope()`, `resolve_capabilities()`, `ScopeCoverage`. The hierarchy is dynamic and metadata-driven — "`direct` means every governed portfolio whose `source_portfolio_type` is direct, not `direct_001`". Total / by-type / individual-portfolio scope is rendered by `frontend/mi-agent-ui/src/components/PortfolioContextSelector.tsx` from `GET /mi/portfolio-context`; aggregation across a lens is in `mi_agent_api/datasets.py`. Ingestion and mapping: `engine/` Gate 1, `agents/onboarding_agent.py`. **Qualification:** the shipped type vocabulary is `direct` / `acquired` (`trakt_core/portfolio.py:45-47`); `PortfolioRegistry.types()` orders and returns any further type present, so a securitisation vehicle is a third governed type the model already holds rather than a shipped enum member. The claim is about the model, not about a securitisation-specific feature. |
| **Portfolio analytics and monitoring** — dashboards, cohorts, concentrations, drill-through, limit monitoring, funded book and pipeline | Evidenced | `analytics/streamlit_app_erm.py`, `frontend/mi-agent-ui/`, `mi_agent/mi_query_executor.py`, `analytics_lib/cohort.py`, `analytics_lib/concentration.py`, `mi_agent_api/evolution.py`, `mi_agent/risk_monitor/`, `config/mi/risk_monitor.yaml`, `mi_agent_api/risk_limits.py`, `mi_agent_api/pipeline_prep.py`. **Merged:** the former "Portfolio Monitoring" tile said the same thing with a threshold attached. |
| **Management reporting** — recurring MI, one consistent answer across finance, risk and operations | Evidenced | `analytics/mi_prep.py`, `mi_agent_pptx/`, `mi_agent_api/temporal_compare.py`, `mi_agent_pptx/insight_resolver.py`, `config/mi/state_library.yaml`. |
| **Investor reporting** — packs, stratifications, commentary | Evidenced | `configs/pptx/investor_pack.yaml`, `analytics_lib/stratify.py`, `config/mi/stratification_catalogue.yaml`, `mi_agent_api/decks.py`. |
| **Regulatory reporting** — ESMA Annex 2 and Annex 12, field-validated, submission-ready XML | Evidenced | `config/regime/annex2_*`, `config/regime/annex12_*`, `engine/` regime projector, `engine/gate_5_delivery/xml_builder_investor.py` validated against the committed XSDs. |
| **Governance and audit** — lineage source header to published figure, validation evidence, reproducible runs | Evidenced | `engine/gate_2_transform/lineage_tracker.py`, `exception_db.py`, `exception_queue.py`, `export_audit_pack.py`, `out/run_manifest.json`. |

**Removed:** the "Omnichannel Intelligence" tile (a delivery mode, not a
capability — its substance is now section 3), the separate "Portfolio
Integration" tile (merged into the lead tile, which made the same claim more
weakly), the separate "Portfolio Monitoring" tile (merged into analytics), and
all eight "Illustrative capabilities" disclosure lists.

---

## 3. Delivery model

Two groups. Flattening them would spend the credibility the page has from
distinguishing what ships from what is planned.

### Available today

| Mode | Class | Evidence |
|---|---|---|
| **Managed service** — "Recurring reporting, regulatory output and governance artefacts, produced with no user interaction." | Evidenced | `apps/blob_trigger_app` (blob / Event Grid triggered pipeline), `mi_agent_pptx/cli.py`, `export_audit_pack.py`, `mi_agent/risk_monitor/monitor.py`. **Wording note:** says *recurring*, not *scheduled*. There is no timer trigger or cron anywhere in the repository; what ships is event-triggered and CLI-invocable. Scheduling is a deployment arrangement, so the page does not claim it as a platform feature. |
| **Trakt Agent workspace** — "The full analytical environment: dashboards, charting, drill-through and portfolio investigation." | Evidenced | `frontend/mi-agent-ui/src/components/` — `DrillThroughPanel`, `RiskLimitsPanel`, `FundedSnapshotPanel`, `GeographyPanel`, `EvolutionPanel`, `ForecastView`, `ArtifactCanvas`. |
| **Microsoft 365 Copilot and Teams** — "Portfolio questions and artefact requests inside the tools your teams already use." | Evidenced | `deploy/copilot-agent/` + `mi_agent_api/copilot_actions.py` — a **declarative agent**, packaged as a Teams app (`deploy/copilot-agent/manifest.json`), exposing exactly three actions: `askTraktMi`, `getLatestInvestorDeck`, `getLatestCanonicalTape` (`docs/copilot_v1_implementation.md`). There is no standalone native Teams bot — no bot registration, no adaptive cards, no proactive messaging — so the copy names the surfaces, not a second product. |

### Roadmap — labelled as such on the page, in muted grey, with no accent

| Mode | Class | Evidence |
|---|---|---|
| **Enterprise agent deployment** — "Trakt running inside a client's own agent estate." | Roadmap | `trakt_core/context.py:33` defines `CHANNEL_ENTERPRISE_AGENT` as *"reserved for a client-owned agent"*. `docs/governed_capability_architecture.md` §2 documents the adapter shape; the only working example is a test fixture (`tests/test_governance_artefacts_and_envelope.py::enterprise_agent_endpoint`). No shipped route module. |
| **Agent-to-agent integration** — "Upstream and downstream systems consulting the governed layer directly." | Roadmap | Same channel vocabulary (`CHANNEL_AGENT_TO_AGENT`) and the same documented adapter shape, plus `docs/governed_capability_architecture.md` §6 *Known gaps*: "no outbound notification seam — an agent-to-agent workflow can call in, but Trakt cannot call back." |

| Closing copy | Class | Evidence |
|---|---|---|
| "The answer is calculated once and distributed; the channel never changes it." | Evidenced | `mi_agent_api/mi_service.py` is the single analytical implementation behind every channel; `test_channel_parity.py`. |
| "Deployments are isolated per client behind Microsoft Entra ID, with retention and document delivery set at onboarding." | Deployment | `mi_agent_api/copilot_auth.py` defaults to `entra` and returns 503 unconfigured; `mi_agent_api/auth.py` Easy-Auth guard; storage roots and retention are environment variables (`deploy/trakt-mi-api/app_settings.example.json`); short-lived server-redeemed document links (`copilot_actions.py`). This one line carries what the old "Why you can rely on it" section said across four cards. |

---

## 4. How it works

| Copy | Class | Evidence |
|---|---|---|
| Source data → governed data layer → analytics and business rules | Evidenced | The gate sequence in `README.md` and `engine/orchestrator/trakt_run.py`. |

**Removed:** the "Delivered as" block (chips reading Copilot · Workspace ·
Reports · Regulatory outputs · Alerts, plus the reconciliation paragraph). The
chips duplicated section 3; the paragraph's best sentence moved to section 1.

---

## 5. Where it applies

Four stages, down from five. The old stages 1 ("Launch or acquire a portfolio")
and 5 ("Integrate additional portfolios") were the same claim written twice.

| Stage | Class | Evidence |
|---|---|---|
| **Onboard** | Evidenced | `agents/onboarding_agent.py`, `docs/onboarding_v1_demo.md`, `synthetic_onboarding_pack/`, multi-client config under `config/clients/`. |
| **Run** | Evidenced | `mi_agent/`, `analytics/`, `mi_agent/risk_monitor/`. |
| **Finance** | Evidenced | `configs/pptx/investor_pack.yaml`, `synthetic_onboarding_pack/warehouse_funding_agreement.md`, `config/regime/annex2_*`, `annex12_*`, `analytics_lib/stratify.py`. |
| **Exit** — "Prepare validated datasets, stratifications and portfolio reporting for sell-side mandates." | Evidenced | `analytics_lib/stratify.py`, `config/mi/stratification_catalogue.yaml`, `export_audit_pack.py`, `config/mna/diligence_scorecard.yaml`, `due_diligence/`. Claims dataset and reporting preparation only — no valuation, no transaction advisory, no buyer-side analytics. |

---

## 6. Example

The only demo surface on the page, and the only place the synthetic-portfolio
disclaimer appears.

| Copy | Class | Evidence |
|---|---|---|
| "Every answer comes from the deterministic engine that serves the workspace and Microsoft 365 Copilot." | Evidenced | Both channels are thin adapters over one service (`docs/mi_shared_service_architecture.md`); the pack is generated through `mi_agent.mi_agent_workflow.run_mi_agent_query` + `mi_agent_api.adapters.adapt_workflow_result` — the same chain. |
| "The portfolio is wholly synthetic, and the page accepts no uploads." | Demonstration | True by construction — no upload endpoint exists; the pack is the only data source. **This is the single instance.** It previously appeared five times (hero, Copilot card, demo card, "What the data is", footer). |
| Client: **Synthetic Demo Lender**, ERE Funding Limited, UK equity release, GBP | Evidenced | `synthetic_demo/config/config_client_SYNTHETIC_ERM.yaml`. |
| "…a lifetime-mortgage portfolio with interest roll-up, funded through a warehouse facility and reported to ESMA Annex 2 for securitisation." | Evidenced | Interest roll-up: `amortisation_type` = "Interest roll-up" for all 36 rows. Warehouse funding: `synthetic_onboarding_pack/warehouse_funding_agreement.md`. Annex 2: `default_regime: ESMA_Annex2` and the delivery report. |
| 36 exposures · £5,382,463 · as at 30 November 2025 | Evidenced | Computed from the canonical CSV; asserted to the penny by `tests/demo_pack_reproducible_test.py`. |
| Every KPI, chart and table in an answer | Evidenced | Engine output. Redacted, never recomputed — see `scripts/build_demo_pack.py`. |
| Governance answer: "62 source headers mapped… 0 parse failures… 2 exceptions… Annex 2 preflight passed" | Evidenced | Read directly from `…_header_mapping_report.json`, `…_transform_report.json`, `validation/…_field_summary.csv`, `…_ESMA_Annex2_delivery_report.json`. |
| "The exceptions are real, and Trakt reports them rather than resolving them for you." | Evidenced | The two `ENUM_INVALID` / BLOCKING exceptions are genuinely present and unresolved in the committed validation output. |
| **"Trakt declines what it cannot derive."** | Evidenced | Every refusal path in `src/lib/intents.ts` and the pack's `unsupported` set. **Promoted** from small grey footnote text to a headline in the accent colour: refusing is the differentiator against an LLM wrapper, and it was the quietest thing on the page. |
| The refusal explanations (temporal comparison, pipeline, arrears, drill-through) | Evidenced / Demonstration | Unchanged. `mi_agent_api/snapshots.py`, `temporal_compare.py`, `evolution.py`, `pipeline_prep.py`, `forecast_bridge.py`; drill-through in `frontend/mi-agent-ui/src/components/DrillThroughPanel.tsx` with role-based access as a deployment arrangement. |
| Session question counter | Demonstration | `DEMO_MAX_QUESTIONS_PER_SESSION`. **Now silent until three questions remain** (`COUNTER_VISIBLE_FROM` in `CopilotDemo.tsx`) — the cap is real and server-enforced, but a counter running from the first question reads as metering on a page whose job is to create appetite. The terminal `limit_reached` message is unchanged; it is produced by the demo backend. |

### Report previews

Unchanged. Document title, page order, the production-pack note and "preview
only, no document is produced" all stand as previously evidenced
(`configs/pptx/investor_pack.yaml`, `mi_agent_pptx/`, `mi_agent_api/decks.py`).

**Removed:** the standalone overview-video section and its placeholder, the
"What this shows" and "What the data is" explainer cards.

---

## 7. Contact and footer

| Copy | Class | Evidence |
|---|---|---|
| "See Trakt applied to your operating model" | Positioning | An offer to demonstrate, not a capability claim. |
| "We will demonstrate Trakt against your own portfolio data, reporting requirements and Microsoft 365 environment." | Positioning | Microsoft 365 integration is evidenced (`deploy/copilot-agent/`). |
| Required fields: name, work email, company. Role and message optional. | — | `src/lib/lead-validation.ts`. Role was required; three fields are enough to hold a conversation and every extra required field costs enquiries. Role is still delivered when supplied (`src/lib/leads.ts`). |
| "We use your details only to respond, and do not add you to a marketing list." | Deployment | **Operational commitment.** True of the delivery adapters as implemented: exactly one destination, no list subscription, and the app itself retains nothing. Honouring it beyond that is a business undertaking — if lead handling changes, this line must change with it. See README § "Who owns incoming leads". |
| Footer: audience line and copyright | Positioning | The audience the brief specifies. |

**Removed:** the footer's synthetic-portfolio paragraph (now once, in section 6),
and the "Watch the overview" secondary CTA, which pointed at the deleted video
section.

---

## The green accent

`mint-400` / `#36c2a8`, taken verbatim from
`frontend/mi-agent-ui/src/index.css:21` — the MI Agent's UI green. In the
product it marks a proven state (`ValidationArtifactView` "Pass",
`RiskLimitsPanel` green status, `PipelineSnapshotPanel` "Clean", `HeaderBar`
"Prod"), and it does the same job here and no other.

| Where | Why |
|---|---|
| The three trust proof points (tick marks; borders from `sm` upward) | Proven properties of the engine. |
| "Every figure is reconciled by construction rather than by comparison." | The page's strongest verification claim. |
| The lead ingestion tile — border and icon only, never a fill | Marks the differentiator without turning into a coloured block. |
| "Available today" in the delivery model | A shipped/not-shipped signal. Roadmap stays muted grey, which does the distinguishing work. |
| "Trakt declines what it cannot derive." | A governance guarantee. |
| "Deterministic" in the hero preview footer | A confirmatory provenance signal. |

Not used for: primary CTAs (periwinkle, so the accent stays meaningful rather
than becoming a second brand colour), body copy, eyebrows, headings,
backgrounds, filled panels, or roadmap items.

**Contrast:** 8.5:1 on `navy-950`, 8.0:1 on `navy-900` — WCAG AA and AAA for
normal text, so no lighter tint was derived.

**Deliberately not used:** `#2E7D5B` (`THEME.positive` / `THEME.rag.green` in
`frontend/mi-agent-ui/src/lib/theme.ts`, mirrored in
`mi_agent/mi_chart_factory.py:78`). That is the chart and RAG **fill** green; it
reaches only 3.8:1 on `navy-950` and fails AA as type.

---

## Excluded — considered and deliberately not said

| Not claimed | Why |
|---|---|
| SOC 2, ISO 27001, or any certification | Trakt does not hold them. No certification is named anywhere on the page. |
| "GDPR compliant" | Compliance is an organisational assessment, not a code property. |
| A hosting geography ("UK-hosted", "EU data residency") | Region is a deployment choice; no region is fixed in repository configuration. |
| Penetration tested / independently assessed | No evidence of a completed test. |
| Regulatory approval, authorisation or endorsement | Trakt produces reporting *artefacts*; it holds no permission. |
| "Bank-grade", "military-grade", "zero-risk", "fully secure" | Unfalsifiable. Absent. |
| "AI-powered", "AI for loan tapes", "chat with your CSV", "upload your loan tape" | Explicitly avoided. The demo's answers are deterministic, not generative. |
| **"Scheduled" delivery as a platform capability** | No timer trigger or cron exists. The page says *recurring*. |
| **Enterprise-agent or agent-to-agent availability** | Reserved channels only. Both appear on the page under an explicit Roadmap label. |
| Uptime or SLA figures | None agreed. |
| Named clients, logos, testimonials, customer counts | None available. |
| "Real-time" | The pipeline is batch and snapshot-based. |
| Any figure not produced by the engine | Every number on the page comes from `data/demo-pack.json`. |

---

## Future capabilities — do not claim yet

Present in the repository as scaffolding, design documents or partial
implementations. None is mentioned on the page.

| Capability | Status |
|---|---|
| Scenario modelling / stress testing | `mi_agent_api/scenario.py`, `analytics/scenario_engine.py` exist, but no client-facing scenario product is described. |
| M&A due-diligence scorecard | `config/mna/diligence_scorecard.yaml`, `due_diligence/` — early. The Exit stage claims dataset preparation only. |
| LLM-assisted query parsing | `mi_agent/llm_query_parser.py` supports an Anthropic path, but the deterministic parser is what ships and what the page describes. The demo uses no model at all. |
| Approval workflow / artefact versioning | `docs/copilot_v1_implementation.md` § "Known v1 limitations" states there is no approval state or artefact versioning. |
| Annex 12 investor XML at production maturity | Implemented, but `docs/xml_readiness_remediation_roadmap.md` lists open items. The page claims the reporting capability, not production certification. |
| Public API access for clients | No public developer API is offered, and "APIs" no longer appears anywhere on the page. |
| Product overview video | No `.mp4`/`.webm`/`.mov` and no hosted video URL exists in this repository. The placeholder section was removed rather than left saying "the recorded walkthrough will appear here", which tells a visitor the site is unfinished. See README § "Adding a product overview video". |
