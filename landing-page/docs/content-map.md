# Content map — every claim on the page, and what backs it

The purpose of this document is to stop marketing running ahead of the product.
Every visible claim is classified:

| Class | Meaning |
|---|---|
| **Evidenced** | Implemented in this repository. The cited file is the proof. |
| **Demonstration** | Describes what the public demo does, here and now. |
| **Illustrative** | A depiction of product workflow whose figures are not engine output. Labelled as such on the page itself. |
| **Deployment** | True of a configured client deployment, not a platform guarantee. Worded to say so. |
| **Roadmap** | Not shipped. Labelled as roadmap on the page itself, never mixed in with live capability. |
| **Positioning** | A statement of what Trakt is for. Not a capability claim. |
| **Excluded** | Considered and deliberately **not** said, because the repository does not support it. |

**Pass 3 repositioned the page.** The previous page presented a governed
reporting dataset; the product had moved beyond it. The page now presents one
governed portfolio layer powering monitoring, forecasting, covenant controls,
reporting and portfolio Q&A. Two Pass-2 statements in this document were found
to be **factually stale against the repository** and are corrected below (see
§7 and the Excluded list): the "no timer trigger or cron anywhere" claim
(`function_app.py:83` is an Azure timer trigger draining the Teams notification
outbox) and the "no Teams bot, no adaptive cards, no proactive messaging" claim
(`mi_agent_api/teams_bot.py`, `trakt_notifications/cards.py` and the
`trakt_notifications/` outbox/delivery pipeline are all three).

The demonstration still runs on three governed books across two reporting
periods (`synthetic_demo/output/multibook/…`, built by
`synthetic_demo/run_multibook_pipeline.sh`). Book identities mirror
`demo_platform/config.py`. Every demo figure is as at 30 June 2026.

Anyone editing page copy should add a row here, or delete one.

The page is nine sections, in this order: value proposition · platform ·
controls & forward risk · governed onboarding · lenses · portfolio
intelligence (the example) · governance & platform · reporting band · contact.

---

## A. Navigation

| Copy | Class | Evidence |
|---|---|---|
| Platform · Controls · Onboarding · Intelligence · Book a demo | Positioning | — |

---

## 1. Value proposition

| Copy | Class | Evidence |
|---|---|---|
| "One governed view of your lending portfolios." | Positioning | The operating-layer claim, plural deliberately: multi-book is the differentiator. Leads on the governed layer rather than on reporting or on AI — both are outputs. |
| "Trakt connects loan data, documents and funding requirements into a single governed layer — then runs monitoring, forecasting, covenant controls, reporting and portfolio Q&A from it." | Evidenced | Layer: ingest→canonical→validate→output pipeline (`engine/orchestrator/trakt_run.py`, gates in `README.md`). Monitoring: `mi_agent/risk_monitor/`, `frontend/mi-agent-ui/`. Forecasting: `mi_agent_api/pipeline_prep.py`, `forecast_bridge.py`, `evolution.py`. Covenant controls: `mi_agent/concentration_tests/`, `config/risk/concentration_test_library.yaml`. Reporting: `mi_agent_pptx/`, `engine/gate_5_delivery/`. Q&A: `mi_agent/mi_query_executor.py`, `deploy/copilot-agent/`. Funding requirements → controls: `mi_agent/risk_monitor/schedule8_extractor.py` + `config/clients/client_001/risk_limits_extracted.yaml`. |
| "Every figure is reconciled by construction rather than by comparison." | Evidenced | One analytical implementation behind every channel (`mi_agent_api/mi_service.py`); parity asserted by `mi_agent_api/tests/test_channel_parity.py`. |
| Proof point: "Deterministic engine — same question, same number, every channel" | Evidenced | Deterministic parser + executor (`mi_agent/llm_query_parser.py` deterministic path, `mi_agent/mi_query_executor.py`); parity tested as above; demo-pack reproducibility asserted by `tests/demo_pack_reproducible_test.py`. |
| Proof point: "Traceable lineage — every published figure ties back to source" | Evidenced | `engine/gate_2_transform/lineage_tracker.py` (Gate 2.5), the committed gate reports, `out/run_manifest.json`. |
| Proof point: "Client-isolated environments and controlled data handling" | Deployment | One deployment per client (`docs/copilot_v1_implementation.md` § "Tenancy"); tenant authorisation enforced in the platform core (`trakt_core/tenancy.py`, `tests/test_governance_context_and_tenancy.py`); HMAC-signed, server-redeemed document delivery (`mi_agent_api/copilot_actions.py`). |
| Interface preview showing three books, a platform total and a sponsor total | Evidenced | Rendered from `data/demo-pack.json`. Unchanged from Pass 2. |
| Preview footer "Deterministic · As at … · Synthetic portfolio" | Demonstration | A provenance label travelling with the figures. |

---

## 2. Platform — "Build the portfolio once. Use it everywhere."

Replaces the Pass-2 "How it works" flow and the capability grid's identity.
The governed layer is deliberately not a tile: it is the frame the outputs sit
on, which makes the "not another output beside your spreadsheets" point
structurally.

| Copy | Class | Evidence |
|---|---|---|
| Data and documents → one governed portfolio layer → every output | Evidenced | The gate sequence in `README.md` and `engine/orchestrator/trakt_run.py`; document intake via `engine/onboarding_agent/file_classifier.py` and `document_extractor.py` (text/markdown; PDF/DOCX is a stated placeholder, so the page never claims a file format). |
| Output chips: Portfolio MI · Forecasting · Risk & covenant controls · Investor reporting · Regulatory reporting · AI & Copilot interaction | Evidenced | Respectively: `analytics/`, `frontend/mi-agent-ui/`; `mi_agent_api/pipeline_prep.py` + `evolution.py` + `forecast_bridge.py`; `mi_agent/concentration_tests/` + `mi_agent/risk_monitor/`; `configs/pptx/investor_pack.yaml` + `mi_agent_api/decks.py`; `config/regime/` + `engine/gate_5_delivery/`; `deploy/copilot-agent/` + `mi_agent_api/copilot_actions.py`. |
| "Management, investor, risk and regulatory views are lenses on the same truth — never separate versions of it." | Evidenced | Single analytical implementation (`mi_agent_api/mi_service.py`); deck slides render from the same MI payloads as the dashboard (`mi_agent_pptx/mi_api.py`). |

---

## 3. Controls and forward risk

New in Pass 3, and the page's strongest differentiator: the Pass-2 page
compressed all of this into the words "limit monitoring".

| Copy | Class | Evidence |
|---|---|---|
| "Trakt structures concentration and covenant requirements from facility documentation into controls your team reviews and activates." | Evidenced | `mi_agent/risk_monitor/schedule8_extractor.py` — deterministic extraction of structured limits (category, value, direction, unit, source snippet, confidence, `needs_review`) from a concentration-limit schedule; committed output `config/clients/client_001/risk_limits_extracted.yaml` (15 limits, 1 flagged for review); tested by `tests/concentration_tests/test_schedule_8.py`. Review/approval before activation: operator-approved `ActiveConfiguration` (`mi_agent/concentration_tests/store.py`, `tests/concentration_tests/test_governance.py`) and the OCC approval workflow (`operations_control/engine.py`). **Deliberate wording:** "structures … for review", never "AI reads your contracts" — extraction is deterministic, text-based and human-reviewed; PDF/DOCX parsing is a placeholder. |
| "Every active control is evaluated three ways — against the funded book, against the expected forecast, and against the full pipeline" | Evidenced | `mi_agent/concentration_tests/forward.py` evaluates each approved test in three explicitly-labelled states: `funded`, `expected_forecast` (pipeline weighted by governed completion probability), `full_pipeline` (labelled a stress, never a prediction). Surfaced in `frontend/mi-agent-ui/src/components/risk/ConcentrationDetailPanel.tsx`, `mi_agent_pptx/concentration.py` and the Teams insight cards. |
| "…with the projected breach horizon when a limit is approaching." | Evidenced | `expected_breach_horizon` and `pipeline_drivers` in `mi_agent/concentration_tests/forward.py`; `identify_emerging_risks`. |
| "Know what is breached today — and what the portfolio is moving toward." | Positioning | The commercial statement of the three-state evaluation above. Carries the green accent as the page's key forward-risk claim. |
| Path chips: Documented requirement → Structured control → Reviewed → Active | Evidenced | The extractor → review → approved-configuration → evaluation chain above. Activation is a human decision; the page keeps the review step visible. |
| Control preview: Geographic concentration ≤ 30% — Funded 24.1% Pass · Expected forecast 28.7% Warning · Including full pipeline 31.4% Projected breach · horizon Nov 2026 · Single obligor 6.2% Pass | Illustrative | The **workflow** depicted is live (rows above); the **figures** are not engine output and the card is labelled "Illustrative" on the page. This is the single sanctioned exception to "every number on the page comes from the engine" — see the Excluded list. Inside this product depiction the product's RAG semantics apply (mint pass / amber warning / rose projected breach) — documented in `app/globals.css`. |

---

## 4. Governed onboarding

New in Pass 3. The Pass-2 page gave onboarding one lifecycle card.

| Copy | Class | Evidence |
|---|---|---|
| "Trakt's onboarding agent interprets source tapes and documentation, proposes mappings and configuration, and routes every decision through human review before activation." | Evidenced | `engine/onboarding_agent/` (60 modules): LLM-assisted mapping under a deterministic-first policy (`llm_assisted_mapping.py`, `llm_policy.py`, `tests` incl. `test_onboarding_deterministic_first.py`, `test_onboarding_llm_cost_policy.py`), mapping review queue and mapping memory; `agents/onboarding_agent.py`. Human review and governed activation: `operations_control/` — workflow engine, onboarding case management, approval before `activate()`, publication, recovery — with its own deployed API and React UI (`frontend/operations-control-ui/`, `.github/workflows/deploy-occ-frontend.yml`) and 17 test modules under `tests/operations_control/`. |
| Steps: Source data and documents → Assisted interpretation → Governed configuration → Live portfolio | Evidenced | The chain above; "Nothing reaches the governed layer unapproved" is the OCC approval gate (`tests/operations_control/test_onboarding.py`, `test_publication.py`). |
| "Less manual configuration, controlled interpretation of requirements, and a repeatable process as additional portfolios are added." | Positioning | Outcome statement of the evidenced chain. **Deliberately softened:** no speed guarantee, no "instant onboarding", no claim that each portfolio is faster than the last — repeatability is the claim the repository supports (`synthetic_onboarding_pack/`, `simulation/` multi-client configs). |
| **Not claimed:** conversational onboarding | Excluded | The OCC Agent (`operations_control/occ_agent/`, 24 modules) is built and tested but not production-enabled — `docs/occ_agent/01_operating_process_implementation.md` §11 lists eleven preconditions and notes the live adapter has never been exercised. The page says "onboarding agent", true of the mapping agent that ships. |

---

## 5. Lenses — "One portfolio truth. Every relevant lens."

| Copy | Class | Evidence |
|---|---|---|
| "Any number of books — direct originations, acquired back books, sponsored securitisations — held in one governed model, each reportable on its own and in aggregate." | Evidenced (with one qualification) | `trakt_core/portfolio.py`: `PortfolioRegistry`, `resolve_scope()`, `resolve_capabilities()`, `ScopeCoverage`. Scope rendering: `frontend/mi-agent-ui/src/components/PortfolioContextSelector.tsx`; aggregation: `mi_agent_api/datasets.py`. **Qualification (unchanged from Pass 2):** the shipped type vocabulary is `direct`/`acquired`; a securitisation vehicle is a further governed type the model already holds (`PortfolioRegistry.types()`), and `spv` is a declared field role in `config/risk/concentration_test_library.yaml` with `spv_id` a monitored dimension in `config/mi/risk_monitor.yaml`. The claim is about the model, not a vehicle-specific feature. |
| "…capability and coverage disclosed per scope." | Evidenced | `ScopeCoverage` / `CapabilityState` disclosure in `trakt_core/portfolio.py`; `PortfolioScopeBanner.tsx`. |
| Lens preview: Consolidated platform + three governed books with balances | Evidenced | Rendered from `data/demo-pack.json` — the same books and figures as the hero preview and the interactive example, so the page carries one platform in one vocabulary. |

---

## 6. Portfolio intelligence (the example)

The only demo surface on the page, and the only place the synthetic-portfolio
disclaimer appears.

| Copy | Class | Evidence |
|---|---|---|
| "Ask portfolio questions in natural language — in the Trakt workspace, in Microsoft Teams, or through Microsoft 365 Copilot." | Evidenced | `mi_agent/mi_query_executor.py` + `mi_agent/interpreter/`; `deploy/copilot-agent/` (declarative agent, Teams app manifest) + `mi_agent_api/copilot_actions.py`; `mi_agent_api/teams_bot.py` (Bot Framework endpoint, JWKS-validated, fail-closed). |
| "Every answer comes from the deterministic engine, so the same question returns the same number in every channel — and Trakt declines what it cannot derive." | Evidenced | `mi_agent_api/mi_service.py`; `test_channel_parity.py`; refusal paths in `src/lib/intents.ts` and the pack's `unsupported` set. |
| "Approved risk findings can also be delivered proactively into Teams." | Evidenced | `trakt_notifications/` (19 modules: `cards.py`, `teams_client.py`, `outbox.py`, `delivery.py`, `trigger.py`, `recipients.py`); approval writes intent, a worker delivers, dedup/supersession by deterministic batch id; timer-driven outbox drain (`function_app.py:83`); tests under `tests/notifications/` incl. `test_end_to_end.py`; `docs/teams_proactive_notifications.md`. **Corrects Pass 2**, which asserted no bot, no cards, no proactive messaging. |
| Delivery strip: "Available today — Trakt workspace · Microsoft Teams & 365 Copilot · Managed service" | Evidenced | Workspace: `frontend/mi-agent-ui/src/components/`. Teams/Copilot: as above. Managed service: `apps/blob_trigger_app` (event-triggered pipeline), `mi_agent_pptx/cli.py`, `export_audit_pack.py`. The former delivery-model section, reduced to its substance; roadmap channels moved to §7. |
| "The portfolios are wholly synthetic, and the page accepts no uploads." | Demonstration | True by construction — no upload endpoint exists; the pack is the only data source. **Single instance on the page.** |
| Everything inside the interactive demo (answers, refusals, report previews, session counter) | Evidenced / Demonstration | Unchanged from Pass 2 — see the Pass-2 appendix below, which remains accurate for the demo surface. |

---

## 7. Governance and platform

| Copy | Class | Evidence |
|---|---|---|
| "AI in Trakt interprets, navigates and accelerates — it never writes your numbers." | Evidenced | LLM use is confined to natural-language → `MIQuerySpec` parsing (shown only the semantic catalogue, never raw data: `mi_agent/llm_query_parser.py`, `mi_agent/interpreter/`) and onboarding mapping suggestion/review (`engine/onboarding_agent/llm_*`). All calculation is deterministic (`mi_workflows/engine.py`: "no I/O, no LLM"); narratives are template-driven (`mi_agent_api/insight_generators.py`). |
| "Deterministic calculation — same question, same number, every channel — parity-tested." | Evidenced | `mi_agent_api/mi_service.py`, `test_channel_parity.py`. |
| "Reviewed configuration — approved by people before activation, and changes are governed." | Evidenced | `operations_control/engine.py` approval/publication; `apps/blob_trigger_app/approvals.py`; operator-approved `ActiveConfiguration` for risk limits; `tests/test_approval_policy.py`. |
| "Traceable outputs — lineage from source header to published figure, with validation evidence and reproducible runs." | Evidenced | `engine/gate_2_transform/lineage_tracker.py`, `exception_db.py` (hash-chained), `export_audit_pack.py`, `tests/test_repin_deterministic.py`. |
| "Client separation — isolated behind Microsoft Entra ID, with tenant authorisation enforced in the platform core." | Deployment + Evidenced | Entra: `mi_agent_api/copilot_auth.py` (defaults to `entra`, 503 unconfigured), `mi_agent_api/auth.py`. Core enforcement: `trakt_core/tenancy.py` — tenant from `ExecutionContext` only, never the request; `TENANT_MISMATCH` / `PORTFOLIO_NOT_AUTHORISED` before any read; `tests/test_governance_context_and_tenancy.py`, `tests/operations_control/test_tenancy.py`. **Wording note:** "controlled separation between organisations on a common platform" — never "multi-tenant SaaS". `config/tenancy.yaml` does not exist; production deployments are single-tenant per client by design. |
| "Built for specialist lending portfolios on a common canonical model with asset-specific configuration — designed so new lending asset classes are added through configuration and verified through the same pipeline, not by rebuilding the platform." | Evidenced (architecture), deliberately not a coverage claim | `docs/asset_class_hardening_framework.md` + `simulation/` (40 files): equity release, bridge and asset/equipment finance generated and driven through the real Gate 1 → MI → risk pathway, seeded determinism enforced in CI (`.github/workflows/hardening-smoke.yml`); key finding: no new canonical fields required (`config/system/fields_registry.yaml` `portfolio_type` + `--extra-aliases-dir` overlays). **The page claims the architecture, never the classes**: no non-equity-release production client exists and regulatory delivery remains two regimes. |
| Roadmap: enterprise agent deployment; agent-to-agent integration | Roadmap | `trakt_core/context.py` reserved channels `CHANNEL_ENTERPRISE_AGENT` / `CHANNEL_AGENT_TO_AGENT`; `docs/governed_capability_architecture.md` documents the adapter shape; only a test fixture exists. Outbound Teams delivery now exists (`trakt_notifications/`); agent callback does not. |

---

## 8. Reporting band

Deliberately a band, not a marquee: reporting is an output of the governed
layer, no longer the page's identity. Three former capability tiles live here.

| Copy | Class | Evidence |
|---|---|---|
| "Recurring packs and submissions are generated from the same governed layer — field-validated, submission-ready and traceable to source." | Evidenced | `mi_agent_pptx/` + `configs/pptx/investor_pack.yaml` + `mi_agent_api/decks.py`; regime projection and delivery `engine/gate_4_projection/`, `gate_4b_delivery/`, `gate_5_delivery/xml_builder_*` validated against committed XSDs; recurring production via `apps/blob_trigger_app` and CLI/API invocation. **Wording note:** *recurring*, with an evidenced timer trigger now in the repository (`function_app.py:83`) — the Pass-2 "no cron anywhere" rationale is retired, but scheduling of client reporting remains a deployment arrangement, so the page still does not promise "scheduled" as a platform feature. |
| Chips: Management reporting · Investor & funding-partner packs · Regulatory submissions · Bespoke analysis | Evidenced | As above, plus `analytics/mi_prep.py`, `analytics_lib/stratify.py`, `mi_agent_api/temporal_compare.py`. |
| **No regime names on the homepage** | Positioning | ESMA Annex 2 / Annex 12 are the delivered regimes but anchor an asset class and a jurisdiction; they belong on product pages. Gate 5 refuses undelivered annexes with a governed message rather than crashing. |

---

## 9. Contact and footer

| Copy | Class | Evidence |
|---|---|---|
| "See Trakt applied to your operating model" | Positioning | An offer to demonstrate, not a capability claim. |
| "We will demonstrate Trakt against your own portfolios, funding requirements and Microsoft 365 environment." | Positioning | Microsoft 365 integration is evidenced (`deploy/copilot-agent/`); funding-requirement interpretation is evidenced (`schedule8_extractor.py`). |
| Lead form fields and the no-marketing-list commitment | Deployment | Unchanged from Pass 2 — `src/lib/lead-validation.ts`, `src/lib/leads.ts`; see README § "Who owns incoming leads". |
| Footer: audience line and copyright | Positioning | The audience the brief specifies. |

---

## The green accent

`mint-400` / `#36c2a8`, taken verbatim from
`frontend/mi-agent-ui/src/index.css:21` — the MI Agent's UI green. In the
product it marks a proven state; it does the same job here and no other.

| Where | Why |
|---|---|
| The three trust proof points (tick marks; borders from `sm` upward) | Proven properties of the engine. |
| "Every figure is reconciled by construction rather than by comparison." | The page's strongest verification claim. |
| "Know what is breached today — and what the portfolio is moving toward." | The page's key forward-risk claim, backed by the three-state evaluation. |
| "Available today" and the three delivery chips it heads | A shipped/not-shipped signal. Roadmap stays muted grey. |
| "Trakt declines what it cannot derive." | A governance guarantee. |
| "Deterministic" in the hero preview footer | A confirmatory provenance signal. |
| "Pass" states inside the control preview | Product RAG semantics inside an interface depiction — see the amber note. |

The amber token remains "synthetic, or a boundary Trakt will not cross", with
one documented exception: inside interface previews that depict product UI,
the product's own RAG semantics apply (amber warning, rose projected breach).
The control preview is provenance-labelled "Illustrative" so neither meaning
leaks into page prose. See `app/globals.css`.

Contrast: 8.5:1 on `navy-950`, 8.0:1 on `navy-900` — WCAG AA/AAA. The chart
fill green `#2E7D5B` is still never used as type (3.8:1, fails AA).

---

## Excluded — considered and deliberately not said

| Not claimed | Why |
|---|---|
| SOC 2, ISO 27001, or any certification | Trakt does not hold them. |
| "GDPR compliant" | An organisational assessment, not a code property. |
| A hosting geography | Region is a deployment choice. |
| Penetration tested / independently assessed | No evidence of a completed test. |
| Regulatory approval, authorisation or endorsement | Trakt produces reporting *artefacts*; it holds no permission. |
| "Bank-grade", "military-grade", "zero-risk", "fully secure" | Unfalsifiable. Absent. |
| "AI-powered", "AI reads your contracts", "chat with your CSV", "upload your loan tape" | The AI that ships is confined to interpretation and mapping suggestion under review; document extraction is deterministic and text-based (PDF/DOCX is a placeholder). |
| "Instant onboarding", "automatic onboarding", per-portfolio speed escalation | The claim the repository supports is a governed, repeatable process — §4. |
| Conversational onboarding | Built (`operations_control/occ_agent/`) but not production-enabled; eleven documented preconditions stand. |
| "Multi-tenant SaaS" | Isolation is code-enforced, but deployments are single-tenant per client and `config/tenancy.yaml` does not exist. |
| Any named asset class as supported today (beyond the demo's synthetic scope), or "every lending asset class" | The hardening framework proves the architecture, not production coverage. |
| Regime/annex names on the homepage | They anchor an asset class and jurisdiction; product pages carry them. |
| Enterprise-agent or agent-to-agent availability | Reserved channels; labelled Roadmap in §7. |
| Uptime or SLA figures | None agreed. |
| Named clients, logos, testimonials, customer counts | None available. |
| "Real-time" | The pipeline is batch and snapshot-based. |
| Any figure not produced by the engine | Every number on the page comes from `data/demo-pack.json`, **except** the control preview in §3, which is labelled "Illustrative" on the page and classified Illustrative here. |

**Retired from this list (Pass 2 → Pass 3):** "no timer trigger or cron
anywhere in the repository" (false — `function_app.py:83`); "no Teams bot, no
adaptive cards, no proactive messaging" (false — `mi_agent_api/teams_bot.py`,
`trakt_notifications/`); "exactly three Copilot actions" (the plugin manifest
has since changed; the page no longer counts actions).

---

## Future capabilities — do not claim yet

| Capability | Status |
|---|---|
| Conversational onboarding (OCC Agent) | Built and tested; not production-enabled (`docs/occ_agent/01_operating_process_implementation.md` §11). |
| PDF/DOCX document interpretation | Explicit placeholder in `engine/onboarding_agent/document_extractor.py`. |
| Additional asset classes as production offerings | Architecture proven via `simulation/` + CI; no non-ERE production client; regulatory delivery remains Annex 2/12. |
| Scenario modelling / stress testing as a product surface | `mi_agent_api/scenario.py` is a small what-if; `analytics/scenario_engine.py` is not wired to the API. |
| LLM-generated commentary | Narratives are template-driven (`insight_generators.py`). |
| Public API access for clients | No public developer API is offered. |
| Multi-tenant operation of a single deployment | `config/tenancy.example.yaml` only; single-tenant is the current production shape. |

---

## Pass 2 appendix — the demo books, period-on-period and Annex exception reasoning

The following Pass-2 material remains accurate for the demonstration surface
and is retained verbatim as the reference for demo figures.

### The books

| Book | `source_portfolio_id` | Provenance | Balance sheet | Exposures | Balance |
|---|---|---|---|---:|---:|
| ALP Origination Book | `alp_origination` | direct | warehoused, destined for SPV2 | 47 | £15,432,544 |
| ALP Acquired Back Book | `alp_acquired` | acquired | warehoused, destined for SPV2 | 37 | £11,974,544 |
| SPV1 Sponsored Securitisation | `spv1_sponsored` | direct | **sold and derecognised**; servicing, risk retention and investor reporting retained | 34 | £9,862,973 |
| **Platform total** (warehoused) | — | — | on balance sheet | **84** | **£27,407,089** |
| **Sponsor total** (incl. SPV1) | — | — | everything the sponsor reports on | **118** | **£37,270,061** |

Sum of books reconciles to the sponsor total to the penny at both reporting
dates. Two totals are carried deliberately: a book the sponsor originated,
securitised and sold is off balance sheet and still carries reporting
obligations.

| Claim | Class | Evidence |
|---|---|---|
| Book identity is a first-class attribute, not a query filter | Evidenced | Stamped at Gate 2 (`engine/provenance.py`); resolved by `trakt_core/portfolio.py`; `synthetic_demo/assemble_multibook.py` refuses rows that cannot name their book. |
| "Show the funded balance by book" | Evidenced | `_balance_by_book` in `scripts/build_demo_pack.py`. |
| Period-on-period movement (31 May → 30 June 2026) | Evidenced | `_period_movement`; prior snapshot SHA-256 asserted before comparison. |
| Annex 2 exception reasoning (three seeded exceptions, reconciled dispositions) | Evidenced | `engine/gate_4b_delivery/annex2_exception_reconciler.py`; the seeded `BLOCKS_DELIVERY` / `DEFAULTED_AT_DELIVERY` / `OUT_OF_REGIME_SCOPE` cases as documented in Pass 2. |
| Session question counter, refusal explanations, report previews | Evidenced / Demonstration | Unchanged — `CopilotDemo.tsx`, `src/lib/intents.ts`, `mi_agent_api/` modules as recorded in Pass 2. |
