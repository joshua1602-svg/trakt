# Trakt Agentic Architecture — Three-State Readiness Review and Gap Analysis

**Scope:** the active agentic pipeline only (blob-triggered orchestration, engine `*_agent` modules, MI Agent, MI API, React dashboard, deck generation, operator console). The legacy pipeline is assessed only for accidental dependencies, duplication, and migration risk.

**Method:** full runtime-path tracing from source-file arrival to output retrieval — entry points, imports, subprocess calls, blob layouts, HTTP routes, auth, CI/CD, and tests — reconciled against `docs/` and `due_diligence/`. Every material claim cites a file (and where useful a line). Findings are labelled **[Confirmed]** (read in code), **[Inferred]** (reasoned, basis stated), or **[Unknown]** (could not be determined from the repository).

**Repository state reviewed:** branch `claude/trakt-agentic-gap-analysis-kaioy8`, HEAD `5aa9805`, July 2026.

**This document contains no code changes and recommends none be made as part of this review.**

---

## 1. Executive conclusion

**Is the current agentic architecture fundamentally suitable for the three-state model?**
Yes, at its core — with one large caveat. The single most valuable property the three-state model needs is already real and enforced in code: **reasoning is separated from deterministic execution**. The LLM never computes a number, never sees loan data, and only proposes a declarative `MIQuerySpec` that a deterministic pandas executor runs (`mi_agent/mi_agent_workflow.py:9-12`, `mi_agent/mi_query_spec.py:9-11`, `mi_agent/mi_query_executor.py:9-16`). The pipeline side is equally disciplined: LLM use is advisory-only, budget-capped, human-gated, and off by default (`apps/blob_trigger_app/llm_recommendations.py:39-77`, `engine/onboarding_agent/llm_policy.py:29-48`). The caveat: everything *around* that core — identity, tenancy, artefact governance, API contracts, ingestion integrity — is built for a single-client, operator-driven managed service (State 1) and does not yet form a boundary an external enterprise agent could safely cross.

**Is Microsoft Copilot primarily an interface extension, or does the architecture require material restructuring?**
It is an interface extension *to the capability layer*, but a material build-out *of the boundary layer*. The deterministic query/calculation services (`mi_agent_api/{evolution,cohorts,forecast_bridge,risk_limits,temporal_compare,scenario,geo}.py`) are exactly the shared services the three-state principle demands, and the deck generator already reuses them in-process (`mi_agent_pptx/mi_api.py:1-8`). What is missing is not capability but contract: no validated identity (the API trusts an unverified `X-MS-CLIENT-PRINCIPAL` header — `mi_agent_api/auth.py:6-11`), no request-scoped tenancy (`auth.py:27-30`), no typed response schemas (no `response_model` anywhere in `mi_agent_api/app.py`), no async job model, no approval state on any output, and zero Copilot/OpenAPI-plugin/MCP surface anywhere in the repo **[Confirmed — repo-wide grep]**.

**Can the same MI Agent capability be reused?**
Yes. The parser → spec → executor → typed-artifact chain is transport-agnostic and reusable behind a new action facade. Two things block direct reuse today: the analyst-quality narrative lives in the React client, not the API (`frontend/mi-agent-ui/src/lib/responsePresenter.ts`, `insights.ts` — the API's `answer` field is a deliberate placeholder, `mi_agent_api/adapters.py:489-502`), and conversation state/follow-up resolution is client-side only (`frontend/mi-agent-ui/src/lib/analysisContext.ts`, history in browser `localStorage`).

**Can all approved artefacts be safely exposed?**
No — because "approved" does not exist as a state for any output artefact. Approval machinery is input-side only (source registration and mapping approval — `apps/blob_trigger_app/approvals.py:29-35`). Published canonicals, decks, and regime outputs carry no approval status, no run ID, no content hash, and live behind mutable `latest/` pointers that are overwritten on every run (`apps/blob_trigger_app/persistence.py:301-325`, `pptx_stage.py:151-200`). "Give me the latest approved investor deck" is unanswerable today **[Confirmed]**.

**Can SFTP feeds be automated into the current pipeline?**
Architecturally yes, cleanly: the pipeline's entry contract is "files in blob + `_READY.json` marker" (`apps/blob_trigger_app/router.py:73-76`), so an SFTP landing service that copies to `raw-v2` and writes the marker would need **zero changes** downstream. But there is no SFTP code anywhere (zero references repo-wide **[Confirmed]**), and the ingestion layer currently lacks checksums, quarantine, dead-lettering, replay protection, and alerting — controls SFTP-grade clients will expect.

**Overall classification: a moderate refactor.**
Not incremental — identity, artefact governance, and the action-contract layer are genuinely missing, and the dashboard's own auth front door is currently removed. Not a re-architecture — the capability layer, deterministic core, orchestration spine, and governance instincts (fail-closed routing, human-gated LLM, provenance stamping) are the right foundations and should be preserved.

---

## 2. Current-state architecture (as implemented)

### 2.1 Component classification

| Component | Classification | Evidence |
|---|---|---|
| Root `function_app.py` (Event Grid trigger) | **Active agentic** — the deployed entry point | `function_app.py:87-96`; deployed by `.github/workflows/main_trakt.yml:36-46` |
| `apps/blob_trigger_app/` (router, approvals, persistence, ops) | **Active agentic** | invoked from root trigger; `router.py:124-146` |
| `engine/orchestrator_agent/`, `engine/onboarding_agent/`, `engine/transformation_agent/`, `engine/validation_agent/` | **Active agentic** | invoked via `orchestrator_invoke.py:223-266` |
| `engine/assembler_agent.py` | **Active agentic** (platform canonical + regime command builder) | `assembler_refresh.py:75-76`, `regime_runner.py:32-38` |
| `engine/gate_1_alignment/semantic_alignment.py` | **Shared kernel** (not legacy) — 17 agentic import sites | `engine/onboarding_agent/semantic_alignment_adapter.py:41` et al. |
| `engine/gate_2_transform`, `gate_3_validation` primitives | **Shared** — cherry-picked via adapters | `transformation_agent/gate2_adapter.py:35`, `validation_agent/rules_adapter.py:31` |
| `engine/gate_4_projection/regime_projector.py` | **Legacy, still on the active path** — subprocessed by the agentic orchestrator | `engine/orchestrator_agent/adapters.py:139-153`, `engine/assembler_agent.py:44` |
| `engine/projection_agent/`, `engine/delivery_xml_agent/` | **Built, dormant** — zero production callers | grep; `delivery_xml_agent/__init__.py:19` |
| `engine/gate_4b_delivery`, `gate_5_delivery` | **Legacy, dormant** | no production callers |
| `engine/orchestrator/trakt_run.py` | **Legacy, orphaned** — README still leads with it | zero non-doc imports |
| `engine/enum_agent/` | **Uncertain** — reached only *through* the legacy Gate 4 subprocess | `gate_4_projection/regime_projector.py:43,48` |
| Root `agents/` package (onboarding v1) | **Legacy** — used only by `ui/onboarding_review.py` and `cli/` | `agents/onboarding_agent.py:1-5` |
| `mi_agent/` (workflow, parser, executor) | **Active agentic** | `mi_agent_api/app.py:1623` |
| `mi_agent/interpreter/`, `mi_agent/mi_runtime.py`, `mi_agent/semantic_resolver.py` | **Built, dormant** — governance-superior parallel stack, unwired | grep: only scripts/tests import them |
| `mi_agent_api/` (FastAPI) | **Active agentic** | deployed as App Service `trakt-mi-api` |
| `frontend/mi-agent-ui/` (React SWA) | **Active agentic** | deployed on every `main` push |
| `mi_agent_pptx/` (`cli.py` + `deck.py` path) | **Active agentic** | invoked in-process by `pptx_stage.py:102-109` |
| `mi_agent_pptx/pptx_builder.py` stack | **Dead** — imported by nothing, but still tested | grep; `tests/mi_agent_pptx/` |
| `mi_agent_operator/` (approval console) | **Active agentic, undeployed** — no CI/CD or IaC artefact exists | grep across `.github/`, `deploy/` |
| `mi_agent/streamlit_mi_agent.py` | **Superseded** thin UI over the same workflow | `docs/phase6e...md:242` |
| `analytics/` (Streamlit ERM dashboard) | **Legacy, still actively deployed** with the repo's best CI | `.github/workflows/deploy_streamlit_dashboard.yml` |
| `snapshot/` (SnapshotStore) | **Built, dormant** — "not wired into the MI Agent runtime yet" | `snapshot/__init__.py:11` |
| Root `exception_db.py` / `exception_queue.py` / `export_audit_pack.py` | **Legacy, disconnected** (hash-chained audit engine, unused) | zero agentic imports |
| `config/system/config_resolver.py` | **Built, dormant** — layered multi-tenant config merge, "not wired" | `config_resolver.py:1-5` |

### 2.2 Textual architecture diagram (current, as-deployed)

```
                       ┌──────────────── STATE-1 OPERATORS ────────────────┐
                       │  ops CLI (approve/promote/rerun/backfill/repin)   │
                       │  operator console (FastAPI + shared token,       │
                       │                    built but NOT deployed)        │
                       └───────────────────────┬───────────────────────────┘
                                               │ approvals / registry writes
 upload pack + _READY.json                     ▼
 ────────────────────────►  Azure Blob "raw-v2"
                                │ Event Grid (BlobCreated)
                                ▼
                     root function_app.py  (the ONLY trigger; no HTTP endpoints)
                                ▼
                 apps/blob_trigger_app/router.py
                 • path parse {client}/{book}/{dataset}/{freq}/{pid}/{period}
                 • schema fingerprint vs source registry (trakt-state/registry)
                 • decisions: deterministic | source_onboarding | drift-halt |
                   auto-approve (materiality policy) | pending_review
                                ▼
                 engine/orchestrator_agent.run_orchestration   (in-process)
                   onboard → [transform] → [validate] → stamp(provenance)
                                ▼
                 engine/assembler_agent  → platform canonical
                                ▼                         ▼
                 subprocess: LEGACY gate_4_projection     mi_agent_pptx.cli
                 regime_projector.py (Annex 2 CSV,        (investor deck,
                 NO XML in production)                    in-process)
                                ▼                         ▼
        Blob "trakt-state"                    Blob "processed-v2"
        events/ runs/ approvals/              accepted/ platform/{latest,period}/
        gates/ llm/ governance/ registry/     regime/ pipeline/ decks/{latest,YYYY-MM}/
                                                        │
                    ┌───────────────────────────────────┴─────────────┐
                    ▼                                                 ▼
        mi_agent_api (FastAPI, App Service)                 analytics/ Streamlit
        24 routes; Easy-Auth header trust;                  (LEGACY dashboard,
        POST /mi/query → chat_routing (deterministic        separate deployment)
        intents) | llm_query_parser (Haiku, spec-only)
        → mi_query_executor (pandas) → typed artifacts
                    ▲
                    │ fetch (currently cross-origin, SWA auth REMOVED)
        frontend/mi-agent-ui (React SWA)
        narrative/insights/follow-ups computed CLIENT-SIDE
```

### 2.3 Key facts about the deployed system

- **Entry points.** Exactly one production trigger (Event Grid blob-created → root `function_app.py:87-96`). There are **no HTTP triggers in the Function App at all** [Confirmed]. Run initiation is therefore file-arrival or operator CLI (`ops rerun`, `backfill`) — there is no "start a run" API.
- **Orchestration.** The whole pipeline (onboard → transform → validate → stamp → assemble → project → deck) executes synchronously inside one Event Grid delivery under a 30-minute function timeout (`host.json:3`), with a no-timeout `subprocess.run` for projection (`regime_runner.py:38`). No Durable Functions, no queues, no job model [Confirmed].
- **Approvals.** Real, but input-side only: `new_source` and `schema_drift` kinds, `pending/approved/promoted/rejected` states (`approvals.py:29-35`), fail-closed routing on drift, human promote via ops CLI or the (undeployed) operator console, plus an auto-approval materiality policy for non-material recurring drift (`approval_policy.py:151-256`) with a governance artefact.
- **MI query flow.** `POST /mi/query` → deterministic intent router first (`chat_routing.py:1236-1313`, no LLM) → otherwise deterministic parse with zero-cost-first LLM fallback (`llm_query_parser.py:2360-2538`, model `claude-haiku-4-5`, spec-only, repair loop, deterministic safety net) → pandas execution → typed artifact envelope (`adapters.py:660-704`).
- **Frontend coupling.** The API deliberately returns a placeholder sentence; the React client generates the narrative, insights, drill-through, and follow-up rewriting (`responsePresenter.ts`, `insights.ts`, `analysisContext.ts`). Chat history is browser `localStorage` (`state/persistence.ts:8`).
- **Infrastructure.** Four Azure services deployed by four different mechanisms; no IaC; no environment separation; no Key Vault (plaintext connection strings in app settings); **zero tests executed in CI** despite ~193 Python + 49 TS test files [Confirmed — grep of `.github/`].

---

## 3. Current end-to-end workflows (as traced)

**Source-file ingestion [Confirmed].** Client/operator uploads a pack folder to `raw-v2/{client}/{book_type}/{dataset}/{freq}/{pid}/{period}/`, data files first, `_READY.json` marker last (`router.py:73-76`). Event Grid fires on every blob; non-marker files are acknowledged and logged; the marker triggers pack download, role classification (`file_roles.py`), schema fingerprinting (`schema_fingerprint.py:126-174`, structure-only), and registry lookup. Decisions: known source + matching fingerprint → deterministic run; new source → onboarding run ending `pending_review`; drift → halt or auto-approve per materiality policy (`router.py:358-382`). Optional `expected_files` list in the marker is verified fail-closed (`router.py:259-270`); there are **no checksums, no quarantine, no dead-letter, no alerting** [Confirmed].

**Canonicalisation [Confirmed].** `run_orchestration` per portfolio: onboarding agent (deterministic mapping kernel = Gate 1 `semantic_alignment.py`, LLM advisory optional) → transformation agent (handoff-contract-driven, reusing frozen Gate 2 primitives) → provenance stamping (`engine/provenance.py`, row-level, fail-closed). Central/platform canonical assembled by `engine/assembler_agent.py`, published to `processed-v2/platform/{client}/latest/` + `/{period}/`.

**Mapping [Confirmed].** Mapping artefacts (`05_*`, `30_*–37_*` series incl. `33_mapping_review_queue`, `34_target_first_decisions`) produced by the onboarding agent; LLM recommendations advisory-only, approval-required (`llm_recommendations.py:220-221`); confirmed mappings promoted into the source registry with a bumped `mapping_version` (`approvals.py:160-181`).

**Validation [Confirmed].** Validation agent produces the `40_*–46_*` package (results, readiness, issues, lineage, blocker diagnostics — `validation_agent.py:587-690`). **Coverage caveat:** the agentic business-rule set is 7 hand-coded checks vs 48 declarative rules in legacy Gate 3 (`rules_adapter.py:372-437` vs `gate_3_validation/validate_business_rules.py:204-720`) [Confirmed].

**MI queries [Confirmed].** As §2.3. Run context is `portfolioId = "{client_id}/{run_id}"` split by hand (`app.py:1388-1390` and ~9 other places); no-`run_id` queries hit the *active/latest* dataset. There is no "latest approved run" concept anywhere in the MI path [Confirmed].

**Investor-deck generation [Confirmed].** Final stage of every successful orchestrated run (`orchestrator_invoke.py:288-296` → `pptx_stage.py` → `mi_agent_pptx/cli.run`). Numbers are taken verbatim from the same compute functions the dashboard uses, in-process, no LLM (`mi_api.py:1-8`, `deck.py:1-8`). Published to `decks/{client}/latest/investor_pack.pptx` + `/{YYYY-MM}/`, pointer blob `latest_investor_pack.json`. **No approval gate; publish failure silently swallowed** (`pptx_stage.py:198-200`); no on-demand generation endpoint exists.

**ESMA Annex 2 XML generation [Confirmed — deliberately not live].** Production produces regime *CSV* via the legacy Gate 4 projector subprocess. The delivery/XML agent is a readiness stage that hard-codes `xml_generated: false` / `ready_for_xml_delivery: false` (`delivery_xml_agent.py:787-804, 856-857`); all four preview modes are disabled in `config/delivery/xml_preview_policy.yaml`; the XSD path map has 11/107 fields sample-confirmed and `production_ready: false` for all 107 [Confirmed]. Docs and code agree here (one stale count: `annex2_production_xml_structure_contract.md:41` says 8 vs actual 11).

**Artefact storage [Confirmed].** All writes are `upload_blob(overwrite=True)` (`storage.py:181-193`). Dated cuts are *asserted* immutable (`platform_snapshots_blob.py:37`) but nothing enforces it. No artefact registry, no content hashes, no SAS/signed URLs anywhere (grep: zero `generate_blob_sas` hits).

**Approval [Confirmed].** Covers source onboarding and schema drift only. Enforced by routing (drift never invokes the orchestrator), promote-only-if-approved (`approvals.py:152-154`). Weaknesses: free-text `decided_by`, single shared operator token, `edit` permitted on already-approved artefacts (`ops.py:222-237`), `--force-publish` break-glass with no second approver.

**Retrieval [Confirmed].** Dashboard/API only. Decks discovered by path convention and streamed as bytes through `GET /mi/decks/download` (`app.py:1250-1271`); canonicals resolved by blob path regex (`platform_snapshots_blob.py:33-34`). No external retrieval contract, no expiring links, no artefact-level authorisation.

**Workflows that could not be confirmed [Unknown]:** the live values of `MI_AGENT_AUTH_ENABLED` and `ANTHROPIC_API_KEY` on the deployed App Service (not in repo; the deploy examples omit both — `deploy/trakt-mi-api/app_settings.example.json`); whether the deployed dashboard currently functions at all given the removed SWA auth + cross-origin API URL (see §9); whether `config/client/mappings/` exists blob-side (referenced by `repin.py:24`, absent in repo).

---

## 4. Three-state readiness assessment

### State 1 — Trakt-managed interface: **Mostly ready**

- **Supported [Confirmed]:** end-to-end blob-native automated runs; fail-closed routing with human approval for new sources/material drift; deterministic MI chatbot + React dashboard; automatic investor deck per run; mapping/validation report packages; run ledger, gate diagnostics, `next_action` operator advice; backfill and repin tooling.
- **Partially supported:** approvals (input-side only, shared-token console, undeployed); exception review (validation issues exist in `43_validation_issues` and run records, but the hash-chained exception queue is disconnected legacy); ESMA XML (readiness machinery exists, production XML deliberately deferred).
- **Missing:** artefact approval/release states; deterministic artefact regeneration by run ID; alerting (discovery is pull-based via `ops list-halted`); operator identity (shared secret, no four-eyes).
- **Blockers (operational):** zero tests in CI; `ops rerun` broken under Azure (writes scratch to read-only `cwd`, depends on vanished `/tmp` pack dir — `ops.py:255,361-363`) [Confirmed]; deck publish failures silent.
- **Governance risks:** approved approvals editable post-approval; auto-approval governance artefact overwritten per pack_key (`layout.py:91-94`); `decided_by` is unverified free text.

### State 2 — Microsoft Copilot-enabled interface: **Partially ready**

- **Supported (as underlying capability) [Confirmed]:** the full deterministic MI question set; portfolio/run-scoped snapshots; period comparison; risk limits; deck discovery + download; dataset contract metadata (`mi_dataset_contract.py`).
- **Partially supported:** structured responses (typed artifacts exist, but no OpenAPI response schemas and a placeholder `answer`); provenance in answers (`sourceNotes`, `queryTrace`, reconciliation exist in the envelope — good raw material for citations).
- **Missing [all Confirmed]:** any Copilot/plugin/MCP/manifest surface (zero references repo-wide); validated-token authentication (header trust only); request-scoped tenancy/RBAC (`portfolioId` never checked against the principal — `app.py:1551`); server-side conversation state; approval-aware retrieval ("latest **approved** X" unanswerable); on-demand artefact generation endpoints; async job/polling model; artefact hand-off primitives (metadata, MIME, expiring links) for SharePoint/Teams/Outlook actions.
- **Architectural blockers:** none fundamental — the gaps are boundary-layer. **Security blockers:** exposing the current API to any external caller without real token validation would allow principal forgery via a client-supplied `X-MS-CLIENT-PRINCIPAL` header if the platform proxy is not in front [Confirmed mechanism; deployment posture Unknown].

### State 3 — Fully embedded architecture: **Partially ready** (ingestion) / **Not ready** (exposure)

- **Supported [Confirmed]:** event-driven file delivery into blob is a proven, working machine-to-machine entry; auto-detection of new data; automated canonicalise → validate → calculate → deck; pack-level idempotency with fingerprint match; period-scoped output history.
- **Partially supported:** raw landing (raw container exists but is not immutable — no WORM/versioning/legal hold); completeness validation (`expected_files` optional, no checksums); exception routing (pending_review states exist; no notification/alerting).
- **Missing [Confirmed]:** SFTP (push or pull) entirely; file authenticity controls (checksums, manifests with hashes, signatures, encryption handling); replay protection at event level (`event_id` is time-salted — a redelivery of a non-terminal pack re-executes, `event_log.py:11-13`, `router.py:280-292`); quarantine/dead-letter; downstream APIs for machine consumers; multi-tenant identity for service accounts; approval-gated exposure of outputs.
- **Blockers:** the 30-minute synchronous execution ceiling; single-YAML source registry with read-modify-write and no ETag (concurrent-write clobber risk — `source_registry.py:112-147`) [Inferred, high confidence]; hard-coded ERM-UK client config in every agentic regime projection (`engine/assembler_agent.py:107`) — a genuine cross-tenant defect once a second client exists [Confirmed].

---

## 5. Copilot-specific gap analysis

Effort categories: Small / Medium / Large / Unknown (no time estimates).

| Required capability | Current implementation | Evidence in repository | Gap | Severity | Recommended architectural response | Dependencies | Effort |
|---|---|---|---|---|---|---|---|
| Get latest approved run | "Latest" = mutable blob pointer; "approved" absent | `platform_snapshots_blob.py:6-8,31-37`; `persistence.py:313-325` | No run approval state; no immutable run identity exposed | **Critical** | Introduce run/artefact registry with lifecycle states; expose `GET /v1/runs/latest?status=approved` | Artefact registry, approval service | Medium |
| Get portfolio summary | `GET /mi/snapshot` deterministic KPIs + MoM | `app.py:526` | Untyped response; no approval scoping; single-tenant | Medium | Wrap behind typed, versioned action endpoint | Response schemas, tenancy | Small |
| Ask MI question | `POST /mi/query` — deterministic-first, LLM spec-only, typed artifacts | `app.py:1549`, `chat_routing.py:1236`, `llm_query_parser.py:2360` | Placeholder narrative; no server-side context; no response schema | High | Move narrative + follow-up resolution server-side; add conversation context parameter; publish schema | Narrative service (port of `responsePresenter`/`insights`), schemas | Medium |
| Compare reporting periods | `GET /mi/evolution/compare` + `temporal_compare.py` | `app.py:1353` | Contract/typing only | Low | Same facade treatment | Schemas | Small |
| List validation exceptions | `43_validation_issues.*` per run; surfaced in run records | `validation_agent.py:599-609`, `run_records.py:58` | No API endpoint; no queryable exception store; hash-chained exception engine disconnected | High | Exception query service over validation issues keyed by run ID | Run registry | Medium |
| List eligibility exceptions | Not implemented as a distinct concept | — | Missing entirely | Medium | Model as rule-tagged subset of validation/business rules | Rule versioning, validation coverage | Medium |
| Retrieve approved artefact | Deck/canonical retrieval by path convention, bytes-proxy | `decks.py:117-194`, `app.py:1250-1271` | No approval filter; no artefact IDs; no metadata | **Critical** | Artefact registry + `GET /v1/artifacts/{id}` honouring approval state | Registry, approval service | Medium |
| Generate approved artefact from latest approved run | Deck generated only as pipeline side-effect; no on-demand generation | `pptx_stage.py:102-109`; no `POST /mi/decks/generate` | No generation API; no async job model | High | Async job endpoints (`POST /v1/jobs/deck` → job ID → poll → artefact ID) reusing `mi_agent_pptx.cli` | Job model, registry | Large |
| Export canonical loan tape | Blob path retrieval only; no endpoint | `platform_snapshots_blob.py` | No governed export action | High | `GET /v1/runs/{id}/canonical` with approval check + link | Registry, secure retrieval | Medium |
| Generate/retrieve ESMA Annex 2 XML | Production XML deliberately not generated (11/107 paths confirmed) | `delivery_xml_agent.py:787-804`; path map `production_ready: false` ×107 | Capability itself incomplete — not a Copilot gap per se | High (product) | Complete XSD path map + promote delivery_xml_agent to producer; only then expose | XML v2 completion | Large |
| Retrieve mapping report | Produced per run (`05_*`, `33_*` etc.), persisted partially to gate artefacts | `onboarding_orchestrator.py:659`, `persistence.py:118-149` | No retrieval endpoint; inconsistent durable persistence | Medium | Register reports as artefacts; expose retrieval action | Registry | Small–Medium |
| Retrieve validation report | `40–46` package on disk; durable only via generic gate loop | `validation_agent.py:587-690` | Same as above | Medium | Same | Registry | Small–Medium |
| Retrieve provenance package | Row-level provenance strong; run-level evidence scattered (events, gates, governance, approvals) | `engine/provenance.py:47-57`; `layout.py:54-94` | No assembled evidence package; artefact-level provenance absent | High | Provenance/evidence assembly service keyed by run ID | Registry, immutable event log | Medium |
| Prepare artefact for distribution (SharePoint/Teams/Outlook) | Nothing — no SAS, no expiry, no metadata, no MIME contract beyond deck download | grep: zero SAS usage | Missing entirely | High | Short-lived signed URL issuance + artefact metadata (hash, MIME, approval, run ID, expiry, correlation ID) | Registry, identity | Medium |
| Identity: Entra ID / OAuth / OBO | Easy-Auth header trust, no token validation; SWA auth config **removed** (commits `5b0cc2f`, `8d224b1`) | `auth.py:6-11`; `staticwebapp.config.json:1-15` vs `docs/auth_setup_runbook.md:125-127` | No verifiable identity for any external caller | **Critical** | Real JWT validation (Entra, JWKS, aud/iss), OBO support for Copilot; restore front-door auth | Entra app registration; SWA Standard tier | Medium |
| Tenant/portfolio/role authorisation | Deployment-per-tenant; 2 roles; `portfolioId` unchecked | `auth.py:27-30,86-113`; `app.py:1551` | No request-scoped authorisation | **Critical** | Principal→client/portfolio entitlement check on every action | Identity | Medium |
| OpenAPI/tool-contract suitability | Auto-generated spec with empty response bodies; `/docs` public; no versioning | no `response_model` in `app.py`; `auth.py:58-59` | Spec unusable for declarative agents; no `/v1` | High | Pydantic response models, versioned action routes, close `/docs` or gate it | — | Medium |
| Async jobs, polling, idempotency | All-sync handlers, 120s gunicorn timeout; no job store | `app.py` (2 async defs, both infra) | Long operations can't be exposed | High | Job resource + idempotency keys for generation actions | Storage for job state | Medium–Large |
| Audit of external access | None (no per-request audit trail tied to identity/artefact) | — | Missing | High | Access audit log with correlation IDs | Identity, registry | Medium |

---

## 6. MI Agent reuse assessment

**Reusable unchanged [Confirmed]:**
- The deterministic execution core: `mi_query_executor.py`, `mi_query_spec.py`, `mi_query_validator.py`, `mi_spec_validation.py`, quantile buckets, chart factory data-shaping.
- The deterministic intent router and service modules: `chat_routing.py`, `evolution.py`, `temporal_compare.py`, `cohorts.py`, `forecast_bridge.py`, `risk_limits.py`, `scenario.py`, `geo.py` — these *are* the shared capability layer, already consumed by three interfaces (React via API, deck via in-process `mi_api.py`, Streamlit via workflow).
- The parser economics: zero-cost-first, LLM repair loop, deterministic safety net (`llm_query_parser.py:2407-2538`).
- The dataset contract (`mi_dataset_contract.py`) — excellent grounding metadata; relocate off `/health`.

**Reusable behind a new API facade:**
- `run_mi_agent_query` (`mi_agent_workflow.py:160`) as the single NL entry, wrapped in a versioned, schema-typed action endpoint with conversation context input.
- Deck discovery/resolution (`decks.py`) behind an artefact-registry-aware retrieval action.

**Needs separating from the frontend [Confirmed]:**
- Narrative generation (`responsePresenter.ts`), insight derivation (`insights.ts`), and follow-up resolution (`analysisContext.ts`) must be ported server-side (or re-implemented as a shared service) — otherwise Copilot receives data plus the placeholder sentence (`adapters.py:489-502`). This is the single largest MI-reuse work item.
- Conversation state: move from `localStorage` to a server-side (or caller-supplied, contract-defined) context so follow-ups work from Copilot, Teams, and the dashboard identically.

**Business logic that must move into shared deterministic services:**
- Little on the numbers side — the separation is already good. The exceptions: measure-format/catalogue knowledge duplicated in TypeScript (`data/catalog.ts`, `MEASURE_FORMAT`), and chart semantics forked between `mi_chart_factory` (Plotly/Streamlit) and `adapters.py` (Recharts/React). Consolidate on one server-side presentation contract.

**Should not be exposed to Copilot:**
- Operator/approval operations (ops CLI, operator console actions), `force_publish`, rerun/backfill/repin, raw run diagnostics, `/health` internals, and anything reading unapproved or halted-run data. Scenario overrides (`scenario.py`) are safe (pure, side-effect-free) but should be labelled as hypothetical, not reported figures.

**Copilot → MI Agent directly, or both → a lower capability layer?**
Both should call the same governed action layer that fronts `run_mi_agent_query` and the deterministic services — i.e. **Copilot should not get a private path, and it should not call the current `/mi/query` as-is**. Concretely:

- *Can Microsoft Copilot call the existing MI Agent API as it stands?* **No.** (1) Authentication: the API validates no tokens; it trusts a platform-injected header a Copilot action cannot legitimately supply (`auth.py:6-11`). (2) No usable response schemas for a declarative-agent manifest. (3) No tenancy/authorisation on `portfolioId`. (4) No approval semantics, so it could serve unapproved/stale data with no way to tell. (5) The narrative the user would expect lives client-side.
- *Which components are reusable?* Parser, spec, executor, intent router, all deterministic service modules, dataset contract, deck resolution.
- *Which require decoupling?* Narrative/insights/follow-ups (from React), chat state (from `localStorage`), dataset/run selection (from env-var deployment identity to request-scoped tenancy).
- *Which stay Trakt-exclusive?* Operator console, approvals/promotion, drill-through workspace UX, remediation and mapping review — matching the stated dashboard-as-specialist-workspace model.
- A note on the dormant stack: `mi_agent/interpreter/` + `mi_runtime.py` are a governance-superior parallel NL boundary (code-marker rejection, hallucinated-field checks, forced clarification) that production bypasses [Confirmed]. Either wire them in or retire them — carrying two NL→spec stacks is drift risk.

---

## 7. Artefact readiness matrix

Legend: ✔ yes · ✖ no · ◐ partial. "Copilot-ready" applies the full-path standard (invocation → authorisation → deterministic execution → persistence → approval → retrieval → audit).

| Artefact | Generated today | Deterministic | Linked to run ID | Versioned | Approval state | Stored | Programmatically retrievable | Externally safe | Copilot-ready | Principal gaps |
|---|---|---|---|---|---|---|---|---|---|---|
| Canonical loan tape | ✔ every run | ✔ | ◐ (path convention `{client}/{period}`; no artefact ID; `RunState.central_canonical_path` local-only) | ◐ (dated cuts; `latest` mutable; overwrite-enforced nothing) | ✖ | ✔ blob `processed-v2` | ◐ (blob path regex; no endpoint) | ✖ | ✖ | No approval state; no immutable identity; no hash |
| Mapping report | ✔ (onboarding `05_*`,`30–37_*`) | ✔ (LLM parts advisory + human-gated) | ◐ (run-local; durable subset by `pack_key`) | ✖ | ◐ (mapping decisions approved; report artefact itself unversioned/unapproved) | ◐ local + partial blob | ✖ no endpoint | ✖ | ✖ | Inconsistent durable persistence; no retrieval API |
| Validation report | ✔ (`40–46` package) | ✔ | ◐ (run-local; blob only via generic gate loop) | ✖ (no rule versioning anywhere in `engine/` [Confirmed]) | ✖ | ◐ | ✖ | ✖ | ✖ | Rule-set unversioned; 7-vs-48 rule coverage regression; no API |
| ESMA Annex 2 XML | ✖ (deliberately: `xml_generated: false`; previews disabled) | ✔ (when built) | ◐ | ✖ | ◐ (elaborate readiness gates exist — the best-governed artefact, and the only ungenerated one) | n/a | ✖ | ✖ | ✖ | 96/107 XSD paths unconfirmed; two dormant builders + one legacy builder |
| Investor deck | ✔ every successful run | ✔ (no LLM in live path) | ✖ (pointer has no run ID; `run_state.json` artefact record self-erasing — `pptx_stage.py:115-117` vs `state.py:134-169`) | ✖ (`latest` overwrite; `YYYY-MM` key → intra-month runs destroy history) | ✖ (ships ungated) | ✔ blob `decks/` | ✔ (`GET /mi/decks`, `/mi/decks/download`) | ✖ | ✖ | No approval, no run linkage, silent publish failure, cross-client pipeline-source fallback (`mi_agent_pptx/cli.py:222`) |
| Exception report | ◐ (`43_validation_issues` + run-record diagnostics; no client-facing artefact) | ✔ | ◐ | ✖ | ✖ | ◐ | ✖ | ✖ | ✖ | Hash-chained exception engine (`exception_db.py`) exists but disconnected; no queryable store |
| Governance evidence package | ◐ (scattered: approvals JSON, auto-approval artefact, agent-session records, event manifests, gate diagnostics) | ✔ | ◐ (by `pack_key`, not run ID) | ✖ (auto-approval artefact overwritten per pack_key) | n/a | ◐ (some local-only, e.g. `GovernanceLogger`) | ✖ | ✖ | ✖ | No assembled evidence package; no immutability; `export_audit_pack.py` (the tool for this) is legacy-disconnected |

Cross-cutting artefact findings: no artefact registry (the only registry is input-side sources); no signed-URL capability (zero SAS usage repo-wide); inconsistent period keying (`YYYY-MM-DD` platform vs `YYYY-MM` decks); artefact metadata (hash, MIME, producer version, approval, run ID) absent on all published outputs [all Confirmed].

---

## 8. SFTP automation assessment

**Current support: none.** Zero SFTP references repo-wide (no `paramiko`/`pysftp`, no ingress/egress code, nothing in requirements) [Confirmed].

**Can SFTP be added as an ingestion adapter without modifying the downstream agentic pipeline? Yes — cleanly.** The pipeline's entire entry contract is: files under `raw-v2/{client}/{book}/{dataset}/{freq}/{pid}/{period}/` plus a `_READY.json` marker written last (`router.py:73-76`; marker metadata read at `azure_io.py:56-73`). An SFTP landing service (push target or pull agent) that stabilises a delivery, copies it into that layout, and writes the marker requires no change to router, orchestrator, approvals, or outputs. The Azure Blob + Event Grid trigger can and should remain the standard entry point. Several ingestion features would come "for free": path-encoded client/portfolio isolation, schema-fingerprint drift detection, fail-closed new-source onboarding, optional `expected_files` manifests, pack-level idempotent skip.

**Controls assessment against the required checklist:**

| Control | Status | Evidence / note |
|---|---|---|
| Client-specific inbound folders | ◐ pattern exists in blob layout; would need mirroring in SFTP namespace | `path_parser.py:106-150` |
| Scheduled polling / event detection | ✔ event-based (Event Grid); no scheduler exists for pull | `function_app.py:87-96` |
| Temporary-file handling / stable-size checks / `.done` files | ✖ at SFTP layer (none exists); ✔ equivalent completion semantics via `_READY.json` marker | `router.py:73-76` |
| Checksums / manifests | ✖ checksums entirely; ◐ `expected_files` name-list only, no hashes | `router.py:259-270` |
| Expected-delivery windows / missing-file alerts / stale-data alerts | ✖ — no scheduling, no alerting anywhere (logging only; deployed `host.json` has no logging block) | `host.json:1-8` |
| Duplicate detection | ◐ pack-level, fingerprint-matched, blocks only prior `processed` | `router.py:280-292` |
| Replay protection | ✖ event-level (time-salted `event_id`; redelivery of non-terminal packs re-executes) | `event_log.py:11-13` |
| File sequencing | ✖ | — |
| Encrypted files / key rotation / IP allow-listing | ✖ (IP allow-listing mentioned only as prose guidance for the operator console) | `runbook:187-198` |
| Quarantine / dead-letter / retries | ✖ all three (retries are implicit Event Grid redelivery via re-raise, no backoff/poison handling in code) | `function_app.py:96` |
| Archive folders / acknowledgement files | ✖ | — |
| Source-system metadata | ◐ `_READY.json` carries `source_metadata` into approvals | `approvals.py:61-88` |
| Immutable raw storage | ✖ — raw container exists; no versioning/WORM/legal hold; all writes overwrite | `storage.py:181-193` |
| Downstream blob-trigger orchestration | ✔ proven | §2 |
| Tenant isolation | ◐ path-based only; single storage account; no per-client credentials/containers | `layout.py` |
| Credential management | ✖ weak — plaintext connection strings in app settings, no Key Vault, no managed identity on the data plane (the only managed-identity data path is in the *legacy* Streamlit app — `analytics/blob_storage.py:46-56`) | `provision.sh:58` |
| Operational resilience | ✖ 30-min synchronous ceiling; no-timeout projection subprocess; single-YAML registry race | `host.json:3`, `regime_runner.py:38`, `source_registry.py:112-147` |

**Push vs pull:** push (servicer exports to Trakt-controlled SFTP) fits the existing marker semantics most directly. Pull (Trakt fetches from servicer SFTP on schedule) additionally needs a scheduler, expected-window tracking, and missing-file alerting — none of which exist. Both converge on the same blob landing + marker contract.

**Verdict:** the adapter seam is genuinely clean — the preferred pattern (SFTP/API → immutable raw landing → common orchestration → … → controlled access) is *compatible* with the repo. But "immutable raw landing" and the integrity/alerting controls are not yet true of the blob layer itself, so SFTP work is roughly one-third adapter, two-thirds hardening the landing zone everything already shares.

---

## 9. Cross-cutting architecture gaps

- **Shared capability layer:** substantially real for MI/analytics (one compute path serves API, deck, Streamlit workflow) — the strongest cross-cutting asset. Not yet true for artefact generation lifecycle (deck generation is pipeline-side-effect only) or for exceptions (two disconnected models).
- **API boundaries:** one flat FastAPI file, no `/v1`, no `APIRouter` modularity, no response models, `/docs`+`/openapi.json` deliberately unauthenticated on a production API (`auth.py:58`) while the operator console correctly disables docs (`operator_app.py:34`) [Confirmed].
- **Schema contracts:** strong internally (field registry, dataset contract, regime rules YAML); absent externally (no committed OpenAPI, React types hand-written with no generation — drift caught only by tests CI never runs).
- **Event model:** single Event Grid subscription; no `eventType` check in code (a BlobDeleted event on the container would be processed as a create — `function_app.py:99-107`); no internal event bus for "run completed / artefact published / approval granted" that downstream interfaces could subscribe to.
- **Job model:** none. Everything is synchronous within trigger or HTTP request. This blocks on-demand generation, long comparisons, and any Copilot action that exceeds a chat timeout.
- **Artefact registry & metadata model:** absent (see §7). The `snapshot/` package contains the right design (content-hashed IDs, idempotent registration, storage-neutral interface) and is unused [Confirmed — `snapshot/__init__.py:11`].
- **Identity:** the deepest gap. No token validation anywhere; SWA front-door auth removed (commits `5b0cc2f`, `8d224b1`) while `docs/auth_setup_runbook.md:125-127` still claims it exists; the SWA build points at the API cross-origin (`VITE_AGENT_API_URL` in the SWA workflow), bypassing the linked-backend path that would inject the principal header. Consequence: either the deployed dashboard is broken (401s) or auth is disabled and a synthetic operator principal is minted (`auth.py:182-183`) [Confirmed mechanism; live setting Unknown].
- **Tenancy:** deployment-per-tenant by env var; two client-ID namespaces coexist (`client_001` in `config/clients/` vs `ERE` in deployment); the layered config resolver that would fix this is built and unwired (`config/system/config_resolver.py:1-5`); the agentic regime projection hard-codes the ERM-UK client config for all tenants (`assembler_agent.py:107`).
- **RBAC:** two roles (`client`/`operator`) with no resource-level scoping; operator console authority is one shared static token.
- **Approvals:** input-side only; auto-approval policy is well-designed (materiality classification, governance artefact) but its evidence is overwritten per pack_key; approved records remain editable (`ops.py:222-237`).
- **Audit/provenance:** excellent at loan-row level (`engine/provenance.py`, fail-closed); good at mapping-decision level (agent session records, approval JSONs); absent at artefact level; no per-request access audit; local-only `GovernanceLogger` writes.
- **Observability:** logging only; the deployed `host.json` carries no App Insights config (the one that does isn't deployed); no alerting; discovery of failures is pull-based.
- **Retries/idempotency:** pack-level idempotency good-but-partial (only blocks `processed`); no event replay protection; no idempotency keys on any API operation.
- **Testing:** ~700+ backend tests of genuinely high quality (golden questions, LLM misbehaviour tables, cost hardening) but **zero executed in CI**; no pytest config; auth disabled by default in the API test suite except `test_auth.py`; no test that an out-of-scope `portfolioId` is refused (there is no such control to test).
- **Deployment/secrets/environments:** four services, four mechanisms, one resource group, no IaC, no staging, no Key Vault, one publish-profile credential remaining; MI API deploy is manual-dispatch only; the *legacy* Streamlit dashboard has the most rigorous pipeline — inverted priorities [Confirmed].
- **Data retention:** none defined; `trakt-state/events/` grows one blob per uploaded file indefinitely; no TTL anywhere.
- **Fail-safe defaults:** mostly good (fail-closed routing, fail-closed auth toggle), with one bad exception: a mis-provisioned MI API silently serves the **synthetic demo dataset** rather than failing (`data_source.py:9-25`, fallback at `:18`) [Confirmed].

---

## 10. Legacy-contamination assessment

**Active dependencies on legacy code [Confirmed]:**
1. **The agentic pipeline's projection step is the legacy Gate 4 projector**, executed as a subprocess (`orchestrator_agent/adapters.py:139-153` → `assembler_agent.py:44,102-116`; again from `regime_runner.py:32-38`). This drags in the legacy enum agent (`regime_projector.py:43,48`) — the very component the agentic `projection_agent/gate4_adapter.py:20-23` was written to avoid — and does so with *inconsistent* enum-review policy: `allow_unreviewed=True` from the orchestrator vs `False` from the router.
2. **Gate 1 `semantic_alignment.py` is a shared kernel, not legacy** — 17 agentic import sites. It should be formally reclassified/relocated; treating it as legacy risks accidental breakage of the entire onboarding path.
3. Gate 2/3 primitives are consumed via deliberate, documented adapters that refuse the dangerous parts (ND5 silent fill, `sys.exit(2)` hard gates) — a healthy pattern.

**Duplicated logic [Confirmed]:** three Annex 2/investor XML builders (legacy Gate 5 ×2 + dormant preview builder); two NL→spec stacks (`llm_query_parser` live, `interpreter/` dormant); two deck engines (`deck.py` live, `pptx_builder.py` dead-but-tested); two exception models (live `43_validation_issues` vs dead hash-chained `exception_db.py`); two snapshot layers (dormant `snapshot/` vs live path-regex discovery); four independent ND-detection regexes; two onboarding agents (root `agents/` v1 vs `engine/onboarding_agent/` v2); business rules 48 (legacy) vs 7 (agentic).

**Ambiguous pipeline selection:** none at runtime — production selection is unambiguous (root trigger → agentic path). The ambiguity is *documentation-level*: the README leads with the orphaned `trakt_run.py` CLI and never mentions the deck stage or the agentic orchestrator; `.funcignore:3-5` mis-describes the deployed entry point [Confirmed].

**Migration risks:**
- The legacy Streamlit dashboard remains deployed on every `main` push with the repo's best CI while being documented as "do not migrate yet" legacy — a standing risk that legacy behaviour continues to be what some clients see.
- The validation-coverage regression (48→7 rules) means switching any consumer from legacy Gate 3 to the validation agent silently loses rule coverage; nothing versions or reconciles the two rule sets.
- Root `agents/onboarding_agent.py` carries a broken default (`_DEFAULT_ALIASES_DIR` points at a directory with no alias YAMLs, `:76`) — evidence it is drifting, and its LLM stack duplicates v2's.
- Confirmation sinks referenced by `mapping_persistence.py` (`aliases_pipeline.yaml`, `fields_registry_pipeline.yaml`) do not exist in-tree — the learning loop has never been exercised through to its declared destination [Confirmed].

**Independent deployability/testability:** the agentic pipeline is not independently testable in CI (no tests run at all), and not independently deployable from legacy: the Function App package zips `engine/` wholesale (`main_trakt.yml:33-44`), so legacy gates ship inside the production artifact — acceptable while Gate 4 is a runtime dependency, but it means legacy code cannot be removed without touching the production package definition.

Wholesale removal is **not** recommended: Gate 4 is load-bearing; Gate 1 is the mapping kernel; Gate 2/3 primitives are consumed by adapters. The exceptions where evidence supports retirement decisions (either wire in or remove): `pptx_builder.py` stack, root `agents/` package, `exception_db.py` family (or, better, harvest its hash-chain design for the artefact registry), `mi_agent/interpreter/` + `mi_runtime.py` + `semantic_resolver.py`, and the orphaned `trakt_run.py` README prominence.

---

## 11. Prioritised gap register

### A. Must have before any Copilot pilot

| Gap | Reason | States | Risk if omitted | Target-state outcome | Evidence |
|---|---|---|---|---|---|
| Real token validation + restored front-door auth | The API trusts a client-suppliable header; SWA auth removed | 2, 3 | Principal forgery; unauthenticated data access | Entra JWT validation (JWKS, aud/iss) in the API; SWA auth or APIM in front; OBO for Copilot | `auth.py:6-11`; `staticwebapp.config.json:1-15`; commits `5b0cc2f`/`8d224b1` |
| Request-scoped tenancy/authorisation | `portfolioId` never checked against principal | 2, 3 | Cross-client data disclosure | Principal→client/portfolio entitlement enforced on every action | `auth.py:27-30`; `app.py:1551` |
| Approval state on runs & artefacts + minimal artefact registry | "Approved" doesn't exist for outputs; `latest` is mutable | 1, 2, 3 | Copilot serves unapproved/stale/draft outputs with no way to tell | Registry with lifecycle (generated→validated→approved→superseded), immutable IDs, run linkage | `approvals.py:29-35`; `persistence.py:301-325`; `pptx_stage.py:180-186` |
| Typed, versioned action endpoints (response models, `/v1`) | OpenAPI spec has empty response bodies; unusable for declarative agents | 2 | Copilot manifest can't be built; contract drift | Pydantic response models on a small governed action set | `app.py` (no `response_model`) |
| Close/gate `/docs` + `/openapi.json` | Full route surface publicly enumerable | 2, 3 | Reconnaissance surface | Gate behind auth (operator console already does this) | `auth.py:58`; `operator_app.py:34` |
| Fail-fast on data-source misconfiguration | Synthetic demo is the silent default dataset | all | Demo data presented as client data via Copilot | Hard failure when no governed source configured | `data_source.py:9-25` |
| Tests in CI (at least API-contract + auth suites) | Zero automation on ~740 tests | all | Regressions ship silently to an external surface | CI gate on API + auth tests minimum | `.github/` grep |

### B. Must have before external production use

| Gap | Reason | States | Risk if omitted | Target-state outcome | Evidence |
|---|---|---|---|---|---|
| Server-side narrative + conversation context | Product-quality answers exist only in React | 2 | Copilot answers are data + filler sentence; follow-ups broken | Shared presentation/narrative service; context contract | `responsePresenter.ts`; `adapters.py:489-502`; `analysisContext.ts` |
| Async job model + idempotency keys | On-demand generation impossible; sync ceiling | 2, 3 | Timeouts, duplicate side-effects | Job resource with polling; idempotent generation actions | `app.py` sync handlers; `host.json:3` |
| Secure artefact hand-off (expiring links, metadata) | No SAS/expiry/MIME/hash/recipient controls | 2, 3 | Uncontrolled redistribution of client artefacts | Short-lived signed URLs + artefact metadata + access audit | zero SAS usage repo-wide |
| Immutable evidence & event log | Overwrite-everything storage; governance artefacts overwritten | 1, 2, 3 | Audit trail can be silently rewritten | Append-only event/evidence store (versioned blobs or ledger); retention policy | `storage.py:181-193`; `layout.py:91-94` |
| Operator identity + four-eyes | Shared static token; free-text `decided_by`; approved records editable | 1, 3 | Approval integrity indefensible to auditors/rating agencies | Entra-backed operator identity; seal approved records; dual control on promote/force-publish | `operator_app.py:53-68`; `ops.py:222-237` |
| Key Vault / managed identity + environment separation + IaC | Plaintext secrets; no staging; imperative provisioning | all | Credential leakage; untestable changes to prod | Managed identity on data plane; declared infra; dev/prod split | `provision.sh:58`; §9 |
| Event replay protection + eventType filtering | Redelivery re-executes non-terminal packs; deletes processed as creates | 3 | Duplicate runs; corrupt state | Event-ID dedup store; explicit event-type allowlist | `event_log.py:11-13`; `function_app.py:99-107` |
| Tenant-correct engine configuration | Regime projection hard-codes ERM-UK config for all clients | 1, 3 | Wrong-regime outputs for a second client | Client config resolved per run (wire `config_resolver.py`) | `assembler_agent.py:95,107` |

### C. Important for scale

- Durable/queued orchestration (escape the 30-minute synchronous ceiling; timeout on projection subprocess) — `host.json:3`, `regime_runner.py:38`.
- Registry concurrency safety (ETag/lease on `source_registry.yaml`) — `source_registry.py:112-147`.
- SFTP ingestion adapter + landing-zone integrity (checksums, quarantine, dead-letter, delivery windows, alerting) — §8.
- Unify snapshot identity: adopt the dormant `snapshot/` design (content-hashed IDs) for the production data layer — `snapshot/keys.py:93-119`.
- Validation-rule parity and versioned rule sets (close the 48→7 gap; version every rule/config contract) — §10.
- Alerting/observability (App Insights on the deployed host.json; failure notifications) — §9.
- Consistent period keying and per-run immutable artefact URIs (fix `YYYY-MM` deck overwrites) — `pptx_stage.py:143-148`.

### D. Optional enhancement

- MCP server exposure of the same action layer (useful for non-Copilot agent ecosystems).
- Harvest `exception_db.py`'s hash-chain design for artefact/evidence integrity, then retire the module.
- Retire or wire the dormant stacks (`interpreter/`, `mi_runtime`, `pptx_builder`, root `agents/`).
- Doc hygiene: fix `.funcignore` mis-description, README `next_action` vocabulary, stale auth runbook, 8-vs-11 XSD count.

---

## 12. Recommended target architecture

Minimum viable target supporting all three states from one capability layer — built by *rearranging and hardening what exists*, not greenfield:

**Shared capability services (mostly exist today):** canonicalisation (onboarding+transformation agents), validation (validation agent, with rule parity + versioning), domain calculations (existing `mi_agent_api` service modules), MI query (parser→spec→executor), artefact generation (deck via `mi_agent_pptx`; XML via completed delivery agent v2), provenance (extend `engine/provenance.py` discipline from rows to artefacts).

**New/completed governance spine:** an **artefact & run registry** (adopt the dormant `snapshot/` identity design: content-hashed immutable IDs, lifecycle states, run linkage, approval stamps); an **approval service** extending the existing source-approval machinery to runs and artefacts, with sealed records and identity-backed decisions; an **append-only evidence log**.

**Action/API layer:** a thin, versioned, schema-typed set of business actions (`get latest approved run`, `ask MI question`, `retrieve artefact`, `generate deck (async job)`, `list exceptions`, `compare periods`) fronting the capability services — consumed identically by the React dashboard, the MI Agent chat, Copilot (declarative agent / OpenAPI plugin), direct API clients, and operator tooling. Real Entra token validation + tenancy enforcement lives here, once, for every interface.

**Ingestion adapters:** existing blob+marker contract stays the canonical entry; SFTP push/pull adapters land into it; API upload lands into it. Landing zone hardened (immutable raw, checksums, quarantine, replay protection).

**Interface clients (thin):** React dashboard (specialist workspace: drill-through, remediation, mapping review, approvals), Trakt MI chat, Copilot agent, operator console — all calling the same actions; narrative service shared server-side.

```
 SFTP push/pull ─┐
 API upload ─────┼─► raw landing (immutable, checksummed) ─► Event Grid ─► router
 Blob drop ──────┘                                                       │
                                                          orchestrator (queued/durable)
                                                          onboard→transform→validate→stamp
                                                          →assemble→project→generate
                                                                    │
                          ┌─ approval service ◄── operator identity │
                          ▼                                         ▼
                 ARTEFACT & RUN REGISTRY  ◄──── evidence log ◄── provenance
                 (immutable IDs, lifecycle, run linkage)
                          ▲
        ┌─────────────────┴──────────────────────────────┐
        │        GOVERNED ACTION LAYER (/v1, typed,      │
        │        Entra-validated, tenant-scoped, jobs)   │
        └───┬──────────┬───────────┬──────────┬──────────┘
         React      MI chat     Copilot     direct API / workflows
        dashboard              agent        (State 3 consumers)
```

Preserved unchanged: the deterministic MI core, the fail-closed routing/approval instincts, the blob layout, the deck compute path, the onboarding governance artefacts.

---

## 13. Minimum viable Copilot pilot

**Demonstrable now (internally, after only the Section-11A items):** ask a quick MI question; get portfolio summary; compare periods; retrieve the latest investor deck; list validation issues for a run. All are served by existing deterministic code paths.

**Requires remediation first (the 11A list):** token validation + tenancy, minimal artefact/run approval semantics, typed response models on the pilot's actions, gated `/docs`, fail-fast data source, CI on the touched suites.

**Pilot action set (all read-only):**
1. `get latest approved portfolio summary` — requires the run-approval stamp.
2. `ask MI question` — via `run_mi_agent_query`, with the server-side narrative port (or, for a first internal pilot, explicitly labelled data-only answers).
3. `retrieve latest mapping report` / `retrieve latest validation report` — after registering reports as artefacts.
4. `retrieve latest approved investor deck` — retrieval only, via expiring link or bytes proxy with audit.
5. `list current validation exceptions`.
6. `compare latest approved run with prior run`.

**Require human confirmation:** none of the above mutate state; any distribution action (post to Teams, attach to Outlook) must require explicit confirmation and should be phase 2.

**Excluded from the first pilot:** deck/tape *generation* (needs the job model); ESMA XML anything (product itself incomplete — 96/107 paths unconfirmed); canonical tape export (bulk client data — needs the secure-retrieval controls); all operator/approval actions; scenario what-ifs (defer until answers can be clearly labelled hypothetical); anything touching pipeline/forecast datasets until the cross-client deck-source fallback (`mi_agent_pptx/cli.py:222`) class of issue is ruled out for the query path.

---

## 14. Final verdict

- **Most important architectural strength:** the enforced separation of reasoning from deterministic execution, and the fact that dashboard, chatbot, and investor deck already consume one shared deterministic compute layer (`mi_agent_workflow` + the `mi_agent_api` service modules + in-process `mi_agent_pptx/mi_api.py`). The three-state principle's hardest requirement is already the codebase's strongest habit.
- **Most important technical gap:** the absence of an artefact & run registry with lifecycle states — everything downstream ("latest approved X", safe retrieval, evidence packages, Copilot answers about provenance) is blocked on it, and every output today is a mutable, metadata-free blob.
- **Most important governance gap:** approval exists only for inputs; client-facing outputs (notably the investor deck) ship ungated, unversioned, and unlinked to their run, while approval decisions themselves rest on a shared token and remain editable after approval.
- **Most important Copilot-specific gap:** identity — no token validation anywhere, the front-door auth removed from the deployed SWA while the runbook claims it exists, and no request-scoped tenancy. Until this is fixed, no external agent can be allowed to reach the API at all.
- **Most important SFTP gap:** the channel simply doesn't exist — but the marker-based landing contract means the gap is an adapter plus landing-zone integrity (checksums, quarantine, replay protection), not a pipeline change.
- **Single highest-priority next architectural step:** build the artefact & run registry with approval lifecycle (adopting the dormant `snapshot/` identity design) and put a small, typed, Entra-validated `/v1` action layer in front of the existing capability services. That one step simultaneously unblocks Copilot, makes State 1 defensible to auditors, and gives State 3 consumers something safe to call — without rewriting any of the deterministic core that already works.
