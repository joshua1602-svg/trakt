# Verified runtime architecture map

Everything below was verified by direct code inspection on this branch, not
taken from documentation. Line references are to the commit under review.

## 1. MI question answering — actual runtime path

```
Client request (React POST /mi/query | Copilot POST /v1/copilot/mi/query)
→ gateway prefix middleware            mi_agent_api/gateway.py:136 (strips /api only to exact registered routes)
→ auth_guard (global dependency)       mi_agent_api/auth.py:200
    React: X-MS-CLIENT-PRINCIPAL header (base64 JSON, platform-injected, not signature-validated)
    Copilot: bypasses auth_guard; own Entra bearer validation (copilot_auth.py, RS256/JWKS, fail-closed 503)
    MI_AGENT_AUTH_ENABLED=false ⇒ synthetic operator principal (dev/test only)
→ identity → ExecutionContext         mi_agent_api/identity.py:128 / :161
    tenant_id = deployment config (MI_AGENT_CLIENT_ID → platform URI segment → "client_001")
    NEVER from header/token/body. require_trustworthy_platform_auth fails closed in production.
→ governed capability                  mi_agent_api/mi_service.py:236 execute_governed_mi_query
    require_scope(mi:query)            mi_service.py:265
    client_id mismatch ⇒ TENANT_MISMATCH  mi_service.py:269
    authorise_portfolio_access         trakt_core/tenancy.py:221 (before any data access)
    describe_active_dataset            mi_agent_api/datasets.py:866
    evaluate_source_approval           trakt_core/policy.py:83 (production: platform_canonical/central_tape only)
→ dataset frame resolution             mi_service.py:493 → datasets.py:678 _resolve_query_frame
    NOTE: uses authorised.requested_portfolio_id (raw request string), not the authorised token —
    the typed accessors resolve_authorised_frame/dataset_snapshot_for (datasets.py:912/:922) have
    ZERO production callers. See defect register.
→ single parse                         mi_service.py:496 → mi_agent/parsed_question.py:96
    → mi_agent/llm_query_parser.py:2516 parse_with_repair (deterministic-first, LLM repair optional)
    → semantics_resolver (mi_workflows/semantics.py:199) stored on parsed.semantics_context
    NOTE: mi_agent/interpreter/* is NOT the runtime parser (dev harness only).
→ recogniser registry                  mi_agent_api/chat_routing.py:2386 try_route
    candidates sorted (-confidence, priority, registration_index)  recogniser_registry.py:217
    capability gate (5 of 14 recognisers declare capabilities)     recogniser_registry.py:263
    handler exception ⇒ WARN + fall through to next candidate      chat_routing.py:2460
→ workflows / capabilities
    portfolio_risk_comparison (prio 65) → mi_workflows/portfolio_risk_comparison.py (engine primitives)
    concentration_analysis   (prio 66) → mi_workflows/concentration_analysis.py (engine primitives)
    period_change_analysis   (prio 85) → mi_agent/period_change/* (own governed calculations)
    legacy routes (scenario 10, cohorts 20/50, forecast 30, funded_bridge 40, geo 60,
      period_movement 70, portfolio_summary 80, temporal_compare 90, risk_limits 100, evolution 110)
      → per-module private math in mi_agent_api/*
    default fallback → mi_agent/mi_agent_workflow.py:161 run_mi_agent_query
      → mi_agent/mi_query_executor.py (point-in-time; no currency/date governance axes)
→ governed envelope                    trakt_core/envelope.py:202 GovernedResult
→ audit                                trakt_core/audit.py:56 (4 emission points, all in mi_service)
→ presenter                            mi_agent_api/presenters.py:20 to_react_payload / :42 Copilot
→ React AppShell / Copilot declarative agent
```

## 2. Portfolio data loading — actual runtime path

```
Source tape → engine/orchestrator (Gates 1-3) → *canonical_typed.csv → blob/local storage
→ runtime: mi_agent_api/data_source.py:159 resolve_data_source() [env-driven, process-global]
   1. MI_AGENT_ANALYTICS_DATASET   (prepared_explicit)
   2. MI_AGENT_PLATFORM_URI/CANONICAL/DIR → out_platform/… (platform_canonical)
   3. MI_AGENT_CENTRAL_TAPE | ONBOARDING_OUTPUT_ROOT+CLIENT_ID+RUN_ID (central_tape)
   4. MI_AGENT_DATA_CSV            (explicit_csv)
   5. synthetic_demo glob, sorted()[0]  (synthetic_demo; refused by policy in production)
→ signature-keyed cache (ETag | path:mtime:size), TTL 30s (MI_AGENT_DATA_CACHE_TTL)
→ per-request: datasets.py:678 _resolve_query_frame(view, portfolio_id)
   funded+run_id → _resolve_run_dataframe (blob: row-filter by source_portfolio_id;
                   local: root/{client_id}/{run_id}/… with cross-client glob fallback snapshots.py:147)
   else → whole active frame (ALL reporting dates — no single-date guard on default path)
→ MI calculation frame
```

## 3. Artefact generation

```
mi_agent_pptx CLI (in-process, no HTTP, client_id from run_state.json)
→ decks stored {client_id}/{period}/investor_pack.pptx (local MI_AGENT_DECK_ROOT or blob)
→ served via governed capability artefact.investor_pack.get (mi_agent_api/artefacts.py:99)
   scope check → requested client_id compared (never used to select) → authorise_portfolio_access
   → tenant selects the deck store → audit
→ React GET /mi/decks/download | Copilot GET /v1/copilot/artifacts/latest
→ Copilot signed download: HMAC-SHA256 token {kind, client_id, expiry}, TTL 300s
   redemption route has NO auth dependency; tenant read from token payload
```

## 4. Security boundaries and sources per path

| Path | Entry | Security boundary | Tenant source | Portfolio source | Reporting date source | Dataset source | Calculation owner | Audit |
|---|---|---|---|---|---|---|---|---|
| MI query (React) | `app.py:1269` | auth_guard + identity fail-closed | deployment env | request body (authorised) | dataset first-row / run_id; asOfDate is a LABEL only | env-resolved active dataset | route-dependent (see §1) | 4 points in mi_service |
| MI query (Copilot) | `copilot_actions.py:359` | Entra JWT validation | deployment env | request body (authorised) | same | same | same capability | same |
| Dashboard GETs (`/mi/snapshot`, `/mi/evolution/*`, `/mi/cohorts`, `/mi/geo/exposure`, `/mi/risk-limits`, `/mi/pipeline/*`, `/mi/forecast/*`, …) | `app.py` various | auth_guard only | n/a (none applied) | query params, NO tenancy authorisation | positional run lists | direct frame resolution, NO source-approval check | per-module private math | NONE |
| Deck download (React) | `app.py:1063` | auth_guard + governed capability | ExecutionContext | authorised | period param (shape-validated) | deck store | n/a | yes |
| Copilot artefact redeem | `copilot_actions.py:539` | HMAC token only (no auth dep) | token payload | token payload | n/a | deck store | n/a | mint-time only |

## 5. Registry consumption verdicts

| Registry | Verdict | Evidence |
|---|---|---|
| MI Semantics Field Registry (`mi_agent/mi_semantics_field_registry.yaml`) | ACTIVE | loaded per query `mi_service.py:483`; parser/validator/executor |
| Business Semantics Registry (`config/business_semantics_registry.yaml`) | ACTIVE via TWO independent loaders | `mi_workflows/semantics.py:163` (workflows + resolver on every parse) and `mi_agent/business_semantics.py:361` (period_change; only this one applies source overrides) |
| Canonical Registry (`config/system/fields_registry.yaml`) | PARTIAL | runtime consumer only `period_change/units.py`; main MI path never reads it |
| Recogniser Registry | ACTIVE | populated `chat_routing.py:2383`, consumed every query |
| Capability Registry (CAP_*) | ACTIVE, partial coverage | gate at `chat_routing.py:2449`; only 5/14 recognisers declare a capability |
| Workflow metadata (`Recogniser.metadata`) | DECLARED, NOT CONSUMED | no runtime reader |
| Portfolio registry overlay (`config/client/portfolio_registry.yaml`) | code ACTIVE, file ABSENT | registry derived from tape provenance columns |
| Tenancy registry (`config/tenancy.yaml`) | code ACTIVE, file ABSENT → open namespace | `tenancy.py:165-205` |

## 6. Dead code register (documented safety seams with no callers)

* `datasets.py:912 resolve_authorised_frame` / `:922 dataset_snapshot_for` — the
  documented "no dataframe without an authorisation token" seam; zero callers.
* `mi_agent/interpreter/*`, `mi_agent/mi_runtime.py` — Phase 8 harness, dev-only.
* `mi_agent/semantic_resolver.py`, `mi_agent/portfolio_reference.py` — no non-test importers.
* `portfolio_context.py:88` calls `_platform_client_id()` with zero args against a
  1-arg signature — permanently-dead branch (TypeError swallowed).

## 7. Audit emission points

`trakt_core/audit.py:56` (never raises; forbidden-key scrub). MI query path has
exactly four emissions, all in `mi_service.py` (:284 governance refusal, :301
storage failure, :315 source refusal, :350 terminal). Artefact capability adds
three (`artefacts.py:166,178,193`). Dashboard GETs and auth-layer 401/403 emit
no audit events.
